"""
LLM generator for grounded response generation using quantized models
Supports MiniCPM-V, Qwen2VL, and text-only models
"""
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig,
    AutoProcessor,
    AutoConfig
)

# Try to import Qwen2VL specific classes, fall back if not available
try:
    from transformers import Qwen2VLForConditionalGeneration
    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False

# Try to import Idefics3 for SmolVLM support
try:
    from transformers import Idefics3ForConditionalGeneration
    HAS_IDEFICS3 = True
except ImportError:
    HAS_IDEFICS3 = False

from typing import List, Dict, Optional, Tuple, Union, Any
from loguru import logger
from dataclasses import dataclass
import re
import time
from PIL import Image
import numpy as np
import warnings
from huggingface_hub import login, whoami
import os

# Suppress some common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class GeneratedResponse:
    """Container for generated response with metadata"""
    response_text: str
    confidence_score: float
    processing_time: float
    context_used: List[Dict]
    grounding_score: float
    citations_needed: List[int]
    model_used: str = "unknown"  # Added for compatibility with LM Studio generator


class LLMGenerator:
    """Generates grounded responses using quantized models (LLaVA or fallback)"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.processor = None  # For multimodal models
        
        # Device configuration with better handling
        if model_config.get('force_cpu', False):
            self.device = torch.device('cpu')
            logger.info("🔧 Forcing CPU-only inference for stability")
        elif model_config.get('device') == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available() and not model_config.get('force_cpu', False):
            self.device = torch.device('cuda')
            logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            logger.info("💻 Using CPU for inference")
            
        self.is_multimodal = model_config.get('use_multimodal', False)
        
        # Configuration
        self.max_length = model_config.get('max_length', 512)
        self.temperature = model_config.get('temperature', 0.7)
        self.max_context_length = model_config.get('max_context_length', 2048)
        self.max_new_tokens = model_config.get('max_new_tokens', 512)
        
        # Setup HuggingFace authentication
        self._setup_huggingface_auth()
        
        self._load_model()
    
    def _setup_huggingface_auth(self):
        """Setup HuggingFace authentication for gated models"""
        try:
            # Check if already authenticated
            user_info = whoami()
            logger.info(f"Already authenticated with HuggingFace as: {user_info['name']}")
            return True
        except Exception:
            # Try to authenticate with token from environment or config
            hf_token = os.getenv('HF_TOKEN') or self.model_config.get('hf_token')
            
            if hf_token:
                try:
                    login(token=hf_token)
                    user_info = whoami()
                    logger.info(f"Successfully authenticated with HuggingFace as: {user_info['name']}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to authenticate with provided token: {e}")
            
            # Authentication not available
            logger.warning("HuggingFace authentication not found.")
            logger.warning("Gated models (MiniCPM-V, Qwen2VL) may not work without authentication.")
            logger.warning("Run 'python setup_hf_login.py' to set up authentication.")
            logger.warning("Or set HF_TOKEN environment variable with your HuggingFace token.")
            return False
    
    def _load_model(self):
        """Load quantized LLM model for offline generation"""
        try:
            model_name = self.model_config.get('model_name', 'openbmb/MiniCPM-V-4_5')
            
            logger.info(f"Loading LLM model: {model_name}")
            
            # Detect model type and load appropriately
            if 'llama' in model_name.lower() and 'vlm' not in model_name.lower():
                # Llama models are text-only (except VLM variants), so disable multimodal
                logger.info("Detected Llama model - loading as text-only")
                self.is_multimodal = False
                self._load_text_model(model_name)
                return
            elif 'qwen' in model_name.lower() and 'vl' not in model_name.lower():
                # Regular Qwen models are text-only
                logger.info("Detected Qwen text model - loading as text-only")
                self.is_multimodal = False
                self._load_text_model(model_name)
                return
            
            # Try to load multimodal models in order of preference
            if self.is_multimodal:
                # Try SmolVLM first (lightweight multimodal)
                if 'smolvlm' in model_name.lower():
                    success = self._load_smolvlm_model(model_name)
                    if success:
                        return
                    else:
                        logger.warning("Failed to load SmolVLM model, trying fallback...")
                
                # Try MiniCPM-V
                if 'minicpm' in model_name.lower():
                    success = self._load_minicpm_model(model_name)
                    if success:
                        return
                    else:
                        logger.warning("Failed to load MiniCPM-V model, trying fallback...")
                
                # Try Qwen2VL as fallback
                fallback_model = self.model_config.get('fallback_model', 'HuggingFaceTB/SmolVLM-500M-Instruct')
                if 'qwen' in fallback_model.lower() and 'vl' in fallback_model.lower():
                    success = self._load_qwen2vl_model(fallback_model)
                    if success:
                        return
                    else:
                        logger.warning("Failed to load Qwen2VL model, falling back to text-only model")
                        self.is_multimodal = False
            
            # Load text-only model (final fallback)
            text_fallback = self.model_config.get('text_fallback', 'Qwen/Qwen3-4B-Instruct-2507')
            self._load_text_model(text_fallback)
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            logger.warning("Falling back to template-based responses")
            # Final fallback to template-based responses
            self.model = None
            self.tokenizer = None
            self.processor = None
    
    def _load_minicpm_model(self, model_name: str) -> bool:
        """Load MiniCPM-V multimodal model"""
        try:
            logger.info(f"Loading MiniCPM-V multimodal model: {model_name}")
            
            # Configure quantization for MiniCPM-V (efficient memory usage)
            if self.model_config.get('quantization', True) and torch.cuda.is_available():
                logger.info("Loading MiniCPM-V with 4-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load MiniCPM-V model with quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.float16,
                    trust_remote_code=self.model_config.get('trust_remote_code', True),
                    low_cpu_mem_usage=True
                )
            else:
                # CPU fallback for MiniCPM-V
                logger.info("Loading MiniCPM-V for CPU or without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32 if self.device.type == 'cpu' else torch.float16,
                    trust_remote_code=self.model_config.get('trust_remote_code', True),
                    low_cpu_mem_usage=True
                )
                if self.device.type == 'cpu':
                    self.model = self.model.to(self.device)
            
            # Load tokenizer for MiniCPM-V
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            # MiniCPM-V uses its own processor
            try:
                self.processor = AutoProcessor.from_pretrained(model_name)
            except:
                # Fallback: use tokenizer as processor
                self.processor = self.tokenizer
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure generation parameters for MiniCPM-V
            self.generation_config = {
                'max_new_tokens': self.model_config.get('max_new_tokens', 512),
                'temperature': self.model_config.get('temperature', 0.7),
                'do_sample': self.model_config.get('do_sample', True),
                'top_p': self.model_config.get('top_p', 0.9),
                'top_k': self.model_config.get('top_k', 50),
                'repetition_penalty': self.model_config.get('repetition_penalty', 1.1),
                'no_repeat_ngram_size': self.model_config.get('no_repeat_ngram_size', 3),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            logger.info(f"✅ MiniCPM-V model loaded successfully on {self.device}")
            logger.info(f"Model vocabulary size: {len(self.tokenizer)}")
            logger.info("🖼️ MiniCPM-V multimodal capabilities enabled (text + image)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MiniCPM-V model: {e}")
            return False
    
    def _load_qwen2vl_model(self, model_name: str) -> bool:
        """Load Qwen2VL multimodal model"""
        try:
            logger.info(f"Loading Qwen2VL multimodal model: {model_name}")
            
            if not HAS_QWEN2VL:
                logger.warning("Qwen2VLForConditionalGeneration not available, using AutoModelForCausalLM")
            
            # Configure quantization for Qwen2VL
            if self.model_config.get('quantization', True) and torch.cuda.is_available():
                logger.info("Loading Qwen2VL with 4-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load Qwen2VL model with quantization
                if HAS_QWEN2VL:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        dtype=torch.float16,
                        trust_remote_code=self.model_config.get('trust_remote_code', True),
                        low_cpu_mem_usage=True,
                        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
                    )
                else:
                    # Fallback to AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        dtype=torch.float16,
                        trust_remote_code=self.model_config.get('trust_remote_code', True),
                        low_cpu_mem_usage=True,
                        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
                    )
            else:
                # CPU fallback for Qwen2VL
                logger.info("Loading Qwen2VL for CPU or without quantization")
                if HAS_QWEN2VL:
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        dtype=torch.float32 if self.device.type == 'cpu' else torch.float16,
                        trust_remote_code=self.model_config.get('trust_remote_code', True),
                        low_cpu_mem_usage=True
                    )
                else:
                    # Fallback to AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=torch.float32 if self.device.type == 'cpu' else torch.float16,
                        trust_remote_code=self.model_config.get('trust_remote_code', True),
                        low_cpu_mem_usage=True
                    )
                
                if self.device.type == 'cpu':
                    self.model = self.model.to(self.device)
            
            # Load processor for Qwen2VL (handles both text and images)
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            # Also load tokenizer for compatibility
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure generation parameters for Qwen2VL
            self.generation_config = {
                'max_new_tokens': self.model_config.get('max_new_tokens', 512),
                'temperature': self.model_config.get('temperature', 0.7),
                'do_sample': self.model_config.get('do_sample', True),
                'top_p': self.model_config.get('top_p', 0.9),
                'top_k': self.model_config.get('top_k', 50),
                'repetition_penalty': self.model_config.get('repetition_penalty', 1.1),
                'no_repeat_ngram_size': self.model_config.get('no_repeat_ngram_size', 3),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            logger.info(f"✅ Qwen2VL model loaded successfully on {self.device}")
            logger.info(f"Model vocabulary size: {len(self.tokenizer)}")
            logger.info("🖼️ Qwen2VL multimodal capabilities enabled (text + image)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2VL model: {e}")
            return False
    
    def _load_smolvlm_model(self, model_name: str) -> bool:
        """Load SmolVLM multimodal model (lightweight alternative)"""
        try:
            logger.info(f"Loading SmolVLM multimodal model: {model_name}")
            
            # Import required classes for SmolVLM/Idefics3
            from transformers import Idefics3ForConditionalGeneration
            
            # Configure quantization for SmolVLM (memory efficient)
            if self.model_config.get('quantization', True) and torch.cuda.is_available():
                logger.info("Loading SmolVLM with 4-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = Idefics3ForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.float16,
                    trust_remote_code=self.model_config.get('trust_remote_code', True),
                    low_cpu_mem_usage=True
                )
            else:
                # CPU fallback for SmolVLM
                logger.info("Loading SmolVLM for CPU or without quantization")
                self.model = Idefics3ForConditionalGeneration.from_pretrained(
                    model_name,
                    dtype=torch.float32 if self.device.type == 'cpu' else torch.float16,
                    trust_remote_code=self.model_config.get('trust_remote_code', True),
                    low_cpu_mem_usage=True
                )
                if self.device.type == 'cpu':
                    self.model = self.model.to(self.device)
            
            # Load processor for SmolVLM (handles both text and images)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=self.model_config.get('trust_remote_code', True)
                )
            except:
                # Fallback: use tokenizer as processor
                self.processor = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=self.model_config.get('trust_remote_code', True)
                )
            
            # Load tokenizer for compatibility
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure generation parameters for SmolVLM
            self.generation_config = {
                'max_new_tokens': self.model_config.get('max_new_tokens', 256),
                'temperature': self.model_config.get('temperature', 0.7),
                'do_sample': self.model_config.get('do_sample', True),
                'top_p': self.model_config.get('top_p', 0.9),
                'top_k': self.model_config.get('top_k', 50),
                'repetition_penalty': self.model_config.get('repetition_penalty', 1.1),
                'no_repeat_ngram_size': self.model_config.get('no_repeat_ngram_size', 3),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            logger.info(f"✅ SmolVLM model loaded successfully on {self.device}")
            logger.info(f"Model vocabulary size: {len(self.tokenizer)}")
            logger.info("🖼️ SmolVLM multimodal capabilities enabled (text + image)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load SmolVLM model: {e}")
            return False
    
    def _load_text_model(self, model_name: str):
        """Load text-only model"""
        try:
            logger.info(f"Loading text-only model: {model_name}")
            
            # Configure quantization for efficiency
            if self.model_config.get('quantization', True) and torch.cuda.is_available():
                logger.info("Loading text model with 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.float16,
                    trust_remote_code=self.model_config.get('trust_remote_code', True),
                    low_cpu_mem_usage=True
                )
            else:
                # CPU fallback or no quantization
                logger.info("Loading text model for CPU or without quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32 if self.device.type == 'cpu' else torch.float16,
                    trust_remote_code=self.model_config.get('trust_remote_code', True),
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure generation parameters
            self.generation_config = {
                'max_new_tokens': self.model_config.get('max_new_tokens', 512),
                'temperature': self.model_config.get('temperature', 0.7),
                'do_sample': self.model_config.get('do_sample', True),
                'top_p': self.model_config.get('top_p', 0.9),
                'top_k': self.model_config.get('top_k', 50),
                'repetition_penalty': self.model_config.get('repetition_penalty', 1.1),
                'no_repeat_ngram_size': self.model_config.get('no_repeat_ngram_size', 3),
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            logger.info(f"✅ Text model loaded successfully on {self.device}")
            logger.info(f"Model vocabulary size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            raise
    
    def generate_grounded_response(self, query: str, context: List[Dict], 
                                 max_length: Optional[int] = None) -> GeneratedResponse:
        """
        Generate a grounded response using only the provided context
        
        Args:
            query: User query
            context: List of retrieved documents with content
            max_length: Maximum response length
            
        Returns:
            GeneratedResponse with grounded answer and metadata
        """
        start_time = time.time()
        max_length = max_length or self.max_length
        
        try:
            if not context:
                return self._generate_no_context_response(query, start_time)
            
            # Prepare context for generation
            context_text = self._prepare_context(context)
            
            # Create grounded prompt
            prompt = self._create_grounded_prompt(query, context_text)
            
            if self.model is None:
                # Fallback to template-based response
                return self._generate_template_response(query, context, start_time)
            
            # Generate response
            response_text = self._generate_with_model(prompt, max_length)
            
            # Validate grounding
            grounding_score = self._validate_response_grounding(response_text, context)
            
            # Extract citation needs
            citations_needed = self._identify_citation_needs(response_text, context)
            
            processing_time = time.time() - start_time
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=min(grounding_score, 0.9),  # Cap confidence
                processing_time=processing_time,
                context_used=context,
                grounding_score=grounding_score,
                citations_needed=citations_needed,
                model_used=self.model_config.get('model_name', 'huggingface-model')
            )
            
        except Exception as e:
            logger.error(f"Error generating grounded response: {e}")
            return self._generate_error_response(query, start_time)
    
    def _prepare_context(self, context: List[Dict]) -> str:
        """Prepare context documents for prompt"""
        context_parts = []
        
        for i, doc in enumerate(context[:5]):  # Limit to top 5 documents
            metadata = doc.get('metadata', {})
            file_path = metadata.get('file_path', f'Document {i+1}')
            
            # Extract content based on document type
            content = ""
            if 'content' in doc:
                if isinstance(doc['content'], list):
                    content = ' '.join([item.get('text', '') for item in doc['content']])
                else:
                    content = str(doc['content'])
            elif 'content_preview' in doc:
                content = doc['content_preview']
            
            if content:
                context_parts.append(f"[Document {i+1}: {file_path}]\n{content[:500]}...")
        
        return '\n\n'.join(context_parts)
    
    def _create_grounded_prompt(self, query: str, context_text: str) -> str:
        """Create a prompt using Llama 3.2's instruction format for grounded responses"""
        # Use Llama 3.2's chat template format
        system_message = """You are a helpful AI assistant that provides accurate answers based only on the provided documents. 
If the documents don't contain enough information to answer the question, clearly state that you don't have enough information.
Always ground your responses in the provided context and cite specific information when possible."""
        
        user_message = f"""Documents:
{context_text}

Question: {query}

Please answer the question based only on the information provided in the documents above."""
        
        # Format using Llama 3.2 chat template
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using fallback format")
        
        # Fallback format if chat template is not available
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return prompt
    
    def _generate_with_model(self, prompt: str, max_length: int) -> str:
        """Generate response using the loaded Llama 3.2 model"""
        try:
            # Tokenize input with proper attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=self.max_context_length,
                padding=True
            )
            
            # Move inputs to the same device as the model
            # Check if model is using device_map (distributed) or single device
            try:
                if hasattr(self.model, 'device'):
                    target_device = self.model.device
                else:
                    # Model is using device_map, get the device of the first parameter
                    target_device = next(self.model.parameters()).device
                
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            except Exception as device_error:
                logger.warning(f"Device placement issue: {device_error}. Falling back to CPU.")
                # Fallback to CPU if there are device issues
                target_device = torch.device('cpu')
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                # Move model to CPU if needed
                if hasattr(self.model, 'to'):
                    self.model = self.model.to(target_device)
            
            # Generate response using the configured generation parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=min(max_length, self.generation_config['max_new_tokens']),
                    temperature=self.generation_config['temperature'],
                    do_sample=self.generation_config['do_sample'],
                    top_p=self.generation_config['top_p'],
                    top_k=self.generation_config['top_k'],
                    repetition_penalty=self.generation_config['repetition_penalty'],
                    no_repeat_ngram_size=self.generation_config['no_repeat_ngram_size'],
                    pad_token_id=self.generation_config['pad_token_id'],
                    eos_token_id=self.generation_config['eos_token_id'],
                    use_cache=True
                )
            
            # Decode only the generated tokens (exclude input prompt)
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up the response
            response = response.strip()
            
            # Remove any remaining special tokens or artifacts
            if response.startswith('<|'):
                # Remove any remaining special tokens at the beginning
                lines = response.split('\n')
                clean_lines = []
                for line in lines:
                    if not line.strip().startswith('<|') and not line.strip().endswith('|>'):
                        clean_lines.append(line)
                response = '\n'.join(clean_lines).strip()
            
            return response if response else "I apologize, but I couldn't generate a proper response."
            
        except Exception as e:
            logger.error(f"Error in Llama 3.2 generation: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def _generate_template_response(self, query: str, context: List[Dict], 
                                  start_time: float) -> GeneratedResponse:
        """Generate template-based response when model is unavailable"""
        try:
            # Simple template-based response using context
            if not context:
                response_text = "I don't have enough information in the available documents to answer your question."
            else:
                # Extract key information from context
                doc_count = len(context)
                file_types = set(doc.get('metadata', {}).get('file_type', 'unknown') for doc in context)
                
                response_text = f"Based on {doc_count} relevant document(s) ({', '.join(file_types)}), "
                
                # Try to extract relevant snippets
                relevant_snippets = []
                for doc in context[:3]:
                    if 'content_preview' in doc:
                        relevant_snippets.append(doc['content_preview'][:100])
                
                if relevant_snippets:
                    response_text += "here's what I found: " + " ".join(relevant_snippets)
                else:
                    response_text += "I found relevant information but need more specific context to provide a detailed answer."
            
            processing_time = time.time() - start_time
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=0.6,  # Lower confidence for template responses
                processing_time=processing_time,
                context_used=context,
                grounding_score=0.8,  # Template responses are grounded by design
                citations_needed=list(range(len(context))),
                model_used="template"
            )
            
        except Exception as e:
            logger.error(f"Error in template generation: {e}")
            return self._generate_error_response(query, start_time)
    
    def _generate_no_context_response(self, query: str, start_time: float) -> GeneratedResponse:
        """Generate response when no context is available"""
        response_text = "I don't have any relevant documents to answer your question. Please try a different query or add more documents to the system."
        
        processing_time = time.time() - start_time
        
        return GeneratedResponse(
            response_text=response_text,
            confidence_score=0.9,  # High confidence in stating no information
            processing_time=processing_time,
            context_used=[],
            grounding_score=1.0,  # Perfectly grounded in stating limitations
            citations_needed=[],
            model_used="none"
        )
    
    def _generate_error_response(self, query: str, start_time: float) -> GeneratedResponse:
        """Generate error response"""
        response_text = "I apologize, but I encountered an error while processing your question. Please try again."
        
        processing_time = time.time() - start_time
        
        return GeneratedResponse(
            response_text=response_text,
            confidence_score=0.1,
            processing_time=processing_time,
            context_used=[],
            grounding_score=0.0,
            citations_needed=[],
            model_used="error"
        )
    
    def _validate_response_grounding(self, response: str, context: List[Dict]) -> float:
        """
        Validate how well the response is grounded in the provided context
        
        Args:
            response: Generated response text
            context: Source context documents
            
        Returns:
            Grounding score between 0.0 and 1.0
        """
        try:
            if not context or not response:
                return 0.0
            
            # Extract key terms from response
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            
            # Extract key terms from context
            context_words = set()
            for doc in context:
                if 'content' in doc:
                    if isinstance(doc['content'], list):
                        for item in doc['content']:
                            if isinstance(item, dict) and 'text' in item:
                                context_words.update(re.findall(r'\b\w+\b', item['text'].lower()))
                elif 'content_preview' in doc:
                    context_words.update(re.findall(r'\b\w+\b', doc['content_preview'].lower()))
            
            # Calculate overlap
            if not response_words:
                return 0.0
            
            overlap = len(response_words.intersection(context_words))
            grounding_score = overlap / len(response_words)
            
            # Penalize if response seems to contain information not in context
            hallucination_indicators = [
                'according to my knowledge', 'i know that', 'generally speaking',
                'it is well known', 'typically', 'usually', 'in general'
            ]
            
            response_lower = response.lower()
            hallucination_penalty = sum(1 for indicator in hallucination_indicators 
                                      if indicator in response_lower) * 0.2
            
            final_score = max(0.0, grounding_score - hallucination_penalty)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error validating response grounding: {e}")
            return 0.5  # Default moderate score
    
    def _identify_citation_needs(self, response: str, context: List[Dict]) -> List[int]:
        """
        Identify which context documents should be cited.
        Uses multiple fields and has a fallback to top-K by similarity.
        """
        citations_needed: List[int] = []

        try:
            response_lower = response.lower()
            response_words = set(re.findall(r'\b\w{3,}\b', response_lower))

            scored_docs: List[Tuple[int, float]] = []

            for i, doc in enumerate(context):
                parts: List[str] = []
                preview = doc.get('content_preview')
                if isinstance(preview, str):
                    parts.append(preview)
                content_val = doc.get('content')
                if isinstance(content_val, list):
                    parts.extend([item.get('text', '') for item in content_val if isinstance(item, dict)])
                elif isinstance(content_val, str):
                    parts.append(content_val)
                meta = doc.get('metadata', {}) or {}
                for k in ('title', 'subject', 'content_snippet', 'summary', 'chunk_text'):
                    v = meta.get(k)
                    if isinstance(v, str):
                        parts.append(v)

                doc_text = ' '.join([p for p in parts if p]).lower()
                if not doc_text:
                    scored_docs.append((i, float(doc.get('similarity_score', 0) or 0)))
                    continue

                doc_words = set(re.findall(r'\b\w{3,}\b', doc_text))
                if not response_words or not doc_words:
                    scored_docs.append((i, float(doc.get('similarity_score', 0) or 0)))
                    continue

                overlap = len(doc_words.intersection(response_words))
                ratio = overlap / max(1, len(response_words))
                if overlap >= 2 or ratio >= 0.02:
                    citations_needed.append(i)

                sim = float(doc.get('similarity_score', 0) or 0)
                score = sim * 0.7 + min(1.0, overlap / 10.0) * 0.3
                scored_docs.append((i, score))

            if not citations_needed and scored_docs:
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                top_k = min(2, len(scored_docs))
                citations_needed = [idx for idx, sc in scored_docs[:top_k] if sc > 0]

            return citations_needed

        except Exception as e:
            logger.error(f"Error identifying citation needs: {e}")
            return list(range(min(len(context), 2)))
    
    def generate_summary(self, documents: List[Dict], max_length: int = 300) -> str:
        """
        Generate a summary of multiple documents
        
        Args:
            documents: List of documents to summarize
            max_length: Maximum summary length
            
        Returns:
            Summary text
        """
        try:
            if not documents:
                return "No documents provided for summarization."
            
            # Prepare content for summarization
            all_content = []
            for doc in documents:
                if 'content_preview' in doc:
                    all_content.append(doc['content_preview'])
                elif 'content' in doc and isinstance(doc['content'], list):
                    content_text = ' '.join([item.get('text', '') for item in doc['content']])
                    all_content.append(content_text[:200])  # Limit per document
            
            combined_content = ' '.join(all_content)
            
            if self.model is None:
                # Template-based summary
                doc_count = len(documents)
                file_types = [doc.get('metadata', {}).get('file_type', 'document') for doc in documents]
                type_counts = {}
                for ft in file_types:
                    type_counts[ft] = type_counts.get(ft, 0) + 1
                
                summary = f"Summary of {doc_count} documents: "
                summary += ', '.join([f"{count} {ftype}(s)" for ftype, count in type_counts.items()])
                summary += f". Content covers: {combined_content[:150]}..."
                
                return summary
            
            # Use model for summarization
            prompt = f"Summarize the following content in {max_length} words or less:\n\n{combined_content[:1500]}\n\nSummary:"
            
            summary = self._generate_with_model(prompt, max_length)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."
    
    def generate_multimodal_response(self, query: str, context: List[Dict], 
                                   image: Optional[Union[str, Image.Image]] = None,
                                   max_length: Optional[int] = None) -> GeneratedResponse:
        """
        Generate a grounded response using both text context and optional image input
        
        Args:
            query: User query
            context: List of retrieved documents with content
            image: Optional image input (PIL Image or path to image)
            max_length: Maximum response length
            
        Returns:
            GeneratedResponse with grounded answer and metadata
        """
        start_time = time.time()
        max_length = max_length or self.max_length
        
        try:
            if not context:
                return self._generate_no_context_response(query, start_time)
            
            # Check if we have multimodal capabilities and an image
            if self.is_multimodal and self.processor and image is not None:
                return self._generate_multimodal_with_llava(query, context, image, max_length, start_time)
            else:
                # Fall back to text-only generation
                return self.generate_grounded_response(query, context, max_length)
                
        except Exception as e:
            logger.error(f"Error generating multimodal response: {e}")
            return self._generate_error_response(query, start_time)
    
    def _generate_multimodal_with_llava(self, query: str, context: List[Dict], 
                                      image: Union[str, Image.Image], max_length: int, 
                                      start_time: float) -> GeneratedResponse:
        """Generate response using LLaVA with both text and image inputs"""
        try:
            # Prepare text context
            context_text = self._prepare_context(context)
            
            # Load and prepare image
            if isinstance(image, str):
                # If image is a path, load it
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            else:
                # If image is numpy array or other format, convert
                pil_image = Image.fromarray(np.array(image)).convert('RGB')
            
            # Create multimodal prompt for LLaVA
            prompt = self._create_multimodal_prompt(query, context_text)
            
            # Process inputs with LLaVA processor
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with LLaVA
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_length, self.generation_config['max_new_tokens']),
                    temperature=self.generation_config['temperature'],
                    do_sample=self.generation_config['do_sample'],
                    top_p=self.generation_config['top_p'],
                    top_k=self.generation_config['top_k'],
                    repetition_penalty=self.generation_config['repetition_penalty'],
                    pad_token_id=self.generation_config['pad_token_id'],
                    eos_token_id=self.generation_config['eos_token_id'],
                    early_stopping=True,
                    use_cache=True
                )
            
            # Decode response
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up response
            response_text = response_text.strip()
            
            # Validate grounding
            grounding_score = self._validate_response_grounding(response_text, context)
            
            # Extract citation needs
            citations_needed = self._identify_citation_needs(response_text, context)
            
            processing_time = time.time() - start_time
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=min(grounding_score + 0.1, 0.95),  # Slight boost for multimodal
                processing_time=processing_time,
                context_used=context,
                grounding_score=grounding_score,
                citations_needed=citations_needed,
                model_used=self.model_config.get('model_name', 'multimodal-model')
            )
            
        except Exception as e:
            logger.error(f"Error in LLaVA multimodal generation: {e}")
            # Fall back to text-only generation
            return self.generate_grounded_response(query, context, max_length)
    
    def _create_multimodal_prompt(self, query: str, context_text: str) -> str:
        """Create a prompt for multimodal generation with MiniCPM-V or Qwen2VL"""
        model_name = self.model_config.get('model_name', '').lower()
        
        if 'minicpm' in model_name:
            # MiniCPM-V format
            prompt = f"""Based on the image and the provided documents, answer the following question. Use only the information from the documents and what you can see in the image.

Documents:
{context_text}

Question: {query}

Answer:"""
        elif 'qwen' in model_name:
            # Qwen2VL format
            prompt = f"""<|im_start|>system
You are a helpful assistant that answers questions based on provided documents and images.<|im_end|>
<|im_start|>user
<img></img>
Based on the image and the provided documents, answer the following question:

Documents:
{context_text}

Question: {query}<|im_end|>
<|im_start|>assistant
"""
        else:
            # Generic multimodal format (fallback)
            prompt = f"""<image>
Based on the image above and the provided documents, answer the following question. Use only the information from the documents and what you can see in the image.

Documents:
{context_text}

Question: {query}

Answer based on the image and documents:"""
        
        return prompt
    
    def supports_multimodal(self) -> bool:
        """Check if the model supports multimodal input"""
        return self.is_multimodal and self.processor is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'multimodal_support': self.supports_multimodal(),
            'device': self.device,
            'model_name': self.model_config.get('model_name', 'Unknown'),
            'quantization_enabled': self.model_config.get('quantization', False)
        }
        
        if self.tokenizer:
            info['vocab_size'] = len(self.tokenizer)
            info['pad_token_id'] = getattr(self.tokenizer, 'pad_token_id', None)
            info['eos_token_id'] = getattr(self.tokenizer, 'eos_token_id', None)
        
        return info