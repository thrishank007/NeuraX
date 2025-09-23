"""
LM Studio-based LLM generator for grounded response generation
Uses OpenAI-compatible API to communicate with local LM Studio server
Supports Gemma (multimodal) as main model and Qwen 4B Thinking as fallback
"""
import requests
import json
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from loguru import logger
from dataclasses import dataclass
from PIL import Image
import base64
import io
import re


@dataclass
class GeneratedResponse:
    """Container for generated response with metadata"""
    response_text: str
    confidence_score: float
    processing_time: float
    context_used: List[Dict]
    grounding_score: float
    citations_needed: List[int]
    model_used: str


class LMStudioGenerator:
    """Generates grounded responses using LM Studio local server"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config.get('base_url', 'http://localhost:1234/v1')
        self.timeout = config.get('timeout', 120)
        
        # Model configuration
        self.gemma_model = config.get('gemma_model', 'google/gemma-3n')  # Main multimodal model
        self.qwen_model = config.get('qwen_model', 'qwen/qwen3-4b-thinking-2507')  # Text fallback
        self.current_model = None
        
        # Generation parameters
        self.max_tokens = config.get('max_tokens', 1024)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.max_context_length = config.get('max_context_length', 4096)
        
        # Test connection to LM Studio
        self._test_connection()
        
        # Set initial model (prefer Gemma for multimodal capabilities)
        self._initialize_model()
    
    def _test_connection(self) -> bool:
        """Test connection to LM Studio server"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                logger.info(f"✅ Connected to LM Studio server at {self.base_url}")
                logger.info(f"Available models: {[model.get('id', 'unknown') for model in models.get('data', [])]}")
                return True
            else:
                logger.error(f"❌ LM Studio server returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to LM Studio server: {e}")
            logger.error("Make sure LM Studio is running and serving models on localhost:1234")
            return False
    
    def _initialize_model(self):
        """Initialize with the primary model (Gemma for multimodal)"""
        try:
            # Try to get available models
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model.get('id', '') for model in models_data.get('data', [])]
                
                # Set current model (prefer Gemma, fallback to Qwen, then use first available)
                if any('gemma' in model.lower() for model in available_models):
                    self.current_model = next(model for model in available_models if 'gemma' in model.lower())
                    logger.info(f"🖼️  Using Gemma model for multimodal capabilities: {self.current_model}")
                elif any('qwen' in model.lower() for model in available_models):
                    self.current_model = next(model for model in available_models if 'qwen' in model.lower())
                    logger.info(f"💭 Using Qwen model for text-only: {self.current_model}")
                elif available_models:
                    self.current_model = available_models[0]
                    logger.info(f"📝 Using available model: {self.current_model}")
                else:
                    logger.warning("No models available in LM Studio")
                    self.current_model = "unknown"
            else:
                logger.warning(f"Could not fetch models, using default: {self.gemma_model}")
                self.current_model = self.gemma_model
                
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            self.current_model = self.gemma_model
    
    def _switch_model(self, preferred_model: str) -> bool:
        """Switch to a specific model if available"""
        try:
            # For LM Studio, we just update our current_model reference
            # The actual model switching happens in LM Studio UI
            old_model = self.current_model
            
            # Check if the preferred model is available
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model.get('id', '') for model in models_data.get('data', [])]
                
                # Find best match
                for model in available_models:
                    if preferred_model.lower() in model.lower():
                        self.current_model = model
                        if old_model != model:
                            logger.info(f"🔄 Switched from {old_model} to {self.current_model}")
                        return True
            
            # If not found, keep current model
            logger.warning(f"Model {preferred_model} not available, keeping {self.current_model}")
            return False
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def _image_to_base64(self, image: Union[str, Image.Image]) -> str:
        """Convert image to base64 string"""
        try:
            if isinstance(image, str):
                # If image is a path, load it
                pil_image = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                pil_image = image.convert('RGB')
            else:
                raise ValueError("Image must be a file path or PIL Image")
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    def generate_grounded_response(self, query: str, context: List[Dict], 
                                 max_length: Optional[int] = None) -> GeneratedResponse:
        """
        Generate a grounded response using only the provided context
        
        Args:
            query: User query
            context: List of retrieved documents with content
            max_length: Maximum response length (unused, handled by LM Studio)
            
        Returns:
            GeneratedResponse with grounded answer and metadata
        """
        start_time = time.time()
        
        try:
            if not context:
                return self._generate_no_context_response(query, start_time)
            
            # Prefer Qwen for text-only queries (better reasoning)
            if not self._has_images_in_context(context):
                self._switch_model("qwen")
            else:
                # Use Gemma for multimodal queries
                self._switch_model("gemma")
            
            # Prepare context for generation
            context_text = self._prepare_context(context)
            
            # Create grounded prompt
            prompt = self._create_grounded_prompt(query, context_text)
            
            # Generate response via LM Studio API
            response_text = self._generate_with_lmstudio(prompt)
            
            # Validate grounding
            grounding_score = self._validate_response_grounding(response_text, context)
            
            # Extract citation needs
            citations_needed = self._identify_citation_needs(response_text, context)
            
            processing_time = time.time() - start_time
            
            return GeneratedResponse(
                response_text=response_text,
                confidence_score=min(grounding_score, 0.9),
                processing_time=processing_time,
                context_used=context,
                grounding_score=grounding_score,
                citations_needed=citations_needed,
                model_used=self.current_model or "unknown"
            )
            
        except Exception as e:
            logger.error(f"Error generating grounded response: {e}")
            return self._generate_error_response(query, start_time)
    
    def generate_multimodal_response(self, query: str, context: List[Dict], 
                                   image: Optional[Union[str, Image.Image]] = None,
                                   max_length: Optional[int] = None) -> GeneratedResponse:
        """
        Generate a multimodal response using both text context and image input
        
        Args:
            query: User query
            context: List of retrieved documents with content
            image: Optional image input (PIL Image or path to image)
            max_length: Maximum response length (unused, handled by LM Studio)
            
        Returns:
            GeneratedResponse with grounded answer and metadata
        """
        start_time = time.time()
        
        try:
            # Switch to Gemma for multimodal capabilities
            self._switch_model("gemma")
            
            if not context:
                return self._generate_no_context_response(query, start_time)
            
            # Prepare context for generation
            context_text = self._prepare_context(context)
            
            # Create multimodal prompt
            messages = self._create_multimodal_messages(query, context_text, image)
            
            # Generate response via LM Studio API
            response_text = self._generate_multimodal_with_lmstudio(messages)
            
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
                model_used=self.current_model or "unknown"
            )
            
        except Exception as e:
            logger.error(f"Error generating multimodal response: {e}")
            return self.generate_grounded_response(query, context, max_length)  # Fallback to text-only
    
    def _has_images_in_context(self, context: List[Dict]) -> bool:
        """Check if context contains image content"""
        for doc in context:
            metadata = doc.get('metadata', {})
            if metadata.get('file_type') in ['image', 'jpg', 'jpeg', 'png', 'bmp', 'gif']:
                return True
        return False
    
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
                context_parts.append(f"[Document {i+1}: {file_path}]\n{content[:800]}...")
        
        return '\n\n'.join(context_parts)
    
    def _create_grounded_prompt(self, query: str, context_text: str) -> str:
        """Create a grounded prompt for text-only generation"""
        # Determine which model prompt format to use
        if self.current_model and 'qwen' in self.current_model.lower():
            # Qwen format (thinking mode)
            system_message = (
                "You are a helpful AI assistant that provides accurate answers based only on the provided documents.\n"
                "Think step by step about the question, analyze the provided documents carefully, and give a grounded response.\n"
                "If the documents don't contain enough information to answer the question, clearly state that you don't have enough information.\n"
                "Always ground your responses in the provided context and cite specific information when possible."
            )

            user_message = (
                f"Documents:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Please think through this step by step and answer the question based only on the information provided in the documents above."
            )
        else:
            # Gemma or general format
            system_message = (
                "You are a helpful AI assistant that provides accurate answers based only on the provided documents.\n"
                "If the documents don't contain enough information to answer the question, clearly state that you don't have enough information.\n"
                "Always ground your responses in the provided context and cite specific information when possible."
            )

            user_message = (
                f"Documents:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Please answer the question based only on the information provided in the documents above."
            )

        # Create a simple concatenated prompt for LM Studio
        prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
        return prompt
    
    def _create_multimodal_messages(self, query: str, context_text: str, 
                                  image: Optional[Union[str, Image.Image]]) -> List[Dict]:
        """Create messages for multimodal generation"""
        system_message = """You are a helpful AI assistant that can analyze both text documents and images. 
Provide accurate answers based only on the provided documents and what you can see in the image.
If the documents or image don't contain enough information to answer the question, clearly state what information is missing.
Always ground your responses in the provided context and visual content."""
        
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text", 
                "text": f"""Documents:
{context_text}

Question: {query}

Please answer based on both the documents and the image provided."""
            }
        ]
        
        # Add image if provided
        if image:
            image_b64 = self._image_to_base64(image)
            if image_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_b64
                    }
                })
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        return messages
    
    def _generate_with_lmstudio(self, prompt: str) -> str:
        """Generate response using LM Studio chat completions API"""
        try:
            # Convert prompt to messages format for better compatibility
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            model_id = self.current_model or self.qwen_model or self.gemma_model or "unknown-model"
            payload = {
                "model": model_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content.strip()
                else:
                    logger.error("No choices in response")
                    return "I apologize, but I couldn't generate a response."
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return "I apologize, but I encountered an error while generating a response."
                
        except requests.exceptions.Timeout:
            logger.error("Request to LM Studio timed out")
            return "I apologize, but the response generation timed out. Please try again."
        except Exception as e:
            logger.error(f"Error in LM Studio generation: {e}")
            return "I apologize, but I encountered an error while generating a response."
    
    def _generate_multimodal_with_lmstudio(self, messages: List[Dict]) -> str:
        """Generate multimodal response using LM Studio chat completions API"""
        try:
            payload = {
                "model": self.current_model or self.gemma_model or self.qwen_model or "unknown-model",
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0]['message']['content']
                    return content.strip()
                else:
                    logger.error("No choices in multimodal response")
                    return "I apologize, but I couldn't generate a multimodal response."
            else:
                logger.error(f"LM Studio multimodal API error: {response.status_code} - {response.text}")
                return "I apologize, but I encountered an error while generating a multimodal response."
                
        except requests.exceptions.Timeout:
            logger.error("Multimodal request to LM Studio timed out")
            return "I apologize, but the multimodal response generation timed out. Please try again."
        except Exception as e:
            logger.error(f"Error in LM Studio multimodal generation: {e}")
            return "I apologize, but I encountered an error while generating a multimodal response."
    
    def _generate_no_context_response(self, query: str, start_time: float) -> GeneratedResponse:
        """Generate response when no context is available"""
        response_text = "I don't have any relevant documents to answer your question. Please try a different query or add more documents to the system."
        
        processing_time = time.time() - start_time
        
        return GeneratedResponse(
            response_text=response_text,
            confidence_score=0.9,
            processing_time=processing_time,
            context_used=[],
            grounding_score=1.0,
            citations_needed=[],
            model_used=self.current_model or "unknown"
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
            model_used=self.current_model or "unknown"
        )
    
    def _validate_response_grounding(self, response: str, context: List[Dict]) -> float:
        """
        Validate how well the response is grounded in the provided context
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
            
            # Penalize hallucination indicators
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
            return 0.5
    
    def _identify_citation_needs(self, response: str, context: List[Dict]) -> List[int]:
        """
        Identify which context documents should be cited.
        - Uses multiple fields (content, preview, metadata) for overlap
        - Falls back to top-K by similarity when overlap is low
        """
        citations_needed: List[int] = []

        try:
            response_lower = response.lower()
            response_words = set(re.findall(r'\b\w{3,}\b', response_lower))

            # Collect candidates with scores for fallback
            scored_docs: List[Tuple[int, float]] = []

            for i, doc in enumerate(context):
                # Gather candidate text from various fields
                parts: List[str] = []
                if isinstance(doc.get('content_preview'), str):
                    parts.append(doc['content_preview'])
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

                # Heuristics: either enough absolute overlap or decent ratio
                if overlap >= 2 or ratio >= 0.02:  # 2% of response words overlap
                    citations_needed.append(i)

                # Track score for fallback: blend similarity and overlap
                sim = float(doc.get('similarity_score', 0) or 0)
                score = sim * 0.7 + min(1.0, overlap / 10.0) * 0.3
                scored_docs.append((i, score))

            # If none selected, pick top-K by score/similarity
            if not citations_needed and scored_docs:
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                top_k = min(2, len(scored_docs))
                citations_needed = [idx for idx, _ in scored_docs[:top_k] if _ > 0]

            return citations_needed

        except Exception as e:
            logger.error(f"Error identifying citation needs: {e}")
            # Default to first few documents if something goes wrong
            return list(range(min(len(context), 2)))
    
    def generate_summary(self, documents: List[Dict], max_length: int = 300) -> str:
        """
        Generate a summary of multiple documents
        """
        try:
            if not documents:
                return "No documents provided for summarization."
            
            # Switch to Qwen for better text summarization
            self._switch_model("qwen")
            
            # Prepare content for summarization
            all_content = []
            for doc in documents:
                if 'content_preview' in doc:
                    all_content.append(doc['content_preview'])
                elif 'content' in doc and isinstance(doc['content'], list):
                    content_text = ' '.join([item.get('text', '') for item in doc['content']])
                    all_content.append(content_text[:400])
            
            combined_content = ' '.join(all_content)
            
            prompt = f"""Please provide a concise summary of the following content in approximately {max_length} words:

{combined_content[:2000]}

Summary:"""
            
            summary = self._generate_with_lmstudio(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."
    
    def supports_multimodal(self) -> bool:
        """Check if the current model supports multimodal input"""
        # Gemma models support multimodal, Qwen text-only models don't
        return bool(self.current_model and 'gemma' in self.current_model.lower())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and LM Studio setup"""
        try:
            # Get models from LM Studio
            response = requests.get(f"{self.base_url}/models", timeout=10)
            models_data = []
            if response.status_code == 200:
                models_data = response.json().get('data', [])
            
            return {
                'current_model': self.current_model,
                'supports_multimodal': self.supports_multimodal(),
                'base_url': self.base_url,
                'available_models': [model.get('id', 'unknown') for model in models_data],
                'gemma_model': self.gemma_model,
                'qwen_model': self.qwen_model,
                'server_connected': len(models_data) > 0,
                'max_tokens': self.max_tokens,
                'temperature': self.temperature
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                'current_model': self.current_model,
                'supports_multimodal': False,
                'base_url': self.base_url,
                'server_connected': False,
                'error': str(e)
            }
    
    def list_available_models(self) -> List[str]:
        """List all models available in LM Studio"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model.get('id', 'unknown') for model in models_data.get('data', [])]
            else:
                logger.error(f"Failed to fetch models: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []