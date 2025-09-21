"""
LLM generator for grounded response generation using quantized models
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
import re
import time


@dataclass
class GeneratedResponse:
    """Container for generated response with metadata"""
    response_text: str
    confidence_score: float
    processing_time: float
    context_used: List[Dict]
    grounding_score: float
    citations_needed: List[int]


class LLMGenerator:
    """Generates grounded responses using quantized Llama model"""
    
    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Configuration
        self.max_length = model_config.get('max_length', 512)
        self.temperature = model_config.get('temperature', 0.7)
        self.max_context_length = model_config.get('max_context_length', 2048)
        
        self._load_model()
    
    def _load_model(self):
        """Load quantized LLM model for offline generation"""
        try:
            model_name = self.model_config.get('model_name', 'microsoft/DialoGPT-medium')
            
            logger.info(f"Loading LLM model: {model_name}")
            
            # Configure quantization for efficiency
            if self.model_config.get('quantization', True) and torch.cuda.is_available():
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
                    torch_dtype=torch.float16
                )
            else:
                # CPU fallback or no quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16
                )
                self.model.to(self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"LLM model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            # Fallback to a simpler model or template-based responses
            self.model = None
            self.tokenizer = None
    
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
                citations_needed=citations_needed
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
        """Create a prompt that encourages grounded responses"""
        prompt = f"""Based ONLY on the provided documents, answer the following question. If the documents don't contain enough information to answer the question, say so clearly.

Documents:
{context_text}

Question: {query}

Answer based only on the documents above:"""
        
        return prompt
    
    def _generate_with_model(self, prompt: str, max_length: int) -> str:
        """Generate response using the loaded model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, 
                                         max_length=self.max_context_length)
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in model generation: {e}")
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
                citations_needed=list(range(len(context)))
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
            citations_needed=[]
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
            citations_needed=[]
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
        Identify which context documents should be cited
        
        Args:
            response: Generated response text
            context: Source context documents
            
        Returns:
            List of context document indices that should be cited
        """
        citations_needed = []
        
        try:
            response_lower = response.lower()
            
            for i, doc in enumerate(context):
                # Check if document content appears to be referenced
                doc_content = ""
                if 'content_preview' in doc:
                    doc_content = doc['content_preview'].lower()
                elif 'content' in doc and isinstance(doc['content'], list):
                    doc_content = ' '.join([item.get('text', '') for item in doc['content']]).lower()
                
                # Simple overlap check - could be made more sophisticated
                if doc_content:
                    doc_words = set(re.findall(r'\b\w{4,}\b', doc_content))  # Words 4+ chars
                    response_words = set(re.findall(r'\b\w{4,}\b', response_lower))
                    
                    overlap = len(doc_words.intersection(response_words))
                    if overlap >= 2:  # Threshold for citation
                        citations_needed.append(i)
            
            return citations_needed
            
        except Exception as e:
            logger.error(f"Error identifying citation needs: {e}")
            return list(range(min(len(context), 3)))  # Default to first 3 documents
    
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