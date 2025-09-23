"""
LLM Generator Factory - Automatically selects between LM Studio and HuggingFace generators
"""
from typing import Dict, Any, Optional, List, Union
from PIL import Image
from loguru import logger

# Import configurations
from config import LLM_CONFIG, LM_STUDIO_CONFIG

# Import generator classes
from generation.lmstudio_generator import LMStudioGenerator, GeneratedResponse


class LLMGeneratorFactory:
    """Factory class to create the appropriate LLM generator"""
    
    @staticmethod
    def create_generator(config: Optional[Dict] = None) -> Union[LMStudioGenerator, Any]:
        """
        Create the appropriate LLM generator based on configuration
        
        Args:
            config: Optional configuration override
            
        Returns:
            LLM generator instance (LMStudioGenerator or fallback to HuggingFace)
        """
        # Use provided config or default to LLM_CONFIG
        if config is None:
            config = LLM_CONFIG
        
        # Check if LM Studio is preferred and available
        if config.get('use_lm_studio', True):
            try:
                # Merge LM Studio specific config
                lm_config = {**LM_STUDIO_CONFIG}
                if config:
                    # Override with any settings from LLM_CONFIG
                    lm_config.update({
                        'max_tokens': config.get('max_new_tokens', config.get('max_tokens', 1024)),
                        'temperature': config.get('temperature', 0.7),
                        'top_p': config.get('top_p', 0.9)
                    })
                
                generator = LMStudioGenerator(lm_config)
                logger.info("✅ Using LM Studio generator with local server")
                return generator
                
            except Exception as e:
                logger.warning(f"Failed to initialize LM Studio generator: {e}")
                logger.info("🔄 Falling back to HuggingFace generator")
                
                # Fall back to HuggingFace generator
                try:
                    from generation.llm_generator import LLMGenerator
                    return LLMGenerator(config)
                except Exception as hf_error:
                    logger.error(f"Failed to initialize HuggingFace generator: {hf_error}")
                    # Return a dummy generator that provides error responses
                    return DummyGenerator()
        else:
            # Use HuggingFace generator explicitly
            try:
                from generation.llm_generator import LLMGenerator
                logger.info("Using HuggingFace generator (explicit choice)")
                return LLMGenerator(config)
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace generator: {e}")
                return DummyGenerator()


class DummyGenerator:
    """Dummy generator that provides error responses when all else fails"""
    
    def __init__(self):
        self.model_config = {}
    
    def generate_grounded_response(self, query: str, context: List[Dict], 
                                 max_length: Optional[int] = None) -> GeneratedResponse:
        """Generate error response"""
        return GeneratedResponse(
            response_text="I apologize, but the AI models are currently unavailable. Please check your LM Studio setup or model configuration.",
            confidence_score=0.0,
            processing_time=0.0,
            context_used=context,
            grounding_score=0.0,
            citations_needed=[],
            model_used="none"
        )
    
    def generate_multimodal_response(self, query: str, context: List[Dict], 
                                   image: Optional[Union[str, Image.Image]] = None,
                                   max_length: Optional[int] = None) -> GeneratedResponse:
        """Generate error response for multimodal"""
        return self.generate_grounded_response(query, context, max_length)
    
    def generate_summary(self, documents: List[Dict], max_length: int = 300) -> str:
        """Generate error summary"""
        return "Summary unavailable - AI models are currently not accessible."
    
    def supports_multimodal(self) -> bool:
        """Dummy doesn't support multimodal"""
        return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get dummy model info"""
        return {
            'model_loaded': False,
            'error': 'No AI models available',
            'suggestion': 'Check LM Studio setup or HuggingFace model configuration'
        }


# Convenience function for backward compatibility
def create_llm_generator(config: Optional[Dict] = None) -> Union[LMStudioGenerator, Any]:
    """
    Create LLM generator with automatic fallback
    
    This is the main function that should be used throughout the application
    to get an LLM generator instance.
    """
    return LLMGeneratorFactory.create_generator(config)