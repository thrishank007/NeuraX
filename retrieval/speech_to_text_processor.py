"""
Speech-to-text processor for voice queries using Whisper
"""
import whisper
import numpy as np
import librosa
from typing import Optional, Dict, Union
from loguru import logger
from pathlib import Path
import tempfile
import io


class SpeechToTextProcessor:
    """Handles voice input queries using local Whisper model"""
    
    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        self.model = None
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model for offline STT"""
        try:
            logger.info(f"Loading Whisper {self.model_size} model for voice queries...")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper STT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper STT model: {e}")
            raise
    
    def transcribe_query(self, audio_data: Union[bytes, str, Path]) -> Optional[str]:
        """
        Transcribe audio data to text for query processing
        
        Args:
            audio_data: Audio data as bytes, file path, or Path object
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            # Handle different input types
            if isinstance(audio_data, bytes):
                # Save bytes to temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                try:
                    result = self._transcribe_file(temp_path)
                finally:
                    # Clean up temporary file
                    Path(temp_path).unlink(missing_ok=True)
                
                return result
                
            elif isinstance(audio_data, (str, Path)):
                return self._transcribe_file(audio_data)
            else:
                logger.error(f"Unsupported audio data type: {type(audio_data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio query: {e}")
            return None
    
    def _transcribe_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Transcribe audio file to text"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Audio file not found: {file_path}")
            return None
        
        try:
            # Validate audio quality first
            if not self.validate_audio_format(file_path):
                return None
            
            # Check audio quality
            quality_score = self._assess_audio_quality(file_path)
            
            if quality_score < 0.3:  # Poor quality threshold
                logger.warning(f"Poor audio quality detected (score: {quality_score:.2f})")
                return self.provide_fallback_message()
            
            # Transcribe using Whisper
            logger.info(f"Transcribing voice query: {file_path}")
            result = self.model.transcribe(str(file_path))
            
            transcribed_text = result['text'].strip()
            confidence = result.get('confidence', 0.0)
            
            # Check transcription confidence
            if len(transcribed_text) < 3:
                logger.warning("Transcription too short, likely poor audio quality")
                return self.provide_fallback_message()
            
            logger.info(f"Voice query transcribed: '{transcribed_text[:50]}...'")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {e}")
            return self.provide_fallback_message()
    
    def validate_audio_format(self, audio_input: Union[bytes, str, Path]) -> bool:
        """
        Validate audio format and basic properties
        
        Args:
            audio_input: Audio data or file path
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            if isinstance(audio_input, (str, Path)):
                file_path = Path(audio_input)
                
                # Check file extension
                if file_path.suffix.lower() not in self.supported_formats:
                    logger.warning(f"Unsupported audio format: {file_path.suffix}")
                    return False
                
                # Check file size (avoid very large files)
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    max_size = 50 * 1024 * 1024  # 50MB limit for voice queries
                    
                    if file_size > max_size:
                        logger.warning(f"Audio file too large: {file_size} bytes")
                        return False
                    
                    if file_size < 1024:  # Very small files likely corrupted
                        logger.warning(f"Audio file too small: {file_size} bytes")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating audio format: {e}")
            return False
    
    def _assess_audio_quality(self, file_path: Path) -> float:
        """
        Assess audio quality for transcription suitability
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            # Load audio for analysis
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            
            # Calculate quality metrics
            rms_energy = np.sqrt(np.mean(audio_data**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            # Check for silence (very low energy)
            if rms_energy < 0.01:
                return 0.1  # Very poor quality
            
            # Check for reasonable audio characteristics
            duration = len(audio_data) / sample_rate
            
            # Very short audio likely incomplete
            if duration < 0.5:
                return 0.2
            
            # Very long audio might be problematic for queries
            if duration > 60:  # 1 minute limit for voice queries
                return 0.3
            
            # Calculate composite quality score
            energy_score = min(rms_energy * 10, 1.0)  # Normalize energy
            duration_score = 1.0 if 1.0 <= duration <= 30.0 else 0.5
            zcr_score = 1.0 if 0.01 <= zero_crossing_rate <= 0.3 else 0.5
            
            quality_score = (energy_score + duration_score + zcr_score) / 3.0
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing audio quality: {e}")
            return 0.5  # Default moderate quality
    
    def handle_poor_audio_quality(self, audio_data: Union[bytes, str, Path]) -> Optional[str]:
        """
        Handle poor audio quality with fallback strategies
        
        Args:
            audio_data: Audio data that failed quality check
            
        Returns:
            Fallback message or None
        """
        logger.warning("Poor audio quality detected, providing fallback message")
        return self.provide_fallback_message()
    
    def provide_fallback_message(self) -> str:
        """Provide fallback message for poor audio quality"""
        return "Audio unclear; please type your query"
    
    def process_voice_query_with_fallback(self, audio_data: Union[bytes, str, Path]) -> Dict[str, any]:
        """
        Process voice query with comprehensive error handling and fallback
        
        Args:
            audio_data: Audio data for transcription
            
        Returns:
            Dict with transcription result and metadata
        """
        result = {
            'transcribed_text': None,
            'success': False,
            'audio_quality': 'unknown',
            'confidence': 0.0,
            'fallback_message': None,
            'processing_time': 0.0
        }
        
        try:
            import time
            start_time = time.time()
            
            # Validate format first
            if not self.validate_audio_format(audio_data):
                result['fallback_message'] = "Unsupported audio format. Please use WAV, MP3, or M4A."
                return result
            
            # Assess quality if it's a file
            if isinstance(audio_data, (str, Path)):
                quality_score = self._assess_audio_quality(Path(audio_data))
                result['audio_quality'] = 'good' if quality_score > 0.6 else 'poor' if quality_score > 0.3 else 'very_poor'
            
            # Attempt transcription
            transcribed_text = self.transcribe_query(audio_data)
            
            result['processing_time'] = time.time() - start_time
            
            if transcribed_text and transcribed_text != self.provide_fallback_message():
                result['transcribed_text'] = transcribed_text
                result['success'] = True
                result['confidence'] = 0.8  # Placeholder - Whisper doesn't always provide confidence
            else:
                result['fallback_message'] = transcribed_text or self.provide_fallback_message()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in voice query processing: {e}")
            result['fallback_message'] = "Voice processing failed. Please type your query."
            return result
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats"""
        return list(self.supported_formats)
    
    def test_microphone_input(self) -> bool:
        """
        Test if microphone input is working (placeholder for future implementation)
        
        Returns:
            True if microphone is accessible, False otherwise
        """
        # This would be implemented when adding real-time microphone support
        logger.info("Microphone test not implemented - using file-based audio input")
        return False