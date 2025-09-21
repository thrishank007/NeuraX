"""
Audio processing module for speech-to-text conversion
"""
import whisper
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import numpy as np


class AudioProcessor:
    """Handles processing of audio files using Whisper STT"""
    
    def __init__(self, model_size: str = "tiny"):
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model for offline STT"""
        try:
            logger.info(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def process_audio(self, audio_path: Path) -> Dict:
        """
        Process an audio file and convert to text
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict containing transcription and audio metadata
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        suffix = audio_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported audio format: {suffix}")
        
        try:
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sample_rate
            
            # Transcribe using Whisper
            logger.info(f"Transcribing audio: {audio_path}")
            result = self.model.transcribe(str(audio_path))
            
            # Extract segments with timestamps
            segments = []
            if 'segments' in result:
                for segment in result['segments']:
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'].strip()
                    })
            
            # Audio quality metrics
            rms_energy = np.sqrt(np.mean(audio_data**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
            
            return {
                'file_path': str(audio_path),
                'file_type': 'audio',
                'transcription': result['text'].strip(),
                'language': result.get('language', 'unknown'),
                'segments': segments,
                'metadata': {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                    'rms_energy': float(rms_energy),
                    'zero_crossing_rate': float(zero_crossing_rate),
                    'file_size': audio_path.stat().st_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            raise
    
    def extract_audio_features(self, audio_path: Path) -> Dict:
        """
        Extract audio features for analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing audio features
        """
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            
            # Extract features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            
            return {
                'mfccs_mean': np.mean(mfccs, axis=1).tolist(),
                'mfccs_std': np.std(mfccs, axis=1).tolist(),
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'chroma_mean': np.mean(chroma, axis=1).tolist(),
                'tempo': float(librosa.beat.tempo(y=audio_data, sr=sample_rate)[0])
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features from {audio_path}: {e}")
            return {}
    
    def segment_audio(self, audio_path: Path, segment_length: float = 30.0) -> List[Dict]:
        """
        Segment long audio files for better processing
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of each segment in seconds
            
        Returns:
            List of segment transcriptions
        """
        try:
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sample_rate
            
            if duration <= segment_length:
                # Process as single segment
                return [self.process_audio(audio_path)]
            
            # Split into segments
            segment_samples = int(segment_length * sample_rate)
            segments = []
            
            for i in range(0, len(audio_data), segment_samples):
                segment_data = audio_data[i:i + segment_samples]
                start_time = i / sample_rate
                end_time = min((i + segment_samples) / sample_rate, duration)
                
                # Save temporary segment
                temp_path = audio_path.parent / f"temp_segment_{i//segment_samples}.wav"
                sf.write(temp_path, segment_data, sample_rate)
                
                try:
                    # Process segment
                    result = self.model.transcribe(str(temp_path))
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'transcription': result['text'].strip(),
                        'language': result.get('language', 'unknown')
                    })
                finally:
                    # Clean up temporary file
                    if temp_path.exists():
                        temp_path.unlink()
            
            return segments
            
        except Exception as e:
            logger.error(f"Error segmenting audio {audio_path}: {e}")
            raise
    
    def batch_process(self, audio_paths: List[Path]) -> List[Dict]:
        """Process multiple audio files"""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.process_audio(audio_path)
                results.append(result)
                logger.info(f"Successfully processed audio: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to process audio {audio_path}: {e}")
                continue
        
        return results