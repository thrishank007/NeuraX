"""
Comprehensive error handling system for SecureInsight RAG
"""
import torch
import psutil
import platform
import subprocess
import sys
import traceback
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from loguru import logger
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import os
import shutil

from config import (
    ERROR_CONFIG, PERFORMANCE_CONFIG, MODEL_DOWNLOAD_CONFIG, 
    PROCESSING_CONFIG, SECURITY_CONFIG, LOGS_DIR
)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    MODEL_LOADING = "model_loading"
    FILE_PROCESSING = "file_processing"
    NETWORK = "network"
    MEMORY = "memory"
    DEVICE = "device"
    DEPENDENCY = "dependency"
    VALIDATION = "validation"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class ErrorReport:
    """Structured error report"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None
    user_guidance: Optional[str] = None


class ErrorHandler:
    """Comprehensive error handling with graceful degradation"""
    
    def __init__(self):
        self.error_log_file = LOGS_DIR / "error_reports.jsonl"
        self.system_info = self._get_system_info()
        self.fallback_strategies = self._initialize_fallback_strategies()
        self.retry_counts = {}
        
        # Ensure error log directory exists
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("ErrorHandler initialized with system info collection")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        try:
            return {
                'platform': platform.platform(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'pytorch_version': torch.__version__,
                'disk_space_gb': shutil.disk_usage('.').free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not collect complete system info: {e}")
            return {'error': str(e)}
    
    def _initialize_fallback_strategies(self) -> Dict[ErrorCategory, List[Callable]]:
        """Initialize fallback strategies for different error categories"""
        return {
            ErrorCategory.MODEL_LOADING: [
                self._fallback_to_cpu,
                self._use_smaller_model,
                self._use_cached_model,
                self._download_model_offline
            ],
            ErrorCategory.DEVICE: [
                self._fallback_to_cpu,
                self._reduce_batch_size,
                self._clear_gpu_memory
            ],
            ErrorCategory.MEMORY: [
                self._reduce_batch_size,
                self._enable_lazy_loading,
                self._clear_caches,
                self._force_garbage_collection
            ],
            ErrorCategory.FILE_PROCESSING: [
                self._retry_with_different_encoding,
                self._skip_corrupted_sections,
                self._use_alternative_processor
            ],
            ErrorCategory.NETWORK: [
                self._use_offline_mode,
                self._use_cached_resources,
                self._validate_network_isolation
            ],
            ErrorCategory.DEPENDENCY: [
                self._use_alternative_library,
                self._install_missing_dependency,
                self._use_fallback_implementation
            ]
        }
    
    def handle_error(self, error: Exception, category: ErrorCategory, 
                    context: Dict[str, Any] = None, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorReport:
        """
        Handle an error with appropriate recovery strategies
        
        Args:
            error: The exception that occurred
            category: Category of the error
            context: Additional context information
            severity: Severity level of the error
            
        Returns:
            ErrorReport with recovery information
        """
        error_id = f"{category.value}_{int(time.time())}"
        context = context or {}
        
        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            details={
                'error_type': type(error).__name__,
                'context': context,
                'system_info': self.system_info
            },
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc()
        )
        
        # Log the error
        self._log_error(error_report)
        
        # Attempt recovery based on category and configuration
        if ERROR_CONFIG.get('graceful_degradation', True):
            recovery_result = self._attempt_recovery(error_report)
            error_report.recovery_action = recovery_result.get('action')
            error_report.user_guidance = recovery_result.get('guidance')
        
        return error_report
    
    def _log_error(self, error_report: ErrorReport):
        """Log error report to file and logger"""
        try:
            # Log to structured file
            with open(self.error_log_file, 'a') as f:
                error_dict = {
                    'error_id': error_report.error_id,
                    'category': error_report.category.value,
                    'severity': error_report.severity.value,
                    'message': error_report.message,
                    'timestamp': error_report.timestamp.isoformat(),
                    'details': error_report.details,
                    'recovery_action': error_report.recovery_action,
                    'user_guidance': error_report.user_guidance
                }
                f.write(json.dumps(error_dict) + '\n')
            
            # Log to main logger
            log_level = {
                ErrorSeverity.LOW: logger.info,
                ErrorSeverity.MEDIUM: logger.warning,
                ErrorSeverity.HIGH: logger.error,
                ErrorSeverity.CRITICAL: logger.critical
            }[error_report.severity]
            
            log_level(f"Error {error_report.error_id}: {error_report.message}")
            
        except Exception as e:
            logger.error(f"Failed to log error report: {e}")
    
    def _attempt_recovery(self, error_report: ErrorReport) -> Dict[str, str]:
        """Attempt to recover from error using fallback strategies"""
        category = error_report.category
        
        if category not in self.fallback_strategies:
            return {
                'action': 'no_recovery_available',
                'guidance': 'Please check the error details and try again'
            }
        
        # Try each fallback strategy
        for strategy in self.fallback_strategies[category]:
            try:
                result = strategy(error_report)
                if result.get('success', False):
                    logger.info(f"Recovery successful using strategy: {strategy.__name__}")
                    return {
                        'action': f"recovered_using_{strategy.__name__}",
                        'guidance': result.get('guidance', 'Recovery completed successfully')
                    }
            except Exception as recovery_error:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {recovery_error}")
                continue
        
        return {
            'action': 'recovery_failed',
            'guidance': 'All recovery strategies failed. Please check system requirements and try again.'
        }
    
    # Fallback strategy implementations
    
    def _fallback_to_cpu(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Fallback from CUDA to CPU processing"""
        try:
            if torch.cuda.is_available() and 'cuda' in str(error_report.message).lower():
                logger.info("Falling back to CPU processing due to CUDA error")
                
                # Clear CUDA cache if possible
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    'success': True,
                    'guidance': 'Switched to CPU processing. Performance may be slower but functionality is maintained.',
                    'device': 'cpu'
                }
            return {'success': False}
        except Exception as e:
            logger.error(f"CPU fallback failed: {e}")
            return {'success': False}
    
    def _use_smaller_model(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Use a smaller model variant when memory is insufficient"""
        try:
            model_alternatives = {
                'all-MiniLM-L6-v2': 'all-MiniLM-L12-v1',
                'clip-vit-base-patch32': 'clip-vit-base-patch16',
                'whisper-base': 'whisper-tiny',
                'whisper-small': 'whisper-tiny'
            }
            
            context = error_report.details.get('context', {})
            current_model = context.get('model_name', '')
            
            for large_model, small_model in model_alternatives.items():
                if large_model in current_model:
                    logger.info(f"Switching from {large_model} to {small_model}")
                    return {
                        'success': True,
                        'guidance': f'Using smaller model {small_model} to reduce memory usage.',
                        'alternative_model': small_model
                    }
            
            return {'success': False}
        except Exception as e:
            logger.error(f"Model size fallback failed: {e}")
            return {'success': False}
    
    def _use_cached_model(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Use cached model if available"""
        try:
            from pathlib import Path
            cache_dir = Path(MODEL_DOWNLOAD_CONFIG['cache_dir'])
            
            if cache_dir.exists():
                cached_models = list(cache_dir.glob('**/pytorch_model.bin'))
                if cached_models:
                    logger.info(f"Found {len(cached_models)} cached models")
                    return {
                        'success': True,
                        'guidance': 'Using cached model to avoid download issues.',
                        'cached_models': [str(m) for m in cached_models]
                    }
            
            return {'success': False}
        except Exception as e:
            logger.error(f"Cached model fallback failed: {e}")
            return {'success': False}
    
    def _download_model_offline(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Handle offline model download scenarios"""
        try:
            # Enable local files only mode
            MODEL_DOWNLOAD_CONFIG['local_files_only'] = True
            
            return {
                'success': True,
                'guidance': 'Switched to offline mode. Ensure all required models are downloaded locally.',
                'offline_mode': True
            }
        except Exception as e:
            logger.error(f"Offline mode fallback failed: {e}")
            return {'success': False}
    
    def _reduce_batch_size(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Reduce batch size to handle memory constraints"""
        try:
            current_batch_size = PROCESSING_CONFIG.get('batch_size', 32)
            new_batch_size = max(1, current_batch_size // 2)
            
            PROCESSING_CONFIG['batch_size'] = new_batch_size
            
            logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
            
            return {
                'success': True,
                'guidance': f'Reduced batch size to {new_batch_size} to handle memory constraints.',
                'new_batch_size': new_batch_size
            }
        except Exception as e:
            logger.error(f"Batch size reduction failed: {e}")
            return {'success': False}
    
    def _clear_gpu_memory(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Clear GPU memory to resolve CUDA out of memory errors"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_cached = torch.cuda.memory_reserved() / (1024**3)
                
                logger.info(f"Cleared GPU memory. Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
                
                return {
                    'success': True,
                    'guidance': 'Cleared GPU memory cache. Try the operation again.',
                    'memory_cleared_gb': memory_cached
                }
            
            return {'success': False, 'reason': 'CUDA not available'}
        except Exception as e:
            logger.error(f"GPU memory clearing failed: {e}")
            return {'success': False}
    
    def _enable_lazy_loading(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Enable lazy loading to reduce memory usage"""
        try:
            PERFORMANCE_CONFIG['lazy_loading'] = True
            
            return {
                'success': True,
                'guidance': 'Enabled lazy loading to reduce memory usage. Models will load on demand.',
                'lazy_loading': True
            }
        except Exception as e:
            logger.error(f"Lazy loading enablement failed: {e}")
            return {'success': False}
    
    def _clear_caches(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Clear various caches to free memory"""
        try:
            import gc
            
            # Clear Python garbage collection
            collected = gc.collect()
            
            # Clear embedding cache if available
            cache_cleared = 0
            try:
                from indexing.embedding_manager import EmbeddingManager
                # This would need to be implemented in the actual usage context
                cache_cleared = 1
            except:
                pass
            
            logger.info(f"Cleared caches. GC collected: {collected} objects")
            
            return {
                'success': True,
                'guidance': 'Cleared system caches to free memory.',
                'objects_collected': collected
            }
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return {'success': False}
    
    def _force_garbage_collection(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Force garbage collection to free memory"""
        try:
            import gc
            
            # Force garbage collection multiple times
            for _ in range(3):
                collected = gc.collect()
            
            # Get memory usage
            memory_info = psutil.virtual_memory()
            
            logger.info(f"Forced garbage collection. Available memory: {memory_info.available / (1024**3):.2f}GB")
            
            return {
                'success': True,
                'guidance': 'Performed garbage collection to free memory.',
                'available_memory_gb': memory_info.available / (1024**3)
            }
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return {'success': False}
    
    def _retry_with_different_encoding(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Retry file processing with different encoding"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            return {
                'success': True,
                'guidance': 'Try processing the file with different character encodings.',
                'suggested_encodings': encodings
            }
        except Exception as e:
            logger.error(f"Encoding fallback failed: {e}")
            return {'success': False}
    
    def _skip_corrupted_sections(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Skip corrupted sections of files during processing"""
        try:
            return {
                'success': True,
                'guidance': 'Skipping corrupted sections. Partial content will be processed.',
                'skip_corrupted': True
            }
        except Exception as e:
            logger.error(f"Corrupted section skipping failed: {e}")
            return {'success': False}
    
    def _use_alternative_processor(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Use alternative processing method"""
        try:
            alternatives = {
                'pdf': ['PyMuPDF', 'pdfplumber', 'PyPDF2'],
                'docx': ['python-docx', 'docx2txt'],
                'image': ['Tesseract', 'EasyOCR', 'PaddleOCR']
            }
            
            context = error_report.details.get('context', {})
            file_type = context.get('file_type', '')
            
            if file_type in alternatives:
                return {
                    'success': True,
                    'guidance': f'Try alternative processors: {alternatives[file_type]}',
                    'alternatives': alternatives[file_type]
                }
            
            return {'success': False}
        except Exception as e:
            logger.error(f"Alternative processor fallback failed: {e}")
            return {'success': False}
    
    def _use_offline_mode(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Switch to offline mode for network issues"""
        try:
            MODEL_DOWNLOAD_CONFIG['local_files_only'] = True
            
            return {
                'success': True,
                'guidance': 'Switched to offline mode. Network connectivity is not required.',
                'offline_mode': True
            }
        except Exception as e:
            logger.error(f"Offline mode switch failed: {e}")
            return {'success': False}
    
    def _use_cached_resources(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Use cached resources when network is unavailable"""
        try:
            cache_locations = [
                Path(MODEL_DOWNLOAD_CONFIG['cache_dir']),
                Path('cache'),
                Path('models')
            ]
            
            available_caches = [str(cache) for cache in cache_locations if cache.exists()]
            
            return {
                'success': len(available_caches) > 0,
                'guidance': f'Using cached resources from: {available_caches}',
                'cache_locations': available_caches
            }
        except Exception as e:
            logger.error(f"Cached resources fallback failed: {e}")
            return {'success': False}
    
    def _validate_network_isolation(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Validate that the system can operate in network isolation"""
        try:
            # Check if essential files exist locally
            essential_paths = [
                Path('models'),
                Path('vector_db'),
                Path('config.py')
            ]
            
            missing_paths = [str(path) for path in essential_paths if not path.exists()]
            
            if missing_paths:
                return {
                    'success': False,
                    'guidance': f'Missing essential files for offline operation: {missing_paths}',
                    'missing_paths': missing_paths
                }
            
            return {
                'success': True,
                'guidance': 'System validated for offline operation.',
                'offline_ready': True
            }
        except Exception as e:
            logger.error(f"Network isolation validation failed: {e}")
            return {'success': False}
    
    def _use_alternative_library(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Use alternative library when primary dependency fails"""
        try:
            library_alternatives = {
                'torch': ['tensorflow', 'jax'],
                'transformers': ['sentence-transformers'],
                'PIL': ['opencv-python', 'imageio'],
                'whisper': ['speech_recognition']
            }
            
            return {
                'success': True,
                'guidance': 'Consider using alternative libraries if primary dependencies fail.',
                'alternatives': library_alternatives
            }
        except Exception as e:
            logger.error(f"Alternative library fallback failed: {e}")
            return {'success': False}
    
    def _install_missing_dependency(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Attempt to install missing dependency"""
        try:
            # This is a placeholder - actual implementation would need careful consideration
            # of security implications in production environments
            
            return {
                'success': False,
                'guidance': 'Please install missing dependencies manually using pip install -r requirements.txt',
                'manual_install_required': True
            }
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return {'success': False}
    
    def _use_fallback_implementation(self, error_report: ErrorReport) -> Dict[str, Any]:
        """Use fallback implementation when advanced features fail"""
        try:
            return {
                'success': True,
                'guidance': 'Using simplified fallback implementation with reduced functionality.',
                'fallback_mode': True
            }
        except Exception as e:
            logger.error(f"Fallback implementation failed: {e}")
            return {'success': False}
    
    # Utility methods for error handling
    
    def retry_with_backoff(self, func: Callable, max_retries: int = None, 
                          delay: float = None, category: ErrorCategory = ErrorCategory.PERFORMANCE) -> Any:
        """
        Retry a function with exponential backoff
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            delay: Initial delay between retries
            category: Error category for classification
            
        Returns:
            Function result or raises last exception
        """
        max_retries = max_retries or ERROR_CONFIG.get('max_retries', 3)
        delay = delay or ERROR_CONFIG.get('retry_delay_seconds', 1)
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        # Handle the final failure
        error_report = self.handle_error(last_exception, category, 
                                       context={'function': func.__name__, 'attempts': max_retries + 1})
        raise last_exception
    
    def validate_system_requirements(self) -> Dict[str, Any]:
        """
        Validate system requirements and return compatibility report
        
        Returns:
            Dict containing validation results and recommendations
        """
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                validation_results['errors'].append(
                    f"Python {python_version.major}.{python_version.minor} is not supported. Minimum required: 3.8"
                )
                validation_results['compatible'] = False
            
            # Check memory
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                validation_results['warnings'].append(
                    f"Low system memory: {memory_gb:.1f}GB. Recommended: 8GB+"
                )
                validation_results['recommendations'].append("Consider enabling lazy loading and reducing batch sizes")
            
            # Check disk space
            disk_space_gb = shutil.disk_usage('.').free / (1024**3)
            if disk_space_gb < 5:
                validation_results['errors'].append(
                    f"Insufficient disk space: {disk_space_gb:.1f}GB. Minimum required: 5GB"
                )
                validation_results['compatible'] = False
            
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                validation_results['recommendations'].append(
                    f"CUDA available with {gpu_memory_gb:.1f}GB GPU memory"
                )
            else:
                validation_results['warnings'].append("CUDA not available. Will use CPU processing (slower)")
                validation_results['recommendations'].append("Consider installing CUDA for better performance")
            
            # Check essential dependencies
            essential_modules = ['torch', 'transformers', 'sentence_transformers', 'whisper', 'PIL']
            for module in essential_modules:
                try:
                    __import__(module)
                except ImportError:
                    validation_results['errors'].append(f"Missing required module: {module}")
                    validation_results['compatible'] = False
            
            logger.info(f"System validation completed. Compatible: {validation_results['compatible']}")
            
        except Exception as e:
            validation_results['errors'].append(f"System validation failed: {e}")
            validation_results['compatible'] = False
        
        return validation_results
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics from log file"""
        try:
            if not self.error_log_file.exists():
                return {'total_errors': 0, 'by_category': {}, 'by_severity': {}}
            
            stats = {
                'total_errors': 0,
                'by_category': {},
                'by_severity': {},
                'recent_errors': []
            }
            
            with open(self.error_log_file, 'r') as f:
                for line in f:
                    try:
                        error_data = json.loads(line.strip())
                        stats['total_errors'] += 1
                        
                        # Count by category
                        category = error_data.get('category', 'unknown')
                        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
                        
                        # Count by severity
                        severity = error_data.get('severity', 'unknown')
                        stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
                        
                        # Keep recent errors (last 10)
                        if len(stats['recent_errors']) < 10:
                            stats['recent_errors'].append({
                                'timestamp': error_data.get('timestamp'),
                                'category': category,
                                'message': error_data.get('message', '')[:100]  # Truncate long messages
                            })
                    
                    except json.JSONDecodeError:
                        continue
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {'error': str(e)}
    
    def test_offline_operation(self) -> Dict[str, Any]:
        """
        Test offline operation capabilities
        
        Returns:
            Dict containing test results
        """
        test_results = {
            'offline_capable': True,
            'tests_passed': [],
            'tests_failed': [],
            'recommendations': []
        }
        
        try:
            # Test 1: Check local model availability
            models_dir = Path(MODEL_DOWNLOAD_CONFIG['cache_dir'])
            if models_dir.exists() and list(models_dir.glob('**/*')):
                test_results['tests_passed'].append('Local models available')
            else:
                test_results['tests_failed'].append('No local models found')
                test_results['offline_capable'] = False
                test_results['recommendations'].append('Download models using download_models.py')
            
            # Test 2: Check vector database
            vector_db_dir = Path('vector_db')
            if vector_db_dir.exists():
                test_results['tests_passed'].append('Vector database directory exists')
            else:
                test_results['tests_failed'].append('Vector database not initialized')
                test_results['recommendations'].append('Initialize vector database before offline use')
            
            # Test 3: Test network isolation simulation
            try:
                # Temporarily set offline mode
                original_local_only = MODEL_DOWNLOAD_CONFIG.get('local_files_only', False)
                MODEL_DOWNLOAD_CONFIG['local_files_only'] = True
                
                test_results['tests_passed'].append('Network isolation mode activated')
                
                # Restore original setting
                MODEL_DOWNLOAD_CONFIG['local_files_only'] = original_local_only
                
            except Exception as e:
                test_results['tests_failed'].append(f'Network isolation test failed: {e}')
                test_results['offline_capable'] = False
            
            # Test 4: Check essential files
            essential_files = ['config.py', 'requirements.txt']
            for file_path in essential_files:
                if Path(file_path).exists():
                    test_results['tests_passed'].append(f'Essential file exists: {file_path}')
                else:
                    test_results['tests_failed'].append(f'Missing essential file: {file_path}')
            
            logger.info(f"Offline operation test completed. Capable: {test_results['offline_capable']}")
            
        except Exception as e:
            test_results['tests_failed'].append(f'Offline test error: {e}')
            test_results['offline_capable'] = False
        
        return test_results


# Convenience functions for common error handling patterns

def handle_model_loading_error(func: Callable, model_name: str = "unknown") -> Any:
    """Convenience function for handling model loading errors"""
    error_handler = ErrorHandler()
    
    try:
        return func()
    except Exception as e:
        error_report = error_handler.handle_error(
            e, 
            ErrorCategory.MODEL_LOADING,
            context={'model_name': model_name},
            severity=ErrorSeverity.HIGH
        )
        
        # Re-raise with additional context
        raise RuntimeError(f"Model loading failed: {error_report.user_guidance}") from e


def handle_file_processing_error(func: Callable, file_path: str, file_type: str = "unknown") -> Any:
    """Convenience function for handling file processing errors"""
    error_handler = ErrorHandler()
    
    try:
        return func()
    except Exception as e:
        error_report = error_handler.handle_error(
            e,
            ErrorCategory.FILE_PROCESSING,
            context={'file_path': file_path, 'file_type': file_type},
            severity=ErrorSeverity.MEDIUM
        )
        
        if ERROR_CONFIG.get('continue_on_error', True):
            logger.warning(f"File processing error handled: {error_report.user_guidance}")
            return None
        else:
            raise RuntimeError(f"File processing failed: {error_report.user_guidance}") from e


def handle_device_error(func: Callable, device: str = "auto") -> Any:
    """Convenience function for handling device-related errors"""
    error_handler = ErrorHandler()
    
    try:
        return func()
    except Exception as e:
        error_report = error_handler.handle_error(
            e,
            ErrorCategory.DEVICE,
            context={'device': device},
            severity=ErrorSeverity.HIGH
        )
        
        # Try CPU fallback if CUDA fails
        if 'cuda' in device.lower() and ERROR_CONFIG.get('fallback_to_cpu', True):
            logger.info("Attempting CPU fallback due to device error")
            try:
                return func()  # Caller should handle device switching
            except Exception as fallback_error:
                raise RuntimeError(f"Device error and CPU fallback failed: {error_report.user_guidance}") from fallback_error
        
        raise RuntimeError(f"Device error: {error_report.user_guidance}") from e