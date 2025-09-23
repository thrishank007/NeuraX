"""
Configuration settings for SecureInsight RAG system
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"
KG_SECURITY_DIR = PROJECT_ROOT / "kg_security"
FEEDBACK_DIR = PROJECT_ROOT / "feedback"
UI_DIR = PROJECT_ROOT / "ui"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, VECTOR_DB_DIR, LOGS_DIR, KG_SECURITY_DIR, FEEDBACK_DIR, UI_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODELS = {
    "text": "sentence-transformers/all-MiniLM-L6-v2",
    "image": "openai/clip-vit-base-patch32",
    "multimodal": "openai/clip-vit-base-patch32"
}

LLM_CONFIG = {
    "model_name": "HuggingFaceTB/SmolVLM-500M-Instruct",  # SmolVLM 500M as lightweight multimodal model
    "fallback_model": "HuggingFaceTB/SmolVLM-500M-Instruct",  # Same model as fallback
    "text_fallback": "Qwen/Qwen3-0.6B",  # Lightweight text-only fallback
    "quantization": True,
    "max_length": 1024,  # Reduced for lighter memory usage
    "temperature": 0.7,
    "device": "auto",  # auto-detect CUDA/CPU
    "dtype": "float16",
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "use_auth_token": False,  # These models don't require auth token
    "trust_remote_code": True,
    "hf_token": None,  # Set this or use HF_TOKEN environment variable for gated models
    "pad_token_id": None,  # Will be set during model loading
    "eos_token_id": None,  # Will be set during model loading
    "max_new_tokens": 256,  # Reduced for lighter memory usage
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    # Multimodal configurations
    "use_multimodal": True,  # Enable multimodal capabilities
    "image_token": "<image>",  # Special token for image inputs
    "max_image_size": 448,  # Maximum image size for models
    "vision_feature_layer": -2,  # Which layer to extract vision features from
    "vision_feature_select_strategy": "default",  # How to select vision features
    # Phi-3 Vision specific settings
    "phi3_config": {
        "slice_mode": True,  # Enable slice mode for better performance
        "max_slice_nums": 9,  # Maximum number of image slices
        "use_image_id": False,  # Whether to use image IDs
        "max_batch_size": 1,  # Batch size for inference
        "sampling": True,  # Enable sampling
        "stream": False  # Disable streaming for batch processing
    },
    # Alternative gated models (require HF authentication)
    "gated_models": {
        "smolvlm": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "qwen": "Qwen/Qwen3-0.6B",
        "minicpm": "openbmb/MiniCPM-V-2_6",
        "qwen2vl": "Qwen/Qwen2-VL-7B-Instruct",
        "llama32": "meta-llama/Llama-3.2-11B-Vision-Instruct"
    }
}

WHISPER_CONFIG = {
    "model_size": "tiny",
    "language": "en",
    "device": "auto",
    "fp16": True,
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6
}

# Vector database settings
CHROMA_CONFIG = {
    "persist_directory": str(VECTOR_DB_DIR),
    "collection_name": "secureinsight_collection",
    "embedding_function": None,  # Will be set by EmbeddingManager
    "distance_function": "cosine",
    "hnsw_space": "cosine",
    "anonymized_telemetry": False,
    "allow_reset": True
}

# UI settings
GRADIO_CONFIG = {
    "server_name": "127.0.0.1",
    "server_port": 7860,
    "share": False
}

STREAMLIT_CONFIG = {
    "server_port": 8501,
    "server_address": "127.0.0.1"
}

# Security and KG settings
KG_CONFIG = {
    "anomaly_threshold": 0.8,
    "max_nodes": 1000,
    "enable_security_layer": True,
    "centrality_threshold": 0.1,
    "outlier_std_threshold": 2.0,
    "min_confidence_score": 0.6,
    "quarantine_enabled": True,
    "audit_log_enabled": True,
    "tamper_detection_enabled": True,
    "graph_layout": "spring",
    "node_size_range": (10, 100),
    "edge_weight_threshold": 0.3
}

# Search and retrieval settings
SIMILARITY_THRESHOLD = 0.7
SEARCH_CONFIG = {
    "default_k": 5,
    "max_results": 50,
    "similarity_threshold": SIMILARITY_THRESHOLD
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "rotation": "10 MB",
    "retention": "1 week",
    "compression": "gz",
    "serialize": True,
    "backtrace": True,
    "diagnose": True
}

# Processing configuration
PROCESSING_CONFIG = {
    "batch_size": 32,
    "max_file_size_mb": 100,
    "supported_image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
    "supported_audio_formats": [".wav", ".mp3", ".m4a", ".flac", ".ogg"],
    "supported_document_formats": [".pdf", ".docx", ".doc", ".txt"],
    "ocr_languages": ["eng"],
    "max_audio_duration_minutes": 30,
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "normalize_embeddings": True,
    "cache_embeddings": True,
    "cache_size_mb": 500,
    "batch_size": 16,
    "device": "auto",
    "precision": "float32"
}

# Feedback system configuration
FEEDBACK_CONFIG = {
    "enable_feedback": True,
    "rating_scale": 5,
    "feedback_log_file": "feedback_logs.jsonl",
    "anonymize_feedback": True,
    "feedback_retention_days": 90,
    "metrics_update_interval": 3600  # seconds
}

# Performance and caching configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_ttl_seconds": 3600,
    "max_cache_size_mb": 1000,
    "lazy_loading": True,
    "gc_threshold": 0.8,  # Memory usage threshold for garbage collection
    "max_concurrent_processes": 4,
    "timeout_seconds": 300,
    "progressive_loading_threshold": 1000,  # Use progressive loading for datasets larger than this
    "memory_monitoring_interval": 15,  # More frequent monitoring for 16GB systems
    "memory_optimization_enabled": True,
    "embedding_compression": "float32",  # Options: float64, float32, float16, int8
    "batch_processing_enabled": True,
    "memory_mapped_indices": True,  # Use memory-mapped files for large indices
    "performance_benchmarking": True,
    "gc_tuning_enabled": True,
    "memory_pressure_threshold": 70  # Lower threshold for 16GB systems
}

# Model download configuration
MODEL_DOWNLOAD_CONFIG = {
    "use_auth_token": False,
    "cache_dir": str(MODELS_DIR),
    "force_download": False,
    "resume_download": True,
    "local_files_only": False,  # Set to True for offline mode
    "revision": "main"
}

# Error handling configuration
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay_seconds": 1,
    "graceful_degradation": True,
    "fallback_to_cpu": True,
    "continue_on_error": True,
    "error_log_level": "ERROR"
}

# Security configuration
SECURITY_CONFIG = {
    "validate_file_paths": True,
    "sanitize_inputs": True,
    "max_upload_size_mb": 100,
    "allowed_file_extensions": [".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png", ".wav", ".mp3"],
    "quarantine_suspicious_files": True,
    "audit_all_operations": True
}