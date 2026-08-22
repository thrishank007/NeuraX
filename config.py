"""
Configuration settings for NeuraX RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env")
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

# LM Studio Configuration - Replaces all HuggingFace model management
LM_STUDIO_CONFIG = {
    # LM Studio server configuration
    "base_url": "http://localhost:1234/v1",  # LM Studio API endpoint
    "timeout": 120,  # Request timeout in seconds
    
    # Model configuration
    "gemma_model": "google/gemma-3n",  # Main multimodal model (has vision capability)
    "qwen_model": "qwen/qwen3-4b-thinking-2507",  # Fallback for text-only (thinking mode)
    
    # Generation parameters
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "max_context_length": 4096,
    
    # Model switching preferences
    "prefer_gemma_for_multimodal": True,  # Use Gemma when images are involved
    "prefer_qwen_for_reasoning": True,  # Use Qwen for complex text reasoning
    
    # Auto-switching based on query type
    "auto_model_switching": True,
    "multimodal_keywords": ["image", "picture", "photo", "visual", "diagram", "chart"],
    "reasoning_keywords": ["analyze", "explain", "reasoning", "think", "logic", "step"],
    
    # Fallback configuration
    "enable_fallback": True,  # Try alternative model if primary fails
    "fallback_timeout": 30  # Shorter timeout for fallback attempts
}

# Cloud LLM Configuration (OpenAI-compatible endpoint)
CLOUD_LLM_CONFIG = {
    "api_url": os.getenv("NEURAX_CLOUD_API_URL", ""),
    "api_key": os.getenv("NEURAX_CLOUD_API_KEY", ""),
    "model": os.getenv("NEURAX_CLOUD_MODEL", ""),
    "max_tokens": int(os.getenv("NEURAX_CLOUD_MAX_TOKENS", "1024")),
    "temperature": float(os.getenv("NEURAX_CLOUD_TEMPERATURE", "0.7")),
    "timeout": int(os.getenv("NEURAX_CLOUD_TIMEOUT", "60")),
}

# Connectivity default flag: choose offline-first (no cloud rerank by default)
# or online-first (cloud rerank enabled by default).
CD_MODE = os.getenv("NEURAX_CD", "offline-first").strip().lower()
if CD_MODE not in {"offline-first", "online-first"}:
    CD_MODE = "offline-first"
CD_ONLINE_FIRST = CD_MODE == "online-first"

# NVIDIA NIM text embeddings (opt-in; leave disabled for the local MiniLM path)
NIM_EMBEDDING_CONFIG = {
    "enabled": os.getenv("NEURAX_NIM_EMBEDDINGS_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
    "api_url": os.getenv("NEURAX_NIM_EMBEDDING_API_URL", "https://integrate.api.nvidia.com/v1/embeddings"),
    "api_key": os.getenv("NEURAX_NIM_API_KEY", ""),
    "model": os.getenv("NEURAX_NIM_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"),
    "dimension": 2048,
    "collection_name": os.getenv("NEURAX_NIM_COLLECTION_NAME", "neurax_nim_nemotron_3_embed_1b_v1"),
    "timeout": int(os.getenv("NEURAX_NIM_EMBEDDING_TIMEOUT", "60")),
}

# Cross-encoder reranking of fused retrieval candidates (same NIM key as
# embeddings; runs after hybrid RRF, before the LLM sees context)
NIM_RERANK_CONFIG = {
    "enabled": os.getenv("NEURAX_NIM_RERANK_ENABLED", "true" if CD_ONLINE_FIRST else "false").lower() in {"1", "true", "yes", "on"},
    "api_url": os.getenv("NEURAX_NIM_RERANK_API_URL", "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"),
    "api_key": os.getenv("NEURAX_NIM_API_KEY", ""),
    # Model id must be one the account can invoke on the reranking
    # endpoint (nv-rerankqa-mistral-4b-v3 is NOT available on this account;
    # the API returns the valid list on a miss).
    "model": os.getenv("NEURAX_NIM_RERANK_MODEL", "nvidia/rerank-qa-mistral-4b"),
    "timeout": int(os.getenv("NEURAX_NIM_RERANK_TIMEOUT", "30")),
    "max_passages": int(os.getenv("NEURAX_NIM_RERANK_MAX_PASSAGES", "20")),
}

# CLIP image vector store (always local — 512-d, separate from text collection)
CLIP_IMAGE_CONFIG = {
    "model": "openai/clip-vit-base-patch32",
    "dimension": 512,
    "collection_name": os.getenv("NEURAX_CLIP_COLLECTION_NAME", "neurax_clip_images_v1"),
}

# Legacy LLM_CONFIG for backward compatibility (now redirects to LM Studio)
LLM_CONFIG = {
    # Deprecated - Use LM_STUDIO_CONFIG instead
    "use_lm_studio": True,  # Flag to use LM Studio instead of HuggingFace
    "model_name": "gemma-3n",  # Simplified name for LM Studio
    "fallback_model": "qwen3-4b-thinking",
    "max_length": 1024,
    "temperature": 0.7,
    "use_multimodal": True,  # Enabled since Gemma supports vision
    # All other HuggingFace-specific settings are ignored when use_lm_studio=True
    "force_cpu": False,  # Not applicable for LM Studio
    "device": "auto",  # Handled by LM Studio
    "quantization": False,  # Handled by LM Studio
    "trust_remote_code": True,  # Not applicable
    "max_new_tokens": 1024,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
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
    "collection_name": "neurax_with_docs",
    "embedding_function": None,  # Will be set by EmbeddingManager
    "distance_function": "cosine",
    "hnsw_space": "cosine",
    "anonymized_telemetry": False,
    "allow_reset": True
}

# UI settings — product UI is Next.js + FastAPI (see frontend/, backend/)
FRONTEND_CONFIG = {
    "url": "http://127.0.0.1:3000",
    "api_url": "http://127.0.0.1:8000",
}

STREAMLIT_CONFIG = {
    "server_port": 8501,
    "server_address": "127.0.0.1",
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

# Graphify document knowledge graph (optional external CLI — not the security NetworkX graph)
# Install separately: uv tool install "graphifyy[openai]"  or  pipx install "graphifyy[openai]"
GRAPHIFY_CONFIG = {
    "enabled": True,
    "executable": os.getenv("GRAPHIFY_EXECUTABLE", "graphify"),
    "workspace_dir": DATA_DIR / "graphify",
    "default_workspace_id": "default",
    "backend": "openai",
    "base_url": os.getenv("GRAPHIFY_OPENAI_BASE_URL", LM_STUDIO_CONFIG["base_url"]),
    "api_key": os.getenv("GRAPHIFY_OPENAI_API_KEY", "lm-studio"),
    "model": os.getenv("GRAPHIFY_MODEL", LM_STUDIO_CONFIG["qwen_model"]),
    "mode": "deep",
    "auto_update_after_ingestion": False,
    "max_concurrency": 1,
    "api_timeout_seconds": 900,
    "process_timeout_seconds": 1800,
    "token_budget": 2000,  # ponytail: Qwen 4B output cap ~908 tok; small chunks prevent hollow responses
    "max_query_output_chars": 12000,
    "max_query_length": 2000,
    "max_visualization_nodes": 1000,
    "max_graph_json_bytes": 50 * 1024 * 1024,
    "max_stdout_capture_bytes": 2 * 1024 * 1024,
    "max_stderr_capture_bytes": 1 * 1024 * 1024,
    "max_rag_context_chars": 4000,
    "max_rag_context_nodes": 25,
    "max_rag_context_edges": 40,
    "allow_non_local_endpoint": False,
    "install_hint": (
        'Graphify is not installed. Install with:\n'
        '  uv tool install "graphifyy[openai]"\n'
        '  # or\n'
        '  pipx install "graphifyy[openai]"\n'
        "Requires Python 3.10+. NeuraX itself can still run without Graphify."
    ),
}

# Search and retrieval settings
SIMILARITY_THRESHOLD = 0.5
SEARCH_CONFIG = {
    "default_k": 5,
    "max_results": 50,
    "similarity_threshold": SIMILARITY_THRESHOLD,
    "enable_query_rewrite": False,  # Opt-in: rewrites queries via LLM before embedding
    "enable_hybrid": True,          # BM25 + dense RRF fusion (disable if corpus is empty)
    "bm25_k": 20,                   # Candidates fetched from BM25 before RRF merge
    "rrf_k": 20,                    # RRF constant (higher = gentler rank penalty)
    "dense_weight": 0.9,            # Dense-favored fusion: equal weights let BM25
    "sparse_weight": 0.1,           #   misses outrank correct dense picks (see evals/)
    "enable_reranking": CD_ONLINE_FIRST,  # NIM cross-encoder rerank of fused candidates
    "rerank_candidates": 20,        # Fused candidates sent to the reranker
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
