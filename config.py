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

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, VECTOR_DB_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODELS = {
    "text": "sentence-transformers/all-MiniLM-L6-v2",
    "image": "openai/clip-vit-base-patch32",
    "multimodal": "openai/clip-vit-base-patch32"
}

LLM_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",  # Fallback for offline
    "quantization": True,
    "max_length": 512,
    "temperature": 0.7
}

WHISPER_CONFIG = {
    "model_size": "tiny",
    "language": "en"
}

# Vector database settings
CHROMA_CONFIG = {
    "persist_directory": str(VECTOR_DB_DIR),
    "collection_name": "secureinsight_collection"
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
    "enable_security_layer": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "rotation": "10 MB"
}