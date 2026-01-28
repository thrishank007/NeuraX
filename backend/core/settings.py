"""
Core settings configuration for NeuraX FastAPI backend
"""

from typing import Optional
from pydantic import BaseSettings, Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "NeuraX Backend"
    app_version: str = "2.0.0"
    debug: bool = False
    api_v1_str: str = "/api/v1"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: list = ["http://localhost:3000", "http://localhost:3001"]
    
    # Database (for future use)
    database_url: Optional[str] = None
    
    # LM Studio Integration
    lm_studio_host: str = "127.0.0.1"
    lm_studio_port: int = 1234
    lm_studio_base_url: str = "http://127.0.0.1:1234"
    
    # Vector Database
    vector_db_dir: str = "./vector_db"
    chroma_collection_name: str = "neurax_documents"
    
    # File Upload
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_types: list = [
        ".pdf", ".doc", ".docx", ".txt", ".md",
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp",
        ".wav", ".mp3", ".m4a", ".flac", ".ogg"
    ]
    
    # Evaluation settings
    evaluation_enabled: bool = True
    evaluation_sample_rate: float = 0.1
    track_latency_metrics: bool = True
    track_retrieval_metrics: bool = True
    track_generation_metrics: bool = True
    
    # Paths
    @property
    def paths(self):
        """Path configurations"""
        return type('Paths', (), {
            'data_dir': Path("./data"),
            'evaluations_dir': Path("./data/evaluations"),
            'models_dir': Path("./models"),
            'vector_db_dir': Path(self.vector_db_dir)
        })()
    
    # Evaluation settings as properties
    @property
    def evaluation(self):
        """Evaluation configurations"""
        return type('EvaluationConfig', (), {
            'enabled': self.evaluation_enabled,
            'sample_rate': self.evaluation_sample_rate,
            'track_latency_metrics': self.track_latency_metrics,
            'track_retrieval_metrics': self.track_retrieval_metrics,
            'track_generation_metrics': self.track_generation_metrics
        })()
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()