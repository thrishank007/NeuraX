"""
Backend settings and configuration
"""

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    """Application settings"""
    paths: 'Paths'
    evaluation: 'EvaluationSettings'

@dataclass
class Paths:
    """Path settings"""
    data_dir: Path = Path("data")
    
@dataclass
class EvaluationSettings:
    """Evaluation settings"""
    enabled: bool = True
    sample_rate: float = 0.1
    track_latency_metrics: bool = True
    track_retrieval_metrics: bool = True
    track_generation_metrics: bool = True

settings = Settings(
    paths=Paths(),
    evaluation=EvaluationSettings()
)
