"""
Health Models

Pydantic models for health check endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class HealthStatusEnum(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status of a single component"""
    
    name: str = Field(..., description="Component name")
    status: HealthStatusEnum = Field(default=HealthStatusEnum.UNKNOWN)
    is_healthy: bool = Field(default=False)
    
    # Details
    message: Optional[str] = None
    latency_ms: Optional[float] = Field(default=None, ge=0.0)
    
    # Metadata
    version: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)
    
    # Additional info
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Basic health status response"""
    
    model_config = ConfigDict(from_attributes=True)
    
    status: HealthStatusEnum = Field(default=HealthStatusEnum.UNKNOWN)
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    uptime: float = Field(default=0.0, ge=0.0, description="Uptime in seconds")
    version: str = Field(default="2.0.0")
    
    # Component summary
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="Component name -> is_healthy mapping"
    )
    
    # Errors
    errors: List[str] = Field(default_factory=list)


class SystemInfo(BaseModel):
    """System information"""
    
    # OS info
    os: str = Field(default="unknown")
    os_version: str = Field(default="unknown")
    hostname: str = Field(default="unknown")
    
    # Python info
    python_version: str = Field(default="unknown")
    
    # Hardware
    cpu_count: int = Field(default=0, ge=0)
    total_memory_gb: float = Field(default=0.0, ge=0.0)
    
    # Process info
    process_id: int = Field(default=0, ge=0)
    process_memory_mb: float = Field(default=0.0, ge=0.0)


class DependencyHealth(BaseModel):
    """Health of external dependencies"""
    
    # LM Studio
    lm_studio: ComponentHealth = Field(default_factory=lambda: ComponentHealth(name="lm_studio"))
    
    # ChromaDB
    chroma_db: ComponentHealth = Field(default_factory=lambda: ComponentHealth(name="chroma_db"))
    
    # File system
    file_system: ComponentHealth = Field(default_factory=lambda: ComponentHealth(name="file_system"))
    
    # Embedding models
    embedding_models: ComponentHealth = Field(default_factory=lambda: ComponentHealth(name="embedding_models"))


class DetailedHealth(BaseModel):
    """Detailed health status response"""
    
    model_config = ConfigDict(from_attributes=True)
    
    # Basic info
    status: HealthStatusEnum = Field(default=HealthStatusEnum.UNKNOWN)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    version: str = Field(default="2.0.0")
    environment: str = Field(default="unknown")
    
    # System info
    system: SystemInfo = Field(default_factory=SystemInfo)
    
    # Component health
    components: List[ComponentHealth] = Field(default_factory=list)
    
    # Dependency health
    dependencies: DependencyHealth = Field(default_factory=DependencyHealth)
    
    # Resource usage
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    disk_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Recent activity
    requests_last_minute: int = Field(default=0, ge=0)
    errors_last_minute: int = Field(default=0, ge=0)
    average_response_time_ms: float = Field(default=0.0, ge=0.0)
    
    # Configuration validation
    config_valid: bool = Field(default=True)
    config_warnings: List[str] = Field(default_factory=list)
    
    # Errors and warnings
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ReadinessResponse(BaseModel):
    """Kubernetes readiness probe response"""
    
    ready: bool = Field(default=False)
    status: str = Field(default="not_ready")
    
    # Component readiness
    components_ready: Dict[str, bool] = Field(default_factory=dict)
    
    # Reasons for not being ready
    reasons: List[str] = Field(default_factory=list)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LivenessResponse(BaseModel):
    """Kubernetes liveness probe response"""
    
    alive: bool = Field(default=True)
    status: str = Field(default="alive")
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    
    # Basic health indicators
    memory_ok: bool = Field(default=True)
    disk_ok: bool = Field(default=True)
    
    # Heartbeat info
    last_heartbeat: Optional[datetime] = None
    heartbeat_interval_seconds: int = Field(default=30)
