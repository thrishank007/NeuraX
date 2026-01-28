"""
Health check endpoints for NeuraX FastAPI backend
"""

import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from backend.services.neurax_service import NeuraXService
from backend.utils.dependencies import get_neurax_service


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: float
    uptime: float
    version: str = "2.0.0"
    components: Dict[str, bool]
    errors: list = []


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model"""
    status: str
    timestamp: float
    uptime: float
    version: str = "2.0.0"
    overall_healthy: bool
    components: Dict[str, bool]
    component_errors: list
    system_info: Dict[str, Any]


@router.get("/", response_model=HealthResponse)
async def health_check(
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """
    Basic health check endpoint
    
    Returns basic system health status and component availability.
    """
    health_status = await neurax_service.get_health_status()
    
    return HealthResponse(
        status=health_status['overall_status'],
        timestamp=time.time(),
        uptime=health_status['uptime'],
        components=health_status['component_status'],
        errors=health_status['component_errors']
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """
    Detailed health check endpoint
    
    Returns comprehensive system health information including
    component status, errors, and system information.
    """
    health_status = await neurax_service.get_health_status()
    
    # Determine overall health
    critical_components = [
        'ingestion_manager', 'embedding_manager', 'vector_store',
        'query_processor', 'citation_generator', 'kg_manager', 'feedback_system'
    ]
    
    overall_healthy = all(
        health_status['component_status'].get(comp, False) 
        for comp in critical_components
    )
    
    # System information
    system_info = {
        'initialized': health_status['initialized'],
        'critical_components_count': len(critical_components),
        'optional_components_count': len(health_status['component_status']) - len(critical_components),
        'total_errors': len(health_status['component_errors'])
    }
    
    return DetailedHealthResponse(
        status=health_status['overall_status'],
        timestamp=time.time(),
        uptime=health_status['uptime'],
        overall_healthy=overall_healthy,
        components=health_status['component_status'],
        component_errors=health_status['component_errors'],
        system_info=system_info
    )


@router.get("/ready")
async def readiness_check(
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """
    Kubernetes-style readiness probe
    
    Returns 200 if the service is ready to handle requests,
    503 if not ready.
    """
    health_status = await neurax_service.get_health_status()
    
    if health_status['overall_status'] == 'healthy' and health_status['initialized']:
        return {"status": "ready"}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe
    
    Returns 200 if the service is alive (basic endpoint responsiveness).
    """
    return {
        "status": "alive",
        "timestamp": time.time()
    }