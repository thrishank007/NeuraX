"""
Core dependencies for NeuraX FastAPI backend
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from core.settings import settings
from services.neurax_service import NeuraXService
from services.evaluation_service import EvaluationService

security = HTTPBearer()

# Global service instances (singleton pattern)
_neurax_service: Optional[NeuraXService] = None
_evaluation_service: Optional[EvaluationService] = None


def get_neurax_service() -> NeuraXService:
    """Get or create NeuraX service instance"""
    global _neurax_service
    if _neurax_service is None:
        _neurax_service = NeuraXService()
    return _neurax_service


def get_evaluation_service() -> EvaluationService:
    """Get or create Evaluation service instance"""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (placeholder for now)"""
    # TODO: Implement proper JWT token verification
    # For now, just return the credentials
    return credentials.credentials