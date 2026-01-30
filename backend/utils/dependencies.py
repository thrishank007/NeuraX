"""
Dependency injection utilities for FastAPI
"""

from typing import Optional
from backend.services.neurax_service import NeuraxService

# Global service instance
_neurax_service: Optional[NeuraxService] = None


def get_neurax_service() -> NeuraxService:
    """Get or create NeuraxService instance"""
    global _neurax_service
    if _neurax_service is None:
        _neurax_service = NeuraxService()
    return _neurax_service
