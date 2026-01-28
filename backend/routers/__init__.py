"""
Router package initialization for NeuraX backend
"""

from . import health
from . import evaluation  
from . import documents

__all__ = ["health", "evaluation", "documents"]