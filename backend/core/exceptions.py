"""
Backend exceptions
"""

from enum import Enum

class ErrorCode(str, Enum):
    """Error codes"""
    UNKNOWN = "UNKNOWN"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"

class EvaluationError(Exception):
    """Evaluation error"""
    pass
