"""
Optional monitoring UI components.

Product UI is the Next.js workspace (frontend/) talking to FastAPI (backend/).
Streamlit remains available only as an optional analytics dashboard.
"""

__all__ = []

try:
    from .streamlit_dashboard import DashboardApp, SystemMetricsCollector

    __all__.extend(["DashboardApp", "SystemMetricsCollector"])
except ImportError:
    pass
