"""
User Interface Components

This module provides the user interface components including:
- Gradio chat interface for document upload and querying
- Streamlit dashboard for system monitoring and visualization
- Interactive knowledge graph visualization
- Voice input and multimodal query support
"""

# Import available components
__all__ = []

try:
    from .gradio_app import create_gradio_interface
    __all__.append('create_gradio_interface')
except ImportError:
    pass

try:
    from .streamlit_dashboard import DashboardApp, SystemMetricsCollector
    __all__.extend(['DashboardApp', 'SystemMetricsCollector'])
except ImportError:
    pass