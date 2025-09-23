"""
Streamlit monitoring dashboard for SecureInsight

Provides comprehensive system monitoring including:
- System performance metrics and health monitoring
- Interactive knowledge graph visualization
- Feedback metrics and trend analysis
- Real-time system status and alerts
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import threading
from loguru import logger

# Import SecureInsight components
from config import (
    STREAMLIT_CONFIG, KG_CONFIG, FEEDBACK_CONFIG, 
    VECTOR_DB_DIR, FEEDBACK_DIR, LOGS_DIR
)
from feedback.feedback_system import FeedbackSystem
from indexing.vector_store import VectorStore
from kg_security.knowledge_graph_manager import KnowledgeGraphManager


class SystemMetricsCollector:
    """Collects and manages system performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.max_history_size = 1000
        self.collection_interval = 5  # seconds
        self._running = False
        self._thread = None
    
    def start_collection(self):
        """Start background metrics collection"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self._thread.start()
            logger.info("Started system metrics collection")
    
    def stop_collection(self):
        """Stop background metrics collection"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        logger.info("Stopped system metrics collection")
    
    def _collect_metrics_loop(self):
        """Background loop for collecting metrics"""
        while self._running:
            try:
                metrics = self.collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def collect_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process-specific metrics (if available)
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics = {
                'timestamp': datetime.now(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'process_memory_mb': process_memory.rss / (1024**2),
                'process_memory_vms_mb': process_memory.vms / (1024**2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'timestamp': datetime.now(),
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'disk_percent': 0,
                'disk_used_gb': 0,
                'disk_total_gb': 0,
                'process_memory_mb': 0,
                'process_memory_vms_mb': 0
            }
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Convert metrics history to pandas DataFrame"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.metrics_history)
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics"""
        if not self.metrics_history:
            return self.collect_current_metrics()
        return self.metrics_history[-1]


class DashboardApp:
    """Main Streamlit dashboard application"""
    
    def __init__(self):
        self.metrics_collector = SystemMetricsCollector()
        self.feedback_system = None
        self.vector_store = None
        self.kg_manager = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize SecureInsight components"""
        try:
            # Initialize feedback system
            self.feedback_system = FeedbackSystem(FEEDBACK_DIR)
            
            # Initialize vector store
            self.vector_store = VectorStore(
                persist_directory=str(VECTOR_DB_DIR),
                collection_name="secureinsight_collection"
            )
            
            # Initialize knowledge graph manager
            self.kg_manager = KnowledgeGraphManager(KG_CONFIG)
            
            logger.info("Dashboard components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing dashboard components: {e}")
            st.error(f"Failed to initialize components: {e}")
    
    def run(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="SecureInsight Dashboard",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Start metrics collection
        if 'metrics_started' not in st.session_state:
            self.metrics_collector.start_collection()
            st.session_state.metrics_started = True
        
        # Sidebar navigation
        st.sidebar.title("🔍 SecureInsight Dashboard")
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["System Metrics", "Knowledge Graph", "Feedback Analysis", "System Health"]
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # Route to selected page
        if page == "System Metrics":
            self.render_system_metrics()
        elif page == "Knowledge Graph":
            self.render_knowledge_graph()
        elif page == "Feedback Analysis":
            self.render_feedback_analysis()
        elif page == "System Health":
            self.render_system_health()
    
    def render_system_metrics(self):
        """Render system performance metrics page"""
        st.title("📊 System Performance Metrics")
        
        # Get current metrics
        current_metrics = self.metrics_collector.get_latest_metrics()
        metrics_df = self.metrics_collector.get_metrics_dataframe()
        
        # Current status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="CPU Usage",
                value=f"{current_metrics['cpu_percent']:.1f}%",
                delta=self._calculate_delta(metrics_df, 'cpu_percent') if not metrics_df.empty else None
            )
        
        with col2:
            st.metric(
                label="Memory Usage",
                value=f"{current_metrics['memory_percent']:.1f}%",
                delta=self._calculate_delta(metrics_df, 'memory_percent') if not metrics_df.empty else None
            )
        
        with col3:
            st.metric(
                label="Disk Usage",
                value=f"{current_metrics['disk_percent']:.1f}%",
                delta=self._calculate_delta(metrics_df, 'disk_percent') if not metrics_df.empty else None
            )
        
        with col4:
            st.metric(
                label="Process Memory",
                value=f"{current_metrics['process_memory_mb']:.1f} MB",
                delta=self._calculate_delta(metrics_df, 'process_memory_mb') if not metrics_df.empty else None
            )
        
        # Time series charts
        if not metrics_df.empty and len(metrics_df) > 1:
            st.subheader("📈 Performance Trends")
            
            # CPU and Memory chart
            fig_cpu_mem = make_subplots(
                rows=2, cols=1,
                subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
                vertical_spacing=0.1
            )
            
            fig_cpu_mem.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['cpu_percent'],
                    mode='lines+markers',
                    name='CPU %',
                    line=dict(color='#e74c3c')
                ),
                row=1, col=1
            )
            
            fig_cpu_mem.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['memory_percent'],
                    mode='lines+markers',
                    name='Memory %',
                    line=dict(color='#3498db')
                ),
                row=2, col=1
            )
            
            fig_cpu_mem.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_cpu_mem, use_container_width=True)
            
            # Process memory chart
            fig_process = go.Figure()
            fig_process.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['process_memory_mb'],
                    mode='lines+markers',
                    name='Process Memory (MB)',
                    line=dict(color='#2ecc71')
                )
            )
            fig_process.update_layout(
                title="Process Memory Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Memory (MB)",
                height=300
            )
            st.plotly_chart(fig_process, use_container_width=True)
        
        # Vector store metrics
        st.subheader("🗄️ Vector Store Metrics")
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", vector_stats.get('total_documents', 0))
                
                # File types distribution
                if vector_stats.get('file_types'):
                    fig_files = px.pie(
                        values=list(vector_stats['file_types'].values()),
                        names=list(vector_stats['file_types'].keys()),
                        title="Document Types Distribution"
                    )
                    st.plotly_chart(fig_files, use_container_width=True)
            
            with col2:
                st.metric("Collection Name", vector_stats.get('collection_name', 'N/A'))
                
                # Embedding types distribution
                if vector_stats.get('embedding_types'):
                    fig_embeddings = px.pie(
                        values=list(vector_stats['embedding_types'].values()),
                        names=list(vector_stats['embedding_types'].keys()),
                        title="Embedding Types Distribution"
                    )
                    st.plotly_chart(fig_embeddings, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading vector store metrics: {e}")
        
        # Processing performance metrics
        st.subheader("⚡ Processing Performance")
        
        # Simulated processing metrics (would be collected from actual operations)
        processing_metrics = self._get_processing_metrics()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Query Time", f"{processing_metrics['avg_query_time']:.2f}s")
        with col2:
            st.metric("Avg Embedding Time", f"{processing_metrics['avg_embedding_time']:.2f}s")
        with col3:
            st.metric("Cache Hit Rate", f"{processing_metrics['cache_hit_rate']:.1f}%")
    
    def _calculate_delta(self, df: pd.DataFrame, column: str) -> Optional[float]:
        """Calculate delta for metrics"""
        if len(df) < 2:
            return None
        
        current = df[column].iloc[-1]
        previous = df[column].iloc[-2]
        return current - previous
    
    def _get_processing_metrics(self) -> Dict[str, float]:
        """Get processing performance metrics (simulated for now)"""
        # In a real implementation, these would be collected from actual operations
        return {
            'avg_query_time': np.random.uniform(0.5, 2.0),
            'avg_embedding_time': np.random.uniform(0.1, 0.5),
            'cache_hit_rate': np.random.uniform(70, 95)
        }
    
    def render_knowledge_graph(self):
        """Render knowledge graph visualization page"""
        st.title("🕸️ Knowledge Graph Visualization")
        
        try:
            # Get graph statistics
            stats = self.kg_manager.get_graph_stats()
            
            # Display basic statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", stats.get('nodes', 0))
            with col2:
                st.metric("Total Edges", stats.get('edges', 0))
            with col3:
                st.metric("Graph Density", f"{stats.get('density', 0):.3f}")
            with col4:
                is_connected = stats.get('is_connected', False)
                st.metric("Connected", "Yes" if is_connected else "No")
            
            # Node and edge type distributions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Node Types Distribution")
                node_types = stats.get('node_types', {})
                if node_types:
                    fig_nodes = px.pie(
                        values=list(node_types.values()),
                        names=list(node_types.keys()),
                        title="Node Types in Knowledge Graph",
                        color_discrete_map={
                            'document': '#3498db',
                            'entity': '#e74c3c',
                            'concept': '#2ecc71',
                            'unknown': '#95a5a6'
                        }
                    )
                    st.plotly_chart(fig_nodes, use_container_width=True)
                else:
                    st.info("No nodes in the knowledge graph yet.")
            
            with col2:
                st.subheader("🔗 Edge Types Distribution")
                edge_types = stats.get('edge_types', {})
                if edge_types:
                    fig_edges = px.pie(
                        values=list(edge_types.values()),
                        names=list(edge_types.keys()),
                        title="Edge Types in Knowledge Graph",
                        color_discrete_map={
                            'contains': '#34495e',
                            'similar_to': '#9b59b6',
                            'derived_from': '#f39c12',
                            'references': '#1abc9c',
                            'unknown': '#bdc3c7'
                        }
                    )
                    st.plotly_chart(fig_edges, use_container_width=True)
                else:
                    st.info("No edges in the knowledge graph yet.")
            
            # Interactive graph visualization
            st.subheader("🌐 Interactive Graph Visualization")
            
            # Visualization controls
            col1, col2, col3 = st.columns(3)
            with col1:
                layout_type = st.selectbox(
                    "Layout Algorithm",
                    ["spring", "circular", "kamada_kawai", "random"],
                    index=0
                )
            with col2:
                node_size_factor = st.slider("Node Size Factor", 0.5, 3.0, 1.0, 0.1)
            with col3:
                show_labels = st.checkbox("Show Node Labels", value=True)
            
            # Filter controls
            st.subheader("🔍 Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_node_types = st.multiselect(
                    "Node Types to Show",
                    options=list(node_types.keys()) if node_types else [],
                    default=list(node_types.keys()) if node_types else []
                )
            
            with col2:
                selected_edge_types = st.multiselect(
                    "Edge Types to Show",
                    options=list(edge_types.keys()) if edge_types else [],
                    default=list(edge_types.keys()) if edge_types else []
                )
            
            with col3:
                confidence_threshold = st.slider(
                    "Minimum Confidence",
                    0.0, 1.0, 0.5, 0.1
                )
            
            # Generate and display visualization
            if stats.get('nodes', 0) > 0:
                viz_data = self._generate_graph_visualization(
                    layout_type=layout_type,
                    node_size_factor=node_size_factor,
                    show_labels=show_labels,
                    selected_node_types=selected_node_types,
                    selected_edge_types=selected_edge_types,
                    confidence_threshold=confidence_threshold
                )
                
                if viz_data:
                    st.plotly_chart(viz_data, use_container_width=True, height=600)
                else:
                    st.warning("No nodes match the current filter criteria.")
            else:
                st.info("No graph data available for visualization. Process some documents first.")
            
            # Anomaly detection results
            st.subheader("🚨 Security Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Run Anomaly Detection"):
                    with st.spinner("Detecting anomalies..."):
                        anomalies = self.kg_manager.detect_anomalies()
                    
                    if anomalies:
                        st.warning(f"Found {len(anomalies)} potential anomalies:")
                        
                        for i, anomaly in enumerate(anomalies[:5]):  # Show first 5
                            with st.expander(f"Anomaly {i+1}: {anomaly.get('anomaly_type', 'Unknown')}"):
                                st.write(f"**Confidence:** {anomaly.get('confidence_score', 0):.2f}")
                                st.write(f"**Description:** {anomaly.get('description', 'No description')}")
                                st.write(f"**Affected Nodes:** {len(anomaly.get('node_ids', []))}")
                                
                                if anomaly.get('recommended_actions'):
                                    st.write("**Recommended Actions:**")
                                    for action in anomaly['recommended_actions']:
                                        st.write(f"- {action}")
                    else:
                        st.success("✅ No anomalies detected in the knowledge graph.")
            
            with col2:
                if st.button("🧪 Run Tamper Simulation"):
                    with st.spinner("Running tamper simulation..."):
                        tamper_data = {
                            'scenarios': ['node_injection', 'edge_manipulation', 'content_modification']
                        }
                        simulation_result = self.kg_manager.simulate_tamper_detection(tamper_data)
                    
                    if simulation_result:
                        st.info(f"Simulation completed:")
                        st.write(f"**Scenarios Tested:** {simulation_result.get('summary', {}).get('total_scenarios', 0)}")
                        st.write(f"**Anomalies Detected:** {simulation_result.get('summary', {}).get('anomalies_detected', 0)}")
                        st.write(f"**Detection Rate:** {simulation_result.get('summary', {}).get('detection_rate', 0):.1%}")
                        
                        detected_anomalies = simulation_result.get('detected_anomalies', [])
                        if detected_anomalies:
                            with st.expander("View Simulation Results"):
                                for anomaly in detected_anomalies:
                                    st.write(f"- **{anomaly.get('anomaly_type', 'Unknown')}**: {anomaly.get('description', 'No description')}")
            
            # Graph export options
            st.subheader("📤 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export Visualization Data"):
                    viz_export_data = self.kg_manager.export_viz_data()
                    
                    # Convert to JSON for download
                    json_data = json.dumps(viz_export_data, indent=2, default=str)
                    
                    st.download_button(
                        label="Download Graph Data (JSON)",
                        data=json_data,
                        file_name=f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📋 Generate Audit Report"):
                    with st.spinner("Generating audit report..."):
                        audit_report = self.kg_manager.generate_audit_report()
                    
                    if audit_report:
                        # Convert to JSON for download
                        json_data = json.dumps(audit_report, indent=2, default=str)
                        
                        st.download_button(
                            label="Download Audit Report (JSON)",
                            data=json_data,
                            file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                        # Show summary
                        st.success("Audit report generated successfully!")
                        st.write(f"**Report ID:** {audit_report.get('report_id', 'N/A')}")
                        st.write(f"**Quarantined Nodes:** {len(audit_report.get('quarantined_nodes', []))}")
                        st.write(f"**Security Events:** {len(audit_report.get('security_events', []))}")
                        st.write(f"**Current Anomalies:** {len(audit_report.get('current_anomalies', []))}")
                    
        except Exception as e:
            st.error(f"Error loading knowledge graph: {e}")
            logger.error(f"Knowledge graph visualization error: {e}")
    
    def _generate_graph_visualization(
        self,
        layout_type: str = "spring",
        node_size_factor: float = 1.0,
        show_labels: bool = True,
        selected_node_types: List[str] = None,
        selected_edge_types: List[str] = None,
        confidence_threshold: float = 0.5
    ) -> Optional[go.Figure]:
        """
        Generate interactive graph visualization using Plotly
        
        Args:
            layout_type: Layout algorithm to use
            node_size_factor: Factor to scale node sizes
            show_labels: Whether to show node labels
            selected_node_types: Node types to include
            selected_edge_types: Edge types to include
            confidence_threshold: Minimum confidence for nodes/edges
            
        Returns:
            Plotly figure or None if no data
        """
        try:
            # Get visualization data from knowledge graph manager
            viz_data = self.kg_manager.export_viz_data()
            
            if not viz_data or not viz_data.get('nodes'):
                return None
            
            # Filter nodes and edges based on criteria
            filtered_nodes = []
            filtered_edges = []
            
            # Filter nodes
            for node in viz_data['nodes']:
                if selected_node_types and node['type'] not in selected_node_types:
                    continue
                if node.get('confidence', 1.0) < confidence_threshold:
                    continue
                filtered_nodes.append(node)
            
            # Get IDs of filtered nodes for edge filtering
            node_ids = {node['id'] for node in filtered_nodes}
            
            # Filter edges
            for edge in viz_data['edges']:
                if edge['source'] not in node_ids or edge['target'] not in node_ids:
                    continue
                if selected_edge_types and edge['type'] not in selected_edge_types:
                    continue
                if edge.get('confidence', 1.0) < confidence_threshold:
                    continue
                filtered_edges.append(edge)
            
            if not filtered_nodes:
                return None
            
            # Create NetworkX graph for layout calculation
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes
            for node in filtered_nodes:
                G.add_node(node['id'], **node)
            
            # Add edges
            for edge in filtered_edges:
                G.add_edge(edge['source'], edge['target'], **edge)
            
            # Calculate layout positions
            if layout_type == "spring":
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout_type == "circular":
                pos = nx.circular_layout(G)
            elif layout_type == "kamada_kawai":
                pos = nx.kamada_kawai_layout(G) if len(G.nodes()) > 1 else nx.spring_layout(G)
            else:  # random
                pos = nx.random_layout(G)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            edge_info = []
            
            for edge in filtered_edges:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_info.append(f"{edge['type']}: {edge.get('confidence', 1.0):.2f}")
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Edges'
            ))
            
            # Add nodes
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            node_hover = []
            
            for node in filtered_nodes:
                x, y = pos[node['id']]
                node_x.append(x)
                node_y.append(y)
                
                # Node label
                label = node.get('label', node['id'])
                if show_labels:
                    node_text.append(label[:20] + '...' if len(label) > 20 else label)
                else:
                    node_text.append('')
                
                # Node color based on type
                node_colors.append(node.get('color', '#95a5a6'))
                
                # Node size based on confidence and size factor
                base_size = node.get('size', 20)
                confidence = node.get('confidence', 1.0)
                size = base_size * confidence * node_size_factor
                node_sizes.append(max(5, min(50, size)))
                
                # Hover information
                hover_text = f"<b>{label}</b><br>"
                hover_text += f"Type: {node['type']}<br>"
                hover_text += f"Confidence: {confidence:.2f}<br>"
                hover_text += f"ID: {node['id']}"
                node_hover.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                textposition="middle center",
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=node_hover,
                name='Nodes'
            ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Knowledge Graph Visualization ({len(filtered_nodes)} nodes, {len(filtered_edges)} edges)",
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Drag to pan, scroll to zoom, hover for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color='gray', size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating graph visualization: {e}")
            st.error(f"Error generating visualization: {e}")
            return None
    
    def render_feedback_analysis(self):
        """Render feedback metrics and analysis page"""
        st.title("📝 Feedback Analysis")
        
        try:
            # Get feedback metrics
            metrics = self.feedback_system.get_feedback_metrics()
            
            # Overview metrics
            st.subheader("📊 Feedback Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Feedback",
                    metrics.get('total_feedback', 0)
                )
            
            with col2:
                avg_rating = metrics.get('average_rating', 0)
                st.metric(
                    "Average Rating",
                    f"{avg_rating:.2f}/5.0",
                    delta=f"{avg_rating - 3.0:+.2f}" if avg_rating > 0 else None
                )
            
            with col3:
                st.metric(
                    "Avg Processing Time",
                    f"{metrics.get('average_processing_time', 0):.2f}s"
                )
            
            with col4:
                latest_date = metrics.get('latest_feedback_date', 'N/A')
                if latest_date != 'N/A':
                    latest_date = datetime.fromisoformat(latest_date).strftime('%Y-%m-%d')
                st.metric(
                    "Latest Feedback",
                    latest_date
                )
            
            # Rating distribution
            if metrics.get('total_feedback', 0) > 0:
                st.subheader("⭐ Rating Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Rating distribution pie chart
                    rating_dist = metrics.get('rating_distribution', {})
                    if rating_dist:
                        fig_ratings = px.pie(
                            values=list(rating_dist.values()),
                            names=[f"{star} Star{'s' if int(star) != 1 else ''}" for star in rating_dist.keys()],
                            title="Rating Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig_ratings, use_container_width=True)
                
                with col2:
                    # Rating distribution bar chart
                    if rating_dist:
                        fig_bar = px.bar(
                            x=list(rating_dist.keys()),
                            y=list(rating_dist.values()),
                            title="Rating Counts",
                            labels={'x': 'Rating', 'y': 'Count'},
                            color=list(rating_dist.values()),
                            color_continuous_scale='RdYlGn'
                        )
                        fig_bar.update_layout(showlegend=False)
                        st.plotly_chart(fig_bar, use_container_width=True)
                
                # Feedback trends over time
                st.subheader("📈 Feedback Trends")
                feedback_trends = metrics.get('feedback_trends', {})
                
                if feedback_trends:
                    # Prepare data for time series
                    dates = list(feedback_trends.keys())
                    counts = [feedback_trends[date]['count'] for date in dates]
                    avg_ratings = [feedback_trends[date]['average_rating'] for date in dates]
                    
                    # Create subplot with secondary y-axis
                    fig_trends = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Daily Feedback Count', 'Daily Average Rating'),
                        vertical_spacing=0.1
                    )
                    
                    # Feedback count over time
                    fig_trends.add_trace(
                        go.Scatter(
                            x=dates,
                            y=counts,
                            mode='lines+markers',
                            name='Feedback Count',
                            line=dict(color='#3498db')
                        ),
                        row=1, col=1
                    )
                    
                    # Average rating over time
                    fig_trends.add_trace(
                        go.Scatter(
                            x=dates,
                            y=avg_ratings,
                            mode='lines+markers',
                            name='Average Rating',
                            line=dict(color='#e74c3c')
                        ),
                        row=2, col=1
                    )
                    
                    fig_trends.update_layout(
                        height=400,
                        showlegend=False,
                        title="Feedback Trends Over Time"
                    )
                    
                    fig_trends.update_xaxes(title_text="Date", row=2, col=1)
                    fig_trends.update_yaxes(title_text="Count", row=1, col=1)
                    fig_trends.update_yaxes(title_text="Rating", row=2, col=1, range=[1, 5])
                    
                    st.plotly_chart(fig_trends, use_container_width=True)
                
                # Common themes analysis
                st.subheader("🏷️ Common Feedback Themes")
                common_themes = metrics.get('common_themes', {})
                
                if common_themes:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Themes bar chart
                        fig_themes = px.bar(
                            x=list(common_themes.values()),
                            y=list(common_themes.keys()),
                            orientation='h',
                            title="Feedback Theme Frequency",
                            labels={'x': 'Frequency', 'y': 'Theme'},
                            color=list(common_themes.values()),
                            color_continuous_scale='viridis'
                        )
                        fig_themes.update_layout(showlegend=False)
                        st.plotly_chart(fig_themes, use_container_width=True)
                    
                    with col2:
                        # Theme insights
                        st.write("**Theme Analysis:**")
                        total_themed_feedback = sum(common_themes.values())
                        
                        for theme, count in sorted(common_themes.items(), key=lambda x: x[1], reverse=True):
                            percentage = (count / total_themed_feedback) * 100 if total_themed_feedback > 0 else 0
                            st.write(f"• **{theme.title()}**: {count} mentions ({percentage:.1f}%)")
                        
                        # Recommendations based on themes
                        st.write("**Recommendations:**")
                        if 'accuracy' in common_themes and common_themes['accuracy'] > 0:
                            st.write("• Focus on improving response accuracy")
                        if 'speed' in common_themes and common_themes['speed'] > 0:
                            st.write("• Optimize processing performance")
                        if 'relevance' in common_themes and common_themes['relevance'] > 0:
                            st.write("• Enhance search relevance algorithms")
                
                # Performance correlation analysis
                st.subheader("🔗 Performance Correlation Analysis")
                
                # Get recent feedback for correlation analysis
                recent_feedback = self.feedback_system.get_recent_feedback(days=30)
                
                if recent_feedback:
                    # Prepare data for correlation analysis
                    ratings = [f.rating for f in recent_feedback]
                    processing_times = [f.processing_time for f in recent_feedback]
                    similarity_scores = []
                    
                    for f in recent_feedback:
                        if f.similarity_scores:
                            similarity_scores.append(max(f.similarity_scores))
                        else:
                            similarity_scores.append(0)
                    
                    # Create correlation plots
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Rating vs Processing Time
                        fig_corr1 = px.scatter(
                            x=processing_times,
                            y=ratings,
                            title="Rating vs Processing Time",
                            labels={'x': 'Processing Time (s)', 'y': 'Rating'},
                            trendline="ols"
                        )
                        st.plotly_chart(fig_corr1, use_container_width=True)
                    
                    with col2:
                        # Rating vs Similarity Score
                        if any(score > 0 for score in similarity_scores):
                            fig_corr2 = px.scatter(
                                x=similarity_scores,
                                y=ratings,
                                title="Rating vs Max Similarity Score",
                                labels={'x': 'Max Similarity Score', 'y': 'Rating'},
                                trendline="ols"
                            )
                            st.plotly_chart(fig_corr2, use_container_width=True)
                        else:
                            st.info("No similarity score data available for correlation analysis.")
                    
                    # Correlation coefficients
                    st.write("**Correlation Analysis:**")
                    
                    if len(ratings) > 1 and len(processing_times) > 1:
                        corr_time = np.corrcoef(ratings, processing_times)[0, 1]
                        st.write(f"• Rating vs Processing Time: {corr_time:.3f}")
                        
                        if corr_time < -0.3:
                            st.warning("⚠️ Strong negative correlation: Longer processing times lead to lower ratings")
                        elif corr_time > 0.3:
                            st.info("ℹ️ Positive correlation: Users may prefer thorough processing")
                    
                    if len(ratings) > 1 and any(score > 0 for score in similarity_scores):
                        valid_indices = [i for i, score in enumerate(similarity_scores) if score > 0]
                        if len(valid_indices) > 1:
                            valid_ratings = [ratings[i] for i in valid_indices]
                            valid_scores = [similarity_scores[i] for i in valid_indices]
                            corr_sim = np.corrcoef(valid_ratings, valid_scores)[0, 1]
                            st.write(f"• Rating vs Similarity Score: {corr_sim:.3f}")
                            
                            if corr_sim > 0.3:
                                st.success("✅ Positive correlation: Higher similarity scores lead to better ratings")
                            elif corr_sim < -0.3:
                                st.warning("⚠️ Negative correlation: Review similarity scoring algorithm")
                
                # Feedback export and integration
                st.subheader("📤 Feedback Management")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Export Feedback Data"):
                        try:
                            export_path = self.feedback_system.export_anonymized_feedback()
                            
                            # Read the exported file for download
                            with open(export_path, 'r', encoding='utf-8') as f:
                                export_data = f.read()
                            
                            st.download_button(
                                label="Download Anonymized Feedback (JSON)",
                                data=export_data,
                                file_name=f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            
                            st.success(f"✅ Feedback exported successfully!")
                            
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                
                with col2:
                    if st.button("🔍 Flag for Anomaly Detection"):
                        try:
                            # Get recent low-rated feedback for anomaly detection
                            flagged_count = 0
                            
                            for feedback in recent_feedback:
                                if self.feedback_system.flag_for_anomaly_detection(feedback):
                                    flagged_count += 1
                            
                            if flagged_count > 0:
                                st.warning(f"⚠️ {flagged_count} feedback entries flagged for anomaly detection")
                                
                                # Integrate with knowledge graph if available
                                if hasattr(self.kg_manager, 'detect_anomalies'):
                                    st.info("Running knowledge graph anomaly detection...")
                                    anomalies = self.kg_manager.detect_anomalies()
                                    
                                    if anomalies:
                                        st.write(f"Found {len(anomalies)} potential anomalies in knowledge graph")
                            else:
                                st.success("✅ No feedback entries require anomaly detection")
                                
                        except Exception as e:
                            st.error(f"Anomaly detection failed: {e}")
                
                with col3:
                    # Time-saved metrics calculation
                    if st.button("⏱️ Calculate Time Saved"):
                        try:
                            time_saved_metrics = self._calculate_time_saved_metrics(recent_feedback)
                            
                            st.metric(
                                "Estimated Time Saved",
                                f"{time_saved_metrics['total_time_saved']:.1f} hours",
                                delta=f"{time_saved_metrics['avg_time_per_query']:.1f}s per query"
                            )
                            
                            st.write(f"**Baseline:** {time_saved_metrics['manual_time_estimate']:.1f}s per manual search")
                            st.write(f"**Automated:** {time_saved_metrics['avg_processing_time']:.1f}s per automated query")
                            st.write(f"**Efficiency Gain:** {time_saved_metrics['efficiency_gain']:.1%}")
                            
                        except Exception as e:
                            st.error(f"Time calculation failed: {e}")
            
            else:
                st.info("No feedback data available yet. Users need to provide feedback on system responses.")
                
                # Show sample feedback form
                st.subheader("📝 Sample Feedback Form")
                st.write("This is how users would provide feedback:")
                
                with st.form("sample_feedback"):
                    sample_query = st.text_input("Query", value="Sample query about documents")
                    sample_response = st.text_area("Response", value="Sample system response")
                    sample_rating = st.slider("Rating", 1, 5, 4)
                    sample_comments = st.text_area("Comments", value="The response was helpful and accurate")
                    
                    if st.form_submit_button("Submit Sample Feedback"):
                        try:
                            feedback_id = self.feedback_system.collect_feedback(
                                query=sample_query,
                                response=sample_response,
                                rating=sample_rating,
                                comments=sample_comments,
                                processing_time=1.5,
                                similarity_scores=[0.85, 0.72, 0.68]
                            )
                            st.success(f"✅ Sample feedback submitted! ID: {feedback_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to submit feedback: {e}")
                    
        except Exception as e:
            st.error(f"Error loading feedback analysis: {e}")
            logger.error(f"Feedback analysis error: {e}")
    
    def _calculate_time_saved_metrics(self, feedback_list: List) -> Dict[str, float]:
        """
        Calculate time-saved metrics based on feedback data
        
        Args:
            feedback_list: List of feedback data
            
        Returns:
            Dictionary containing time-saved metrics
        """
        if not feedback_list:
            return {
                'total_time_saved': 0.0,
                'avg_time_per_query': 0.0,
                'manual_time_estimate': 300.0,  # 5 minutes baseline
                'avg_processing_time': 0.0,
                'efficiency_gain': 0.0
            }
        
        # Calculate average processing time
        processing_times = [f.processing_time for f in feedback_list]
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Estimate manual search time (baseline)
        # Assume manual search takes 5-10 minutes depending on complexity
        manual_time_estimate = 300.0  # 5 minutes in seconds
        
        # Calculate time saved per query
        time_saved_per_query = manual_time_estimate - avg_processing_time
        
        # Calculate total time saved
        total_time_saved_seconds = time_saved_per_query * len(feedback_list)
        total_time_saved_hours = total_time_saved_seconds / 3600
        
        # Calculate efficiency gain
        efficiency_gain = time_saved_per_query / manual_time_estimate if manual_time_estimate > 0 else 0
        
        return {
            'total_time_saved': total_time_saved_hours,
            'avg_time_per_query': time_saved_per_query,
            'manual_time_estimate': manual_time_estimate,
            'avg_processing_time': avg_processing_time,
            'efficiency_gain': efficiency_gain
        }
    
    def render_system_health(self):
        """Render system health monitoring page"""
        st.title("🏥 System Health Monitor")
        
        # Health status indicators
        health_status = self._check_system_health()
        
        # Overall health score
        overall_score = np.mean(list(health_status.values()))
        health_color = "🟢" if overall_score > 0.8 else "🟡" if overall_score > 0.6 else "🔴"
        
        st.metric(
            label=f"{health_color} Overall System Health",
            value=f"{overall_score:.1%}"
        )
        
        # Individual component health
        st.subheader("Component Health Status")
        
        for component, status in health_status.items():
            status_icon = "✅" if status > 0.8 else "⚠️" if status > 0.6 else "❌"
            st.metric(
                label=f"{status_icon} {component.replace('_', ' ').title()}",
                value=f"{status:.1%}"
            )
        
        # System alerts
        st.subheader("🚨 System Alerts")
        alerts = self._get_system_alerts(health_status)
        
        if alerts:
            for alert in alerts:
                st.warning(f"⚠️ {alert}")
        else:
            st.success("✅ No active alerts")
        
        # Resource usage warnings
        current_metrics = self.metrics_collector.get_latest_metrics()
        
        if current_metrics['cpu_percent'] > 80:
            st.error(f"🔥 High CPU usage: {current_metrics['cpu_percent']:.1f}%")
        
        if current_metrics['memory_percent'] > 85:
            st.error(f"💾 High memory usage: {current_metrics['memory_percent']:.1f}%")
        
        if current_metrics['disk_percent'] > 90:
            st.error(f"💿 High disk usage: {current_metrics['disk_percent']:.1f}%")
    
    def _check_system_health(self) -> Dict[str, float]:
        """Check health status of system components"""
        health_status = {}
        
        try:
            # Check vector store health
            vector_stats = self.vector_store.get_collection_stats()
            health_status['vector_store'] = 1.0 if vector_stats else 0.0
        except:
            health_status['vector_store'] = 0.0
        
        try:
            # Check feedback system health
            feedback_metrics = self.feedback_system.get_feedback_metrics()
            health_status['feedback_system'] = 1.0 if feedback_metrics is not None else 0.0
        except:
            health_status['feedback_system'] = 0.0
        
        try:
            # Check knowledge graph health
            kg_stats = self.kg_manager.get_graph_stats()
            health_status['knowledge_graph'] = 1.0 if kg_stats else 0.0
        except:
            health_status['knowledge_graph'] = 0.0
        
        # Check system resources
        current_metrics = self.metrics_collector.get_latest_metrics()
        health_status['cpu_health'] = max(0, (100 - current_metrics['cpu_percent']) / 100)
        health_status['memory_health'] = max(0, (100 - current_metrics['memory_percent']) / 100)
        health_status['disk_health'] = max(0, (100 - current_metrics['disk_percent']) / 100)
        
        return health_status
    
    def _get_system_alerts(self, health_status: Dict[str, float]) -> List[str]:
        """Generate system alerts based on health status"""
        alerts = []
        
        for component, status in health_status.items():
            if status < 0.5:
                alerts.append(f"{component.replace('_', ' ').title()} is experiencing issues")
            elif status < 0.8:
                alerts.append(f"{component.replace('_', ' ').title()} performance is degraded")
        
        return alerts


def main():
    """Main entry point for the Streamlit dashboard"""
    try:
        dashboard = DashboardApp()
        dashboard.run()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")


if __name__ == "__main__":
    main()