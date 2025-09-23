"""
Knowledge Graph Manager for SecureInsight

Implements NetworkX-based knowledge graph construction, entity linking,
and visualization export for security analysis and dashboard integration.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass, asdict, field
from loguru import logger
import pickle

from config import KG_CONFIG


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph"""
    node_id: str
    node_type: str  # 'document', 'entity', 'concept'
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph"""
    source_id: str
    target_id: str
    edge_type: str  # 'contains', 'references', 'similar_to', 'derived_from'
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class KnowledgeGraphManager:
    """
    Manages the construction and maintenance of a security-focused knowledge graph
    using NetworkX for entity linking and cross-modal document relationships.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Knowledge Graph Manager
        
        Args:
            config: Configuration dictionary, defaults to KG_CONFIG
        """
        self.config = config or KG_CONFIG
        self.graph = nx.MultiDiGraph()
        self.node_embeddings = {}
        self.entity_index = {}  # Maps content hashes to node IDs
        self.similarity_threshold = self.config.get('SIMILARITY_THRESHOLD', 0.7)
        self.max_nodes = self.config.get('MAX_NODES', 10000)
        
        logger.info("KnowledgeGraphManager initialized")
    
    def build_graph_from_documents(self, documents: List[Dict[str, Any]]) -> nx.MultiDiGraph:
        """
        Build knowledge graph from a list of processed documents
        
        Args:
            documents: List of processed document dictionaries
            
        Returns:
            NetworkX MultiDiGraph representing the knowledge graph
        """
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        try:
            # Clear existing graph
            self.graph.clear()
            self.node_embeddings.clear()
            self.entity_index.clear()
            
            # Process each document
            for doc in documents:
                self.add_document_to_graph(doc)
            
            # Create cross-document relationships
            self._create_cross_document_links()
            
            logger.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            return self.graph
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise
    
    def add_document_to_graph(self, document: Dict[str, Any]) -> None:
        """
        Add a single document to the knowledge graph
        
        Args:
            document: Processed document dictionary with content, metadata, and embeddings
        """
        try:
            # Create document node
            doc_id = self._generate_node_id(document.get('file_path', ''), 'document')
            
            doc_node = GraphNode(
                node_id=doc_id,
                node_type='document',
                content=document.get('content', ''),
                embedding=document.get('embedding'),
                metadata={
                    'file_path': document.get('file_path', ''),
                    'file_type': document.get('file_type', ''),
                    'timestamp': document.get('timestamp', datetime.now().isoformat()),
                    'size': len(document.get('content', '')),
                    'source_type': document.get('file_type', 'unknown')
                }
            )
            
            # Add document node to graph
            self._add_node_to_graph(doc_node)
            
            # Extract and link entities from document content
            entities = self._extract_entities(document)
            for entity in entities:
                entity_node = self._create_entity_node(entity, document)
                self._add_node_to_graph(entity_node)
                
                # Create edge from document to entity
                edge = GraphEdge(
                    source_id=doc_id,
                    target_id=entity_node.node_id,
                    edge_type='contains',
                    confidence=entity.get('confidence', 0.8),
                    metadata={'extraction_method': 'content_analysis'}
                )
                self._add_edge_to_graph(edge)
            
            # Handle multimodal content
            if document.get('ocr_text'):
                ocr_node = GraphNode(
                    node_id=self._generate_node_id(doc_id, 'ocr'),
                    node_type='concept',
                    content=document['ocr_text'],
                    metadata={'derived_from': 'ocr', 'parent_document': doc_id}
                )
                self._add_node_to_graph(ocr_node)
                
                edge = GraphEdge(
                    source_id=doc_id,
                    target_id=ocr_node.node_id,
                    edge_type='derived_from',
                    confidence=0.9
                )
                self._add_edge_to_graph(edge)
            
            if document.get('transcription'):
                transcript_node = GraphNode(
                    node_id=self._generate_node_id(doc_id, 'transcript'),
                    node_type='concept',
                    content=document['transcription'],
                    metadata={'derived_from': 'speech_to_text', 'parent_document': doc_id}
                )
                self._add_node_to_graph(transcript_node)
                
                edge = GraphEdge(
                    source_id=doc_id,
                    target_id=transcript_node.node_id,
                    edge_type='derived_from',
                    confidence=0.85
                )
                self._add_edge_to_graph(edge)
                
        except Exception as e:
            logger.error(f"Error adding document to graph: {str(e)}")
            raise
    
    def _add_node_to_graph(self, node: GraphNode) -> None:
        """Add a node to the NetworkX graph with proper attributes"""
        # Check if node already exists to avoid duplicates
        if node.node_id in self.graph.nodes():
            return
            
        if self.graph.number_of_nodes() >= self.max_nodes:
            logger.warning(f"Maximum nodes ({self.max_nodes}) reached, skipping node addition")
            return
            
        # Convert node to dictionary for NetworkX
        node_attrs = asdict(node)
        
        # Handle numpy arrays for embeddings
        if node.embedding is not None:
            self.node_embeddings[node.node_id] = node.embedding
            node_attrs['has_embedding'] = True
        else:
            node_attrs['has_embedding'] = False
        
        # Remove embedding from attributes (stored separately)
        node_attrs.pop('embedding', None)
        
        # Convert datetime to string
        if isinstance(node_attrs['timestamp'], datetime):
            node_attrs['timestamp'] = node_attrs['timestamp'].isoformat()
        
        self.graph.add_node(node.node_id, **node_attrs)
        
        # Update entity index for similarity matching
        content_hash = self._hash_content(node.content)
        self.entity_index[content_hash] = node.node_id
    
    def _add_edge_to_graph(self, edge: GraphEdge) -> None:
        """Add an edge to the NetworkX graph with proper attributes"""
        edge_attrs = asdict(edge)
        
        # Convert datetime to string
        if isinstance(edge_attrs['timestamp'], datetime):
            edge_attrs['timestamp'] = edge_attrs['timestamp'].isoformat()
        
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=f"{edge.edge_type}_{edge.source_id}_{edge.target_id}",
            **edge_attrs
        )
    
    def _extract_entities(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from document content using simple heuristics
        
        Args:
            document: Document dictionary
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        content = document.get('content', '')
        
        if not content:
            return entities
        
        # Simple entity extraction - can be enhanced with NLP libraries
        # Extract potential entities (capitalized words, numbers, etc.)
        words = content.split()
        
        for i, word in enumerate(words):
            # Clean word of punctuation for better matching
            clean_word = word.strip('.,!?;:"()[]{}')
            
            # Capitalized words (potential proper nouns)
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append({
                    'text': clean_word,
                    'type': 'proper_noun',
                    'confidence': 0.6,
                    'position': i
                })
            
            # Numbers (potential identifiers, dates, etc.)
            if clean_word.isdigit() and len(clean_word) >= 4:
                entities.append({
                    'text': clean_word,
                    'type': 'identifier',
                    'confidence': 0.7,
                    'position': i
                })
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity['text'] not in seen:
                seen.add(entity['text'])
                unique_entities.append(entity)
        
        return unique_entities[:50]  # Limit to prevent graph explosion
    
    def _create_entity_node(self, entity: Dict[str, Any], source_doc: Dict[str, Any]) -> GraphNode:
        """Create a graph node from an extracted entity"""
        entity_id = self._generate_node_id(entity['text'], 'entity')
        
        return GraphNode(
            node_id=entity_id,
            node_type='entity',
            content=entity['text'],
            confidence=entity.get('confidence', 0.5),
            metadata={
                'entity_type': entity.get('type', 'unknown'),
                'source_document': source_doc.get('file_path', ''),
                'position': entity.get('position', 0)
            }
        )
    
    def _create_cross_document_links(self) -> None:
        """Create similarity-based links between documents and entities"""
        logger.info("Creating cross-document similarity links")
        
        nodes_with_embeddings = [
            node_id for node_id in self.graph.nodes()
            if node_id in self.node_embeddings
        ]
        
        # Calculate pairwise similarities
        for i, node1_id in enumerate(nodes_with_embeddings):
            for node2_id in nodes_with_embeddings[i+1:]:
                # Skip if nodes don't exist (may have been skipped due to max_nodes limit)
                if node1_id not in self.graph.nodes() or node2_id not in self.graph.nodes():
                    continue
                    
                similarity = self._calculate_similarity(
                    self.node_embeddings[node1_id],
                    self.node_embeddings[node2_id]
                )
                
                if similarity > self.similarity_threshold:
                    edge = GraphEdge(
                        source_id=node1_id,
                        target_id=node2_id,
                        edge_type='similar_to',
                        weight=similarity,
                        confidence=similarity,
                        metadata={'similarity_score': similarity}
                    )
                    self._add_edge_to_graph(edge)
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def _generate_node_id(self, content: str, node_type: str) -> str:
        """Generate a unique node ID based on content and type"""
        content_hash = self._hash_content(content)
        return f"{node_type}_{content_hash[:12]}"
    
    def _hash_content(self, content: str) -> str:
        """Generate a hash for content deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def export_viz_data(self) -> Dict[str, Any]:
        """
        Export graph data for visualization in dashboard
        
        Returns:
            Dictionary containing nodes and edges data for visualization
        """
        try:
            nodes_data = []
            edges_data = []
            
            # Export nodes
            for node_id in self.graph.nodes():
                node_attrs = self.graph.nodes[node_id]
                
                nodes_data.append({
                    'id': node_id,
                    'label': node_attrs.get('content', '')[:50] + '...' if len(node_attrs.get('content', '')) > 50 else node_attrs.get('content', ''),
                    'type': node_attrs.get('node_type', 'unknown'),
                    'confidence': node_attrs.get('confidence', 1.0),
                    'size': min(max(len(node_attrs.get('content', '')), 10), 50),
                    'color': self._get_node_color(node_attrs.get('node_type', 'unknown')),
                    'metadata': node_attrs.get('metadata', {})
                })
            
            # Export edges
            for source, target, key, edge_attrs in self.graph.edges(keys=True, data=True):
                edges_data.append({
                    'source': source,
                    'target': target,
                    'type': edge_attrs.get('edge_type', 'unknown'),
                    'weight': edge_attrs.get('weight', 1.0),
                    'confidence': edge_attrs.get('confidence', 1.0),
                    'color': self._get_edge_color(edge_attrs.get('edge_type', 'unknown'))
                })
            
            return {
                'nodes': nodes_data,
                'edges': edges_data,
                'stats': {
                    'total_nodes': len(nodes_data),
                    'total_edges': len(edges_data),
                    'node_types': self._get_node_type_counts(),
                    'edge_types': self._get_edge_type_counts()
                }
            }
            
        except Exception as e:
            logger.error(f"Error exporting visualization data: {str(e)}")
            return {'nodes': [], 'edges': [], 'stats': {}}
    
    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type in visualization"""
        color_map = {
            'document': '#3498db',  # Blue
            'entity': '#e74c3c',    # Red
            'concept': '#2ecc71',   # Green
            'unknown': '#95a5a6'    # Gray
        }
        return color_map.get(node_type, '#95a5a6')
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Get color for edge type in visualization"""
        color_map = {
            'contains': '#34495e',     # Dark gray
            'similar_to': '#9b59b6',   # Purple
            'derived_from': '#f39c12', # Orange
            'references': '#1abc9c',   # Teal
            'unknown': '#bdc3c7'       # Light gray
        }
        return color_map.get(edge_type, '#bdc3c7')
    
    def _get_node_type_counts(self) -> Dict[str, int]:
        """Get count of nodes by type"""
        counts = {}
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id].get('node_type', 'unknown')
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _get_edge_type_counts(self) -> Dict[str, int]:
        """Get count of edges by type"""
        counts = {}
        for _, _, _, edge_attrs in self.graph.edges(keys=True, data=True):
            edge_type = edge_attrs.get('edge_type', 'unknown')
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'node_types': self._get_node_type_counts(),
            'edge_types': self._get_edge_type_counts()
        }
    
    def save_graph(self, filepath: Path) -> None:
        """Save the graph to disk"""
        try:
            graph_data = {
                'graph': nx.node_link_data(self.graph),
                'embeddings': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                              for k, v in self.node_embeddings.items()},
                'entity_index': self.entity_index,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(graph_data, f)
                
            logger.info(f"Graph saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
            raise
    
    def load_graph(self, filepath: Path) -> None:
        """Load the graph from disk"""
        try:
            with open(filepath, 'rb') as f:
                graph_data = pickle.load(f)
            
            self.graph = nx.node_link_graph(graph_data['graph'])
            self.node_embeddings = {k: np.array(v) if isinstance(v, list) else v 
                                   for k, v in graph_data['embeddings'].items()}
            self.entity_index = graph_data['entity_index']
            
            logger.info(f"Graph loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            raise

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the knowledge graph using the AnomalyDetector
        
        Returns:
            List of anomaly reports as dictionaries
        """
        try:
            from kg_security.anomaly_detector import AnomalyDetector
            
            detector = AnomalyDetector(self.config)
            anomaly_reports = detector.detect_anomalies(self.graph)
            
            # Convert to dictionaries for easier handling
            anomalies = []
            for report in anomaly_reports:
                anomaly_dict = asdict(report)
                # Convert datetime to string for JSON serialization
                if isinstance(anomaly_dict['timestamp'], datetime):
                    anomaly_dict['timestamp'] = anomaly_dict['timestamp'].isoformat()
                anomalies.append(anomaly_dict)
            
            logger.info(f"Detected {len(anomalies)} anomalies in knowledge graph")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []

    def quarantine_nodes(self, node_ids: List[str]) -> None:
        """
        Quarantine suspicious nodes by marking them and optionally isolating them
        
        Args:
            node_ids: List of node IDs to quarantine
        """
        try:
            quarantined_count = 0
            
            for node_id in node_ids:
                if node_id in self.graph.nodes():
                    # Mark node as quarantined
                    self.graph.nodes[node_id]['quarantined'] = True
                    self.graph.nodes[node_id]['quarantine_timestamp'] = datetime.now().isoformat()
                    self.graph.nodes[node_id]['quarantine_reason'] = 'anomaly_detection'
                    
                    # Log security event
                    self._log_security_event(
                        event_type='node_quarantine',
                        node_id=node_id,
                        description=f"Node {node_id} quarantined due to anomaly detection",
                        severity='medium'
                    )
                    
                    quarantined_count += 1
                    logger.warning(f"Quarantined node: {node_id}")
            
            logger.info(f"Quarantined {quarantined_count} nodes")
            
        except Exception as e:
            logger.error(f"Error quarantining nodes: {str(e)}")
            raise

    def generate_audit_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive audit report for security analysis
        
        Returns:
            Dictionary containing audit report data
        """
        try:
            # Get basic graph statistics
            stats = self.get_graph_stats()
            
            # Get quarantined nodes
            quarantined_nodes = []
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                if node_data.get('quarantined', False):
                    quarantined_nodes.append({
                        'node_id': node_id,
                        'quarantine_timestamp': node_data.get('quarantine_timestamp'),
                        'quarantine_reason': node_data.get('quarantine_reason'),
                        'node_type': node_data.get('node_type'),
                        'content_preview': node_data.get('content', '')[:100] + '...' if len(node_data.get('content', '')) > 100 else node_data.get('content', '')
                    })
            
            # Get security events from logs
            security_events = self._get_recent_security_events()
            
            # Detect current anomalies
            current_anomalies = self.detect_anomalies()
            
            # Calculate security metrics
            security_metrics = self._calculate_security_metrics()
            
            audit_report = {
                'report_id': f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'graph_statistics': stats,
                'quarantined_nodes': quarantined_nodes,
                'security_events': security_events,
                'current_anomalies': current_anomalies,
                'security_metrics': security_metrics,
                'recommendations': self._generate_security_recommendations(current_anomalies, quarantined_nodes)
            }
            
            logger.info(f"Generated audit report with {len(current_anomalies)} anomalies and {len(quarantined_nodes)} quarantined nodes")
            return audit_report
            
        except Exception as e:
            logger.error(f"Error generating audit report: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def simulate_tamper_detection(self, tampered_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate tamper detection for demonstration purposes
        
        Args:
            tampered_data: Dictionary containing tampered data scenarios
            
        Returns:
            Dictionary containing tamper detection results
        """
        try:
            from kg_security.anomaly_detector import AnomalyDetector
            
            detector = AnomalyDetector(self.config)
            
            # Get tamper scenarios from input or use defaults
            scenarios = tampered_data.get('scenarios', ['node_injection', 'edge_manipulation', 'content_modification'])
            
            # Run tamper simulation
            tamper_reports = detector.simulate_tamper_detection(self.graph, scenarios)
            
            # Convert reports to dictionaries
            tamper_results = []
            for report in tamper_reports:
                result_dict = asdict(report)
                if isinstance(result_dict['timestamp'], datetime):
                    result_dict['timestamp'] = result_dict['timestamp'].isoformat()
                tamper_results.append(result_dict)
            
            # Log security event for simulation
            self._log_security_event(
                event_type='tamper_simulation',
                description=f"Tamper simulation completed with {len(scenarios)} scenarios",
                severity='info',
                metadata={'scenarios': scenarios, 'detected_anomalies': len(tamper_results)}
            )
            
            simulation_result = {
                'simulation_id': f"tamper_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'scenarios_tested': scenarios,
                'detected_anomalies': tamper_results,
                'summary': {
                    'total_scenarios': len(scenarios),
                    'anomalies_detected': len(tamper_results),
                    'detection_rate': len(tamper_results) / len(scenarios) if scenarios else 0
                }
            }
            
            logger.info(f"Tamper simulation completed: {len(tamper_results)} anomalies detected from {len(scenarios)} scenarios")
            return simulation_result
            
        except Exception as e:
            logger.error(f"Error in tamper simulation: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def integrate_feedback_analysis(self, feedback_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Integrate feedback analysis results with knowledge graph security layer.
        
        Args:
            feedback_analysis: Results from feedback system pattern analysis
            
        Returns:
            List of security events generated from feedback analysis
        """
        try:
            security_events = []
            
            # Process flagged feedback for anomaly detection
            flagged_analysis = feedback_analysis.get('flagged_analysis', {})
            if flagged_analysis.get('flagged_percentage', 0) > 15:  # More than 15% flagged
                event = {
                    'event_type': 'high_feedback_flagging',
                    'description': f"High percentage of flagged feedback detected: {flagged_analysis['flagged_percentage']:.1f}%",
                    'severity': 'medium',
                    'metadata': {
                        'flagged_percentage': flagged_analysis['flagged_percentage'],
                        'total_flagged': flagged_analysis.get('total_flagged', 0),
                        'flagging_reasons': flagged_analysis.get('flagging_reasons_distribution', {})
                    }
                }
                security_events.append(event)
                self._log_security_event(**event)
            
            # Process performance correlation issues
            performance_correlation = feedback_analysis.get('performance_correlation', {})
            if performance_correlation.get('performance_issues_detected', False):
                event = {
                    'event_type': 'performance_correlation_issue',
                    'description': 'Performance issues detected through feedback correlation analysis',
                    'severity': 'medium',
                    'metadata': {
                        'processing_time_correlation': performance_correlation.get('processing_time_correlation', 0),
                        'similarity_score_correlation': performance_correlation.get('similarity_score_correlation', 0)
                    }
                }
                security_events.append(event)
                self._log_security_event(**event)
            
            # Process problematic queries
            problematic_queries = feedback_analysis.get('problematic_queries', [])
            if len(problematic_queries) > 3:  # Multiple problematic query patterns
                event = {
                    'event_type': 'multiple_problematic_queries',
                    'description': f'Multiple query patterns with poor ratings detected: {len(problematic_queries)} patterns',
                    'severity': 'low',
                    'metadata': {
                        'problematic_query_count': len(problematic_queries),
                        'average_rating': sum(q.get('average_rating', 0) for q in problematic_queries) / len(problematic_queries) if problematic_queries else 0
                    }
                }
                security_events.append(event)
                self._log_security_event(**event)
                
            # Process rating trends
            rating_trends = feedback_analysis.get('rating_trends', {})
            if rating_trends.get('trend_direction') == 'declining':
                event = {
                    'event_type': 'declining_rating_trend',
                    'description': 'Declining rating trend detected in user feedback',
                    'severity': 'medium',
                    'metadata': {
                        'trend_direction': rating_trends['trend_direction'],
                        'overall_average': rating_trends.get('overall_average', 0)
                    }
                }
                security_events.append(event)
                self._log_security_event(**event)
           
            logger.info(f"Integrated feedback analysis: generated {len(security_events)} security events")
            return security_events
            
        except Exception as e:
            logger.error(f"Error integrating feedback analysis: {str(e)}")
            return []
    
    def correlate_feedback_with_graph_anomalies(self, feedback_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Correlate feedback-flagged items with knowledge graph anomalies.
        
        Args:
            feedback_items: List of flagged feedback items
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            # Detect current graph anomalies
            graph_anomalies = self.detect_anomalies()
            
            # Analyze temporal correlation
            temporal_correlation = self._analyze_temporal_correlation(feedback_items, graph_anomalies)
            
            # Analyze content correlation
            content_correlation = self._analyze_content_correlation(feedback_items, graph_anomalies)
            
            # Generate insights
            correlation_insights = self._generate_correlation_insights(
                temporal_correlation, content_correlation, feedback_items, graph_anomalies
            )
            
            correlation_result = {
                'correlation_timestamp': datetime.now().isoformat(),
                'feedback_items_analyzed': len(feedback_items),
                'graph_anomalies_detected': len(graph_anomalies),
                'temporal_correlation': temporal_correlation,
                'content_correlation': content_correlation,
                'correlation_insights': correlation_insights,
                'recommended_actions': self._generate_correlation_recommendations(correlation_insights)
            }
            
            # Log correlation event
            self._log_security_event(
                event_type='feedback_anomaly_correlation',
                description=f"Correlation analysis completed: {len(feedback_items)} feedback items vs {len(graph_anomalies)} anomalies",
                severity='info',
                metadata={
                    'correlation_strength': temporal_correlation.get('correlation_strength', 0),
                    'insights_count': len(correlation_insights)
                }
            )
            
            logger.info(f"Completed feedback-anomaly correlation analysis with {len(correlation_insights)} insights")
            return correlation_result
            
        except Exception as e:
            logger.error(f"Error correlating feedback with graph anomalies: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_temporal_correlation(self, feedback_items: List[Dict[str, Any]], graph_anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal correlation between feedback and anomalies"""
        try:
            # Parse timestamps
            feedback_times = []
            for item in feedback_items:
                try:
                    timestamp = datetime.fromisoformat(item.get('timestamp', ''))
                    feedback_times.append(timestamp)
                except (ValueError, KeyError):
                    continue
            
            anomaly_times = []
            for anomaly in graph_anomalies:
                try:
                    timestamp = datetime.fromisoformat(anomaly.get('timestamp', ''))
                    anomaly_times.append(timestamp)
                except (ValueError, KeyError):
                    continue
            
            if not feedback_times or not anomaly_times:
                return {'correlation_strength': 0, 'temporal_matches': 0}
            
            # Find temporal clusters (within time window)
            time_window_hours = 1
            matches = 0
            
            for fb_time in feedback_times:
                for an_time in anomaly_times:
                    time_diff = abs((fb_time - an_time).total_seconds()) / 3600  # Convert to hours
                    if time_diff <= time_window_hours:
                        matches += 1
            
            correlation_strength = matches / max(len(feedback_times), len(anomaly_times))
            
            return {
                'correlation_strength': correlation_strength,
                'temporal_matches': matches,
                'feedback_count': len(feedback_times),
                'anomaly_count': len(anomaly_times),
                'time_window_hours': time_window_hours
            }
            
        except Exception as e:
            logger.error(f"Error analyzing temporal correlation: {str(e)}")
            return {'correlation_strength': 0, 'temporal_matches': 0}
    
    def _analyze_content_correlation(self, feedback_items: List[Dict[str, Any]], graph_anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content-based correlation between feedback and anomalies"""
        try:
            content_matches = 0
            total_comparisons = 0
            
            for feedback_item in feedback_items:
                feedback_query = feedback_item.get('query', '').lower()
                feedback_comments = feedback_item.get('comments', '').lower()
                feedback_content = f"{feedback_query} {feedback_comments}"
                
                for anomaly in graph_anomalies:
                    total_comparisons += 1
                    anomaly_description = anomaly.get('description', '').lower()
                    anomaly_node_content = anomaly.get('node_content', '').lower()
                    anomaly_content = f"{anomaly_description} {anomaly_node_content}"
                    
                    # Simple content similarity check
                    feedback_words = set(feedback_content.split())
                    anomaly_words = set(anomaly_content.split())
                    
                    if feedback_words and anomaly_words:
                        overlap = len(feedback_words.intersection(anomaly_words))
                        similarity = overlap / len(feedback_words.union(anomaly_words))
                        
                        if similarity > 0.1:  # 10% content similarity threshold
                            content_matches += 1
            
            correlation_strength = content_matches / total_comparisons if total_comparisons > 0 else 0
            
            return {
                'correlation_strength': correlation_strength,
                'content_matches': content_matches,
                'total_comparisons': total_comparisons,
                'similarity_threshold': 0.1
            }
            
        except Exception as e:
            logger.error(f"Error analyzing content correlation: {str(e)}")
            return {'correlation_strength': 0, 'content_matches': 0}
    
    def _generate_correlation_insights(self, temporal_correlation: Dict, content_correlation: Dict, 
                                     feedback_items: List, graph_anomalies: List) -> List[Dict[str, Any]]:
        """Generate insights from correlation analysis"""
        insights = []
        
        # Temporal correlation insights
        temporal_strength = temporal_correlation.get('correlation_strength', 0)
        if temporal_strength > 0.2:
            insights.append({
                'type': 'temporal_correlation',
                'severity': 'medium' if temporal_strength > 0.5 else 'low',
                'description': f"Strong temporal correlation detected ({temporal_strength:.2f})",
                'details': f"Found {temporal_correlation.get('temporal_matches', 0)} temporal matches",
                'confidence': min(temporal_strength * 2, 1.0)
            })
        
        # Content correlation insights
        content_strength = content_correlation.get('correlation_strength', 0)
        if content_strength > 0.1:
            insights.append({
                'type': 'content_correlation',
                'severity': 'medium' if content_strength > 0.3 else 'low',
                'description': f"Content correlation detected ({content_strength:.2f})",
                'details': f"Found {content_correlation.get('content_matches', 0)} content matches",
                'confidence': min(content_strength * 3, 1.0)
            })
        
        # High flagging rate insights
        flagging_rate = len(feedback_items) / max(len(feedback_items) + len(graph_anomalies), 1)
        if flagging_rate > 0.2:
            insights.append({
                'type': 'high_flagging_rate',
                'severity': 'medium' if flagging_rate > 0.4 else 'low',
                'description': f"High feedback flagging rate detected ({flagging_rate:.2f})",
                'details': f"Flagged {len(feedback_items)} feedback items vs {len(graph_anomalies)} anomalies",
                'confidence': min(flagging_rate * 2, 1.0)
            })
        
        # Anomaly type analysis
        if graph_anomalies:
            anomaly_types = {}
            for anomaly in graph_anomalies:
                anomaly_type = anomaly.get('anomaly_type', 'unknown')
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            # Check for centrality anomalies
            if 'centrality' in anomaly_types and anomaly_types['centrality'] > 3:
                insights.append({
                    'type': 'centrality_anomalies',
                    'severity': 'medium',
                    'description': f"Multiple centrality anomalies detected",
                    'details': f"Found {anomaly_types['centrality']} centrality-based anomalies",
                    'confidence': 0.8
                })
            
            # Check for suspicious content patterns
            if 'suspicious_content' in anomaly_types:
                insights.append({
                    'type': 'suspicious_content_pattern',
                    'severity': 'high',
                    'description': f"Suspicious content patterns detected",
                    'details': f"Found {anomaly_types['suspicious_content']} suspicious content anomalies",
                    'confidence': 0.9
                })
        
        # Feedback pattern analysis
        if feedback_items:
            low_rating_count = sum(1 for item in feedback_items if item.get('rating', 5) <= 2)
            if low_rating_count > len(feedback_items) * 0.5:
                insights.append({
                    'type': 'low_rating_pattern',
                    'severity': 'medium',
                    'description': f"High proportion of low ratings",
                    'details': f"{low_rating_count} out of {len(feedback_items)} feedback items have low ratings",
                    'confidence': 0.7
                })
        
        return insights
    
    def _generate_correlation_recommendations(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on correlation insights"""
        recommendations = []
        
        for insight in insights:
            insight_type = insight.get('type')
            severity = insight.get('severity')
            
            if insight_type == 'temporal_correlation' and severity == 'medium':
                recommendations.append("Investigate temporal patterns - feedback issues may be triggering graph anomalies")
            
            elif insight_type == 'content_correlation':
                recommendations.append("Review content quality - similar issues detected in feedback and graph anomalies")
            
            elif insight_type == 'high_flagging_rate':
                recommendations.append("High flagging rate detected - consider reviewing system performance and data quality")
            
            elif insight_type == 'centrality_anomalies':
                recommendations.append("Multiple centrality anomalies detected - review graph structure and node relationships")
            
            elif insight_type == 'suspicious_content_pattern':
                recommendations.append("URGENT: Suspicious content patterns detected - immediate security review recommended")
            
            elif insight_type == 'low_rating_pattern':
                recommendations.append("High proportion of low ratings - investigate system accuracy and relevance")
        
        if not recommendations:
            recommendations.append("No significant correlations detected - continue monitoring")
        
        return recommendations
    
    def _log_security_event(self, event_type: str, description: str, severity: str = 'info', 
                           node_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a security event (placeholder - should integrate with SecurityEventLogger)"""
        try:
            # This would normally integrate with the SecurityEventLogger
            # For now, just log the event
            logger.info(f"Security Event [{severity.upper()}] {event_type}: {description}")
            if metadata:
                logger.debug(f"Event metadata: {metadata}")
        except Exception as e:
            logger.error(f"Failed to log security event: {str(e)}")

    def _get_recent_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent security events from the audit trail
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent security events
        """
        try:
            events = self.graph.graph.get('security_events', [])
            
            # Sort by timestamp (most recent first)
            sorted_events = sorted(events, key=lambda x: x['timestamp'], reverse=True)
            
            return sorted_events[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving security events: {str(e)}")
            return []

    def _calculate_security_metrics(self) -> Dict[str, Any]:
        """
        Calculate security-related metrics for the knowledge graph
        
        Returns:
            Dictionary containing security metrics
        """
        try:
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()
            
            # Count quarantined nodes
            quarantined_count = sum(1 for node_id in self.graph.nodes() 
                                  if self.graph.nodes[node_id].get('quarantined', False))
            
            # Count security events by severity
            events = self.graph.graph.get('security_events', [])
            event_counts = {}
            for event in events:
                severity = event.get('severity', 'unknown')
                event_counts[severity] = event_counts.get(severity, 0) + 1
            
            # Calculate graph health metrics
            isolated_nodes = len(list(nx.isolates(self.graph)))
            
            # Calculate connectivity metrics
            if total_nodes > 0:
                quarantine_rate = quarantined_count / total_nodes
                isolation_rate = isolated_nodes / total_nodes
            else:
                quarantine_rate = 0
                isolation_rate = 0
            
            metrics = {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'quarantined_nodes': quarantined_count,
                'isolated_nodes': isolated_nodes,
                'quarantine_rate': quarantine_rate,
                'isolation_rate': isolation_rate,
                'security_events_by_severity': event_counts,
                'total_security_events': len(events),
                'graph_density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph) if total_nodes > 0 else True
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating security metrics: {str(e)}")
            return {}

    def _generate_security_recommendations(self, anomalies: List[Dict[str, Any]], 
                                         quarantined_nodes: List[Dict[str, Any]]) -> List[str]:
        """
        Generate security recommendations based on current state
        
        Args:
            anomalies: List of current anomalies
            quarantined_nodes: List of quarantined nodes
            
        Returns:
            List of security recommendations
        """
        recommendations = []
        
        try:
            # Analyze anomaly patterns
            if len(anomalies) > 10:
                recommendations.append("High number of anomalies detected. Consider reviewing data ingestion processes.")
            
            # Check for specific anomaly types
            anomaly_types = [anomaly.get('anomaly_type', '') for anomaly in anomalies]
            
            if 'high_centrality_outlier' in anomaly_types:
                recommendations.append("High centrality outliers detected. Investigate potential hub injection attacks.")
            
            if 'isolated_nodes' in anomaly_types:
                recommendations.append("Isolated nodes found. Check for data corruption or injection attempts.")
            
            if 'suspicious_content_pattern' in anomaly_types:
                recommendations.append("Suspicious content patterns detected. Review content validation processes.")
            
            if 'rapid_node_creation' in anomaly_types:
                recommendations.append("Rapid node creation detected. Implement rate limiting for data ingestion.")
            
            # Analyze quarantine status
            if len(quarantined_nodes) > 0:
                recommendations.append(f"{len(quarantined_nodes)} nodes currently quarantined. Review and validate quarantine decisions.")
            
            # General recommendations
            if not recommendations:
                recommendations.append("No immediate security concerns detected. Continue regular monitoring.")
            
            recommendations.append("Regularly backup the knowledge graph and maintain audit trails.")
            recommendations.append("Consider implementing automated response procedures for high-severity anomalies.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating security recommendations: {str(e)}")
            return ["Error generating recommendations. Manual security review recommended."]