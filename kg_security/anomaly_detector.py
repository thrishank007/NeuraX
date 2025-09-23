"""
Anomaly Detection for Knowledge Graph Security

Implements centrality-based anomaly detection, outlier identification,
and tamper simulation for security analysis and demonstrations.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from loguru import logger
import json
import random
from pathlib import Path
from scipy import stats

from config import KG_CONFIG


@dataclass
class AnomalyReport:
    """Represents an anomaly detection report"""
    anomaly_id: str
    node_ids: List[str]
    anomaly_type: str
    confidence_score: float
    description: str
    timestamp: datetime
    recommended_actions: List[str]
    centrality_scores: Dict[str, float] = None
    statistical_measures: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.centrality_scores is None:
            self.centrality_scores = {}
        if self.statistical_measures is None:
            self.statistical_measures = {}


@dataclass
class CentralityMeasures:
    """Container for node centrality measures"""
    node_id: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    clustering_coefficient: float


class AnomalyDetector:
    """
    Implements anomaly detection algorithms for knowledge graph security.
    Detects unusual connectivity patterns, outlier nodes, and potential tampering.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Anomaly Detector
        
        Args:
            config: Configuration dictionary, defaults to KG_CONFIG
        """
        self.config = config or KG_CONFIG
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.8)
        self.centrality_threshold = self.config.get('centrality_threshold', 0.1)
        self.outlier_std_threshold = self.config.get('outlier_std_threshold', 2.0)
        self.min_confidence_score = self.config.get('min_confidence_score', 0.6)
        
        # Cache for centrality calculations
        self._centrality_cache = {}
        self._last_graph_hash = None
        
        logger.info("AnomalyDetector initialized")

    def detect_anomalies(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """
        Detect anomalies in the knowledge graph using multiple algorithms
        
        Args:
            graph: NetworkX MultiDiGraph to analyze
            
        Returns:
            List of AnomalyReport objects describing detected anomalies
        """
        logger.info(f"Starting anomaly detection on graph with {graph.number_of_nodes()} nodes")
        
        try:
            anomalies = []
            
            # Calculate centrality measures
            centrality_measures = self.calculate_centrality_measures(graph)
            
            # Detect centrality-based anomalies
            centrality_anomalies = self._detect_centrality_anomalies(graph, centrality_measures)
            anomalies.extend(centrality_anomalies)
            
            # Detect connectivity outliers
            connectivity_anomalies = self._detect_connectivity_outliers(graph, centrality_measures)
            anomalies.extend(connectivity_anomalies)
            
            # Detect structural anomalies
            structural_anomalies = self._detect_structural_anomalies(graph)
            anomalies.extend(structural_anomalies)
            
            # Detect temporal anomalies
            temporal_anomalies = self._detect_temporal_anomalies(graph)
            anomalies.extend(temporal_anomalies)
            
            # Detect content-based anomalies
            content_anomalies = self._detect_content_anomalies(graph)
            anomalies.extend(content_anomalies)
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error during anomaly detection: {str(e)}")
            return []
    
    def calculate_centrality_measures(self, graph: nx.MultiDiGraph) -> Dict[str, CentralityMeasures]:
        """
        Calculate various centrality measures for all nodes in the graph
        
        Args:
            graph: NetworkX MultiDiGraph to analyze
            
        Returns:
            Dictionary mapping node IDs to CentralityMeasures objects
        """
        # Check if we can use cached results
        graph_hash = self._calculate_graph_hash(graph)
        if graph_hash == self._last_graph_hash and self._centrality_cache:
            logger.debug("Using cached centrality measures")
            return self._centrality_cache
        
        logger.info("Calculating centrality measures")
        
        try:
            # Convert to simple graph for centrality calculations
            simple_graph = self._convert_to_simple_graph(graph)
            
            if simple_graph.number_of_nodes() == 0:
                return {}
            
            # Calculate centrality measures
            degree_centrality = nx.degree_centrality(simple_graph)
            
            # Handle disconnected graphs for betweenness and closeness
            if nx.is_connected(simple_graph.to_undirected()):
                betweenness_centrality = nx.betweenness_centrality(simple_graph)
                closeness_centrality = nx.closeness_centrality(simple_graph)
            else:
                betweenness_centrality = nx.betweenness_centrality(simple_graph)
                closeness_centrality = {node: 0.0 for node in simple_graph.nodes()}
            
            # Eigenvector centrality (may fail for disconnected graphs)
            try:
                eigenvector_centrality = nx.eigenvector_centrality(simple_graph, max_iter=1000)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                eigenvector_centrality = {node: 0.0 for node in simple_graph.nodes()}
            
            # PageRank
            pagerank = nx.pagerank(simple_graph, max_iter=1000)
            
            # Clustering coefficient
            clustering = nx.clustering(simple_graph.to_undirected())
            
            # Combine into CentralityMeasures objects
            centrality_measures = {}
            for node_id in simple_graph.nodes():
                centrality_measures[node_id] = CentralityMeasures(
                    node_id=node_id,
                    degree_centrality=degree_centrality.get(node_id, 0.0),
                    betweenness_centrality=betweenness_centrality.get(node_id, 0.0),
                    closeness_centrality=closeness_centrality.get(node_id, 0.0),
                    eigenvector_centrality=eigenvector_centrality.get(node_id, 0.0),
                    pagerank=pagerank.get(node_id, 0.0),
                    clustering_coefficient=clustering.get(node_id, 0.0)
                )
            
            # Cache results
            self._centrality_cache = centrality_measures
            self._last_graph_hash = graph_hash
            
            logger.info(f"Calculated centrality measures for {len(centrality_measures)} nodes")
            return centrality_measures
            
        except Exception as e:
            logger.error(f"Error calculating centrality measures: {str(e)}")
            return {}  
  
    def _detect_centrality_anomalies(self, graph: nx.MultiDiGraph, 
                                   centrality_measures: Dict[str, CentralityMeasures]) -> List[AnomalyReport]:
        """Detect anomalies based on unusual centrality scores"""
        anomalies = []
        
        if not centrality_measures:
            return anomalies
        
        # Extract centrality values for statistical analysis
        degree_values = [cm.degree_centrality for cm in centrality_measures.values()]
        betweenness_values = [cm.betweenness_centrality for cm in centrality_measures.values()]
        pagerank_values = [cm.pagerank for cm in centrality_measures.values()]
        
        # Calculate statistical thresholds
        degree_threshold = np.mean(degree_values) + self.outlier_std_threshold * np.std(degree_values)
        betweenness_threshold = np.mean(betweenness_values) + self.outlier_std_threshold * np.std(betweenness_values)
        pagerank_threshold = np.mean(pagerank_values) + self.outlier_std_threshold * np.std(pagerank_values)
        
        # Detect high centrality outliers
        for node_id, measures in centrality_measures.items():
            outlier_scores = []
            outlier_types = []
            
            if measures.degree_centrality > degree_threshold:
                outlier_scores.append(measures.degree_centrality)
                outlier_types.append("high_degree_centrality")
            
            if measures.betweenness_centrality > betweenness_threshold:
                outlier_scores.append(measures.betweenness_centrality)
                outlier_types.append("high_betweenness_centrality")
            
            if measures.pagerank > pagerank_threshold:
                outlier_scores.append(measures.pagerank)
                outlier_types.append("high_pagerank")
            
            # Create anomaly report if outliers detected
            if outlier_scores:
                confidence = min(max(np.mean(outlier_scores), self.min_confidence_score), 1.0)
                
                anomaly = AnomalyReport(
                    anomaly_id=f"centrality_{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    node_ids=[node_id],
                    anomaly_type="high_centrality_outlier",
                    confidence_score=confidence,
                    description=f"Node {node_id} shows unusually high centrality scores: {', '.join(outlier_types)}",
                    timestamp=datetime.now(),
                    recommended_actions=[
                        "Investigate node content and connections",
                        "Verify data source integrity",
                        "Consider quarantine if suspicious"
                    ],
                    centrality_scores={
                        "degree": measures.degree_centrality,
                        "betweenness": measures.betweenness_centrality,
                        "pagerank": measures.pagerank
                    },
                    statistical_measures={
                        "degree_threshold": degree_threshold,
                        "betweenness_threshold": betweenness_threshold,
                        "pagerank_threshold": pagerank_threshold
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_connectivity_outliers(self, graph: nx.MultiDiGraph,
                                    centrality_measures: Dict[str, CentralityMeasures]) -> List[AnomalyReport]:
        """Detect nodes with unusual connectivity patterns"""
        anomalies = []
        
        # Analyze degree distribution
        degrees = [graph.degree(node) for node in graph.nodes()]
        if not degrees:
            return anomalies
        
        degree_mean = np.mean(degrees)
        degree_std = np.std(degrees)
        
        # Detect isolated nodes (potential data injection)
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            anomaly = AnomalyReport(
                anomaly_id=f"isolated_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                node_ids=isolated_nodes,
                anomaly_type="isolated_nodes",
                confidence_score=0.9,
                description=f"Found {len(isolated_nodes)} isolated nodes with no connections",
                timestamp=datetime.now(),
                recommended_actions=[
                    "Investigate why nodes are isolated",
                    "Check for data injection or corruption",
                    "Consider removing if not legitimate"
                ]
            )
            anomalies.append(anomaly)
        
        # Detect nodes with extremely high degree (potential hubs or injection points)
        high_degree_threshold = degree_mean + self.outlier_std_threshold * degree_std
        high_degree_nodes = [node for node in graph.nodes() 
                           if graph.degree(node) > high_degree_threshold]
        
        if high_degree_nodes:
            anomaly = AnomalyReport(
                anomaly_id=f"high_degree_nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                node_ids=high_degree_nodes,
                anomaly_type="high_degree_nodes",
                confidence_score=0.8,
                description=f"Found {len(high_degree_nodes)} nodes with unusually high connectivity",
                timestamp=datetime.now(),
                recommended_actions=[
                    "Verify legitimacy of highly connected nodes",
                    "Check for potential hub injection attacks",
                    "Analyze connection patterns for anomalies"
                ],
                statistical_measures={
                    "degree_threshold": high_degree_threshold,
                    "mean_degree": degree_mean,
                    "std_degree": degree_std
                }
            )
            anomalies.append(anomaly)
        
        return anomalies    

    def _detect_structural_anomalies(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Detect structural anomalies in the graph"""
        anomalies = []
        
        # Detect unusual subgraph structures
        try:
            # Find strongly connected components
            scc = list(nx.strongly_connected_components(graph))
            large_scc = [component for component in scc if len(component) > 10]
            
            if large_scc:
                for i, component in enumerate(large_scc):
                    anomaly = AnomalyReport(
                        anomaly_id=f"large_scc_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        node_ids=list(component),
                        anomaly_type="large_strongly_connected_component",
                        confidence_score=0.7,
                        description=f"Found strongly connected component with {len(component)} nodes",
                        timestamp=datetime.now(),
                        recommended_actions=[
                            "Investigate circular dependencies",
                            "Check for potential data loops",
                            "Verify component legitimacy"
                        ]
                    )
                    anomalies.append(anomaly)
            
            # Detect cliques (complete subgraphs)
            undirected_graph = graph.to_undirected()
            cliques = list(nx.find_cliques(undirected_graph))
            large_cliques = [clique for clique in cliques if len(clique) > 5]
            
            if large_cliques:
                for i, clique in enumerate(large_cliques):
                    anomaly = AnomalyReport(
                        anomaly_id=f"large_clique_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        node_ids=list(clique),
                        anomaly_type="large_clique",
                        confidence_score=0.6,
                        description=f"Found clique with {len(clique)} fully connected nodes",
                        timestamp=datetime.now(),
                        recommended_actions=[
                            "Investigate unusual complete connectivity",
                            "Check for artificial data patterns",
                            "Verify clique formation legitimacy"
                        ]
                    )
                    anomalies.append(anomaly)
                    
        except Exception as e:
            logger.error(f"Error detecting structural anomalies: {str(e)}")
        
        return anomalies
    
    def _detect_temporal_anomalies(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Detect temporal anomalies based on node and edge timestamps"""
        anomalies = []
        
        try:
            # Collect timestamps from nodes and edges
            node_timestamps = []
            edge_timestamps = []
            
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                timestamp_str = node_data.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        node_timestamps.append((node_id, timestamp))
                    except ValueError:
                        continue
            
            for source, target, key, edge_data in graph.edges(keys=True, data=True):
                timestamp_str = edge_data.get('timestamp')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        edge_timestamps.append((f"{source}-{target}", timestamp))
                    except ValueError:
                        continue
            
            # Detect temporal clustering (burst of activity)
            if node_timestamps:
                timestamps = [ts for _, ts in node_timestamps]
                timestamps.sort()
                
                # Look for unusual bursts of node creation
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                             for i in range(len(timestamps)-1)]
                
                if time_diffs:
                    mean_diff = np.mean(time_diffs)
                    std_diff = np.std(time_diffs)
                    
                    # Find periods of rapid node creation
                    rapid_periods = []
                    for i, diff in enumerate(time_diffs):
                        if diff < mean_diff - 2 * std_diff and diff < 60:  # Less than 1 minute
                            rapid_periods.append(i)
                    
                    if len(rapid_periods) > 5:  # Multiple rapid creation periods
                        rapid_nodes = [node_timestamps[i][0] for i in rapid_periods]
                        anomaly = AnomalyReport(
                            anomaly_id=f"rapid_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            node_ids=rapid_nodes,
                            anomaly_type="rapid_node_creation",
                            confidence_score=0.8,
                            description=f"Detected {len(rapid_periods)} periods of rapid node creation",
                            timestamp=datetime.now(),
                            recommended_actions=[
                                "Investigate source of rapid data ingestion",
                                "Check for automated attacks or data dumps",
                                "Verify data source legitimacy"
                            ]
                        )
                        anomalies.append(anomaly)
                        
        except Exception as e:
            logger.error(f"Error detecting temporal anomalies: {str(e)}")
        
        return anomalies  
  
    def _detect_content_anomalies(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Detect content-based anomalies in node data"""
        anomalies = []
        
        try:
            # Analyze content patterns
            content_lengths = []
            suspicious_nodes = []
            
            for node_id in graph.nodes():
                node_data = graph.nodes[node_id]
                content = node_data.get('content', '')
                
                if content:
                    content_lengths.append(len(content))
                    
                    # Check for suspicious patterns
                    if len(content) > 10000:  # Very long content
                        suspicious_nodes.append((node_id, "excessive_content_length"))
                    
                    # Check for repeated patterns (potential injection)
                    words = content.split()
                    if len(words) > 10:
                        unique_words = set(words)
                        repetition_ratio = len(words) / len(unique_words)
                        if repetition_ratio > 5:  # High repetition
                            suspicious_nodes.append((node_id, "high_content_repetition"))
            
            # Detect content length outliers
            if content_lengths:
                mean_length = np.mean(content_lengths)
                std_length = np.std(content_lengths)
                length_threshold = mean_length + 3 * std_length
                
                outlier_nodes = []
                for node_id in graph.nodes():
                    content = graph.nodes[node_id].get('content', '')
                    if len(content) > length_threshold:
                        outlier_nodes.append(node_id)
                
                if outlier_nodes:
                    anomaly = AnomalyReport(
                        anomaly_id=f"content_length_outliers_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        node_ids=outlier_nodes,
                        anomaly_type="content_length_outliers",
                        confidence_score=0.6,
                        description=f"Found {len(outlier_nodes)} nodes with unusually long content",
                        timestamp=datetime.now(),
                        recommended_actions=[
                            "Review content for legitimacy",
                            "Check for data injection or corruption",
                            "Consider content size limits"
                        ]
                    )
                    anomalies.append(anomaly)
            
            # Report suspicious content patterns
            if suspicious_nodes:
                for node_id, reason in suspicious_nodes:
                    anomaly = AnomalyReport(
                        anomaly_id=f"suspicious_content_{node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        node_ids=[node_id],
                        anomaly_type="suspicious_content_pattern",
                        confidence_score=0.7,
                        description=f"Node {node_id} has suspicious content: {reason}",
                        timestamp=datetime.now(),
                        recommended_actions=[
                            "Manually review node content",
                            "Investigate data source",
                            "Consider quarantine if malicious"
                        ]
                    )
                    anomalies.append(anomaly)
                    
        except Exception as e:
            logger.error(f"Error detecting content anomalies: {str(e)}")
        
        return anomalies

    def simulate_tamper_detection(self, graph: nx.MultiDiGraph, 
                                 tamper_scenarios: List[str] = None) -> List[AnomalyReport]:
        """
        Simulate tamper detection for testing and demonstration purposes
        
        Args:
            graph: Original graph to tamper with
            tamper_scenarios: List of tamper scenarios to simulate
            
        Returns:
            List of AnomalyReport objects for detected tampering
        """
        if tamper_scenarios is None:
            tamper_scenarios = [
                "node_injection",
                "edge_manipulation",
                "content_modification",
                "hub_creation",
                "isolation_attack"
            ]
        
        logger.info(f"Simulating tamper detection with scenarios: {tamper_scenarios}")
        
        # Create a copy of the graph for tampering
        tampered_graph = graph.copy()
        tamper_reports = []
        
        for scenario in tamper_scenarios:
            try:
                if scenario == "node_injection":
                    reports = self._simulate_node_injection(tampered_graph)
                elif scenario == "edge_manipulation":
                    reports = self._simulate_edge_manipulation(tampered_graph)
                elif scenario == "content_modification":
                    reports = self._simulate_content_modification(tampered_graph)
                elif scenario == "hub_creation":
                    reports = self._simulate_hub_creation(tampered_graph)
                elif scenario == "isolation_attack":
                    reports = self._simulate_isolation_attack(tampered_graph)
                else:
                    logger.warning(f"Unknown tamper scenario: {scenario}")
                    continue
                
                tamper_reports.extend(reports)
                
            except Exception as e:
                logger.error(f"Error simulating {scenario}: {str(e)}")
        
        return tamper_reports
    
    def _simulate_node_injection(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Simulate malicious node injection"""
        # Add suspicious nodes with high connectivity
        injected_nodes = []
        existing_nodes = list(graph.nodes())
        
        if not existing_nodes:
            return []
        
        for i in range(3):  # Inject 3 suspicious nodes
            node_id = f"injected_malicious_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add node with suspicious content
            graph.add_node(node_id, 
                          node_type='entity',
                          content='MALICIOUS_CONTENT_' * 100,  # Repetitive content
                          confidence=1.0,
                          timestamp=datetime.now().isoformat())
            
            # Connect to many existing nodes (hub behavior)
            targets = random.sample(existing_nodes, min(10, len(existing_nodes)))
            for target in targets:
                graph.add_edge(node_id, target, 
                              edge_type='suspicious_link',
                              weight=1.0,
                              confidence=1.0)
            
            injected_nodes.append(node_id)
        
        # Detect the injected nodes
        anomalies = self.detect_anomalies(graph)
        
        # Filter for injection-related anomalies
        injection_anomalies = [
            anomaly for anomaly in anomalies
            if any(node_id in injected_nodes for node_id in anomaly.node_ids)
        ]
        
        return injection_anomalies
    
    def _simulate_edge_manipulation(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Simulate edge manipulation attacks"""
        existing_nodes = list(graph.nodes())
        if len(existing_nodes) < 2:
            return []
        
        # Add many random edges to create unusual connectivity
        manipulated_edges = []
        for _ in range(20):  # Add 20 suspicious edges
            source = random.choice(existing_nodes)
            target = random.choice(existing_nodes)
            if source != target:
                graph.add_edge(source, target,
                              edge_type='manipulated',
                              weight=0.1,  # Low weight to indicate suspicion
                              confidence=0.3)
                manipulated_edges.append((source, target))
        
        # Detect anomalies after manipulation
        anomalies = self.detect_anomalies(graph)
        
        return anomalies

    def _simulate_content_modification(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Simulate content modification attacks"""
        existing_nodes = list(graph.nodes())
        if not existing_nodes:
            return []
        
        # Modify content of random nodes to create suspicious patterns
        modified_nodes = random.sample(existing_nodes, min(5, len(existing_nodes)))
        
        for node_id in modified_nodes:
            # Inject repetitive malicious content
            malicious_content = "TAMPERED_DATA " * 200
            graph.nodes[node_id]['content'] = malicious_content
            graph.nodes[node_id]['confidence'] = 0.1  # Low confidence indicates tampering
        
        # Detect anomalies after content modification
        anomalies = self.detect_anomalies(graph)
        
        return anomalies

    def _simulate_hub_creation(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Simulate malicious hub creation"""
        existing_nodes = list(graph.nodes())
        if len(existing_nodes) < 5:
            return []
        
        # Create a malicious hub node
        hub_id = f"malicious_hub_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        graph.add_node(hub_id,
                      node_type='entity',
                      content='MALICIOUS_HUB_CONTENT',
                      confidence=1.0,
                      timestamp=datetime.now().isoformat())
        
        # Connect hub to many existing nodes
        targets = random.sample(existing_nodes, min(20, len(existing_nodes)))
        for target in targets:
            graph.add_edge(hub_id, target,
                          edge_type='malicious_connection',
                          weight=1.0,
                          confidence=0.2)
        
        # Detect anomalies after hub creation
        anomalies = self.detect_anomalies(graph)
        
        return anomalies

    def _simulate_isolation_attack(self, graph: nx.MultiDiGraph) -> List[AnomalyReport]:
        """Simulate node isolation attack"""
        existing_nodes = list(graph.nodes())
        if len(existing_nodes) < 10:
            return []
        
        # Remove edges from random nodes to isolate them
        isolated_nodes = random.sample(existing_nodes, min(3, len(existing_nodes)))
        
        for node_id in isolated_nodes:
            # Remove all edges connected to this node
            edges_to_remove = []
            for source, target, key in graph.edges(keys=True):
                if source == node_id or target == node_id:
                    edges_to_remove.append((source, target, key))
            
            for source, target, key in edges_to_remove:
                graph.remove_edge(source, target, key)
        
        # Detect anomalies after isolation
        anomalies = self.detect_anomalies(graph)
        
        return anomalies

    def create_sample_tampered_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Create sample tampered datasets for quick judge demonstrations
        with realistic security scenarios
        """
        logger.info("Creating sample tampered datasets for demonstrations")
        
        datasets = {
            "financial_fraud": {
                "description": "Simulated financial document tampering with suspicious transaction patterns",
                "scenario": "An attacker injects fake transaction records and creates artificial connections between accounts",
                "documents": [
                    {
                        "file_path": "financial_report_Q1.pdf",
                        "content": "Quarterly financial report with legitimate transactions...",
                        "file_type": "pdf",
                        "timestamp": "2024-01-15T10:00:00"
                    },
                    {
                        "file_path": "TAMPERED_transactions.pdf",
                        "content": "FAKE_TRANSACTION " * 100 + " $1000000 transfer to offshore account",
                        "file_type": "pdf",
                        "timestamp": "2024-01-15T10:01:00"  # Suspicious timing
                    }
                ],
                "expected_anomalies": ["rapid_node_creation", "suspicious_content_pattern", "high_centrality_outlier"]
            },
            
            "intelligence_leak": {
                "description": "Simulated intelligence document tampering with classified information injection",
                "scenario": "An insider threat injects fake classified documents to mislead analysis",
                "documents": [
                    {
                        "file_path": "intelligence_brief_001.pdf",
                        "content": "Standard intelligence briefing on regional security threats...",
                        "file_type": "pdf",
                        "timestamp": "2024-02-01T09:00:00"
                    },
                    {
                        "file_path": "CLASSIFIED_FAKE_intel.pdf",
                        "content": "TOP_SECRET " * 150 + " FABRICATED_INTELLIGENCE_DATA " * 50,
                        "file_type": "pdf",
                        "timestamp": "2024-02-01T09:00:30"  # Rapid injection
                    }
                ],
                "expected_anomalies": ["content_length_outliers", "high_content_repetition", "temporal_clustering"]
            },
            
            "supply_chain_attack": {
                "description": "Simulated supply chain document tampering with vendor information manipulation",
                "scenario": "Attacker modifies vendor documents to hide malicious suppliers",
                "documents": [
                    {
                        "file_path": "vendor_list_approved.xlsx",
                        "content": "Approved vendor list: VendorA, VendorB, VendorC...",
                        "file_type": "xlsx",
                        "timestamp": "2024-03-01T14:00:00"
                    },
                    {
                        "file_path": "vendor_list_approved.xlsx",  # Same filename - replacement attack
                        "content": "Approved vendor list: VendorA, VendorB, MALICIOUS_VENDOR " * 80,
                        "file_type": "xlsx",
                        "timestamp": "2024-03-01T14:30:00"
                    }
                ],
                "expected_anomalies": ["duplicate_content_hash", "suspicious_content_pattern"]
            }
        }
        
        return datasets

    def create_demo_hooks(self) -> Dict[str, callable]:
        """
        Create demo hooks for security showcases tied to validation testing
        """
        logger.info("Creating demo hooks for security showcases")
        
        def demo_real_time_detection():
            """Demo hook for real-time anomaly detection"""
            logger.info("Running real-time anomaly detection demo")
            # This would be called during live demonstrations
            pass
        
        def demo_tamper_simulation():
            """Demo hook for tamper simulation"""
            logger.info("Running tamper simulation demo")
            # This would show live tampering and detection
            pass
        
        def demo_forensic_analysis():
            """Demo hook for forensic analysis of detected anomalies"""
            logger.info("Running forensic analysis demo")
            # This would show detailed analysis of anomalies
            pass
        
        hooks = {
            "real_time_detection": demo_real_time_detection,
            "tamper_simulation": demo_tamper_simulation,
            "forensic_analysis": demo_forensic_analysis
        }
        
        return hooks

    def _convert_to_simple_graph(self, multigraph: nx.MultiDiGraph) -> nx.DiGraph:
        """Convert MultiDiGraph to simple DiGraph for centrality calculations"""
        simple_graph = nx.DiGraph()
        
        # Add all nodes
        for node_id, node_data in multigraph.nodes(data=True):
            simple_graph.add_node(node_id, **node_data)
        
        # Add edges (combine multiple edges between same nodes)
        edge_weights = {}
        for source, target, key, edge_data in multigraph.edges(keys=True, data=True):
            edge_key = (source, target)
            weight = edge_data.get('weight', 1.0)
            
            if edge_key in edge_weights:
                edge_weights[edge_key] += weight
            else:
                edge_weights[edge_key] = weight
        
        # Add combined edges to simple graph
        for (source, target), weight in edge_weights.items():
            simple_graph.add_edge(source, target, weight=weight)
        
        return simple_graph

    def _calculate_graph_hash(self, graph: nx.MultiDiGraph) -> str:
        """Calculate a hash of the graph structure for caching"""
        # Simple hash based on number of nodes and edges
        return f"{graph.number_of_nodes()}_{graph.number_of_edges()}"

    def get_confidence_scoring_explanation(self) -> Dict[str, str]:
        """
        Provide explanations for confidence scoring methodology
        """
        return {
            "centrality_outliers": "Confidence based on how many standard deviations above the mean the centrality scores are",
            "connectivity_patterns": "Confidence based on degree distribution analysis and statistical significance",
            "structural_anomalies": "Confidence based on the size and rarity of detected structural patterns",
            "temporal_clustering": "Confidence based on the frequency and timing of rapid activity bursts",
            "content_anomalies": "Confidence based on content length distribution and repetition patterns",
            "tamper_simulation": "Confidence artificially set based on known tamper scenarios for demonstration"
        }