"""
Metrics collector implementation with RAGAS integration for SecureInsight.

Provides comprehensive evaluation metrics including RAGAS metrics for RAG system
evaluation, processing time tracking, and efficiency benchmarking.
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAGAS not available: {e}. Metrics collection will use fallback implementations.")
    RAGAS_AVAILABLE = False


@dataclass
class QueryMetrics:
    """Data structure for individual query metrics."""
    query_id: str
    query: str
    response: str
    contexts: List[str]
    ground_truth: Optional[str]
    processing_time: float
    timestamp: datetime
    
    # RAGAS metrics
    context_precision_score: Optional[float] = None
    context_recall_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    answer_relevancy_score: Optional[float] = None
    
    # Custom metrics
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    similarity_scores: List[float] = None
    num_retrieved_docs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['similarity_scores'] = self.similarity_scores or []
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryMetrics':
        """Create QueryMetrics from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['similarity_scores'] = data.get('similarity_scores', [])
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Data structure for benchmarking results."""
    benchmark_id: str
    benchmark_type: str  # 'time_efficiency', 'accuracy', 'throughput'
    baseline_value: float
    measured_value: float
    improvement_percentage: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary format."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """
    Comprehensive metrics collector with RAGAS integration.
    
    Provides capabilities for:
    - RAGAS evaluation metrics (context_relevancy, answer_correctness, faithfulness, answer_relevancy)
    - Processing time tracking and efficiency metrics
    - Time saved benchmarking with baseline measurements
    - Feedback metrics aggregation and analysis
    - Performance monitoring and reporting
    """
    
    def __init__(self, metrics_dir: str = "feedback/metrics"):
        """
        Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics data
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.metrics_dir / "query_metrics").mkdir(exist_ok=True)
        (self.metrics_dir / "benchmarks").mkdir(exist_ok=True)
        (self.metrics_dir / "aggregated").mkdir(exist_ok=True)
        (self.metrics_dir / "reports").mkdir(exist_ok=True)
        
        self.query_metrics_file = self.metrics_dir / "query_metrics" / "metrics.jsonl"
        self.benchmarks_file = self.metrics_dir / "benchmarks" / "benchmarks.jsonl"
        
        # Initialize baseline measurements for benchmarking
        self.baseline_measurements = self._load_baseline_measurements()
        
        logger.info(f"MetricsCollector initialized with directory: {self.metrics_dir}")
        logger.info(f"RAGAS integration: {'enabled' if RAGAS_AVAILABLE else 'disabled (fallback mode)'}")
    
    def collect_query_metrics(
        self,
        query_id: str,
        query: str,
        response: str,
        contexts: List[str],
        processing_time: float,
        ground_truth: Optional[str] = None,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
        similarity_scores: Optional[List[float]] = None,
        num_retrieved_docs: int = 0
    ) -> QueryMetrics:
        """
        Collect comprehensive metrics for a query-response pair.
        
        Args:
            query_id: Unique identifier for the query
            query: Original user query
            response: System response
            contexts: Retrieved contexts used for generation
            processing_time: Total processing time
            ground_truth: Optional ground truth answer for evaluation
            retrieval_time: Time spent on retrieval
            generation_time: Time spent on generation
            similarity_scores: Similarity scores from retrieval
            num_retrieved_docs: Number of documents retrieved
            
        Returns:
            QueryMetrics object with all computed metrics
        """
        try:
            # Create base metrics object
            metrics = QueryMetrics(
                query_id=query_id,
                query=query,
                response=response,
                contexts=contexts,
                ground_truth=ground_truth,
                processing_time=processing_time,
                timestamp=datetime.now(),
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                similarity_scores=similarity_scores or [],
                num_retrieved_docs=num_retrieved_docs
            )
            
            # Calculate RAGAS metrics if available and ground truth provided
            if RAGAS_AVAILABLE and ground_truth and contexts:
                ragas_scores = self._calculate_ragas_metrics(query, response, contexts, ground_truth)
                metrics.context_precision_score = ragas_scores.get('context_precision')
                metrics.context_recall_score = ragas_scores.get('context_recall')
                metrics.faithfulness_score = ragas_scores.get('faithfulness')
                metrics.answer_relevancy_score = ragas_scores.get('answer_relevancy')
            elif not RAGAS_AVAILABLE:
                # Use fallback metrics
                fallback_scores = self._calculate_fallback_metrics(query, response, contexts, ground_truth)
                metrics.context_precision_score = fallback_scores.get('context_precision')
                metrics.context_recall_score = fallback_scores.get('context_recall')
                metrics.faithfulness_score = fallback_scores.get('faithfulness')
                metrics.answer_relevancy_score = fallback_scores.get('answer_relevancy')
            
            # Store metrics
            self._store_query_metrics(metrics)
            
            logger.info(f"Collected metrics for query {query_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for query {query_id}: {e}")
            raise
    
    def _calculate_ragas_metrics(
        self, 
        query: str, 
        response: str, 
        contexts: List[str], 
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Calculate RAGAS metrics for the query-response pair.
        
        Args:
            query: User query
            response: System response
            contexts: Retrieved contexts
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary containing RAGAS metric scores
        """
        try:
            # Prepare dataset for RAGAS evaluation
            data = {
                'question': [query],
                'answer': [response],
                'contexts': [contexts],
                'ground_truth': [ground_truth]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Define metrics to evaluate
            metrics_to_evaluate = [
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ]
            
            # Run RAGAS evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_evaluate
            )
            
            # Extract scores
            scores = {
                'context_precision': float(result['context_precision']) if 'context_precision' in result else None,
                'context_recall': float(result['context_recall']) if 'context_recall' in result else None,
                'faithfulness': float(result['faithfulness']) if 'faithfulness' in result else None,
                'answer_relevancy': float(result['answer_relevancy']) if 'answer_relevancy' in result else None
            }
            
            logger.debug(f"RAGAS scores calculated: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate RAGAS metrics: {e}")
            return {}
    
    def _calculate_fallback_metrics(
        self, 
        query: str, 
        response: str, 
        contexts: List[str], 
        ground_truth: Optional[str]
    ) -> Dict[str, float]:
        """
        Calculate fallback metrics when RAGAS is not available.
        
        Args:
            query: User query
            response: System response
            contexts: Retrieved contexts
            ground_truth: Ground truth answer (optional)
            
        Returns:
            Dictionary containing fallback metric scores
        """
        try:
            scores = {}
            
            # Context relevancy: measure overlap between query and contexts
            if contexts:
                query_words = set(query.lower().split())
                context_words = set()
                for context in contexts:
                    context_words.update(context.lower().split())
                
                if query_words and context_words:
                    overlap = len(query_words.intersection(context_words))
                    scores['context_precision'] = overlap / len(query_words)
                    scores['context_recall'] = overlap / len(context_words) if context_words else 0.0
                else:
                    scores['context_precision'] = 0.0
                    scores['context_recall'] = 0.0
            else:
                scores['context_precision'] = 0.0
                scores['context_recall'] = 0.0
            
            # Answer correctness: compare with ground truth if available
            if ground_truth:
                response_words = set(response.lower().split())
                truth_words = set(ground_truth.lower().split())
                
                if response_words and truth_words:
                    overlap = len(response_words.intersection(truth_words))
                    scores['answer_correctness'] = overlap / len(truth_words)
                else:
                    scores['answer_correctness'] = 0.0
            else:
                scores['answer_correctness'] = None
            
            # Faithfulness: measure overlap between response and contexts
            if contexts:
                response_words = set(response.lower().split())
                context_words = set()
                for context in contexts:
                    context_words.update(context.lower().split())
                
                if response_words and context_words:
                    overlap = len(response_words.intersection(context_words))
                    scores['faithfulness'] = overlap / len(response_words)
                else:
                    scores['faithfulness'] = 0.0
            else:
                scores['faithfulness'] = 0.0
            
            # Answer relevancy: measure overlap between query and response
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            
            if query_words and response_words:
                overlap = len(query_words.intersection(response_words))
                scores['answer_relevancy'] = overlap / len(query_words)
            else:
                scores['answer_relevancy'] = 0.0
            
            logger.debug(f"Fallback scores calculated: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate fallback metrics: {e}")
            return {}
    
    def _store_query_metrics(self, metrics: QueryMetrics) -> None:
        """
        Store query metrics to local storage.
        
        Args:
            metrics: QueryMetrics object to store
        """
        try:
            with open(self.query_metrics_file, 'a', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to store query metrics: {e}")
            raise
    
    def benchmark_time_efficiency(
        self,
        task_type: str,
        measured_time: float,
        task_complexity: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark time efficiency against baseline measurements.
        
        Args:
            task_type: Type of task (e.g., 'document_search', 'multimodal_query')
            measured_time: Actual time taken for the task
            task_complexity: Complexity level ('simple', 'medium', 'complex')
            metadata: Additional metadata about the task
            
        Returns:
            BenchmarkResult with efficiency comparison
        """
        try:
            # Get baseline time for this task type and complexity
            baseline_key = f"{task_type}_{task_complexity}"
            baseline_time = self.baseline_measurements.get(baseline_key, {}).get('manual_time', 60.0)  # Default 1 minute
            
            # Calculate improvement
            time_saved = baseline_time - measured_time
            improvement_percentage = (time_saved / baseline_time) * 100 if baseline_time > 0 else 0.0
            
            # Create benchmark result
            benchmark = BenchmarkResult(
                benchmark_id=f"time_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                benchmark_type="time_efficiency",
                baseline_value=baseline_time,
                measured_value=measured_time,
                improvement_percentage=improvement_percentage,
                timestamp=datetime.now(),
                metadata={
                    'task_type': task_type,
                    'task_complexity': task_complexity,
                    'time_saved_seconds': time_saved,
                    **(metadata or {})
                }
            )
            
            # Store benchmark result
            self._store_benchmark_result(benchmark)
            
            logger.info(f"Time efficiency benchmark: {improvement_percentage:.1f}% improvement "
                       f"({time_saved:.2f}s saved) for {task_type}")
            
            return benchmark
            
        except Exception as e:
            logger.error(f"Failed to benchmark time efficiency: {e}")
            raise
    
    def _store_benchmark_result(self, benchmark: BenchmarkResult) -> None:
        """
        Store benchmark result to local storage.
        
        Args:
            benchmark: BenchmarkResult object to store
        """
        try:
            with open(self.benchmarks_file, 'a', encoding='utf-8') as f:
                json.dump(benchmark.to_dict(), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to store benchmark result: {e}")
            raise
    
    def _load_baseline_measurements(self) -> Dict[str, Dict[str, float]]:
        """
        Load baseline measurements for benchmarking.
        
        Returns:
            Dictionary containing baseline measurements
        """
        baseline_file = self.metrics_dir / "baselines.json"
        
        # Default baseline measurements (in seconds)
        default_baselines = {
            'document_search_simple': {'manual_time': 30.0, 'description': 'Simple keyword search in documents'},
            'document_search_medium': {'manual_time': 120.0, 'description': 'Complex search across multiple documents'},
            'document_search_complex': {'manual_time': 300.0, 'description': 'Research-level document analysis'},
            'multimodal_query_simple': {'manual_time': 60.0, 'description': 'Basic image-text search'},
            'multimodal_query_medium': {'manual_time': 180.0, 'description': 'Cross-modal analysis'},
            'multimodal_query_complex': {'manual_time': 600.0, 'description': 'Complex multimodal reasoning'},
            'audio_transcription_simple': {'manual_time': 300.0, 'description': 'Manual audio transcription'},
            'audio_transcription_medium': {'manual_time': 900.0, 'description': 'Complex audio analysis'},
            'knowledge_extraction_simple': {'manual_time': 600.0, 'description': 'Basic information extraction'},
            'knowledge_extraction_medium': {'manual_time': 1800.0, 'description': 'Comprehensive knowledge mapping'}
        }
        
        try:
            if baseline_file.exists():
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baselines = json.load(f)
                # Merge with defaults for any missing entries
                for key, value in default_baselines.items():
                    if key not in baselines:
                        baselines[key] = value
            else:
                baselines = default_baselines
                # Save default baselines
                with open(baseline_file, 'w', encoding='utf-8') as f:
                    json.dump(baselines, f, indent=2, ensure_ascii=False)
                logger.info(f"Created default baseline measurements at {baseline_file}")
            
            return baselines
            
        except Exception as e:
            logger.error(f"Failed to load baseline measurements: {e}")
            return default_baselines
    
    def aggregate_feedback_metrics(self, feedback_system) -> Dict[str, Any]:
        """
        Aggregate feedback metrics with performance metrics.
        
        Args:
            feedback_system: FeedbackSystem instance
            
        Returns:
            Dictionary containing aggregated metrics
        """
        try:
            # Get feedback metrics
            feedback_metrics = feedback_system.get_feedback_metrics()
            
            # Get query metrics
            query_metrics = self._get_aggregated_query_metrics()
            
            # Get benchmark results
            benchmark_metrics = self._get_aggregated_benchmark_metrics()
            
            # Combine all metrics
            aggregated = {
                'timestamp': datetime.now().isoformat(),
                'feedback_metrics': feedback_metrics,
                'query_performance_metrics': query_metrics,
                'benchmark_metrics': benchmark_metrics,
                'correlation_analysis': self._analyze_feedback_performance_correlation(
                    feedback_metrics, query_metrics
                )
            }
            
            # Store aggregated metrics
            self._store_aggregated_metrics(aggregated)
            
            logger.info("Successfully aggregated feedback and performance metrics")
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate feedback metrics: {e}")
            return {}
    
    def _get_aggregated_query_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated query performance metrics.
        
        Returns:
            Dictionary containing aggregated query metrics
        """
        try:
            metrics_list = list(self._read_all_query_metrics())
            
            if not metrics_list:
                return {'error': 'No query metrics available'}
            
            # Calculate aggregated statistics
            processing_times = [m.processing_time for m in metrics_list]
            retrieval_times = [m.retrieval_time for m in metrics_list]
            generation_times = [m.generation_time for m in metrics_list]
            
            # RAGAS metrics aggregation
            ragas_metrics = {}
            for metric_name in ['context_precision_score', 'context_recall_score', 
                               'faithfulness_score', 'answer_relevancy_score']:
                scores = [getattr(m, metric_name) for m in metrics_list 
                         if hasattr(m, metric_name) and getattr(m, metric_name) is not None]
                if scores:
                    ragas_metrics[metric_name] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'count': len(scores)
                    }
            
            # Similarity scores analysis
            all_similarity_scores = []
            for m in metrics_list:
                if m.similarity_scores:
                    all_similarity_scores.extend(m.similarity_scores)
            
            aggregated = {
                'total_queries': len(metrics_list),
                'processing_time_stats': {
                    'mean': np.mean(processing_times),
                    'std': np.std(processing_times),
                    'min': np.min(processing_times),
                    'max': np.max(processing_times),
                    'p95': np.percentile(processing_times, 95)
                },
                'retrieval_time_stats': {
                    'mean': np.mean(retrieval_times),
                    'std': np.std(retrieval_times),
                    'min': np.min(retrieval_times),
                    'max': np.max(retrieval_times)
                },
                'generation_time_stats': {
                    'mean': np.mean(generation_times),
                    'std': np.std(generation_times),
                    'min': np.min(generation_times),
                    'max': np.max(generation_times)
                },
                'ragas_metrics': ragas_metrics,
                'similarity_scores_stats': {
                    'mean': np.mean(all_similarity_scores) if all_similarity_scores else 0.0,
                    'std': np.std(all_similarity_scores) if all_similarity_scores else 0.0,
                    'count': len(all_similarity_scores)
                },
                'average_retrieved_docs': np.mean([m.num_retrieved_docs for m in metrics_list])
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate query metrics: {e}")
            return {}
    
    def _get_aggregated_benchmark_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated benchmark metrics.
        
        Returns:
            Dictionary containing aggregated benchmark metrics
        """
        try:
            benchmarks = list(self._read_all_benchmarks())
            
            if not benchmarks:
                return {'error': 'No benchmark data available'}
            
            # Group by benchmark type
            by_type = {}
            for benchmark in benchmarks:
                bench_type = benchmark.benchmark_type
                if bench_type not in by_type:
                    by_type[bench_type] = []
                by_type[bench_type].append(benchmark)
            
            # Calculate statistics for each type
            aggregated = {}
            for bench_type, type_benchmarks in by_type.items():
                improvements = [b.improvement_percentage for b in type_benchmarks]
                time_saved = [b.baseline_value - b.measured_value for b in type_benchmarks]
                
                aggregated[bench_type] = {
                    'count': len(type_benchmarks),
                    'improvement_percentage_stats': {
                        'mean': np.mean(improvements),
                        'std': np.std(improvements),
                        'min': np.min(improvements),
                        'max': np.max(improvements)
                    },
                    'time_saved_stats': {
                        'mean': np.mean(time_saved),
                        'std': np.std(time_saved),
                        'total': np.sum(time_saved)
                    },
                    'latest_benchmark': max(type_benchmarks, key=lambda x: x.timestamp).to_dict()
                }
            
            # Overall statistics
            all_improvements = [b.improvement_percentage for b in benchmarks]
            all_time_saved = [b.baseline_value - b.measured_value for b in benchmarks]
            
            aggregated['overall'] = {
                'total_benchmarks': len(benchmarks),
                'average_improvement_percentage': np.mean(all_improvements),
                'total_time_saved_seconds': np.sum(all_time_saved),
                'total_time_saved_hours': np.sum(all_time_saved) / 3600
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate benchmark metrics: {e}")
            return {}
    
    def _analyze_feedback_performance_correlation(
        self, 
        feedback_metrics: Dict[str, Any], 
        query_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between feedback and performance metrics.
        
        Args:
            feedback_metrics: Aggregated feedback metrics
            query_metrics: Aggregated query performance metrics
            
        Returns:
            Dictionary containing correlation analysis
        """
        try:
            correlation_analysis = {
                'feedback_performance_insights': [],
                'recommendations': []
            }
            
            # Analyze average rating vs processing time
            avg_rating = feedback_metrics.get('average_rating', 0)
            avg_processing_time = query_metrics.get('processing_time_stats', {}).get('mean', 0)
            
            if avg_rating < 3.0 and avg_processing_time > 5.0:
                correlation_analysis['feedback_performance_insights'].append(
                    "Low ratings correlate with high processing times"
                )
                correlation_analysis['recommendations'].append(
                    "Optimize system performance to improve user satisfaction"
                )
            
            # Analyze RAGAS metrics vs feedback
            ragas_metrics = query_metrics.get('ragas_metrics', {})
            if ragas_metrics:
                avg_faithfulness = ragas_metrics.get('faithfulness_score', {}).get('mean', 0)
                avg_relevancy = ragas_metrics.get('answer_relevancy_score', {}).get('mean', 0)
                
                if avg_rating < 3.0 and (avg_faithfulness < 0.7 or avg_relevancy < 0.7):
                    correlation_analysis['feedback_performance_insights'].append(
                        "Low ratings correlate with poor RAGAS scores"
                    )
                    correlation_analysis['recommendations'].append(
                        "Improve retrieval quality and answer generation"
                    )
            
            # Analyze similarity scores vs feedback
            avg_similarity = query_metrics.get('similarity_scores_stats', {}).get('mean', 0)
            if avg_rating < 3.0 and avg_similarity < 0.5:
                correlation_analysis['feedback_performance_insights'].append(
                    "Low ratings correlate with poor similarity scores"
                )
                correlation_analysis['recommendations'].append(
                    "Review embedding quality and retrieval thresholds"
                )
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze feedback-performance correlation: {e}")
            return {}
    
    def _store_aggregated_metrics(self, aggregated_metrics: Dict[str, Any]) -> None:
        """
        Store aggregated metrics to file.
        
        Args:
            aggregated_metrics: Aggregated metrics to store
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            aggregated_file = self.metrics_dir / "aggregated" / f"aggregated_metrics_{timestamp}.json"
            
            with open(aggregated_file, 'w', encoding='utf-8') as f:
                json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stored aggregated metrics to {aggregated_file}")
            
        except Exception as e:
            logger.error(f"Failed to store aggregated metrics: {e}")
            raise
    
    def _read_all_query_metrics(self):
        """
        Generator to read all query metrics from storage.
        
        Yields:
            QueryMetrics objects
        """
        if not self.query_metrics_file.exists():
            return
        
        try:
            with open(self.query_metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            yield QueryMetrics.from_dict(data)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning(f"Skipping invalid metrics line: {e}")
                            continue
        except Exception as e:
            logger.error(f"Failed to read query metrics file: {e}")
            raise
    
    def _read_all_benchmarks(self):
        """
        Generator to read all benchmark results from storage.
        
        Yields:
            BenchmarkResult objects
        """
        if not self.benchmarks_file.exists():
            return
        
        try:
            with open(self.benchmarks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                            yield BenchmarkResult(**data)
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            logger.warning(f"Skipping invalid benchmark line: {e}")
                            continue
        except Exception as e:
            logger.error(f"Failed to read benchmarks file: {e}")
            raise
    
    def generate_metrics_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report for the specified period.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Dictionary containing comprehensive metrics report
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter metrics by date
            recent_query_metrics = [
                m for m in self._read_all_query_metrics() 
                if m.timestamp >= cutoff_date
            ]
            
            recent_benchmarks = [
                b for b in self._read_all_benchmarks() 
                if b.timestamp >= cutoff_date
            ]
            
            # Generate report
            report = {
                'report_period': f"Last {days} days",
                'report_timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_queries_analyzed': len(recent_query_metrics),
                    'total_benchmarks_run': len(recent_benchmarks),
                    'average_processing_time': np.mean([m.processing_time for m in recent_query_metrics]) if recent_query_metrics else 0,
                    'average_improvement_percentage': np.mean([b.improvement_percentage for b in recent_benchmarks]) if recent_benchmarks else 0
                },
                'performance_trends': self._analyze_performance_trends(recent_query_metrics),
                'ragas_analysis': self._analyze_ragas_trends(recent_query_metrics),
                'efficiency_gains': self._analyze_efficiency_gains(recent_benchmarks),
                'recommendations': self._generate_performance_recommendations(recent_query_metrics, recent_benchmarks)
            }
            
            # Store report
            self._store_metrics_report(report)
            
            logger.info(f"Generated metrics report for {len(recent_query_metrics)} queries and {len(recent_benchmarks)} benchmarks")
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate metrics report: {e}")
            return {}
    
    def _analyze_performance_trends(self, metrics_list: List[QueryMetrics]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if not metrics_list:
            return {'error': 'No metrics data available'}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics_list, key=lambda x: x.timestamp)
        
        # Calculate daily averages
        daily_performance = {}
        for metric in sorted_metrics:
            date_key = metric.timestamp.date().isoformat()
            if date_key not in daily_performance:
                daily_performance[date_key] = []
            daily_performance[date_key].append(metric.processing_time)
        
        # Calculate trends
        daily_averages = {
            date: np.mean(times) for date, times in daily_performance.items()
        }
        
        dates = sorted(daily_averages.keys())
        if len(dates) >= 2:
            recent_avg = daily_averages[dates[-1]]
            earlier_avg = daily_averages[dates[0]]
            trend = 'improving' if recent_avg < earlier_avg else 'declining' if recent_avg > earlier_avg else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'daily_averages': daily_averages,
            'trend_direction': trend,
            'performance_variance': np.std(list(daily_averages.values())) if daily_averages else 0
        }
    
    def _analyze_ragas_trends(self, metrics_list: List[QueryMetrics]) -> Dict[str, Any]:
        """Analyze RAGAS metrics trends"""
        if not metrics_list:
            return {'error': 'No metrics data available'}
        
        ragas_analysis = {}
        
        for metric_name in ['context_precision_score', 'context_recall_score', 
                           'faithfulness_score', 'answer_relevancy_score']:
            scores = [getattr(m, metric_name) for m in metrics_list 
                     if hasattr(m, metric_name) and getattr(m, metric_name) is not None]
            
            if scores:
                ragas_analysis[metric_name] = {
                    'current_average': np.mean(scores),
                    'trend': 'stable',  # Could be enhanced with time-series analysis
                    'quality_assessment': 'good' if np.mean(scores) > 0.7 else 'needs_improvement'
                }
        
        return ragas_analysis
    
    def _analyze_efficiency_gains(self, benchmarks: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze efficiency gains from benchmarks"""
        if not benchmarks:
            return {'error': 'No benchmark data available'}
        
        total_time_saved = sum(b.baseline_value - b.measured_value for b in benchmarks)
        avg_improvement = np.mean([b.improvement_percentage for b in benchmarks])
        
        # Group by task type
        by_task_type = {}
        for benchmark in benchmarks:
            task_type = benchmark.metadata.get('task_type', 'unknown')
            if task_type not in by_task_type:
                by_task_type[task_type] = []
            by_task_type[task_type].append(benchmark)
        
        task_analysis = {}
        for task_type, task_benchmarks in by_task_type.items():
            task_improvements = [b.improvement_percentage for b in task_benchmarks]
            task_analysis[task_type] = {
                'average_improvement': np.mean(task_improvements),
                'count': len(task_benchmarks)
            }
        
        return {
            'total_time_saved_hours': total_time_saved / 3600,
            'average_improvement_percentage': avg_improvement,
            'task_type_analysis': task_analysis,
            'efficiency_rating': 'excellent' if avg_improvement > 80 else 'good' if avg_improvement > 50 else 'needs_improvement'
        }
    
    def _generate_performance_recommendations(
        self, 
        metrics_list: List[QueryMetrics], 
        benchmarks: List[BenchmarkResult]
    ) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        if metrics_list:
            avg_processing_time = np.mean([m.processing_time for m in metrics_list])
            if avg_processing_time > 5.0:
                recommendations.append("High average processing time detected - consider performance optimization")
            
            # RAGAS-based recommendations
            ragas_scores = {}
            for metric_name in ['context_precision_score', 'context_recall_score', 
                               'faithfulness_score', 'answer_relevancy_score']:
                scores = [getattr(m, metric_name) for m in metrics_list 
                         if hasattr(m, metric_name) and getattr(m, metric_name) is not None]
                if scores:
                    ragas_scores[metric_name] = np.mean(scores)
            
            if ragas_scores.get('context_precision_score', 1.0) < 0.7:
                recommendations.append("Low context precision - review retrieval algorithm")
            if ragas_scores.get('context_recall_score', 1.0) < 0.7:
                recommendations.append("Low context recall - expand retrieval scope")
            if ragas_scores.get('faithfulness_score', 1.0) < 0.7:
                recommendations.append("Low faithfulness score - improve answer grounding")
            if ragas_scores.get('answer_relevancy_score', 1.0) < 0.7:
                recommendations.append("Low answer relevancy - enhance query understanding")
        
        if benchmarks:
            avg_improvement = np.mean([b.improvement_percentage for b in benchmarks])
            if avg_improvement < 50:
                recommendations.append("Low efficiency gains - investigate automation opportunities")
        
        if not recommendations:
            recommendations.append("System performance appears optimal - continue monitoring")
        
        return recommendations
    
    def _store_metrics_report(self, report: Dict[str, Any]) -> None:
        """Store metrics report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.metrics_dir / "reports" / f"metrics_report_{timestamp}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stored metrics report to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to store metrics report: {e}")
            raise