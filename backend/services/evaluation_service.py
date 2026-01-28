"""
Evaluation Service

Provides RAG evaluation functionality:
- Retrieval quality metrics (MRR, Recall@K, Precision@K, NDCG)
- Generation quality metrics (grounding, faithfulness, relevance)
- Latency tracking
- A/B testing support
"""

import time
import uuid
import json
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from loguru import logger

from core.settings import settings
from typing import Union
from enum import Enum

# Simplified models since we don't have the actual models yet
class EvaluationStatusEnum(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Basic model classes (simplified)
class EvaluationMetrics:
    def __init__(self):
        self.retrieval = None
        self.generation = None
        self.latency = None
        self.overall_quality_score = 0.0
        self.total_test_cases = 0

class TestCase:
    def __init__(self, query: str, expected_documents: List[str]):
        self.query = query
        self.expected_documents = expected_documents

class EvaluationConfig:
    def __init__(self, name: str, description: str = None, test_cases: List[TestCase] = None):
        self.name = name
        self.description = description
        self.test_cases = test_cases or []
        self.k_values = [1, 3, 5, 10]
        self.similarity_thresholds = [0.5]
        self.include_generation_metrics = True
        self.include_latency_metrics = True
        self.max_concurrent_requests = 5

class EvaluationRun:
    def __init__(self, run_id: str, config: EvaluationConfig):
        self.run_id = run_id
        self.name = config.name
        self.description = config.description
        self.config = config
        self.status = EvaluationStatusEnum.PENDING
        self.created_at = datetime.utcnow()
        self.completed_at = None
        self.total_cases = len(config.test_cases) if config.test_cases else 0
        self.metrics = None
        self.results = []

class LatencyMetrics:
    def __init__(self):
        self.average_total_latency_ms = 0
        self.p50_total_latency_ms = 0
        self.p90_total_latency_ms = 0
        self.p99_total_latency_ms = 0
        self.average_embedding_latency_ms = 0
        self.average_retrieval_latency_ms = 0
        self.average_generation_latency_ms = 0
        self.latency_buckets = {}
        self.error_rate = 0.0

class RetrievalMetrics:
    def __init__(self):
        self.mrr = 0.0
        self.precision_at_k = {}
        self.recall_at_k = {}
        self.ndcg_at_k = {}
        self.average_precision = 0.0

class GenerationMetrics:
    def __init__(self):
        self.grounding_score = 0.0
        self.coherence = 0.0
        self.completeness = 0.0
        self.answer_relevance = 0.0

class EvaluationResult:
    pass


class EvaluationService:
    """
    RAG Evaluation Service
    
    Provides comprehensive evaluation of the RAG pipeline including:
    - Retrieval quality (precision, recall, MRR, NDCG)
    - Generation quality (grounding, faithfulness)
    - Latency metrics (p50, p90, p99)
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or Path("./evaluations")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory tracking
        self._active_runs: Dict[str, EvaluationRun] = {}
        self._evaluation_history: List[EvaluationRun] = []
        
        # Real-time metrics
        self._latency_samples: List[Dict] = []
        self._retrieval_samples: List[Dict] = []
        self._generation_samples: List[Dict] = []
        
        # Sampling control
        self._sample_count = 0
        self._sample_rate = 0.1  # Default sample rate
        
        self.logger = logger.bind(component="EvaluationService")
    
    # ==================== Real-time Tracking ====================
    
    def should_sample(self) -> bool:
        """Determine if current request should be sampled"""
        # Always sample for now
        self._sample_count += 1
        return (self._sample_count % 10) == 0  # 10% sample rate
    
    def track_latency(
        self,
        total_ms: float,
        embedding_ms: Optional[float] = None,
        retrieval_ms: Optional[float] = None,
        generation_ms: Optional[float] = None
    ) -> None:
        """Track latency metrics"""
        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_ms": total_ms,
            "embedding_ms": embedding_ms,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms
        }
        
        self._latency_samples.append(sample)
        
        # Keep only recent samples
        max_samples = 10000
        if len(self._latency_samples) > max_samples:
            self._latency_samples = self._latency_samples[-max_samples:]
    
    def track_retrieval(
        self,
        query: str,
        retrieved_ids: List[str],
        relevant_ids: Optional[List[str]] = None,
        similarity_scores: Optional[List[float]] = None
    ) -> None:
        """Track retrieval metrics"""
        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:200],  # Truncate
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "similarity_scores": similarity_scores,
            "k": len(retrieved_ids)
        }
        
        self._retrieval_samples.append(sample)
        
        max_samples = 5000
        if len(self._retrieval_samples) > max_samples:
            self._retrieval_samples = self._retrieval_samples[-max_samples:]
    
    def track_generation(
        self,
        query: str,
        response: str,
        context_used: List[str],
        grounding_score: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Track generation metrics"""
        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:200],
            "response_length": len(response),
            "context_used": context_used,
            "grounding_score": grounding_score,
            "confidence": confidence
        }
        
        self._generation_samples.append(sample)
        
        max_samples = 3000
        if len(self._generation_samples) > max_samples:
            self._generation_samples = self._generation_samples[-max_samples:]
            return
        
        sample = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:200],
            "response_length": len(response),
            "context_count": len(context_used),
            "grounding_score": grounding_score,
            "confidence": confidence
        }
        
        self._generation_samples.append(sample)
        
        max_samples = 5000
        if len(self._generation_samples) > max_samples:
            self._generation_samples = self._generation_samples[-max_samples:]
    
    # ==================== Metrics Calculation ====================
    
    def calculate_retrieval_metrics(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """Calculate retrieval metrics for a single query"""
        metrics = {
            "precision_at_k": {},
            "recall_at_k": {},
            "hit_rate_at_k": {},
            "mrr": 0.0,
            "ndcg_at_k": {}
        }
        
        if not relevant_ids:
            return metrics
        
        relevant_set = set(relevant_ids)
        
        # Calculate metrics for each k
        for k in k_values:
            top_k = retrieved_ids[:k]
            hits = len(set(top_k) & relevant_set)
            
            # Precision@K
            metrics["precision_at_k"][k] = hits / k if k > 0 else 0
            
            # Recall@K
            metrics["recall_at_k"][k] = hits / len(relevant_set) if relevant_set else 0
            
            # Hit Rate@K (binary: at least one relevant in top k)
            metrics["hit_rate_at_k"][k] = 1.0 if hits > 0 else 0.0
            
            # NDCG@K
            if k > 0:
                dcg = sum(
                    1 / (i + 2)  # log2(i+2) approximation
                    for i, doc_id in enumerate(top_k)
                    if doc_id in relevant_set
                )
                ideal_dcg = sum(1 / (i + 2) for i in range(min(k, len(relevant_set))))
                metrics["ndcg_at_k"][k] = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_set:
                metrics["mrr"] = 1 / (i + 1)
                break
        
        return metrics
    
    def calculate_generation_metrics(
        self,
        query: str,
        response: str,
        context: List[str],
        expected_answer: Optional[str] = None
    ) -> Dict[str, float]:
        """Calculate generation quality metrics"""
        metrics = {
            "grounding_score": 0.0,
            "coherence": 0.0,
            "completeness": 0.0
        }
        
        if not response:
            return metrics
        
        # Simple grounding check (word overlap with context)
        response_words = set(response.lower().split())
        context_text = " ".join(context).lower()
        context_words = set(context_text.split())
        
        if response_words:
            overlap = len(response_words & context_words)
            metrics["grounding_score"] = min(1.0, overlap / len(response_words))
        
        # Coherence (simple heuristics)
        # Check for proper sentence structure
        sentences = response.split(".")
        valid_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        metrics["coherence"] = min(1.0, valid_sentences / max(len(sentences), 1))
        
        # Completeness (if expected answer provided)
        if expected_answer:
            expected_words = set(expected_answer.lower().split())
            if expected_words:
                overlap = len(response_words & expected_words)
                metrics["completeness"] = overlap / len(expected_words)
        else:
            # Estimate based on response length relative to query
            query_len = len(query.split())
            response_len = len(response.split())
            metrics["completeness"] = min(1.0, response_len / max(query_len * 5, 1))
        
        return metrics
    
    # ==================== Aggregated Metrics ====================
    
    def get_aggregated_metrics(
        self,
        time_range_hours: int = 24
    ) -> EvaluationMetrics:
        """Get aggregated metrics for a time range"""
        cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Filter samples by time
        latency_samples = [
            s for s in self._latency_samples
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]
        
        retrieval_samples = [
            s for s in self._retrieval_samples
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]
        
        generation_samples = [
            s for s in self._generation_samples
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]
        
        # Calculate latency metrics
        latency_metrics = self._aggregate_latency(latency_samples)
        
        # Calculate retrieval metrics
        retrieval_metrics = self._aggregate_retrieval(retrieval_samples)
        
        # Calculate generation metrics
        generation_metrics = self._aggregate_generation(generation_samples)
        
        # Overall quality score
        overall_score = (
            retrieval_metrics.mrr * 0.3 +
            generation_metrics.grounding_score * 0.3 +
            generation_metrics.answer_relevance * 0.2 +
            (1 - latency_metrics.error_rate) * 0.2
        )
        
        return EvaluationMetrics(
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            latency=latency_metrics,
            overall_quality_score=overall_score,
            total_test_cases=len(latency_samples)
        )
    
    def _aggregate_latency(self, samples: List[Dict]) -> LatencyMetrics:
        """Aggregate latency samples"""
        if not samples:
            return LatencyMetrics()
        
        totals = sorted([s["total_ms"] for s in samples])
        n = len(totals)
        
        # Calculate percentiles
        p50_idx = int(n * 0.50)
        p90_idx = int(n * 0.90)
        p99_idx = int(n * 0.99)
        
        # Latency buckets
        buckets = defaultdict(int)
        for t in totals:
            if t < 100:
                buckets["<100ms"] += 1
            elif t < 500:
                buckets["100-500ms"] += 1
            elif t < 1000:
                buckets["500ms-1s"] += 1
            elif t < 2000:
                buckets["1-2s"] += 1
            elif t < 5000:
                buckets["2-5s"] += 1
            else:
                buckets[">5s"] += 1
        
        # Component latencies
        embedding_samples = [s["embedding_ms"] for s in samples if s.get("embedding_ms")]
        retrieval_samples = [s["retrieval_ms"] for s in samples if s.get("retrieval_ms")]
        generation_samples = [s["generation_ms"] for s in samples if s.get("generation_ms")]
        
        return LatencyMetrics(
            average_total_latency_ms=sum(totals) / n,
            p50_total_latency_ms=totals[p50_idx] if p50_idx < n else 0,
            p90_total_latency_ms=totals[p90_idx] if p90_idx < n else 0,
            p99_total_latency_ms=totals[p99_idx] if p99_idx < n else 0,
            average_embedding_latency_ms=sum(embedding_samples) / len(embedding_samples) if embedding_samples else 0,
            average_retrieval_latency_ms=sum(retrieval_samples) / len(retrieval_samples) if retrieval_samples else 0,
            average_generation_latency_ms=sum(generation_samples) / len(generation_samples) if generation_samples else 0,
            latency_buckets=dict(buckets),
            total_requests_measured=n
        )
    
    def _aggregate_retrieval(self, samples: List[Dict]) -> RetrievalMetrics:
        """Aggregate retrieval samples"""
        if not samples:
            return RetrievalMetrics()
        
        # For samples with ground truth
        with_gt = [s for s in samples if s.get("relevant_ids")]
        
        if not with_gt:
            # Just basic stats
            avg_scores = []
            for s in samples:
                if s.get("similarity_scores"):
                    avg_scores.append(sum(s["similarity_scores"]) / len(s["similarity_scores"]))
            
            return RetrievalMetrics(
                average_relevance_score=sum(avg_scores) / len(avg_scores) if avg_scores else 0,
                total_queries_evaluated=len(samples)
            )
        
        # Aggregate metrics from samples with ground truth
        precision_sums = defaultdict(float)
        recall_sums = defaultdict(float)
        hit_rate_sums = defaultdict(float)
        ndcg_sums = defaultdict(float)
        mrr_sum = 0.0
        
        for sample in with_gt:
            metrics = self.calculate_retrieval_metrics(
                sample["retrieved_ids"],
                sample["relevant_ids"]
            )
            
            for k, v in metrics["precision_at_k"].items():
                precision_sums[k] += v
            for k, v in metrics["recall_at_k"].items():
                recall_sums[k] += v
            for k, v in metrics["hit_rate_at_k"].items():
                hit_rate_sums[k] += v
            for k, v in metrics["ndcg_at_k"].items():
                ndcg_sums[k] += v
            mrr_sum += metrics["mrr"]
        
        n = len(with_gt)
        
        return RetrievalMetrics(
            precision_at_k={k: v / n for k, v in precision_sums.items()},
            recall_at_k={k: v / n for k, v in recall_sums.items()},
            hit_rate_at_k={k: v / n for k, v in hit_rate_sums.items()},
            ndcg_at_k={k: v / n for k, v in ndcg_sums.items()},
            mrr=mrr_sum / n,
            total_queries_evaluated=n
        )
    
    def _aggregate_generation(self, samples: List[Dict]) -> GenerationMetrics:
        """Aggregate generation samples"""
        if not samples:
            return GenerationMetrics()
        
        grounding_scores = [s["grounding_score"] for s in samples if s.get("grounding_score") is not None]
        confidences = [s["confidence"] for s in samples if s.get("confidence") is not None]
        response_lengths = [s["response_length"] for s in samples]
        context_counts = [s["context_count"] for s in samples]
        
        return GenerationMetrics(
            grounding_score=sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0,
            answer_relevance=sum(confidences) / len(confidences) if confidences else 0,
            average_response_length=sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            average_citation_count=sum(context_counts) / len(context_counts) if context_counts else 0,
            total_responses_evaluated=len(samples)
        )
    
    # ==================== Evaluation Runs ====================
    
    async def create_evaluation_run(
        self,
        config: EvaluationConfig,
        neurax_service: "NeuraXService"
    ) -> str:
        """Create and start an evaluation run"""
        run_id = str(uuid.uuid4())[:8]
        
        run = EvaluationRun(
            run_id=run_id,
            name=config.name,
            config=config,
            status=EvaluationStatusEnum.PENDING,
            created_at=datetime.utcnow()
        )
        
        self._active_runs[run_id] = run
        
        # Start evaluation in background
        asyncio.create_task(self._execute_evaluation(run, neurax_service))
        
        return run_id
    
    async def _execute_evaluation(
        self,
        run: EvaluationRun,
        neurax_service: "NeuraXService"
    ) -> None:
        """Execute an evaluation run"""
        run.status = EvaluationStatusEnum.RUNNING
        run.started_at = datetime.utcnow()
        
        try:
            test_cases = run.config.test_cases or []
            run.total_cases = len(test_cases)
            
            for i, test_case in enumerate(test_cases):
                try:
                    result = await self._evaluate_test_case(
                        test_case,
                        neurax_service,
                        run.config
                    )
                    run.results.append(result)
                    run.completed_cases += 1
                    
                except Exception as e:
                    run.results.append(EvaluationResult(
                        test_id=test_case.test_id,
                        query=test_case.query,
                        error=str(e)
                    ))
                    run.failed_cases += 1
                
                run.progress_percentage = (i + 1) / run.total_cases * 100
            
            # Calculate aggregate metrics
            run.metrics = self._calculate_run_metrics(run.results)
            
            run.status = EvaluationStatusEnum.COMPLETED
            run.completed_at = datetime.utcnow()
            run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            
            # Save to history
            self._evaluation_history.append(run)
            self._save_run(run)
            
        except Exception as e:
            run.status = EvaluationStatusEnum.FAILED
            run.errors.append(str(e))
            self.logger.exception(f"Evaluation run {run.run_id} failed: {e}")
        
        finally:
            # Remove from active runs
            self._active_runs.pop(run.run_id, None)
    
    async def _evaluate_test_case(
        self,
        test_case: TestCase,
        neurax_service: "NeuraXService",
        config: EvaluationConfig
    ) -> EvaluationResult:
        """Evaluate a single test case"""
        start_time = time.time()
        
        # Perform search
        search_result = await neurax_service.search(
            query=test_case.query,
            k=max(config.k_values),
            similarity_threshold=config.similarity_thresholds[0],
            generate_response=True
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract results
        retrieved_ids = [r.get("document_id", "") for r in search_result.get("results", [])]
        generated_response = search_result.get("generated_response")
        
        # Calculate metrics
        retrieval_scores = self.calculate_retrieval_metrics(
            retrieved_ids,
            test_case.relevant_document_ids,
            config.k_values
        )
        
        generation_scores = self.calculate_generation_metrics(
            test_case.query,
            generated_response or "",
            [r.get("content_preview", "") for r in search_result.get("results", [])],
            test_case.expected_answer
        )
        
        return EvaluationResult(
            test_id=test_case.test_id,
            query=test_case.query,
            retrieved_document_ids=retrieved_ids,
            generated_response=generated_response,
            retrieval_scores=retrieval_scores,
            generation_scores=generation_scores,
            latency_ms=latency_ms,
            precision=retrieval_scores.get("precision_at_k", {}).get(5, 0),
            recall=retrieval_scores.get("recall_at_k", {}).get(5, 0)
        )
    
    def _calculate_run_metrics(
        self,
        results: List[EvaluationResult]
    ) -> EvaluationMetrics:
        """Calculate aggregate metrics for evaluation run"""
        valid_results = [r for r in results if not r.error]
        
        if not valid_results:
            return EvaluationMetrics()
        
        # Aggregate retrieval
        retrieval = RetrievalMetrics()
        k_values = [1, 3, 5, 10]
        
        precision_sums = {k: 0.0 for k in k_values}
        recall_sums = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        
        for r in valid_results:
            for k in k_values:
                precision_sums[k] += r.retrieval_scores.get("precision_at_k", {}).get(k, 0)
                recall_sums[k] += r.retrieval_scores.get("recall_at_k", {}).get(k, 0)
            mrr_sum += r.retrieval_scores.get("mrr", 0)
        
        n = len(valid_results)
        retrieval.precision_at_k = {k: v / n for k, v in precision_sums.items()}
        retrieval.recall_at_k = {k: v / n for k, v in recall_sums.items()}
        retrieval.mrr = mrr_sum / n
        retrieval.total_queries_evaluated = n
        
        # Aggregate generation
        generation = GenerationMetrics()
        grounding_sum = sum(r.generation_scores.get("grounding_score", 0) for r in valid_results)
        generation.grounding_score = grounding_sum / n
        generation.total_responses_evaluated = n
        
        # Aggregate latency
        latencies = [r.latency_ms for r in valid_results]
        latency = LatencyMetrics(
            average_total_latency_ms=sum(latencies) / n,
            total_requests_measured=n
        )
        
        return EvaluationMetrics(
            retrieval=retrieval,
            generation=generation,
            latency=latency,
            overall_quality_score=(retrieval.mrr + generation.grounding_score) / 2,
            total_test_cases=n
        )
    
    def _save_run(self, run: EvaluationRun) -> None:
        """Save evaluation run to disk"""
        filepath = self.storage_dir / f"run_{run.run_id}.json"
        with open(filepath, "w") as f:
            json.dump(run.model_dump(mode="json"), f, indent=2, default=str)
    
    # ==================== Query Methods ====================
    
    def get_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Get evaluation run by ID"""
        if run_id in self._active_runs:
            return self._active_runs[run_id]
        
        # Check history
        for run in self._evaluation_history:
            if run.run_id == run_id:
                return run
        
        # Check disk
        filepath = self.storage_dir / f"run_{run_id}.json"
        if filepath.exists():
            with open(filepath) as f:
                return EvaluationRun.model_validate(json.load(f))
        
        return None
    
    def list_runs(
        self,
        limit: int = 20,
        status: Optional[EvaluationStatusEnum] = None
    ) -> List[EvaluationRun]:
        """List evaluation runs"""
        runs = list(self._active_runs.values()) + self._evaluation_history
        
        if status:
            runs = [r for r in runs if r.status == status]
        
        # Sort by created_at descending
        runs.sort(key=lambda r: r.created_at, reverse=True)
        
        return runs[:limit]


# Singleton instance
_evaluation_service: Optional[EvaluationService] = None


def get_evaluation_service() -> EvaluationService:
    """Get the singleton evaluation service"""
    global _evaluation_service
    if _evaluation_service is None:
        _evaluation_service = EvaluationService()
    return _evaluation_service
