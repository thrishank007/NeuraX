"""
Evaluation Models

Pydantic models for RAG evaluation dashboard.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class EvaluationStatusEnum(str, Enum):
    """Status of evaluation run"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricTypeEnum(str, Enum):
    """Types of evaluation metrics"""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    LATENCY = "latency"
    END_TO_END = "end_to_end"


class RetrievalMetrics(BaseModel):
    """Metrics for retrieval quality"""
    
    model_config = ConfigDict(from_attributes=True)
    
    # Precision and Recall
    precision_at_k: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    )
    recall_at_k: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    )
    
    # Ranking metrics
    mrr: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean Reciprocal Rank"
    )
    ndcg_at_k: Dict[int, float] = Field(
        default_factory=lambda: {5: 0.0, 10: 0.0}
    )
    map_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean Average Precision"
    )
    
    # Hit rate
    hit_rate_at_k: Dict[int, float] = Field(
        default_factory=lambda: {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    )
    
    # Relevance
    average_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance_distribution: Dict[str, int] = Field(
        default_factory=lambda: {"high": 0, "medium": 0, "low": 0, "none": 0}
    )
    
    # Coverage
    document_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of corpus covered in retrieval"
    )
    
    total_queries_evaluated: int = Field(default=0, ge=0)


class GenerationMetrics(BaseModel):
    """Metrics for generation quality"""
    
    model_config = ConfigDict(from_attributes=True)
    
    # Grounding metrics
    grounding_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How well response is grounded in retrieved context"
    )
    faithfulness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Factual accuracy of response"
    )
    
    # Relevance metrics
    answer_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance of answer to query"
    )
    context_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance of retrieved context to query"
    )
    
    # Quality metrics
    completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How completely the question is answered"
    )
    coherence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Logical flow and readability"
    )
    
    # Hallucination detection
    hallucination_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rate of hallucinated content"
    )
    
    # Citation metrics
    citation_precision: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Accuracy of citations"
    )
    citation_recall: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Coverage of citations"
    )
    
    # Response characteristics
    average_response_length: float = Field(default=0.0, ge=0.0)
    average_citation_count: float = Field(default=0.0, ge=0.0)
    
    total_responses_evaluated: int = Field(default=0, ge=0)


class LatencyMetrics(BaseModel):
    """Metrics for system latency"""
    
    model_config = ConfigDict(from_attributes=True)
    
    # End-to-end latency
    average_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p50_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p90_total_latency_ms: float = Field(default=0.0, ge=0.0)
    p99_total_latency_ms: float = Field(default=0.0, ge=0.0)
    
    # Component latencies
    average_embedding_latency_ms: float = Field(default=0.0, ge=0.0)
    average_retrieval_latency_ms: float = Field(default=0.0, ge=0.0)
    average_generation_latency_ms: float = Field(default=0.0, ge=0.0)
    
    # Latency distribution
    latency_buckets: Dict[str, int] = Field(
        default_factory=lambda: {
            "<100ms": 0,
            "100-500ms": 0,
            "500ms-1s": 0,
            "1-2s": 0,
            "2-5s": 0,
            ">5s": 0
        }
    )
    
    # Throughput
    requests_per_second: float = Field(default=0.0, ge=0.0)
    concurrent_requests_handled: int = Field(default=0, ge=0)
    
    # Timeouts and errors
    timeout_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    
    total_requests_measured: int = Field(default=0, ge=0)


class EvaluationMetrics(BaseModel):
    """Combined evaluation metrics"""
    
    retrieval: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = Field(default_factory=GenerationMetrics)
    latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
    
    # Overall scores
    overall_quality_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Weighted average of all quality metrics"
    )
    
    # Evaluation metadata
    evaluation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    evaluation_duration_seconds: float = Field(default=0.0, ge=0.0)
    total_test_cases: int = Field(default=0, ge=0)


class TestCase(BaseModel):
    """Single test case for evaluation"""
    
    test_id: str = Field(..., description="Unique test case identifier")
    query: str = Field(..., min_length=1, max_length=10000)
    
    # Expected results (ground truth)
    expected_answer: Optional[str] = Field(default=None, max_length=50000)
    relevant_document_ids: List[str] = Field(default_factory=list)
    relevance_scores: Optional[Dict[str, float]] = Field(default=None)
    
    # Test metadata
    category: Optional[str] = None
    difficulty: Optional[str] = Field(
        default=None,
        pattern="^(easy|medium|hard)$"
    )
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TestDataset(BaseModel):
    """Dataset for evaluation"""
    
    dataset_id: str = Field(..., description="Unique dataset identifier")
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    
    test_cases: List[TestCase] = Field(default_factory=list)
    
    # Dataset metadata
    total_cases: int = Field(default=0, ge=0)
    categories: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    # Dataset source
    source: Optional[str] = None
    version: str = Field(default="1.0.0")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation run"""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)
    
    # Test dataset
    dataset_id: Optional[str] = None
    test_cases: Optional[List[TestCase]] = None
    
    # Evaluation parameters
    metrics_to_evaluate: List[MetricTypeEnum] = Field(
        default_factory=lambda: [
            MetricTypeEnum.RETRIEVAL,
            MetricTypeEnum.GENERATION,
            MetricTypeEnum.LATENCY
        ]
    )
    k_values: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    
    # Search parameters to test
    similarity_thresholds: List[float] = Field(
        default_factory=lambda: [0.5]
    )
    
    # Sampling
    sample_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of test cases to sample"
    )
    random_seed: Optional[int] = Field(default=None)
    
    # A/B testing
    enable_ab_testing: bool = Field(default=False)
    variants: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Result of a single test case evaluation"""
    
    test_id: str
    query: str
    
    # Actual results
    retrieved_document_ids: List[str] = Field(default_factory=list)
    generated_response: Optional[str] = None
    
    # Scores
    retrieval_scores: Dict[str, float] = Field(default_factory=dict)
    generation_scores: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0.0)
    
    # Comparison with ground truth
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Errors
    error: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvaluationRun(BaseModel):
    """Complete evaluation run"""
    
    run_id: str = Field(..., description="Unique run identifier")
    name: str
    status: EvaluationStatusEnum = Field(default=EvaluationStatusEnum.PENDING)
    
    # Configuration
    config: EvaluationConfig
    
    # Results
    metrics: Optional[EvaluationMetrics] = None
    results: List[EvaluationResult] = Field(default_factory=list)
    
    # Progress
    total_cases: int = Field(default=0, ge=0)
    completed_cases: int = Field(default=0, ge=0)
    failed_cases: int = Field(default=0, ge=0)
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Errors
    errors: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None


class EvaluationHistory(BaseModel):
    """History of evaluation runs"""
    
    runs: List[EvaluationRun] = Field(default_factory=list)
    total_runs: int = Field(default=0, ge=0)
    
    # Trends
    average_quality_score_trend: List[Dict[str, Any]] = Field(default_factory=list)
    latency_trend: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Best/worst runs
    best_run_id: Optional[str] = None
    worst_run_id: Optional[str] = None


class EvaluationComparisonRequest(BaseModel):
    """Request to compare evaluation runs"""
    
    run_ids: List[str] = Field(..., min_length=2, max_length=10)
    metrics_to_compare: List[str] = Field(default_factory=list)


class EvaluationComparisonResponse(BaseModel):
    """Response for evaluation comparison"""
    
    runs: List[Dict[str, Any]] = Field(default_factory=list)
    comparison: Dict[str, Any] = Field(default_factory=dict)
    winner: Optional[str] = None
    insights: List[str] = Field(default_factory=list)
