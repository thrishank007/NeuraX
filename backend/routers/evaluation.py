"""
Evaluation Router

API endpoints for RAG evaluation:
- Real-time metrics
- Evaluation runs
- Test datasets
- Export reports
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from backend.core.settings import settings
from backend.core.dependencies import get_neurax_service
from backend.services.evaluation_service import get_evaluation_service, EvaluationService
from backend.models.evaluation import (
    EvaluationMetrics,
    EvaluationConfig,
    EvaluationRun,
    TestCase,
    TestDataset,
    EvaluationStatusEnum,
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# ==================== Request/Response Models ====================

class MetricsResponse(BaseModel):
    retrieval: dict
    generation: dict
    latency: dict
    overall_quality_score: float
    total_test_cases: int
    time_range_hours: int
    timestamp: str


class RunCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    test_cases: Optional[List[TestCase]] = None
    dataset_id: Optional[str] = None
    k_values: List[int] = Field(default=[1, 3, 5, 10])
    similarity_thresholds: List[float] = Field(default=[0.5])
    include_generation_metrics: bool = True
    include_latency_metrics: bool = True
    max_concurrent_requests: int = Field(default=5, ge=1, le=50)


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    message: str


class RunListResponse(BaseModel):
    runs: List[dict]
    total: int


class DatasetCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    test_cases: List[TestCase]
    version: Optional[str] = None


class DatasetCreateResponse(BaseModel):
    dataset_id: str
    status: str
    message: str


class ExportRequest(BaseModel):
    run_id: str
    format: str = Field(default="json", pattern="^(json|csv|html)$")


# ==================== Dependencies ====================

def get_eval_service() -> EvaluationService:
    return get_evaluation_service()


# ==================== Endpoints ====================

@router.get("/metrics", response_model=MetricsResponse)
async def get_aggregated_metrics(
    time_range_hours: int = 24,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Get aggregated evaluation metrics for the specified time range."""
    try:
        metrics = eval_service.get_aggregated_metrics(time_range_hours)
        
        return MetricsResponse(
            retrieval=metrics.retrieval.model_dump() if metrics.retrieval else {},
            generation=metrics.generation.model_dump() if metrics.generation else {},
            latency=metrics.latency.model_dump() if metrics.latency else {},
            overall_quality_score=metrics.overall_quality_score,
            total_test_cases=metrics.total_test_cases,
            time_range_hours=time_range_hours,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieval")
async def get_retrieval_metrics(
    time_range_hours: int = 24,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Get detailed retrieval metrics."""
    metrics = eval_service.get_aggregated_metrics(time_range_hours)
    return {
        "metrics": metrics.retrieval.model_dump() if metrics.retrieval else {},
        "time_range_hours": time_range_hours,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/generation")
async def get_generation_metrics(
    time_range_hours: int = 24,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Get detailed generation metrics."""
    metrics = eval_service.get_aggregated_metrics(time_range_hours)
    return {
        "metrics": metrics.generation.model_dump() if metrics.generation else {},
        "time_range_hours": time_range_hours,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/latency")
async def get_latency_metrics(
    time_range_hours: int = 24,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Get detailed latency metrics."""
    metrics = eval_service.get_aggregated_metrics(time_range_hours)
    return {
        "metrics": metrics.latency.model_dump() if metrics.latency else {},
        "time_range_hours": time_range_hours,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/runs", response_model=RunCreateResponse)
async def create_evaluation_run(
    request: RunCreateRequest,
    background_tasks: BackgroundTasks,
    neurax_service = Depends(get_neurax_service),
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Start a new evaluation run."""
    try:
        config = EvaluationConfig(
            name=request.name,
            description=request.description,
            test_cases=request.test_cases,
            dataset_id=request.dataset_id,
            k_values=request.k_values,
            similarity_thresholds=request.similarity_thresholds,
            include_generation_metrics=request.include_generation_metrics,
            include_latency_metrics=request.include_latency_metrics,
            max_concurrent_requests=request.max_concurrent_requests
        )
        
        run_id = await eval_service.create_evaluation_run(config, neurax_service)
        
        return RunCreateResponse(
            run_id=run_id,
            status="started",
            message=f"Evaluation run {run_id} started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=RunListResponse)
async def list_evaluation_runs(
    limit: int = 20,
    status: Optional[str] = None,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """List evaluation runs."""
    try:
        status_enum = EvaluationStatusEnum(status) if status else None
        runs = eval_service.list_runs(limit=limit, status=status_enum)
        
        return RunListResponse(
            runs=[
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "status": r.status.value if hasattr(r.status, 'value') else r.status,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "total_cases": r.total_cases,
                    "overall_score": r.metrics.overall_quality_score if r.metrics else None,
                    "mrr": r.metrics.retrieval.mrr if r.metrics and r.metrics.retrieval else None,
                    "grounding_score": r.metrics.generation.grounding_score if r.metrics and r.metrics.generation else None
                }
                for r in runs
            ],
            total=len(runs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}")
async def get_evaluation_run(
    run_id: str,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Get details of a specific evaluation run."""
    run = eval_service.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    return run.model_dump(mode="json")


@router.post("/runs/{run_id}/cancel")
async def cancel_evaluation_run(
    run_id: str,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Cancel a running evaluation."""
    run = eval_service.get_run(run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    if run.status not in [EvaluationStatusEnum.PENDING, EvaluationStatusEnum.RUNNING]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel run in {run.status} status"
        )
    
    run.status = EvaluationStatusEnum.CANCELLED
    
    return {"status": "cancelled", "run_id": run_id}


@router.post("/datasets", response_model=DatasetCreateResponse)
async def create_test_dataset(
    request: DatasetCreateRequest,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Create a new test dataset."""
    import uuid
    
    dataset_id = str(uuid.uuid4())[:8]
    
    dataset = TestDataset(
        dataset_id=dataset_id,
        name=request.name,
        description=request.description,
        test_cases=request.test_cases,
        created_at=datetime.utcnow(),
        version=request.version
    )
    
    # Store dataset (in production, this would go to a database)
    # For now, we just return success
    
    return DatasetCreateResponse(
        dataset_id=dataset_id,
        status="created",
        message=f"Dataset {request.name} created with {len(request.test_cases)} test cases"
    )


@router.get("/datasets")
async def list_test_datasets(
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """List available test datasets."""
    # In production, this would query a database
    return {"datasets": [], "total": 0}


@router.post("/export")
async def export_evaluation_report(
    request: ExportRequest,
    eval_service: EvaluationService = Depends(get_eval_service)
):
    """Export evaluation report in specified format."""
    from fastapi.responses import StreamingResponse
    import json
    import io
    
    run = eval_service.get_run(request.run_id)
    
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {request.run_id} not found")
    
    if request.format == "json":
        content = json.dumps(run.model_dump(mode="json"), indent=2)
        media_type = "application/json"
        filename = f"evaluation_{request.run_id}.json"
        
    elif request.format == "csv":
        # Create CSV from results
        lines = ["test_id,query,precision,recall,latency_ms,error"]
        for r in run.results:
            lines.append(
                f'"{r.test_id}","{r.query[:100]}",{r.precision or ""},'
                f'{r.recall or ""},{r.latency_ms or ""},"{r.error or ""}"'
            )
        content = "\n".join(lines)
        media_type = "text/csv"
        filename = f"evaluation_{request.run_id}.csv"
        
    else:  # HTML
        content = f"""
        <html>
        <head><title>Evaluation Report - {run.name}</title></head>
        <body>
            <h1>{run.name}</h1>
            <p>Status: {run.status}</p>
            <p>Total Cases: {run.total_cases}</p>
            <p>Overall Score: {run.metrics.overall_quality_score if run.metrics else 'N/A'}</p>
        </body>
        </html>
        """
        media_type = "text/html"
        filename = f"evaluation_{request.run_id}.html"
    
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
