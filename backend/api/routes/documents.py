from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from backend.api.dependencies import get_component_registry
from backend.api.errors import APIError
from backend.api.schemas.documents import DocumentItem, DocumentListResponse, JobStatusResponse
from backend.services import document_service
from backend.services.component_registry import ComponentRegistry
from backend.services.job_service import job_service

router = APIRouter(tags=["documents"])


@router.get("/api/documents", response_model=DocumentListResponse)
def list_documents(
    registry: ComponentRegistry = Depends(get_component_registry),
) -> DocumentListResponse:
    try:
        docs = document_service.list_documents(registry)
        return DocumentListResponse(
            documents=[DocumentItem(**d) for d in docs],
            total=len(docs),
        )
    except Exception as exc:
        raise APIError("processing_error", "Failed to list documents", 500) from exc


@router.post("/api/documents", response_model=JobStatusResponse, status_code=202)
async def upload_documents(
    files: List[UploadFile] = File(...),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> JobStatusResponse:
    if not files:
        raise APIError("validation_error", "No files uploaded", 400)

    saved = []
    errors = []
    for f in files:
        try:
            data = await f.read()
            path = document_service.save_upload(f.filename or "upload.bin", data)
            saved.append(path)
        except ValueError as ve:
            errors.append(str(ve))
        except Exception as exc:
            errors.append(f"{f.filename}: upload failed")

    if not saved:
        raise APIError(
            "upload_rejected",
            "No valid files to process",
            400,
            details={"errors": errors},
        )

    job = document_service.start_indexing_job(registry, saved)
    if errors:
        job.errors.extend(errors)
        job.logs.extend([f"SKIP: {e}" for e in errors])
    return JobStatusResponse(**job.to_dict())


@router.get("/api/documents/{doc_id}", response_model=DocumentItem)
def get_document(
    doc_id: str,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> DocumentItem:
    if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise APIError("validation_error", "Invalid document id", 400)
    doc = document_service.get_document(registry, doc_id)
    if not doc:
        raise APIError("not_found", "Document not found", 404)
    return DocumentItem(**doc)


@router.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict:
    if ".." in doc_id or "/" in doc_id or "\\" in doc_id:
        raise APIError("validation_error", "Invalid document id", 400)
    try:
        existing = document_service.get_document(registry, doc_id)
        if not existing:
            raise APIError("not_found", "Document not found", 404)
        document_service.delete_document(registry, doc_id)
        return {"status": "deleted", "id": doc_id}
    except APIError:
        raise
    except Exception as exc:
        raise APIError("processing_error", "Failed to delete document", 500) from exc


@router.get("/api/index/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_service.get(job_id)
    if not job:
        raise APIError("job_not_found", "Indexing job not found", 404)
    return JobStatusResponse(**job.to_dict())


@router.post("/api/index/jobs/{job_id}/cancel", response_model=JobStatusResponse)
def cancel_job(job_id: str) -> JobStatusResponse:
    job = job_service.request_cancel(job_id)
    if not job:
        raise APIError("job_not_found", "Indexing job not found", 404)
    return JobStatusResponse(**job.to_dict())
