from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_component_registry
from backend.api.errors import APIError
from backend.api.schemas.documents import DocumentItem
from backend.services import document_service
from backend.services.component_registry import ComponentRegistry

router = APIRouter(tags=["sources"])


@router.get("/api/sources/{source_id}", response_model=DocumentItem)
def get_source(
    source_id: str,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> DocumentItem:
    """Source inspection reuses document store metadata (no invented fields)."""
    if ".." in source_id or "/" in source_id or "\\" in source_id:
        raise APIError("validation_error", "Invalid source id", 400)
    doc = document_service.get_document(registry, source_id)
    if not doc:
        raise APIError("not_found", "Source not found", 404)
    return DocumentItem(**doc)
