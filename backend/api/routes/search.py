from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.api.dependencies import get_component_registry
from backend.api.errors import APIError
from backend.api.schemas.search import SearchResponse, SearchResultItem
from backend.services import document_service, search_service
from backend.services.component_registry import ComponentRegistry

router = APIRouter(tags=["search"])


@router.post("/api/search", response_model=SearchResponse)
async def search_endpoint(
    query: str = Form(""),
    modality: str = Form("text"),
    similarity_threshold: float | None = Form(None),
    k: int | None = Form(None),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> SearchResponse:
    image_path = None
    audio_path = None
    try:
        if image is not None and image.filename:
            data = await image.read()
            image_path = str(document_service.save_upload(image.filename, data))
        if audio is not None and audio.filename:
            data = await audio.read()
            audio_path = str(document_service.save_upload(audio.filename, data))

        payload = search_service.search(
            registry,
            query=query,
            modality=modality,
            similarity_threshold=similarity_threshold,
            k=k,
            image_path=image_path,
            audio_path=audio_path,
        )
        return SearchResponse(
            query=payload["query"],
            query_type=payload["query_type"],
            results=[SearchResultItem(**r) for r in payload["results"]],
            total_results=payload["total_results"],
            similarity_threshold=payload["similarity_threshold"],
            processing_time=payload["processing_time"],
            transcription=payload.get("transcription"),
        )
    except ValueError as ve:
        raise APIError("validation_error", str(ve), 400) from ve
    except Exception as exc:
        raise APIError("processing_error", "Search failed", 500) from exc
