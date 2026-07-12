from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_component_registry
from backend.api.errors import APIError
from backend.api.schemas.search import ChatRequest, ChatResponse, CitationItem, FeedbackRequest, SearchResultItem
from backend.services import chat_service
from backend.services.component_registry import ComponentRegistry

router = APIRouter(tags=["chat"])


@router.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(
    body: ChatRequest,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> ChatResponse:
    try:
        result = chat_service.chat(
            registry,
            query=body.query,
            similarity_threshold=body.similarity_threshold,
            max_docs=body.max_docs,
        )
        return ChatResponse(
            query=result["query"],
            response=result["response"],
            citations=[CitationItem(**c) for c in result["citations"]],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            sources=[SearchResultItem(**s) for s in result["sources"]],
            model_used=result["model_used"],
            lm_studio_available=result["lm_studio_available"],
        )
    except ValueError as ve:
        raise APIError("validation_error", str(ve), 400) from ve
    except Exception as exc:
        raise APIError("processing_error", "Chat generation failed", 500) from exc


@router.post("/api/chat/stream")
def chat_stream(
    body: ChatRequest,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> StreamingResponse:
    def event_gen():
        for item in chat_service.chat_stream_events(
            registry,
            query=body.query,
            similarity_threshold=body.similarity_threshold,
            max_docs=body.max_docs,
        ):
            event = item["event"]
            data = json.dumps(item["data"])
            yield f"event: {event}\ndata: {data}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/api/feedback")
def feedback(
    body: FeedbackRequest,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict:
    try:
        fb = registry.ensure_feedback()
        feedback_id = fb.collect_feedback(
            query=body.query,
            response=body.response,
            rating=body.rating,
            comments=body.comments or "",
            query_metadata={"interface": "nextjs"},
        )
        return {"status": "ok", "feedback_id": feedback_id}
    except Exception as exc:
        raise APIError("processing_error", "Failed to store feedback", 500) from exc
