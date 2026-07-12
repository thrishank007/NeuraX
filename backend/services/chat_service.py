"""Chat: retrieve + generate + cite — mirrors Gradio generate_response_handler."""
from __future__ import annotations

import time
from typing import Any, Generator, Optional

from backend.services.component_registry import ComponentRegistry
from backend.services.search_service import search
from backend.services.status_service import probe_lm_studio


def _citation_to_dict(c: Any) -> dict:
    return {
        "citation_id": getattr(c, "citation_id", 0),
        "source_document": getattr(c, "source_document", ""),
        "source_type": getattr(c, "source_type", ""),
        "content_snippet": getattr(c, "content_snippet", ""),
        "confidence_score": float(getattr(c, "confidence_score", 0.0) or 0.0),
        "file_path": getattr(c, "file_path", ""),
        "page_number": getattr(c, "page_number", None),
        "expandable_link": getattr(c, "expandable_link", ""),
        "timestamp": getattr(c, "timestamp", ""),
    }


def chat(
    registry: ComponentRegistry,
    *,
    query: str,
    similarity_threshold: Optional[float] = None,
    max_docs: int = 5,
) -> dict[str, Any]:
    start = time.time()
    lm = probe_lm_studio()

    search_payload = search(
        registry,
        query=query,
        modality="text",
        similarity_threshold=similarity_threshold,
        k=max_docs,
    )
    raw_results = search_payload.get("raw_results") or []
    sources = search_payload.get("results") or []

    if not raw_results:
        return {
            "query": query,
            "response": "No relevant documents found for context. Try lowering the similarity threshold or indexing more documents.",
            "citations": [],
            "confidence": 0.0,
            "processing_time": time.time() - start,
            "sources": sources,
            "model_used": "",
            "lm_studio_available": lm["lm_studio_reachable"],
        }

    llm, citation_gen = registry.ensure_generation()
    generated = llm.generate_grounded_response(query=query, context=raw_results[:max_docs])
    citation_indices = generated.citations_needed if generated.citations_needed else None
    citations = citation_gen.generate_citations(
        response=generated.response_text,
        sources=raw_results[:max_docs],
        citation_indices=citation_indices,
    )

    return {
        "query": query,
        "response": generated.response_text,
        "citations": [_citation_to_dict(c) for c in citations],
        "confidence": float(getattr(generated, "confidence_score", 0.0) or 0.0),
        "processing_time": time.time() - start,
        "sources": sources[:max_docs],
        "model_used": getattr(generated, "model_used", "") or "",
        "lm_studio_available": lm["lm_studio_reachable"],
    }


def chat_stream_events(
    registry: ComponentRegistry,
    *,
    query: str,
    similarity_threshold: Optional[float] = None,
    max_docs: int = 5,
) -> Generator[dict[str, Any], None, None]:
    """SSE-friendly event generator. Retrieval behavior identical to chat()."""
    yield {"event": "status", "data": {"phase": "retrieval_started"}}
    try:
        result = chat(
            registry,
            query=query,
            similarity_threshold=similarity_threshold,
            max_docs=max_docs,
        )
        yield {"event": "retrieval", "data": {"sources": result["sources"]}}
        yield {"event": "status", "data": {"phase": "generation_started"}}
        yield {
            "event": "message",
            "data": {
                "response": result["response"],
                "confidence": result["confidence"],
                "model_used": result["model_used"],
            },
        }
        yield {"event": "citations", "data": {"citations": result["citations"]}}
        yield {
            "event": "done",
            "data": {
                "processing_time": result["processing_time"],
                "lm_studio_available": result["lm_studio_available"],
            },
        }
    except Exception as exc:
        yield {"event": "error", "data": {"code": "processing_error", "message": str(exc)}}
