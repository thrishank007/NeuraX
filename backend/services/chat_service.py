"""Chat: retrieve + generate + cite using domain generators."""
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


def _prepare_chat(
    registry: ComponentRegistry,
    *,
    query: str,
    similarity_threshold: Optional[float],
    max_docs: int,
    use_knowledge_graph: bool,
    mode: str,
) -> dict[str, Any]:
    """Shared retrieval + generator setup for chat() and chat_stream_events().

    Returns either a ready-to-emit payload ('early_return') when generation
    is impossible, or the pieces both callers need.
    """
    start = time.time()
    lm = probe_lm_studio()
    graph_warning: Optional[str] = None
    graph_used = False

    search_payload = search(
        registry,
        query=query,
        modality="text",
        similarity_threshold=similarity_threshold,
        k=max_docs,
    )
    raw_results = search_payload.get("raw_results") or []
    sources = search_payload.get("results") or []

    # Optional Graphify document-graph enrichment (non-fatal)
    context_docs = list(raw_results[:max_docs])
    if use_knowledge_graph:
        try:
            graphify = registry.ensure_graphify()
            if graphify is None:
                graph_warning = "Graphify service unavailable; continuing with vector RAG only."
            else:
                ctx, warn = graphify.build_rag_context(query)
                if warn:
                    graph_warning = warn
                if ctx:
                    context_docs = context_docs + [
                        {
                            "content": ctx,
                            "file_path": "graphify://document-graph",
                            "file_type": "knowledge_graph",
                            "metadata": {
                                "file_type": "knowledge_graph",
                                "source": "graphify",
                                "label": "GRAPHIFY_CONTEXT",
                            },
                            "similarity_score": 0.0,
                        }
                    ]
                    graph_used = True
        except Exception as exc:
            graph_warning = f"Graph context skipped: {exc}"

    if mode == "cloud":
        from config import CLOUD_LLM_CONFIG
        from generation.cloud_generator import CloudGenerator
        llm = CloudGenerator(CLOUD_LLM_CONFIG)
        citation_gen = registry.ensure_generation()[1]  # reuse citation generator
    else:
        if not lm["lm_studio_reachable"]:
            return {
                "early_return": {
                    "query": query,
                    "response": "LM Studio is not running or unreachable at http://localhost:1234/v1. Please start LM Studio with a model loaded, or switch to Cloud mode in the header above.",
                    "citations": [],
                    "confidence": 0.0,
                    "processing_time": time.time() - start,
                    "sources": sources[:max_docs],
                    "model_used": "",
                    "lm_studio_available": False,
                    "graph_context_used": graph_used,
                    "graph_warning": graph_warning,
                }
            }
        llm, citation_gen = registry.ensure_generation()

    return {
        "start": start,
        "lm": lm,
        "llm": llm,
        "citation_gen": citation_gen,
        "context_docs": context_docs,
        "raw_results": raw_results,
        "sources": sources,
        "graph_used": graph_used,
        "graph_warning": graph_warning,
    }


def _build_citations(
    citation_gen: Any,
    generated: Any,
    raw_results: list,
    max_docs: int,
    graph_used: bool,
) -> list[dict]:
    citation_indices = generated.citations_needed if generated.citations_needed else None
    citations = citation_gen.generate_citations(
        response=generated.response_text,
        sources=raw_results[:max_docs],
        citation_indices=citation_indices,
    )
    citation_dicts = [_citation_to_dict(c) for c in citations]
    if graph_used:
        citation_dicts.append(
            {
                "citation_id": len(citation_dicts) + 1,
                "source_document": "Graphify Knowledge Graph",
                "source_type": "knowledge_graph",
                "content_snippet": "Document-graph relationships (EXTRACTED/INFERRED/AMBIGUOUS)",
                "confidence_score": 0.5,
                "file_path": "graphify://document-graph",
                "page_number": None,
                "expandable_link": "/graph",
                "timestamp": "",
            }
        )
    return citation_dicts


def chat(
    registry: ComponentRegistry,
    *,
    query: str,
    similarity_threshold: Optional[float] = None,
    max_docs: int = 5,
    use_knowledge_graph: bool = False,
    mode: str = "local",
) -> dict[str, Any]:
    prep = _prepare_chat(
        registry,
        query=query,
        similarity_threshold=similarity_threshold,
        max_docs=max_docs,
        use_knowledge_graph=use_knowledge_graph,
        mode=mode,
    )
    if "early_return" in prep:
        return prep["early_return"]

    generated = prep["llm"].generate_grounded_response(query=query, context=prep["context_docs"])
    citations = _build_citations(
        prep["citation_gen"], generated, prep["raw_results"], max_docs, prep["graph_used"]
    )

    return {
        "query": query,
        "response": generated.response_text,
        "citations": citations,
        "confidence": float(getattr(generated, "confidence_score", 0.0) or 0.0),
        "processing_time": time.time() - prep["start"],
        "sources": prep["sources"][:max_docs],
        "model_used": getattr(generated, "model_used", "") or "",
        "lm_studio_available": prep["lm"]["lm_studio_reachable"],
        "graph_context_used": prep["graph_used"],
        "graph_warning": prep["graph_warning"],
    }


def chat_stream_events(
    registry: ComponentRegistry,
    *,
    query: str,
    similarity_threshold: Optional[float] = None,
    max_docs: int = 5,
    use_knowledge_graph: bool = False,
    mode: str = "local",
) -> Generator[dict[str, Any], None, None]:
    """SSE event generator. Retrieval identical to chat(); generation
    streams token-by-token when the active generator supports it."""
    yield {"event": "status", "data": {"phase": "retrieval_started"}}
    try:
        prep = _prepare_chat(
            registry,
            query=query,
            similarity_threshold=similarity_threshold,
            max_docs=max_docs,
            use_knowledge_graph=use_knowledge_graph,
            mode=mode,
        )
        if "early_return" in prep:
            result = prep["early_return"]
            yield {"event": "retrieval", "data": {"sources": result["sources"]}}
            yield {
                "event": "message",
                "data": {
                    "response": result["response"],
                    "confidence": 0.0,
                    "model_used": "",
                },
            }
            yield {
                "event": "done",
                "data": {
                    "processing_time": result["processing_time"],
                    "lm_studio_available": False,
                    "graph_context_used": result.get("graph_context_used", False),
                    "graph_warning": result.get("graph_warning"),
                },
            }
            return

        yield {"event": "retrieval", "data": {"sources": prep["sources"][:max_docs]}}
        if prep["graph_warning"] or prep["graph_used"]:
            yield {
                "event": "status",
                "data": {
                    "phase": "graph_context",
                    "graph_context_used": prep["graph_used"],
                    "graph_warning": prep["graph_warning"],
                },
            }
        yield {"event": "status", "data": {"phase": "generation_started"}}

        llm = prep["llm"]
        stream_fn = getattr(llm, "generate_grounded_response_stream", None)
        if stream_fn is not None:
            token_gen = stream_fn(query=query, context=prep["context_docs"])
            while True:
                try:
                    delta = next(token_gen)
                except StopIteration as stop:
                    generated = stop.value
                    break
                yield {"event": "token", "data": {"delta": delta}}
        else:
            generated = llm.generate_grounded_response(query=query, context=prep["context_docs"])
            if generated.response_text:
                yield {"event": "token", "data": {"delta": generated.response_text}}

        citations = _build_citations(
            prep["citation_gen"], generated, prep["raw_results"], max_docs, prep["graph_used"]
        )
        yield {
            "event": "message",
            "data": {
                "response": generated.response_text,
                "confidence": float(getattr(generated, "confidence_score", 0.0) or 0.0),
                "model_used": getattr(generated, "model_used", "") or "",
            },
        }
        yield {"event": "citations", "data": {"citations": citations}}
        yield {
            "event": "done",
            "data": {
                "processing_time": time.time() - prep["start"],
                "lm_studio_available": prep["lm"]["lm_studio_reachable"],
                "graph_context_used": prep["graph_used"],
                "graph_warning": prep["graph_warning"],
            },
        }
    except Exception as exc:
        yield {"event": "error", "data": {"code": "processing_error", "message": str(exc)}}
