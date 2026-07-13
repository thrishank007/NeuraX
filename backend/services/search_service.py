"""Search orchestration via QueryProcessor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from backend.services.component_registry import ComponentRegistry


def _format_result(raw: dict) -> dict:
    file_path = raw.get("file_path") or raw.get("metadata", {}).get("file_path", "")
    if not file_path and isinstance(raw.get("metadata"), dict):
        file_path = raw["metadata"].get("file_path", "")
    meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    file_path = file_path or meta.get("file_path", "")
    preview = raw.get("content_preview") or raw.get("content") or ""
    if isinstance(preview, (list, dict)):
        preview = str(preview)[:200]
    return {
        "id": str(raw.get("document_id") or raw.get("id") or meta.get("id") or file_path or ""),
        "file_path": file_path,
        "file_name": Path(file_path).name if file_path else "",
        "file_type": raw.get("file_type") or meta.get("file_type", ""),
        "similarity_score": float(raw.get("similarity_score", 0.0) or 0.0),
        "content_preview": str(preview)[:500],
        "metadata": meta or {k: v for k, v in raw.items() if k not in ("content", "content_preview")},
        "text_similarity": raw.get("text_similarity"),
        "image_similarity": raw.get("image_similarity"),
    }


def search(
    registry: ComponentRegistry,
    *,
    query: str = "",
    modality: str = "text",
    similarity_threshold: Optional[float] = None,
    k: Optional[int] = None,
    image_path: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> dict[str, Any]:
    _, _, query_processor = registry.ensure_vector_stack()
    if query_processor is None:
        raise RuntimeError("Query processor unavailable")

    if similarity_threshold is not None:
        query_processor.update_similarity_threshold(similarity_threshold)

    transcription = None

    if modality == "text":
        if not query.strip():
            raise ValueError("Query text is required")
        result = query_processor.process_text_query(query, k=k)
    elif modality == "image":
        if not image_path:
            raise ValueError("Image file is required for image search")
        result = query_processor.process_image_query(image_path, k=k)
        query = query or "[Image Query]"
    elif modality == "voice":
        if not audio_path:
            raise ValueError("Audio file is required for voice search")
        stt = registry.ensure_stt()
        stt_result = stt.process_voice_query_with_fallback(audio_path)
        if not stt_result.get("success"):
            raise ValueError(stt_result.get("fallback_message", "Voice processing failed"))
        transcription = stt_result["transcribed_text"]
        result = query_processor.process_text_query(transcription, k=k)
        query = transcription
    elif modality == "multimodal":
        if not query.strip() or not image_path:
            raise ValueError("Both text and image are required for multimodal search")
        result = query_processor.process_multimodal_query(query, image_path, k=k)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    return {
        "query": query,
        "query_type": result.query_type,
        "results": [_format_result(r) for r in result.results],
        "total_results": result.total_results,
        "similarity_threshold": result.similarity_threshold,
        "processing_time": result.processing_time,
        "transcription": transcription,
        "raw_results": result.results,
    }
