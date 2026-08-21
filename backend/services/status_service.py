"""Health and model status probes."""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from config import LM_STUDIO_CONFIG, PROCESSING_CONFIG, SECURITY_CONFIG
from backend.services.component_registry import ComponentRegistry


def probe_lm_studio() -> dict[str, Any]:
    base = LM_STUDIO_CONFIG.get("base_url", "http://localhost:1234/v1")
    url = base.rstrip("/").replace("localhost", "127.0.0.1") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = payload.get("data", []) or []
        ids = [m.get("id", "") for m in models if isinstance(m, dict)]
        return {
            "lm_studio_reachable": True,
            "models_loaded": len(models),
            "model_ids": ids,
            "message": f"{len(models)} model(s) available" if models else "Server up, no models loaded",
        }
    except Exception as exc:
        return {
            "lm_studio_reachable": False,
            "models_loaded": 0,
            "model_ids": [],
            "message": f"LM Studio unreachable: {type(exc).__name__}",
            "details": {"reason": type(exc).__name__},
        }


def build_system_status(registry: ComponentRegistry) -> dict[str, Any]:
    lm = probe_lm_studio()
    collection: dict[str, Any] = {}
    vector_status = "not_initialized"
    if registry.vector_store is not None:
        try:
            collection = registry.vector_store.get_collection_stats() or {}
            vector_status = "ok"
        except Exception:
            vector_status = "error"
    else:
        # Lightweight open for status only
        try:
            from config import CHROMA_CONFIG
            from indexing.vector_store import VectorStore

            vs = VectorStore(
                persist_directory=CHROMA_CONFIG["persist_directory"],
                collection_name=CHROMA_CONFIG["collection_name"],
            )
            collection = vs.get_collection_stats() or {}
            vector_status = "ok"
            try:
                if hasattr(vs, "memory_manager"):
                    vs.memory_manager.stop_monitoring()
            except Exception:
                pass
        except Exception:
            vector_status = "error"

    lm_status = "ok" if lm["lm_studio_reachable"] and lm["models_loaded"] > 0 else (
        "degraded" if lm["lm_studio_reachable"] else "unavailable"
    )

    formats = (
        PROCESSING_CONFIG["supported_document_formats"]
        + PROCESSING_CONFIG["supported_image_formats"]
        + PROCESSING_CONFIG["supported_audio_formats"]
    )

    overall = "ok"
    if vector_status == "error":
        overall = "degraded"
    if lm_status == "unavailable":
        overall = "degraded"

    graphify_status: dict[str, Any] = {"available": False, "enabled": False}
    try:
        graphify = registry.ensure_graphify()
        if graphify is not None:
            st = graphify.get_status()
            graphify_status = {
                "available": st.available,
                "enabled": st.enabled,
                "version": st.version,
                "corpus_file_count": st.corpus_file_count,
                "artifacts_available": st.artifacts_available,
                "build_running": st.build_running,
                "model": st.model,
            }
    except Exception as exc:
        graphify_status = {"available": False, "enabled": False, "error": type(exc).__name__}

    from config import CLOUD_LLM_CONFIG
    cloud_configured = bool(CLOUD_LLM_CONFIG.get("api_url")) and bool(CLOUD_LLM_CONFIG.get("model"))
    cloud_status = {
        "configured": cloud_configured,
        "api_url": CLOUD_LLM_CONFIG.get("api_url", "")[:50] if cloud_configured else "",
        "model": CLOUD_LLM_CONFIG.get("model", "") if cloud_configured else "",
    }

    return {
        "overall": overall,
        "backend": "ok",
        "vector_store": vector_status,
        "lm_studio": lm_status,
        "components": registry.component_flags(),
        "collection": collection,
        "supported_formats": formats,
        "max_upload_mb": SECURITY_CONFIG.get("max_upload_size_mb", 100),
        "offline_mode": not cloud_configured,
        "cloud_llm": cloud_status,
        "graphify": graphify_status,
    }


def build_models_status(registry: ComponentRegistry) -> dict[str, Any]:
    lm = probe_lm_studio()
    details: dict[str, Any] = {}
    current = None
    multimodal = False
    if registry.llm_generator is not None:
        try:
            details = registry.llm_generator.get_model_info() or {}
            current = details.get("current_model")
            multimodal = bool(details.get("supports_multimodal", False))
        except Exception:
            pass
    return {
        **lm,
        "current_model": current,
        "supports_multimodal": multimodal,
        "details": details,
    }
