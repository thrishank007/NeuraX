"""Document upload, indexing, listing, and deletion."""
from __future__ import annotations

import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from loguru import logger

from backend.config import MAX_UPLOAD_MB, UPLOAD_DIR
from backend.services.component_registry import ComponentRegistry
from backend.services.job_service import IndexJob, job_service
from config import PROCESSING_CONFIG, SECURITY_CONFIG


SUPPORTED_FORMATS = (
    PROCESSING_CONFIG["supported_document_formats"]
    + PROCESSING_CONFIG["supported_image_formats"]
    + PROCESSING_CONFIG["supported_audio_formats"]
)


def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^\w.\- ()\[\]]+", "_", base)
    if not base or base in (".", ".."):
        base = f"upload_{uuid.uuid4().hex[:8]}"
    return base[:180]


def validate_extension(filename: str) -> Tuple[bool, str]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
    return True, ext


def validate_size(size_bytes: int) -> Tuple[bool, str]:
    max_bytes = SECURITY_CONFIG.get("max_upload_size_mb", MAX_UPLOAD_MB) * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File too large ({size_bytes / (1024 * 1024):.1f}MB > {max_bytes // (1024 * 1024)}MB)"
    return True, ""


def save_upload(filename: str, data: bytes) -> Path:
    ok, msg = validate_extension(filename)
    if not ok:
        raise ValueError(msg)
    ok, msg = validate_size(len(data))
    if not ok:
        raise ValueError(msg)

    safe = _safe_filename(filename)
    dest_dir = UPLOAD_DIR / uuid.uuid4().hex
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / safe

    # Path traversal guard
    if not str(dest.resolve()).startswith(str(UPLOAD_DIR.resolve())):
        raise ValueError("Invalid upload path")

    dest.write_bytes(data)
    return dest


def _embed_and_store(registry: ComponentRegistry, result: dict) -> Optional[str]:
    embedding_manager, vector_store, _ = registry.ensure_vector_stack()
    if result.get("file_type") == "image":
        text = result.get("ocr_text") or result.get("content")
        if text:
            emb = embedding_manager.embed_text(text)
            if emb is not None and getattr(emb, "size", 0):
                result["text_embedding"] = emb[0]
                result["embedding_type"] = result.get("embedding_type") or "text"
    elif result.get("file_type") == "audio":
        text = result.get("transcription")
        if text:
            emb = embedding_manager.embed_text(text)
            if emb is not None and getattr(emb, "size", 0):
                result["text_embedding"] = emb[0]
                result["embedding_type"] = result.get("embedding_type") or "text"
    else:
        content = result.get("content")
        if content:
            emb = embedding_manager.embed_text(content)
            if emb is not None and getattr(emb, "size", 0):
                result["text_embedding"] = emb[0]
                result["embedding_type"] = result.get("embedding_type") or "text"

    if "text_embedding" not in result:
        raise ValueError("No embeddable content extracted")

    # Capture ids by counting before/after is fragile; use metadata path match after add
    before = set()
    try:
        existing = vector_store.collection.get(include=["metadatas"])
        before = set(existing.get("ids") or [])
    except Exception:
        before = set()

    vector_store.add_documents([result], np.array([result["text_embedding"]]))

    try:
        after = vector_store.collection.get(include=["metadatas"])
        new_ids = [i for i in (after.get("ids") or []) if i not in before]
        if new_ids:
            return new_ids[-1]
        # Fallback: match by file_path
        file_path = str(result.get("file_path", ""))
        for i, meta in zip(after.get("ids") or [], after.get("metadatas") or []):
            if meta and meta.get("file_path") == file_path:
                return i
    except Exception as exc:
        logger.warning(f"Could not resolve new document id: {exc}")
    return None


def process_files_job(registry: ComponentRegistry, job: IndexJob, file_paths: List[Path]) -> None:
    job.status = "running"
    job.total = len(file_paths)
    job.touch()
    ingestion = registry.ensure_ingestion()
    registry.ensure_vector_stack()

    for i, path in enumerate(file_paths):
        if job.cancel_requested:
            job.status = "cancelled"
            job.logs.append("Cancellation requested — stopping")
            job.touch()
            return
        try:
            job.logs.append(f"Processing {path.name}...")
            job.progress = i / max(len(file_paths), 1)
            job.touch()
            result = ingestion.process_file(str(path))
            if not result:
                job.errors.append(f"Failed to process {path.name}")
                job.logs.append(f"FAILED {path.name}")
            else:
                doc_id = _embed_and_store(registry, result)
                if doc_id:
                    job.document_ids.append(doc_id)
                job.processed += 1
                job.logs.append(f"OK {path.name} ({result.get('file_type', 'unknown')})")
        except Exception as exc:
            logger.error(f"Indexing error for {path}: {exc}")
            job.errors.append(f"{path.name}: {exc}")
            job.logs.append(f"ERROR {path.name}: {exc}")
        finally:
            job.progress = (i + 1) / max(len(file_paths), 1)
            job.touch()

    job.status = "failed" if job.processed == 0 and job.errors else "completed"
    job.progress = 1.0
    job.touch()


def start_indexing_job(registry: ComponentRegistry, file_paths: List[Path]) -> IndexJob:
    job = job_service.create_job(total=len(file_paths))
    thread = threading.Thread(
        target=process_files_job,
        args=(registry, job, file_paths),
        daemon=True,
    )
    thread.start()
    return job


def list_documents(registry: ComponentRegistry, limit: int = 200) -> List[dict]:
    _, vector_store, _ = registry.ensure_vector_stack()
    data = vector_store.collection.get(
        limit=limit,
        include=["metadatas", "documents"],
    )
    items = []
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        content = docs[i] if i < len(docs) and docs[i] else ""
        file_path = meta.get("file_path", "")
        preview = (content or "")[:200]
        if content and len(content) > 200:
            preview += "..."
        items.append(
            {
                "id": doc_id,
                "file_path": file_path,
                "file_name": Path(file_path).name if file_path else doc_id,
                "file_type": meta.get("file_type", ""),
                "embedding_type": meta.get("embedding_type", ""),
                "timestamp": meta.get("timestamp", ""),
                "content_preview": preview,
                "metadata": meta,
            }
        )
    return items


def get_document(registry: ComponentRegistry, doc_id: str) -> Optional[dict]:
    _, vector_store, _ = registry.ensure_vector_stack()
    data = vector_store.collection.get(ids=[doc_id], include=["metadatas", "documents"])
    if not data.get("ids"):
        return None
    meta = (data.get("metadatas") or [{}])[0] or {}
    content = (data.get("documents") or [""])[0] or ""
    file_path = meta.get("file_path", "")
    return {
        "id": doc_id,
        "file_path": file_path,
        "file_name": Path(file_path).name if file_path else doc_id,
        "file_type": meta.get("file_type", ""),
        "embedding_type": meta.get("embedding_type", ""),
        "timestamp": meta.get("timestamp", ""),
        "content_preview": content[:2000],
        "metadata": meta,
    }


def delete_document(registry: ComponentRegistry, doc_id: str) -> None:
    _, vector_store, _ = registry.ensure_vector_stack()
    vector_store.delete_documents([doc_id])
