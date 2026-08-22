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


def _filter_cross_file_duplicates(
    chunks: List[dict], existing_metas: List[dict], file_path: str
) -> Tuple[List[dict], int]:
    """Drop chunks whose text is already indexed under a different file.

    Identical chunk content adds no retrieval signal (e.g. the same file
    uploaded twice lands in different upload folders), but chunks of the
    file being re-ingested must survive: they are replaced by the upsert
    flow, not duplicated. Returns (kept, skipped_count).
    """
    foreign_hashes = {
        m.get("content_hash")
        for m in existing_metas or []
        if m.get("content_hash") and m.get("file_path") != file_path
    }
    if not foreign_hashes:
        return chunks, 0
    kept = [c for c in chunks if c.get("content_hash") not in foreign_hashes]
    return kept, len(chunks) - len(kept)


def _embed_and_store(registry: ComponentRegistry, result: dict) -> Optional[str]:
    """Chunk text documents, embed each chunk, upsert with deterministic IDs.

    Non-text modalities (image OCR, audio transcription) are still stored as
    a single embedding — chunking them is future work once we have enough
    content to measure benefit.
    """
    from indexing.text_chunker import chunk_document, deterministic_chunk_id

    embedding_manager, vector_store, _ = registry.ensure_vector_stack()
    embedding_model = embedding_manager.text_embedding_model
    embedding_dim = embedding_manager.text_embedding_dimension

    file_type = result.get("file_type", "")

    # ── Non-text modalities: CLIP image vector store (512-d) ────────────────
    if file_type == "image":
        file_path = str(result.get("file_path", ""))
        emb = embedding_manager.embed_image(file_path)
        if emb is None or not getattr(emb, "size", 0):
            raise ValueError(f"Failed to generate CLIP embedding for {file_path}")

        image_vs = registry.ensure_image_vector_stack()
        from indexing.text_chunker import deterministic_chunk_id
        chunk_id = deterministic_chunk_id(file_path, 0, "clip-vit-base-patch32")

        ocr_text = str(result.get("ocr_text") or "").strip()
        doc_text = ocr_text if ocr_text else f"Image: {Path(file_path).name}"

        meta = {
            "file_path": file_path,
            "file_type": "image",
            "embedding_type": "image",
            "embedding_provider": "clip",
            "embedding_model": "openai/clip-vit-base-patch32",
        }
        image_vs.collection.upsert(
            ids=[chunk_id],
            embeddings=[emb[0].tolist()],
            metadatas=[meta],
            documents=[doc_text[:1000]],
        )
        logger.info(f"Indexed image file to CLIP image store: {file_path}")
        return chunk_id

    if file_type == "audio":
        text = str(result.get("transcription") or "").strip()
        if not text:
            text = f"Audio: {Path(result.get('file_path', '')).name}"
        emb = embedding_manager.embed_text(text)
        if emb is None or not getattr(emb, "size", 0):
            raise ValueError("No embeddable content extracted from audio")

        image_vs = registry.ensure_image_vector_stack()
        file_path = str(result.get("file_path", ""))
        from indexing.text_chunker import deterministic_chunk_id
        chunk_id = deterministic_chunk_id(file_path, 0, embedding_manager.text_embedding_model)

        meta = {
            "file_path": file_path,
            "file_type": "audio",
            "embedding_type": "text",
            "embedding_provider": "nvidia_nim" if embedding_manager.uses_nim_embeddings else "minilm",
            "embedding_model": embedding_manager.text_embedding_model,
        }
        image_vs.collection.upsert(
            ids=[chunk_id],
            embeddings=[emb[0].tolist()],
            metadatas=[meta],
            documents=[str(text)[:1000]],
        )
        logger.info(f"Indexed audio file to image/audio store: {file_path}")
        return chunk_id

    # ── Text documents: chunk → dedupe → embed → upsert ────────────────────
    chunks = chunk_document(result)
    if not chunks:
        raise ValueError("No embeddable content extracted")

    file_path = str(result.get("file_path", ""))
    doc_metadata = result.get("metadata") or {}

    try:
        existing_metas = vector_store.collection.get(include=["metadatas"]).get("metadatas") or []
    except Exception as exc:
        logger.warning(f"Could not read existing chunks for dedup, indexing all: {exc}")
        existing_metas = []
    chunks, skipped = _filter_cross_file_duplicates(chunks, existing_metas, file_path)
    if skipped:
        logger.info(
            f"Skipped {skipped} chunk(s) for {file_path}: identical content already indexed under another file"
        )
    if not chunks:
        logger.warning(f"All chunks of {file_path} duplicate existing content; nothing indexed")
        return None

    texts = [c["content"] for c in chunks]
    embeddings = embedding_manager.embed_text(texts)  # batch; passage mode via NIM or MiniLM

    ids: list[str] = []
    chunk_dicts: list[dict] = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = deterministic_chunk_id(file_path, i, embedding_model)
        ids.append(chunk_id)
        chunk_dicts.append({
            "content": chunk["content"],
            "file_path": file_path,
            "file_type": file_type,
            "embedding_type": "text",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "page": chunk.get("page"),
            "content_hash": chunk["content_hash"],
            "embedding_provider": "nvidia_nim" if embedding_manager.uses_nim_embeddings else "minilm",
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dim,
            # doc-level metadata
            "title": doc_metadata.get("title", ""),
            "author": doc_metadata.get("author", ""),
        })

    # Upsert: delete existing IDs for this file first (handles re-ingest cleanly)
    try:
        existing = vector_store.collection.get(
            where={"file_path": file_path}, include=[]
        )
        old_ids = existing.get("ids") or []
        if old_ids:
            vector_store.collection.delete(ids=old_ids)
            logger.info(f"Removed {len(old_ids)} old chunk(s) for {file_path} before re-index")
    except Exception as exc:
        logger.warning(f"Could not check existing chunks for {file_path}: {exc}")

    # Store all chunks in one call using low-level Chroma API for explicit IDs
    vector_store.collection.upsert(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[
            {k: v for k, v in d.items() if k != "content" and isinstance(v, (str, int, float, bool)) and v is not None}
            for d in chunk_dicts
        ],
        documents=[d["content"] for d in chunk_dicts],
    )
    logger.info(f"Indexed {len(chunks)} chunk(s) for {file_path}")
    return ids[0] if ids else None


def process_files_job(registry: ComponentRegistry, job: IndexJob, file_paths: List[Path]) -> None:
    job.status = "running"
    job.total = len(file_paths)
    job.touch()
    ingestion = registry.ensure_ingestion()
    registry.ensure_vector_stack()
    graphify = None
    try:
        graphify = registry.ensure_graphify()
    except Exception as exc:
        logger.warning(f"Graphify not available during ingestion: {exc}")

    corpus_ready = 0
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

                # Persist a safe copy into the Graphify corpus (non-fatal)
                if graphify is not None:
                    try:
                        entry = graphify.persist_to_corpus(
                            path,
                            original_filename=path.name,
                            file_type=str(result.get("file_type") or ""),
                        )
                        corpus_ready += 1
                        job.logs.append(
                            f"Graph corpus: {entry.get('message', 'ready')} ({entry.get('stored_filename')})"
                        )
                    except Exception as gexc:
                        logger.warning(f"Graphify corpus copy failed for {path.name}: {gexc}")
                        job.logs.append(f"Graph corpus skipped for {path.name}: {gexc}")
        except Exception as exc:
            logger.error(f"Indexing error for {path}: {exc}")
            job.errors.append(f"{path.name}: {exc}")
            job.logs.append(f"ERROR {path.name}: {exc}")
        finally:
            job.progress = (i + 1) / max(len(file_paths), 1)
            job.touch()

    # Optional auto-update after the batch finishes (never rolls back vector indexing)
    if graphify is not None and corpus_ready > 0:
        try:
            auto = bool(getattr(graphify, "config", {}).get("auto_update_after_ingestion", False))
            if auto:
                job.logs.append("Auto-updating Graphify knowledge graph...")
                job.touch()
                build = graphify.update_graph()
                if build.success:
                    job.logs.append("Graphify update completed")
                else:
                    job.logs.append(f"Graphify update skipped/failed: {build.message}")
            else:
                job.logs.append(
                    "Files ready for Graphify. Open Knowledge Graph to Build/Update the document graph."
                )
        except Exception as exc:
            logger.warning(f"Graphify auto-update failed (non-fatal): {exc}")
            job.logs.append(f"Graphify auto-update failed (non-fatal): {exc}")

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
