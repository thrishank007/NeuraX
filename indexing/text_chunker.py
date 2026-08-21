"""Recursive text chunker with source metadata preservation."""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* on sentence/paragraph boundaries into overlapping chunks.

    Boundary preference (most to least preferred):
      paragraph break → sentence end → word boundary → hard cut
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to cut at a paragraph break
        cut = text.rfind("\n\n", start, end)
        if cut == -1 or cut <= start:
            # Sentence boundary (. ! ?)
            m = None
            for m in re.finditer(r"[.!?]\s+", text[start:end]):
                pass  # walk to the last match
            cut = (start + m.end()) if m else -1

        if cut == -1 or cut <= start:
            # Word boundary
            cut = text.rfind(" ", start, end)

        if cut == -1 or cut <= start:
            cut = end  # hard cut

        chunks.append(text[start:cut].strip())
        start = max(start + 1, cut - overlap)

    return [c for c in chunks if c]


def chunk_document(result: dict[str, Any], chunk_size: int = 800, overlap: int = 100) -> list[dict[str, Any]]:
    """Produce a list of chunk dicts from a processed document result.

    Each chunk contains:
      content       – the chunk text
      chunk_index   – 0-based position within the document
      source_file   – original file path
      file_type     – pdf / docx / txt / …
      page          – page number if available (None for non-paged formats)
      content_hash  – sha256 of chunk text (for dedup)
      plus the document-level metadata keys
    """
    file_path = str(result.get("file_path", ""))
    file_type = str(result.get("file_type", ""))
    doc_metadata = result.get("metadata") or {}

    # Assemble raw text preserving page information
    content = result.get("content", "")
    raw_sections: list[tuple[str, int | None]] = []  # (text, page_or_None)

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "").strip()
                page = item.get("page") or item.get("paragraph")
                if text:
                    raw_sections.append((text, page))
            elif isinstance(item, str) and item.strip():
                raw_sections.append((item.strip(), None))
    elif isinstance(content, str) and content.strip():
        raw_sections.append((content.strip(), None))

    if not raw_sections:
        return []

    # Flatten into chunks, carrying the page of the section that started each chunk
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    for section_text, page in raw_sections:
        for piece in _split_text(section_text, chunk_size, overlap):
            if not piece:
                continue
            content_hash = hashlib.sha256(piece.encode()).hexdigest()
            chunk: dict[str, Any] = {
                "content": piece,
                "chunk_index": chunk_index,
                "source_file": file_path,
                "file_type": file_type,
                "page": page,
                "content_hash": content_hash,
            }
            # Carry useful top-level document metadata
            for key in ("title", "author", "subject", "encoding"):
                if key in doc_metadata:
                    chunk[key] = doc_metadata[key]
            chunks.append(chunk)
            chunk_index += 1

    return chunks


def deterministic_chunk_id(file_path: str, chunk_index: int, embedding_model: str) -> str:
    """Stable, collision-resistant ID for a chunk.

    Deterministic across re-ingestions of identical content so upsert
    can avoid duplicates.  Changing the embedding model produces a
    different ID (forcing a fresh index entry) — ponytail: model version
    is the only versioning dimension we need right now; add index_version
    when multi-version collections become necessary.
    """
    key = f"{file_path}|{chunk_index}|{embedding_model}"
    return hashlib.sha256(key.encode()).hexdigest()[:40]
