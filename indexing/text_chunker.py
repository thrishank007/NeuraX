"""Recursive text chunker with source metadata preservation."""
from __future__ import annotations

import hashlib
import re
from typing import Any


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* on sentence/paragraph boundaries into overlapping chunks.

    Boundary preference (most to least preferred):
      paragraph break → sentence end → word boundary → hard cut

    A boundary is only accepted when it lies at least ``min_len`` past the
    chunk start.  Without that floor, a boundary just ahead of ``start``
    (e.g. the tail of the previous cut, or a long unpunctuated run) is
    re-found on every iteration and the window crawls forward one character
    at a time, emitting near-duplicate slivers of the same sentence.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    min_len = max(1, min(chunk_size // 2, chunk_size - overlap))

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to cut at a paragraph break
        cut = text.rfind("\n\n", start + min_len, end)
        if cut == -1 or cut <= start:
            # Sentence boundary (. ! ?)
            m = None
            for m in re.finditer(r"[.!?]\s+", text[start + min_len : end]):
                pass  # walk to the last match
            cut = (start + min_len + m.end()) if m else -1

        if cut == -1 or cut <= start:
            # Word boundary
            cut = text.rfind(" ", start + min_len, end)

        if cut == -1 or cut <= start:
            cut = end  # hard cut

        chunks.append(text[start:cut].strip())
        # Advance past the cut, keeping up to ``overlap`` chars of context.
        # When the chunk is shorter than the overlap, honour progress over
        # overlap so the window can never stall (or crawl) on the same cut.
        start = min(cut, max(start + 1, cut - overlap))

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
    seen_hashes: set[str] = set()
    chunk_index = 0
    for section_text, page in raw_sections:
        for piece in _split_text(section_text, chunk_size, overlap):
            if not piece:
                continue
            content_hash = hashlib.sha256(piece.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue  # repeated header/footer-style sections carry no new signal
            seen_hashes.add(content_hash)
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
