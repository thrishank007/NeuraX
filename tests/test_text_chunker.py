"""Regression tests for indexing.text_chunker.

The split loop historically degenerated on text with a sentence boundary
early in the window followed by a long unpunctuated run (PDF-extracted
CVs): it re-found the same stale boundary every iteration and advanced one
character at a time, emitting ~100 near-duplicate slivers of one sentence.
"""
import re

from indexing.text_chunker import _split_text, chunk_document


def test_no_rotation_slivers_after_early_sentence_end():
    # ~480 chars of sentences, then a long unpunctuated bullet run — the
    # shape that made the old loop crawl forward one character per chunk.
    text = (" ".join(f"Filler sentence {i} with plenty of words to consume window space." for i in range(7))
            + " "
            + "".join(f"Built scalable ML system variant {i} from prototype to production\n" for i in range(35)))
    chunks = _split_text(text, chunk_size=800, overlap=100)

    assert len(chunks) <= 6  # ceil(len / stride) + tail, not ~100
    assert all(len(c) > 200 for c in chunks[:-1])
    assert all(c for c in chunks)
    # no chunk may be a substring of another (rotation artifact)
    for i, c in enumerate(chunks):
        if len(c) >= 60:
            assert not any(c in o for j, o in enumerate(chunks) if j != i)


def test_sentence_boundaries_preferred_and_overlap_kept():
    text = " ".join(
        f"Sentence number {i} with enough padding words to matter here." for i in range(60)
    )
    chunks = _split_text(text, chunk_size=800, overlap=100)

    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)
    # consecutive chunks share ~overlap chars of trailing context
    for a, b in zip(chunks, chunks[1:]):
        assert b[:40] in a


def test_short_and_empty_text_passthrough():
    assert _split_text("short text", 800, 100) == ["short text"]
    assert _split_text("   ", 800, 100) == []


def test_full_coverage_on_mixed_punctuation_text():
    parts = []
    for i in range(40):
        parts.append(f"Item {i} description. " + "unpunctuated technical tokens " * 12)
    text = "\n".join(parts)

    chunks = _split_text(text, chunk_size=800, overlap=100)

    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    source, joined = norm(text), norm(" ".join(chunks))
    for i in range(0, len(source) - 60, 100):
        assert source[i : i + 60] in joined


def test_chunk_document_dedupes_identical_sections():
    result = {
        "file_path": "test.pdf",
        "file_type": "pdf",
        "content": [
            {"text": "Header repeated on every page", "page": 1},
            {"text": "Header repeated on every page", "page": 2},
            {"text": "Real body content " * 30, "page": 2},
        ],
    }
    chunks = chunk_document(result)

    contents = [c["content"] for c in chunks]
    assert len(contents) == len(set(contents))
    assert sum(1 for c in contents if c == "Header repeated on every page") == 1
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
