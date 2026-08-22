"""Cross-file duplicate chunk filtering at ingest time."""
from backend.services.document_service import _filter_cross_file_duplicates


def _chunk(text_hash: str) -> dict:
    return {"content": f"text-{text_hash}", "content_hash": text_hash}


def test_skips_chunk_hashed_under_another_file():
    chunks = [_chunk("a"), _chunk("b")]
    existing = [{"content_hash": "a", "file_path": "other.pdf"}]
    kept, skipped = _filter_cross_file_duplicates(chunks, existing, "this.pdf")
    assert [c["content_hash"] for c in kept] == ["b"]
    assert skipped == 1


def test_keeps_chunk_hashed_under_same_file_being_reingested():
    chunks = [_chunk("a")]
    existing = [{"content_hash": "a", "file_path": "this.pdf"}]
    kept, skipped = _filter_cross_file_duplicates(chunks, existing, "this.pdf")
    assert kept == chunks
    assert skipped == 0


def test_no_foreign_hashes_keeps_everything():
    chunks = [_chunk("a"), _chunk("b")]
    existing = [{"content_hash": "c", "file_path": "this.pdf"}, {"file_path": "other.pdf"}]
    kept, skipped = _filter_cross_file_duplicates(chunks, existing, "this.pdf")
    assert kept == chunks
    assert skipped == 0


def test_empty_collection_keeps_everything():
    chunks = [_chunk("a")]
    kept, skipped = _filter_cross_file_duplicates(chunks, [], "this.pdf")
    assert kept == chunks
    assert skipped == 0
