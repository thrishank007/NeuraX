"""Unit tests for the NIM reranker client (HTTP mocked — no network)."""
from unittest.mock import MagicMock, patch

from retrieval.nim_reranker import NimReranker

CONFIG = {
    "api_url": "https://example.invalid/v1/retrieval/reranking",
    "api_key": "test-key",
    "model": "nvidia/rerank-qa-mistral-4b",
    "timeout": 5,
    "max_passages": 20,
}


def _response(payload: dict) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = payload
    return mock


def test_rerank_orders_by_returned_rankings():
    r = NimReranker(CONFIG)
    with patch("retrieval.nim_reranker.requests.post") as post:
        post.return_value = _response({"rankings": [{"index": 2}, {"index": 0}, {"index": 1}]})
        order = r.rerank("q", ["a", "b", "c"])
    assert order == [2, 0, 1]
    body = post.call_args.kwargs["json"]
    assert body["query"] == {"text": "q"}
    assert body["passages"] == [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert body["model"] == CONFIG["model"]


def test_rerank_skips_blank_passages_and_maps_back():
    r = NimReranker(CONFIG)
    with patch("retrieval.nim_reranker.requests.post") as post:
        post.return_value = _response({"rankings": [{"index": 1}, {"index": 0}]})
        order = r.rerank("q", ["a", "   ", "c"])
    # blank passage compacted away; indices map back to 0 and 2; the blank
    # slot is appended rather than dropped
    assert order == [2, 0, 1]


def test_rerank_single_passage_is_identity_without_http():
    r = NimReranker(CONFIG)
    with patch("retrieval.nim_reranker.requests.post") as post:
        assert r.rerank("q", ["only"]) == [0]
        post.assert_not_called()


def test_rerank_dicts_degrades_to_input_order_on_api_error():
    r = NimReranker(CONFIG)
    results = [{"content": "a"}, {"content": "b"}]
    with patch("retrieval.nim_reranker.requests.post") as post:
        post.side_effect = RuntimeError("boom")
        out = r.rerank_dicts("q", results, top_k=1)
    assert out == [{"content": "a"}]


def test_rerank_dicts_respects_top_k_and_marks_rank():
    r = NimReranker(CONFIG)
    results = [{"content": f"p{i}"} for i in range(5)]
    with patch("retrieval.nim_reranker.requests.post") as post:
        post.return_value = _response({"rankings": [{"index": 3}, {"index": 0}]})
        out = r.rerank_dicts("q", results, top_k=2)
    assert [x["content"] for x in out] == ["p3", "p0"]
    assert [x["rerank_rank"] for x in out] == [1, 2]


def test_missing_config_raises():
    import pytest
    with pytest.raises(ValueError):
        NimReranker({"api_url": "", "api_key": "", "model": ""})
