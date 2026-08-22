"""Backend API tests — health, validation, error shape, jobs."""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "neurax-api"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "health" in r.json()


def test_system_status(client):
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    assert "backend" in body
    assert "vector_store" in body
    assert "lm_studio" in body
    # offline_mode mirrors cloud configuration rather than being constant:
    # environments with NEURAX_CLOUD_* credentials legitimately report online
    from config import CLOUD_LLM_CONFIG

    cloud_configured = bool(CLOUD_LLM_CONFIG.get("api_url")) and bool(CLOUD_LLM_CONFIG.get("model"))
    assert body["offline_mode"] is (not cloud_configured)


def test_models_status(client):
    r = client.get("/api/models/status")
    assert r.status_code == 200
    body = r.json()
    assert "lm_studio_reachable" in body
    assert "message" in body


def test_upload_rejected_bad_extension(client):
    files = [("files", ("malware.exe", b"not-a-real-exe", "application/octet-stream"))]
    r = client.post("/api/documents", files=files)
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] in ("upload_rejected", "validation_error")
    # No stack traces
    assert "Traceback" not in r.text


def test_upload_empty_list(client):
    r = client.post("/api/documents", files=[])
    # FastAPI may 422 on missing files
    assert r.status_code in (400, 422)


def test_chat_malformed(client):
    r = client.post("/api/chat", json={})
    assert r.status_code == 422


def test_chat_empty_query(client):
    r = client.post("/api/chat", json={"query": ""})
    assert r.status_code == 422


def test_chat_stream_early_return_sse(client, monkeypatch):
    """LM Studio unreachable: SSE carries retrieval + message + done, no tokens."""
    from backend.services import chat_service

    def fake_prepare(registry, *, query, **kwargs):
        return {
            "early_return": {
                "query": query,
                "response": "LM Studio guidance text",
                "confidence": 0.0,
                "processing_time": 0.01,
                "sources": [],
                "model_used": "",
                "lm_studio_available": False,
                "graph_context_used": False,
                "graph_warning": None,
            }
        }

    monkeypatch.setattr(chat_service, "_prepare_chat", fake_prepare)
    with client.stream("POST", "/api/chat/stream", json={"query": "hi"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "\n".join(resp.iter_lines())

    assert "event: retrieval" in text
    assert "event: message" in text
    assert "LM Studio guidance text" in text
    assert "event: done" in text
    assert "event: token" not in text


def test_chat_stream_tokens_flow_through_sse(client, monkeypatch):
    """Streaming-capable generator: token deltas precede the final message."""
    from generation.lmstudio_generator import GeneratedResponse
    from backend.services import chat_service

    class FakeStreamLLM:
        def generate_grounded_response_stream(self, query, context):
            yield "Hel"
            yield "lo"
            return GeneratedResponse(
                response_text="Hello", confidence_score=0.8, processing_time=0.01,
                context_used=context, grounding_score=0.8, citations_needed=[0],
                model_used="fake-model",
            )

    def fake_prepare(registry, *, query, **kwargs):
        return {
            "start": 0.0,
            "lm": {"lm_studio_reachable": True},
            "llm": FakeStreamLLM(),
            "citation_gen": None,
            "context_docs": [{"content": "c", "file_path": "f"}],
            "raw_results": [],
            "sources": [],
            "graph_used": False,
            "graph_warning": None,
        }

    monkeypatch.setattr(chat_service, "_prepare_chat", fake_prepare)
    monkeypatch.setattr(chat_service, "_build_citations", lambda *a, **k: [])
    with client.stream("POST", "/api/chat/stream", json={"query": "hi"}) as resp:
        assert resp.status_code == 200
        text = "\n".join(resp.iter_lines())

    assert '"delta": "Hel"' in text
    assert '"delta": "lo"' in text
    assert "event: citations" in text
    assert "event: done" in text
    assert text.index("event: token") < text.index("event: message")


def test_job_not_found(client):
    r = client.get("/api/index/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "job_not_found"


def test_cancel_job_not_found(client):
    r = client.post("/api/index/jobs/does-not-exist/cancel")
    assert r.status_code == 404


def test_delete_missing_document(client):
    with patch("backend.api.routes.documents.document_service.get_document", return_value=None):
        with patch(
            "backend.api.routes.documents.document_service.list_documents",
            return_value=[],
        ):
            r = client.delete("/api/documents/nonexistent-id")
            assert r.status_code == 404
            assert "error" in r.json()
            assert "Traceback" not in r.text


def test_search_text_validation(client):
    r = client.post(
        "/api/search",
        data={"query": "", "modality": "text"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "validation_error"


def test_unhandled_error_no_stack(client):
    from backend.api.errors import unhandled_error_handler
    from starlette.requests import Request

    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    request = Request(scope)

    import asyncio

    resp = asyncio.get_event_loop().run_until_complete(
        unhandled_error_handler(request, RuntimeError("secret boom"))
    )
    assert resp.status_code == 500
    body = resp.body.decode()
    assert "secret boom" not in body
    assert "Traceback" not in body
    assert "internal_error" in body


def test_upload_valid_txt_starts_job(client):
    """Upload a small text file; job is accepted (indexing may fail if models missing)."""
    content = b"NeuraX baseline test document about secure offline retrieval."
    files = [("files", ("baseline.txt", content, "text/plain"))]
    r = client.post("/api/documents", files=files)
    assert r.status_code == 202
    body = r.json()
    assert "job_id" in body
    assert body["status"] in ("queued", "running", "completed", "failed")

    job_r = client.get(f"/api/index/jobs/{body['job_id']}")
    assert job_r.status_code == 200
    assert job_r.json()["job_id"] == body["job_id"]


def test_lm_studio_probe_shape(client):
    r = client.get("/api/models/status")
    data = r.json()
    assert isinstance(data["lm_studio_reachable"], bool)
    if not data["lm_studio_reachable"]:
        assert "unreachable" in data["message"].lower() or "LM Studio" in data["message"]
