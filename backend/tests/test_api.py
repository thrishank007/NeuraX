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
    assert body["offline_mode"] is True


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
