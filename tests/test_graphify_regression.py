"""Regression: NeuraX stays healthy without Graphify; security KG untouched."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_import_graphify_service_without_package():
    """Importing our integration must not require the graphifyy package."""
    # Ensure graphify package is not required at import time
    with patch.dict(sys.modules, {"graphify": None}):
        # re-import cleanly
        import importlib
        import kg_security.graphify_service as mod

        importlib.reload(mod)
        svc = mod.GraphifyService(
            {
                "enabled": True,
                "executable": "graphify-not-installed-xyz",
                "workspace_dir": Path("data") / "graphify_test_ws",
                "base_url": "http://127.0.0.1:1234/v1",
                "api_key": "x",
                "model": "m",
                "install_hint": "hint",
                "allow_non_local_endpoint": False,
            }
        )
        assert svc.is_available() is False
        st = svc.get_status()
        assert st.available is False


def test_knowledge_graph_manager_still_works():
    from kg_security.knowledge_graph_manager import KnowledgeGraphManager

    kg = KnowledgeGraphManager()
    assert kg is not None
    assert hasattr(kg, "graph")
    # Empty graph: connectivity stats may raise; ensure core API still exists
    assert hasattr(kg, "add_document_to_graph")
    assert hasattr(kg, "export_viz_data")
    # Security graph remains a distinct class from GraphifyService
    from kg_security.graphify_service import GraphifyService

    assert KnowledgeGraphManager is not GraphifyService


def test_api_starts_and_graphify_status(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    r2 = client.get("/api/graphify/status")
    assert r2.status_code == 200
    body = r2.json()
    assert "available" in body
    assert "install_hint" in body
    assert "enabled" in body


def test_system_status_includes_graphify(client):
    r = client.get("/api/system/status")
    assert r.status_code == 200
    body = r.json()
    assert "graphify" in body
    assert "components" in body


def test_chat_schema_accepts_kg_flag(client):
    # Will likely fail generation without models, but validation should pass shape
    # Empty retrieval path returns 200 with message
    with patch("backend.services.chat_service.search") as mock_search:
        mock_search.return_value = {
            "raw_results": [],
            "results": [],
        }
        with patch("backend.services.chat_service.probe_lm_studio") as mock_lm:
            mock_lm.return_value = {"lm_studio_reachable": False}
            r = client.post(
                "/api/chat",
                json={
                    "query": "hello world",
                    "use_knowledge_graph": True,
                    "max_docs": 3,
                },
            )
            assert r.status_code == 200
            body = r.json()
            assert "graph_context_used" in body
            assert isinstance(body["graph_context_used"], bool)
