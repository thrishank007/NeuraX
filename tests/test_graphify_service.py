"""Unit and integration tests for GraphifyService (fake CLI, no real Graphify)."""
from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kg_security.graphify_schema import (
    GraphParseError,
    load_graph_json,
    parse_graph_payload,
)
from kg_security.graphify_service import (
    GraphifyService,
    is_loopback_url,
    redact_secrets,
    sanitize_filename,
)


FIXTURE_GRAPH = {
    "directed": True,
    "multigraph": False,
    "nodes": [
        {
            "id": "n1",
            "label": "Alpha",
            "file_type": "concept",
            "source_file": "a.txt",
            "community": 0,
        },
        {
            "id": "n2",
            "label": "Beta",
            "file_type": "entity",
            "source_file": "b.txt",
            "community": 0,
        },
        {
            "id": "n3",
            "label": "Gamma",
            "file_type": "concept",
            "source_file": "a.txt",
            "community": 1,
        },
    ],
    "links": [
        {
            "source": "n1",
            "target": "n2",
            "relation": "related_to",
            "confidence": "extracted",
            "confidence_score": 0.9,
            "source_file": "a.txt",
        },
        {
            "source": "n2",
            "target": "n3",
            "relation": "mentions",
            "confidence": "inferred",
            "confidence_score": 0.6,
        },
    ],
}


_FAKE_GRAPHIFY_PY = textwrap.dedent(
    """\
    import json, sys, time
    from pathlib import Path
    argv = sys.argv[1:]
    if not argv:
        print("fake-graphify 0.0.1-test")
        sys.exit(0)
    cmd = argv[0]
    if cmd in ("--version", "-V"):
        print("fake-graphify 0.0.1-test")
        sys.exit(0)
    if cmd in ("--help", "-h"):
        print("Usage: graphify extract|query|explain|path|update")
        print("  extract <path> --backend --model --out --max-concurrency --api-timeout --mode --token-budget --force")
        print("  query <q> --graph --budget")
        print("  explain <node> --graph")
        print("  path <a> <b> --graph")
        print("  update <path>")
        sys.exit(0)
    if cmd == "extract" and "--help" in argv:
        print("extract --backend --model --out DIR --max-concurrency N --api-timeout S --mode deep --token-budget N --force")
        sys.exit(0)
    if cmd == "query" and "--help" in argv:
        print("query --graph --budget")
        sys.exit(0)
    if cmd == "extract":
        out = None
        path = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else "."
        i = 0
        while i < len(argv):
            if argv[i] == "--out" and i + 1 < len(argv):
                out = argv[i + 1]
                i += 2
                continue
            i += 1
        out_dir = Path(out) if out else Path(path) / "graphify-out"
        out_dir.mkdir(parents=True, exist_ok=True)
        graph = {
            "directed": True,
            "multigraph": False,
            "nodes": [
                {"id": "n1", "label": "Alpha", "file_type": "concept", "source_file": "a.txt", "community": 0},
                {"id": "n2", "label": "Beta", "file_type": "entity", "source_file": "b.txt", "community": 0},
            ],
            "links": [
                {"source": "n1", "target": "n2", "relation": "related_to", "confidence": "extracted", "confidence_score": 0.9}
            ],
        }
        (out_dir / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
        (out_dir / "graph.html").write_text("<html><body>graph</body></html>", encoding="utf-8")
        (out_dir / "GRAPH_REPORT.md").write_text("# Report\\nOK", encoding="utf-8")
        print("extract complete")
        sys.exit(0)
    if cmd == "update":
        print("update complete")
        sys.exit(0)
    if cmd == "query":
        if "--fail" in argv:
            print("error", file=sys.stderr)
            sys.exit(2)
        print("Query result: Alpha relates to Beta")
        sys.exit(0)
    if cmd == "explain":
        print("Explain: Alpha is a concept from a.txt")
        sys.exit(0)
    if cmd == "path":
        print("Path: Alpha -> related_to -> Beta")
        sys.exit(0)
    if cmd == "sleep-timeout":
        time.sleep(30)
        sys.exit(0)
    print(f"unknown {cmd}", file=sys.stderr)
    sys.exit(1)
    """
)


def _write_fake_graphify(bin_dir: Path) -> Path:
    """Create a fake graphify executable that handles extract/query/explain/path."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    py = bin_dir / "fake_graphify.py"
    py.write_text(_FAKE_GRAPHIFY_PY, encoding="utf-8")
    if os.name == "nt":
        bat = bin_dir / "graphify.bat"
        bat.write_text(
            f'@echo off\r\n"{sys.executable}" "{py}" %*\r\n',
            encoding="utf-8",
        )
        return bat
    script = bin_dir / "graphify"
    script.write_text(
        f"#!/usr/bin/env {sys.executable}\n" + _FAKE_GRAPHIFY_PY,
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


@pytest.fixture
def fake_graphify(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    exe = _write_fake_graphify(bin_dir)
    # Put fake on PATH
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    workspace = tmp_path / "ws"
    cfg = {
        "enabled": True,
        "executable": str(exe),
        "workspace_dir": workspace,
        "default_workspace_id": "default",
        "backend": "openai",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio-secret-key",
        "model": "test-model",
        "mode": "deep",
        "auto_update_after_ingestion": False,
        "max_concurrency": 1,
        "api_timeout_seconds": 30,
        "process_timeout_seconds": 60,
        "token_budget": 1000,
        "max_query_output_chars": 12000,
        "max_query_length": 2000,
        "max_visualization_nodes": 1000,
        "max_graph_json_bytes": 5 * 1024 * 1024,
        "max_stdout_capture_bytes": 1024 * 1024,
        "max_stderr_capture_bytes": 512 * 1024,
        "max_rag_context_chars": 4000,
        "max_rag_context_nodes": 25,
        "max_rag_context_edges": 40,
        "allow_non_local_endpoint": False,
        "install_hint": "install graphify",
    }
    svc = GraphifyService(cfg)
    return svc, cfg, exe


# ---- Unit: helpers ----
def test_is_loopback_url():
    assert is_loopback_url("http://localhost:1234/v1")
    assert is_loopback_url("http://127.0.0.1:1234/v1")
    assert is_loopback_url("http://[::1]:1234/v1")
    assert not is_loopback_url("http://example.com/v1")


def test_redact_secrets():
    text = "OPENAI_API_KEY=sk-supersecret123 and password=hunter2"
    red = redact_secrets(text)
    assert "sk-supersecret123" not in red
    assert "hunter2" not in red
    assert "REDACTED" in red


def test_sanitize_filename():
    assert ".." not in sanitize_filename("../etc/passwd")
    assert "/" not in sanitize_filename("a/b/c.txt")
    assert sanitize_filename("ok file.pdf").endswith(".pdf")


def test_parse_graph_links():
    g = parse_graph_payload(FIXTURE_GRAPH)
    assert len(g.nodes) == 3
    assert len(g.edges) == 2


def test_parse_graph_edges_key():
    payload = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"source": "a", "target": "b", "relation": "x"}],
    }
    g = parse_graph_payload(payload)
    assert len(g.edges) == 1


def test_malformed_graph_rejection():
    with pytest.raises(GraphParseError):
        parse_graph_payload([1, 2, 3])
    with pytest.raises(GraphParseError):
        parse_graph_payload({"nodes": [{"label": "no id"}]})


def test_missing_node_edge_rejection():
    with pytest.raises(GraphParseError):
        parse_graph_payload(
            {
                "nodes": [{"id": "a"}],
                "links": [{"source": "a", "target": "missing"}],
            }
        )


def test_load_graph_json_file(tmp_path):
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(FIXTURE_GRAPH), encoding="utf-8")
    g = load_graph_json(p)
    assert len(g.nodes) == 3


# ---- Service unit/integration with fake CLI ----
def test_executable_detection(fake_graphify):
    svc, _, exe = fake_graphify
    assert svc.is_available()
    assert svc.resolve_executable()
    assert Path(svc.resolve_executable()).exists() or svc.resolve_executable() == str(exe)


def test_version_detection(fake_graphify):
    svc, _, _ = fake_graphify
    ver = svc.get_version()
    assert ver is not None
    assert "fake-graphify" in ver or "0.0.1" in ver


def test_command_construction_no_shell(fake_graphify, monkeypatch):
    svc, _, _ = fake_graphify
    # force capability detection with real help from fake
    caps = svc.detect_capabilities(force=True)
    corpus = svc.corpus_dir("default")
    out = svc.output_dir("default")
    cmd = svc._build_extract_command(corpus, out, force=True, caps=caps)
    assert cmd[0]
    assert "extract" in cmd
    assert "--backend" in cmd or not caps.extract_backend
    # Ensure shell never used: inspect _run source contract via a spy
    calls = []
    real_run = svc._run

    def spy(command, **kwargs):
        calls.append(list(command))
        return real_run(command, **kwargs)

    monkeypatch.setattr(svc, "_run", spy)
    # also patch subprocess to ensure shell=False
    import subprocess as sp

    original = sp.run

    def tracked(*args, **kwargs):
        assert kwargs.get("shell") is False or kwargs.get("shell") is None or kwargs.get("shell") is False
        # our code always passes shell=False
        assert kwargs.get("shell") is False
        return original(*args, **kwargs)

    monkeypatch.setattr(sp, "run", tracked)
    (corpus / "doc.txt").write_text("hello knowledge graph", encoding="utf-8")
    result = svc.build_graph("default")
    assert result.success
    assert calls


def test_lm_studio_env_construction(fake_graphify):
    svc, cfg, _ = fake_graphify
    env = svc.build_child_env()
    assert env["OPENAI_BASE_URL"] == cfg["base_url"]
    assert env["OPENAI_API_KEY"] == cfg["api_key"]
    assert env["OPENAI_MODEL"] == cfg["model"]
    assert env["GRAPHIFY_QUERY_LOG_DISABLE"] == "1"


def test_non_local_endpoint_rejected(fake_graphify):
    svc, cfg, _ = fake_graphify
    svc.config["base_url"] = "https://api.openai.com/v1"
    svc.config["allow_non_local_endpoint"] = False
    ok, warn = svc._endpoint_check()
    assert not ok
    assert warn


def test_path_containment_and_sanitize(fake_graphify, tmp_path):
    svc, _, _ = fake_graphify
    src = tmp_path / "note.txt"
    src.write_text("content", encoding="utf-8")
    entry = svc.persist_to_corpus(src, original_filename="../../evil.txt")
    stored = Path(entry["path"])
    assert stored.exists()
    # must stay under corpus
    assert "corpus" in str(stored)
    assert ".." not in stored.name


def test_duplicate_file_handling(fake_graphify, tmp_path):
    svc, _, _ = fake_graphify
    a = tmp_path / "a.txt"
    a.write_text("same", encoding="utf-8")
    e1 = svc.persist_to_corpus(a, original_filename="a.txt")
    e2 = svc.persist_to_corpus(a, original_filename="a.txt")
    assert e2["reused"] is True
    b = tmp_path / "b.txt"
    b.write_text("different", encoding="utf-8")
    e3 = svc.persist_to_corpus(b, original_filename="a.txt")
    assert e3["reused"] is False
    assert e3["stored_filename"] != e1["stored_filename"] or e3["sha256"] != e1["sha256"]


def test_manifest_updates(fake_graphify, tmp_path):
    svc, _, _ = fake_graphify
    f = tmp_path / "m.md"
    f.write_text("# title", encoding="utf-8")
    svc.persist_to_corpus(f, original_filename="m.md", file_type="document")
    files = svc.list_corpus_files("default")
    assert len(files) == 1
    assert files[0]["original_filename"] == "m.md"
    assert "sha256" in files[0]


def test_query_length_validation(fake_graphify):
    svc, _, _ = fake_graphify
    r = svc.query_graph("default", "")
    assert not r.success
    long_q = "x" * 5000
    r2 = svc.query_graph("default", long_q)
    assert not r2.success


def test_missing_graphify_behavior(tmp_path):
    cfg = {
        "enabled": True,
        "executable": str(tmp_path / "does-not-exist-graphify"),
        "workspace_dir": tmp_path / "gws",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "x",
        "model": "m",
        "install_hint": "please install",
        "allow_non_local_endpoint": False,
        "process_timeout_seconds": 10,
        "max_query_length": 2000,
    }
    svc = GraphifyService(cfg)
    assert not svc.is_available()
    st = svc.get_status()
    assert st.available is False
    assert "install" in st.install_hint.lower() or "please" in st.install_hint.lower()
    build = svc.build_graph()
    assert not build.success
    assert build.error == "graphify_not_found"


def test_full_build_query_explain_path(fake_graphify, tmp_path):
    svc, _, _ = fake_graphify
    doc = tmp_path / "corpus_doc.txt"
    doc.write_text("Alpha and Beta knowledge", encoding="utf-8")
    svc.persist_to_corpus(doc, original_filename="corpus_doc.txt")
    build = svc.build_graph("default")
    assert build.success, build.message
    arts = svc.get_artifact_paths("default")
    assert arts["graph_json"]
    assert arts["graph_html"]
    assert arts["graph_report"]
    stats = svc.get_graph_stats("default")
    assert stats["node_count"] >= 2
    q = svc.query_graph("default", "How are Alpha and Beta related?")
    assert q.success
    assert "Alpha" in q.output
    ex = svc.explain_node("default", "Alpha")
    assert ex.success
    path = svc.find_path("default", "Alpha", "Beta")
    assert path.success


def test_nonzero_exit(fake_graphify, monkeypatch):
    svc, _, _ = fake_graphify
    # After a successful build, force query to fail via patched _run
    from kg_security.graphify_service import SubprocessResult

    def fail_run(command, **kwargs):
        return SubprocessResult(
            command=list(command),
            returncode=2,
            stdout="",
            stderr="boom",
            duration_seconds=0.01,
            error=None,
        )

    # need graph file for require_graph_path
    out = svc.output_dir("default")
    (out / "graph.json").write_text(json.dumps(FIXTURE_GRAPH), encoding="utf-8")
    monkeypatch.setattr(svc, "_run", fail_run)
    r = svc.query_graph("default", "anything")
    assert not r.success
    assert "exit_2" in (r.error or "")


def test_timeout_behavior(fake_graphify, monkeypatch):
    svc, _, _ = fake_graphify
    from kg_security.graphify_service import SubprocessResult

    def timeout_run(command, **kwargs):
        return SubprocessResult(
            command=list(command),
            returncode=-1,
            stdout="",
            stderr="",
            duration_seconds=1.0,
            timed_out=True,
            error="Process timed out after 1s",
        )

    out = svc.output_dir("default")
    (out / "graph.json").write_text(json.dumps(FIXTURE_GRAPH), encoding="utf-8")
    monkeypatch.setattr(svc, "_run", timeout_run)
    r = svc.query_graph("default", "slow")
    assert not r.success
    assert "timeout" in (r.error or "").lower() or r.subprocess


def test_rag_context_fallback_when_missing(tmp_path):
    cfg = {
        "enabled": True,
        "executable": str(tmp_path / "missing"),
        "workspace_dir": tmp_path / "ws",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "x",
        "model": "m",
        "install_hint": "install me",
        "allow_non_local_endpoint": False,
        "max_rag_context_chars": 1000,
        "max_rag_context_nodes": 10,
        "max_rag_context_edges": 10,
        "max_query_length": 2000,
    }
    svc = GraphifyService(cfg)
    ctx, warn = svc.build_rag_context("hello")
    assert ctx is None
    assert warn


def test_secret_not_in_build_logs(fake_graphify, tmp_path, monkeypatch):
    svc, cfg, _ = fake_graphify
    doc = tmp_path / "d.txt"
    doc.write_text("data", encoding="utf-8")
    svc.persist_to_corpus(doc)
    # inject secret into stderr via spy wrapping real run
    from kg_security.graphify_service import SubprocessResult

    real = svc._run

    def leaky(command, **kwargs):
        res = real(command, **kwargs)
        return SubprocessResult(
            command=res.command,
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr + f"\nOPENAI_API_KEY={cfg['api_key']}\n",
            duration_seconds=res.duration_seconds,
            timed_out=res.timed_out,
            error=res.error,
        )

    monkeypatch.setattr(svc, "_run", leaky)
    result = svc.build_graph()
    assert cfg["api_key"] not in (result.logs or "")
    assert "REDACTED" in (result.logs or "") or cfg["api_key"] not in (result.logs or "")
