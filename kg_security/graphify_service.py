"""
Graphify document knowledge-graph integration for NeuraX.

Invokes the optional external Graphify CLI via subprocess (never shell=True).
Does not import Graphify package internals so NeuraX stays runnable without it.
Separate from the security NetworkX KnowledgeGraphManager.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from loguru import logger

from config import GRAPHIFY_CONFIG, PROCESSING_CONFIG, SECURITY_CONFIG
from kg_security.graphify_schema import (
    GraphParseError,
    GraphifyGraph,
    compact_graph_context,
    compute_graph_stats,
    filter_graph,
    load_graph_json,
)


SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[a-z0-9\-._~+/]+=*"),
    re.compile(r"sk-[a-zA-Z0-9]{10,}"),
)


def redact_secrets(text: str) -> str:
    if not text:
        return ""
    out = text
    for pat in SECRET_PATTERNS:
        out = pat.sub(lambda m: m.group(0).split("=")[0].split(":")[0] + "=[REDACTED]", out)
    # Common env-style values that might leak from stderr
    for key in ("OPENAI_API_KEY", "GRAPHIFY_OPENAI_API_KEY", "API_KEY"):
        out = re.sub(
            rf"(?i)({re.escape(key)}\s*[=:]\s*)(\S+)",
            r"\1[REDACTED]",
            out,
        )
    return out


def is_loopback_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")
    except Exception:
        return False


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    base = base.replace("\\", "_").replace("/", "_")
    base = re.sub(r"[^\w.\- ()\[\]]+", "_", base)
    if not base or base in {".", ".."}:
        base = f"file_{hashlib.sha256(name.encode('utf-8', errors='replace')).hexdigest()[:10]}"
    return base[:180]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding=encoding)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def resolve_under(root: Path, candidate: Path) -> Path:
    """Resolve candidate and ensure it stays under root (no traversal/symlink escape)."""
    root_resolved = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {candidate}") from exc
    return resolved


@dataclass
class SubprocessResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stdout"] = redact_secrets(d.get("stdout") or "")
        d["stderr"] = redact_secrets(d.get("stderr") or "")
        d["error"] = redact_secrets(d.get("error") or "") if d.get("error") else None
        return d


@dataclass
class GraphifyCliCapabilities:
    extract_backend: bool = False
    extract_model: bool = False
    extract_out: bool = False
    extract_max_concurrency: bool = False
    extract_api_timeout: bool = False
    extract_mode: bool = False
    extract_token_budget: bool = False
    extract_force: bool = False
    query_graph: bool = False
    query_budget: bool = False
    explain_graph: bool = False
    path_graph: bool = False
    has_update: bool = False
    raw_help: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphifyBuildResult:
    success: bool
    workspace_id: str
    mode: str  # build | update | rebuild
    message: str
    duration_seconds: float = 0.0
    artifacts: Dict[str, Optional[str]] = field(default_factory=dict)
    logs: str = ""
    error: Optional[str] = None
    subprocess: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphifyQueryResult:
    success: bool
    kind: str  # query | explain | path
    output: str
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    subprocess: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["output"] = redact_secrets(d.get("output") or "")
        if d.get("error"):
            d["error"] = redact_secrets(d["error"])
        return d


@dataclass
class GraphifyStatus:
    enabled: bool
    available: bool
    executable: Optional[str]
    version: Optional[str]
    install_hint: str
    workspace_id: str
    corpus_file_count: int
    last_build_time: Optional[str]
    build_running: bool
    artifacts_available: bool
    artifact_paths: Dict[str, Optional[str]]
    lm_studio_base_url: str
    model: str
    backend: str
    auto_update_after_ingestion: bool
    endpoint_allowed: bool
    endpoint_warning: Optional[str] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphifyService:
    """Optional Graphify CLI integration with managed corpus workspaces."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or GRAPHIFY_CONFIG)
        self.workspace_root = Path(self.config.get("workspace_dir", Path("data") / "graphify"))
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self._build_lock = threading.RLock()
        self._build_running = False
        self._last_error: Optional[str] = None
        self._capabilities: Optional[GraphifyCliCapabilities] = None
        self._capabilities_checked_at = 0.0
        self._version_cache: Optional[str] = None
        self._version_checked_at = 0.0

    # ------------------------------------------------------------------
    # Paths / workspace
    # ------------------------------------------------------------------
    def default_workspace_id(self) -> str:
        return str(self.config.get("default_workspace_id", "default"))

    def _sanitize_workspace_id(self, workspace_id: str) -> str:
        wid = (workspace_id or self.default_workspace_id()).strip()
        wid = re.sub(r"[^a-zA-Z0-9_\-]", "_", wid)[:64]
        if not wid or wid in {".", ".."}:
            wid = self.default_workspace_id()
        return wid

    def workspace_dir(self, workspace_id: str) -> Path:
        wid = self._sanitize_workspace_id(workspace_id)
        path = resolve_under(self.workspace_root, self.workspace_root / wid)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def corpus_dir(self, workspace_id: str) -> Path:
        d = self.workspace_dir(workspace_id) / "corpus"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def output_dir(self, workspace_id: str) -> Path:
        d = self.workspace_dir(workspace_id) / "graphify-out"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def manifest_path(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "manifest.json"

    def meta_path(self, workspace_id: str) -> Path:
        return self.workspace_dir(workspace_id) / "build_meta.json"

    def get_artifact_paths(self, workspace_id: str) -> Dict[str, Optional[str]]:
        out = self.output_dir(workspace_id)
        candidates = {
            "graph_json": out / "graph.json",
            "graph_html": out / "graph.html",
            "graph_report": out / "GRAPH_REPORT.md",
        }
        result: Dict[str, Optional[str]] = {}
        for key, path in candidates.items():
            try:
                resolved = resolve_under(self.workspace_root, path)
                result[key] = str(resolved) if resolved.exists() else None
            except ValueError:
                result[key] = None
        return result

    # ------------------------------------------------------------------
    # Executable detection
    # ------------------------------------------------------------------
    def _configured_executable(self) -> str:
        return str(self.config.get("executable") or os.getenv("GRAPHIFY_EXECUTABLE") or "graphify")

    def resolve_executable(self) -> Optional[str]:
        configured = self._configured_executable()
        # Absolute / relative path
        p = Path(configured)
        if p.is_file():
            return str(p.resolve())
        found = shutil.which(configured)
        if found:
            return found

        # Backend often runs without venv\Scripts on PATH. Probe next to the
        # active interpreter and the repo venv so graphifyy installs are found.
        name = configured
        candidates: List[Path] = []
        try:
            import sys

            scripts = Path(sys.executable).resolve().parent
            candidates.extend(
                [
                    scripts / name,
                    scripts / f"{name}.exe",
                    scripts / "Scripts" / name,
                    scripts / "Scripts" / f"{name}.exe",
                ]
            )
        except Exception:
            pass
        repo_root = Path(__file__).resolve().parents[1]
        candidates.extend(
            [
                repo_root / "venv" / "Scripts" / f"{name}.exe",
                repo_root / "venv" / "Scripts" / name,
                repo_root / "venv" / "bin" / name,
                repo_root / ".venv" / "Scripts" / f"{name}.exe",
                repo_root / ".venv" / "bin" / name,
            ]
        )
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return str(candidate.resolve())
            except OSError:
                continue
        return None

    def is_available(self) -> bool:
        return self.resolve_executable() is not None

    def get_version(self) -> Optional[str]:
        now = time.time()
        if self._version_cache is not None and (now - self._version_checked_at) < 60:
            return self._version_cache
        exe = self.resolve_executable()
        if not exe:
            self._version_cache = None
            self._version_checked_at = now
            return None
        result = self._run([exe, "--version"], timeout=15)
        version = None
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip().splitlines()[0][:120]
        elif result.stderr.strip():
            # some CLIs print version to stderr
            version = result.stderr.strip().splitlines()[0][:120]
        self._version_cache = version
        self._version_checked_at = now
        return version

    def detect_capabilities(self, force: bool = False) -> GraphifyCliCapabilities:
        now = time.time()
        if self._capabilities is not None and not force and (now - self._capabilities_checked_at) < 300:
            return self._capabilities
        caps = GraphifyCliCapabilities()
        exe = self.resolve_executable()
        if not exe:
            self._capabilities = caps
            self._capabilities_checked_at = now
            return caps

        help_chunks: List[str] = []
        for args in ([exe, "--help"], [exe, "extract", "--help"], [exe, "query", "--help"]):
            res = self._run(args, timeout=20)
            help_chunks.append(res.stdout or "")
            help_chunks.append(res.stderr or "")
        help_text = "\n".join(help_chunks)
        caps.raw_help = help_text[:5000]

        def has(flag: str) -> bool:
            return flag in help_text

        caps.extract_backend = has("--backend")
        caps.extract_model = has("--model")
        caps.extract_out = has("--out")
        caps.extract_max_concurrency = has("--max-concurrency")
        caps.extract_api_timeout = has("--api-timeout")
        caps.extract_mode = has("--mode")
        caps.extract_token_budget = has("--token-budget")
        caps.extract_force = has("--force") or "force" in help_text.lower()
        caps.query_graph = has("--graph")
        caps.query_budget = has("--budget")
        caps.explain_graph = has("--graph")
        caps.path_graph = has("--graph")
        caps.has_update = bool(re.search(r"\bupdate\b", help_text))

        self._capabilities = caps
        self._capabilities_checked_at = now
        return caps

    # ------------------------------------------------------------------
    # Environment / endpoint safety
    # ------------------------------------------------------------------
    def _endpoint_check(self) -> Tuple[bool, Optional[str]]:
        base_url = str(self.config.get("base_url") or "")
        allow_remote = bool(self.config.get("allow_non_local_endpoint", False))
        if not base_url:
            return False, "GRAPHIFY base_url is empty"
        if allow_remote or is_loopback_url(base_url):
            return True, None
        return (
            False,
            f"Non-loopback model endpoint rejected: {base_url}. "
            "Set GRAPHIFY_CONFIG['allow_non_local_endpoint']=True to override.",
        )

    def build_child_env(self) -> Dict[str, str]:
        """Controlled environment for Graphify child processes (no full env dump in logs)."""
        env = os.environ.copy()
        base_url = str(self.config.get("base_url") or "")
        api_key = str(self.config.get("api_key") or "lm-studio")
        model = str(self.config.get("model") or "")
        env["OPENAI_BASE_URL"] = base_url
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_MODEL"] = model
        env["GRAPHIFY_QUERY_LOG_DISABLE"] = "1"
        # Prefer OPENAI_* for openai backend; also set GRAPHIFY aliases if used by some versions
        env.setdefault("GRAPHIFY_OPENAI_BASE_URL", base_url)
        env.setdefault("GRAPHIFY_OPENAI_API_KEY", api_key)
        return env

    # ------------------------------------------------------------------
    # Subprocess runner
    # ------------------------------------------------------------------
    def _truncate(self, text: str, max_bytes: int) -> str:
        raw = text.encode("utf-8", errors="replace")
        if len(raw) <= max_bytes:
            return text
        return raw[:max_bytes].decode("utf-8", errors="replace") + "\n...[truncated]"

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[float] = None,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> SubprocessResult:
        cmd = [str(c) for c in command]
        timeout = float(timeout if timeout is not None else self.config.get("process_timeout_seconds", 1800))
        max_out = int(self.config.get("max_stdout_capture_bytes", 2 * 1024 * 1024))
        max_err = int(self.config.get("max_stderr_capture_bytes", 1 * 1024 * 1024))
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                shell=False,  # never use shell=True
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
            duration = time.perf_counter() - start
            return SubprocessResult(
                command=cmd,
                returncode=int(proc.returncode),
                stdout=self._truncate(proc.stdout or "", max_out),
                stderr=self._truncate(proc.stderr or "", max_err),
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            stdout = ""
            stderr = ""
            if exc.stdout:
                stdout = self._truncate(
                    exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", "replace"),
                    max_out,
                )
            if exc.stderr:
                stderr = self._truncate(
                    exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", "replace"),
                    max_err,
                )
            return SubprocessResult(
                command=cmd,
                returncode=-1,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=True,
                error=f"Process timed out after {timeout}s",
            )
        except FileNotFoundError:
            duration = time.perf_counter() - start
            return SubprocessResult(
                command=cmd,
                returncode=-1,
                stdout="",
                stderr="",
                duration_seconds=duration,
                error="Executable not found",
            )
        except Exception as exc:
            duration = time.perf_counter() - start
            return SubprocessResult(
                command=cmd,
                returncode=-1,
                stdout="",
                stderr="",
                duration_seconds=duration,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------
    def _load_manifest(self, workspace_id: str) -> Dict[str, Any]:
        path = self.manifest_path(workspace_id)
        if not path.exists():
            return {"version": 1, "files": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"version": 1, "files": {}}
            data.setdefault("version", 1)
            data.setdefault("files", {})
            return data
        except Exception as exc:
            logger.warning(f"Corrupt graphify manifest, resetting: {exc}")
            return {"version": 1, "files": {}}

    def _save_manifest(self, workspace_id: str, manifest: Dict[str, Any]) -> None:
        atomic_write_text(
            self.manifest_path(workspace_id),
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )

    def _load_meta(self, workspace_id: str) -> Dict[str, Any]:
        path = self.meta_path(workspace_id)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_meta(self, workspace_id: str, meta: Dict[str, Any]) -> None:
        atomic_write_text(self.meta_path(workspace_id), json.dumps(meta, indent=2, ensure_ascii=False))

    def _allowed_extensions(self) -> set[str]:
        formats = set(SECURITY_CONFIG.get("allowed_file_extensions") or [])
        formats.update(PROCESSING_CONFIG.get("supported_document_formats") or [])
        formats.update(PROCESSING_CONFIG.get("supported_image_formats") or [])
        formats.update(PROCESSING_CONFIG.get("supported_audio_formats") or [])
        # Graphify-friendly extras for text corpora
        formats.update({".md", ".rst", ".csv", ".json", ".py", ".ts", ".tsx", ".js", ".jsx"})
        return {f.lower() if f.startswith(".") else f".{f.lower()}" for f in formats}

    def _max_file_bytes(self) -> int:
        mb = int(
            SECURITY_CONFIG.get("max_upload_size_mb")
            or PROCESSING_CONFIG.get("max_file_size_mb")
            or 100
        )
        return mb * 1024 * 1024

    def persist_to_corpus(
        self,
        source_path: Path | str,
        *,
        workspace_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Copy a validated source file into the managed Graphify corpus.
        Returns manifest entry dict. Never follows user-controlled symlinks.
        """
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise ValueError(f"Source file not found: {src}")
        # Refuse symlink sources
        if src.is_symlink():
            raise ValueError("Symlink sources are not allowed in Graphify corpus")

        original = original_filename or src.name
        safe_name = sanitize_filename(original)
        ext = Path(safe_name).suffix.lower()
        if ext not in self._allowed_extensions():
            raise ValueError(f"Extension not allowed for corpus: {ext}")

        size = src.stat().st_size
        if size > self._max_file_bytes():
            raise ValueError(f"File too large for corpus: {size} bytes")

        file_hash = sha256_file(src)
        corpus = self.corpus_dir(wid)
        manifest = self._load_manifest(wid)
        files: Dict[str, Any] = manifest.get("files") or {}

        # Reuse existing entry with same hash
        for stored_name, entry in files.items():
            if entry.get("sha256") == file_hash:
                dest = resolve_under(corpus, corpus / stored_name)
                return {
                    "workspace_id": wid,
                    "stored_filename": stored_name,
                    "original_filename": entry.get("original_filename", original),
                    "sha256": file_hash,
                    "path": str(dest),
                    "reused": True,
                    "message": "Already present in Graphify corpus (same hash)",
                }

        # Deterministic unique name on collision with different content
        candidate = safe_name
        stem = Path(safe_name).stem
        suffix = Path(safe_name).suffix
        n = 1
        while candidate in files:
            existing = files[candidate]
            if existing.get("sha256") == file_hash:
                break
            candidate = f"{stem}_{n}{suffix}"
            n += 1
            if n > 1000:
                candidate = f"{stem}_{file_hash[:10]}{suffix}"
                break

        dest = resolve_under(corpus, corpus / candidate)
        if dest.exists() and dest.is_symlink():
            raise ValueError("Refusing to write through existing symlink")
        data = src.read_bytes()
        atomic_write_bytes(dest, data)

        entry = {
            "original_filename": original,
            "stored_filename": candidate,
            "sha256": file_hash,
            "size": size,
            "file_type": file_type or ext.lstrip("."),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "source_path_hint": src.name,
        }
        files[candidate] = entry
        manifest["files"] = files
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest(wid, manifest)
        return {
            "workspace_id": wid,
            "stored_filename": candidate,
            "original_filename": original,
            "sha256": file_hash,
            "path": str(dest),
            "reused": False,
            "message": "Copied to Graphify corpus; ready for graph indexing",
        }

    def list_corpus_files(self, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        manifest = self._load_manifest(wid)
        return list((manifest.get("files") or {}).values())

    def corpus_file_count(self, workspace_id: Optional[str] = None) -> int:
        return len(self.list_corpus_files(workspace_id))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self, workspace_id: Optional[str] = None) -> GraphifyStatus:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        enabled = bool(self.config.get("enabled", True))
        exe = self.resolve_executable() if enabled else None
        available = bool(exe)
        version = self.get_version() if available else None
        artifacts = self.get_artifact_paths(wid)
        artifacts_available = bool(artifacts.get("graph_json"))
        meta = self._load_meta(wid)
        endpoint_ok, endpoint_warn = self._endpoint_check()
        caps = self.detect_capabilities() if available else GraphifyCliCapabilities()

        return GraphifyStatus(
            enabled=enabled,
            available=available,
            executable=exe,
            version=version,
            install_hint=str(self.config.get("install_hint") or ""),
            workspace_id=wid,
            corpus_file_count=self.corpus_file_count(wid),
            last_build_time=meta.get("last_build_time"),
            build_running=self._build_running,
            artifacts_available=artifacts_available,
            artifact_paths=artifacts,
            lm_studio_base_url=str(self.config.get("base_url") or ""),
            model=str(self.config.get("model") or ""),
            backend=str(self.config.get("backend") or "openai"),
            auto_update_after_ingestion=bool(self.config.get("auto_update_after_ingestion", False)),
            endpoint_allowed=endpoint_ok,
            endpoint_warning=endpoint_warn,
            capabilities=caps.to_dict(),
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------
    def _build_extract_command(
        self,
        corpus: Path,
        out_dir: Path,
        *,
        force: bool = False,
        caps: Optional[GraphifyCliCapabilities] = None,
    ) -> List[str]:
        exe = self.resolve_executable()
        if not exe:
            raise FileNotFoundError("graphify executable not found")
        caps = caps or self.detect_capabilities()
        cmd: List[str] = [exe, "extract", str(corpus)]
        backend = str(self.config.get("backend") or "openai")
        model = str(self.config.get("model") or "")
        mode = self.config.get("mode")
        if caps.extract_backend:
            cmd.extend(["--backend", backend])
        if caps.extract_model and model:
            cmd.extend(["--model", model])
        if caps.extract_out:
            cmd.extend(["--out", str(out_dir)])
        if caps.extract_max_concurrency:
            cmd.extend(["--max-concurrency", str(int(self.config.get("max_concurrency", 1)))])
        if caps.extract_api_timeout:
            cmd.extend(["--api-timeout", str(float(self.config.get("api_timeout_seconds", 900)))])
        if caps.extract_mode and mode:
            cmd.extend(["--mode", str(mode)])
        if caps.extract_token_budget and self.config.get("token_budget"):
            cmd.extend(["--token-budget", str(int(self.config["token_budget"]))])
        if force and caps.extract_force:
            cmd.append("--force")
        return cmd

    def build_graph(self, workspace_id: Optional[str] = None, force: bool = False) -> GraphifyBuildResult:
        return self._run_build(workspace_id, mode="rebuild" if force else "build", force=force)

    def update_graph(self, workspace_id: Optional[str] = None) -> GraphifyBuildResult:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        artifacts = self.get_artifact_paths(wid)
        if not artifacts.get("graph_json"):
            return self._run_build(wid, mode="build", force=False)
        caps = self.detect_capabilities()
        if caps.has_update:
            return self._run_update(wid)
        # Fallback: full extract (Graphify extract is incremental by default for many versions)
        return self._run_build(wid, mode="update", force=False)

    def _run_update(self, workspace_id: str) -> GraphifyBuildResult:
        if not self.config.get("enabled", True):
            return GraphifyBuildResult(
                success=False,
                workspace_id=workspace_id,
                mode="update",
                message="Graphify integration is disabled in configuration",
                error="disabled",
            )
        if self._build_running or not self._build_lock.acquire(blocking=False):
            return GraphifyBuildResult(
                success=False,
                workspace_id=workspace_id,
                mode="update",
                message="A graph build is already running",
                error="build_locked",
            )
        self._build_running = True
        try:
            exe = self.resolve_executable()
            if not exe:
                return self._missing_result(workspace_id, "update")
            ok, warn = self._endpoint_check()
            if not ok:
                return GraphifyBuildResult(
                    success=False,
                    workspace_id=workspace_id,
                    mode="update",
                    message=warn or "Endpoint not allowed",
                    error="endpoint_rejected",
                )
            corpus = self.corpus_dir(workspace_id)
            if not any(corpus.iterdir()):
                return GraphifyBuildResult(
                    success=False,
                    workspace_id=workspace_id,
                    mode="update",
                    message="Corpus is empty. Upload documents first.",
                    error="empty_corpus",
                )
            cmd = [exe, "update", str(corpus)]
            env = self.build_child_env()
            # Point Graphify output into workspace via env when supported
            env["GRAPHIFY_OUT"] = str(self.output_dir(workspace_id))
            result = self._run(cmd, env=env, cwd=self.workspace_dir(workspace_id))
            return self._finalize_build(workspace_id, "update", result)
        finally:
            self._build_running = False
            try:
                self._build_lock.release()
            except RuntimeError:
                pass

    def _missing_result(self, workspace_id: str, mode: str) -> GraphifyBuildResult:
        hint = str(self.config.get("install_hint") or "Install graphifyy[openai] via uv tool or pipx.")
        self._last_error = "graphify_not_found"
        return GraphifyBuildResult(
            success=False,
            workspace_id=workspace_id,
            mode=mode,
            message=hint,
            error="graphify_not_found",
        )

    def _run_build(
        self,
        workspace_id: Optional[str],
        *,
        mode: str,
        force: bool,
    ) -> GraphifyBuildResult:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        if not self.config.get("enabled", True):
            return GraphifyBuildResult(
                success=False,
                workspace_id=wid,
                mode=mode,
                message="Graphify integration is disabled in configuration",
                error="disabled",
            )
        if self._build_running or not self._build_lock.acquire(blocking=False):
            return GraphifyBuildResult(
                success=False,
                workspace_id=wid,
                mode=mode,
                message="A graph build is already running",
                error="build_locked",
            )
        self._build_running = True
        try:
            exe = self.resolve_executable()
            if not exe:
                return self._missing_result(wid, mode)
            ok, warn = self._endpoint_check()
            if not ok:
                self._last_error = "endpoint_rejected"
                return GraphifyBuildResult(
                    success=False,
                    workspace_id=wid,
                    mode=mode,
                    message=warn or "Endpoint not allowed",
                    error="endpoint_rejected",
                )
            corpus = self.corpus_dir(wid)
            if not any(corpus.iterdir()):
                return GraphifyBuildResult(
                    success=False,
                    workspace_id=wid,
                    mode=mode,
                    message="Corpus is empty. Upload and process documents first.",
                    error="empty_corpus",
                )
            out_dir = self.output_dir(wid)
            if force:
                # Rebuild from scratch: clear prior artifacts only inside workspace
                for name in ("graph.json", "graph.html", "GRAPH_REPORT.md"):
                    p = out_dir / name
                    try:
                        if p.exists() and p.is_file() and not p.is_symlink():
                            resolve_under(self.workspace_root, p).unlink()
                    except Exception as exc:
                        logger.warning(f"Could not remove artifact {p}: {exc}")

            caps = self.detect_capabilities()
            cmd = self._build_extract_command(corpus, out_dir, force=force, caps=caps)
            env = self.build_child_env()
            env["GRAPHIFY_OUT"] = str(out_dir)
            result = self._run(cmd, env=env, cwd=self.workspace_dir(wid))
            return self._finalize_build(wid, mode, result)
        finally:
            self._build_running = False
            try:
                self._build_lock.release()
            except RuntimeError:
                pass

    def _finalize_build(
        self,
        workspace_id: str,
        mode: str,
        result: SubprocessResult,
    ) -> GraphifyBuildResult:
        logs = redact_secrets((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""))
        artifacts = self.get_artifact_paths(workspace_id)
        graph_path = artifacts.get("graph_json")
        if result.timed_out:
            self._last_error = "timeout"
            return GraphifyBuildResult(
                success=False,
                workspace_id=workspace_id,
                mode=mode,
                message=result.error or "Graphify build timed out",
                duration_seconds=result.duration_seconds,
                artifacts=artifacts,
                logs=logs,
                error="timeout",
                subprocess=result.to_dict(),
            )
        if result.returncode != 0:
            self._last_error = f"exit_{result.returncode}"
            return GraphifyBuildResult(
                success=False,
                workspace_id=workspace_id,
                mode=mode,
                message=redact_secrets(result.error or result.stderr or f"Graphify exited with code {result.returncode}"),
                duration_seconds=result.duration_seconds,
                artifacts=artifacts,
                logs=logs,
                error=f"exit_{result.returncode}",
                subprocess=result.to_dict(),
            )
        if not graph_path:
            self._last_error = "missing_graph_json"
            return GraphifyBuildResult(
                success=False,
                workspace_id=workspace_id,
                mode=mode,
                message="Graphify finished but graph.json was not produced",
                duration_seconds=result.duration_seconds,
                artifacts=artifacts,
                logs=logs,
                error="missing_graph_json",
                subprocess=result.to_dict(),
            )
        # Validate graph.json
        try:
            load_graph_json(
                Path(graph_path),
                max_bytes=int(self.config.get("max_graph_json_bytes", 50 * 1024 * 1024)),
            )
        except GraphParseError as exc:
            self._last_error = "invalid_graph_json"
            return GraphifyBuildResult(
                success=False,
                workspace_id=workspace_id,
                mode=mode,
                message=f"Graphify output is invalid: {exc}",
                duration_seconds=result.duration_seconds,
                artifacts=artifacts,
                logs=logs,
                error="invalid_graph_json",
                subprocess=result.to_dict(),
            )

        # Generate interactive HTML visualization via `graphify tree`
        out_dir = self.output_dir(workspace_id)
        html_out = out_dir / "graph.html"
        try:
            exe = self.resolve_executable()
            if exe:
                tree_cmd = [
                    exe, "tree",
                    "--graph", str(out_dir / "graph.json"),
                    "--output", str(html_out),
                ]
                tree_result = self._run(tree_cmd, cwd=self.workspace_dir(workspace_id), timeout=120)
                if tree_result.returncode != 0:
                    logger.warning(f"graphify tree failed (non-fatal): {tree_result.stderr[:500]}")
        except Exception as exc:
            logger.warning(f"graphify tree error (non-fatal): {exc}")

        artifacts = self.get_artifact_paths(workspace_id)  # refresh after tree run
        now = datetime.now(timezone.utc).isoformat()
        meta = self._load_meta(workspace_id)
        meta["last_build_time"] = now
        meta["last_build_mode"] = mode
        meta["last_build_success"] = True
        self._save_meta(workspace_id, meta)
        self._last_error = None
        return GraphifyBuildResult(
            success=True,
            workspace_id=workspace_id,
            mode=mode,
            message="Knowledge graph built successfully",
            duration_seconds=result.duration_seconds,
            artifacts=artifacts,
            logs=logs,
            subprocess=result.to_dict(),
        )

    # ------------------------------------------------------------------
    # Query / explain / path
    # ------------------------------------------------------------------
    def _validate_query_text(self, text: str, *, field_name: str = "question") -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError(f"{field_name} must not be empty")
        max_len = int(self.config.get("max_query_length", 2000))
        if len(cleaned) > max_len:
            raise ValueError(f"{field_name} exceeds maximum length ({max_len})")
        return cleaned

    def _require_graph_path(self, workspace_id: str) -> Path:
        artifacts = self.get_artifact_paths(workspace_id)
        path = artifacts.get("graph_json")
        if not path:
            raise FileNotFoundError(
                "No graph.json found. Build the knowledge graph first from the Knowledge Graph page."
            )
        return Path(path)

    def _parse_query_side_info(self, output: str, graph: Optional[GraphifyGraph]) -> Tuple[List[Dict[str, Any]], List[str]]:
        nodes: List[Dict[str, Any]] = []
        sources: List[str] = []
        if not graph:
            return nodes, sources
        # Best-effort: find node labels mentioned in output
        lower_out = output.lower()
        for n in graph.nodes:
            label = n.label or n.id
            if label and label.lower() in lower_out:
                nodes.append(n.to_dict())
                if n.source_file and n.source_file not in sources:
                    sources.append(n.source_file)
            if len(nodes) >= 30:
                break
        return nodes, sources

    def query_graph(
        self,
        workspace_id: Optional[str],
        question: str,
        budget: Optional[int] = None,
    ) -> GraphifyQueryResult:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        try:
            question = self._validate_query_text(question, field_name="question")
        except ValueError as exc:
            return GraphifyQueryResult(success=False, kind="query", output="", error=str(exc))

        exe = self.resolve_executable()
        if not exe:
            return GraphifyQueryResult(
                success=False,
                kind="query",
                output="",
                error=str(self.config.get("install_hint") or "Graphify not installed"),
            )
        try:
            graph_path = self._require_graph_path(wid)
        except FileNotFoundError as exc:
            return GraphifyQueryResult(success=False, kind="query", output="", error=str(exc))

        caps = self.detect_capabilities()
        cmd: List[str] = [exe, "query", question]
        if caps.query_graph:
            cmd.extend(["--graph", str(graph_path)])
        if budget is not None and caps.query_budget:
            cmd.extend(["--budget", str(int(budget))])

        env = self.build_child_env()
        result = self._run(cmd, env=env, timeout=min(300, float(self.config.get("process_timeout_seconds", 1800))))
        return self._finalize_query("query", result, graph_path)

    def explain_node(self, workspace_id: Optional[str], node_name: str) -> GraphifyQueryResult:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        try:
            node_name = self._validate_query_text(node_name, field_name="node_name")
        except ValueError as exc:
            return GraphifyQueryResult(success=False, kind="explain", output="", error=str(exc))

        exe = self.resolve_executable()
        if not exe:
            return GraphifyQueryResult(
                success=False,
                kind="explain",
                output="",
                error=str(self.config.get("install_hint") or "Graphify not installed"),
            )
        try:
            graph_path = self._require_graph_path(wid)
        except FileNotFoundError as exc:
            return GraphifyQueryResult(success=False, kind="explain", output="", error=str(exc))

        caps = self.detect_capabilities()
        cmd: List[str] = [exe, "explain", node_name]
        if caps.explain_graph:
            cmd.extend(["--graph", str(graph_path)])
        env = self.build_child_env()
        result = self._run(cmd, env=env, timeout=min(300, float(self.config.get("process_timeout_seconds", 1800))))
        return self._finalize_query("explain", result, graph_path)

    def find_path(
        self,
        workspace_id: Optional[str],
        source: str,
        target: str,
    ) -> GraphifyQueryResult:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        try:
            source = self._validate_query_text(source, field_name="source")
            target = self._validate_query_text(target, field_name="target")
        except ValueError as exc:
            return GraphifyQueryResult(success=False, kind="path", output="", error=str(exc))

        exe = self.resolve_executable()
        if not exe:
            return GraphifyQueryResult(
                success=False,
                kind="path",
                output="",
                error=str(self.config.get("install_hint") or "Graphify not installed"),
            )
        try:
            graph_path = self._require_graph_path(wid)
        except FileNotFoundError as exc:
            return GraphifyQueryResult(success=False, kind="path", output="", error=str(exc))

        caps = self.detect_capabilities()
        cmd: List[str] = [exe, "path", source, target]
        if caps.path_graph:
            cmd.extend(["--graph", str(graph_path)])
        env = self.build_child_env()
        result = self._run(cmd, env=env, timeout=min(300, float(self.config.get("process_timeout_seconds", 1800))))
        return self._finalize_query("path", result, graph_path)

    def _finalize_query(
        self,
        kind: str,
        result: SubprocessResult,
        graph_path: Path,
    ) -> GraphifyQueryResult:
        max_chars = int(self.config.get("max_query_output_chars", 12000))
        output = redact_secrets((result.stdout or "").strip() or (result.stderr or "").strip())
        if len(output) > max_chars:
            output = output[: max_chars - 20] + "\n...[truncated]"

        graph: Optional[GraphifyGraph] = None
        try:
            graph = load_graph_json(
                graph_path,
                max_bytes=int(self.config.get("max_graph_json_bytes", 50 * 1024 * 1024)),
            )
        except GraphParseError:
            graph = None

        nodes, sources = self._parse_query_side_info(output, graph)

        if result.timed_out:
            return GraphifyQueryResult(
                success=False,
                kind=kind,
                output=output,
                nodes=nodes,
                sources=sources,
                error=result.error or "timeout",
                duration_seconds=result.duration_seconds,
                subprocess=result.to_dict(),
            )
        if result.returncode != 0:
            return GraphifyQueryResult(
                success=False,
                kind=kind,
                output=output,
                nodes=nodes,
                sources=sources,
                error=redact_secrets(result.error or f"exit_{result.returncode}"),
                duration_seconds=result.duration_seconds,
                subprocess=result.to_dict(),
            )
        if not output:
            return GraphifyQueryResult(
                success=False,
                kind=kind,
                output="",
                error="Graphify returned empty output",
                duration_seconds=result.duration_seconds,
                subprocess=result.to_dict(),
            )
        return GraphifyQueryResult(
            success=True,
            kind=kind,
            output=output,
            nodes=nodes,
            sources=sources,
            duration_seconds=result.duration_seconds,
            subprocess=result.to_dict(),
        )

    # ------------------------------------------------------------------
    # Load / stats / filters / RAG context
    # ------------------------------------------------------------------
    def load_graph(self, workspace_id: Optional[str] = None) -> GraphifyGraph:
        wid = self._sanitize_workspace_id(workspace_id or self.default_workspace_id())
        path = self._require_graph_path(wid)
        return load_graph_json(
            path,
            max_bytes=int(self.config.get("max_graph_json_bytes", 50 * 1024 * 1024)),
        )

    def get_graph_stats(self, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        graph = self.load_graph(workspace_id)
        return compute_graph_stats(graph)

    def get_filtered_graph(
        self,
        workspace_id: Optional[str] = None,
        *,
        node_type: Optional[str] = None,
        community: Optional[str] = None,
        source_file: Optional[str] = None,
        confidence: Optional[str] = None,
        min_confidence_score: Optional[float] = None,
        limit_nodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        graph = self.load_graph(workspace_id)
        filtered = filter_graph(
            graph,
            node_type=node_type,
            community=community,
            source_file=source_file,
            confidence=confidence,
            min_confidence_score=min_confidence_score,
        )
        limit = limit_nodes or int(self.config.get("max_visualization_nodes", 1000))
        if len(filtered.nodes) > limit:
            keep = {n.id for n in filtered.nodes[:limit]}
            filtered = GraphifyGraph(
                nodes=filtered.nodes[:limit],
                edges=[e for e in filtered.edges if e.source in keep and e.target in keep],
                directed=filtered.directed,
                multigraph=filtered.multigraph,
                raw_meta=filtered.raw_meta,
            )
        data = filtered.to_dict()
        data["stats"] = compute_graph_stats(filtered)
        return data

    def node_labels(self, workspace_id: Optional[str] = None, limit: int = 500) -> List[str]:
        try:
            graph = self.load_graph(workspace_id)
        except Exception:
            return []
        labels = []
        seen = set()
        for n in graph.nodes:
            label = n.label or n.id
            if label not in seen:
                seen.add(label)
                labels.append(label)
            if len(labels) >= limit:
                break
        return labels

    def build_rag_context(self, question: str, workspace_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (context_text, warning). Warning is set on soft failures.
        Never raises for missing Graphify.
        """
        if not self.config.get("enabled", True):
            return None, None
        if not self.is_available():
            return None, "Graphify not installed; continuing without graph context."
        try:
            result = self.query_graph(workspace_id, question)
            if not result.success:
                return None, f"Graph context unavailable: {result.error or 'query failed'}"
            # Prefer compact structured context if graph loads
            try:
                graph = self.load_graph(workspace_id)
                # Prefer nodes recovered from query output; fall back to full capped graph
                if result.nodes:
                    ids = {n.get("id") for n in result.nodes if n.get("id")}
                    subset = GraphifyGraph(
                        nodes=[n for n in graph.nodes if n.id in ids] or graph.nodes[:25],
                        edges=[
                            e
                            for e in graph.edges
                            if e.source in ids or e.target in ids
                        ][:40]
                        or graph.edges[:40],
                        directed=graph.directed,
                        multigraph=graph.multigraph,
                    )
                else:
                    subset = graph
                ctx = compact_graph_context(
                    subset,
                    max_chars=int(self.config.get("max_rag_context_chars", 4000)),
                    max_nodes=int(self.config.get("max_rag_context_nodes", 25)),
                    max_edges=int(self.config.get("max_rag_context_edges", 40)),
                    question=question,
                )
                # Append raw Graphify answer truncated
                answer = result.output[:1500]
                combined = ctx + "\n\n### Graphify Query Answer\n" + answer
                max_chars = int(self.config.get("max_rag_context_chars", 4000)) + 1500
                if len(combined) > max_chars:
                    combined = combined[: max_chars - 20] + "\n...[truncated]"
                return combined, None
            except GraphParseError as exc:
                # Still return text answer
                text = "### Knowledge Graph Context (Graphify)\n" + result.output
                max_chars = int(self.config.get("max_rag_context_chars", 4000))
                if len(text) > max_chars:
                    text = text[: max_chars - 20] + "\n...[truncated]"
                return text, f"Partial graph context (parse warning: {exc})"
        except Exception as exc:
            logger.warning(f"Graph RAG context failed: {exc}")
            return None, f"Graph context skipped: {exc}"

    def read_artifact_text(self, workspace_id: str, kind: str) -> Tuple[Optional[str], Optional[str]]:
        """Read graph.html / graph.json / GRAPH_REPORT.md as text when safe."""
        paths = self.get_artifact_paths(workspace_id)
        key_map = {
            "html": "graph_html",
            "json": "graph_json",
            "report": "graph_report",
            "graph.html": "graph_html",
            "graph.json": "graph_json",
            "GRAPH_REPORT.md": "graph_report",
        }
        key = key_map.get(kind, kind)
        path = paths.get(key)
        if not path:
            return None, f"Artifact not available: {kind}"
        p = Path(path)
        try:
            resolve_under(self.workspace_root, p)
            if p.stat().st_size > int(self.config.get("max_graph_json_bytes", 50 * 1024 * 1024)):
                return None, "Artifact too large to inline"
            return p.read_text(encoding="utf-8", errors="replace"), None
        except Exception as exc:
            return None, str(exc)

    def escape_html_label(self, value: str) -> str:
        return html.escape(value or "", quote=True)
