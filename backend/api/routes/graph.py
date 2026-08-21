"""Security KG export + Graphify document knowledge graph API."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.dependencies import get_component_registry
from backend.api.errors import APIError
from backend.services.component_registry import ComponentRegistry

router = APIRouter(tags=["graph"])


class GraphifyQueryBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    budget: Optional[int] = Field(default=None, ge=1, le=50000)
    workspace_id: str = "default"


class GraphifyExplainBody(BaseModel):
    node_name: str = Field(..., min_length=1, max_length=500)
    workspace_id: str = "default"


class GraphifyPathBody(BaseModel):
    source: str = Field(..., min_length=1, max_length=500)
    target: str = Field(..., min_length=1, max_length=500)
    workspace_id: str = "default"


class GraphifyBuildBody(BaseModel):
    workspace_id: str = "default"
    force: bool = False


def _svc(registry: ComponentRegistry):
    svc = registry.ensure_graphify()
    if svc is None:
        raise APIError(
            "not_found",
            "Graphify service could not be initialized",
            503,
        )
    return svc


# ---------------------------------------------------------------------------
# Existing security knowledge graph (NetworkX KnowledgeGraphManager)
# ---------------------------------------------------------------------------
@router.get("/api/graph")
def get_graph(
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict:
    kg = registry.ensure_kg()
    if kg is None:
        raise APIError(
            "not_found",
            "Knowledge graph is not available",
            404,
        )
    try:
        data = kg.export_viz_data()
        stats = kg.get_graph_stats() if hasattr(kg, "get_graph_stats") else {}
        return {"graph": data, "stats": stats, "kind": "security"}
    except Exception as exc:
        raise APIError("processing_error", "Failed to export graph", 500) from exc


# ---------------------------------------------------------------------------
# Graphify document knowledge graph
# ---------------------------------------------------------------------------
@router.get("/api/graphify/status")
def graphify_status(
    workspace_id: str = Query(default="default"),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    return svc.get_status(workspace_id).to_dict()


@router.post("/api/graphify/build")
def graphify_build(
    body: GraphifyBuildBody,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    result = svc.build_graph(body.workspace_id, force=body.force)
    return result.to_dict()


@router.post("/api/graphify/update")
def graphify_update(
    body: GraphifyBuildBody,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    result = svc.update_graph(body.workspace_id)
    return result.to_dict()


@router.post("/api/graphify/rebuild")
def graphify_rebuild(
    body: GraphifyBuildBody,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    result = svc.build_graph(body.workspace_id, force=True)
    return result.to_dict()


@router.get("/api/graphify/stats")
def graphify_stats(
    workspace_id: str = Query(default="default"),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    try:
        return {"stats": svc.get_graph_stats(workspace_id)}
    except FileNotFoundError as exc:
        raise APIError("not_found", str(exc), 404) from exc
    except Exception as exc:
        raise APIError("processing_error", f"Failed to load graph stats: {exc}", 400) from exc


@router.get("/api/graphify/data")
def graphify_data(
    workspace_id: str = Query(default="default"),
    node_type: Optional[str] = None,
    community: Optional[str] = None,
    source_file: Optional[str] = None,
    confidence: Optional[str] = None,
    min_confidence_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    try:
        return svc.get_filtered_graph(
            workspace_id,
            node_type=node_type,
            community=community,
            source_file=source_file,
            confidence=confidence,
            min_confidence_score=min_confidence_score,
        )
    except FileNotFoundError as exc:
        raise APIError("not_found", str(exc), 404) from exc
    except Exception as exc:
        raise APIError("processing_error", f"Failed to load graph: {exc}", 400) from exc


@router.get("/api/graphify/nodes")
def graphify_nodes(
    workspace_id: str = Query(default="default"),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    return {"labels": svc.node_labels(workspace_id)}


@router.post("/api/graphify/query")
def graphify_query(
    body: GraphifyQueryBody,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    result = svc.query_graph(body.workspace_id, body.question, budget=body.budget)
    return result.to_dict()


@router.post("/api/graphify/explain")
def graphify_explain(
    body: GraphifyExplainBody,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    result = svc.explain_node(body.workspace_id, body.node_name)
    return result.to_dict()


@router.post("/api/graphify/path")
def graphify_path(
    body: GraphifyPathBody,
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    result = svc.find_path(body.workspace_id, body.source, body.target)
    return result.to_dict()


@router.get("/api/graphify/corpus")
def graphify_corpus(
    workspace_id: str = Query(default="default"),
    registry: ComponentRegistry = Depends(get_component_registry),
) -> dict[str, Any]:
    svc = _svc(registry)
    files = svc.list_corpus_files(workspace_id)
    return {"files": files, "count": len(files)}


@router.get("/api/graphify/artifacts/{kind}")
def graphify_artifact(
    kind: str,
    workspace_id: str = Query(default="default"),
    registry: ComponentRegistry = Depends(get_component_registry),
):
    """Download graph.html | graph.json | report (GRAPH_REPORT.md)."""
    svc = _svc(registry)
    kind_map = {
        "html": "graph_html",
        "json": "graph_json",
        "report": "graph_report",
        "graph.html": "graph_html",
        "graph.json": "graph_json",
        "GRAPH_REPORT.md": "graph_report",
    }
    if kind not in kind_map:
        raise APIError("validation_error", f"Unknown artifact kind: {kind}", 400)
    paths = svc.get_artifact_paths(workspace_id)
    path = paths.get(kind_map[kind])
    if not path or not Path(path).exists():
        raise APIError("not_found", f"Artifact not available: {kind}", 404)
    media = {
        "graph_html": "text/html",
        "graph_json": "application/json",
        "graph_report": "text/markdown",
    }[kind_map[kind]]
    filename = Path(path).name
    # HTML must be served inline so it renders in an <iframe>.
    # Other artifacts are attachments (download).
    if kind_map[kind] == "graph_html":
        return FileResponse(path, media_type=media, filename=filename, content_disposition_type="inline")
    return FileResponse(path, media_type=media, filename=filename)
