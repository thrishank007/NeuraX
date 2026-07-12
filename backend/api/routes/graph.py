from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_component_registry
from backend.api.errors import APIError
from backend.services.component_registry import ComponentRegistry

router = APIRouter(tags=["graph"])


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
        return {"graph": data, "stats": stats}
    except Exception as exc:
        raise APIError("processing_error", "Failed to export graph", 500) from exc
