from __future__ import annotations

from backend.services.component_registry import ComponentRegistry, get_registry


def get_component_registry() -> ComponentRegistry:
    return get_registry()
