from fastapi import APIRouter, Depends

from backend.api.dependencies import get_component_registry
from backend.api.schemas.system import ModelsStatusResponse, SystemStatusResponse
from backend.services.component_registry import ComponentRegistry
from backend.services.status_service import build_models_status, build_system_status

router = APIRouter(tags=["system"])


@router.get("/api/system/status", response_model=SystemStatusResponse)
def system_status(
    registry: ComponentRegistry = Depends(get_component_registry),
) -> SystemStatusResponse:
    return SystemStatusResponse(**build_system_status(registry))


@router.get("/api/models/status", response_model=ModelsStatusResponse)
def models_status(
    registry: ComponentRegistry = Depends(get_component_registry),
) -> ModelsStatusResponse:
    return ModelsStatusResponse(**build_models_status(registry))
