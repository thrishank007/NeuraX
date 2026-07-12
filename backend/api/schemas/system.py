from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SystemStatusResponse(BaseModel):
    overall: str
    backend: str = "ok"
    vector_store: str = "unknown"
    lm_studio: str = "unknown"
    components: dict[str, bool] = Field(default_factory=dict)
    collection: dict[str, Any] = Field(default_factory=dict)
    supported_formats: list[str] = Field(default_factory=list)
    max_upload_mb: int = 100
    offline_mode: bool = True


class ModelsStatusResponse(BaseModel):
    lm_studio_reachable: bool
    models_loaded: int = 0
    model_ids: list[str] = Field(default_factory=list)
    current_model: Optional[str] = None
    supports_multimodal: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
