from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    id: str
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""
    embedding_type: str = ""
    timestamp: str = ""
    content_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    total: int = 0
    processed: int = 0
    logs: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    cancel_requested: bool = False
