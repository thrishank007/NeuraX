from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = ""
    modality: Literal["text", "image", "voice", "multimodal"] = "text"
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    k: Optional[int] = Field(default=None, ge=1, le=50)


class SearchResultItem(BaseModel):
    id: str = ""
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""
    similarity_score: float = 0.0
    content_preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    text_similarity: Optional[float] = None
    image_similarity: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    query_type: str
    results: List[SearchResultItem]
    total_results: int
    similarity_threshold: float
    processing_time: float
    transcription: Optional[str] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_docs: int = Field(default=5, ge=1, le=10)


class CitationItem(BaseModel):
    citation_id: int
    source_document: str = ""
    source_type: str = ""
    content_snippet: str = ""
    confidence_score: float = 0.0
    file_path: str = ""
    page_number: Optional[int] = None
    expandable_link: str = ""
    timestamp: str = ""


class ChatResponse(BaseModel):
    query: str
    response: str
    citations: List[CitationItem]
    confidence: float = 0.0
    processing_time: float = 0.0
    sources: List[SearchResultItem] = Field(default_factory=list)
    model_used: str = ""
    lm_studio_available: bool = True


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5)
    comments: str = ""
