"""Lazy component registry for domain managers used by the API layer."""
from __future__ import annotations

import threading
from typing import Any, Optional

from loguru import logger

from config import CHROMA_CONFIG, CLIP_IMAGE_CONFIG, LLM_CONFIG, NIM_EMBEDDING_CONFIG, SEARCH_CONFIG


class ComponentRegistry:
    """Process-wide lazy holders for NeuraX domain components."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.ingestion_manager = None
        self.embedding_manager = None
        self.vector_store = None
        self.query_processor = None
        self.stt_processor = None
        self.llm_generator = None
        self.citation_generator = None
        self.feedback_system = None
        self.kg_manager = None  # existing NetworkX security graph
        self.graphify_service = None  # optional Graphify document graph
        self.image_vector_store = None  # dedicated 512-d CLIP collection
        self._init_errors: list[str] = []

    def component_flags(self) -> dict[str, bool]:
        return {
            "ingestion_manager": self.ingestion_manager is not None,
            "embedding_manager": self.embedding_manager is not None,
            "vector_store": self.vector_store is not None,
            "query_processor": self.query_processor is not None,
            "stt_processor": self.stt_processor is not None,
            "llm_generator": self.llm_generator is not None,
            "citation_generator": self.citation_generator is not None,
            "feedback_system": self.feedback_system is not None,
            "kg_manager": self.kg_manager is not None,
            "graphify_service": self.graphify_service is not None,
        }

    def ensure_ingestion(self):
        with self._lock:
            if self.ingestion_manager is None:
                from ingestion.ingestion_manager import IngestionManager

                self.ingestion_manager = IngestionManager()
            return self.ingestion_manager

    def ensure_vector_stack(self):
        """Initialize embedding manager, vector store, and query processor."""
        with self._lock:
            if self.embedding_manager is None:
                from indexing.embedding_manager import EmbeddingManager

                logger.info("Initializing embedding manager...")
                self.embedding_manager = EmbeddingManager()

            if self.vector_store is None:
                from indexing.vector_store import VectorStore

                logger.info("Initializing vector store...")
                collection_name = (
                    NIM_EMBEDDING_CONFIG["collection_name"]
                    if NIM_EMBEDDING_CONFIG["enabled"]
                    else CHROMA_CONFIG["collection_name"]
                )
                embedding_dimension = (
                    NIM_EMBEDDING_CONFIG["dimension"]
                    if NIM_EMBEDDING_CONFIG["enabled"]
                    else 384
                )
                self.vector_store = VectorStore(
                    persist_directory=CHROMA_CONFIG["persist_directory"],
                    collection_name=collection_name,
                    embedding_dimension=embedding_dimension,
                )

            if self.query_processor is None and self.embedding_manager and self.vector_store:
                from retrieval.query_processor import QueryProcessor

                # Try to get LLM generator for query rewriting
                llm_gen = None
                try:
                    llm_gen = self.ensure_generation()[0]
                except Exception as exc:
                    logger.debug(f"LLM generator unavailable for query rewriting: {exc}")

                logger.info("Initializing query processor...")
                self.query_processor = QueryProcessor(
                    self.embedding_manager,
                    self.vector_store,
                    SEARCH_CONFIG,
                    llm_generator=llm_gen,
                )
            return self.embedding_manager, self.vector_store, self.query_processor

    def ensure_image_vector_stack(self):
        """Lazy init for the dedicated CLIP image vector store (512-d)."""
        with self._lock:
            if self.image_vector_store is None:
                from indexing.vector_store import VectorStore

                logger.info("Initializing CLIP image vector store...")
                self.image_vector_store = VectorStore(
                    persist_directory=CHROMA_CONFIG["persist_directory"],
                    collection_name=CLIP_IMAGE_CONFIG["collection_name"],
                    embedding_dimension=CLIP_IMAGE_CONFIG["dimension"],
                )
            return self.image_vector_store

    def ensure_stt(self):
        with self._lock:
            if self.stt_processor is None:
                from retrieval.speech_to_text_processor import SpeechToTextProcessor

                self.stt_processor = SpeechToTextProcessor()
            return self.stt_processor

    def ensure_generation(self):
        with self._lock:
            if self.llm_generator is None:
                from generation.llm_factory import create_llm_generator

                self.llm_generator = create_llm_generator(LLM_CONFIG)
            if self.citation_generator is None:
                from generation.citation_generator import CitationGenerator

                self.citation_generator = CitationGenerator()
            return self.llm_generator, self.citation_generator

    def ensure_feedback(self):
        with self._lock:
            if self.feedback_system is None:
                from feedback.feedback_system import FeedbackSystem

                self.feedback_system = FeedbackSystem()
            return self.feedback_system

    def ensure_kg(self) -> Optional[Any]:
        """Existing security-focused NetworkX knowledge graph manager."""
        with self._lock:
            if self.kg_manager is None:
                try:
                    from kg_security.knowledge_graph_manager import KnowledgeGraphManager

                    self.kg_manager = KnowledgeGraphManager()
                except Exception as exc:
                    logger.warning(f"Knowledge graph unavailable: {exc}")
                    self._init_errors.append(str(exc))
                    return None
            return self.kg_manager

    def ensure_graphify(self) -> Optional[Any]:
        """Optional Graphify document graph service (CLI-backed, never hard-fails)."""
        with self._lock:
            if self.graphify_service is None:
                try:
                    from kg_security.graphify_service import GraphifyService

                    self.graphify_service = GraphifyService()
                except Exception as exc:
                    logger.warning(f"Graphify service unavailable: {exc}")
                    self._init_errors.append(f"graphify: {exc}")
                    return None
            return self.graphify_service

    def shutdown(self) -> None:
        with self._lock:
            if self.vector_store is not None:
                try:
                    if hasattr(self.vector_store, "memory_manager"):
                        self.vector_store.memory_manager.stop_monitoring()
                except Exception as exc:
                    logger.warning(f"Vector store shutdown warning: {exc}")
            self.embedding_manager = None
            self.vector_store = None
            self.image_vector_store = None
            self.query_processor = None
            self.stt_processor = None
            self.llm_generator = None
            self.citation_generator = None
            self.feedback_system = None
            self.kg_manager = None
            self.graphify_service = None
            self.ingestion_manager = None


_registry: Optional[ComponentRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ComponentRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = ComponentRegistry()
        return _registry


def reset_registry() -> None:
    """Test helper to clear process-wide registry."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.shutdown()
        _registry = None
