"""NVIDIA NIM text embedding client."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import requests


class NvidiaNimEmbeddingProvider:
    """Generate normalized text embeddings with NVIDIA's hosted NIM API."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.api_url = str(config.get("api_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", ""))
        self.model = str(config.get("model", ""))
        self.timeout = int(config.get("timeout", 60))

        if not self.api_url:
            raise ValueError("NEURAX_NIM_EMBEDDING_API_URL is required when NVIDIA NIM embeddings are enabled")
        if not self.api_key:
            raise ValueError("NEURAX_NIM_API_KEY is required when NVIDIA NIM embeddings are enabled")
        if not self.model:
            raise ValueError("NEURAX_NIM_EMBEDDING_MODEL is required when NVIDIA NIM embeddings are enabled")

    def embed_documents(self, texts: Iterable[str]) -> np.ndarray:
        """Embed document chunks using NVIDIA's passage retrieval mode."""
        return self._embed(texts, input_type="passage")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query using NVIDIA's query retrieval mode."""
        return self._embed([query], input_type="query")

    def _embed(self, texts: Iterable[str], *, input_type: str) -> np.ndarray:
        cleaned_texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
        if not cleaned_texts:
            raise ValueError("NVIDIA NIM embedding input must contain at least one non-empty string")

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": cleaned_texts,
                "input_type": input_type,
                "encoding_format": "float",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        items = sorted(payload.get("data", []), key=lambda item: item.get("index", -1))
        if len(items) != len(cleaned_texts):
            raise RuntimeError("NVIDIA NIM returned an unexpected number of embeddings")

        embeddings = np.asarray([item.get("embedding") for item in items], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] == 0:
            raise RuntimeError("NVIDIA NIM returned invalid embedding vectors")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("NVIDIA NIM returned a zero-norm embedding")
        return embeddings / norms
