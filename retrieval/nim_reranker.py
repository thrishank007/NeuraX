"""NVIDIA NIM cross-encoder reranker.

Reorders fused retrieval candidates by query-passage relevance using the
hosted NIM reranking API. Degrades gracefully: on any failure the caller
keeps the original (RRF) order, so reranking can never take retrieval
down.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import requests
from loguru import logger


class NimReranker:
    """Rerank passages against a query via NVIDIA's reranking endpoint."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.api_url = str(config.get("api_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", ""))
        self.model = str(config.get("model", "nvidia/nv-rerankqa-mistral-4b-v3"))
        self.timeout = int(config.get("timeout", 30))
        self.max_passages = int(config.get("max_passages", 20))

        if not self.api_url:
            raise ValueError("NEURAX_NIM_RERANK_API_URL is required when reranking is enabled")
        if not self.api_key:
            raise ValueError("NEURAX_NIM_API_KEY is required when reranking is enabled")

    def rerank(self, query: str, passages: Sequence[str]) -> List[int]:
        """Return passage indices ordered most→least relevant.

        Raises on transport/API errors so callers can decide; returns the
        identity order never — degradation is the caller's policy.
        """
        texts = [p for p in passages if p and p.strip()]
        if len(texts) <= 1:
            return list(range(len(passages)))

        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": {"text": query},
                "passages": [{"text": t} for t in texts[: self.max_passages]],
                "truncate": "END",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        rankings = response.json().get("rankings") or []
        # Map compacted (non-empty) positions back to original indices.
        non_empty = [i for i, p in enumerate(passages) if p and p.strip()]
        ordered = [non_empty[r["index"]] for r in rankings if 0 <= r.get("index", -1) < len(non_empty)]
        seen = set(ordered)
        ordered.extend(i for i in range(len(passages)) if i not in seen)
        return ordered

    def rerank_dicts(self, query: str, results: List[dict], top_k: int) -> List[dict]:
        """Rerank a list of result dicts (with 'content'), return top_k.

        Non-fatal: on API failure logs and returns the input order.
        """
        if not results:
            return results
        passages = [str(r.get("content") or r.get("document") or "") for r in results[: self.max_passages]]
        try:
            order = self.rerank(query, passages)
        except Exception as exc:
            logger.warning(f"NIM rerank failed, keeping RRF order: {exc}")
            return results[:top_k]
        reranked = [results[i] for i in order[:top_k]]
        for rank, r in enumerate(reranked):
            r["rerank_rank"] = rank + 1
        return reranked
