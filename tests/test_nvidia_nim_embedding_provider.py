from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from indexing.nvidia_nim_embedding_provider import NvidiaNimEmbeddingProvider
from indexing.vector_store import VectorStore
from retrieval.query_processor import QueryProcessor


def _provider() -> NvidiaNimEmbeddingProvider:
    return NvidiaNimEmbeddingProvider(
        {
            "api_url": "https://integrate.api.nvidia.com/v1/embeddings",
            "api_key": "test-key",
            "model": "nvidia/nemotron-3-embed-1b",
            "timeout": 30,
        }
    )


def _response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ]
    }
    response.raise_for_status = MagicMock()
    return response


def test_embed_documents_uses_passage_mode_and_normalizes_vectors():
    provider = _provider()

    with patch(
        "indexing.nvidia_nim_embedding_provider.requests.post",
        return_value=_response([[3.0, 4.0], [0.0, 2.0]]),
    ) as post:
        embeddings = provider.embed_documents(["first chunk", "second chunk"])

    assert np.allclose(embeddings, [[0.6, 0.8], [0.0, 1.0]])
    assert post.call_args.kwargs["json"]["input_type"] == "passage"
    assert post.call_args.kwargs["json"]["input"] == ["first chunk", "second chunk"]


def test_embed_query_uses_query_mode():
    provider = _provider()

    with patch(
        "indexing.nvidia_nim_embedding_provider.requests.post",
        return_value=_response([[1.0, 0.0]]),
    ) as post:
        provider.embed_query("what does this document say?")

    assert post.call_args.kwargs["json"]["input_type"] == "query"


def test_query_processor_uses_query_embedding_mode():
    class EmbeddingManager:
        def __init__(self):
            self.queries = []

        def embed_query(self, query):
            self.queries.append(query)
            return np.array([[1.0, 0.0]], dtype=np.float32)

    class Collection:
        def count(self):
            return 0

        def get(self, include):
            return {"documents": []}

    class VectorStoreStub:
        collection = Collection()

        def similarity_search(self, query_embedding, k, filters):
            return []

    embedding_manager = EmbeddingManager()
    processor = QueryProcessor(
        embedding_manager,
        VectorStoreStub(),
        {"similarity_threshold": 0.5, "max_results": 5},
    )

    processor.process_text_query("find the eligibility criteria")

    assert embedding_manager.queries == ["find the eligibility criteria"]


def test_vector_store_rejects_wrong_dimension_instead_of_truncating():
    vector_store = VectorStore.__new__(VectorStore)
    vector_store.collection_name = "nim-text"
    vector_store.embedding_dimension = 2048

    with pytest.raises(ValueError, match="2048 dimensions"):
        vector_store._standardize_query_embedding(np.ones(384, dtype=np.float32))
