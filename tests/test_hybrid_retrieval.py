from unittest.mock import MagicMock
import numpy as np
import pytest

from retrieval.query_processor import QueryProcessor
from indexing.vector_store import VectorStore


def test_rrf_merge_combines_ranks_correctly():
    dense_results = [
        {"id": "doc1", "similarity_score": 0.9, "document": "doc1 text"},
        {"id": "doc2", "similarity_score": 0.8, "document": "doc2 text"},
    ]
    bm25_results = [
        {"id": "doc2", "bm25_score": 5.0, "document": "doc2 text"},
        {"id": "doc3", "bm25_score": 4.0, "document": "doc3 text"},
    ]

    # RRF k = 60
    # doc1: 1/(60+1) = 1/61 = 0.01639
    # doc2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.01613 + 0.01639 = 0.03252
    # doc3: 1/(60+2) = 1/62 = 0.01613
    merged = QueryProcessor._rrf_merge(dense_results, bm25_results, k=3, rrf_k=60)

    assert len(merged) == 3
    assert merged[0]["id"] == "doc2"  # Ranked 1st because it appeared in both
    assert merged[1]["id"] == "doc1"
    assert merged[2]["id"] == "doc3"
    assert merged[0]["rrf_score"] > merged[1]["rrf_score"]
    assert merged[1]["rrf_score"] > merged[2]["rrf_score"]


def test_bm25_search_ranks_matching_documents():
    vector_store = VectorStore.__new__(VectorStore)
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["doc_a", "doc_b", "doc_c"],
        "documents": [
            "Network security firewall policies and protocols",
            "Deep learning neural networks and embeddings",
            "Cooking recipes for homemade pasta and sauce",
        ],
        "metadatas": [{"cat": "security"}, {"cat": "ml"}, {"cat": "food"}],
    }
    vector_store.collection = mock_collection

    results = vector_store.bm25_search("firewall security policy", k=5)

    assert len(results) > 0
    assert results[0]["id"] == "doc_a"
    assert results[0]["query_type"] == "bm25"
    assert results[0]["metadata"]["cat"] == "security"


def test_process_text_query_hybrid_flow():
    class DummyEmbeddingManager:
        text_embedding_model = "test-model"
        uses_nim_embeddings = True

        def embed_query(self, query):
            return np.array([[0.1, 0.2]], dtype=np.float32)

    class DummyCollection:
        def count(self):
            return 2

        def get(self, include=None):
            return {
                "ids": ["doc1", "doc2"],
                "documents": ["alpha beta gamma", "delta epsilon gamma"],
                "metadatas": [{"source": "doc1.txt"}, {"source": "doc2.txt"}],
            }

    class DummyVectorStore:
        collection = DummyCollection()

        def similarity_search(self, query_emb, k, filters):
            return [
                {
                    "id": "doc1",
                    "similarity_score": 0.85,
                    "distance": 0.15,
                    "metadata": {"source": "doc1.txt"},
                    "document": "alpha beta gamma",
                }
            ]

        def bm25_search(self, query, k):
            return [
                {
                    "id": "doc2",
                    "bm25_score": 2.5,
                    "similarity_score": 0.0,
                    "distance": 0.0,
                    "metadata": {"source": "doc2.txt"},
                    "document": "delta epsilon gamma",
                    "query_type": "bm25",
                }
            ]

    processor = QueryProcessor(
        embedding_manager=DummyEmbeddingManager(),
        vector_store=DummyVectorStore(),
        config={
            "similarity_threshold": 0.5,
            "max_results": 5,
            "enable_hybrid": True,
            "bm25_k": 10,
            "rrf_k": 60,
        },
    )

    res = processor.process_text_query("delta gamma")
    assert res.query_type == "hybrid"
    assert len(res.results) == 2
    doc_ids = [r["document_id"] for r in res.results]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
