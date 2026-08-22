"""
Query processor for multimodal queries and retrieval coordination
"""
import numpy as np
from typing import List, Dict, Optional, Union
from loguru import logger
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from indexing.embedding_manager import EmbeddingManager
from indexing.vector_store import VectorStore


@dataclass
class QueryResult:
    """Result of a query operation"""
    query: str
    results: List[Dict]
    query_type: str
    processing_time: float
    total_results: int
    similarity_threshold: float


@dataclass
class SearchResult:
    """Individual search result"""
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict
    source_type: str
    file_path: str


class QueryProcessor:
    """Handles multimodal queries and coordinates retrieval"""

    def __init__(self, embedding_manager: EmbeddingManager,
                 vector_store: VectorStore, config: Dict,
                 llm_generator=None):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.config = config
        self.llm_generator = llm_generator

        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_results = config.get('max_results', 10)
        self.enable_cross_modal = config.get('enable_cross_modal', True)

        # Spelling correction cache
        self._vocab = None
        self._vocab_doc_count = 0

        # Cross-encoder reranker (lazy: only built when enabled + configured)
        self._reranker = None
        self._reranker_checked = False

        if self.llm_generator:
            logger.info("Query processor initialized (LLM query rewriting enabled)")
        else:
            logger.info("Query processor initialized")

    def _get_reranker(self):
        """Lazy NIM reranker; None when disabled, unconfigured, or broken.

        Gated on the processor's own config (not the global NIM flag) so
        unit tests passing custom configs never touch the network.
        """
        if self._reranker_checked:
            return self._reranker
        self._reranker_checked = True
        if not self.config.get('enable_reranking', False):
            return None
        from config import NIM_RERANK_CONFIG

        if not NIM_RERANK_CONFIG.get("enabled") or not NIM_RERANK_CONFIG.get("api_key"):
            return None
        try:
            from retrieval.nim_reranker import NimReranker

            self._reranker = NimReranker(NIM_RERANK_CONFIG)
            logger.info(f"NIM reranker ready: {NIM_RERANK_CONFIG.get('model')}")
        except Exception as exc:
            logger.warning(f"Reranker unavailable, fusion order stands: {exc}")
        return self._reranker

    # ── Vocabulary / spelling ────────────────────────────────────────────────

    def _get_vocabulary(self) -> set:
        """Get or rebuild the vocabulary from all documents in the vector store"""
        try:
            current_count = self.vector_store.collection.count()
        except Exception:
            current_count = 0

        if self._vocab is None or self._vocab_doc_count != current_count:
            try:
                all_docs = self.vector_store.collection.get(include=['documents'])
                docs_text = all_docs.get('documents', [])
                import re
                vocab = set()
                for text in docs_text:
                    if text:
                        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                        vocab.update(words)
                self._vocab = vocab
                self._vocab_doc_count = current_count
                logger.info(f"Built spelling vocabulary of {len(vocab)} words from {current_count} documents")
            except Exception as e:
                logger.error(f"Failed to build spelling vocabulary: {e}")
                self._vocab = set()
                self._vocab_doc_count = 0
        return self._vocab

    def _correct_query_spelling(self, query: str) -> str:
        """Correct spelling errors in query using local vocabulary"""
        try:
            from rapidfuzz import process, fuzz
            import re
            vocab = self._get_vocabulary()
            if not vocab:
                return query

            words = query.split()
            corrected_words = []
            for word in words:
                clean_word = re.sub(r'[^a-zA-Z]', '', word).lower()
                if len(clean_word) < 3 or clean_word in vocab:
                    corrected_words.append(word)
                    continue

                match = process.extractOne(clean_word, vocab, scorer=fuzz.WRatio)
                if match:
                    best_match, score, _ = match
                    if score >= 85:
                        if word[0].isupper():
                            best_match = best_match.capitalize()
                        corrected_words.append(best_match)
                        continue
                corrected_words.append(word)
            return " ".join(corrected_words)
        except Exception as e:
            logger.warning(f"Error performing spelling correction: {e}")
            return query

    # ── LLM query rewrite ────────────────────────────────────────────────────

    def _rewrite_query_with_llm(self, query: str) -> str:
        """Rewrite query using LLM for better retrieval. Returns original on any failure."""
        try:
            import requests as _requests

            prompt = (
                "Rewrite this search query to be cleaner and more specific for "
                "document retrieval. Return ONLY the rewritten query, no explanation.\n\n"
                f"Query: {query}"
            )

            payload = {
                "model": self.llm_generator.current_model or "unknown",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 64,
                "temperature": 0.3,
                "top_p": 0.9,
                "stream": False,
            }

            response = _requests.post(
                f"{self.llm_generator.base_url}/chat/completions",
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    rewritten = result["choices"][0]["message"]["content"].strip()
                    if rewritten:
                        return rewritten

            return query
        except Exception as e:
            logger.debug(f"LLM query rewrite failed, using original: {e}")
            return query

    # ── RRF merge ────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(dense_results: list, bm25_results: list, k: int, rrf_k: int = 60,
                   dense_weight: float = 1.0, sparse_weight: float = 1.0) -> list:
        """Reciprocal Rank Fusion of dense and BM25 result lists.

        Score(d) = sum(weight_i / (rrf_k + rank_i)) across all lists.
        Production callers pass dense-favored weights: equal weights let a
        BM25 distractor outrank a chunk that dense already placed in the
        top-k, degrading recall below dense-only on paraphrase-style
        queries (measured on the synthetic eval corpus).
        Returns top-k by RRF score, carrying through all metadata.
        """
        scores: dict = {}
        meta_map: dict = {}

        for rank, r in enumerate(dense_results, 1):
            doc_id = r["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)
            meta_map[doc_id] = r

        for rank, r in enumerate(bm25_results, 1):
            doc_id = r["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight / (rrf_k + rank)
            if doc_id not in meta_map:
                meta_map[doc_id] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        merged = []
        for doc_id, rrf_score in ranked:
            entry = meta_map[doc_id].copy()
            entry["rrf_score"] = rrf_score
            entry.setdefault("similarity_score", 0.0)
            merged.append(entry)
        return merged

    # ── Main query methods ───────────────────────────────────────────────────

    def process_text_query(self, query: str, filters: Optional[Dict] = None,
                           k: int = None) -> QueryResult:
        """
        Process a text query and return relevant results.

        When enable_hybrid=True (default), combines dense vector search with
        BM25 keyword search via Reciprocal Rank Fusion for better recall.
        """
        start_time = datetime.now()
        k = k or self.max_results

        try:
            logger.info(f"Processing text query: {query[:50]}...")

            # Spelling correction
            corrected_query = self._correct_query_spelling(query)
            if corrected_query != query:
                logger.info(f"Corrected query from '{query}' to '{corrected_query}'")
                query = corrected_query

            # Optional LLM query rewriting
            if self.llm_generator and self.config.get('enable_query_rewrite', False):
                rewritten = self._rewrite_query_with_llm(query)
                if rewritten != query:
                    logger.info(f"Rewrote query from '{query}' to '{rewritten}'")
                    query = rewritten

            # Dense embedding
            query_embedding = self.embedding_manager.embed_query(query)[0]

            use_hybrid = self.config.get('enable_hybrid', True)
            bm25_k = self.config.get('bm25_k', 20)
            rrf_k = self.config.get('rrf_k', 20)
            dense_weight = self.config.get('dense_weight', 0.9)
            sparse_weight = self.config.get('sparse_weight', 0.1)

            # Dense retrieval — fetch more candidates when hybrid (pre-RRF)
            dense_k = k * 3 if use_hybrid else k
            dense_results = self.vector_store.similarity_search(
                query_embedding, k=dense_k, filters=filters
            )

            # Optional cross-encoder reranking: rerank a wider fused
            # candidate set down to k before the LLM sees any context.
            reranker = self._get_reranker()
            rerank_candidates = self.config.get('rerank_candidates', 20)

            if use_hybrid:
                bm25_results = self.vector_store.bm25_search(query, k=bm25_k)
                if bm25_results:
                    merge_k = max(k, rerank_candidates) if reranker else k
                    search_results = self._rrf_merge(
                        dense_results, bm25_results, k=merge_k, rrf_k=rrf_k,
                        dense_weight=dense_weight, sparse_weight=sparse_weight,
                    )
                    logger.debug(
                        f"Hybrid RRF: {len(dense_results)} dense + "
                        f"{len(bm25_results)} BM25 -> {len(search_results)} merged"
                    )
                else:
                    # Corpus empty — BM25 returns nothing, fall back to dense
                    search_results = dense_results[:max(k, rerank_candidates) if reranker else k]
            else:
                search_results = dense_results[:max(k, rerank_candidates) if reranker else k]

            if reranker is not None and search_results:
                search_results = reranker.rerank_dicts(query, search_results, top_k=k)

            # Keep any result with a positive RRF or similarity score
            filtered_results = [
                r for r in search_results
                if r.get('rrf_score', r.get('similarity_score', 0.0)) > 0
            ]

            formatted_results = self._format_search_results(filtered_results)
            processing_time = (datetime.now() - start_time).total_seconds()

            return QueryResult(
                query=query,
                results=formatted_results,
                query_type='hybrid' if use_hybrid else 'text',
                processing_time=processing_time,
                total_results=len(filtered_results),
                similarity_threshold=self.similarity_threshold,
            )

        except Exception as e:
            logger.error(f"Error processing text query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=query,
                results=[],
                query_type='text',
                processing_time=processing_time,
                total_results=0,
                similarity_threshold=self.similarity_threshold,
            )

    def process_image_query(self, image: Union[str, Path, Image.Image],
                            filters: Optional[Dict] = None, k: int = None) -> QueryResult:
        """
        Process an image query and return relevant results.
        """
        start_time = datetime.now()
        k = k or self.max_results

        try:
            logger.info("Processing image query...")
            image_embedding = self.embedding_manager.embed_image(image)[0]
            search_results = self.vector_store.similarity_search(
                image_embedding, k=k, filters=filters
            )
            filtered_results = [
                r for r in search_results
                if r['similarity_score'] >= self.similarity_threshold
            ]
            formatted_results = self._format_search_results(filtered_results)
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query="[Image Query]",
                results=formatted_results,
                query_type='image',
                processing_time=processing_time,
                total_results=len(filtered_results),
                similarity_threshold=self.similarity_threshold,
            )
        except Exception as e:
            logger.error(f"Error processing image query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query="[Image Query]",
                results=[],
                query_type='image',
                processing_time=processing_time,
                total_results=0,
                similarity_threshold=self.similarity_threshold,
            )

    def process_multimodal_query(self, text_query: str,
                                 image: Union[str, Path, Image.Image],
                                 filters: Optional[Dict] = None,
                                 k: int = None) -> QueryResult:
        """
        Process a multimodal query combining text and image.
        """
        start_time = datetime.now()
        k = k or self.max_results

        try:
            logger.info(f"Processing multimodal query: {text_query[:30]}... + image")

            text_results = self.process_text_query(text_query, filters, k * 2)
            image_results = self.process_image_query(image, filters, k * 2)

            combined_results = {}
            for result in text_results.results:
                doc_id = result['document_id']
                combined_results[doc_id] = result
                combined_results[doc_id]['text_similarity'] = result['similarity_score']

            for result in image_results.results:
                doc_id = result['document_id']
                if doc_id in combined_results:
                    text_sim = combined_results[doc_id]['text_similarity']
                    image_sim = result['similarity_score']
                    combined_results[doc_id]['similarity_score'] = (text_sim + image_sim) / 2
                    combined_results[doc_id]['image_similarity'] = image_sim
                else:
                    combined_results[doc_id] = result
                    combined_results[doc_id]['image_similarity'] = result['similarity_score']

            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['similarity_score'],
                reverse=True,
            )[:k]

            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=f"{text_query} + [Image]",
                results=sorted_results,
                query_type='multimodal',
                processing_time=processing_time,
                total_results=len(sorted_results),
                similarity_threshold=self.similarity_threshold,
            )
        except Exception as e:
            logger.error(f"Error processing multimodal query: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                query=f"{text_query} + [Image]",
                results=[],
                query_type='multimodal',
                processing_time=processing_time,
                total_results=0,
                similarity_threshold=self.similarity_threshold,
            )

    # ── Formatting helpers ───────────────────────────────────────────────────

    def _format_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """Format search results into standardized format"""
        formatted = []
        for result in search_results:
            metadata = result.get('metadata', {})
            content = result.get('document', '')
            preview = content[:200].strip().replace('\n', ' ') if content else ''
            if not preview:
                preview = self._generate_content_preview(metadata)
            formatted.append({
                'document_id': result['id'],
                'similarity_score': result.get('similarity_score', 0.0),
                'rrf_score': result.get('rrf_score'),
                'file_path': metadata.get('file_path', ''),
                'file_type': metadata.get('file_type', 'unknown'),
                'embedding_type': metadata.get('embedding_type', 'unknown'),
                'metadata': metadata,
                'timestamp': metadata.get('timestamp', ''),
                'content': content,
                'content_preview': preview,
            })
        return formatted

    def _generate_content_preview(self, metadata: Dict, max_length: int = 200) -> str:
        """Generate a content preview from metadata"""
        for source in [
            metadata.get('title', ''),
            metadata.get('subject', ''),
            metadata.get('content_snippet', ''),
            metadata.get('file_path', ''),
        ]:
            if source and len(source.strip()) > 10:
                preview = source.strip()
                return preview[:max_length] + "..." if len(preview) > max_length else preview
        return f"Document: {metadata.get('file_type', 'unknown')} file"

    # ── Utility methods ──────────────────────────────────────────────────────

    def get_similar_documents(self, document_id: str, k: int = 5) -> List[Dict]:
        """Find documents similar to a given document"""
        try:
            similar_docs = self.vector_store.get_similar_documents(document_id, k)
            return self._format_search_results(similar_docs)
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def update_similarity_threshold(self, threshold: float) -> None:
        """Update the similarity threshold for filtering results"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"Updated similarity threshold to: {threshold}")
        else:
            logger.warning(f"Invalid threshold value: {threshold}. Must be between 0.0 and 1.0")

    def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Generate query suggestions based on partial input"""
        suggestions = []
        if len(partial_query) >= 3:
            common_expansions = {
                'sec': ['security', 'section', 'second'],
                'doc': ['document', 'documentation', 'doctor'],
                'img': ['image', 'imaging'],
                'aud': ['audio', 'audit', 'audience'],
                'sys': ['system', 'systematic'],
                'net': ['network', 'networking'],
                'dat': ['data', 'database', 'date'],
            }
            for prefix, expansions in common_expansions.items():
                if partial_query.lower().startswith(prefix):
                    suggestions.extend(
                        [exp for exp in expansions if exp.startswith(partial_query.lower())]
                    )
        return suggestions[:limit]
