"""
Query processor for multimodal queries and retrieval coordination
"""
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
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
                 vector_store: VectorStore, config: Dict):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.config = config
        
        # Configuration parameters
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_results = config.get('max_results', 10)
        self.enable_cross_modal = config.get('enable_cross_modal', True)
        
        logger.info("Query processor initialized")
    
    def process_text_query(self, query: str, filters: Optional[Dict] = None,
                          k: int = None) -> QueryResult:
        """
        Process a text query and return relevant results
        
        Args:
            query: Text query string
            filters: Optional metadata filters
            k: Number of results to return (defaults to config)
            
        Returns:
            QueryResult with search results
        """
        start_time = datetime.now()
        k = k or self.max_results
        
        try:
            logger.info(f"Processing text query: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_manager.embed_text(query)[0]
            
            # Perform similarity search
            search_results = self.vector_store.similarity_search(
                query_embedding, k=k, filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in search_results 
                if r['similarity_score'] >= self.similarity_threshold
            ]
            
            # Format results
            formatted_results = self._format_search_results(filtered_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=query,
                results=formatted_results,
                query_type='text',
                processing_time=processing_time,
                total_results=len(filtered_results),
                similarity_threshold=self.similarity_threshold
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
                similarity_threshold=self.similarity_threshold
            )
    
    def process_image_query(self, image: Union[str, Path, Image.Image],
                           filters: Optional[Dict] = None, k: int = None) -> QueryResult:
        """
        Process an image query and return relevant results
        
        Args:
            image: Image path or PIL Image object
            filters: Optional metadata filters
            k: Number of results to return
            
        Returns:
            QueryResult with search results
        """
        start_time = datetime.now()
        k = k or self.max_results
        
        try:
            logger.info("Processing image query...")
            
            # Generate image embedding
            image_embedding = self.embedding_manager.embed_image(image)[0]
            
            # Perform similarity search
            search_results = self.vector_store.similarity_search(
                image_embedding, k=k, filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                r for r in search_results 
                if r['similarity_score'] >= self.similarity_threshold
            ]
            
            # Format results
            formatted_results = self._format_search_results(filtered_results)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query="[Image Query]",
                results=formatted_results,
                query_type='image',
                processing_time=processing_time,
                total_results=len(filtered_results),
                similarity_threshold=self.similarity_threshold
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
                similarity_threshold=self.similarity_threshold
            )
    
    def process_multimodal_query(self, text_query: str, 
                                image: Union[str, Path, Image.Image],
                                filters: Optional[Dict] = None, 
                                k: int = None) -> QueryResult:
        """
        Process a multimodal query combining text and image
        
        Args:
            text_query: Text component of query
            image: Image component of query
            filters: Optional metadata filters
            k: Number of results to return
            
        Returns:
            QueryResult with combined search results
        """
        start_time = datetime.now()
        k = k or self.max_results
        
        try:
            logger.info(f"Processing multimodal query: {text_query[:30]}... + image")
            
            # Get results from both modalities
            text_results = self.process_text_query(text_query, filters, k * 2)
            image_results = self.process_image_query(image, filters, k * 2)
            
            # Combine and deduplicate results
            combined_results = {}
            
            # Add text results
            for result in text_results.results:
                doc_id = result['document_id']
                combined_results[doc_id] = result
                combined_results[doc_id]['text_similarity'] = result['similarity_score']
            
            # Add image results and combine scores
            for result in image_results.results:
                doc_id = result['document_id']
                if doc_id in combined_results:
                    # Average the similarities for multimodal scoring
                    text_sim = combined_results[doc_id]['text_similarity']
                    image_sim = result['similarity_score']
                    combined_results[doc_id]['similarity_score'] = (text_sim + image_sim) / 2
                    combined_results[doc_id]['image_similarity'] = image_sim
                else:
                    combined_results[doc_id] = result
                    combined_results[doc_id]['image_similarity'] = result['similarity_score']
            
            # Sort by combined similarity and take top-k
            sorted_results = sorted(
                combined_results.values(),
                key=lambda x: x['similarity_score'],
                reverse=True
            )[:k]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query=f"{text_query} + [Image]",
                results=sorted_results,
                query_type='multimodal',
                processing_time=processing_time,
                total_results=len(sorted_results),
                similarity_threshold=self.similarity_threshold
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
                similarity_threshold=self.similarity_threshold
            )
    
    def _format_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """Format search results into standardized format"""
        formatted = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            
            formatted_result = {
                'document_id': result['id'],
                'similarity_score': result['similarity_score'],
                'file_path': metadata.get('file_path', ''),
                'file_type': metadata.get('file_type', 'unknown'),
                'embedding_type': metadata.get('embedding_type', 'unknown'),
                'metadata': metadata,
                'timestamp': metadata.get('timestamp', ''),
                'content_preview': self._generate_content_preview(metadata)
            }
            
            formatted.append(formatted_result)
        
        return formatted
    
    def _generate_content_preview(self, metadata: Dict, max_length: int = 200) -> str:
        """Generate a content preview from metadata"""
        # Try to extract meaningful preview text
        preview_sources = [
            metadata.get('title', ''),
            metadata.get('subject', ''),
            metadata.get('content_snippet', ''),
            metadata.get('file_path', '')
        ]
        
        for source in preview_sources:
            if source and len(source.strip()) > 10:
                preview = source.strip()
                if len(preview) > max_length:
                    preview = preview[:max_length] + "..."
                return preview
        
        return f"Document: {metadata.get('file_type', 'unknown')} file"
    
    def get_similar_documents(self, document_id: str, k: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document
        
        Args:
            document_id: ID of reference document
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
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
        """
        Generate query suggestions based on partial input
        
        Args:
            partial_query: Partial query string
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        # This is a simple implementation - could be enhanced with more sophisticated methods
        suggestions = []
        
        if len(partial_query) >= 3:
            # Basic keyword expansion suggestions
            common_expansions = {
                'sec': ['security', 'section', 'second'],
                'doc': ['document', 'documentation', 'doctor'],
                'img': ['image', 'imaging'],
                'aud': ['audio', 'audit', 'audience'],
                'sys': ['system', 'systematic'],
                'net': ['network', 'networking'],
                'dat': ['data', 'database', 'date']
            }
            
            for prefix, expansions in common_expansions.items():
                if partial_query.lower().startswith(prefix):
                    suggestions.extend([exp for exp in expansions if exp.startswith(partial_query.lower())])
        
        return suggestions[:limit]