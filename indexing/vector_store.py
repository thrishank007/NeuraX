"""
Vector store for efficient similarity search and retrieval
"""
import numpy as np
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
from loguru import logger
import json
import pickle
from datetime import datetime
from config import SIMILARITY_THRESHOLD, KG_CONFIG, PERFORMANCE_CONFIG

from .memory_manager import MemoryManager, ProgressiveLoader, MemoryOptimizer
from .performance_benchmarker import PerformanceBenchmarker


class VectorStore:
    """Enhanced vector store with multimodal search capabilities and memory optimization"""
    
    def __init__(self, persist_directory: str, collection_name: str = "secureinsight_collection"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Initialize memory management
        self.memory_manager = MemoryManager(
            gc_threshold=PERFORMANCE_CONFIG.get('gc_threshold', 0.8)
        )
        self.progressive_loader = ProgressiveLoader(
            chunk_size=PERFORMANCE_CONFIG.get('batch_size', 32)
        )
        self.benchmarker = PerformanceBenchmarker()
        
        # Start memory monitoring
        self.memory_manager.start_monitoring()
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initializes the ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # Use get_or_create_collection to ensure the collection exists
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Successfully connected to collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        """
        Add embeddings to the vector store.
        
        Args:
            ids: List of unique identifiers for the embeddings.
            embeddings: List of embeddings to be added.
            metadatas: List of metadata dictionaries associated with each embedding.
        """
        try:
            # Validate input lengths
            if len(ids) != len(embeddings) or len(ids) != len(metadatas):
                raise ValueError("Length of ids, embeddings, and metadatas must match")
            
            # Convert embeddings to list if they are numpy arrays
            embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(ids)} embeddings to the vector store")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise
    
    def add_document(self, document: Dict) -> None:
        """Add a single document and its embedding to the vector store"""
        if not isinstance(document, dict):
            raise ValueError("Document must be a dictionary")

        embedding_vector = None
        embedding_type = document.get('embedding_type')

        for key, default_type in (
            ('embedding', embedding_type),
            ('text_embedding', 'text'),
            ('image_embedding', 'image'),
            ('multimodal_embedding', 'multimodal'),
        ):
            value = document.get(key)
            if value is None:
                continue

            array = np.asarray(value, dtype=np.float32)
            if array.size == 0:
                continue
            if array.ndim > 1:
                array = array[0]

            embedding_vector = array
            if not embedding_type:
                embedding_type = default_type
            break

        if embedding_vector is None:
            raise ValueError("Document does not contain an embedding to store")

        doc_copy = document.copy()
        doc_copy['embedding_type'] = embedding_type or 'text'

        self.add_documents(
            [doc_copy],
            np.array([embedding_vector], dtype=np.float32)
        )


    def add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> None:
        """
        Add documents with their embeddings to the store with memory optimization
        
        Args:
            documents: List of processed documents with metadata
            embeddings: Corresponding embeddings array
        """
        with self.benchmarker.benchmark_operation("add_documents", len(documents)):
            try:
                if len(documents) != len(embeddings):
                    raise ValueError("Number of documents must match number of embeddings")
                
                # Optimize embeddings for memory efficiency
                optimized_embeddings = MemoryOptimizer.optimize_numpy_arrays([embeddings])[0]
                
                # Check memory pressure and use progressive loading if needed
                if len(documents) > 1000 or self.memory_manager.check_memory_pressure():
                    logger.info("Using progressive loading for large document batch")
                    self._add_documents_progressively(documents, optimized_embeddings)
                else:
                    self._add_documents_batch(documents, optimized_embeddings)
                
                logger.info(f"Added {len(documents)} documents to vector store")
                
            except Exception as e:
                logger.error(f"Failed to add documents to vector store: {e}")
                raise
    
    def _add_documents_batch(self, documents: List[Dict], embeddings: np.ndarray) -> None:
        """Add documents in a single batch"""
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        embedding_list = []
        document_contents = []
        
        for i, doc in enumerate(documents):
            # Generate unique ID
            doc_id = f"{doc.get('file_type', 'unknown')}_{i}_{datetime.now().timestamp()}"
            ids.append(doc_id)
            
            # Prepare document content
            content = doc.get('content', '')
            if isinstance(content, list):
                # If content is a list of paragraphs/chunks, join them
                content = ' '.join([item.get('text', str(item)) if isinstance(item, dict) else str(item) for item in content])
            elif isinstance(content, dict):
                content = content.get('text', str(content))
            document_contents.append(str(content))
            
            # Prepare metadata
            metadata = {
                'file_path': doc.get('file_path', ''),
                'file_type': doc.get('file_type', ''),
                'embedding_type': doc.get('embedding_type', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add additional metadata based on file type
            if 'metadata' in doc:
                for key, value in doc['metadata'].items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[f"meta_{key}"] = value
            
            metadatas.append(metadata)
            embedding_list.append(embeddings[i].tolist())
        
        # Add to collection with documents
        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            metadatas=metadatas,
            documents=document_contents
        )
    
    def _add_documents_progressively(self, documents: List[Dict], embeddings: np.ndarray) -> None:
        """Add documents progressively in chunks to manage memory"""
        chunk_size = self.progressive_loader.chunk_size
        
        for i in range(0, len(documents), chunk_size):
            # Check memory pressure before each chunk
            if self.memory_manager.check_memory_pressure():
                logger.warning("Memory pressure detected, forcing garbage collection")
                self.memory_manager.force_gc()
            
            # Process chunk
            end_idx = min(i + chunk_size, len(documents))
            doc_chunk = documents[i:end_idx]
            embedding_chunk = embeddings[i:end_idx]
            
            self._add_documents_batch(doc_chunk, embedding_chunk)
            
            logger.debug(f"Added document chunk {i//chunk_size + 1}: {end_idx}/{len(documents)} documents")
    
    def _standardize_query_embedding(self, query_embedding: np.ndarray, target_dim: int = 384) -> np.ndarray:
        """
        Standardize query embedding dimension to match collection embeddings
        
        Args:
            query_embedding: Input query embedding
            target_dim: Target dimension (default 384)
            
        Returns:
            Standardized query embedding
        """
        current_dim = query_embedding.shape[0] if query_embedding.ndim == 1 else query_embedding.shape[-1]
        
        if current_dim == target_dim:
            return query_embedding
        elif current_dim > target_dim:
            # Truncate to target dimension
            logger.warning(f"Truncating query embedding from {current_dim} to {target_dim} dimensions")
            return query_embedding[:target_dim] if query_embedding.ndim == 1 else query_embedding[..., :target_dim]
        else:
            # Pad with zeros to reach target dimension
            logger.warning(f"Padding query embedding from {current_dim} to {target_dim} dimensions")
            if query_embedding.ndim == 1:
                padding = np.zeros(target_dim - current_dim)
                return np.concatenate([query_embedding, padding])
            else:
                padding_shape = list(query_embedding.shape)
                padding_shape[-1] = target_dim - current_dim
                padding = np.zeros(padding_shape)
                return np.concatenate([query_embedding, padding], axis=-1)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 5, 
                         filters: Optional[Dict] = None, 
                         similarity_threshold: Optional[float] = None) -> List[Dict]:
        """
        Perform similarity search and return top-k results with performance monitoring
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity threshold (uses config default if None)
            
        Returns:
            List of search results with metadata and scores
        """
        with self.benchmarker.benchmark_operation("similarity_search", 1):
            try:
                # Use config threshold if not provided
                if similarity_threshold is None:
                    similarity_threshold = SIMILARITY_THRESHOLD
                
                # Standardize and optimize query embedding
                standardized_query = self._standardize_query_embedding(query_embedding)
                optimized_query = MemoryOptimizer.optimize_numpy_arrays([standardized_query])[0]
                
                # Prepare query
                query_embeddings = [optimized_query.tolist()]
                
                # Build where clause for filtering
                where_clause = {}
                if filters:
                    for key, value in filters.items():
                        where_clause[key] = value
                
                # Perform search with more results to filter by threshold
                search_k = min(k * 3, 100)  # Get more results to filter
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=search_k,
                    where=where_clause if where_clause else None,
                    include=['documents', 'metadatas', 'distances']  # Include documents in results
                )
                
                # Format and filter results by similarity threshold
                search_results = []
                for i in range(len(results['ids'][0])):
                    similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_score >= similarity_threshold:
                        result = {
                            'id': results['ids'][0][i],
                            'distance': results['distances'][0][i],
                            'similarity_score': similarity_score,
                            'metadata': results['metadatas'][0][i],
                            'document': results['documents'][0][i] if results.get('documents') and len(results['documents'][0]) > i else None
                        }
                        search_results.append(result)
                
                # Return top-k results
                return search_results[:k]
                
            except Exception as e:
                logger.error(f"Failed to perform similarity search: {e}")
                return []
    
    def hybrid_search(self, text_query_embedding: Optional[np.ndarray] = None, 
                     image_query_embedding: Optional[np.ndarray] = None,
                     k: int = 5, filters: Optional[Dict] = None,
                     similarity_threshold: Optional[float] = None) -> List[Dict]:
        """
        Perform cross-modal search across text and image embeddings
        
        Args:
            text_query_embedding: Text query embedding vector
            image_query_embedding: Image query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results from both modalities
        """
        try:
            all_results = []
            
            # Search with text query embedding if provided
            if text_query_embedding is not None:
                text_results = self.similarity_search(
                    text_query_embedding, k, filters, similarity_threshold
                )
                # Mark results as from text query
                for result in text_results:
                    result['query_type'] = 'text'
                all_results.extend(text_results)
            
            # Search with image query embedding if provided
            if image_query_embedding is not None:
                image_results = self.similarity_search(
                    image_query_embedding, k, filters, similarity_threshold
                )
                # Mark results as from image query
                for result in image_results:
                    result['query_type'] = 'image'
                all_results.extend(image_results)
            
            # Remove duplicates based on document ID
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result['id'] not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result['id'])
            
            # Sort by similarity score and return top-k
            unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return unique_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return []
    
    def cross_modal_search(self, query_embedding: np.ndarray, target_modality: str = None,
                          k: int = 5, filters: Optional[Dict] = None,
                          similarity_threshold: Optional[float] = None) -> List[Dict]:
        """
        Perform cross-modal search (e.g., text query finding images)
        
        Args:
            query_embedding: Query embedding vector
            target_modality: Target modality to search ('text', 'image', or None for all)
            k: Number of results to return
            filters: Optional metadata filters
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results
        """
        try:
            # Add modality filter if specified
            search_filters = filters.copy() if filters else {}
            if target_modality:
                search_filters['embedding_type'] = target_modality
            
            return self.similarity_search(
                query_embedding, k, search_filters, similarity_threshold
            )
            
        except Exception as e:
            logger.error(f"Failed to perform cross-modal search: {e}")
            return []
    
    def get_similar_documents(self, doc_id: str, k: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document
        
        Args:
            doc_id: ID of the reference document
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        try:
            # Get the document embedding
            doc_result = self.collection.get(ids=[doc_id], include=['embeddings'])
            
            if not doc_result['embeddings']:
                logger.warning(f"Document {doc_id} not found")
                return []
            
            # Use the document's embedding as query
            doc_embedding = np.array(doc_result['embeddings'][0])
            
            # Search for similar documents (excluding the original)
            results = self.similarity_search(doc_embedding, k + 1)
            
            # Filter out the original document
            similar_docs = [r for r in results if r['id'] != doc_id]
            
            return similar_docs[:k]
            
        except Exception as e:
            logger.error(f"Failed to find similar documents for {doc_id}: {e}")
            return []
    
    def export_index(self, export_path: Path) -> None:
        """
        Export the vector index for portable deployment
        
        Args:
            export_path: Path to export the index
        """
        try:
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            # Get all data from collection
            all_data = self.collection.get(include=['embeddings', 'metadatas'])
            
            # Save to pickle file
            export_data = {
                'collection_name': self.collection_name,
                'ids': all_data['ids'],
                'embeddings': all_data['embeddings'],
                'metadatas': all_data['metadatas'],
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(export_path / 'vector_index.pkl', 'wb') as f:
                pickle.dump(export_data, f)
            
            logger.info(f"Vector index exported to: {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export index: {e}")
            raise
    
    def import_index(self, import_path: Path) -> None:
        """
        Import a vector index from portable deployment
        
        Args:
            import_path: Path to import the index from
        """
        try:
            import_file = Path(import_path) / 'vector_index.pkl'
            
            if not import_file.exists():
                raise FileNotFoundError(f"Import file not found: {import_file}")
            
            # Load data
            with open(import_file, 'rb') as f:
                import_data = pickle.load(f)
            
            # Clear existing collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add imported data
            if import_data['ids']:
                self.collection.add(
                    ids=import_data['ids'],
                    embeddings=import_data['embeddings'],
                    metadatas=import_data['metadatas']
                )
            
            logger.info(f"Vector index imported from: {import_path}")
            
        except Exception as e:
            logger.error(f"Failed to import index: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze types
            sample_data = self.collection.get(limit=min(100, count), include=['metadatas'])
            
            file_types = {}
            embedding_types = {}
            
            for metadata in sample_data['metadatas']:
                file_type = metadata.get('file_type', 'unknown')
                embedding_type = metadata.get('embedding_type', 'unknown')
                
                file_types[file_type] = file_types.get(file_type, 0) + 1
                embedding_types[embedding_type] = embedding_types.get(embedding_type, 0) + 1
            
            return {
                'total_documents': count,
                'file_types': file_types,
                'embedding_types': embedding_types,
                'collection_name': self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics for the vector store"""
        memory_stats = self.memory_manager.get_memory_stats()
        collection_stats = self.get_collection_stats()
        
        return {
            'system_memory': {
                'total_gb': memory_stats.total_memory,
                'used_gb': memory_stats.used_memory,
                'available_gb': memory_stats.available_memory,
                'usage_percent': memory_stats.memory_percent,
                'process_memory_mb': memory_stats.process_memory
            },
            'collection_stats': collection_stats,
            'gc_stats': memory_stats.gc_collections,
            'recommendations': self.memory_manager.get_memory_recommendations()
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage for the vector store"""
        logger.info("Optimizing vector store memory usage")
        
        # Force garbage collection
        gc_result = self.memory_manager.force_gc()
        
        # Get memory stats before and after
        memory_before = self.memory_manager.get_memory_stats()
        
        # Optimize for large dataset if needed
        if memory_before.memory_percent > 70:
            self.memory_manager.optimize_for_large_dataset()
        
        memory_after = self.memory_manager.get_memory_stats()
        
        optimization_result = {
            'gc_collections': gc_result,
            'memory_before_percent': memory_before.memory_percent,
            'memory_after_percent': memory_after.memory_percent,
            'memory_freed_mb': (memory_before.used_memory - memory_after.used_memory) * 1024,
            'recommendations': self.memory_manager.get_memory_recommendations()
        }
        
        logger.info(f"Memory optimization completed: "
                   f"{memory_before.memory_percent:.1f}% -> {memory_after.memory_percent:.1f}")
        
        return optimization_result    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for vector store operations"""
        return self.benchmarker.generate_performance_report()
    
    def benchmark_search_performance(self, test_embeddings: List[np.ndarray], 
                                   k_values: List[int] = None) -> Dict[int, Any]:
        """
        Benchmark search performance with different k values
        
        Args:
            test_embeddings: List of test query embeddings
            k_values: List of k values to test
            
        Returns:
            Dictionary mapping k values to benchmark results
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]
        
        return self.benchmarker.benchmark_vector_search(self, test_embeddings, k_values)
    
    def __del__(self):
        """Cleanup when vector store is destroyed"""
        try:
            if hasattr(self, 'memory_manager'):
                self.memory_manager.stop_monitoring()
        except Exception:
            pass  # Ignore cleanup errors