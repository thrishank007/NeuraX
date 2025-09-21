"""
Vector store for efficient similarity search and retrieval
"""
import numpy as np
import faiss
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
from loguru import logger
import json
import pickle
from datetime import datetime


class VectorStore:
    """Enhanced vector store with multimodal search capabilities"""
    
    def __init__(self, persist_directory: str, collection_name: str = "secureinsight_collection"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistent storage"""
        try:
            # Create persistent directory if it doesn't exist
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> None:
        """
        Add documents with their embeddings to the store
        
        Args:
            documents: List of processed documents with metadata
            embeddings: Corresponding embeddings array
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Prepare data for ChromaDB
            ids = []
            metadatas = []
            embedding_list = []
            
            for i, doc in enumerate(documents):
                # Generate unique ID
                doc_id = f"{doc.get('file_type', 'unknown')}_{i}_{datetime.now().timestamp()}"
                ids.append(doc_id)
                
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
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embedding_list,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 5, 
                         filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform similarity search and return top-k results
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with metadata and scores
        """
        try:
            # Prepare query
            query_embeddings = [query_embedding.tolist()]
            
            # Build where clause for filtering
            where_clause = {}
            if filters:
                for key, value in filters.items():
                    where_clause[key] = value
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=k,
                where=where_clause if where_clause else None
            )
            
            # Format results
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'metadata': results['metadatas'][0][i]
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def hybrid_search(self, text_query: str = None, image_query: Optional[np.ndarray] = None,
                     k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform cross-modal search across text and image embeddings
        
        Args:
            text_query: Text query string
            image_query: Image embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results from both modalities
        """
        try:
            all_results = []
            
            # Search with text query if provided
            if text_query is not None:
                # This would need the embedding manager to generate text embedding
                # For now, we'll assume the query embedding is provided
                pass
            
            # Search with image query if provided
            if image_query is not None:
                image_results = self.similarity_search(image_query, k, filters)
                all_results.extend(image_results)
            
            # Sort by similarity score and return top-k
            all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return all_results[:k]
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
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