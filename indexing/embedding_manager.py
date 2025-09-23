"""
Embedding manager for multimodal content
"""
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import List, Dict, Union, Optional
from loguru import logger
from pathlib import Path
import pickle
import hashlib
from tqdm import tqdm
import os

from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity


class EmbeddingManager:
    """Manages embeddings for text, images, and multimodal content"""
    
    def __init__(self, device: Optional[str] = None, cache_dir: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.error_handler = ErrorHandler()
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        logger.info("EmbeddingManager initialized, about to load models...")
        
        # Setup caching
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_cache = {}
        
        self._load_models()
        
        # Final verification that models are properly loaded
        logger.info("Performing final model verification...")
        if self.text_model is None:
            raise RuntimeError("Text model failed to load")
        if not hasattr(self.text_model, 'encode'):
            raise RuntimeError(f"Text model is invalid type: {type(self.text_model)}")
        if self.clip_model is None:
            raise RuntimeError("CLIP model failed to load")
        if self.clip_processor is None:
            raise RuntimeError("CLIP processor failed to load")
        
        logger.info("✅ EmbeddingManager initialization completed successfully")
    
    def _collect_text_fragments(self, value) -> List[str]:
        fragments = []

        def collect(item):
            if item is None:
                return
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    fragments.append(stripped)
            elif isinstance(item, dict):
                for key in ('text', 'content', 'transcription', 'ocr_text', 'value', 'summary'):
                    if key in item:
                        collect(item[key])
            elif isinstance(item, (list, tuple, set)):
                for sub_item in item:
                    collect(sub_item)
            else:
                text_value = str(item).strip()
                if text_value:
                    fragments.append(text_value)

        collect(value)
        return fragments

    def _prepare_text_inputs(self, texts: Union[str, List, Dict, None]) -> List[str]:
        if texts is None:
            return []
        if isinstance(texts, str):
            stripped = texts.strip()
            return [stripped] if stripped else []
        if isinstance(texts, list):
            if all(isinstance(item, str) for item in texts):
                cleaned = [item.strip() for item in texts if isinstance(item, str) and item.strip()]
                return cleaned
            fragments = self._collect_text_fragments(texts)
            combined = ' '.join(fragments).strip()
            return [combined] if combined else []
        if isinstance(texts, dict):
            fragments = self._collect_text_fragments(texts)
            combined = ' '.join(fragments).strip()
            return [combined] if combined else []
        text_repr = str(texts).strip()
        return [text_repr] if text_repr else []

    def _load_models(self):
        """Load embedding models with error handling"""
        try:
            # Load text model
            logger.info("Loading text embedding model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info(f"Text model loaded successfully: {type(self.text_model)}")
            
            try:
                self.text_model = self.text_model.to(self.device)
                logger.info(f"Text model moved to {self.device}: {type(self.text_model)}")
            except Exception as device_error:
                logger.warning(f"Failed to move text model to {self.device}, using CPU: {device_error}")
                self.device = 'cpu'
                self.text_model = self.text_model.to(self.device)
                logger.info(f"Text model moved to CPU: {type(self.text_model)}")
            
            # Load CLIP model
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info(f"CLIP model loaded successfully: {type(self.clip_model)}")
            
            try:
                self.clip_model = self.clip_model.to(self.device)
                logger.info(f"CLIP model moved to {self.device}: {type(self.clip_model)}")
            except Exception as device_error:
                logger.warning(f"Failed to move CLIP model to {self.device}, using CPU: {device_error}")
                self.device = 'cpu'
                self.clip_model = self.clip_model.to(self.device)
                logger.info(f"CLIP model moved to CPU: {type(self.clip_model)}")
            
            logger.info("All embedding models loaded successfully")
            logger.info(f"Final model types - Text: {type(self.text_model)}, CLIP: {type(self.clip_model)}, Processor: {type(self.clip_processor)}")
            
        except Exception as e:
            logger.critical(f"Model loading failed: {e}")
            # Simple CPU fallback without complex error handling
            if self.device != 'cpu':
                logger.info("Attempting CPU fallback for model loading")
                self.device = 'cpu'
                try:
                    # Retry loading with CPU
                    logger.info("Retrying model loading with CPU...")
                    self.text_model = SentenceTransformer('all-MiniLM-L6-v2').to('cpu')
                    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cpu')
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    logger.info("Successfully loaded models with CPU fallback")
                    logger.info(f"Final model types after CPU fallback - Text: {type(self.text_model)}, CLIP: {type(self.clip_model)}")
                except Exception as fallback_error:
                    logger.critical(f"Model loading failed even with CPU fallback: {fallback_error}")
                    raise
            else:
                logger.critical("Already using CPU, cannot fallback further")
                raise
    
    def _get_cache_key(self, content: str, embedding_type: str) -> str:
        """Generate cache key for content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{embedding_type}_{content_hash}"
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _standardize_embedding_dimension(self, embedding: np.ndarray, target_dim: int = 384) -> np.ndarray:
        """
        Standardize embedding dimension to ensure compatibility across different models
        
        Args:
            embedding: Input embedding array
            target_dim: Target dimension (default 384 for text model compatibility)
            
        Returns:
            Standardized embedding with target dimension
        """
        current_dim = embedding.shape[0] if embedding.ndim == 1 else embedding.shape[-1]
        
        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate to target dimension
            logger.warning(f"Truncating embedding from {current_dim} to {target_dim} dimensions")
            return embedding[:target_dim] if embedding.ndim == 1 else embedding[..., :target_dim]
        else:
            # Pad with zeros to reach target dimension
            logger.warning(f"Padding embedding from {current_dim} to {target_dim} dimensions")
            if embedding.ndim == 1:
                padding = np.zeros(target_dim - current_dim)
                return np.concatenate([embedding, padding])
            else:
                padding_shape = list(embedding.shape)
                padding_shape[-1] = target_dim - current_dim
                padding = np.zeros(padding_shape)
                return np.concatenate([embedding, padding], axis=-1)
    
    def embed_text(self, texts: Union[str, List[str]], normalize: bool = True, 
                   use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for text content
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings for cosine similarity
            use_cache: Whether to use caching mechanism
            
        Returns:
            Numpy array of embeddings
        """
        texts = self._prepare_text_inputs(texts)

        if not texts:
            raise ValueError("No valid text content provided for embedding.")

        try:
            embeddings_list = []
            uncached_texts = []
            uncached_indices = []
            
            # Check cache first if enabled
            if use_cache:
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text, "text")
                    if cache_key in self._embedding_cache:
                        embeddings_list.append(self._embedding_cache[cache_key])
                    else:
                        embeddings_list.append(None)
                        uncached_texts.append(text)
                        uncached_indices.append(i)
            else:
                uncached_texts = texts
                uncached_indices = list(range(len(texts)))
                embeddings_list = [None] * len(texts)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                if self.text_model is None:
                    logger.error("Text embedding model not loaded. Cannot generate embeddings.")
                    raise RuntimeError("Text embedding model not initialized")
                
                # Debug: Check if text_model is the correct type
                logger.info(f"DEBUG: text_model type: {type(self.text_model)}")
                logger.info(f"DEBUG: text_model value: {self.text_model}")
                logger.info(f"DEBUG: text_model has encode: {hasattr(self.text_model, 'encode')}")
                
                if not hasattr(self.text_model, 'encode'):
                    logger.error(f"text_model is not a SentenceTransformer model. Type: {type(self.text_model)}, Value: {self.text_model}")
                    raise RuntimeError(f"Text model is invalid type: {type(self.text_model)}")
                
                new_embeddings = self.text_model.encode(uncached_texts, convert_to_numpy=True)
                
                # Standardize dimensions and normalize if requested
                if normalize:
                    new_embeddings = np.array([
                        self._normalize_embedding(self._standardize_embedding_dimension(emb)) 
                        for emb in new_embeddings
                    ])
                else:
                    new_embeddings = np.array([
                        self._standardize_embedding_dimension(emb) 
                        for emb in new_embeddings
                    ])
                
                # Update cache and results
                for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    idx = uncached_indices[i]
                    embeddings_list[idx] = embedding
                    
                    if use_cache:
                        cache_key = self._get_cache_key(text, "text")
                        self._embedding_cache[cache_key] = embedding
            
            return np.array(embeddings_list)
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
    
    def embed_image(self, images: Union[str, Path, Image.Image, List], 
                   normalize: bool = True, use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for images using CLIP
        
        Args:
            images: Single image path/PIL Image or list of images
            normalize: Whether to normalize embeddings for cosine similarity
            use_cache: Whether to use caching mechanism
            
        Returns:
            Numpy array of embeddings
        """
        if not isinstance(images, list):
            images = [images]
        
        try:
            embeddings_list = []
            uncached_images = []
            uncached_indices = []
            pil_images = []
            
            # Convert paths to PIL Images and check cache
            for i, img in enumerate(images):
                if isinstance(img, (str, Path)):
                    pil_img = Image.open(img)
                    img_path = str(img)
                elif isinstance(img, Image.Image):
                    pil_img = img
                    img_path = f"pil_image_{hash(img.tobytes())}"
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                
                pil_images.append(pil_img)
                
                # Check cache
                if use_cache:
                    cache_key = self._get_cache_key(img_path, "image")
                    if cache_key in self._embedding_cache:
                        embeddings_list.append(self._embedding_cache[cache_key])
                    else:
                        embeddings_list.append(None)
                        uncached_images.append((pil_img, img_path))
                        uncached_indices.append(i)
                else:
                    embeddings_list.append(None)
                    uncached_images.append((pil_img, img_path))
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached images
            if uncached_images:
                uncached_pil_images = [img[0] for img in uncached_images]
                
                # Process images
                inputs = self.clip_processor(images=uncached_pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    new_embeddings = image_features.cpu().numpy()
                
                # Standardize dimensions and normalize if requested
                if normalize:
                    new_embeddings = np.array([
                        self._normalize_embedding(self._standardize_embedding_dimension(emb)) 
                        for emb in new_embeddings
                    ])
                else:
                    new_embeddings = np.array([
                        self._standardize_embedding_dimension(emb) 
                        for emb in new_embeddings
                    ])
                
                # Update cache and results
                for i, ((pil_img, img_path), embedding) in enumerate(zip(uncached_images, new_embeddings)):
                    idx = uncached_indices[i]
                    embeddings_list[idx] = embedding
                    
                    if use_cache:
                        cache_key = self._get_cache_key(img_path, "image")
                        self._embedding_cache[cache_key] = embedding
            
            return np.array(embeddings_list)
            
        except Exception as e:
            logger.error(f"Error generating image embeddings: {e}")
            raise
    
    def embed_multimodal(self, text: str, image: Union[str, Path, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Generate multimodal embeddings for text-image pairs
        
        Args:
            text: Text content
            image: Image path or PIL Image
            
        Returns:
            Dict containing text and image embeddings
        """
        try:
            # Convert image if needed
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            else:
                pil_image = image
            
            # Process inputs
            inputs = self.clip_processor(
                text=[text], 
                images=[pil_image], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                text_embeds = outputs.text_embeds.cpu().numpy()
                image_embeds = outputs.image_embeds.cpu().numpy()
            
            # Standardize dimensions for compatibility
            text_embeds_std = self._standardize_embedding_dimension(text_embeds[0])
            image_embeds_std = self._standardize_embedding_dimension(image_embeds[0])
            
            return {
                'text_embedding': text_embeds_std,
                'image_embedding': image_embeds_std,
                'combined_embedding': np.concatenate([text_embeds_std, image_embeds_std])
            }
            
        except Exception as e:
            logger.error(f"Error generating multimodal embeddings: {e}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def batch_embed_documents(self, documents: List[Dict], show_progress: bool = True) -> List[Dict]:
        """
        Generate embeddings for a batch of processed documents with progress indicators
        
        Args:
            documents: List of processed documents from ingestion
            show_progress: Whether to show progress bar
            
        Returns:
            List of documents with embeddings added
        """
        results = []
        
        # Setup progress bar
        iterator = tqdm(documents, desc="Generating embeddings") if show_progress else documents
        
        for doc in iterator:
            try:
                doc_with_embeddings = doc.copy()
                
                if doc['file_type'] in ['pdf', 'docx']:
                    # Text document
                    text_content = []
                    for content in doc['content']:
                        text_content.append(content['text'])
                    
                    if text_content:
                        full_text = ' '.join(text_content)
                        embedding = self.embed_text(full_text, normalize=True, use_cache=True)
                        doc_with_embeddings['embedding'] = embedding[0]
                        doc_with_embeddings['embedding_type'] = 'text'
                
                elif doc['file_type'] in ['image', 'screenshot']:
                    # Image document
                    image_path = doc['file_path']
                    image_embedding = self.embed_image(image_path, normalize=True, use_cache=True)
                    doc_with_embeddings['embedding'] = image_embedding[0]
                    doc_with_embeddings['embedding_type'] = 'image'
                    
                    # Also embed OCR text if available
                    if doc.get('ocr_text'):
                        text_embedding = self.embed_text(doc['ocr_text'], normalize=True, use_cache=True)
                        doc_with_embeddings['text_embedding'] = text_embedding[0]
                        
                        # Create multimodal embedding
                        multimodal = self.embed_multimodal(doc['ocr_text'], image_path)
                        doc_with_embeddings['multimodal_embedding'] = multimodal['combined_embedding']
                
                elif doc['file_type'] == 'audio':
                    # Audio document (use transcription)
                    if doc.get('transcription'):
                        embedding = self.embed_text(doc['transcription'], normalize=True, use_cache=True)
                        doc_with_embeddings['embedding'] = embedding[0]
                        doc_with_embeddings['embedding_type'] = 'text'
                
                results.append(doc_with_embeddings)
                
                if show_progress:
                    iterator.set_postfix({'processed': doc.get('file_path', 'unknown')})
                else:
                    logger.info(f"Generated embeddings for: {doc['file_path']}")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {doc.get('file_path', 'unknown')}: {e}")
                continue
        
        return results
    
    def batch_embed_texts(self, texts: List[str], batch_size: int = 32, 
                         show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of texts with progress indicators
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch embedding texts") if show_progress else range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embed_text(batch_texts, normalize=True, use_cache=True)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def batch_embed_images(self, images: List[Union[str, Path, Image.Image]], 
                          batch_size: int = 16, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a batch of images with progress indicators
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(images), batch_size), desc="Batch embedding images") if show_progress else range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_embeddings = self.embed_image(batch_images, normalize=True, use_cache=True)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def save_embeddings(self, embeddings_data: List[Dict], save_path: Path):
        """Save embeddings to disk"""
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(embeddings_data, f)
            logger.info(f"Embeddings saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            raise
    
    def load_embeddings(self, load_path: Path) -> List[Dict]:
        """Load embeddings from disk"""
        try:
            with open(load_path, 'rb') as f:
                embeddings_data = pickle.load(f)
            logger.info(f"Embeddings loaded from: {load_path}")
            return embeddings_data
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def save_cache(self) -> None:
        """Save embedding cache to disk"""
        try:
            cache_file = self.cache_dir / "embedding_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.info(f"Embedding cache saved to: {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load_cache(self) -> None:
        """Load embedding cache from disk"""
        try:
            cache_file = self.cache_dir / "embedding_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Embedding cache loaded from: {cache_file}")
            else:
                logger.info("No existing cache found, starting with empty cache")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self._embedding_cache = {}
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'total_cached_embeddings': len(self._embedding_cache),
            'cache_size_mb': sum(emb.nbytes for emb in self._embedding_cache.values()) / (1024 * 1024)
        }