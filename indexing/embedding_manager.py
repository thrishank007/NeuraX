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


class EmbeddingManager:
    """Manages embeddings for text, images, and multimodal content"""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.text_model = None
        self.clip_model = None
        self.clip_processor = None
        
        self._load_models()
    
    def _load_models(self):
        """Load embedding models"""
        try:
            # Load text embedding model
            logger.info("Loading text embedding model...")
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_model.to(self.device)
            
            # Load CLIP model for images and multimodal
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
            logger.info("All embedding models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding models: {e}")
            raise
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text content
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.text_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating text embeddings: {e}")
            raise
    
    def embed_image(self, images: Union[str, Path, Image.Image, List]) -> np.ndarray:
        """
        Generate embeddings for images using CLIP
        
        Args:
            images: Single image path/PIL Image or list of images
            
        Returns:
            Numpy array of embeddings
        """
        if not isinstance(images, list):
            images = [images]
        
        # Convert paths to PIL Images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img))
            elif isinstance(img, Image.Image):
                pil_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        try:
            # Process images
            inputs = self.clip_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embeddings = image_features.cpu().numpy()
            
            return embeddings
            
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
            
            return {
                'text_embedding': text_embeds[0],
                'image_embedding': image_embeds[0],
                'combined_embedding': np.concatenate([text_embeds[0], image_embeds[0]])
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
    
    def batch_embed_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for a batch of processed documents
        
        Args:
            documents: List of processed documents from ingestion
            
        Returns:
            List of documents with embeddings added
        """
        results = []
        
        for doc in documents:
            try:
                doc_with_embeddings = doc.copy()
                
                if doc['file_type'] in ['pdf', 'docx']:
                    # Text document
                    text_content = []
                    for content in doc['content']:
                        text_content.append(content['text'])
                    
                    if text_content:
                        full_text = ' '.join(text_content)
                        embedding = self.embed_text(full_text)
                        doc_with_embeddings['embedding'] = embedding[0]
                        doc_with_embeddings['embedding_type'] = 'text'
                
                elif doc['file_type'] in ['image', 'screenshot']:
                    # Image document
                    image_path = doc['file_path']
                    image_embedding = self.embed_image(image_path)
                    doc_with_embeddings['embedding'] = image_embedding[0]
                    doc_with_embeddings['embedding_type'] = 'image'
                    
                    # Also embed OCR text if available
                    if doc.get('ocr_text'):
                        text_embedding = self.embed_text(doc['ocr_text'])
                        doc_with_embeddings['text_embedding'] = text_embedding[0]
                        
                        # Create multimodal embedding
                        multimodal = self.embed_multimodal(doc['ocr_text'], image_path)
                        doc_with_embeddings['multimodal_embedding'] = multimodal['combined_embedding']
                
                elif doc['file_type'] == 'audio':
                    # Audio document (use transcription)
                    if doc.get('transcription'):
                        embedding = self.embed_text(doc['transcription'])
                        doc_with_embeddings['embedding'] = embedding[0]
                        doc_with_embeddings['embedding_type'] = 'text'
                
                results.append(doc_with_embeddings)
                logger.info(f"Generated embeddings for: {doc['file_path']}")
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for {doc.get('file_path', 'unknown')}: {e}")
                continue
        
        return results
    
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