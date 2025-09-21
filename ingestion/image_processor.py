"""
Image processing module for screenshots and images
"""
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import numpy as np


class ImageProcessor:
    """Handles processing of images and screenshots"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def process_image(self, image_path: Path) -> Dict:
        """
        Process an image file and extract text via OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing OCR text and image metadata
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        suffix = image_path.suffix.lower()
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {suffix}")
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(image, lang='eng')
            
            # Get image metadata
            width, height = image.size
            
            # Calculate image statistics for quality assessment
            img_array = np.array(image)
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            return {
                'file_path': str(image_path),
                'file_type': 'image',
                'ocr_text': ocr_text.strip(),
                'metadata': {
                    'width': width,
                    'height': height,
                    'format': image.format,
                    'mode': image.mode,
                    'brightness': float(brightness),
                    'contrast': float(contrast),
                    'file_size': image_path.stat().st_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_screenshot(self, screenshot_path: Path, context: Optional[str] = None) -> Dict:
        """
        Process a screenshot with additional context
        
        Args:
            screenshot_path: Path to screenshot
            context: Optional context about when/why screenshot was taken
            
        Returns:
            Dict containing processed screenshot data
        """
        result = self.process_image(screenshot_path)
        
        # Add screenshot-specific metadata
        result['screenshot_context'] = context or ''
        result['file_type'] = 'screenshot'
        
        # Enhanced OCR for screenshots (often contain UI elements)
        try:
            image = Image.open(screenshot_path)
            
            # Try different OCR configurations for better UI text extraction
            custom_config = r'--oem 3 --psm 6'
            enhanced_ocr = pytesseract.image_to_string(image, config=custom_config)
            
            if len(enhanced_ocr.strip()) > len(result['ocr_text']):
                result['ocr_text'] = enhanced_ocr.strip()
                result['ocr_method'] = 'enhanced'
            else:
                result['ocr_method'] = 'standard'
                
        except Exception as e:
            logger.warning(f"Enhanced OCR failed for {screenshot_path}: {e}")
            result['ocr_method'] = 'standard'
        
        return result
    
    def extract_image_features(self, image_path: Path) -> Dict:
        """
        Extract visual features from image for embedding
        
        Args:
            image_path: Path to image
            
        Returns:
            Dict containing visual features
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Basic color analysis
            img_array = np.array(image)
            
            # Color histogram features
            hist_r = np.histogram(img_array[:,:,0], bins=32)[0]
            hist_g = np.histogram(img_array[:,:,1], bins=32)[0]
            hist_b = np.histogram(img_array[:,:,2], bins=32)[0]
            
            # Dominant colors
            pixels = img_array.reshape(-1, 3)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
            dominant_color_idx = np.argmax(counts)
            dominant_color = unique_colors[dominant_color_idx].tolist()
            
            return {
                'color_histogram': {
                    'red': hist_r.tolist(),
                    'green': hist_g.tolist(),
                    'blue': hist_b.tolist()
                },
                'dominant_color': dominant_color,
                'average_color': np.mean(pixels, axis=0).tolist(),
                'color_variance': np.var(pixels, axis=0).tolist()
            }
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return {}
    
    def batch_process(self, image_paths: List[Path]) -> List[Dict]:
        """Process multiple images"""
        results = []
        for image_path in image_paths:
            try:
                result = self.process_image(image_path)
                results.append(result)
                logger.info(f"Successfully processed image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                continue
        
        return results