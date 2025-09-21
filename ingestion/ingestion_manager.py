"""
Ingestion manager to orchestrate all document processors
"""
from pathlib import Path
from typing import List, Dict, Optional, Union
from loguru import logger
import mimetypes
from tqdm import tqdm

from .document_processor import DocumentProcessor
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .notes_processor import NotesProcessor


class IngestionManager:
    """Orchestrates the processing of different file types"""
    
    def __init__(self):
        # Initialize processors
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.notes_processor = NotesProcessor()
        
        # File type mappings
        self.file_type_map = {
            # Document types
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.txt': 'document',
            
            # Image types
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.webp': 'image',
            
            # Audio types
            '.wav': 'audio',
            '.mp3': 'audio',
            '.m4a': 'audio',
            '.flac': 'audio',
            '.ogg': 'audio'
        }
        
        logger.info("Ingestion manager initialized with all processors")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of all supported file formats"""
        return list(self.file_type_map.keys())
    
    def detect_file_type(self, file_path: Path) -> str:
        """
        Detect the type of file for processing
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type category ('document', 'image', 'audio', 'unknown')
        """
        if not file_path.exists():
            return 'unknown'
        
        suffix = file_path.suffix.lower()
        
        # Check our mapping first
        if suffix in self.file_type_map:
            return self.file_type_map[suffix]
        
        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('text/'):
                return 'document'
            elif mime_type.startswith('image/'):
                return 'image'
            elif mime_type.startswith('audio/'):
                return 'audio'
        
        return 'unknown'
    
    def process_file(self, file_path: Union[str, Path]) -> Optional[Dict]:
        """
        Process a single file using the appropriate processor
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed document dict or None if failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        file_type = self.detect_file_type(file_path)
        
        try:
            logger.info(f"Processing {file_type} file: {file_path}")
            
            if file_type == 'document':
                return self.document_processor.process_file(file_path)
            elif file_type == 'image':
                return self.image_processor.process_image(file_path)
            elif file_type == 'audio':
                return self.audio_processor.process_audio(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type} for {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return None
    
    def process_batch(self, file_paths: List[Union[str, Path]], 
                     show_progress: bool = True) -> List[Dict]:
        """
        Process multiple files in batch with progress tracking
        
        Args:
            file_paths: List of file paths to process
            show_progress: Whether to show progress bar
            
        Returns:
            List of successfully processed documents
        """
        results = []
        failed_files = []
        
        # Convert to Path objects
        paths = [Path(p) for p in file_paths]
        
        # Group files by type for efficient batch processing
        files_by_type = {
            'document': [],
            'image': [],
            'audio': []
        }
        
        for path in paths:
            file_type = self.detect_file_type(path)
            if file_type in files_by_type:
                files_by_type[file_type].append(path)
        
        # Process each type in batch
        progress_bar = tqdm(total=len(paths), desc="Processing files") if show_progress else None
        
        try:
            # Process documents
            if files_by_type['document']:
                try:
                    doc_results = self.document_processor.batch_process(files_by_type['document'])
                    results.extend(doc_results)
                    if progress_bar:
                        progress_bar.update(len(files_by_type['document']))
                except Exception as e:
                    logger.error(f"Batch document processing failed: {e}")
                    failed_files.extend(files_by_type['document'])
            
            # Process images
            if files_by_type['image']:
                try:
                    img_results = self.image_processor.batch_process(files_by_type['image'])
                    results.extend(img_results)
                    if progress_bar:
                        progress_bar.update(len(files_by_type['image']))
                except Exception as e:
                    logger.error(f"Batch image processing failed: {e}")
                    failed_files.extend(files_by_type['image'])
            
            # Process audio files
            if files_by_type['audio']:
                try:
                    audio_results = self.audio_processor.batch_process(files_by_type['audio'])
                    results.extend(audio_results)
                    if progress_bar:
                        progress_bar.update(len(files_by_type['audio']))
                except Exception as e:
                    logger.error(f"Batch audio processing failed: {e}")
                    failed_files.extend(files_by_type['audio'])
            
        finally:
            if progress_bar:
                progress_bar.close()
        
        # Log summary
        logger.info(f"Batch processing complete: {len(results)} successful, {len(failed_files)} failed")
        if failed_files:
            logger.warning(f"Failed files: {[str(f) for f in failed_files]}")
        
        return results
    
    def process_note(self, text_content: str, source: Optional[str] = None, 
                    context: Optional[Dict] = None) -> Optional[Dict]:
        """
        Process a text note using the notes processor
        
        Args:
            text_content: The text content to process
            source: Optional source identifier
            context: Optional context metadata
            
        Returns:
            Processed note document or None if failed
        """
        try:
            return self.notes_processor.process_note(text_content, source, context)
        except Exception as e:
            logger.error(f"Failed to process note: {e}")
            return None
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True, 
                         show_progress: bool = True) -> List[Dict]:
        """
        Process all supported files in a directory
        
        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            show_progress: Whether to show progress bar
            
        Returns:
            List of processed documents
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Find all supported files
        supported_extensions = set(self.get_supported_formats())
        files_to_process = []
        
        if recursive:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    files_to_process.append(file_path)
        else:
            for file_path in directory_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    files_to_process.append(file_path)
        
        logger.info(f"Found {len(files_to_process)} supported files in {directory_path}")
        
        if not files_to_process:
            logger.warning("No supported files found in directory")
            return []
        
        return self.process_batch(files_to_process, show_progress)
    
    def get_processing_stats(self, results: List[Dict]) -> Dict:
        """
        Get statistics about processed documents
        
        Args:
            results: List of processed documents
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {'total': 0}
        
        stats = {
            'total': len(results),
            'by_type': {},
            'total_size': 0,
            'processing_errors': 0
        }
        
        for doc in results:
            file_type = doc.get('file_type', 'unknown')
            stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1
            
            # Add file size if available
            if 'metadata' in doc and 'file_size' in doc['metadata']:
                stats['total_size'] += doc['metadata']['file_size']
        
        return stats
    
    def validate_file_before_processing(self, file_path: Path) -> bool:
        """
        Validate file before processing
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file can be processed, False otherwise
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return False
            
            if not file_path.is_file():
                return False
            
            # Check file size (avoid very large files that might cause issues)
            file_size = file_path.stat().st_size
            max_size = 100 * 1024 * 1024  # 100MB limit
            
            if file_size > max_size:
                logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return False
            
            # Check if file type is supported
            file_type = self.detect_file_type(file_path)
            if file_type == 'unknown':
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False