"""
Notes processor for free-form text ingestion
"""
from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime
import re
import hashlib


class NotesProcessor:
    """Handles processing of free-form text notes"""
    
    def __init__(self):
        self.supported_types = {'note', 'text'}
    
    def process_note(self, text_content: str, source: Optional[str] = None, 
                    context: Optional[Dict] = None) -> Dict:
        """
        Process a text note and prepare it for embedding
        
        Args:
            text_content: The raw text content
            source: Optional source identifier
            context: Optional context metadata
            
        Returns:
            Dict containing processed note data
        """
        if not text_content or not text_content.strip():
            raise ValueError("Note content cannot be empty")
        
        try:
            # Clean and validate text
            cleaned_text = self._clean_text(text_content)
            
            # Generate unique ID for the note
            note_id = self._generate_note_id(cleaned_text, source)
            
            # Extract basic statistics
            word_count = len(cleaned_text.split())
            char_count = len(cleaned_text)
            line_count = len(cleaned_text.split('\n'))
            
            # Extract potential entities (simple keyword extraction)
            keywords = self._extract_keywords(cleaned_text)
            
            return {
                'note_id': note_id,
                'file_path': f"note_{note_id}",
                'file_type': 'note',
                'content': [{
                    'text': cleaned_text,
                    'source': source or 'direct_input'
                }],
                'metadata': {
                    'source': source or 'direct_input',
                    'timestamp': datetime.now().isoformat(),
                    'word_count': word_count,
                    'char_count': char_count,
                    'line_count': line_count,
                    'keywords': keywords,
                    'context': context or {}
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing note: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and sanitize text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters but keep newlines and tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def _generate_note_id(self, text: str, source: Optional[str] = None) -> str:
        """Generate a unique ID for the note"""
        # Create hash from content and timestamp
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        timestamp_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:4]
        
        if source:
            source_hash = hashlib.md5(source.encode()).hexdigest()[:4]
            return f"{source_hash}_{content_hash}_{timestamp_hash}"
        else:
            return f"note_{content_hash}_{timestamp_hash}"
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract potential keywords from text"""
        try:
            # Simple keyword extraction - remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
            }
            
            # Extract words (alphanumeric, 3+ characters)
            words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower())
            
            # Filter out stop words and count frequency
            word_freq = {}
            for word in words:
                if word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:max_keywords]]
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            return []
    
    def validate_note_content(self, text_content: str) -> bool:
        """
        Validate note content for basic requirements
        
        Args:
            text_content: Text to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not text_content:
            return False
        
        # Check minimum length
        if len(text_content.strip()) < 3:
            return False
        
        # Check for reasonable character distribution
        if len(set(text_content.strip())) < 2:
            return False
        
        return True
    
    def batch_process_notes(self, notes_data: List[Dict]) -> List[Dict]:
        """
        Process multiple notes in batch
        
        Args:
            notes_data: List of dicts with 'text', optional 'source' and 'context'
            
        Returns:
            List of processed note documents
        """
        results = []
        
        for i, note_data in enumerate(notes_data):
            try:
                text = note_data.get('text', '')
                source = note_data.get('source', f'batch_note_{i}')
                context = note_data.get('context', {})
                
                if self.validate_note_content(text):
                    result = self.process_note(text, source, context)
                    results.append(result)
                    logger.info(f"Successfully processed note from source: {source}")
                else:
                    logger.warning(f"Invalid note content from source: {source}")
                    
            except Exception as e:
                logger.error(f"Failed to process note {i}: {e}")
                continue
        
        return results
    
    def extract_note_metadata(self, note_data: Dict) -> Dict:
        """
        Extract additional metadata from processed note
        
        Args:
            note_data: Processed note data
            
        Returns:
            Enhanced metadata dict
        """
        try:
            metadata = note_data.get('metadata', {}).copy()
            text = note_data.get('content', [{}])[0].get('text', '')
            
            # Add text analysis
            sentences = text.split('.')
            metadata['sentence_count'] = len([s for s in sentences if s.strip()])
            
            # Check for potential structured content
            has_bullets = bool(re.search(r'[•\-\*]\s', text))
            has_numbers = bool(re.search(r'\d+[\.\)]\s', text))
            has_urls = bool(re.search(r'https?://', text))
            has_emails = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
            
            metadata.update({
                'has_bullet_points': has_bullets,
                'has_numbered_list': has_numbers,
                'has_urls': has_urls,
                'has_emails': has_emails,
                'is_structured': has_bullets or has_numbers
            })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract enhanced metadata: {e}")
            return note_data.get('metadata', {})