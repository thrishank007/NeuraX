"""
Citation generator for source traceability and expandable links
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import re
import hashlib
from datetime import datetime
from pathlib import Path


@dataclass
class Citation:
    """Individual citation with metadata"""
    citation_id: int
    source_document: str
    source_type: str
    content_snippet: str
    confidence_score: float
    expandable_link: str
    file_path: str
    page_number: Optional[int] = None
    timestamp: str = ""


class CitationGenerator:
    """Creates verifiable citations linking generated content to sources"""
    
    def __init__(self):
        self.citation_counter = 0
        self.citation_cache = {}
        logger.info("Citation generator initialized")
    
    def generate_citations(self, response: str, sources: List[Dict], 
                          citation_indices: Optional[List[int]] = None) -> List[Citation]:
        """
        Generate numbered citations for a response based on source documents
        
        Args:
            response: Generated response text
            sources: List of source documents used
            citation_indices: Optional list of source indices to cite
            
        Returns:
            List of Citation objects
        """
        try:
            citations = []
            
            # Use provided indices or cite all sources
            indices_to_cite = citation_indices if citation_indices is not None else list(range(len(sources)))
            
            for i, source_idx in enumerate(indices_to_cite):
                if source_idx < len(sources):
                    source = sources[source_idx]
                    citation = self._create_citation(i + 1, source, response)
                    citations.append(citation)
            
            logger.info(f"Generated {len(citations)} citations")
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            return []
    
    def _create_citation(self, citation_number: int, source: Dict, response: str) -> Citation:
        """Create a single citation from source document"""
        try:
            metadata = source.get('metadata', {})
            
            # Extract source information
            file_path = metadata.get('file_path', 'Unknown source')
            source_type = metadata.get('file_type', 'document')
            
            # Generate content snippet
            content_snippet = self._extract_content_snippet(source, response)
            
            # Calculate confidence score
            confidence_score = self._calculate_citation_confidence(source, response)
            
            # Generate expandable link
            expandable_link = self._generate_expandable_link(source)
            
            # Extract page number if available
            page_number = self._extract_page_number(source)
            
            citation = Citation(
                citation_id=citation_number,
                source_document=Path(file_path).name if file_path else f"Document {citation_number}",
                source_type=source_type,
                content_snippet=content_snippet,
                confidence_score=confidence_score,
                expandable_link=expandable_link,
                file_path=file_path,
                page_number=page_number,
                timestamp=datetime.now().isoformat()
            )
            
            return citation
            
        except Exception as e:
            logger.error(f"Error creating citation: {e}")
            return self._create_fallback_citation(citation_number, source)
    
    def _extract_content_snippet(self, source: Dict, response: str, max_length: int = 150) -> str:
        """Extract relevant content snippet from source"""
        try:
            # Get source content
            source_content = ""
            
            if 'content_preview' in source:
                source_content = source['content_preview']
            elif 'content' in source:
                if isinstance(source['content'], list):
                    # Extract text from content list
                    content_parts = []
                    for item in source['content']:
                        if isinstance(item, dict) and 'text' in item:
                            content_parts.append(item['text'])
                    source_content = ' '.join(content_parts)
                else:
                    source_content = str(source['content'])
            
            if not source_content:
                return "Content not available"
            
            # Find most relevant snippet based on response overlap
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            
            # Split content into sentences
            sentences = re.split(r'[.!?]+', source_content)
            
            best_sentence = ""
            best_overlap = 0
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(response_words.intersection(sentence_words))
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = sentence.strip()
            
            # Use best sentence or first meaningful sentence
            snippet = best_sentence if best_sentence else sentences[0].strip()
            
            # Truncate if too long
            if len(snippet) > max_length:
                snippet = snippet[:max_length] + "..."
            
            return snippet
            
        except Exception as e:
            logger.error(f"Error extracting content snippet: {e}")
            return "Content snippet unavailable"
    
    def _calculate_citation_confidence(self, source: Dict, response: str) -> float:
        """Calculate confidence score for citation accuracy"""
        try:
            # Base confidence on similarity score if available
            base_confidence = source.get('similarity_score', 0.5)
            
            # Adjust based on content overlap
            source_content = source.get('content_preview', '')
            if not source_content and 'content' in source:
                if isinstance(source['content'], list):
                    source_content = ' '.join([item.get('text', '') for item in source['content']])
            
            if source_content:
                # Calculate word overlap
                response_words = set(re.findall(r'\b\w{3,}\b', response.lower()))
                source_words = set(re.findall(r'\b\w{3,}\b', source_content.lower()))
                
                if response_words:
                    overlap_ratio = len(response_words.intersection(source_words)) / len(response_words)
                    confidence_adjustment = overlap_ratio * 0.3  # Up to 30% adjustment
                    base_confidence = min(1.0, base_confidence + confidence_adjustment)
            
            return round(base_confidence, 2)
            
        except Exception as e:
            logger.error(f"Error calculating citation confidence: {e}")
            return 0.5
    
    def _generate_expandable_link(self, source: Dict) -> str:
        """Generate expandable link identifier for UI"""
        try:
            # Create unique identifier for the source
            file_path = source.get('metadata', {}).get('file_path', '')
            doc_id = source.get('document_id', source.get('id', ''))
            
            # Generate hash for unique link ID
            link_content = f"{file_path}_{doc_id}_{datetime.now().timestamp()}"
            link_hash = hashlib.md5(link_content.encode()).hexdigest()[:8]
            
            return f"expand_{link_hash}"
            
        except Exception as e:
            logger.error(f"Error generating expandable link: {e}")
            return f"expand_{self.citation_counter}"
    
    def _extract_page_number(self, source: Dict) -> Optional[int]:
        """Extract page number if available from source"""
        try:
            # Check if page information is in content
            if 'content' in source and isinstance(source['content'], list):
                for item in source['content']:
                    if isinstance(item, dict) and 'page' in item:
                        return item['page']
            
            # Check metadata
            metadata = source.get('metadata', {})
            if 'page' in metadata:
                return metadata['page']
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting page number: {e}")
            return None
    
    def _create_fallback_citation(self, citation_number: int, source: Dict) -> Citation:
        """Create fallback citation when normal creation fails"""
        return Citation(
            citation_id=citation_number,
            source_document=f"Document {citation_number}",
            source_type="unknown",
            content_snippet="Citation content unavailable",
            confidence_score=0.3,
            expandable_link=f"expand_fallback_{citation_number}",
            file_path="",
            timestamp=datetime.now().isoformat()
        )
    
    def create_expandable_links(self, citations: List[Citation]) -> List[Dict]:
        """
        Create expandable link data for UI integration
        
        Args:
            citations: List of Citation objects
            
        Returns:
            List of expandable link dictionaries
        """
        try:
            expandable_links = []
            
            for citation in citations:
                link_data = {
                    'citation_id': citation.citation_id,
                    'link_id': citation.expandable_link,
                    'source_document': citation.source_document,
                    'file_path': citation.file_path,
                    'content_snippet': citation.content_snippet,
                    'full_content_available': bool(citation.file_path),
                    'source_type': citation.source_type,
                    'confidence_score': citation.confidence_score,
                    'page_number': citation.page_number
                }
                
                expandable_links.append(link_data)
            
            return expandable_links
            
        except Exception as e:
            logger.error(f"Error creating expandable links: {e}")
            return []
    
    def validate_citation_accuracy(self, citation: Citation, source: Dict) -> bool:
        """
        Validate citation accuracy against source
        
        Args:
            citation: Citation to validate
            source: Original source document
            
        Returns:
            True if citation is accurate, False otherwise
        """
        try:
            # Check if file path matches
            source_path = source.get('metadata', {}).get('file_path', '')
            if citation.file_path != source_path:
                logger.warning(f"Citation file path mismatch: {citation.file_path} vs {source_path}")
                return False
            
            # Check if content snippet exists in source
            source_content = ""
            if 'content_preview' in source:
                source_content = source['content_preview']
            elif 'content' in source and isinstance(source['content'], list):
                source_content = ' '.join([item.get('text', '') for item in source['content']])
            
            if source_content and citation.content_snippet not in source_content:
                # Allow for partial matches due to truncation
                snippet_words = set(re.findall(r'\b\w+\b', citation.content_snippet.lower()))
                source_words = set(re.findall(r'\b\w+\b', source_content.lower()))
                
                overlap_ratio = len(snippet_words.intersection(source_words)) / len(snippet_words) if snippet_words else 0
                
                if overlap_ratio < 0.5:  # Less than 50% overlap
                    logger.warning(f"Citation content mismatch for citation {citation.citation_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating citation accuracy: {e}")
            return False
    
    def format_citations_for_display(self, citations: List[Citation]) -> str:
        """
        Format citations for display in UI
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citation text
        """
        try:
            if not citations:
                return ""
            
            formatted_citations = []
            
            for citation in citations:
                # Format: [1] Document.pdf (Page 5): "Content snippet..." [Expand]
                citation_text = f"[{citation.citation_id}] {citation.source_document}"
                
                if citation.page_number:
                    citation_text += f" (Page {citation.page_number})"
                
                citation_text += f': "{citation.content_snippet}"'
                
                if citation.confidence_score < 0.7:
                    citation_text += f" (Confidence: {citation.confidence_score:.1f})"
                
                formatted_citations.append(citation_text)
            
            return "\n".join(formatted_citations)
            
        except Exception as e:
            logger.error(f"Error formatting citations: {e}")
            return "Citations unavailable"
    
    def get_citation_statistics(self, citations: List[Citation]) -> Dict:
        """Get statistics about citations"""
        try:
            if not citations:
                return {'total': 0}
            
            stats = {
                'total': len(citations),
                'by_type': {},
                'avg_confidence': 0.0,
                'high_confidence': 0,
                'with_page_numbers': 0
            }
            
            total_confidence = 0
            
            for citation in citations:
                # Count by type
                source_type = citation.source_type
                stats['by_type'][source_type] = stats['by_type'].get(source_type, 0) + 1
                
                # Confidence statistics
                total_confidence += citation.confidence_score
                if citation.confidence_score >= 0.8:
                    stats['high_confidence'] += 1
                
                # Page number availability
                if citation.page_number:
                    stats['with_page_numbers'] += 1
            
            stats['avg_confidence'] = round(total_confidence / len(citations), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating citation statistics: {e}")
            return {'total': 0, 'error': str(e)}