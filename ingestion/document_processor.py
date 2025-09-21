"""
Document processing module for PDFs and DOCs
"""
import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import pytesseract
from PIL import Image
import io


class DocumentProcessor:
    """Handles extraction of text from PDF and DOC files"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.doc'}
    
    def process_file(self, file_path: Path) -> Dict:
        """
        Process a document file and extract text content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing extracted text and metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._process_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            return self._process_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _process_pdf(self, file_path: Path) -> Dict:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(file_path)
            text_content = []
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_content.append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })
                
                # Extract images for OCR if text is sparse
                if len(text.strip()) < 50:  # Likely image-heavy page
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        text_content.append({
                            'page': page_num + 1,
                            'text': ocr_text.strip(),
                            'source': 'ocr'
                        })
            
            doc.close()
            
            return {
                'file_path': str(file_path),
                'file_type': 'pdf',
                'content': text_content,
                'total_pages': len(doc),
                'metadata': {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'subject': doc.metadata.get('subject', '')
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def _process_docx(self, file_path: Path) -> Dict:
        """Extract text from DOCX files"""
        try:
            doc = Document(file_path)
            text_content = []
            
            for i, paragraph in enumerate(doc.paragraphs):
                if paragraph.text.strip():
                    text_content.append({
                        'paragraph': i + 1,
                        'text': paragraph.text.strip()
                    })
            
            # Extract text from tables
            for table_idx, table in enumerate(doc.tables):
                table_text = []
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    table_text.append(' | '.join(row_text))
                
                if table_text:
                    text_content.append({
                        'table': table_idx + 1,
                        'text': '\n'.join(table_text)
                    })
            
            return {
                'file_path': str(file_path),
                'file_type': 'docx',
                'content': text_content,
                'metadata': {
                    'title': doc.core_properties.title or '',
                    'author': doc.core_properties.author or '',
                    'subject': doc.core_properties.subject or ''
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            raise
    
    def batch_process(self, file_paths: List[Path]) -> List[Dict]:
        """Process multiple documents"""
        results = []
        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                results.append(result)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        return results