"""
Gradio-based chat interface for SecureInsight multimodal RAG system
"""
import gradio as gr
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from loguru import logger
import time
import json
import numpy as np
from datetime import datetime

# Import system components
from ingestion.ingestion_manager import IngestionManager
from indexing.embedding_manager import EmbeddingManager
from indexing.vector_store import VectorStore
from retrieval.query_processor import QueryProcessor
from retrieval.speech_to_text_processor import SpeechToTextProcessor
from generation.llm_factory import create_llm_generator
from generation.citation_generator import CitationGenerator
from feedback.feedback_system import FeedbackSystem
from config import (
    GRADIO_CONFIG, PROCESSING_CONFIG, SECURITY_CONFIG, 
    FEEDBACK_CONFIG, ERROR_CONFIG, SEARCH_CONFIG, LLM_CONFIG
)


class SecureInsightGradioApp:
    """Main Gradio application for SecureInsight"""
    
    def __init__(self):
        """Initialize the Gradio application"""
        self.ingestion_manager = IngestionManager()
        self.embedding_manager = None  # Will be initialized lazily
        self.vector_store = None  # Will be initialized lazily
        self.query_processor = None  # Will be initialized lazily
        self.stt_processor = None  # Will be initialized lazily
        self.llm_generator = None  # Will be initialized lazily
        self.citation_generator = None  # Will be initialized lazily
        self.feedback_system = None  # Will be initialized lazily
        
        # Track processing state
        self.processing_state = {
            'is_processing': False,
            'current_files': [],
            'processed_count': 0,
            'total_count': 0,
            'errors': []
        }
        
        # Track query state
        self.query_state = {
            'is_querying': False,
            'last_query': '',
            'last_results': [],
            'query_history': []
        }
        
        # Supported file formats from config
        self.supported_formats = PROCESSING_CONFIG['supported_document_formats'] + \
                               PROCESSING_CONFIG['supported_image_formats'] + \
                               PROCESSING_CONFIG['supported_audio_formats']
        
        logger.info("SecureInsight Gradio app initialized")
    
    def _initialize_components(self):
        """Lazy initialization of heavy components"""
        try:
            if self.embedding_manager is None:
                logger.info("Initializing embedding manager...")
                self.embedding_manager = EmbeddingManager()
            
            if self.vector_store is None:
                logger.info("Initializing vector store...")
                from config import CHROMA_CONFIG
                self.vector_store = VectorStore(
                    persist_directory=CHROMA_CONFIG['persist_directory'],
                    collection_name=CHROMA_CONFIG['collection_name']
                )
            
            if self.query_processor is None and self.embedding_manager and self.vector_store:
                logger.info("Initializing query processor...")
                self.query_processor = QueryProcessor(
                    self.embedding_manager, 
                    self.vector_store, 
                    SEARCH_CONFIG
                )
            
            if self.stt_processor is None:
                logger.info("Initializing speech-to-text processor...")
                self.stt_processor = SpeechToTextProcessor()
            
            if self.llm_generator is None:
                logger.info("Initializing LLM generator...")
                self.llm_generator = create_llm_generator(LLM_CONFIG)
                # Log model info
                model_info = self.llm_generator.get_model_info()
                if 'current_model' in model_info:
                    logger.info(f"Using model: {model_info['current_model']}")
                    if model_info.get('supports_multimodal', False):
                        logger.info("Multimodal capabilities enabled")
            
            if self.citation_generator is None:
                logger.info("Initializing citation generator...")
                self.citation_generator = CitationGenerator()
            
            if self.feedback_system is None:
                logger.info("Initializing feedback system...")
                self.feedback_system = FeedbackSystem()
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def validate_uploaded_files(self, files: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Validate uploaded files and return valid and invalid file lists
        
        Args:
            files: List of uploaded file objects from Gradio
            
        Returns:
            Tuple of (valid_files, error_messages)
        """
        valid_files = []
        error_messages = []
        
        if not files:
            return valid_files, ["No files uploaded"]
        
        for file_obj in files:
            try:
                # Get file path from Gradio file object
                if hasattr(file_obj, 'name'):
                    file_path = Path(file_obj.name)
                else:
                    file_path = Path(str(file_obj))
                
                # Check if file exists
                if not file_path.exists():
                    error_messages.append(f"File not found: {file_path.name}")
                    continue
                
                # Check file extension
                file_ext = file_path.suffix.lower()
                if file_ext not in self.supported_formats:
                    error_messages.append(
                        f"Unsupported format: {file_path.name} ({file_ext}). "
                        f"Supported formats: {', '.join(self.supported_formats)}"
                    )
                    continue
                
                # Check file size
                file_size = file_path.stat().st_size
                max_size = SECURITY_CONFIG['max_upload_size_mb'] * 1024 * 1024
                
                if file_size > max_size:
                    error_messages.append(
                        f"File too large: {file_path.name} "
                        f"({file_size / (1024*1024):.1f}MB > {SECURITY_CONFIG['max_upload_size_mb']}MB)"
                    )
                    continue
                
                # File is valid
                valid_files.append(str(file_path))
                
            except Exception as e:
                error_messages.append(f"Error validating file: {e}")
        
        return valid_files, error_messages
    
    def process_uploaded_files(self, files: List[Any], progress=gr.Progress()) -> Tuple[str, str, str]:
        """
        Process uploaded files and return results
        
        Args:
            files: List of uploaded file objects from Gradio
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (status_message, processing_log, file_list_html)
        """
        if self.processing_state['is_processing']:
            return (
                "⚠️ Processing already in progress. Please wait...",
                "Processing in progress...",
                ""
            )
        
        try:
            # Initialize components if needed
            self._initialize_components()
            
            # Validate files
            valid_files, validation_errors = self.validate_uploaded_files(files)
            
            if validation_errors and not valid_files:
                error_msg = "❌ No valid files to process:\n" + "\n".join(validation_errors)
                return error_msg, "\n".join(validation_errors), ""
            
            # Update processing state
            self.processing_state.update({
                'is_processing': True,
                'current_files': valid_files,
                'processed_count': 0,
                'total_count': len(valid_files),
                'errors': validation_errors.copy()
            })
            
            # Process files with progress tracking
            processed_docs = []
            processing_log = []
            
            if validation_errors:
                processing_log.extend([f"⚠️ {error}" for error in validation_errors])
            
            progress(0, desc="Starting file processing...")
            
            for i, file_path in enumerate(valid_files):
                try:
                    progress((i + 1) / len(valid_files), desc=f"Processing {Path(file_path).name}...")
                    
                    # Process individual file
                    start_time = time.time()
                    result = self.ingestion_manager.process_file(file_path)
                    processing_time = time.time() - start_time
                    
                    if result:
                        processed_docs.append(result)
                        self.processing_state['processed_count'] += 1
                        
                        processing_log.append(
                            f"✅ {Path(file_path).name} - "
                            f"{result.get('file_type', 'unknown')} - "
                            f"{processing_time:.2f}s"
                        )
                        
                        # Generate embeddings and store
                        if self.embedding_manager and self.vector_store:
                            try:
                                # Generate embeddings based on content type
                                if result.get('file_type') == 'image':
                                    # For images, use both OCR text and image embeddings
                                    if result.get('ocr_text'):
                                        text_embedding = self.embedding_manager.embed_text(result['ocr_text'])
                                        if text_embedding.size:
                                            result['text_embedding'] = text_embedding[0]
                                            if not result.get('embedding_type'):
                                                result['embedding_type'] = 'text'


                                    # Image embedding would be handled here
                                    # image_embedding = self.embedding_manager.embed_image(image_path)
                                    
                                elif result.get('file_type') == 'audio':
                                    # For audio, use transcription
                                    if result.get('transcription'):
                                        text_embedding = self.embedding_manager.embed_text(result['transcription'])
                                        if text_embedding.size:
                                            result['text_embedding'] = text_embedding[0]
                                            if not result.get('embedding_type'):
                                                result['embedding_type'] = 'text'


                                else:
                                    # For documents, use extracted text
                                    if result.get('content'):
                                        text_embedding = self.embedding_manager.embed_text(result['content'])
                                        if text_embedding.size:
                                            result['text_embedding'] = text_embedding[0]
                                            if not result.get('embedding_type'):
                                                result['embedding_type'] = 'text'


                                # Store in vector database
                                # Format result for the new vector store format
                                documents_to_add = [result]
                                embeddings_to_add = np.array([result['text_embedding']])
                                self.vector_store.add_documents(documents_to_add, embeddings_to_add)
                                processing_log.append(f"  📊 Embeddings generated and stored")
                                
                            except Exception as e:
                                processing_log.append(f"  ⚠️ Embedding generation failed: {e}")
                    else:
                        self.processing_state['errors'].append(f"Failed to process {Path(file_path).name}")
                        processing_log.append(f"❌ {Path(file_path).name} - Processing failed")
                
                except Exception as e:
                    error_msg = f"Error processing {Path(file_path).name}: {e}"
                    self.processing_state['errors'].append(error_msg)
                    processing_log.append(f"❌ {error_msg}")
            
            # Generate summary
            success_count = len(processed_docs)
            total_files = len(valid_files)
            error_count = total_files - success_count
            
            if success_count > 0:
                status_msg = f"✅ Processing complete: {success_count}/{total_files} files processed successfully"
                if error_count > 0:
                    status_msg += f" ({error_count} errors)"
            else:
                status_msg = f"❌ Processing failed: No files processed successfully"
            
            # Generate file list HTML
            file_list_html = self._generate_file_list_html(processed_docs)
            
            return status_msg, "\n".join(processing_log), file_list_html
            
        except Exception as e:
            error_msg = f"❌ Processing error: {e}"
            logger.error(f"File processing error: {e}")
            return error_msg, str(e), ""
        
        finally:
            # Reset processing state
            self.processing_state['is_processing'] = False
    
    def _generate_file_list_html(self, processed_docs: List[Dict]) -> str:
        """Generate HTML for displaying processed files"""
        if not processed_docs:
            return "<p>No files processed yet.</p>"
        
        html = "<div style='max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;'>"
        html += "<h4>📁 Processed Files</h4>"
        
        for doc in processed_docs:
            file_name = Path(doc.get('file_path', 'Unknown')).name
            file_type = doc.get('file_type', 'unknown')
            timestamp = doc.get('timestamp', 'Unknown')
            
            # File type icon
            type_icons = {
                'document': '📄',
                'image': '🖼️',
                'audio': '🎵',
                'unknown': '❓'
            }
            icon = type_icons.get(file_type, '❓')
            
            # Content preview
            content_preview = ""
            if doc.get('content'):
                preview_text = str(doc['content'])[:100]
                if len(str(doc['content'])) > 100:
                    preview_text += "..."
                content_preview = f"<br><small style='color: #666;'>{preview_text}</small>"
            
            html += f"""
            <div style='margin: 5px 0; padding: 8px; background: #f9f9f9; border-radius: 3px;'>
                <strong>{icon} {file_name}</strong> 
                <span style='color: #666; font-size: 0.9em;'>({file_type})</span>
                <br><small style='color: #888;'>Processed: {timestamp}</small>
                {content_preview}
            </div>
            """
        
        html += "</div>"
        return html
    
    def clear_processing_state(self) -> Tuple[str, str, str]:
        """Clear the processing state and UI"""
        self.processing_state = {
            'is_processing': False,
            'current_files': [],
            'processed_count': 0,
            'total_count': 0,
            'errors': []
        }
        
        return (
            "🔄 Ready for new files",
            "Processing state cleared",
            "<p>No files processed yet.</p>"
        )
    
    def process_text_query(self, query_text: str, similarity_threshold: float = None) -> Tuple[str, str]:
        """
        Process a text query and return formatted results
        
        Args:
            query_text: The text query to search for
            similarity_threshold: Optional similarity threshold override
            
        Returns:
            Tuple of (status_message, results_html)
        """
        if not query_text or not query_text.strip():
            return "❌ Please enter a search query", ""
        
        if self.query_state['is_querying']:
            return "⚠️ Query already in progress. Please wait...", ""
        
        try:
            # Initialize components if needed
            self._initialize_components()
            
            if not self.query_processor:
                return "❌ Query processor not available. Please process some files first.", ""
            
            # Update query state
            self.query_state['is_querying'] = True
            self.query_state['last_query'] = query_text
            
            # Update similarity threshold if provided
            if similarity_threshold is not None:
                self.query_processor.update_similarity_threshold(similarity_threshold)
            
            # Process the query
            start_time = time.time()
            query_result = self.query_processor.process_text_query(query_text)
            processing_time = time.time() - start_time
            
            # Store results
            self.query_state['last_results'] = query_result.results
            self.query_state['query_history'].append({
                'query': query_text,
                'type': 'text',
                'timestamp': datetime.now().isoformat(),
                'results_count': len(query_result.results),
                'processing_time': processing_time
            })
            
            # Format results
            if query_result.results:
                status_msg = f"✅ Found {len(query_result.results)} results in {processing_time:.2f}s"
                results_html = self._format_query_results_html(query_result.results, query_text, 'text')
            else:
                status_msg = f"ℹ️ No results found for '{query_text}' (threshold: {query_result.similarity_threshold:.2f})"
                results_html = "<p>No results found. Try adjusting the similarity threshold or rephrasing your query.</p>"
            
            return status_msg, results_html
            
        except Exception as e:
            logger.error(f"Error processing text query: {e}")
            return f"❌ Query processing error: {e}", ""
        
        finally:
            self.query_state['is_querying'] = False
    
    def process_image_query(self, image_file, similarity_threshold: float = None) -> Tuple[str, str]:
        """
        Process an image query and return formatted results
        
        Args:
            image_file: Uploaded image file from Gradio
            similarity_threshold: Optional similarity threshold override
            
        Returns:
            Tuple of (status_message, results_html)
        """
        if image_file is None:
            return "❌ Please upload an image for search", ""
        
        if self.query_state['is_querying']:
            return "⚠️ Query already in progress. Please wait...", ""
        
        try:
            # Initialize components if needed
            self._initialize_components()
            
            if not self.query_processor:
                return "❌ Query processor not available. Please process some files first.", ""
            
            # Update query state
            self.query_state['is_querying'] = True
            self.query_state['last_query'] = "[Image Query]"
            
            # Update similarity threshold if provided
            if similarity_threshold is not None:
                self.query_processor.update_similarity_threshold(similarity_threshold)
            
            # Process the image query
            start_time = time.time()
            image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
            query_result = self.query_processor.process_image_query(image_path)
            processing_time = time.time() - start_time
            
            # Store results
            self.query_state['last_results'] = query_result.results
            self.query_state['query_history'].append({
                'query': '[Image Query]',
                'type': 'image',
                'timestamp': datetime.now().isoformat(),
                'results_count': len(query_result.results),
                'processing_time': processing_time
            })
            
            # Format results
            if query_result.results:
                status_msg = f"✅ Found {len(query_result.results)} results for image query in {processing_time:.2f}s"
                results_html = self._format_query_results_html(query_result.results, "[Image Query]", 'image')
            else:
                status_msg = f"ℹ️ No results found for image query (threshold: {query_result.similarity_threshold:.2f})"
                results_html = "<p>No results found. Try adjusting the similarity threshold or using a different image.</p>"
            
            return status_msg, results_html
            
        except Exception as e:
            logger.error(f"Error processing image query: {e}")
            return f"❌ Image query processing error: {e}", ""
        
        finally:
            self.query_state['is_querying'] = False
    
    def process_voice_query(self, audio_file, similarity_threshold: float = None) -> Tuple[str, str, str]:
        """
        Process a voice query and return formatted results
        
        Args:
            audio_file: Uploaded audio file from Gradio
            similarity_threshold: Optional similarity threshold override
            
        Returns:
            Tuple of (status_message, transcription_text, results_html)
        """
        if audio_file is None:
            return "❌ Please upload an audio file for voice search", "", ""
        
        if self.query_state['is_querying']:
            return "⚠️ Query already in progress. Please wait...", "", ""
        
        try:
            # Initialize components if needed
            self._initialize_components()
            
            if not self.query_processor or not self.stt_processor:
                return "❌ Voice query components not available. Please check system setup.", "", ""
            
            # Update query state
            self.query_state['is_querying'] = True
            
            # Process voice input with STT
            audio_path = audio_file.name if hasattr(audio_file, 'name') else str(audio_file)
            stt_result = self.stt_processor.process_voice_query_with_fallback(audio_path)
            
            if not stt_result['success']:
                fallback_msg = stt_result.get('fallback_message', 'Voice processing failed')
                return f"⚠️ {fallback_msg}", fallback_msg, ""
            
            transcribed_text = stt_result['transcribed_text']
            
            # Update similarity threshold if provided
            if similarity_threshold is not None:
                self.query_processor.update_similarity_threshold(similarity_threshold)
            
            # Process the transcribed text as a query
            start_time = time.time()
            query_result = self.query_processor.process_text_query(transcribed_text)
            processing_time = time.time() - start_time
            
            # Store results
            self.query_state['last_query'] = transcribed_text
            self.query_state['last_results'] = query_result.results
            self.query_state['query_history'].append({
                'query': transcribed_text,
                'type': 'voice',
                'timestamp': datetime.now().isoformat(),
                'results_count': len(query_result.results),
                'processing_time': processing_time,
                'audio_quality': stt_result.get('audio_quality', 'unknown')
            })
            
            # Format results
            if query_result.results:
                status_msg = f"✅ Found {len(query_result.results)} results for voice query in {processing_time:.2f}s"
                results_html = self._format_query_results_html(query_result.results, transcribed_text, 'voice')
            else:
                status_msg = f"ℹ️ No results found for '{transcribed_text}' (threshold: {query_result.similarity_threshold:.2f})"
                results_html = "<p>No results found. Try adjusting the similarity threshold or rephrasing your query.</p>"
            
            return status_msg, transcribed_text, results_html
            
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            return f"❌ Voice query processing error: {e}", "", ""
        
        finally:
            self.query_state['is_querying'] = False
    
    def process_multimodal_query(self, text_query: str, image_file, similarity_threshold: float = None) -> Tuple[str, str]:
        """
        Process a multimodal query combining text and image
        
        Args:
            text_query: Text component of the query
            image_file: Image component of the query
            similarity_threshold: Optional similarity threshold override
            
        Returns:
            Tuple of (status_message, results_html)
        """
        if not text_query or not text_query.strip():
            return "❌ Please enter text for multimodal search", ""
        
        if image_file is None:
            return "❌ Please upload an image for multimodal search", ""
        
        if self.query_state['is_querying']:
            return "⚠️ Query already in progress. Please wait...", ""
        
        try:
            # Initialize components if needed
            self._initialize_components()
            
            if not self.query_processor:
                return "❌ Query processor not available. Please process some files first.", ""
            
            # Update query state
            self.query_state['is_querying'] = True
            self.query_state['last_query'] = f"{text_query} + [Image]"
            
            # Update similarity threshold if provided
            if similarity_threshold is not None:
                self.query_processor.update_similarity_threshold(similarity_threshold)
            
            # Process the multimodal query
            start_time = time.time()
            image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
            query_result = self.query_processor.process_multimodal_query(text_query, image_path)
            processing_time = time.time() - start_time
            
            # Store results
            self.query_state['last_results'] = query_result.results
            self.query_state['query_history'].append({
                'query': f"{text_query} + [Image]",
                'type': 'multimodal',
                'timestamp': datetime.now().isoformat(),
                'results_count': len(query_result.results),
                'processing_time': processing_time
            })
            
            # Format results
            if query_result.results:
                status_msg = f"✅ Found {len(query_result.results)} results for multimodal query in {processing_time:.2f}s"
                results_html = self._format_query_results_html(query_result.results, f"{text_query} + [Image]", 'multimodal')
            else:
                status_msg = f"ℹ️ No results found for multimodal query (threshold: {query_result.similarity_threshold:.2f})"
                results_html = "<p>No results found. Try adjusting the similarity threshold or modifying your query.</p>"
            
            return status_msg, results_html
            
        except Exception as e:
            logger.error(f"Error processing multimodal query: {e}")
            return f"❌ Multimodal query processing error: {e}", ""
        
        finally:
            self.query_state['is_querying'] = False
    
    def _format_query_results_html(self, results: List[Dict], query: str, query_type: str) -> str:
        """Format query results as HTML for display"""
        if not results:
            return "<p>No results found.</p>"
        
        html = f"""
        <div style='max-height: 500px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: #fafafa;'>
            <h4>🔍 Search Results for: "{query}" ({query_type})</h4>
            <p style='color: #666; margin-bottom: 15px;'>Found {len(results)} results</p>
        """
        
        for i, result in enumerate(results, 1):
            similarity_score = result.get('similarity_score', 0.0)
            file_path = result.get('file_path', 'Unknown')
            file_name = Path(file_path).name if file_path != 'Unknown' else 'Unknown'
            file_type = result.get('file_type', 'unknown')
            content_preview = result.get('content_preview', 'No preview available')
            
            # Color code similarity scores
            if similarity_score >= 0.8:
                score_color = '#28a745'  # Green
                score_label = 'Excellent'
            elif similarity_score >= 0.6:
                score_color = '#ffc107'  # Yellow
                score_label = 'Good'
            else:
                score_color = '#dc3545'  # Red
                score_label = 'Fair'
            
            # File type icons
            type_icons = {
                'document': '📄',
                'image': '🖼️',
                'audio': '🎵',
                'unknown': '❓'
            }
            icon = type_icons.get(file_type, '❓')
            
            html += f"""
            <div style='margin: 10px 0; padding: 12px; background: white; border-radius: 6px; border-left: 4px solid {score_color};'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                    <strong>{icon} {file_name}</strong>
                    <span style='background: {score_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;'>
                        {similarity_score:.3f} ({score_label})
                    </span>
                </div>
                <div style='color: #666; font-size: 0.9em; margin-bottom: 6px;'>
                    Type: {file_type.title()} | Path: {file_path}
                </div>
                <div style='color: #333; font-size: 0.95em; line-height: 1.4;'>
                    {content_preview}
                </div>
            """
            
            # Add multimodal specific information
            if query_type == 'multimodal' and 'text_similarity' in result:
                text_sim = result.get('text_similarity', 0.0)
                image_sim = result.get('image_similarity', 0.0)
                html += f"""
                <div style='margin-top: 8px; font-size: 0.8em; color: #666;'>
                    Text similarity: {text_sim:.3f} | Image similarity: {image_sim:.3f}
                </div>
                """
            
            html += "</div>"
        
        html += "</div>"
        return html
    
    def clear_query_results(self) -> Tuple[str, str, str, str]:
        """Clear query results and reset interface"""
        self.query_state = {
            'is_querying': False,
            'last_query': '',
            'last_results': [],
            'query_history': []
        }
        
        return (
            "🔄 Ready for new queries",
            "",  # Clear text query
            "",  # Clear transcription
            "<p>No search results yet. Enter a query above to get started.</p>"  # Clear results
        )
    
    def get_query_history_html(self) -> str:
        """Get formatted query history"""
        if not self.query_state['query_history']:
            return "<p>No queries yet.</p>"
        
        html = "<div style='max-height: 200px; overflow-y: auto;'>"
        html += "<h5>📋 Recent Queries</h5>"
        
        for query_info in reversed(self.query_state['query_history'][-10:]):  # Last 10 queries
            query_type = query_info['type']
            query_text = query_info['query']
            results_count = query_info['results_count']
            timestamp = query_info['timestamp']
            processing_time = query_info.get('processing_time', 0.0)
            
            type_icons = {
                'text': '💬',
                'image': '🖼️',
                'voice': '🎤',
                'multimodal': '🔀'
            }
            icon = type_icons.get(query_type, '❓')
            
            html += f"""
            <div style='margin: 5px 0; padding: 6px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em;'>
                <strong>{icon} {query_text[:50]}{'...' if len(query_text) > 50 else ''}</strong>
                <br><small style='color: #666;'>
                    {results_count} results | {processing_time:.2f}s | {timestamp[:19]}
                </small>
            </div>
            """
        
        html += "</div>"
        return html
    
    def generate_response_with_citations(self, query: str, search_results: List[Dict]) -> Tuple[str, str, str, float]:
        """
        Generate a response with citations based on search results
        
        Args:
            query: The user's query
            search_results: List of search results to use as context
            
        Returns:
            Tuple of (response_text, citations_html, confidence_score, processing_time)
        """
        try:
            # Initialize components if needed
            self._initialize_components()
            
            if not self.llm_generator or not self.citation_generator:
                return "❌ Response generation not available", "", 0.0, 0.0
            
            if not search_results:
                return "ℹ️ No context available for response generation. Please search for relevant documents first.", "", 0.0, 0.0
            
            start_time = time.time()
            
            # Generate response using LLM
            generated_response = self.llm_generator.generate_grounded_response(
                query=query,
                context=search_results
            )
            
            # Generate citations
            # If the model didn't flag any specific sources, let the generator pick defaults
            citation_indices = generated_response.citations_needed if generated_response.citations_needed else None
            citations = self.citation_generator.generate_citations(
                response=generated_response.response_text,
                sources=search_results,
                citation_indices=citation_indices
            )
            
            processing_time = time.time() - start_time
            
            # Format citations as HTML
            citations_html = self._format_citations_html(citations)
            
            return (
                generated_response.response_text,
                citations_html,
                generated_response.confidence_score,
                processing_time
            )
            
        except Exception as e:
            logger.error(f"Error generating response with citations: {e}")
            return f"❌ Response generation error: {e}", "", 0.0, 0.0
    
    def _format_citations_html(self, citations: List) -> str:
        """Format citations as expandable HTML"""
        if not citations:
            return "<p>No citations available.</p>"
        
        html = "<div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9;'>"
        html += "<h4>📚 Sources & Citations</h4>"
        
        for citation in citations:
            confidence_color = "#28a745" if citation.confidence_score >= 0.8 else "#ffc107" if citation.confidence_score >= 0.6 else "#dc3545"
            
            html += f"""
            <details style='margin: 10px 0; padding: 10px; background: white; border-radius: 6px; border-left: 4px solid {confidence_color};'>
                <summary style='cursor: pointer; font-weight: bold; padding: 5px 0;'>
                    [{citation.citation_id}] {Path(citation.file_path).name} 
                    <span style='color: #666; font-weight: normal; font-size: 0.9em;'>
                        (Confidence: {citation.confidence_score:.2f})
                    </span>
                </summary>
                <div style='margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 4px;'>
                    <p><strong>Source:</strong> {citation.source_type.title()}</p>
                    <p><strong>File:</strong> {citation.file_path}</p>
                    {f"<p><strong>Page:</strong> {citation.page_number}</p>" if citation.page_number else ""}
                    <p><strong>Content Preview:</strong></p>
                    <div style='background: white; padding: 8px; border-radius: 4px; font-style: italic; border-left: 3px solid #007bff;'>
                        {citation.content_snippet}
                    </div>
                    <p style='font-size: 0.8em; color: #666; margin-top: 8px;'>
                        <strong>Timestamp:</strong> {citation.timestamp}
                    </p>
                </div>
            </details>
            """
        
        html += "</div>"
        return html
    
    def submit_feedback(self, query: str, response: str, rating: int, comments: str = "") -> str:
        """
        Submit user feedback for a query-response pair
        
        Args:
            query: The original query
            response: The generated response
            rating: Rating from 1-5
            comments: Optional feedback comments
            
        Returns:
            Status message
        """
        try:
            # Initialize components if needed
            self._initialize_components()
            
            if not self.feedback_system:
                return "❌ Feedback system not available"
            
            if not query or not response:
                return "❌ Query and response are required for feedback"
            
            if not (1 <= rating <= 5):
                return "❌ Rating must be between 1 and 5"
            
            # Submit feedback
            feedback_id = self.feedback_system.collect_feedback(
                query=query,
                response=response,
                rating=rating,
                comments=comments,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'interface': 'gradio'
                }
            )
            
            return f"✅ Thank you for your feedback! (ID: {feedback_id})"
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return f"❌ Feedback submission error: {e}"
    
    def get_system_info(self) -> str:
        """Get system information for display"""
        try:
            supported_formats_str = ", ".join(self.supported_formats)
            max_size = SECURITY_CONFIG['max_upload_size_mb']
            
            info = f"""
            ## 🔧 System Information
            
            **Supported Formats:** {supported_formats_str}
            
            **Maximum File Size:** {max_size} MB
            
            **Processing Status:** {'🟡 Processing...' if self.processing_state['is_processing'] else '🟢 Ready'}
            
            **Query Status:** {'🟡 Querying...' if self.query_state['is_querying'] else '🟢 Ready'}
            
            **Components Status:**
            - Ingestion Manager: ✅ Ready
            - Embedding Manager: {'✅ Ready' if self.embedding_manager else '⏳ Not initialized'}
            - Vector Store: {'✅ Ready' if self.vector_store else '⏳ Not initialized'}
            - Query Processor: {'✅ Ready' if self.query_processor else '⏳ Not initialized'}
            - Speech-to-Text: {'✅ Ready' if self.stt_processor else '⏳ Not initialized'}
            - LLM Generator: {'✅ Ready' if self.llm_generator else '⏳ Not initialized'}
            - Citation Generator: {'✅ Ready' if self.citation_generator else '⏳ Not initialized'}
            - Feedback System: {'✅ Ready' if self.feedback_system else '⏳ Not initialized'}
            """
            
            return info
            
        except Exception as e:
            return f"❌ Error getting system info: {e}"
    
    def create_interface(self) -> gr.Blocks:
        """
        Modern & Attractive SecureInsight Interface
        """

        css = r"""
        .gradio-container { 
            max-width: 1800px !important; 
            margin: 0 auto; 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial; 
            background: #393C3C !important; 
        }

        /* Sidebar */
        .sidebar { 
            background: #C3C7C7; 
            padding: 16px; 
            border-right: 1px solid #e0e0e0; 
            height: 100%; 
            border-radius: 12px; 
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .chat-list-item { 
            padding: 12px 16px; 
            margin-bottom: 8px; 
            border-radius: 10px; 
            cursor: pointer; 
            background: #7C8383; 
            font-weight: 500;
            transition: all 0.2s ease-in-out;
        }
        .chat-list-item:hover { 
            background: #000000; 
            color: #ffffff; 
            transform: scale(1.02);
        }

        /* Header Bar */
        .header-bar { 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            padding: 10px 16px; 
            margin-bottom: 12px; 
            border-bottom: 2px solid #e0e0e0; 
        }
        .header-title { 
            font-weight: 700; 
            font-size: 22px; 
            color: #111827; 
        }
        .header-bar button { 
            background:#2563eb; 
            color: white; 
            border:none; 
            border-radius:6px; 
            padding:6px 14px; 
            font-size:14px; 
            cursor:pointer; 
            transition: background 0.2s ease-in-out;
        }
        .header-bar button:hover { background:#1e40af; }

        /* Main Chat Panel */
        .main-chat { 
            background: #ffffff; 
            border-radius: 14px; 
            padding: 18px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
            min-height: 650px; 
        }
        .chatbox { 
            border: 1px solid #d1d5db; 
            border-radius: 10px; 
            padding: 10px;
        }

        /* Right Panel */
        .right-panel { 
            background: #ffffff; 
            padding: 18px; 
            border-radius: 14px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
            min-height: 100%; 
        }
        .right-panel h3, .right-panel label { 
            color: #111827 !important; 
        }

        /* Inputs */
        input, textarea, select, .gr-textbox, .gr-text-input {
            background: #f9fafb !important; 
            color: #111827 !important; 
            border: 1px solid #d1d5db !important; 
            border-radius: 8px !important; 
            padding: 8px;
        }

        /* Buttons */
        button { 
            font-weight: 600; 
            border-radius: 8px !important; 
            transition: all 0.2s ease-in-out;
        }
        button:hover { transform: scale(1.02); }

        /* Tabs */
        .tab-nav button { 
            font-weight: 600; 
            padding: 8px 14px; 
            color: #F1A151;
        }

        /* Force all Gradio component labels to be visible */
        label, .block-label, .wrap label, .form label {
            color: #000000 !important;  /* black text */
            font-weight: 600;
        }

        """

        with gr.Blocks(css=css) as interface:
            with gr.Row():
                # -------------------------
                # LEFT: Sidebar Chat List
                # -------------------------
                with gr.Column(scale=2, min_width=240, elem_classes="sidebar"):
                    gr.Markdown("### 💬 Chats")
                    gr.HTML(
                        "<div class='chat-list-item'>Unnamed Chat</div>"
                        "<div class='chat-list-item'>Project Notes</div>"
                        "<div class='chat-list-item'>Research Draft</div>"
                    )

                # -------------------------
                # CENTER: Main Workspace
                # -------------------------
                with gr.Column(scale=6, min_width=800):
                    with gr.Column(elem_classes="main-chat"):
                        gr.HTML(
                            "<div class='header-bar'>"
                            "<span class='header-title'>🔒 SecureInsight - Multimodal RAG System</span>"
                            "<button>Eject</button>"
                            "</div>"
                        )

                        with gr.Tabs():
                            # -------- File Upload --------
                            with gr.TabItem("📤 File Upload"):
                                file_upload = gr.File(
                                    label="Upload Files",
                                    file_count="multiple",
                                    file_types=self.supported_formats,
                                )
                                process_btn = gr.Button("🚀 Process Files", variant="primary")
                                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                                status_display = gr.Textbox(label="📊 Processing Status", value="🔄 Ready", interactive=False)
                                processing_log = gr.Textbox(label="📋 Processing Log", lines=8, interactive=False)
                                processed_files_display = gr.HTML("<p>No files processed yet.</p>")

                            # -------- Query --------
                            with gr.TabItem("🔍 Search & Query"):
                                text_query_input = gr.Textbox(label="Enter Query")
                                text_search_btn = gr.Button("🔍 Search Text", variant="primary")
                                search_results_display = gr.HTML("<p>No results yet.</p>")

                            # -------- AI Response --------
                            with gr.TabItem("💬 AI Response"):
                                response_query_input = gr.Textbox(label="Enter your question")
                                generate_response_btn = gr.Button("🤖 Generate Response", variant="primary")
                                generated_response_display = gr.Textbox(label="Response", interactive=False, lines=8)
                                citations_display = gr.HTML("<p>No citations yet.</p>")

                            # -------- Feedback --------
                            with gr.TabItem("📝 Feedback"):
                                feedback_rating = gr.Slider(label="Rating (1-5)", minimum=1, maximum=5, value=3, step=1)
                                feedback_comments = gr.Textbox(label="Comments", lines=2)
                                submit_feedback_btn = gr.Button("Submit Feedback", variant="secondary")
                                feedback_status = gr.Textbox(label="Feedback Status", value="Ready", interactive=False)

                            # -------- Help --------
                            with gr.TabItem("❓ Help & Instructions"):
                                gr.Markdown("## Instructions\n- Upload files\n- Run queries\n- Generate responses\n- Provide feedback")

                # -------------------------
                # RIGHT: Configuration Panel
                # -------------------------
                with gr.Column(scale=3, min_width=280, elem_classes="right-panel"):
                    gr.Markdown("### ⚙️ Advanced Configuration")

                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder='Example: "Only answer in rhymes"',
                        lines=3
                    )

                    with gr.Accordion("General", open=False):
                        gr.Checkbox(label="Enable feature X")
                        gr.Checkbox(label="Enable feature Y")

                    with gr.Accordion("Sampling", open=False):
                        gr.Slider(label="Temperature", minimum=0, maximum=2, step=0.1, value=1.0)
                        gr.Slider(label="Top-p", minimum=0, maximum=1, step=0.05, value=0.95)

            # -------------------------
            # Event handlers
            # -------------------------
            process_btn.click(
                fn=self.process_uploaded_files,
                inputs=[file_upload],
                outputs=[status_display, processing_log, processed_files_display],
                show_progress=True
            )

            clear_btn.click(
                fn=self.clear_processing_state,
                outputs=[status_display, processing_log, processed_files_display]
            )

            text_search_btn.click(
                fn=self.process_text_query,
                inputs=[text_query_input],
                outputs=[search_results_display],
                show_progress=True
            )

            generate_response_btn.click(
                fn=self.generate_response_with_citations,
                inputs=[response_query_input],
                outputs=[generated_response_display, citations_display],
                show_progress=True
            )

            submit_feedback_btn.click(
                fn=self.submit_feedback,
                inputs=[response_query_input, generated_response_display, feedback_rating, feedback_comments],
                outputs=[feedback_status]
            )

        return interface



def create_gradio_interface() -> gr.Blocks:
    """
    Factory function to create the Gradio interface
    
    Returns:
        Configured Gradio Blocks interface
    """
    app = SecureInsightGradioApp()
    return app.create_interface()


def launch_gradio_app(
    server_name: str = None,
    server_port: int = None,
    share: bool = False,
    debug: bool = False
) -> None:
    """
    Launch the Gradio application
    
    Args:
        server_name: Server hostname (default from config)
        server_port: Server port (default from config)
        share: Whether to create public link
        debug: Enable debug mode
    """
    try:
        # Use config defaults if not specified
        server_name = server_name or GRADIO_CONFIG['server_name']
        server_port = server_port or GRADIO_CONFIG['server_port']
        share = share or GRADIO_CONFIG.get('share', False)
        
        # Create and launch interface
        interface = create_gradio_interface()
        
        logger.info(f"Launching Gradio app on {server_name}:{server_port}")
        
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            debug=debug,
            show_error=True,
            quiet=not debug
        )
        
    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {e}")
        raise


if __name__ == "__main__":
    # Launch with default settings
    launch_gradio_app(debug=True)
