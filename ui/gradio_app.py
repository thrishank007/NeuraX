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
from datetime import datetime

# Import system components
from ingestion.ingestion_manager import IngestionManager
from indexing.embedding_manager import EmbeddingManager
from indexing.vector_store import VectorStore
from retrieval.query_processor import QueryProcessor
from retrieval.speech_to_text_processor import SpeechToTextProcessor
from generation.llm_generator import LLMGenerator
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
                self.vector_store = VectorStore()
            
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
                self.llm_generator = LLMGenerator(LLM_CONFIG)
            
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
                                self.vector_store.add_document(result)
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
                context_documents=search_results
            )
            
            # Generate citations
            citations = self.citation_generator.generate_citations(
                response=generated_response.response_text,
                sources=search_results,
                citation_indices=generated_response.citations_needed
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
        """Create the main Gradio interface"""
        
        with gr.Blocks(
            title="SecureInsight - Multimodal RAG System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1400px !important;
            }
            .file-upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #f9f9f9;
            }
            .query-section {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                background-color: #f8f9fa;
            }
            .status-box {
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; }
            .results-container {
                max-height: 600px;
                overflow-y: auto;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown(
                """
                # 🔒 SecureInsight - Multimodal RAG System
                
                Upload and process documents, images, and audio files for secure offline analysis.
                Supports PDFs, DOCs, images (JPG, PNG, etc.), and audio files (WAV, MP3, etc.).
                """
            )
            
            # Create tabs for different functionalities
            with gr.Tabs():
                # File Upload Tab
                with gr.TabItem("📤 File Upload", id="upload_tab"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # File upload section
                            gr.Markdown("## 📤 File Upload & Processing")
                            
                            file_upload = gr.File(
                                label="Upload Files",
                                file_count="multiple",
                                file_types=self.supported_formats,
                                elem_classes=["file-upload-area"]
                            )
                            
                            with gr.Row():
                                process_btn = gr.Button(
                                    "🚀 Process Files", 
                                    variant="primary",
                                    size="lg"
                                )
                                clear_btn = gr.Button(
                                    "🗑️ Clear", 
                                    variant="secondary"
                                )
                            
                            # Processing status
                            status_display = gr.Textbox(
                                label="📊 Processing Status",
                                value="🔄 Ready for new files",
                                interactive=False,
                                lines=2
                            )
                        
                        with gr.Column(scale=1):
                            # System information
                            gr.Markdown("## ℹ️ System Info")
                            system_info = gr.Markdown(
                                value=self.get_system_info()
                            )
                            
                            refresh_info_btn = gr.Button(
                                "🔄 Refresh Info",
                                size="sm"
                            )
                    
                    # Processing log and results
                    with gr.Row():
                        with gr.Column():
                            processing_log = gr.Textbox(
                                label="📋 Processing Log",
                                lines=10,
                                interactive=False,
                                placeholder="Processing details will appear here..."
                            )
                        
                        with gr.Column():
                            processed_files_display = gr.HTML(
                                label="📁 Processed Files",
                                value="<p>No files processed yet.</p>"
                            )
                
                # Query Interface Tab
                with gr.TabItem("🔍 Search & Query", id="query_tab"):
                    gr.Markdown("## 🔍 Multimodal Search Interface")
                    gr.Markdown("Search through your processed documents using text, images, or voice queries.")
                    
                    # Query controls
                    with gr.Row():
                        with gr.Column(scale=3):
                            # Text Query Section
                            with gr.Group(elem_classes=["query-section"]):
                                gr.Markdown("### 💬 Text Query")
                                text_query_input = gr.Textbox(
                                    label="Enter your search query",
                                    placeholder="e.g., 'security protocols', 'network configuration', 'user authentication'...",
                                    lines=2
                                )
                                
                                with gr.Row():
                                    text_search_btn = gr.Button(
                                        "🔍 Search Text", 
                                        variant="primary"
                                    )
                                    clear_text_btn = gr.Button(
                                        "🗑️ Clear", 
                                        variant="secondary",
                                        size="sm"
                                    )
                            
                            # Image Query Section
                            with gr.Group(elem_classes=["query-section"]):
                                gr.Markdown("### 🖼️ Image Query")
                                image_query_input = gr.File(
                                    label="Upload an image to search for similar content",
                                    file_types=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
                                )
                                
                                image_search_btn = gr.Button(
                                    "🔍 Search by Image", 
                                    variant="primary"
                                )
                            
                            # Voice Query Section
                            with gr.Group(elem_classes=["query-section"]):
                                gr.Markdown("### 🎤 Voice Query")
                                voice_query_input = gr.File(
                                    label="Upload an audio file with your spoken query",
                                    file_types=['.wav', '.mp3', '.m4a', '.flac', '.ogg']
                                )
                                
                                with gr.Row():
                                    voice_search_btn = gr.Button(
                                        "🔍 Search by Voice", 
                                        variant="primary"
                                    )
                                    # Placeholder for future microphone input
                                    # mic_btn = gr.Button(
                                    #     "🎙️ Record", 
                                    #     variant="secondary",
                                    #     size="sm"
                                    # )
                                
                                voice_transcription = gr.Textbox(
                                    label="Voice Transcription",
                                    placeholder="Transcribed text will appear here...",
                                    interactive=False,
                                    lines=2
                                )
                            
                            # Multimodal Query Section
                            with gr.Group(elem_classes=["query-section"]):
                                gr.Markdown("### 🔀 Multimodal Query")
                                gr.Markdown("Combine text and image for more precise search results.")
                                
                                multimodal_text = gr.Textbox(
                                    label="Text component",
                                    placeholder="Describe what you're looking for...",
                                    lines=1
                                )
                                
                                multimodal_image = gr.File(
                                    label="Image component",
                                    file_types=['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
                                )
                                
                                multimodal_search_btn = gr.Button(
                                    "🔍 Multimodal Search", 
                                    variant="primary"
                                )
                        
                        with gr.Column(scale=1):
                            # Search Settings
                            gr.Markdown("### ⚙️ Search Settings")
                            
                            similarity_threshold = gr.Slider(
                                label="Similarity Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.05,
                                info="Higher values = more strict matching"
                            )
                            
                            # Query Status
                            query_status = gr.Textbox(
                                label="🔍 Query Status",
                                value="🔄 Ready for queries",
                                interactive=False,
                                lines=2
                            )
                            
                            # Clear All Results
                            clear_all_btn = gr.Button(
                                "🗑️ Clear All Results",
                                variant="secondary",
                                size="sm"
                            )
                            
                            # Query History
                            with gr.Accordion("📋 Query History", open=False):
                                query_history_display = gr.HTML(
                                    value="<p>No queries yet.</p>"
                                )
                    
                    # Search Results
                    with gr.Row():
                        search_results_display = gr.HTML(
                            label="🔍 Search Results",
                            value="<p>No search results yet. Enter a query above to get started.</p>",
                            elem_classes=["results-container"]
                        )
                
                # Response Generation Tab
                with gr.TabItem("💬 AI Response", id="response_tab"):
                    gr.Markdown("## 💬 AI Response Generation with Citations")
                    gr.Markdown("Generate AI responses based on your search results with full source citations.")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Response Generation Section
                            with gr.Group(elem_classes=["query-section"]):
                                gr.Markdown("### 🤖 Generate Response")
                                
                                response_query_input = gr.Textbox(
                                    label="Enter your question",
                                    placeholder="Ask a question about your documents...",
                                    lines=2
                                )
                                
                                with gr.Row():
                                    generate_response_btn = gr.Button(
                                        "🤖 Generate Response", 
                                        variant="primary",
                                        size="lg"
                                    )
                                    clear_response_btn = gr.Button(
                                        "🗑️ Clear", 
                                        variant="secondary"
                                    )
                                
                                # Response Status
                                response_status = gr.Textbox(
                                    label="🤖 Generation Status",
                                    value="🔄 Ready to generate responses",
                                    interactive=False,
                                    lines=2
                                )
                        
                        with gr.Column(scale=1):
                            # Response Settings
                            gr.Markdown("### ⚙️ Response Settings")
                            
                            response_similarity_threshold = gr.Slider(
                                label="Context Similarity Threshold",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.05,
                                info="Higher values = more relevant context"
                            )
                            
                            max_context_docs = gr.Slider(
                                label="Maximum Context Documents",
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1,
                                info="Number of documents to use as context"
                            )
                    
                    # Generated Response Display
                    with gr.Row():
                        with gr.Column():
                            generated_response_display = gr.Textbox(
                                label="🤖 Generated Response",
                                placeholder="Generated response will appear here...",
                                lines=8,
                                interactive=False
                            )
                            
                            # Response Metadata
                            with gr.Row():
                                response_confidence = gr.Textbox(
                                    label="Confidence Score",
                                    placeholder="N/A",
                                    interactive=False,
                                    scale=1
                                )
                                response_time = gr.Textbox(
                                    label="Processing Time",
                                    placeholder="N/A",
                                    interactive=False,
                                    scale=1
                                )
                        
                        with gr.Column():
                            citations_display = gr.HTML(
                                label="📚 Citations & Sources",
                                value="<p>No citations yet. Generate a response to see sources.</p>",
                                elem_classes=["results-container"]
                            )
                    
                    # Feedback Section
                    with gr.Group(elem_classes=["query-section"]):
                        gr.Markdown("### 📝 Feedback")
                        gr.Markdown("Help us improve by rating the response quality.")
                        
                        with gr.Row():
                            feedback_rating = gr.Slider(
                                label="Rating (1-5 stars)",
                                minimum=1,
                                maximum=5,
                                value=3,
                                step=1
                            )
                            
                            feedback_comments = gr.Textbox(
                                label="Comments (optional)",
                                placeholder="Any additional feedback...",
                                lines=2,
                                scale=2
                            )
                        
                        with gr.Row():
                            submit_feedback_btn = gr.Button(
                                "📝 Submit Feedback",
                                variant="secondary"
                            )
                            
                            feedback_status = gr.Textbox(
                                label="Feedback Status",
                                placeholder="Ready to receive feedback",
                                interactive=False,
                                scale=2
                            )
            
            # Help section
            with gr.Accordion("❓ Help & Instructions", open=False):
                gr.Markdown(
                    f"""
                    ### How to Use
                    
                    #### File Upload & Processing
                    1. **Upload Files**: Click the upload area or drag and drop files
                    2. **Supported Formats**: {', '.join(self.supported_formats)}
                    3. **File Size Limit**: {SECURITY_CONFIG['max_upload_size_mb']} MB per file
                    4. **Processing**: Click "Process Files" to start ingestion
                    5. **Monitor**: Watch the processing log for real-time updates
                    
                    #### Search & Query
                    1. **Text Search**: Enter natural language queries to find relevant documents
                    2. **Image Search**: Upload an image to find visually similar content
                    3. **Voice Search**: Upload audio files with spoken queries (uses speech-to-text)
                    4. **Multimodal Search**: Combine text and image for more precise results
                    5. **Adjust Threshold**: Use the similarity slider to control result strictness
                    
                    ### File Types
                    
                    - **📄 Documents**: PDF, DOCX, DOC, TXT - Text extraction and indexing
                    - **🖼️ Images**: JPG, PNG, BMP, TIFF - OCR text extraction
                    - **🎵 Audio**: WAV, MP3, M4A, FLAC - Speech-to-text transcription
                    
                    ### Search Features
                    
                    - **Cross-modal search**: Text queries can find images and vice versa
                    - **Similarity scoring**: All results include confidence scores
                    - **Content preview**: See snippets of matching content
                    - **Source traceability**: Direct links to original files
                    
                    ### Security Features
                    
                    - All processing happens offline
                    - Files are validated before processing
                    - No data leaves your system
                    - Comprehensive error handling
                    
                    ### Troubleshooting
                    
                    - **File not supported**: Check the supported formats list
                    - **File too large**: Reduce file size or split into smaller files
                    - **Processing failed**: Check the processing log for details
                    - **No search results**: Try lowering the similarity threshold
                    - **Voice unclear**: Upload clearer audio or type your query instead
                    - **System slow**: Try processing fewer files at once
                    """
                )
            
            # Event handlers for file upload tab
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
            
            refresh_info_btn.click(
                fn=self.get_system_info,
                outputs=[system_info]
            )
            
            # Event handlers for query tab
            
            # Text search
            text_search_btn.click(
                fn=self.process_text_query,
                inputs=[text_query_input, similarity_threshold],
                outputs=[query_status, search_results_display],
                show_progress=True
            )
            
            # Clear text query
            clear_text_btn.click(
                fn=lambda: "",
                outputs=[text_query_input]
            )
            
            # Image search
            image_search_btn.click(
                fn=self.process_image_query,
                inputs=[image_query_input, similarity_threshold],
                outputs=[query_status, search_results_display],
                show_progress=True
            )
            
            # Voice search
            voice_search_btn.click(
                fn=self.process_voice_query,
                inputs=[voice_query_input, similarity_threshold],
                outputs=[query_status, voice_transcription, search_results_display],
                show_progress=True
            )
            
            # Multimodal search
            multimodal_search_btn.click(
                fn=self.process_multimodal_query,
                inputs=[multimodal_text, multimodal_image, similarity_threshold],
                outputs=[query_status, search_results_display],
                show_progress=True
            )
            
            # Clear all results
            clear_all_btn.click(
                fn=self.clear_query_results,
                outputs=[query_status, text_query_input, voice_transcription, search_results_display]
            )
            
            # Update query history when any search is performed
            def update_history():
                return self.get_query_history_html()
            
            # Connect history updates to search buttons
            for btn in [text_search_btn, image_search_btn, voice_search_btn, multimodal_search_btn]:
                btn.click(
                    fn=update_history,
                    outputs=[query_history_display],
                    show_progress=False
                )
            
            # Event handlers for response generation tab
            
            # Generate response with citations
            def generate_response_handler(query_text, similarity_threshold, max_docs):
                if not query_text or not query_text.strip():
                    return "❌ Please enter a question", "", "", "", ""
                
                # First, search for relevant documents
                search_status, search_html = self.process_text_query(query_text, similarity_threshold)
                
                if "No results found" in search_status:
                    return "ℹ️ No relevant documents found for context", "", "", "", ""
                
                # Get search results from the last query
                search_results = self.query_state.get('last_results', [])[:max_docs]
                
                # Generate response with citations
                response_text, citations_html, confidence, proc_time = self.generate_response_with_citations(
                    query_text, search_results
                )
                
                status = f"✅ Response generated in {proc_time:.2f}s"
                confidence_display = f"{confidence:.2f}" if confidence > 0 else "N/A"
                time_display = f"{proc_time:.2f}s" if proc_time > 0 else "N/A"
                
                return status, response_text, citations_html, confidence_display, time_display
            
            generate_response_btn.click(
                fn=generate_response_handler,
                inputs=[response_query_input, response_similarity_threshold, max_context_docs],
                outputs=[response_status, generated_response_display, citations_display, response_confidence, response_time],
                show_progress=True
            )
            
            # Clear response
            clear_response_btn.click(
                fn=lambda: ("🔄 Ready to generate responses", "", "<p>No citations yet. Generate a response to see sources.</p>", "", ""),
                outputs=[response_status, generated_response_display, citations_display, response_confidence, response_time]
            )
            
            # Submit feedback
            def submit_feedback_handler(query, response, rating, comments):
                if not query or not response:
                    return "❌ No response to provide feedback for"
                
                return self.submit_feedback(query, response, int(rating), comments)
            
            submit_feedback_btn.click(
                fn=submit_feedback_handler,
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
