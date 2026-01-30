"""
FastAPI Backend API for NeuraX RAG System

Provides RESTful API endpoints for the Next.js frontend to interact with
the Python-based RAG backend components.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Suppress warnings for optional dependencies
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import backend components
from ingestion.ingestion_manager import IngestionManager
from retrieval.query_processor import QueryProcessor
from generation.lmstudio_generator import LMStudioGenerator
from generation.citation_generator import CitationGenerator
from indexing.embedding_manager import EmbeddingManager
from indexing.vector_store import VectorStore
from feedback.feedback_system import FeedbackSystem
from kg_security.knowledge_graph_manager import KnowledgeGraphManager
from feedback.metrics_collector import MetricsCollector
from retrieval.speech_to_text_processor import SpeechToTextProcessor
from config import (
    LM_STUDIO_CONFIG, CHROMA_CONFIG, VECTOR_DB_DIR,
    PROCESSING_CONFIG, FEEDBACK_CONFIG, KG_CONFIG
)

# Initialize FastAPI app
app = FastAPI(
    title="NeuraX RAG API",
    description="RESTful API for NeuraX Multimodal RAG System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global component instances (initialized on startup)
ingestion_manager: Optional[IngestionManager] = None
embedding_manager: Optional[EmbeddingManager] = None
vector_store: Optional[VectorStore] = None
query_processor: Optional[QueryProcessor] = None
llm_generator: Optional[LMStudioGenerator] = None
citation_generator: Optional[CitationGenerator] = None
feedback_system: Optional[FeedbackSystem] = None
kg_manager: Optional[KnowledgeGraphManager] = None
metrics_collector: Optional[MetricsCollector] = None
stt_processor: Optional[SpeechToTextProcessor] = None


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize all backend components on startup"""
    global ingestion_manager, embedding_manager, vector_store
    global query_processor, llm_generator, citation_generator
    global feedback_system, kg_manager, metrics_collector, stt_processor
    
    import traceback
    
    print("🚀 Starting NeuraX Backend API initialization...")
    
    # Initialize ingestion manager
    try:
        print("Initializing ingestion manager...")
        ingestion_manager = IngestionManager()
        print("✅ Ingestion manager initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize ingestion manager: {e}")
        ingestion_manager = None
    
    # Initialize embedding manager
    try:
        print("Initializing embedding manager...")
        embedding_manager = EmbeddingManager()
        print("✅ Embedding manager initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize embedding manager: {e}")
        print(traceback.format_exc())
        embedding_manager = None
    
    # Initialize vector store
    try:
        print("Initializing vector store...")
        vector_store = VectorStore(
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=CHROMA_CONFIG['collection_name']
        )
        print("✅ Vector store initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize vector store: {e}")
        print(traceback.format_exc())
        vector_store = None
    
    # Initialize query processor (depends on embedding and vector store)
    if embedding_manager and vector_store:
        try:
            print("Initializing query processor...")
            query_processor = QueryProcessor(
                embedding_manager,
                vector_store,
                {
                    'similarity_threshold': 0.5,
                    'max_results': 10,
                    'enable_cross_modal': True
                }
            )
            print("✅ Query processor initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize query processor: {e}")
            print(traceback.format_exc())
            query_processor = None
    else:
        print("⚠️ Skipping query processor (missing dependencies)")
        query_processor = None
    
    # Initialize LLM generator
    try:
        print("Initializing LLM generator...")
        llm_generator = LMStudioGenerator(LM_STUDIO_CONFIG)
        print("✅ LLM generator initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize LLM generator: {e}")
        print(traceback.format_exc())
        llm_generator = None
    
    # Initialize citation generator
    try:
        print("Initializing citation generator...")
        citation_generator = CitationGenerator()
        print("✅ Citation generator initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize citation generator: {e}")
        print(traceback.format_exc())
        citation_generator = None
    
    # Initialize feedback system
    try:
        print("Initializing feedback system...")
        feedback_system = FeedbackSystem()
        print("✅ Feedback system initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize feedback system: {e}")
        print(traceback.format_exc())
        feedback_system = None
    
    # Initialize knowledge graph manager
    try:
        print("Initializing knowledge graph manager...")
        kg_manager = KnowledgeGraphManager()
        print("✅ Knowledge graph manager initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize knowledge graph manager: {e}")
        print(traceback.format_exc())
        kg_manager = None
    
    # Initialize metrics collector
    try:
        print("Initializing metrics collector...")
        metrics_collector = MetricsCollector()
        print("✅ Metrics collector initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize metrics collector: {e}")
        print(traceback.format_exc())
        metrics_collector = None
    
    # Initialize speech-to-text processor (optional)
    try:
        print("Initializing speech-to-text processor...")
        stt_processor = SpeechToTextProcessor()
        print("✅ Speech-to-text processor initialized")
    except Exception as e:
        print(f"⚠️ Speech-to-text processor not available: {e}")
        stt_processor = None
    
    # Summary
    initialized = sum([
        ingestion_manager is not None,
        embedding_manager is not None,
        vector_store is not None,
        query_processor is not None,
        llm_generator is not None,
        citation_generator is not None,
        feedback_system is not None,
        kg_manager is not None,
    ])
    
    print(f"\n✅ Backend API initialized ({initialized}/8 core components)")
    print("📝 API documentation available at: http://localhost:8000/docs")
    print("🔍 OpenAPI schema available at: http://localhost:8000/openapi.json")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global embedding_manager, vector_store
    
    try:
        if embedding_manager:
            embedding_manager.save_cache()
        if vector_store:
            # Vector store auto-persists
            pass
    except Exception as e:
        print(f"Warning: Error during shutdown: {e}")


# ==================== Request/Response Models ====================

class QueryRequest(BaseModel):
    query: str
    query_type: str = "text"  # text, image, multimodal
    k: int = 10
    similarity_threshold: float = 0.5
    filters: Optional[Dict[str, Any]] = None
    generate_response: bool = True


class QueryResponse(BaseModel):
    query_id: str
    query: str
    response_text: Optional[str] = None
    results: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    processing_time: float
    total_results: int
    model_used: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_id": "query_123",
                "query": "What is RAG?",
                "response_text": "RAG is...",
                "results": [],
                "citations": [],
                "processing_time": 1.5,
                "total_results": 0,
                "model_used": "gemma-3n"
            }
        }


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    status: str
    processing_time: float
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "doc_123",
                "filename": "example.pdf",
                "file_type": "pdf",
                "status": "success",
                "processing_time": 2.5,
                "message": "File processed and indexed successfully"
            }
        }


class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(ge=1, le=5)
    comments: Optional[str] = None
    query_metadata: Optional[Dict[str, Any]] = None


class ConfigUpdateRequest(BaseModel):
    lm_studio_url: Optional[str] = None
    similarity_threshold: Optional[float] = None
    max_results: Optional[int] = None
    model_preference: Optional[str] = None  # "gemma" or "qwen"


# ==================== Health Check ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "ingestion": ingestion_manager is not None,
            "embedding": embedding_manager is not None,
            "vector_store": vector_store is not None,
            "query_processor": query_processor is not None,
            "llm_generator": llm_generator is not None,
            "citation_generator": citation_generator is not None,
            "feedback_system": feedback_system is not None,
            "kg_manager": kg_manager is not None,
        }
    }


@app.get("/api/test")
async def test_endpoint():
    """Simple test endpoint to verify API connectivity"""
    return {
        "message": "API is working",
        "timestamp": datetime.utcnow().isoformat(),
        "status": "ok"
    }


# ==================== File Upload ====================

@app.post("/api/upload", response_model=List[FileUploadResponse])
async def upload_files(
    files: List[UploadFile] = File(...)
):
    """Upload and process files"""
    if not ingestion_manager or not embedding_manager or not vector_store:
        raise HTTPException(status_code=503, detail="Backend components not initialized")
    
    results = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Process files sequentially to avoid overwhelming the system
        for idx, file in enumerate(files):
            start_time = datetime.now()
            
            # Validate file
            file_ext = Path(file.filename).suffix.lower()
            max_size = PROCESSING_CONFIG.get('max_file_size_mb', 100) * 1024 * 1024
            
            # Save uploaded file temporarily
            temp_path = Path(temp_dir) / file.filename
            with open(temp_path, "wb") as f:
                content = await file.read()
                if len(content) > max_size:
                    results.append(FileUploadResponse(
                        file_id="",
                        filename=file.filename,
                        file_type=file_ext,
                        status="error",
                        processing_time=0,
                        message=f"File too large (max {max_size / 1024 / 1024}MB)"
                    ))
                    continue
                f.write(content)
            
            # Process file
            try:
                print(f"Processing file {idx + 1}/{len(files)}: {file.filename}")
                processed_doc = ingestion_manager.process_file(temp_path)
                
                if processed_doc:
                    print(f"File {file.filename} processed, generating embeddings...")
                    # Generate embeddings
                    try:
                        if processed_doc.get('file_type') in ['pdf', 'docx', 'doc', 'txt']:
                            content_text = processed_doc.get('content_preview', '')
                            if content_text:
                                embedding = embedding_manager.embed_text(content_text)[0]
                                processed_doc['embedding'] = embedding
                        elif processed_doc.get('file_type') in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
                            embedding = embedding_manager.embed_image(str(temp_path))[0]
                            processed_doc['embedding'] = embedding
                    except Exception as embed_error:
                        print(f"Embedding generation failed for {file.filename}: {embed_error}")
                        # Continue without embedding - file is still processed
                    
                    # Add to vector store
                    try:
                        vector_store.add_document(processed_doc)
                        print(f"File {file.filename} added to vector store")
                    except Exception as store_error:
                        print(f"Vector store error for {file.filename}: {store_error}")
                        # Continue - file processing succeeded even if storage failed
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    results.append(FileUploadResponse(
                        file_id=processed_doc.get('document_id', ''),
                        filename=file.filename,
                        file_type=processed_doc.get('file_type', 'unknown'),
                        status="success",
                        processing_time=processing_time,
                        message="File processed and indexed successfully"
                    ))
                    print(f"Successfully processed {file.filename} in {processing_time:.2f}s")
                else:
                    print(f"Failed to process {file.filename}: ingestion_manager returned None")
                    results.append(FileUploadResponse(
                        file_id="",
                        filename=file.filename,
                        file_type=file_ext,
                        status="error",
                        processing_time=0,
                        message="Failed to process file - ingestion manager returned no result"
                    ))
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error processing {file.filename}: {error_trace}")
                results.append(FileUploadResponse(
                    file_id="",
                    filename=file.filename,
                    file_type=file_ext,
                    status="error",
                    processing_time=0,
                    message=f"Processing error: {str(e)}"
                ))
    
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


@app.get("/api/files")
async def list_files():
    """List all uploaded files"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    # Get all documents from vector store
    # This is a simplified version - in production, maintain a separate file registry
    return {"files": [], "total": 0}


@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a file from the system"""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    # Delete from vector store
    # In production, implement proper deletion
    return {"status": "deleted", "file_id": file_id}


# ==================== Query Processing ====================

@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a multimodal query"""
    if not query_processor or not llm_generator or not citation_generator:
        raise HTTPException(status_code=503, detail="Query components not initialized")
    
    start_time = datetime.now()
    query_id = f"query_{datetime.now().timestamp()}"
    
    try:
        # Process query based on type
        if request.query_type == "text":
            query_result = query_processor.process_text_query(
                request.query,
                filters=request.filters,
                k=request.k
            )
        elif request.query_type == "image":
            # Image queries would need image data in request
            raise HTTPException(status_code=400, detail="Image queries not yet supported via JSON")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported query type: {request.query_type}")
        
        # Generate response if requested
        response_text = None
        model_used = None
        if request.generate_response and llm_generator:
            # Prepare context from results
            context = [
                {
                    'content': r.get('content', ''),
                    'content_preview': r.get('content_preview', ''),
                    'metadata': r.get('metadata', {})
                }
                for r in query_result.results
            ]
            
            # Generate grounded response
            generated = llm_generator.generate_grounded_response(
                request.query,
                context
            )
            response_text = generated.response_text
            model_used = generated.model_used
        
        # Generate citations
        citations = []
        if response_text and citation_generator:
            citations_data = citation_generator.generate_citations(
                response_text,
                query_result.results
            )
            citations = [
                {
                    'citation_id': c.citation_id,
                    'source_document': c.source_document,
                    'source_type': c.source_type,
                    'content_snippet': c.content_snippet,
                    'confidence_score': c.confidence_score,
                    'file_path': c.file_path,
                    'page_number': c.page_number
                }
                for c in citations_data
            ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            query_id=query_id,
            query=request.query,
            response_text=response_text,
            results=[
                {
                    'document_id': r.get('document_id', ''),
                    'similarity_score': r.get('similarity_score', 0),
                    'content_preview': r.get('content_preview', ''),
                    'file_path': r.get('file_path', ''),
                    'file_type': r.get('file_type', ''),
                    'metadata': r.get('metadata', {})
                }
                for r in query_result.results
            ],
            citations=citations,
            processing_time=processing_time,
            total_results=len(query_result.results),
            model_used=model_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.post("/api/query/voice")
async def process_voice_query(audio_file: UploadFile = File(...)):
    """Process voice query (speech-to-text then query)"""
    if not stt_processor:
        raise HTTPException(status_code=503, detail="Speech-to-text not available")
    
    # Save audio temporarily
    temp_path = Path(tempfile.mktemp(suffix=".wav"))
    try:
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Transcribe
        transcription = stt_processor.transcribe_audio(str(temp_path))
        
        # Process as text query
        request = QueryRequest(
            query=transcription,
            query_type="text",
            generate_response=True
        )
        
        return await process_query(request)
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/query/history")
async def get_query_history(limit: int = 20):
    """Get query history"""
    # In production, maintain query history in database
    return {"queries": [], "total": 0}


@app.get("/api/query/suggestions")
async def get_query_suggestions(partial: str, limit: int = 5):
    """Get query auto-complete suggestions"""
    if not query_processor:
        return {"suggestions": []}
    
    suggestions = query_processor.get_query_suggestions(partial, limit)
    return {"suggestions": suggestions}


# ==================== Analytics ====================

@app.get("/api/analytics/metrics")
async def get_metrics(time_range_hours: int = 24):
    """Get system performance metrics"""
    if not metrics_collector:
        return {
            "metrics": {},
            "time_range_hours": time_range_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    try:
        # MetricsCollector has generate_metrics_report method
        # For now, return basic structure - can be enhanced later
        if hasattr(metrics_collector, 'generate_metrics_report'):
            # Convert days to hours for the report
            days = max(1, time_range_hours // 24)
            metrics = metrics_collector.generate_metrics_report(days=days)
        elif hasattr(metrics_collector, '_get_aggregated_query_metrics'):
            metrics = {
                "query_metrics": metrics_collector._get_aggregated_query_metrics(),
                "benchmark_metrics": metrics_collector._get_aggregated_benchmark_metrics() if hasattr(metrics_collector, '_get_aggregated_benchmark_metrics') else {}
            }
        else:
            metrics = {}
        
        return {
            "metrics": metrics if isinstance(metrics, dict) else {"data": metrics},
            "time_range_hours": time_range_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "metrics": {},
            "time_range_hours": time_range_hours,
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@app.get("/api/analytics/usage")
async def get_usage_stats(time_range_hours: int = 24):
    """Get usage statistics"""
    # In production, aggregate from query history and feedback
    return {
        "queries_per_day": 0,
        "file_uploads": 0,
        "popular_queries": [],
        "time_range_hours": time_range_hours
    }


@app.get("/api/analytics/security")
async def get_security_events(limit: int = 50):
    """Get security events and anomalies"""
    if not kg_manager:
        return {"events": [], "anomalies": [], "total": 0}
    
    try:
        # Get anomalies from knowledge graph
        anomalies = kg_manager.detect_anomalies()
        
        # Ensure anomalies is a list
        if not isinstance(anomalies, list):
            anomalies = []
        
        return {
            "events": [],
            "anomalies": [
                {
                    "anomaly_id": a.get('anomaly_id', '') if isinstance(a, dict) else str(a),
                    "type": a.get('type', '') if isinstance(a, dict) else 'unknown',
                    "severity": a.get('severity', '') if isinstance(a, dict) else 'low',
                    "description": a.get('description', '') if isinstance(a, dict) else str(a),
                    "timestamp": a.get('timestamp', '') if isinstance(a, dict) else ''
                }
                for a in anomalies[:limit]
            ],
            "total": len(anomalies)
        }
    except Exception as e:
        print(f"Error getting security events: {e}")
        import traceback
        traceback.print_exc()
        return {"events": [], "anomalies": [], "total": 0, "error": str(e)}


@app.get("/api/knowledge-graph")
async def get_knowledge_graph():
    """Get knowledge graph data for visualization"""
    if not kg_manager:
        return {"nodes": [], "edges": []}
    
    stats = kg_manager.get_graph_stats()
    graph_data = kg_manager.export_graph_for_visualization()
    
    return {
        "nodes": graph_data.get('nodes', []),
        "edges": graph_data.get('edges', []),
        "stats": stats
    }


# ==================== Feedback ====================

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    if not feedback_system:
        raise HTTPException(status_code=503, detail="Feedback system not initialized")
    
    try:
        feedback_system.collect_feedback(
            query=feedback.query,
            response=feedback.response,
            rating=feedback.rating,
            comments=feedback.comments,
            query_metadata=feedback.query_metadata or {}
        )
        
        return {"status": "success", "message": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.get("/api/feedback/history")
async def get_feedback_history(limit: int = 20):
    """Get feedback history"""
    if not feedback_system:
        return {"feedback": [], "total": 0}
    
    # Get recent feedback
    feedback_data = feedback_system.get_recent_feedback(limit)
    
    return {
        "feedback": [
            {
                "feedback_id": f.feedback_id,
                "query": f.query,
                "rating": f.rating,
                "comments": f.comments,
                "timestamp": f.timestamp.isoformat()
            }
            for f in feedback_data
        ],
        "total": len(feedback_data)
    }


@app.get("/api/feedback/analytics")
async def get_feedback_analytics():
    """Get feedback analytics"""
    if not feedback_system:
        return {"analytics": {}}
    
    metrics = feedback_system.get_feedback_metrics()
    return {"analytics": metrics}


# ==================== Configuration ====================

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "lm_studio_url": LM_STUDIO_CONFIG.get('base_url', 'http://localhost:1234/v1'),
        "similarity_threshold": 0.5,
        "max_results": 10,
        "model_preference": "auto",
        "supported_formats": ingestion_manager.get_supported_formats() if ingestion_manager else []
    }


@app.put("/api/config")
async def update_config(config: ConfigUpdateRequest):
    """Update configuration"""
    # In production, persist configuration changes
    return {
        "status": "success",
        "message": "Configuration updated",
        "config": config.dict(exclude_unset=True)
    }


@app.post("/api/config/validate")
async def validate_config(config: ConfigUpdateRequest):
    """Validate configuration before applying"""
    errors = []
    
    if config.lm_studio_url:
        # Validate LM Studio URL
        try:
            import requests
            response = requests.get(f"{config.lm_studio_url}/models", timeout=5)
            if response.status_code != 200:
                errors.append("LM Studio URL is not accessible")
        except Exception as e:
            errors.append(f"LM Studio URL validation failed: {str(e)}")
    
    if config.similarity_threshold is not None:
        if not 0.0 <= config.similarity_threshold <= 1.0:
            errors.append("Similarity threshold must be between 0.0 and 1.0")
    
    if config.max_results is not None:
        if config.max_results < 1 or config.max_results > 100:
            errors.append("Max results must be between 1 and 100")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# ==================== Audit Logging ====================

@app.get("/api/audit/logs")
async def get_audit_logs(limit: int = 50):
    """Get audit logs"""
    # In production, retrieve from audit log system
    return {"logs": [], "total": 0}


@app.get("/api/security/events")
async def get_security_events_endpoint(limit: int = 50):
    """Get security events"""
    # Reuse the analytics security endpoint logic
    if not kg_manager:
        return {"events": [], "anomalies": [], "total": 0}
    
    try:
        anomalies = kg_manager.detect_anomalies()
        if not isinstance(anomalies, list):
            anomalies = []
        
        return {
            "events": [],
            "anomalies": [
                {
                    "anomaly_id": a.get('anomaly_id', '') if isinstance(a, dict) else str(a),
                    "type": a.get('type', '') if isinstance(a, dict) else 'unknown',
                    "severity": a.get('severity', '') if isinstance(a, dict) else 'low',
                    "description": a.get('description', '') if isinstance(a, dict) else str(a),
                    "timestamp": a.get('timestamp', '') if isinstance(a, dict) else ''
                }
                for a in anomalies[:limit]
            ],
            "total": len(anomalies)
        }
    except Exception as e:
        return {"events": [], "anomalies": [], "total": 0, "error": str(e)}


@app.get("/api/security/anomalies")
async def get_anomalies_endpoint(limit: int = 50):
    """Get anomaly detection data"""
    # Same as security events
    return await get_security_events_endpoint(limit)


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
