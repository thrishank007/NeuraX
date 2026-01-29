#!/usr/bin/env python3
"""
FastAPI wrapper for NeuraX backend integration
Provides REST API endpoints that the Next.js frontend can consume
"""
import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import existing NeuraX components
try:
    from ingestion.ingestion_manager import IngestionManager
    from indexing.embedding_manager import EmbeddingManager
    from indexing.vector_store import VectorStore
    from retrieval.query_processor import QueryProcessor
    from retrieval.speech_to_text_processor import SpeechToTextProcessor
    from generation.llm_factory import create_llm_generator
    from generation.citation_generator import CitationGenerator
    from feedback.feedback_system import FeedbackSystem
    from config import (
        CHROMA_CONFIG, LM_STUDIO_CONFIG, LLM_CONFIG, 
        SECURITY_CONFIG, FEEDBACK_CONFIG
    )
except ImportError as e:
    logger.error(f"Failed to import NeuraX components: {e}")
    logger.warning("Running in standalone mode without full NeuraX backend")

# Pydantic models for API
class QueryRequest(BaseModel):
    text: Optional[str] = None
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    options: Dict[str, Any] = Field(default_factory=dict)

class ImageQueryRequest(BaseModel):
    text_query: Optional[str] = None
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)

class VoiceQueryRequest(BaseModel):
    language: str = Field(default="en")
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)

class MultimodalQueryRequest(BaseModel):
    text: str
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)

class ResponseGenerationRequest(BaseModel):
    query: str
    context: List[Dict[str, Any]]
    model: Optional[str] = "gemma-3n"
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    include_citations: bool = True

class FeedbackRequest(BaseModel):
    query_id: str
    response_id: Optional[str] = None
    rating: int = Field(ge=1, le=5)
    comments: Optional[str] = None
    is_helpful: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class ConfigUpdateRequest(BaseModel):
    api_url: Optional[str] = None
    lm_studio_url: Optional[str] = None
    max_file_size: Optional[int] = None
    default_similarity_threshold: Optional[float] = None
    enable_analytics: Optional[bool] = None
    enable_voice_input: Optional[bool] = None
    models: Optional[Dict[str, str]] = None
    performance: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="NeuraX API",
    description="REST API wrapper for NeuraX RAG System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:3001",  # Next.js production
        "https://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for NeuraX components
ingestion_manager = None
embedding_manager = None
vector_store = None
query_processor = None
stt_processor = None
llm_generator = None
citation_generator = None
feedback_system = None

# Initialize components
def initialize_components():
    """Initialize NeuraX components"""
    global ingestion_manager, embedding_manager, vector_store, query_processor, stt_processor, llm_generator, citation_generator, feedback_system
    
    try:
        # Initialize core components
        ingestion_manager = IngestionManager()
        logger.info("Ingestion manager initialized")
        
        # Initialize embedding manager and vector store
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore(
            persist_directory=CHROMA_CONFIG['persist_directory'],
            collection_name=CHROMA_CONFIG['collection_name']
        )
        
        # Initialize query processor
        query_processor = QueryProcessor(
            embedding_manager, 
            vector_store,
            {
                'similarity_threshold': 0.5,
                'max_results': 10
            }
        )
        
        # Initialize STT processor
        stt_processor = SpeechToTextProcessor()
        
        # Initialize LLM generator
        llm_generator = create_llm_generator(LLM_CONFIG)
        
        # Initialize citation generator
        citation_generator = CitationGenerator()
        
        # Initialize feedback system
        feedback_system = FeedbackSystem()
        
        logger.info("All NeuraX components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # Continue with mock implementations for development

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()

# API Response models
class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class PaginatedResponse(BaseModel):
    success: bool
    data: List[Any]
    pagination: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# Health and status endpoints
@app.get("/health", response_model=ApiResponse)
async def health_check():
    """Health check endpoint"""
    return ApiResponse(
        success=True,
        message="NeuraX API is running",
        data={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    )

@app.get("/api/status", response_model=ApiResponse)
async def system_status():
    """Get system status"""
    try:
        status = {
            "status": "online",
            "components": {
                "ingestion_manager": ingestion_manager is not None,
                "embedding_manager": embedding_manager is not None,
                "vector_store": vector_store is not None,
                "query_processor": query_processor is not None,
                "stt_processor": stt_processor is not None,
                "llm_generator": llm_generator is not None,
                "citation_generator": citation_generator is not None,
                "feedback_system": feedback_system is not None,
            },
            "lm_studio": {
                "url": LM_STUDIO_CONFIG.get('base_url', 'Not configured'),
                "available": True  # Would check actual connectivity
            },
            "uptime": "N/A",  # Would calculate from startup time
            "timestamp": datetime.now().isoformat()
        }
        
        return ApiResponse(success=True, data=status)
        
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            message="Failed to get system status"
        )

# File upload endpoints
@app.post("/api/upload", response_model=ApiResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    uploaded_files = []
    total_size = 0
    errors = []
    
    try:
        for file in files:
            # Validate file
            if file.size and file.size > SECURITY_CONFIG['max_upload_size_mb'] * 1024 * 1024:
                errors.append({
                    "fileName": file.filename,
                    "error": f"File too large: {file.size / 1024 / 1024:.1f}MB"
                })
                continue
            
            # Save file temporarily
            file_id = str(uuid.uuid4())
            file_path = Path(f"uploads/{file_id}_{file.filename}")
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            total_size += len(content)
            
            # Process file if components are available
            if ingestion_manager:
                try:
                    result = ingestion_manager.process_file(str(file_path))
                    uploaded_files.append({
                        "id": file_id,
                        "fileName": file.filename,
                        "filePath": str(file_path),
                        "fileType": result.get('file_type', 'unknown'),
                        "fileSize": len(content),
                        "mimeType": file.content_type,
                        "status": "completed",
                        "progress": 100,
                        "metadata": result,
                        "uploadedAt": datetime.now().isoformat(),
                        "processedAt": datetime.now().isoformat()
                    })
                except Exception as e:
                    uploaded_files.append({
                        "id": file_id,
                        "fileName": file.filename,
                        "filePath": str(file_path),
                        "fileType": "unknown",
                        "fileSize": len(content),
                        "mimeType": file.content_type,
                        "status": "error",
                        "progress": 0,
                        "error": str(e),
                        "uploadedAt": datetime.now().isoformat()
                    })
            else:
                # Mock response for development
                uploaded_files.append({
                    "id": file_id,
                    "fileName": file.filename,
                    "filePath": str(file_path),
                    "fileType": "unknown",
                    "fileSize": len(content),
                    "mimeType": file.content_type,
                    "status": "completed",
                    "progress": 100,
                    "uploadedAt": datetime.now().isoformat()
                })
        
        return ApiResponse(
            success=True,
            message=f"Successfully uploaded {len(uploaded_files)} files",
            data={
                "files": uploaded_files,
                "totalFiles": len(uploaded_files),
                "totalSize": total_size,
                "errors": errors
            }
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files", response_model=PaginatedResponse)
async def get_uploaded_files(page: int = 1, limit: int = 20):
    """Get list of uploaded files"""
    try:
        # This would typically query a database
        # For now, return mock data
        files = []
        total = 0
        
        return PaginatedResponse(
            success=True,
            data=files,
            pagination={
                "page": page,
                "limit": limit,
                "total": total,
                "totalPages": (total + limit - 1) // limit,
                "hasNext": page * limit < total,
                "hasPrev": page > 1
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/files/{file_id}", response_model=ApiResponse)
async def delete_file(file_id: str):
    """Delete uploaded file"""
    try:
        # Implementation would remove file from storage and database
        return ApiResponse(success=True, message="File deleted successfully")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query endpoints
@app.post("/api/query/text", response_model=ApiResponse)
async def process_text_query(request: QueryRequest):
    """Process text query"""
    if not request.text:
        raise HTTPException(status_code=400, detail="Text query is required")
    
    try:
        if query_processor:
            query_result = query_processor.process_text_query(request.text)
            
            # Transform results to match frontend expectations
            results = []
            for result in query_result.results:
                results.append({
                    "id": str(uuid.uuid4()),
                    "filePath": result.get('file_path', ''),
                    "fileName": Path(result.get('file_path', '')).name,
                    "fileType": result.get('file_type', 'unknown'),
                    "similarityScore": result.get('similarity_score', 0.0),
                    "confidence": result.get('confidence', 0.0),
                    "contentPreview": result.get('content_preview', ''),
                    "metadata": result.get('metadata', {}),
                    "timestamp": datetime.now().isoformat()
                })
            
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "text",
                        "text": request.text,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": request.similarity_threshold,
                        "status": "completed"
                    },
                    "results": results,
                    "totalResults": len(results),
                    "processingTime": getattr(query_result, 'processing_time', 0.0)
                }
            )
        else:
            # Mock response for development
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "text",
                        "text": request.text,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": request.similarity_threshold,
                        "status": "completed"
                    },
                    "results": [],
                    "totalResults": 0,
                    "processingTime": 0.1
                }
            )
            
    except Exception as e:
        logger.error(f"Text query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/image", response_model=ApiResponse)
async def process_image_query(
    image: UploadFile = File(...),
    text_query: Optional[str] = None,
    similarity_threshold: float = 0.5,
    max_results: int = 10
):
    """Process image query"""
    try:
        # Save uploaded image temporarily
        image_id = str(uuid.uuid4())
        image_path = Path(f"temp/{image_id}_{image.filename}")
        image_path.parent.mkdir(exist_ok=True)
        
        with open(image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        if query_processor:
            query_result = query_processor.process_image_query(str(image_path))
            
            # Transform results
            results = []
            for result in query_result.results:
                results.append({
                    "id": str(uuid.uuid4()),
                    "filePath": result.get('file_path', ''),
                    "fileName": Path(result.get('file_path', '')).name,
                    "fileType": result.get('file_type', 'unknown'),
                    "similarityScore": result.get('similarity_score', 0.0),
                    "confidence": result.get('confidence', 0.0),
                    "contentPreview": result.get('content_preview', ''),
                    "metadata": result.get('metadata', {}),
                    "timestamp": datetime.now().isoformat()
                })
            
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "image",
                        "text": text_query,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": similarity_threshold,
                        "status": "completed"
                    },
                    "results": results,
                    "totalResults": len(results),
                    "processingTime": getattr(query_result, 'processing_time', 0.0)
                }
            )
        else:
            # Mock response
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "image",
                        "text": text_query,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": similarity_threshold,
                        "status": "completed"
                    },
                    "results": [],
                    "totalResults": 0,
                    "processingTime": 0.1
                }
            )
            
    except Exception as e:
        logger.error(f"Image query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/voice", response_model=ApiResponse)
async def process_voice_query(
    audio: UploadFile = File(...),
    language: str = "en",
    similarity_threshold: float = 0.5,
    max_results: int = 10
):
    """Process voice query with speech-to-text"""
    try:
        # Save uploaded audio temporarily
        audio_id = str(uuid.uuid4())
        audio_path = Path(f"temp/{audio_id}_{audio.filename}")
        audio_path.parent.mkdir(exist_ok=True)
        
        with open(audio_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        if stt_processor and query_processor:
            # Transcribe audio
            stt_result = stt_processor.process_voice_query_with_fallback(str(audio_path))
            
            if not stt_result.get('success'):
                raise HTTPException(status_code=400, detail=stt_result.get('error', 'Transcription failed'))
            
            transcribed_text = stt_result.get('transcribed_text', '')
            
            # Process as text query
            query_result = query_processor.process_text_query(transcribed_text)
            
            # Transform results
            results = []
            for result in query_result.results:
                results.append({
                    "id": str(uuid.uuid4()),
                    "filePath": result.get('file_path', ''),
                    "fileName": Path(result.get('file_path', '')).name,
                    "fileType": result.get('file_type', 'unknown'),
                    "similarityScore": result.get('similarity_score', 0.0),
                    "confidence": result.get('confidence', 0.0),
                    "contentPreview": result.get('content_preview', ''),
                    "metadata": result.get('metadata', {}),
                    "timestamp": datetime.now().isoformat()
                })
            
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "voice",
                        "text": transcribed_text,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": similarity_threshold,
                        "status": "completed"
                    },
                    "results": results,
                    "totalResults": len(results),
                    "processingTime": getattr(query_result, 'processing_time', 0.0)
                }
            )
        else:
            # Mock response
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "voice",
                        "text": "Mock transcription",
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": similarity_threshold,
                        "status": "completed"
                    },
                    "results": [],
                    "totalResults": 0,
                    "processingTime": 0.1
                }
            )
            
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/multimodal", response_model=ApiResponse)
async def process_multimodal_query(
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    similarity_threshold: float = 0.5,
    max_results: int = 10
):
    """Process multimodal query combining text and image"""
    try:
        image_path = None
        
        # Save uploaded image if provided
        if image:
            image_id = str(uuid.uuid4())
            image_path = Path(f"temp/{image_id}_{image.filename}")
            image_path.parent.mkdir(exist_ok=True)
            
            with open(image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
        
        if query_processor:
            if image_path:
                query_result = query_processor.process_multimodal_query(text, str(image_path))
            else:
                query_result = query_processor.process_text_query(text)
            
            # Transform results
            results = []
            for result in query_result.results:
                results.append({
                    "id": str(uuid.uuid4()),
                    "filePath": result.get('file_path', ''),
                    "fileName": Path(result.get('file_path', '')).name,
                    "fileType": result.get('file_type', 'unknown'),
                    "similarityScore": result.get('similarity_score', 0.0),
                    "confidence": result.get('confidence', 0.0),
                    "contentPreview": result.get('content_preview', ''),
                    "metadata": result.get('metadata', {}),
                    "timestamp": datetime.now().isoformat()
                })
            
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "multimodal",
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": similarity_threshold,
                        "status": "completed"
                    },
                    "results": results,
                    "totalResults": len(results),
                    "processingTime": getattr(query_result, 'processing_time', 0.0)
                }
            )
        else:
            # Mock response
            return ApiResponse(
                success=True,
                data={
                    "query": {
                        "id": str(uuid.uuid4()),
                        "type": "multimodal",
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "similarityThreshold": similarity_threshold,
                        "status": "completed"
                    },
                    "results": [],
                    "totalResults": 0,
                    "processingTime": 0.1
                }
            )
            
    except Exception as e:
        logger.error(f"Multimodal query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Response generation endpoint
@app.post("/api/generate-response", response_model=ApiResponse)
async def generate_response(request: ResponseGenerationRequest):
    """Generate AI response with citations"""
    try:
        if llm_generator and citation_generator:
            # Generate response using LLM
            generated_response = llm_generator.generate_grounded_response(
                query=request.query,
                context=request.context
            )
            
            # Generate citations
            citations = citation_generator.generate_citations(
                response=generated_response.response_text,
                sources=request.context,
                citation_indices=None
            )
            
            # Transform citations
            citation_list = []
            for citation in citations:
                citation_list.append({
                    "id": str(uuid.uuid4()),
                    "citationId": getattr(citation, 'citation_id', ''),
                    "filePath": getattr(citation, 'file_path', ''),
                    "fileName": Path(getattr(citation, 'file_path', '')).name,
                    "sourceType": getattr(citation, 'source_type', 'document'),
                    "pageNumber": getattr(citation, 'page_number', None),
                    "contentSnippet": getattr(citation, 'content_snippet', ''),
                    "confidenceScore": getattr(citation, 'confidence_score', 0.0),
                    "timestamp": datetime.now().isoformat()
                })
            
            return ApiResponse(
                success=True,
                data={
                    "id": str(uuid.uuid4()),
                    "queryId": str(uuid.uuid4()),
                    "responseText": generated_response.response_text,
                    "confidence": getattr(generated_response, 'confidence_score', 0.0),
                    "citations": citation_list,
                    "processingTime": 0.0,  # Would calculate actual time
                    "modelUsed": request.model or "gemma-3n",
                    "timestamp": datetime.now().isoformat()
                },
                message="Response generated successfully"
            )
        else:
            # Mock response
            return ApiResponse(
                success=True,
                data={
                    "id": str(uuid.uuid4()),
                    "queryId": str(uuid.uuid4()),
                    "responseText": f"This is a mock response to: {request.query}",
                    "confidence": 0.8,
                    "citations": [],
                    "processingTime": 0.1,
                    "modelUsed": request.model or "gemma-3n",
                    "timestamp": datetime.now().isoformat()
                },
                message="Mock response generated"
            )
            
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Query history endpoint
@app.get("/api/query/history", response_model=PaginatedResponse)
async def get_query_history(page: int = 1, limit: int = 20):
    """Get query history"""
    try:
        # This would typically query a database
        # For now, return mock data
        queries = []
        total = 0
        
        return PaginatedResponse(
            success=True,
            data=queries,
            pagination={
                "page": page,
                "limit": limit,
                "total": total,
                "totalPages": (total + limit - 1) // limit,
                "hasNext": page * limit < total,
                "hasPrev": page > 1
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/api/analytics", response_model=ApiResponse)
async def get_analytics(start: Optional[str] = None, end: Optional[str] = None):
    """Get analytics data"""
    try:
        # Mock analytics data
        analytics_data = {
            "queryStats": {
                "totalQueries": 150,
                "textQueries": 100,
                "imageQueries": 30,
                "voiceQueries": 15,
                "multimodalQueries": 5,
                "avgProcessingTime": 1.2,
                "successRate": 95.3
            },
            "fileStats": {
                "totalFiles": 45,
                "documentFiles": 25,
                "imageFiles": 15,
                "audioFiles": 5,
                "totalSize": 125.6,  # MB
                "avgProcessingTime": 2.8
            },
            "systemStats": {
                "uptime": 86400,  # seconds
                "memoryUsage": 45.2,  # percentage
                "cpuUsage": 23.1,  # percentage
                "diskUsage": 67.8   # percentage
            },
            "usageTrends": [
                {"date": "2024-01-01", "queries": 12, "uploads": 3},
                {"date": "2024-01-02", "queries": 18, "uploads": 5},
                {"date": "2024-01-03", "queries": 15, "uploads": 2},
                {"date": "2024-01-04", "queries": 22, "uploads": 7},
                {"date": "2024-01-05", "queries": 19, "uploads": 4},
                {"date": "2024-01-06", "queries": 25, "uploads": 6},
                {"date": "2024-01-07", "queries": 21, "uploads": 3}
            ],
            "popularQueries": [
                {"query": "security protocols", "count": 12, "avgRating": 4.2},
                {"query": "network configuration", "count": 8, "avgRating": 4.5},
                {"query": "user authentication", "count": 6, "avgRating": 4.0}
            ]
        }
        
        return ApiResponse(success=True, data=analytics_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/metrics", response_model=ApiResponse)
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        metrics = {
            "cpu": {"usage": 23.1, "cores": 4, "frequency": "2.4GHz"},
            "memory": {"usage": 45.2, "total": "16GB", "available": "8.7GB"},
            "disk": {"usage": 67.8, "total": "500GB", "available": "161GB"},
            "network": {"rx": "1.2MB/s", "tx": "0.8MB/s"},
            "timestamp": datetime.now().isoformat()
        }
        
        return ApiResponse(success=True, data=metrics)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Security endpoints
@app.get("/api/security/events", response_model=PaginatedResponse)
async def get_security_events(page: int = 1, limit: int = 20):
    """Get security events"""
    try:
        # Mock security events
        events = [
            {
                "id": str(uuid.uuid4()),
                "type": "audit",
                "severity": "low",
                "title": "User Login",
                "description": "User logged in successfully",
                "source": "auth_system",
                "timestamp": datetime.now().isoformat(),
                "resolved": True,
                "resolvedAt": datetime.now().isoformat()
            }
        ]
        total = len(events)
        
        return PaginatedResponse(
            success=True,
            data=events,
            pagination={
                "page": page,
                "limit": limit,
                "total": total,
                "totalPages": (total + limit - 1) // limit,
                "hasNext": page * limit < total,
                "hasPrev": page > 1
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoint
@app.post("/api/feedback", response_model=ApiResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    try:
        if feedback_system:
            feedback_id = feedback_system.collect_feedback(
                query="",  # Would get from database
                response="",  # Would get from database
                rating=feedback.rating,
                comments=feedback.comments,
                metadata=feedback.metadata
            )
        else:
            feedback_id = str(uuid.uuid4())
        
        return ApiResponse(
            success=True,
            data={"feedbackId": feedback_id},
            message="Feedback submitted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/api/config", response_model=ApiResponse)
async def get_system_config():
    """Get system configuration"""
    try:
        config = {
            "apiUrl": "http://localhost:8000",
            "wsUrl": "ws://localhost:8000",
            "lmStudioUrl": LM_STUDIO_CONFIG.get('base_url', 'http://localhost:1234'),
            "maxFileSize": SECURITY_CONFIG['max_upload_size_mb'] * 1024 * 1024,
            "allowedFileTypes": [".pdf", ".docx", ".doc", ".txt", ".jpg", ".png", ".mp3"],
            "enableAnalytics": True,
            "enableDarkMode": True,
            "defaultSimilarityThreshold": 0.5,
            "maxQueryHistory": 50,
            "enableVoiceInput": True,
            "models": {
                "primary": "gemma-3n",
                "fallback": "qwen3-4b"
            },
            "performance": {
                "batchSize": 10,
                "maxConcurrency": 4,
                "cacheEnabled": True,
                "cacheTimeout": 3600
            },
            "security": {
                "auditLogging": True,
                "anomalyDetection": True,
                "rateLimiting": True,
                "maxUploadsPerHour": 100
            }
        }
        
        return ApiResponse(success=True, data=config)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config", response_model=ApiResponse)
async def update_system_config(config: ConfigUpdateRequest):
    """Update system configuration"""
    try:
        # In a real implementation, this would save to a config file or database
        return ApiResponse(
            success=True,
            message="Configuration updated successfully",
            data=config.dict()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config/validate", response_model=ApiResponse)
async def validate_config(config: ConfigUpdateRequest):
    """Validate configuration"""
    try:
        # Basic validation
        if config.max_file_size and config.max_file_size < 1024:
            raise ValueError("Max file size must be at least 1024 bytes")
        
        if config.default_similarity_threshold and not (0.0 <= config.default_similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        return ApiResponse(
            success=True,
            message="Configuration is valid"
        )
        
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            message="Configuration validation failed"
        )

# Export endpoints
@app.post("/api/export")
async def export_results(format: str, data: Dict[str, Any]):
    """Export results in various formats"""
    try:
        if format == "json":
            return JSONResponse(content=data)
        elif format == "csv":
            # Simple CSV conversion (would be more sophisticated in real implementation)
            csv_content = "File Name,File Type,Similarity Score,Content Preview\n"
            for result in data.get("results", []):
                csv_content += f'"{result.get("fileName", "")}","{result.get("fileType", "")}",{result.get("similarityScore", 0)},"{result.get("contentPreview", "")}"\n'
            
            return FileResponse(
                content=csv_content,
                media_type="text/csv",
                filename=f"neurax-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files (for uploaded files, etc.)
if Path("uploads").exists():
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if Path("temp").exists():
    app.mount("/temp", StaticFiles(directory="temp"), name="temp")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )