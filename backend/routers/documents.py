"""
Document processing router for NeuraX
"""

import time
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel

from core.dependencies import get_neurax_service
from services.neurax_service import NeuraXService

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    status: str
    chunks_created: int
    processing_time: float


class DocumentListResponse(BaseModel):
    documents: List[dict]
    total: int


class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[dict]
    total_results: int
    search_time: float


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    include_sources: bool = True


class ChatResponse(BaseModel):
    success: bool
    response: str
    sources: List[dict]
    session_id: str
    generation_time: float


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """Upload and process a document"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file type
    allowed_extensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process document
        result = await neurax_service.ingest_document(
            file_data=file_content,
            filename=file.filename,
            content_type=file.content_type or "unknown"
        )
        
        if result['success']:
            return DocumentResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """List uploaded documents"""
    try:
        # TODO: Implement actual document listing
        # For now, return mock data
        mock_documents = [
            {
                'document_id': f'doc_{i}',
                'filename': f'document_{i}.pdf',
                'status': 'processed',
                'upload_time': time.time() - (i * 3600),  # Mock timestamps
                'chunks_count': 10 + i * 5,
                'file_size': 1024 * 1024 * (i + 1)  # Mock file sizes
            }
            for i in range(10)
        ]
        
        total = len(mock_documents)
        documents = mock_documents[offset:offset + limit]
        
        return DocumentListResponse(
            documents=documents,
            total=total
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    query: str = Form(...),
    limit: int = Form(10),
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """Search through uploaded documents"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = await neurax_service.search_documents(query, limit)
        
        if result['success']:
            return SearchResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result['error'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """Delete a document"""
    try:
        # TODO: Implement actual document deletion
        return {
            'success': True,
            'message': f'Document {document_id} deleted successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    neurax_service: NeuraXService = Depends(get_neurax_service)
):
    """Chat with the document knowledge base"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        # Search for relevant context
        search_result = await neurax_service.search_documents(request.message, limit=5)
        
        if not search_result['success']:
            raise HTTPException(status_code=500, detail=search_result['error'])
        
        context = search_result['results']
        
        # Generate response
        generation_result = await neurax_service.generate_response(
            query=request.message,
            context=context
        )
        
        if not generation_result['success']:
            raise HTTPException(status_code=500, detail=generation_result['error'])
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())[:8]
        
        return ChatResponse(
            success=True,
            response=generation_result['response'],
            sources=generation_result['sources'] if request.include_sources else [],
            session_id=session_id,
            generation_time=generation_result['generation_time']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))