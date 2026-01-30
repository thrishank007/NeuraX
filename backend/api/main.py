from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingestion_manager import IngestionManager
from retrieval.query_processor import QueryProcessor
from generation.lmstudio_generator import LMStudioGenerator
from indexing.vector_store import VectorStore

app = FastAPI(title="NeuraX API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
ingestion_manager = IngestionManager()
query_processor = QueryProcessor()
vector_store = VectorStore()

@app.get("/")
async def root():
    return {"message": "NeuraX API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "database": "connected",
            "lm_studio": "checking..."
        }
    }

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process files"""
    try:
        results = []
        for file in files:
            # Save file
            file_path = f"./data/uploads/{file.filename}"
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process file
            result = ingestion_manager.process_file(file_path)
            results.append({
                "filename": file.filename,
                "status": "processed",
                "result": result
            })
        
        return JSONResponse({"status": "success", "files": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def process_query(query: dict):
    """Process multimodal query"""
    try:
        query_text = query.get("text", "")
        query_type = query.get("type", "text")
        
        # Process query
        results = query_processor.process(query_text)
        
        return JSONResponse({
            "status": "success",
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/files")
async def list_files():
    """List uploaded files"""
    try:
        upload_dir = "./data/uploads"
        files = os.listdir(upload_dir) if os.path.exists(upload_dir) else []
        return JSONResponse({"status": "success", "files": files})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)