import sys
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import shutil

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main_launcher import SecureInsightLauncher

app = FastAPI(title="NeuraX API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

launcher = SecureInsightLauncher()

@app.on_event("startup")
async def startup_event():
    # Run initialization in a separate thread to avoid blocking
    def init():
        if not launcher.validate_system():
            print("System validation failed")
        if not launcher.initialize_components():
            print("Component initialization failed")
    
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init)

@app.get("/api/health")
async def health_check():
    return launcher._perform_health_check()

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    for file in files:
        temp_path = upload_dir / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Process file using ingestion manager
            result = launcher.ingestion_manager.process_file(str(temp_path))
            if result:
                # Add to vector store
                if launcher.vector_store and launcher.embedding_manager:
                    # Depending on the content type, we might need to embed text or image
                    if result.get('content'):
                        embedding = launcher.embedding_manager.embed_text(result['content'])
                        result['embedding'] = embedding
                    elif result.get('image_path'):
                        embedding = launcher.embedding_manager.embed_image(result['image_path'])
                        result['embedding'] = embedding
                        
                    launcher.vector_store.add_document(result)
                results.append({"filename": file.filename, "status": "success", "id": result.get('metadata', {}).get('document_id')})
            else:
                results.append({"filename": file.filename, "status": "error", "message": "Processing failed"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
        finally:
            # We might want to keep the file if it's referenced by metadata, 
            # but for now let's follow the system's logic.
            # Some processors might move the file or store it.
            pass
            
    return results

@app.get("/api/files")
async def list_files():
    # If the vector store doesn't support listing, we'll return an empty list or cached list
    return {"files": []}

@app.post("/api/query")
async def process_query(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    try:
        result = {}
        if image:
            upload_dir = Path("uploads/queries")
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_image_path = upload_dir / image.filename
            with open(temp_image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            
            result = launcher.query_processor.process_multimodal_query(query, str(temp_image_path))
        else:
            result = launcher.query_processor.process_text_query(query)
        
        # Generate response using LLM if available
        if launcher.llm_generator and result.get('results'):
            context = result['results']
            llm_response = launcher.llm_generator.generate_grounded_response(query, context)
            result['generated_response'] = llm_response.response_text
            
            # Generate citations
            if launcher.citation_generator:
                citations = launcher.citation_generator.generate_citations(llm_response.response_text, context)
                result['citations'] = citations
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/metrics")
async def get_metrics():
    if launcher.metrics_collector:
        return launcher.metrics_collector.get_all_metrics()
    return {}

@app.get("/api/knowledge-graph")
async def get_knowledge_graph():
    if launcher.kg_manager:
        return launcher.kg_manager.get_graph_stats()
    return {}

@app.post("/api/feedback")
async def submit_feedback(feedback: Dict[str, Any] = Body(...)):
    if launcher.feedback_system:
        launcher.feedback_system.collect_feedback(**feedback)
        return {"status": "success"}
    return {"status": "error", "message": "Feedback system not available"}

@app.get("/api/config")
async def get_config():
    try:
        from config import LLM_CONFIG, SEARCH_CONFIG
        return {
            "llm": LLM_CONFIG,
            "search": SEARCH_CONFIG
        }
    except ImportError:
        return {}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
