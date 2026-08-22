"""
NeuraX FastAPI application.

Thin HTTP layer over existing domain modules. Launch:

    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path for domain imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.api.errors import APIError, api_error_handler, http_exception_handler, unhandled_error_handler
from backend.api.routes import chat, documents, graph, health, search, sources, system
from backend.config import CORS_ORIGINS, UPLOAD_DIR
from backend.services.component_registry import get_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("NeuraX API starting (lazy component init)")
    yield
    logger.info("NeuraX API shutting down")
    get_registry().shutdown()


app = FastAPI(
    title="NeuraX API",
    description="Local-first multimodal RAG service layer",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(health.router)
app.include_router(system.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(chat.router)
app.include_router(sources.router)
app.include_router(graph.router)


@app.get("/")
def root():
    return {
        "service": "neurax-api",
        "docs": "/docs",
        "health": "/api/health",
    }
