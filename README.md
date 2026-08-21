# NeuraX - Offline Multimodal RAG System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

## Overview
NeuraX is a production-ready offline multimodal Retrieval-Augmented Generation (RAG) system designed for NTRO's SIH 2025 problem statement. It provides secure, air-gapped document intelligence with advanced multimodal capabilities and enterprise-grade security features.

## Demo:

[![Watch the video](https://img.youtube.com/vi/2qcBRtBl5q8/0.jpg)](https://youtu.be/2qcBRtBl5q8)

## ✨ Key Features

### 🔒 **Security & Privacy**
- **Complete Offline Operation**: Zero internet dependencies, air-gapped deployment
- **Knowledge Graph Security**: Real-time anomaly detection and tamper protection
- **Audit Logging**: Comprehensive activity tracking and compliance monitoring
- **Data Sovereignty**: All processing occurs locally with no external API calls

### 🤖 **Advanced AI Capabilities**
- **Multimodal Understanding**: Process documents, images, and audio seamlessly
- **Cross-Modal Search**: Find relevant content across different data types
- **LM Studio Integration**: Local LLM hosting with Gemma 3n (multimodal) and Qwen3 4B (reasoning)
- **CLIP Embeddings**: State-of-the-art visual-text similarity matching
- **Intelligent Citations**: Numbered references with confidence scores and expandable sources

### 📁 **Comprehensive Format Support**
- **Documents**: PDF, DOCX, DOC, TXT with OCR fallback
- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP with visual similarity search
- **Audio**: WAV, MP3, M4A, FLAC, OGG with speech-to-text processing
- **Batch Processing**: Handle multiple files simultaneously with progress tracking

### 🚀 **Production Features**
- **Auto-Deployment**: One-click executable generation with PyInstaller
- **USB Portability**: Export complete system to USB for air-gapped deployment
- **Performance Optimization**: Memory-efficient processing with GPU acceleration
- **Error Resilience**: Graceful degradation and comprehensive error handling
- **Real-time Feedback**: User feedback collection and performance metrics

## 🏗️ Architecture

```text
Next.js frontend (frontend/, :3000)
    ↓ HTTP / SSE
FastAPI service layer (backend/, :8000)
    ↓
Existing Python domain modules (ingestion, indexing, retrieval, generation)
    ↓
ChromaDB · embeddings · Whisper · CLIP · LM Studio
```

**Product UI is Next.js only.** Streamlit (`:8501`) is optional analytics, not the main interface.

### Core Components
- **LM Studio Integration**: Local LLM server for multimodal and reasoning tasks
- **ChromaDB**: Persistent vector database for semantic search
- **CLIP Embeddings**: Visual-text cross-modal understanding
- **Whisper STT**: Speech-to-text for audio processing
- **NetworkX**: Security-focused knowledge graph (anomaly / tamper monitoring)
- **Graphify (optional)**: Document knowledge graph for corpus intelligence (CLI integration)
- **FastAPI**: Thin HTTP API over domain modules
- **Next.js UI**: Primary workspace (Chat, Documents, Sources, Graph, Settings)
- **Streamlit Dashboard**: Optional analytics only

## 🛠️ System Requirements

### Minimum Requirements
- **Python**: 3.8+ (3.9+ recommended for optimal performance)
- **Memory**: 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: 5GB free space (models are managed via LM Studio)
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### Recommended Setup
- **Memory**: 16GB+ RAM for smooth operation
- **GPU**: 6GB+ VRAM for accelerated processing (CPU fallback available)
- **Storage**: 10GB+ for cache and data processing
- **Network**: None required during operation (offline-first design)

### Dependencies
- **LM Studio**: For local LLM hosting (Gemma 3n + Qwen3 4B)
- **Tesseract OCR**: For document text extraction (auto-bundled)
- **FFmpeg**: For audio processing (platform-specific installation)
- **Graphify (optional)**: Document knowledge graph CLI — see [Knowledge Graph (Graphify)](#-knowledge-graph-graphify-document-intelligence)

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/thrishank007/NeuraX.git
cd NeuraX

# Run automated setup
python install_dependencies.py

# Setup LM Studio integration
python migrate_to_lmstudio.py

# Launch product UI (Next.js + FastAPI)
pwsh scripts/dev.ps1
```

### Next.js + FastAPI (product UI)

```bash
# Backend (repo root, venv active)
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# Frontend
cd frontend
cp .env.local.example .env.local   # Windows: copy .env.local.example .env.local
npm install
npm run dev
```

Or start both (Windows):

```powershell
pwsh scripts/dev.ps1
```

| Surface | URL |
|---|---|
| Next.js workspace | http://127.0.0.1:3000 |
| FastAPI | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |
| Streamlit (optional) | http://127.0.0.1:8501 |

Environment templates: [`.env.example`](.env.example), [`frontend/.env.local.example`](frontend/.env.local.example).

## 🕸️ Knowledge Graph (Graphify document intelligence)

NeuraX has **two different graph layers**. They are not interchangeable.

| Layer | Module | Purpose |
|---|---|---|
| **Security graph** | `kg_security/knowledge_graph_manager.py` (NetworkX) | Anomaly detection, tamper monitoring, security analytics |
| **Document graph** | `kg_security/graphify_service.py` + Graphify CLI | Build / query a knowledge graph over uploaded documents |

The product **Knowledge Graph** page (`/graph`) is the Graphify document-intelligence UI. The security graph remains available for analytics/export and is **not** removed or renamed.

### Why Graphify is optional

- NeuraX supports older Python runtimes in some deployments.
- Graphify (`graphifyy`) requires **Python 3.10+**.
- Therefore Graphify is **never** a mandatory import-time dependency.
- NeuraX talks to Graphify only through its **installed CLI** (`subprocess`, never `shell=True`).
- If Graphify is missing, NeuraX still launches; the UI shows install guidance.

### Install Graphify (separate from NeuraX)

```bash
# Recommended: isolated tool install
uv tool install "graphifyy[openai]"
# or
pipx install "graphifyy[openai]"
```

Verify:

```bash
graphify --version
graphify extract --help
```

Optional: point NeuraX at a custom binary:

```bash
# Windows PowerShell
$env:GRAPHIFY_EXECUTABLE = "C:\path\to\graphify.exe"
```

### LM Studio setup for Graphify

Graphify semantic extraction uses the same local OpenAI-compatible endpoint as NeuraX:

1. Start LM Studio and serve a chat model (e.g. Qwen) on `http://localhost:1234/v1`.
2. NeuraX sets `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` for Graphify child processes.
3. By default only **loopback** endpoints are allowed (`localhost`, `127.0.0.1`, `::1`).

### Configuration (`GRAPHIFY_CONFIG` in `config.py`)

| Key | Meaning |
|---|---|
| `enabled` | Master switch for Graphify integration |
| `executable` / `GRAPHIFY_EXECUTABLE` | CLI name or absolute path |
| `workspace_dir` | `data/graphify/` managed workspaces |
| `base_url` | LM Studio OpenAI-compatible base URL |
| `api_key` / `GRAPHIFY_OPENAI_API_KEY` | Dummy local key (default `lm-studio`) |
| `model` / `GRAPHIFY_MODEL` | Model id for extraction |
| `mode` | e.g. `deep` when supported by CLI |
| `auto_update_after_ingestion` | If true, run incremental update after upload batch |
| `max_concurrency` | Passed to extract when supported |
| `api_timeout_seconds` / `process_timeout_seconds` | Timeouts |
| `allow_non_local_endpoint` | Must be true to allow non-loopback model URLs |
| `max_rag_context_*` | Caps for optional graph-enhanced RAG |

### Workflow

1. Upload documents in the **Documents** page (vector indexing runs as usual).
2. Successfully processed files are copied into a **managed Graphify corpus** under `data/graphify/default/corpus/` (sanitized names, SHA-256 manifest, no path traversal).
3. Open **Knowledge Graph** and choose:
   - **Build Knowledge Graph** — extract from corpus
   - **Update Knowledge Graph** — incremental update when available
   - **Rebuild From Scratch** — clear artifacts and re-extract
4. View stats, filter by type/community/source/confidence, run **query / explain / path**.
5. Download artifacts: `graph.html`, `graph.json`, `GRAPH_REPORT.md`.
6. In **Chat**, optionally enable **Use Knowledge Graph Context** to append compact graph relationships to vector RAG (never replaces ChromaDB retrieval). Inferred edges are labeled `EXTRACTED` / `INFERRED` / `AMBIGUOUS`.

### Offline / privacy

- Graph builds call **local** LM Studio only (default).
- Corpus and artifacts stay under `data/graphify/` (gitignored).
- No Graphify package code is imported into core NeuraX modules.

### Troubleshooting

| Symptom | What to do |
|---|---|
| Graphify executable not found | Install with `uv tool install "graphifyy[openai]"` or `pipx install "graphifyy[openai]"`; ensure `graphify` is on `PATH` or set `GRAPHIFY_EXECUTABLE` |
| Unsupported Python for Graphify | Use Python 3.10+ for the Graphify tool only; NeuraX can keep an older runtime |
| LM Studio unavailable | Start LM Studio server on the configured base URL; load a chat model |
| Empty graph | Ensure corpus has files (upload + process documents first), then rebuild |
| Graph build timeout | Increase `process_timeout_seconds` / `api_timeout_seconds`; reduce corpus size or concurrency |
| Corrupted `graph.json` | Rebuild from scratch; check disk space and LM Studio logs |

### Tests

```bash
# Graphify unit + integration (uses a fake CLI; Graphify package not required)
pytest tests/test_graphify_service.py tests/test_graphify_regression.py -q

# Existing API suite
pytest backend/tests -q
```

### Option 2: Manual Installation
```bash
# Clone repository
git clone https://github.com/thrishank007/NeuraX.git
cd NeuraX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (platform-specific)
# Ubuntu/Debian: sudo apt-get install tesseract-ocr ffmpeg
# macOS: brew install tesseract ffmpeg
# Windows: Automated via install_dependencies.py

# Launch system
python main_launcher.py
```

### Option 3: Portable Executable
```bash
# Build portable executable
python build_executables.py

# Deploy to USB or air-gapped system
# Executable will be in packages/ directory
```

## 🎯 LM Studio Setup (Required)

NeuraX uses LM Studio for local LLM hosting, providing better performance and easier model management:

### 1. Install LM Studio
- Download from [https://lmstudio.ai/](https://lmstudio.ai/)
- Install and launch the application

### 2. Download Models
In LM Studio, search for and download:
- **Gemma 3n**: For multimodal queries (text + images)
- **Qwen3 4B Thinking 2507**: For complex reasoning tasks

### 3. Start Local Server
1. Go to "Local Server" tab in LM Studio
2. Load your preferred model (Gemma for multimodal, Qwen for reasoning)
3. Start server on `localhost:1234`
4. Verify server is running with green status indicator

### 4. Test Integration
```bash
python test_lmstudio_integration.py
```

## 💻 Usage Examples

### Basic Document Processing
```python
# Upload documents via the Next.js Documents workspace
# Supported: PDF, DOCX, DOC, TXT files
# Automatic text extraction and indexing

# Query via Chat workspace or POST /api/chat
query = "What are the main findings in the research?"
# System returns relevant passages with citations
```

### Multimodal Search
```python
# Upload images along with documents
# Supported: JPG, PNG, BMP, TIFF, WEBP

# Cross-modal queries
query = "Find documents related to this chart"
# System matches visual content with textual descriptions
```

### Audio Processing
```python
# Upload audio files
# Supported: WAV, MP3, M4A, FLAC, OGG

# Audio-to-text search
query = "What was discussed about budget planning?"
# System transcribes audio and searches content
```

## 📂 Project Structure
```
NeuraX/
├── 📁 ingestion/              # Multimodal data processors
│   ├── document_processor.py  # PDF, DOCX, DOC, TXT processing
│   ├── image_processor.py     # Image analysis and OCR
│   ├── audio_processor.py     # Speech-to-text conversion
│   ├── notes_processor.py     # Structured note processing
│   └── ingestion_manager.py   # Orchestrates all processors
│
├── 📁 indexing/               # Vector embeddings and storage
│   ├── embedding_manager.py   # CLIP + text embeddings
│   ├── vector_store.py        # ChromaDB interface
│   ├── cache_manager.py       # Embedding cache optimization
│   ├── memory_manager.py      # Memory usage optimization
│   └── performance_benchmarker.py # Performance monitoring
│
├── 📁 retrieval/              # Query processing
│   ├── query_processor.py     # Multimodal query handling
│   └── speech_to_text_processor.py # Audio query processing
│
├── 📁 generation/             # LLM integration
│   ├── lmstudio_generator.py  # LM Studio API client
│   ├── llm_factory.py         # Model selection logic
│   ├── llm_generator.py       # Legacy HF integration
│   └── citation_generator.py  # Citation formatting
│
├── 📁 kg_security/            # Knowledge graph security
│   ├── knowledge_graph_manager.py # Graph construction
│   ├── anomaly_detector.py    # Security monitoring
│   ├── security_event_logger.py # Audit logging
│   └── feedback_integration.py # User feedback processing
│
├── 📁 feedback/               # Feedback system
│   ├── feedback_system.py     # User feedback collection
│   ├── metrics_collector.py   # Performance metrics
│   └── 📁 exports/            # Feedback data exports
│
├── 📁 backend/                # FastAPI thin service layer
│   ├── main.py                # App factory, CORS, lifespan
│   ├── api/routes/            # HTTP endpoints
│   ├── services/              # Domain orchestration for HTTP
│   └── tests/                 # API tests
│
├── 📁 frontend/               # Next.js App Router (product UI)
│   ├── app/                   # Routes (chat, documents, …)
│   ├── features/              # Feature UI
│   ├── components/            # Shared UI
│   └── tests/                 # Playwright tests
│
├── 📁 ui/                     # Optional Streamlit analytics
│   └── streamlit_dashboard.py
│
├── 📁 docs/migration/         # Migration notes and parity
├── 📁 models/                 # Local model cache (LM Studio managed)
├── 📁 data/                   # Input data and uploads
├── 📁 vector_db/              # ChromaDB persistent storage
├── 📁 cache/                  # Embedding and processing cache
├── 📁 logs/                   # System logs and error reports
│
├── 🔧 config.py               # Central domain configuration
├── 🚀 main_launcher.py        # Domain runtime / optional Streamlit
├── 📋 requirements.txt        # Python dependencies
├── 🛠️ install_dependencies.py # Automated setup script
├── 📦 build_executables.py    # Portable build script
├── PRODUCT.md / DESIGN.md     # Product and design direction
└── scripts/dev.ps1            # Primary: API + Next.js launcher
```

## 🧪 Tests

```bash
# Backend API
pytest backend/tests -q

# Frontend
cd frontend
npm run typecheck
npm run build
npx playwright install chromium   # once
npm run test                      # requires API + frontend running
```

## 🔌 Offline deployment notes

- Install Python deps, Node deps, embedding models, Whisper, and LM Studio models while online.
- Run with no required external APIs: frontend → local FastAPI → local Chroma/LM Studio.
- Bind hosts explicitly for trusted LAN; keep `NEURAX_CORS_ORIGINS` tight (no `*`).
- Product UI: `pwsh scripts/dev.ps1` or run FastAPI + `npm run dev` in `frontend/`.

## 🩺 Troubleshooting

| Symptom | Check |
|---|---|
| Status bar: Backend unavailable | `uvicorn backend.main:app --host 127.0.0.1 --port 8000` |
| LM Studio unavailable | Local Server on port 1234; load a model |
| Empty search / weak answers | Index documents first; lower similarity threshold |
| Upload rejected | Extension allowlist and max size in Settings |
| CORS errors in browser | `NEURAX_CORS_ORIGINS` includes `http://127.0.0.1:3000` |


## 🔧 Configuration

### Core Settings (`config.py`)
```python
# LM Studio Configuration
LM_STUDIO_CONFIG = {
    "base_url": "http://localhost:1234/v1",
    "gemma_model": "google/gemma-3n",           # Multimodal model
    "qwen_model": "qwen/qwen3-4b-thinking-2507", # Reasoning model
    "auto_model_switching": True,               # Auto switch based on query type
}

# Security Configuration
SECURITY_CONFIG = {
    "allowed_file_extensions": [
        ".pdf", ".docx", ".doc", ".txt",        # Documents
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", # Images
        ".wav", ".mp3", ".m4a", ".flac", ".ogg" # Audio
    ],
    "max_file_size_mb": 100,
    "enable_audit_logging": True,
}
```

### Advanced Configuration
- **Performance tuning**: Memory thresholds, batch sizes, GPU settings
- **Security policies**: File validation, audit logging, anomaly detection
- **UI customization**: Interface themes, component visibility
- **Model preferences**: LLM selection, embedding models, fallback strategies

## 🧪 Testing & Validation

### Automated Testing Suite
```bash
# Run complete test suite
python -m pytest tests/

# Test specific components
python test_image_query_no_ocr.py     # Image processing
python test_multimodal_simple.py      # Multimodal queries  
python test_lmstudio_integration.py   # LM Studio integration
python test_final_verification.py     # End-to-end validation
```

### Manual Testing
```bash
# Test file upload interface
python test_file_upload_interface_fix.py

# Validate system performance  
python test_vector_store.py

# Check citation generation
python test_citation_fix.py
```

## 🚢 Deployment Options

### Option 1: Standard Installation
- Install Python dependencies via pip
- Setup LM Studio separately
- Run via `python main_launcher.py`

### Option 2: Portable Executable
```bash
# Build self-contained executable
python build_executables.py

# Generates:
# - NeuraX-Windows-x64.zip
# - USB_Deployment/ folder for air-gapped systems
```

### Option 3: USB Deployment
```bash
# Create USB-ready package
python build_executables.py --usb-deployment

# Copy USB_Deployment/ contents to USB drive
# Includes autorun.inf for Windows systems
```

### Air-Gapped Deployment
1. Build executable on internet-connected system
2. Copy package to air-gapped environment
3. Install LM Studio and download models offline
4. Run executable with zero internet dependencies

## 📊 Performance Metrics

### Processing Speeds
- **Document Indexing**: 50-100 documents/minute
- **Image Processing**: 25-50 images/minute  
- **Audio Transcription**: Real-time (1x speed with Whisper-tiny)
- **Query Response**: 200-500ms average
- **Vector Search**: 4.7+ items/second similarity search

### Resource Usage
- **Memory**: 4-8GB typical usage (scales with data size)
- **Storage**: 100MB base + data size + cache
- **GPU**: Optional but recommended for large datasets
- **CPU**: Efficient with multi-core utilization

## 🛡️ Security Features

### Data Protection
- **Local Processing**: All data remains on local system
- **Encrypted Storage**: Vector database encryption at rest
- **Audit Trails**: Comprehensive activity logging
- **Access Control**: File type and size validation

### Anomaly Detection
- **Knowledge Graph Monitoring**: Real-time graph analysis
- **Behavioral Analysis**: Unusual query pattern detection
- **Tamper Detection**: Content integrity verification
- **Alert System**: Automated security event notifications

## 🤝 Contributing & Support

### Development Setup
```bash
# Clone for development
git clone https://github.com/thrishank007/NeuraX.git
cd NeuraX

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests before committing
python -m pytest tests/
```

### Known Issues & Solutions
- **Tesseract OCR**: Auto-bundled in executables, manual install for dev
- **GPU Memory**: Adjust batch sizes in config for lower VRAM systems
- **LM Studio Connection**: Ensure server is running on localhost:1234
- **Large Files**: Use batch processing for datasets >1GB

### Documentation
- **API Reference**: `/docs/api/` (generated from code)
- **Architecture Guide**: `/docs/architecture.md`
- **Deployment Guide**: `/docs/deployment.md`
- **Troubleshooting**: `/docs/troubleshooting.md`

## 📈 Roadmap

### Current Version (v1.0)
- ✅ Complete offline multimodal RAG system
- ✅ LM Studio integration with Gemma 3n + Qwen3 4B
- ✅ Cross-modal search capabilities
- ✅ Portable executable generation
- ✅ Enterprise security features

### Future Enhancements (v1.1+)
- 🔄 Additional LLM integrations (Ollama, LocalAI)
- 🔄 Enhanced video processing capabilities
- 🔄 Multi-language support expansion
- 🔄 Advanced analytics dashboard
- 🔄 Distributed deployment options

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **NTRO SIH 2025**: Problem statement and requirements definition
- **Hugging Face**: CLIP and Transformer models
- **LM Studio**: Local LLM hosting platform
- **ChromaDB**: Vector database infrastructure
- **Next.js / FastAPI**: Product web UI and API

---

**Built with ❤️ for secure, offline AI document intelligence**

For detailed documentation, visit: [Documentation](./docs/)  
For support and issues: [GitHub Issues](https://github.com/thrishank007/NeuraX/issues)
