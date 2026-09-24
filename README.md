# NeuraX — Offline-First Multimodal RAG System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

NeuraX is a production-ready multimodal Retrieval-Augmented Generation (RAG) system built for NTRO's SIH 2025 problem statement. It runs **offline-first**: every core capability — ingestion, embedding, hybrid retrieval, reranking, and generation — works on local infrastructure with zero internet dependency. Optional cloud services (NVIDIA NIM embeddings/reranking, an OpenAI-compatible cloud LLM) can be enabled individually for online deployments via a single `NEURAX_STRATEGY` switch.

## Demo:

[![Watch the video](https://img.youtube.com/vi/2qcBRtBl5q8/0.jpg)](https://youtu.be/2qcBRtBl5q8)

## ✨ Key Features

### 🔒 **Security & Privacy**
- **Offline-First Operation**: `offline_first` strategy keeps everything local — zero egress by default
- **Knowledge Graph Security**: Real-time anomaly detection and tamper protection over the security graph
- **Audit Logging**: Comprehensive activity tracking and compliance monitoring
- **Data Sovereignty**: All document processing, vector storage, and generation occur locally unless a cloud service is explicitly enabled

### 🤖 **Retrieval & Generation**
- **Hybrid Retrieval**: Dense vector search fused with BM25 via dense-favored weighted Reciprocal Rank Fusion (RRF)
- **Optional Cross-Encoder Reranking**: NVIDIA NIM reranking stage after RRF fusion (opt-in, needs API key)
- **Token-by-Token Streaming**: Server-Sent Events streaming for both LM Studio (local) and cloud generation
- **LM Studio Integration**: Local LLM hosting with Gemma 3n (multimodal) and Qwen3 4B (reasoning)
- **Optional Cloud LLM**: Any OpenAI-compatible endpoint (e.g. Groq) with local/cloud mode toggle in the Chat UI; cloud failures degrade gracefully to guidance, never fake success
- **CLIP Embeddings**: Visual–text cross-modal similarity matching
- **Intelligent Citations**: Numbered references with confidence scores and expandable sources

### 📁 **Comprehensive Format Support**
- **Documents**: PDF, DOCX, DOC, TXT with OCR fallback
- **Images**: JPG, JPEG, PNG, BMP, TIFF, WEBP with visual similarity search
- **Audio**: WAV, MP3, M4A, FLAC, OGG with speech-to-text processing
- **Batch Processing**: Multiple files with per-job progress tracking and cancellation

### 🧪 **Evaluation & Maintenance Tooling**
- **RAG Eval Harness**: precision@k / recall@k / MRR plus latency benchmarks across dense, BM25, and hybrid modes (`scripts/eval_rag.py`)
- **Synthetic Golden Set Generation**: hand-labeled golden set from live corpus chunks (`scripts/make_eval_dataset.py`)
- **Corpus Reindex**: re-chunk and re-embed indexed files through the production ingestion path (`scripts/reindex_corpus.py`)
- **Embedding Cache**: NIM query embeddings are cached and bounded, so repeat queries skip the API

### 🚀 **Production Features**
- **Auto-Deployment**: One-click executable generation with PyInstaller
- **USB Portability**: Export complete system to USB for air-gapped deployment
- **Error Resilience**: Graceful degradation and comprehensive error handling
- **Real-time Feedback**: User feedback collection and performance metrics

## 🏗️ Architecture

```text
Next.js frontend (frontend/, :3000)
    ↓ HTTP / SSE
FastAPI service layer (backend/, :8000)
    ↓
Domain modules: ingestion · indexing · retrieval · generation
    ↓
Dense (ChromaDB + MiniLM or NIM) + BM25 → weighted RRF → optional NIM rerank
    ↓
LM Studio (local) or cloud LLM · Whisper · CLIP · Graphify KG
```

**Product UI is Next.js only.** Streamlit (`:8501`) is optional analytics, not the main interface.

### Core Components
- **FastAPI**: Thin HTTP/SSE API over domain modules (`backend/`)
- **Next.js UI**: Primary workspace — Chat, Documents, Search, Sources, Graph, Settings (`frontend/`)
- **ChromaDB**: Persistent vector database; separate collections for local MiniLM, NIM, and CLIP image embeddings
- **Hybrid Retrieval**: BM25 + dense candidates fused by weighted RRF (`retrieval/query_processor.py`)
- **NIM Reranker**: Optional cross-encoder reranking of fused candidates (`retrieval/nim_reranker.py`)
- **LM Studio**: Local LLM server for multimodal and reasoning tasks (`generation/lmstudio_generator.py`)
- **Cloud Generator**: Optional OpenAI-compatible cloud LLM with streaming (`generation/cloud_generator.py`)
- **Whisper STT**: Speech-to-text for audio processing
- **CLIP Embeddings**: Visual-text cross-modal understanding
- **NetworkX Security Graph**: Anomaly / tamper monitoring (`kg_security/`)
- **Graphify (optional)**: Document knowledge graph over the uploaded corpus (CLI integration)

Details: [docs/architecture.md](docs/architecture.md) · API reference: [docs/api.md](docs/api.md)

## 🛠️ System Requirements

- **Python**: 3.9+ (3.10+ recommended)
- **Node.js**: 18+ (for the Next.js frontend)
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **GPU**: optional, 6GB+ VRAM accelerates embedding/Whisper (CPU fallback available)
- **Storage**: 10GB+ free for cache, vector DB, and data
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+

### External dependencies
- **LM Studio**: local LLM hosting (Gemma 3n + Qwen3 4B) — required for local generation
- **Tesseract OCR**: document text extraction (auto-bundled in executables, manual install for dev)
- **FFmpeg**: audio processing (platform-specific installation)
- **Graphify (optional)**: document knowledge graph CLI — see [Knowledge Graph (Graphify)](#️-knowledge-graph-graphify-document-intelligence)

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/thrishank007/NeuraX.git
cd NeuraX

# Automated setup (venv + dependencies + system tools)
python install_dependencies.py

# Configure environment (all optional — defaults are offline/local)
cp .env.example .env        # Windows: copy .env.example .env

# Launch product UI (FastAPI + Next.js)
pwsh scripts/dev.ps1
```

### Option 2: Manual Installation
```bash
git clone https://github.com/thrishank007/NeuraX.git
cd NeuraX

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# System dependencies
# Ubuntu/Debian: sudo apt-get install tesseract-ocr ffmpeg
# macOS:         brew install tesseract ffmpeg
# Windows:       automated via install_dependencies.py

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then the frontend:

```bash
cd frontend
cp .env.local.example .env.local   # Windows: copy .env.local.example .env.local
npm install
npm run dev
```

| Surface | URL |
|---|---|
| Next.js workspace | http://127.0.0.1:3000 |
| FastAPI | http://127.0.0.1:8000 |
| Interactive API docs | http://127.0.0.1:8000/docs |
| Streamlit (optional analytics) | http://127.0.0.1:8501 |

Environment templates: [`.env.example`](.env.example), [`frontend/.env.local.example`](frontend/.env.local.example).

### Option 3: Portable Executable
```bash
python build_executables.py

# Deploy to USB or air-gapped system
# Executable will be in packages/ directory
```

## ⚙️ Deployment Strategy (`NEURAX_STRATEGY`)

One switch controls the local-vs-cloud posture. Individual `NEURAX_*` flags always override the strategy default.

| | `offline_first` (default) | `online_first` |
|---|---|---|
| NIM embeddings | off unless `NEURAX_NIM_EMBEDDINGS_ENABLED=true` | on when `NEURAX_NIM_API_KEY` exists |
| NIM reranking | off unless `NEURAX_NIM_RERANK_ENABLED=true` | on when `NEURAX_NIM_API_KEY` exists |
| Chat default mode | LM Studio (local) | cloud LLM |

```bash
# .env
NEURAX_STRATEGY=online_first
NEURAX_NIM_API_KEY=your-nvidia-api-key
NEURAX_CLOUD_API_URL=https://api.groq.com/openai/v1
NEURAX_CLOUD_API_KEY=your-api-key
NEURAX_CLOUD_MODEL=llama-3.3-70b-versatile
```

Full environment-variable reference and air-gapped checklist: [docs/deployment.md](docs/deployment.md).

## 🎯 LM Studio Setup (Required for local generation)

1. **Install** from [https://lmstudio.ai/](https://lmstudio.ai/)
2. **Download models** — in LM Studio, search for and download:
   - **Gemma 3n**: multimodal queries (text + images)
   - **Qwen3 4B Thinking 2507**: complex reasoning tasks
3. **Start the local server**: Local Server tab → load a model → serve on `localhost:1234`
4. **Verify** in NeuraX: the status bar / Settings page shows the LM Studio model state; the backend logs the connection on startup.

## 💻 Usage Examples

### Chat with your documents
```bash
# In the Chat workspace (or POST /api/chat / /api/chat/stream)
query = "What are the main findings in the research?"
# Answers stream token-by-token with numbered citations and similarity scores.
```

### Multimodal Search
```python
# Upload images along with documents (JPG, PNG, BMP, TIFF, WEBP)
# Cross-modal queries via the Search workspace
query = "Find documents related to this chart"
# CLIP matches visual content with textual descriptions
```

### Audio Processing
```python
# Upload audio files (WAV, MP3, M4A, FLAC, OGG)
# Audio is transcribed with Whisper and indexed as searchable text
query = "What was discussed about budget planning?"
```

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

More detail, including troubleshooting: [docs/troubleshooting.md](docs/troubleshooting.md#graphify).

## 🧰 Maintenance & Evaluation Scripts

```bash
# Re-index the indexed corpus through the current chunker/embedding path
# (run after changing chunking logic so deterministic chunk IDs stay in sync)
venv\Scripts\python.exe scripts\reindex_corpus.py                # all indexed files
venv\Scripts\python.exe scripts\reindex_corpus.py docs\a.pdf     # explicit subset

# Retrieval evaluation: precision@k, recall@k, MRR + latency across modes
venv\Scripts\python.exe scripts\eval_rag.py --k 5 --runs 5       # dense / bm25 / hybrid

# Regenerate the synthetic golden set from live corpus chunks
venv\Scripts\python.exe scripts\make_eval_dataset.py
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
│   ├── nvidia_nim_embedding_provider.py  # Optional NIM cloud embeddings
│   ├── text_chunker.py        # Deterministic chunking (stable chunk IDs)
│   ├── vector_store.py        # ChromaDB interface
│   ├── cache_manager.py       # Embedding cache optimization
│   ├── memory_manager.py      # Memory usage optimization
│   └── performance_benchmarker.py # Performance monitoring
│
├── 📁 retrieval/              # Query processing
│   ├── query_processor.py     # Hybrid retrieval: dense + BM25 → weighted RRF
│   ├── nim_reranker.py        # Optional NIM cross-encoder reranking
│   └── speech_to_text_processor.py # Audio query processing
│
├── 📁 generation/             # LLM integration
│   ├── lmstudio_generator.py  # LM Studio API client (streaming)
│   ├── cloud_generator.py     # Optional OpenAI-compatible cloud LLM (streaming)
│   ├── llm_factory.py         # Model selection logic
│   ├── llm_generator.py       # Legacy HF integration
│   └── citation_generator.py  # Citation formatting
│
├── 📁 kg_security/            # Knowledge graphs
│   ├── knowledge_graph_manager.py # Security graph construction
│   ├── anomaly_detector.py    # Security monitoring
│   ├── security_event_logger.py   # Audit logging
│   ├── feedback_integration.py    # User feedback processing
│   └── graphify_service.py    # Graphify CLI integration (document graph)
│
├── 📁 feedback/               # Feedback system
│   ├── feedback_system.py     # User feedback collection
│   ├── metrics_collector.py   # Performance metrics
│   └── 📁 exports/            # Feedback data exports
│
├── 📁 backend/                # FastAPI thin service layer
│   ├── main.py                # App factory, CORS, lifespan
│   ├── api/routes/            # HTTP + SSE endpoints
│   ├── services/              # Domain orchestration for HTTP
│   └── tests/                 # API tests
│
├── 📁 frontend/               # Next.js App Router (product UI)
│   ├── app/                   # Routes (chat, documents, search, sources, graph, settings)
│   ├── features/              # Feature UI
│   ├── components/            # Shared UI
│   └── tests/                 # Playwright tests
│
├── 📁 ui/                     # Optional Streamlit analytics
│   └── streamlit_dashboard.py
│
├── 📁 scripts/                # Dev + ops tooling
│   ├── dev.ps1                # Start API + Next.js
│   ├── reindex_corpus.py      # Re-chunk/re-embed indexed files
│   ├── eval_rag.py            # Retrieval quality + latency eval
│   └── make_eval_dataset.py   # Synthetic golden-set generation
│
├── 📁 tests/                  # Domain unit/integration tests
├── 📁 docs/                   # Architecture, deployment, API, troubleshooting
├── 📁 models/                 # Local model cache (LM Studio managed)
├── 📁 data/                   # Uploads + Graphify workspaces
├── 📁 vector_db/              # ChromaDB persistent storage
├── 📁 cache/                  # Embedding and processing cache
├── 📁 logs/                   # System logs and error reports
│
├── 🔧 config.py               # Central domain configuration
├── 🚀 main_launcher.py        # Domain runtime / optional Streamlit
├── 📋 requirements.txt        # Python dependencies
├── 🛠️ install_dependencies.py # Automated setup script
├── 📦 build_executables.py    # Portable build script
├── 📄 PRODUCT.md / DESIGN.md  # Product and design direction
└── 🧪 pytest.ini              # Collects tests/ + backend/tests/
```

## 🧪 Tests

```bash
# All Python tests (pytest.ini collects tests/ and backend/tests/)
pytest

# Domain units only — e.g. hybrid retrieval, reranker, chunker, streaming, Graphify
pytest tests -q

# Backend API suite
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
- Keep `NEURAX_STRATEGY=offline_first` (the default) so no cloud service activates implicitly.
- Bind hosts explicitly for trusted LAN; keep `NEURAX_CORS_ORIGINS` tight (no `*`).
- Product UI: `pwsh scripts/dev.ps1` or run FastAPI + `npm run dev` in `frontend/`.

Full deployment guide: [docs/deployment.md](docs/deployment.md).

## 🔧 Configuration

### Environment variables (`.env`)

See [`.env.example`](.env.example) for the full annotated template:

| Variable | Default | Purpose |
|---|---|---|
| `NEURAX_STRATEGY` | `offline_first` | `offline_first` / `online_first` service posture |
| `NEURAX_API_HOST` / `NEURAX_API_PORT` | `127.0.0.1` / `8000` | FastAPI bind address |
| `NEURAX_CORS_ORIGINS` | localhost:3000 variants | Allowed browser origins |
| `NEURAX_UPLOAD_DIR` / `NEURAX_MAX_UPLOAD_MB` | `data/uploads` / `100` | Upload storage and cap |
| `NEURAX_CLOUD_*` | unset | OpenAI-compatible cloud LLM (URL, key, model, tokens, temperature, timeout) |
| `NEURAX_NIM_API_KEY` | unset | Shared key for NIM embeddings + reranking |
| `NEURAX_NIM_EMBEDDINGS_ENABLED` | strategy default | `true`/`false` NIM embeddings override |
| `NEURAX_NIM_EMBEDDING_MODEL` | `nvidia/nemotron-3-embed-1b` | NIM embedding model |
| `NEURAX_NIM_COLLECTION_NAME` | `neurax_nim_nemotron_3_embed_1b_v1` | ChromaDB collection for NIM vectors |
| `NEURAX_NIM_RERANK_ENABLED` | strategy default | `true`/`false` reranking override |
| `NEURAX_NIM_RERANK_MODEL` | `nvidia/rerank-qa-mistral-4b` | NIM reranking model |

### Domain config (`config.py`)

```python
# LM Studio (local generation)
LM_STUDIO_CONFIG = {
    "base_url": "http://localhost:1234/v1",
    "gemma_model": "google/gemma-3n",            # Multimodal model
    "qwen_model": "qwen/qwen3-4b-thinking-2507", # Reasoning model
    "auto_model_switching": True,                # Auto switch based on query type
}

# Hybrid retrieval + reranking
SEARCH_CONFIG = {
    "enable_hybrid": True,   # BM25 + dense RRF fusion
    "bm25_k": 20,            # BM25 candidates before RRF merge
    "rrf_k": 20,             # RRF constant
    "dense_weight": 0.9,     # Dense-favored fusion (see evals)
    "sparse_weight": 0.1,
    "enable_reranking": False,  # Requires NIM key + explicit enable
    "rerank_candidates": 20,
}
```

Advanced knobs (performance, security policy, KG thresholds, feedback) live alongside these in `config.py`.

## 🩺 Troubleshooting

| Symptom | Check |
|---|---|
| Status bar: Backend unavailable | `uvicorn backend.main:app --host 127.0.0.1 --port 8000` |
| LM Studio unavailable | Local Server on port 1234; load a model |
| Empty search / weak answers | Index documents first; lower similarity threshold; check hybrid mode is on |
| Cloud chat errors | Key/model in `NEURAX_CLOUD_*`; errors degrade to a guidance delta, not a crash |
| Upload rejected | Extension allowlist and max size in Settings |
| CORS errors in browser | `NEURAX_CORS_ORIGINS` includes `http://127.0.0.1:3000` |

Extended guide (incl. Graphify): [docs/troubleshooting.md](docs/troubleshooting.md).

## 🚢 Deployment Options

### Option 1: Standard Installation
- Install Python dependencies via pip; set up LM Studio separately
- Run `uvicorn backend.main:app` + `cd frontend && npm run dev` (or `pwsh scripts/dev.ps1`)

### Option 2: Portable Executable
```bash
python build_executables.py
# Generates:
# - NeuraX-Windows-x64.zip
# - USB_Deployment/ folder for air-gapped systems
```

### Option 3: USB / Air-Gapped Deployment
```bash
python build_executables.py --usb-deployment
# Copy USB_Deployment/ contents to USB drive (includes autorun.inf for Windows)
```

Air-gapped checklist: [docs/deployment.md](docs/deployment.md#air-gapped-checklist).

## 📊 Performance Notes

- **Query embeddings**: cached across repeated queries (bounded LRU); NIM repeat queries skip the API entirely
- **Hybrid retrieval**: dense candidates fetched at 3× k before RRF merge; fusion is O(candidates)
- **Streaming**: first token reaches the UI as soon as the generator emits it — no full-response wait
- **Memory**: typical usage 4–8GB, scales with corpus and cache size; GC tuning enabled in `PERFORMANCE_CONFIG`
- Measure on your own corpus with `scripts/eval_rag.py` (reports per-mode latency alongside quality metrics)

## 🛡️ Security Features

- **Local Processing**: all data remains on the local system under `offline_first`
- **Audit Trails**: comprehensive activity logging (`kg_security/security_event_logger.py`)
- **Anomaly Detection**: knowledge-graph monitoring, behavioral analysis, tamper detection
- **Access Control**: file type and size validation; quarantine of suspicious files
- **Graphify Sandbox**: sanitized corpus names, SHA-256 manifest, no path traversal, loopback-only model endpoints by default

## 🤝 Contributing & Support

```bash
git clone https://github.com/thrishank007/NeuraX.git
cd NeuraX
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest black flake8

pytest   # run tests before committing
```

### Known Issues & Notes
- **Tesseract OCR**: auto-bundled in executables, manual install for dev
- **GPU Memory**: adjust batch sizes in config for lower VRAM systems
- **LM Studio Connection**: ensure server is running on localhost:1234
- **NIM rerank model ids**: must be invocable by your account; the API returns the valid list on a miss
- **Large Files**: use batch processing for datasets >1GB

### Documentation
- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **API reference**: [docs/api.md](docs/api.md) (interactive: http://127.0.0.1:8000/docs)
- **Deployment guide**: [docs/deployment.md](docs/deployment.md)
- **Troubleshooting**: [docs/troubleshooting.md](docs/troubleshooting.md)
- **Migration notes**: [docs/migration/](docs/migration/)

## 📈 Roadmap

### Current Version
- ✅ Offline-first multimodal RAG with Next.js + FastAPI
- ✅ Hybrid retrieval (BM25 + dense, weighted RRF) and optional NIM reranking
- ✅ Token-by-token streaming for local and cloud generation
- ✅ Retrieval evaluation harness and corpus reindex tooling
- ✅ Graphify document knowledge graph + security graph monitoring
- ✅ `NEURAX_STRATEGY` deployment switch (offline_first / online_first)
- ✅ Portable executable generation

### Future Enhancements
- 🔄 Additional LLM integrations (Ollama, LocalAI)
- 🔄 Enhanced video processing capabilities
- 🔄 Multi-language support expansion
- 🔄 Distributed deployment options

## 📄 License

This project is licensed under the MIT License.

## 🏆 Acknowledgments

- **NTRO SIH 2025**: Problem statement and requirements definition
- **Hugging Face**: CLIP and Transformer models
- **LM Studio**: Local LLM hosting platform
- **ChromaDB**: Vector database infrastructure
- **Next.js / FastAPI**: Product web UI and API

---

**Built with ❤️ for secure, offline AI document intelligence**

For support and issues: [GitHub Issues](https://github.com/thrishank007/NeuraX/issues)
