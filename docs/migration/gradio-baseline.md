# Gradio Baseline Verification

**Date:** 2026-07-12  
**Branch:** `feat/nextjs-frontend-migration`  
**Source:** `stable` @ `85d2d09` (+ local WIP commit `6b3e217`)

## Startup

| Item | Value |
|---|---|
| Primary command | `python main_launcher.py --mode gradio_only` |
| Direct command | `python -c "from ui.gradio_app import launch_gradio_app; launch_gradio_app()"` |
| Module entry | `ui/gradio_app.py` → `launch_gradio_app()` |
| Expected URL | `http://127.0.0.1:7860` |
| Streamlit dashboard | `http://127.0.0.1:8501` (separate) |
| LM Studio | `http://127.0.0.1:1234/v1` |

### Startup verification results

| Check | Result |
|---|---|
| Import `NeuraXGradioApp` / `create_gradio_interface` | **Success** |
| Config load (`GRADIO_CONFIG`, formats, Chroma) | **Success** |
| Chroma collection `neurax_with_docs` | **Connected** (`total_documents`: 0 at baseline) |
| LM Studio `/models` | **Reachable** — 4 models loaded |
| Full browser session of live Gradio server | **Not fully exercised in automation** (heavy process; interface factory smoke-tested) |
| Existing pytest suite | **None found** at repo root beyond integration helpers inside launchers |

## Architecture map (current)

```text
Current Gradio UI (ui/gradio_app.py)
    → NeuraXGradioApp orchestration (lazy component init)
    → processors and ingestion (ingestion/*)
    → embeddings and ChromaDB (indexing/*)
    → retrieval (retrieval/query_processor.py, speech_to_text_processor.py)
    → LM Studio generation (generation/lmstudio_generator.py via llm_factory)
    → citations + feedback (generation/citation_generator.py, feedback/*)
```

## Available pages / tabs

1. **File Upload** — multi-file upload, process, clear, status, processing log, session file list, system info  
2. **Search & Query** — text / image / voice / multimodal search, similarity threshold, history, results  
3. **AI Response** — generate grounded answer, citations, confidence, latency, feedback  
4. **Help accordion** — usage instructions  

## Feature inventory (code-verified)

| Feature | Present | Notes |
|---|---|---|
| Document uploads | Yes | Gradio `File` multi |
| Supported file types | Yes | pdf/docx/doc/txt; jpg/jpeg/png/bmp/tiff/webp; wav/mp3/m4a/flac/ogg |
| Indexing progress | Yes | Gradio progress + log text |
| Document listing | Partial | Session processed list only; not full Chroma inventory |
| Delete indexed documents | No UI | `VectorStore.delete_documents` exists |
| Text questions | Yes | Search + AI Response tabs |
| Multimodal queries | Yes | Image / voice / text+image |
| Source citations | Yes | `CitationGenerator` + HTML |
| Source previews | Yes | Snippet + path + confidence |
| Chat history | Partial | In-session query history (last 10), not persistent chat |
| Model status | Partial | Component ready flags; LM Studio detail limited |
| LM Studio connectivity | Prerequisite | Offline check improved in `error_handler.py` |
| Settings | Partial | Similarity threshold, max context docs |
| Graph views | No (Gradio) | Streamlit + `kg_security` |
| Import/export | No UI | VectorStore methods exist |
| Streaming generation | No | LM Studio client `stream: False` |
| Error states | Yes | Text status messages |

## Successful workflows (verified at code + import level)

- Module import and interface construction path available  
- Chroma persistence directory opens  
- LM Studio reachable with models loaded  
- Ingestion → embed → store pipeline present in `process_uploaded_files`  
- Query paths call `QueryProcessor` methods unchanged  
- Response path: search results → `generate_grounded_response` → citations  

## Broken / limited workflows

| Item | Classification | Notes |
|---|---|---|
| Empty index searches | Expected empty state | Collection has 0 docs at baseline |
| Session-only file list after restart | Product limitation | Not persisted as UI list |
| No delete from Gradio | Missing UI | Backend method available |
| No streaming tokens | Design | Non-streaming LM Studio API |
| Memory pressure warnings | Environment | Host showed ~92% RAM during VectorStore init |

## API / model prerequisites

- **Python 3.8+** with project venv and `requirements.txt`  
- **LM Studio** local server with at least one model for generation  
- **Embedding models** via sentence-transformers / CLIP (downloaded once; offline thereafter)  
- **Whisper** for audio; **Tesseract** for OCR  
- **ChromaDB** local under `vector_db/`  

## Retrieval and citation behavior (baseline — do not change)

1. Text query → `embed_text` → `similarity_search` → filter by threshold (default from `SEARCH_CONFIG`, UI default 0.5)  
2. AI response uses last search results (max context docs slider, default 5)  
3. Citations from `CitationGenerator.generate_citations` with file path, type, snippet, confidence, optional page  

## Approximate performance expectations

- Component lazy init can take tens of seconds on first use (embeddings)  
- Indexing: per-file; progress is sequential  
- Generation: blocked on LM Studio request timeout up to 120s (`LM_STUDIO_CONFIG`)  

## Separation of concerns for failures

| Category | At baseline |
|---|---|
| Application defects | None proven in import path; full E2E UI not automated |
| Missing local infrastructure | None for LM Studio (running) |
| Missing models | None (4 models loaded) |
| Environment | High system memory pressure noted |

## Migration gate (closed)

This baseline recorded Gradio behavior before the Next.js cutover.  
**Gradio has been removed** (`ui/gradio_app.py` deleted; `gradio` dependency dropped).  
Product UI is Next.js + FastAPI only. Keep this file as historical reference.
