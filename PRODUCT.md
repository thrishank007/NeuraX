# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- Intelligence analysts and security operators who require air-gapped, zero-cloud data processing for sensitive documentation
- Technical teams and engineers running local LLMs (via LM Studio) seeking multimodal document intelligence without external APIs
- Operators evaluating retrieval quality and document provenance with full citation auditability

## Product Purpose

NeuraX is a local-first, offline multimodal Retrieval-Augmented Generation (RAG) workspace. It enables ingestion, cross-modal querying, and evidence-grounded analysis across text documents (PDF, DOCX, TXT), images (PNG, JPG, TIFF), and audio files (WAV, MP3, FLAC). Success means rapid, trustworthy question-answering with visible provenance, reliable local index health, and zero data egress.

## Positioning

A completely offline, local-first multimodal RAG system operating entirely on local infrastructure (FastAPI, ChromaDB, CLIP, Whisper, LM Studio) with real-time knowledge graph tamper detection and zero cloud telemetry.

## Operating Context

- Air-gapped workstations or private local servers with CPU/GPU compute
- Local LM Studio server hosting multimodal models (e.g. Gemma 3n) and reasoning models (e.g. Qwen3 4B)
- Dedicated analytical workspace divided into five views: Chat (Q&A with citation drawer), Documents (batch upload, progress, inventory), Sources (retrieval inspection), Knowledge Graph (corpus & security graph analysis), and Settings (diagnostics and threshold controls)

## Capabilities and Constraints

- **Capabilities**: Multimodal file ingestion with OCR and speech-to-text; cross-modal search via CLIP & ChromaDB; streaming grounded generation via LM Studio; citation score tracking; tamper-evident knowledge graph monitoring.
- **Technical Constraints**: Zero internet connectivity during operation; Next.js frontend (:3000) talking over HTTP/SSE to FastAPI service layer (:8000); strictly local vector and cache storage.
- **Non-Goals**: Hosted multi-tenant auth, cloud vector databases, or telemetry/analytics SDKs.

## Brand Commitments

- **Name**: NeuraX
- **Voice**: Calm, precise, diagnostic, and transparent about system status and limitations
- **Incumbent Visual System**: Preserved in `DESIGN.md` (IBM Plex Sans / IBM Plex Mono typography, warm paper light mode, dark olive/slate dark mode, deep teal functional accent)

## Evidence on Hand

- Operational domain packages: `backend/`, `ingestion/`, `indexing/`, `retrieval/`, `generation/`, `kg_security/`
- Next.js 15 App Router frontend in `frontend/`
- Documentation and architecture specs in `README.md` and `DEPLOYMENT_SUMMARY.md`
- Regression test suite in `tests/`

## Product Principles

1. **Status over chrome** — Connectivity, index health, and local model states remain visible at all times.
2. **Sources are first-class** — Every answer provides inspectable citation provenance and similarity scores.
3. **Dense, calm productivity** — Maximum information density and clarity without decorative distractions.
4. **Honest empty and error states** — Explicit diagnostics for offline, unindexed, or disconnected states; never fake generation success.
5. **Local-first integrity** — Privacy and zero external data leakage are absolute product invariants.

## Accessibility & Inclusion

- WCAG AA contrast compliance (≥ 4.5:1 for body and essential labels across light and dark themes)
- Full keyboard operability for chat composer, navigation, and source inspection
- ARIA live regions for background indexing status and token streaming
