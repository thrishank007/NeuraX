# NeuraX - Production-Ready Multimodal RAG System

A complete offline-first Retrieval-Augmented Generation (RAG) system with multimodal capabilities, featuring a modern Next.js frontend and Python FastAPI backend.

## Architecture Overview

### Backend (Python)
- **FastAPI REST API** (`backend/api/main.py`) - RESTful API wrapper
- **Document Processing** - PDF, DOCX, TXT, Images, Audio with OCR and STT
- **Vector Store** - ChromaDB for semantic search
- **LLM Integration** - LM Studio (Gemma 3n for multimodal, Qwen 4B for reasoning)
- **Knowledge Graph** - NetworkX-based security and anomaly detection
- **Feedback System** - User feedback collection and metrics

### Frontend (Next.js)
- **Next.js 14+** with App Router
- **TypeScript** strict mode
- **Tailwind CSS** + shadcn/ui components
- **Responsive Design** with dark mode
- **Real-time Updates** via WebSocket (optional)

## Quick Start

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install FastAPI and uvicorn if not already installed
pip install fastapi uvicorn python-multipart

# Start the FastAPI backend
cd backend/api
python main.py
# or
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd neurax-frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.local.example .env.local
# Edit .env.local with your configuration

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 3. LM Studio Setup (Optional but Recommended)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load models:
   - **Gemma 3n** (for multimodal queries)
   - **Qwen3 4B Thinking** (for text reasoning)
3. Start the local server on port 1234
4. The backend will automatically connect to LM Studio

## Features

### Document Management
- Upload PDF, DOCX, TXT, Images (JPG, PNG, etc.), Audio (WAV, MP3, etc.)
- Automatic processing and indexing
- Batch upload support
- File type validation and size limits

### Multimodal Queries
- **Text Queries** - Natural language questions
- **Voice Queries** - Speech-to-text input
- **Image Queries** - Visual search (coming soon)
- **Multimodal** - Combined text + image queries

### Results & Citations
- Numbered citations with expandable sources
- Confidence scores and similarity metrics
- Document previews
- Export capabilities (JSON, CSV)

### Analytics Dashboard
- Performance metrics (retrieval, generation, latency)
- Usage statistics
- Security alerts and anomaly detection
- Knowledge graph visualization

### Configuration
- LM Studio connection settings
- Search parameters (similarity threshold, max results)
- Model preferences
- Performance tuning

## API Endpoints

### File Upload
- `POST /api/upload` - Upload and process files
- `GET /api/files` - List uploaded files
- `DELETE /api/files/{file_id}` - Delete a file

### Query Processing
- `POST /api/query` - Process text query
- `POST /api/query/voice` - Process voice query
- `GET /api/query/history` - Get query history
- `GET /api/query/suggestions` - Get auto-complete suggestions

### Analytics
- `GET /api/analytics/metrics` - Get performance metrics
- `GET /api/analytics/usage` - Get usage statistics
- `GET /api/analytics/security` - Get security events
- `GET /api/knowledge-graph` - Get knowledge graph data

### Feedback
- `POST /api/feedback` - Submit feedback
- `GET /api/feedback/history` - Get feedback history
- `GET /api/feedback/analytics` - Get feedback analytics

### Configuration
- `GET /api/config` - Get current configuration
- `PUT /api/config` - Update configuration
- `POST /api/config/validate` - Validate configuration

### Health
- `GET /api/health` - Health check

## Development

### Backend Development

```bash
# Run with auto-reload
uvicorn backend.api.main:app --reload

# Run tests (if available)
pytest
```

### Frontend Development

```bash
# Development server with hot reload
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint

# Production build
npm run build
npm start
```

## Project Structure

```
.
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   ├── routers/                  # API route modules
│   ├── services/                 # Business logic services
│   ├── models/                   # Pydantic models
│   └── core/                     # Core utilities
│
├── neurax-frontend/              # Next.js frontend
│   ├── app/                      # Next.js app directory
│   ├── components/               # React components
│   ├── lib/                      # Utilities and API clients
│   └── public/                   # Static assets
│
├── ingestion/                    # Document processing
├── indexing/                     # Vector store and embeddings
├── retrieval/                    # Query processing
├── generation/                   # LLM integration
├── kg_security/                  # Knowledge graph and security
├── feedback/                     # Feedback system
└── config.py                     # Configuration
```

## Configuration

### Environment Variables

**Backend:**
- Configure in `config.py` or via environment variables
- LM Studio URL: `http://localhost:1234/v1` (default)

**Frontend:**
- `NEXT_PUBLIC_API_URL` - Backend API URL (default: `http://localhost:8000`)
- `NEXT_PUBLIC_LM_STUDIO_URL` - LM Studio URL (default: `http://localhost:1234`)
- `NEXT_PUBLIC_MAX_FILE_SIZE` - Max file size in bytes (default: 100MB)

## Deployment

### Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

### Manual Deployment

1. **Backend:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run with production server
   uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
   ```

2. **Frontend:**
   ```bash
   cd neurax-frontend
   npm install
   npm run build
   npm start
   ```

## Offline Operation

The system is designed for offline/air-gapped deployment:

- All models run locally (LM Studio)
- No external API dependencies
- Local vector database (ChromaDB)
- Offline document processing
- Local feedback storage

## Security

- Input validation and sanitization
- File type and size validation
- CORS configuration
- Audit logging
- Anomaly detection
- Knowledge graph security layer

## Performance

- Efficient embedding caching
- Batch processing support
- Memory optimization
- Progressive loading for large datasets
- Connection pooling

## Troubleshooting

### Backend Issues

1. **LM Studio not connecting:**
   - Ensure LM Studio is running on port 1234
   - Check `LM_STUDIO_CONFIG` in `config.py`

2. **Vector store errors:**
   - Check `vector_db/` directory permissions
   - Ensure ChromaDB is properly installed

3. **Import errors:**
   - Verify all dependencies in `requirements.txt` are installed
   - Check Python path configuration

### Frontend Issues

1. **API connection errors:**
   - Verify backend is running on port 8000
   - Check `NEXT_PUBLIC_API_URL` in `.env.local`

2. **Build errors:**
   - Run `npm install` to ensure all dependencies are installed
   - Check TypeScript errors with `npm run type-check`

## Contributing

1. Follow code style guidelines
2. Write tests for new features
3. Update documentation
4. Ensure TypeScript strict mode compliance
5. Test on multiple browsers

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions, please refer to the project documentation or create an issue in the repository.
