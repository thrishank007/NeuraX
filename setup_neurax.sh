#!/bin/bash
# NeuraX Frontend Setup Script
# This script sets up the complete NeuraX RAG system with both frontend and backend

set -e

echo "🚀 Setting up NeuraX RAG System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main_launcher.py" ]; then
    print_error "Please run this script from the NeuraX project root directory"
    exit 1
fi

# Create directory structure
print_status "Creating directory structure..."
mkdir -p neurax-frontend
mkdir -p logs
mkdir -p data
mkdir -p vector_db
mkdir -p models

# Setup Backend Dependencies
print_status "Setting up Python backend dependencies..."
pip3 install -r requirements.txt
pip3 install -r backend/requirements-api.txt

# Setup Frontend
print_status "Setting up Next.js frontend..."
cd neurax-frontend

# Install Node.js dependencies
if ! command -v npm &> /dev/null; then
    print_error "Node.js and npm are required but not installed."
    print_error "Please install Node.js 18+ from https://nodejs.org/"
    exit 1
fi

print_status "Installing Node.js dependencies..."
npm install

# Setup environment variables
print_status "Setting up environment configuration..."
if [ ! -f ".env.local" ]; then
    cp .env.example .env.local 2>/dev/null || true
    print_warning "Created .env.local from template. Please configure it with your settings."
fi

# Build the frontend
print_status "Building frontend for production..."
npm run build

cd ..

# Setup LM Studio (optional)
print_status "Checking LM Studio integration..."
if [ ! -f "lm_studio_config.json" ]; then
    print_warning "LM Studio configuration not found."
    print_warning "Please ensure LM Studio is running on localhost:1234 for full functionality."
    print_warning "You can download LM Studio from https://lmstudio.ai/"
fi

# Create startup scripts
print_status "Creating startup scripts..."

# Frontend startup script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting NeuraX Frontend..."
cd neurax-frontend
npm run dev
EOF

# Backend startup script
cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting NeuraX Backend API..."
cd backend
python api_server.py
EOF

# Full system startup script
cat > start_system.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Complete NeuraX RAG System..."

# Function to cleanup background processes on exit
cleanup() {
    echo "Shutting down NeuraX system..."
    kill $(jobs -p) 2>/dev/null || true
    exit
}

trap cleanup SIGINT SIGTERM

# Start backend in background
echo "Starting backend API server..."
./start_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "Starting frontend application..."
./start_frontend.sh &
FRONTEND_PID=$!

echo ""
echo "✅ NeuraX system is starting up..."
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait
EOF

# Make scripts executable
chmod +x start_frontend.sh
chmod +x start_backend.sh
chmod +x start_system.sh

# Create a comprehensive README for the complete system
print_status "Creating system documentation..."
cat > NEURAX_SETUP.md << 'EOF'
# NeuraX RAG System - Complete Setup Guide

## System Overview

The NeuraX RAG (Retrieval-Augmented Generation) system consists of:

1. **Python Backend** - Core RAG functionality with multimodal processing
2. **FastAPI Wrapper** - REST API for frontend integration
3. **Next.js Frontend** - Modern web interface

## Quick Start

### Option 1: Start Everything at Once
```bash
./start_system.sh
```

### Option 2: Start Components Separately

**Terminal 1 - Backend:**
```bash
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
./start_frontend.sh
```

## Access Points

- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs
- **System Status**: http://localhost:8000/api/status

## Features

### 📁 File Upload
- Drag-and-drop interface
- Support for PDF, DOCX, DOC, TXT, Images, Audio
- Real-time processing status
- Batch upload capabilities

### 🔍 Multimodal Search
- **Text Queries**: Natural language search
- **Image Search**: Visual similarity search
- **Voice Search**: Speech-to-text queries
- **Multimodal**: Combine text and image search

### 📊 Analytics Dashboard
- System performance metrics
- Query statistics
- Usage trends
- Security monitoring

### ⚙️ Configuration
- Model selection (Gemma 3n, Qwen3 4b)
- Performance tuning
- Security settings
- System preferences

## Backend API Endpoints

### File Operations
- `POST /api/upload` - Upload files
- `GET /api/files` - List uploaded files
- `DELETE /api/files/{id}` - Delete file

### Query Operations
- `POST /api/query/text` - Text search
- `POST /api/query/image` - Image search
- `POST /api/query/voice` - Voice search
- `POST /api/query/multimodal` - Combined search

### Response Generation
- `POST /api/generate-response` - AI response with citations

### Analytics & Monitoring
- `GET /api/analytics` - Analytics data
- `GET /api/security/events` - Security events
- `GET /api/audit/logs` - Audit logs

### Configuration
- `GET /api/config` - Get system config
- `PUT /api/config` - Update configuration
- `POST /api/config/validate` - Validate config

## System Requirements

### Minimum Requirements
- Python 3.8+
- Node.js 18+
- 8GB RAM
- 20GB disk space

### Recommended Requirements
- Python 3.10+
- Node.js 20+
- 16GB RAM
- 50GB disk space
- GPU (optional, for faster processing)

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find and kill processes using ports
   lsof -ti:3000 | xargs kill -9  # Frontend
   lsof -ti:8000 | xargs kill -9  # Backend
   ```

2. **Module Not Found Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   npm install
   ```

3. **Frontend Build Errors**
   ```bash
   # Clear cache and rebuild
   cd neurax-frontend
   rm -rf .next node_modules
   npm install
   npm run build
   ```

4. **Backend Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Logs and Debugging

- **Frontend Logs**: Browser developer console
- **Backend Logs**: Check terminal output or log files
- **API Logs**: Check FastAPI server logs
- **System Logs**: Check logs/ directory

## Configuration

### Environment Variables (.env.local)

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_LM_STUDIO_URL=http://localhost:1234

# File Upload Settings
NEXT_PUBLIC_MAX_FILE_SIZE=104857600
NEXT_PUBLIC_ALLOWED_FILE_TYPES=.pdf,.docx,.doc,.txt,.jpg,.png,.mp3,.wav,.m4a,.flac,.ogg,.bmp,.tiff,.webp

# Feature Flags
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_DARK_MODE=true
NEXT_PUBLIC_ENABLE_VOICE_INPUT=true

# Query Settings
NEXT_PUBLIC_DEFAULT_SIMILARITY_THRESHOLD=0.5
NEXT_PUBLIC_MAX_QUERY_HISTORY=50
```

### Backend Configuration (config.py)

Key configuration sections:
- **LM_STUDIO_CONFIG**: LM Studio connection settings
- **CHROMA_CONFIG**: Vector database settings
- **SECURITY_CONFIG**: Security and file upload limits
- **FEEDBACK_CONFIG**: Feedback system settings

## Development

### Frontend Development
```bash
cd neurax-frontend
npm run dev
```

### Backend Development
```bash
cd backend
python api_server.py
```

### Adding New Features

1. **Frontend**: Add components in `neurax-frontend/components/`
2. **Backend**: Add endpoints in `backend/api_server.py`
3. **Types**: Update type definitions in `neurax-frontend/lib/types/`

## Production Deployment

### Static Export
```bash
cd neurax-frontend
npm run build
# Static files will be in .next/
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t neurax-frontend neurax-frontend/
docker run -p 3000:3000 neurax-frontend
```

### Security Considerations

1. **Environment Variables**: Never commit sensitive data
2. **File Uploads**: Validate file types and sizes
3. **API Endpoints**: Implement rate limiting
4. **Network**: Use HTTPS in production
5. **Authentication**: Add auth if needed

## Support

- **Issues**: GitHub Issues
- **Documentation**: This README and inline comments
- **API Docs**: http://localhost:8000/api/docs

## License

MIT License - see LICENSE file for details.
EOF

# Final setup completion
print_success "✅ NeuraX RAG System setup completed!"
echo ""
echo "🚀 To start the system:"
echo "   ./start_system.sh"
echo ""
echo "📱 Access the application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/api/docs"
echo ""
echo "📚 Read NEURAX_SETUP.md for detailed documentation"
echo ""
echo "🎯 Key Features:"
echo "   • Multimodal file upload and processing"
echo "   • Text, image, voice, and multimodal search"
echo "   • AI response generation with citations"
echo "   • Real-time analytics dashboard"
echo "   • Query history and management"
echo "   • Comprehensive system settings"
echo ""
print_success "Happy exploring with NeuraX! 🎉"