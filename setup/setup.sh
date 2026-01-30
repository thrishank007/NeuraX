#!/bin/bash

# NeuraX Complete Setup Script
# This script automates the entire setup process for NeuraX

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logo
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════╗
║         NeuraX Setup & Installation        ║
║   Offline Multimodal RAG System            ║
╚═══════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.8+ is required but not installed"
        exit 1
    fi
    print_status "Python $(python3 --version) found"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js 18+ is required but not installed"
        exit 1
    fi
    print_status "Node.js $(node --version) found"
    
    # Check npm/yarn
    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed"
        exit 1
    fi
    print_status "npm $(npm --version) found"
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        print_status "Docker $(docker --version) found (optional)"
    else
        print_warning "Docker not found (optional, but recommended)"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is required but not installed"
        exit 1
    fi
    print_status "Git $(git --version) found"
}

# Create directory structure
setup_directories() {
    echo -e "\nSetting up directory structure..."
    
    mkdir -p backend/api/routes
    mkdir -p backend/api/middleware
    mkdir -p backend/logs
    mkdir -p frontend/.next
    mkdir -p data/uploads
    mkdir -p data/cache
    mkdir -p vector_db
    mkdir -p models
    
    print_status "Directory structure created"
}

# Setup environment files
setup_env_files() {
    echo -e "\nSetting up environment files..."
    
    # Backend .env
    if [ ! -f backend/.env ]; then
        cat > backend/.env << EOF
# Backend Configuration
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# LM Studio Configuration
LM_STUDIO_URL=http://localhost:1234
GEMMA_MODEL=google/gemma-3n
QWEN_MODEL=qwen/qwen3-4b-thinking-2507

# Database Configuration
VECTOR_DB_PATH=./vector_db
CHROMA_DB_PERSIST=True

# File Upload Configuration
MAX_FILE_SIZE_MB=100
UPLOAD_DIR=./data/uploads
ALLOWED_EXTENSIONS=.pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.bmp,.tiff,.webp,.wav,.mp3,.m4a,.flac,.ogg

# Security Configuration
ENABLE_AUDIT_LOGGING=True
ENABLE_ANOMALY_DETECTION=True

# Performance Configuration
BATCH_SIZE=32
GPU_ENABLED=True
CACHE_DIR=./data/cache
EOF
        print_status "Backend .env created"
    else
        print_warning "Backend .env already exists, skipping"
    fi
    
    # Frontend .env.local
    if [ ! -f frontend/.env.local ]; then
        cat > frontend/.env.local << EOF
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# LM Studio Configuration
NEXT_PUBLIC_LM_STUDIO_URL=http://localhost:1234

# Upload Configuration
NEXT_PUBLIC_MAX_FILE_SIZE=104857600
NEXT_PUBLIC_ALLOWED_FILE_TYPES=.pdf,.docx,.doc,.txt,.jpg,.jpeg,.png,.bmp,.tiff,.webp,.wav,.mp3,.m4a,.flac,.ogg

# Feature Flags
NEXT_PUBLIC_ENABLE_VOICE_INPUT=true
NEXT_PUBLIC_ENABLE_ANALYTICS=true
NEXT_PUBLIC_ENABLE_FEEDBACK=true

# UI Configuration
NEXT_PUBLIC_APP_NAME=NeuraX
NEXT_PUBLIC_APP_VERSION=1.0.0
EOF
        print_status "Frontend .env.local created"
    else
        print_warning "Frontend .env.local already exists, skipping"
    fi
}

# Install backend dependencies
install_backend() {
    echo -e "\nInstalling backend dependencies..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_status "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    print_status "Backend dependencies installed"
    
    # Install additional API dependencies
    pip install fastapi uvicorn python-multipart websockets aiofiles
    print_status "API dependencies installed"
    
    deactivate
    cd ..
}

# Install frontend dependencies
install_frontend() {
    echo -e "\nInstalling frontend dependencies..."
    
    cd frontend
    
    # Initialize npm project if package.json doesn't exist
    if [ ! -f package.json ]; then
        cat > package.json << 'EOF'
{
  "name": "neurax-frontend",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.3.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "@types/node": "^20.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^14.0.0"
  }
}
EOF
        print_status "package.json created"
    fi
    
    # Install npm packages
    npm install
    print_status "Frontend dependencies installed"
    
    cd ..
}

# Initialize database
init_database() {
    echo -e "\nInitializing vector database..."
    
    cd backend
    source venv/bin/activate
    
    python3 << EOF
from indexing.vector_store import VectorStore
import os

# Initialize ChromaDB
vector_store = VectorStore()
print("Vector database initialized successfully")
EOF
    
    deactivate
    cd ..
    
    print_status "Database initialized"
}

# Create FastAPI wrapper
create_api_wrapper() {
    echo -e "\nCreating API wrapper..."
    
    mkdir -p backend/api/routes
    
    # Main API file
    cat > backend/api/main.py << 'EOF'
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
EOF
    
    print_status "API wrapper created"
}

# Create verification script
create_verification_script() {
    echo -e "\nCreating verification script..."
    
    cat > verify_installation.sh << 'EOF'
#!/bin/bash

# Verification Script for NeuraX

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "NeuraX Installation Verification"
echo "================================="

# Test 1: Check backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ Backend API is running${NC}"
else
    echo -e "${RED}✗ Backend API is not running${NC}"
    echo "Start it with: cd backend && source venv/bin/activate && uvicorn api.main:app"
fi

# Test 2: Check frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✓ Frontend is running${NC}"
else
    echo -e "${RED}✗ Frontend is not running${NC}"
    echo "Start it with: cd frontend && npm run dev"
fi

# Test 3: Check LM Studio
if curl -s http://localhost:1234/v1/models > /dev/null; then
    echo -e "${GREEN}✓ LM Studio is running${NC}"
else
    echo -e "${RED}✗ LM Studio is not running${NC}"
    echo "Please start LM Studio and load a model."
fi

# Test 4: Check database
if [ -d "vector_db" ] && [ "$(ls -A vector_db)" ]; then
    echo -e "${GREEN}✓ Vector database is initialized${NC}"
else
    echo -e "${RED}✗ Vector database not initialized${NC}"
    echo "Run: ./setup/init_database.sh"
fi

echo ""
echo "Verification complete!"
EOF
    
    chmod +x verify_installation.sh
    print_status "Verification script created"
}

# Main installation flow
main() {
    echo "Starting NeuraX installation..."
    echo ""
    
    check_prerequisites
    setup_directories
    setup_env_files
    install_backend
    install_frontend
    init_database
    create_api_wrapper
    create_verification_script
    
    echo -e "\n${GREEN}╔═══════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Installation Complete! 🎉             ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start LM Studio and load Gemma 3n or Qwen3 4B model"
    echo "2. Start backend: cd backend && source venv/bin/activate && uvicorn api.main:app --reload"
    echo "3. Start frontend: cd frontend && npm run dev"
    echo "4. Run verification: ./verify_installation.sh"
    echo ""
    echo "Access the application at: http://localhost:3000"
    echo "API documentation at: http://localhost:8000/docs"
}

# Run main installation
main