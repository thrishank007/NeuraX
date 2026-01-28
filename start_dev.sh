#!/bin/bash

# NeuraX Development Server Launcher
# This script starts both the backend FastAPI server and frontend Next.js app

set -e

echo "🚀 Starting NeuraX Development Environment"

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

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Install frontend dependencies
print_status "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p data/vector_db
mkdir -p data/evaluations
mkdir -p logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export NEXT_PUBLIC_API_URL="http://localhost:8000"

print_success "Environment setup complete!"

# Function to start backend
start_backend() {
    print_status "Starting FastAPI backend server..."
    cd backend
    
    # Run with uvicorn
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info &
    BACKEND_PID=$!
    
    cd ..
    echo $BACKEND_PID > backend.pid
    print_success "Backend server started (PID: $BACKEND_PID)"
    print_status "Backend API available at: http://localhost:8000"
    print_status "API documentation available at: http://localhost:8000/docs"
}

# Function to start frontend
start_frontend() {
    print_status "Starting Next.js frontend..."
    cd frontend
    
    # Start Next.js development server
    npm run dev &
    FRONTEND_PID=$!
    
    cd ..
    echo $FRONTEND_PID > frontend.pid
    print_success "Frontend server started (PID: $FRONTEND_PID)"
    print_status "Frontend available at: http://localhost:3000"
}

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down servers..."
    
    # Kill backend
    if [ -f backend.pid ]; then
        BACKEND_PID=$(cat backend.pid)
        kill $BACKEND_PID 2>/dev/null || true
        rm backend.pid
    fi
    
    # Kill frontend
    if [ -f frontend.pid ]; then
        FRONTEND_PID=$(cat frontend.pid)
        kill $FRONTEND_PID 2>/dev/null || true
        rm frontend.pid
    fi
    
    print_success "All servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Start both servers
start_backend
sleep 3  # Give backend time to start
start_frontend

print_success "🎉 NeuraX is now running!"
print_status "Backend API: http://localhost:8000"
print_status "Frontend:    http://localhost:3000"
print_status "API Docs:    http://localhost:8000/docs"
print_warning "Press Ctrl+C to stop all servers"

# Keep script running
wait