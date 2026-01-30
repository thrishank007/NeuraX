#!/bin/bash

# Run all NeuraX tests

echo "Running NeuraX test suite..."
echo "============================"
echo ""

# Run backend tests
echo "Running backend tests..."
cd backend
source venv/bin/activate

# Run integration tests
if [ -d "../tests/integration" ]; then
    echo "Running integration tests..."
    python -m pytest ../tests/integration/ -v
else
    echo "No integration tests found"
fi

deactivate
cd ..

# Run frontend tests (if available)
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    echo ""
    echo "Running frontend tests..."
    cd frontend
    npm test
    cd ..
else
    echo "No frontend tests configured"
fi

echo ""
echo "Test suite complete!"