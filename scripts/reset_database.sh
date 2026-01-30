#!/bin/bash

# Reset NeuraX vector database

echo "Resetting NeuraX vector database..."

# Stop services if running
if [ -f .backend.pid ]; then
    echo "Stopping backend..."
    kill $(cat .backend.pid)
    rm .backend.pid
fi

# Remove existing database
if [ -d "vector_db" ]; then
    echo "Removing existing database..."
    rm -rf vector_db
fi

# Reinitialize database
echo "Reinitializing database..."
cd backend
source venv/bin/activate

python3 << EOF
from indexing.vector_store import VectorStore

# Initialize fresh database
vector_store = VectorStore()
print("✓ New vector database created")
EOF

deactivate
cd ..

echo "Database reset complete!"