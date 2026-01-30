#!/bin/bash

# Initialize vector database for NeuraX

echo "Initializing NeuraX vector database..."

cd backend
source venv/bin/activate

python3 << EOF
from indexing.vector_store import VectorStore
import os

# Initialize ChromaDB
print("Initializing ChromaDB...")
vector_store = VectorStore()

# Verify initialization
if vector_store:
    print("✓ Vector database initialized successfully")
    print(f"Database path: {vector_store.db_path}")
else:
    print("✗ Failed to initialize database")
    exit(1)
EOF

deactivate
cd ..

echo "Database initialization complete!"