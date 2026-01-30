#!/bin/bash

# Download models for NeuraX

echo "NeuraX Model Download Script"
echo "============================"
echo ""

echo "This script helps you set up the required models for NeuraX."
echo ""

echo "Required Models:"
echo "1. Gemma 3n - For multimodal queries (text + images)"
echo "2. Qwen3 4B Thinking 2507 - For complex reasoning tasks"
echo ""

echo "Please follow these steps:"
echo ""

echo "1. Install LM Studio from https://lmstudio.ai/"
echo "2. Launch LM Studio application"
echo "3. Go to the 'Models' tab"
echo "4. Search for 'Gemma 3n' and download it"
echo "5. Search for 'Qwen3 4B Thinking 2507' and download it"
echo "6. Go to 'Local Server' tab"
echo "7. Load the models you just downloaded"
echo "8. Start the server on port 1234"
echo ""

echo "After completing these steps, verify the server is running:"
echo "curl http://localhost:1234/v1/models"
echo ""

echo "Note: Models are managed by LM Studio, not downloaded directly by this script."
echo "This ensures better performance and easier model management."
echo ""

echo "For more information, refer to the LM Studio documentation."