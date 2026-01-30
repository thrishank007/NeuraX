#!/bin/bash

# Test script to verify NeuraX setup

echo "Testing NeuraX Setup"
echo "===================="
echo ""

# Test 1: Check if setup script exists
echo "1. Checking setup script..."
if [ -f "setup/setup.sh" ]; then
    echo "   ✓ setup.sh exists"
else
    echo "   ✗ setup.sh missing"
fi

# Test 2: Check if service scripts exist
echo "2. Checking service scripts..."
if [ -f "scripts/start_all.sh" ] && [ -f "scripts/stop_all.sh" ]; then
    echo "   ✓ Service scripts exist"
else
    echo "   ✗ Service scripts missing"
fi

# Test 3: Check Docker configuration
echo "3. Checking Docker configuration..."
if [ -f "docker/docker-compose.yml" ]; then
    echo "   ✓ Docker compose file exists"
else
    echo "   ✗ Docker compose file missing"
fi

# Test 4: Check API wrapper
echo "4. Checking API wrapper..."
if [ -f "backend/api/main.py" ]; then
    echo "   ✓ API wrapper exists"
else
    echo "   ✗ API wrapper missing"
fi

# Test 5: Check frontend structure
echo "5. Checking frontend structure..."
if [ -f "frontend/package.json" ] && [ -f "frontend/pages/index.tsx" ]; then
    echo "   ✓ Frontend structure exists"
else
    echo "   ✗ Frontend structure incomplete"
fi

# Test 6: Check documentation
echo "6. Checking documentation..."
if [ -f "docs/SETUP_GUIDE.md" ] && [ -f "docs/TROUBLESHOOTING.md" ]; then
    echo "   ✓ Documentation exists"
else
    echo "   ✗ Documentation missing"
fi

# Test 7: Check test files
echo "7. Checking test files..."
if [ -f "tests/integration/test_api_connection.py" ]; then
    echo "   ✓ Test files exist"
else
    echo "   ✗ Test files missing"
fi

echo ""
echo "Setup verification complete!"
echo ""
echo "To run the full setup, execute:"
echo "  ./setup/setup.sh"
echo ""
echo "To start services:"
echo "  ./scripts/start_all.sh"