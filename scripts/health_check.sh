#!/bin/bash

# Comprehensive health check for NeuraX

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "NeuraX System Health Check"
echo "=========================="
echo ""

# Function to check service
check_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Checking $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" $url)
    
    if [ "$response" -eq "$expected" ]; then
        echo -e "${GREEN}✓ HEALTHY${NC}"
        return 0
    else
        echo -e "${RED}✗ UNHEALTHY (HTTP $response)${NC}"
        return 1
    fi
}

# Function to check port
check_port() {
    local name=$1
    local port=$2
    
    echo -n "Checking $name (port $port)... "
    
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✓ OPEN${NC}"
        return 0
    else
        echo -e "${RED}✗ CLOSED${NC}"
        return 1
    fi
}

# Check backend API
check_service "Backend API" "http://localhost:8000/health" 200

# Check frontend
check_service "Frontend" "http://localhost:3000" 200

# Check LM Studio
check_service "LM Studio" "http://localhost:1234/v1/models" 200

# Check ports
check_port "Backend Port" 8000
check_port "Frontend Port" 3000
check_port "LM Studio Port" 1234

# Check database
echo -n "Checking vector database... "
if [ -d "vector_db" ] && [ "$(ls -A vector_db)" ]; then
    echo -e "${GREEN}✓ EXISTS${NC}"
else
    echo -e "${YELLOW}⚠ EMPTY${NC}"
fi

# Check disk space
echo -n "Checking disk space... "
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    echo -e "${GREEN}✓ OK (${DISK_USAGE}% used)${NC}"
else
    echo -e "${YELLOW}⚠ LOW (${DISK_USAGE}% used)${NC}"
fi

# Check memory
echo -n "Checking memory... "
if command -v free &> /dev/null; then
    MEM_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [ "$MEM_USAGE" -lt 90 ]; then
        echo -e "${GREEN}✓ OK (${MEM_USAGE}% used)${NC}"
    else
        echo -e "${YELLOW}⚠ HIGH (${MEM_USAGE}% used)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ CANNOT DETERMINE${NC}"
fi

echo ""
echo "Health check complete!"