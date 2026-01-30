#!/bin/bash

# Stop all NeuraX services

echo "Stopping NeuraX services..."

# Stop backend
if [ -f .backend.pid ]; then
    kill $(cat .backend.pid)
    rm .backend.pid
    echo "Backend stopped"
fi

# Stop frontend
if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid)
    rm .frontend.pid
    echo "Frontend stopped"
fi

echo "All services stopped"