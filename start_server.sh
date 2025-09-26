#!/bin/bash
# CometAI Server Startup Script
# Simple script to start the CometAI server with proper environment setup

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting CometAI Server${NC}"
echo "=========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables for production
export ENVIRONMENT=production
export HOST=0.0.0.0
export PORT=8080
export LOG_LEVEL=INFO

# Create logs directory
mkdir -p logs

echo -e "${GREEN}✅ Starting CometAI server on http://0.0.0.0:8080${NC}"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python server.py --host 0.0.0.0 --port 8080
