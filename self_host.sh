#!/bin/bash
# CometAI Self-Hosting Setup Script
# Simple setup for self-hosting CometAI

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🏠 CometAI Self-Hosting Setup${NC}"
echo "================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 found${NC}"

# Setup application
echo "Setting up CometAI..."

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Get network info
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")

echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo "Starting CometAI server..."
echo -e "${YELLOW}Local access: http://localhost:8080${NC}"
echo -e "${YELLOW}Network access: http://${LOCAL_IP}:8080${NC}"
echo ""
echo "For external access:"
echo "1. Forward port 8080 on your router"
echo "2. Use your public IP with port 8080"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python server.py --host 0.0.0.0 --port 8080
