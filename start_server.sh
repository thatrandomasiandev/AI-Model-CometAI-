#!/bin/bash
# CometAI Self-Hosted Server Startup Script

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🏠 Starting CometAI Self-Hosted Server${NC}"
echo "====================================="

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

# Get local IP for network access
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "localhost")

echo -e "${GREEN}✅ Starting CometAI server...${NC}"
echo -e "${YELLOW}Local access: http://localhost:8080${NC}"
echo -e "${YELLOW}Network access: http://${LOCAL_IP}:8080${NC}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python server.py --host 0.0.0.0 --port 8080
