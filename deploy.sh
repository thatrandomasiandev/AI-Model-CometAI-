#!/bin/bash
# CometAI Deployment Script
# This script helps deploy CometAI to a production server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="cometai"
APP_USER="cometai"
APP_DIR="/opt/cometai"
SERVICE_FILE="cometai.service"
PYTHON_VERSION="3.11"

echo -e "${GREEN}🚀 CometAI Production Deployment Script${NC}"
echo "========================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}❌ This script should not be run as root${NC}"
   echo "Please run as a regular user with sudo privileges"
   exit 1
fi

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check system requirements
echo "Checking system requirements..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VER" < "3.9" ]]; then
    print_error "Python 3.9+ is required, found $PYTHON_VER"
    exit 1
fi

print_status "Python $PYTHON_VER found"

# Check available memory
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ $MEMORY_GB -lt 16 ]]; then
    print_warning "System has ${MEMORY_GB}GB RAM. 16GB+ recommended for optimal performance"
fi

print_status "System has ${MEMORY_GB}GB RAM"

# Create application user
echo "Setting up application user..."
if ! id "$APP_USER" &>/dev/null; then
    sudo useradd -r -s /bin/false -d "$APP_DIR" "$APP_USER"
    print_status "Created user $APP_USER"
else
    print_status "User $APP_USER already exists"
fi

# Create application directory
echo "Setting up application directory..."
sudo mkdir -p "$APP_DIR"
sudo mkdir -p "$APP_DIR/logs"
sudo chown -R "$APP_USER:$APP_USER" "$APP_DIR"
print_status "Created application directory $APP_DIR"

# Copy application files
echo "Copying application files..."
sudo cp -r . "$APP_DIR/"
sudo chown -R "$APP_USER:$APP_USER" "$APP_DIR"
print_status "Application files copied"

# Create virtual environment
echo "Setting up Python virtual environment..."
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/venv"
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"
print_status "Virtual environment created and dependencies installed"

# Install systemd service
echo "Installing systemd service..."
sudo cp "$SERVICE_FILE" "/etc/systemd/system/"
sudo systemctl daemon-reload
sudo systemctl enable "$APP_NAME"
print_status "Systemd service installed and enabled"

# Configure firewall (if ufw is available)
if command -v ufw &> /dev/null; then
    echo "Configuring firewall..."
    sudo ufw allow 8080/tcp
    print_status "Firewall configured to allow port 8080"
fi

# Start the service
echo "Starting CometAI service..."
sudo systemctl start "$APP_NAME"

# Wait a moment for startup
sleep 5

# Check service status
if sudo systemctl is-active --quiet "$APP_NAME"; then
    print_status "CometAI service is running"
    
    # Test the API
    echo "Testing API endpoint..."
    if curl -f http://localhost:8080/health &>/dev/null; then
        print_status "API is responding"
    else
        print_warning "API is not responding yet (model may still be loading)"
    fi
else
    print_error "CometAI service failed to start"
    echo "Check logs with: sudo journalctl -u $APP_NAME -f"
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo "Service management commands:"
echo "  Start:   sudo systemctl start $APP_NAME"
echo "  Stop:    sudo systemctl stop $APP_NAME"
echo "  Restart: sudo systemctl restart $APP_NAME"
echo "  Status:  sudo systemctl status $APP_NAME"
echo "  Logs:    sudo journalctl -u $APP_NAME -f"
echo ""
echo "API endpoints:"
echo "  Health:  http://localhost:8080/health"
echo "  Chat:    http://localhost:8080/api/chat"
echo "  Docs:    http://localhost:8080/docs (if not in production mode)"
echo ""
echo -e "${YELLOW}Note: The model will download and load on first startup, which may take several minutes.${NC}"
