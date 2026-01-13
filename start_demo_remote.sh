#!/bin/bash
# Script to start the Maritime Surveillance demo on a remote Linux device
# with access from a Windows laptop via SSH tunnel

echo "================================================"
echo "Maritime Surveillance Intelligence Generator"
echo "Powered by HP ZGX Nano AI Station - Remote Start"
echo "================================================"
echo ""

# Get the hostname/IP of the Linux server
HOSTNAME=$(hostname -I | awk '{print $1}')
echo "Server Information:"
echo "  Hostname/IP: $HOSTNAME"
echo ""

# Check if virtual environment exists
if [ ! -d "navy-env" ]; then
    echo "✗ Virtual environment not found!"
    echo "  Please run install.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source navy-env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Check if main.py exists
if [ ! -f "backend/main.py" ]; then
    echo "✗ backend/main.py not found!"
    exit 1
fi

# Check if frontend exists
if [ ! -f "frontend/index.html" ]; then
    echo "⚠ frontend/index.html not found - UI may not load"
fi

echo "================================================"
echo "Starting Maritime Surveillance Demo Server..."
echo "================================================"
echo ""
echo "Server will be accessible at:"
echo "  Local:  http://localhost:8000"
echo "  Remote: http://${HOSTNAME}:8000"
echo ""
echo "For Windows laptop access via SSH tunnel:"
echo "  1. Open PowerShell or Command Prompt on your Windows laptop"
echo "  2. Run: ssh -L 8000:localhost:8000 $USER@${HOSTNAME}"
echo "  3. Then open browser to: http://localhost:8000"
echo ""
echo "Or use VS Code Remote SSH and forward port 8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

# Start the FastAPI server
python3 backend/main.py