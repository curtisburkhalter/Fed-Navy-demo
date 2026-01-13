#!/bin/bash
# Script to start the Maritime Surveillance demo on a remote Linux device
# with access from a Windows laptop via SSH tunnel

echo "================================================"
echo "Maritime Surveillance Intelligence Generator"
echo "Powered by HP ZGX Nano AI Station - Remote Start"
echo "Using BLIP-2 FLAN-T5-XL + TinyLlama for intelligence reports"
echo "================================================"
echo ""

# Get the hostname/IP of the Linux server
SERVER_IP=$(hostname -I | awk '{print $1}')

echo "Server Information:"
echo "  Hostname/IP: $SERVER_IP"
echo ""

# Kill any existing processes on the ports
echo "Cleaning up old processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null
sleep 2

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

echo ""
echo "======================================"
echo "✅ Demo is running!"
echo "======================================"
echo ""
echo "Access the demo from your Windows laptop:"
echo "👉 http://${SERVER_IP}:8000"
echo ""
echo "Backend API endpoints:"
echo "  - Status: http://${SERVER_IP}:8000/"
echo "  - Load Models: http://${SERVER_IP}:8000/load_models"
echo "  - Process Text: http://${SERVER_IP}:8000/mask_pii"
echo ""
echo "Instructions:"
echo "1. Open the web interface in your browser"
echo "2. Select image to analyze"
echo "3. Click Analyze Image"
echo "4. Click 'Process & Compare' to see both models in action"
echo ""
echo "⚠️  Note: Models use ~32GB each. Ensure sufficient GPU memory!"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "======================================"


# Start the FastAPI server
python3 backend/main.py
