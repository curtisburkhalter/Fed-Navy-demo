#!/bin/bash
# Installation script for Maritime Surveillance Intelligence Generator
# Powered by HP ZGX Nano AI Station

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "Maritime Surveillance Intelligence Generator"
echo "Installation Script"
echo "Powered by HP ZGX Nano AI Station"
echo "============================================================"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Found Python: ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv navy-env
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source navy-env/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r backend/requirements.txt

# Install llama-cpp-python with CUDA support for ZGX Nano
echo ""
echo "Installing llama-cpp-python with CUDA support..."
echo "This may take several minutes to compile..."

# For Grace Blackwell (GB10) - use CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

echo ""
echo "============================================================"
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Download the VLM model: ./download_models.sh"
echo "2. Start the demo: ./start_demo.sh"
echo ""
echo "Or for remote access: ./start_demo_remote.sh"
echo "============================================================"