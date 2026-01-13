#!/bin/bash
# Download LLaVA model for Maritime Surveillance Intelligence Generator
# Powered by HP ZGX Nano AI Station

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

echo "============================================================"
echo "Maritime Surveillance Intelligence Generator"
echo "Model Download Script"
echo "============================================================"
echo ""

mkdir -p "${MODELS_DIR}"
cd "${MODELS_DIR}"

# LLaVA 1.6 Mistral 7B - Excellent multimodal performance
# Using Q4_K_M quantization for optimal performance/quality balance

MODEL_URL="https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf"
MODEL_FILE="llava-v1.6-mistral-7b.Q4_K_M.gguf"

CLIP_URL="https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf"
CLIP_FILE="mmproj-model-f16.gguf"

echo "Downloading LLaVA 1.6 Mistral 7B model (Q4_K_M)..."
echo "This is approximately 4GB and may take several minutes."
echo ""

if [ -f "${MODEL_FILE}" ]; then
    echo "Model file already exists: ${MODEL_FILE}"
    echo "Skipping download. Delete the file to re-download."
else
    echo "Downloading: ${MODEL_URL}"
    wget --progress=bar:force -O "${MODEL_FILE}" "${MODEL_URL}"
    echo "Model downloaded successfully!"
fi

echo ""
echo "Downloading CLIP vision encoder (mmproj)..."

if [ -f "${CLIP_FILE}" ]; then
    echo "CLIP file already exists: ${CLIP_FILE}"
    echo "Skipping download. Delete the file to re-download."
else
    echo "Downloading: ${CLIP_URL}"
    wget --progress=bar:force -O "${CLIP_FILE}" "${CLIP_URL}"
    echo "CLIP encoder downloaded successfully!"
fi

echo ""
echo "============================================================"
echo "Download complete!"
echo ""
echo "Model files:"
ls -lh "${MODELS_DIR}"
echo ""
echo "You can now run the demo with: ./start_demo.sh"
echo "============================================================"