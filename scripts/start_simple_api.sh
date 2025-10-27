#!/bin/bash
# Start Simple API server

set -e

echo "🚀 Starting dots.ocr Simple API..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements-unified.txt"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if model exists
if [ ! -d "weights/DotsOCR" ]; then
    echo "📥 Model not found, downloading..."
    python3 tools/download_model.py
fi

# Set environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_DTYPE=${TORCH_DTYPE:-auto}

echo "✓ Model: weights/DotsOCR"
echo "✓ GPU: $CUDA_VISIBLE_DEVICES"
echo "✓ Port: 5000"
echo ""
echo "Starting server..."

# Start server
python api/simple/ocr_api_server.py
