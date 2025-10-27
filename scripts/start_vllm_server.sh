#!/bin/bash
# Start vLLM server for DotsOCR model

set -e

cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════"
echo "🚀 Starting vLLM Server for DotsOCR"
echo "═══════════════════════════════════════════════════════"
echo ""

# Activate venv
source venv/bin/activate

# Check if model exists
if [ ! -d "weights/DotsOCR" ]; then
    echo "❌ Model not found at weights/DotsOCR"
    echo ""
    echo "Download it first:"
    echo "  python tools/download_model.py"
    echo ""
    exit 1
fi

echo "✓ Model found"
echo ""

# Register model to vLLM
hf_model_path=./weights/DotsOCR
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH

echo "Starting vLLM server on port 8000..."
echo "This may take 1-2 minutes to load..."
echo ""

# Start vLLM
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --chat-template-content-format string \
    --served-model-name model \
    --trust-remote-code \
    --port 8000

