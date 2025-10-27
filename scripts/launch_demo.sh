#!/bin/bash
# Launch Gradio demo with vLLM backend

set -e

cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════"
echo "🎨 dots.ocr Gradio Demo Launcher"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Run setup first:"
    echo "  ./scripts/setup_local.sh"
    echo ""
    exit 1
fi

echo "✓ Virtual environment found"
echo ""

# Activate venv
source venv/bin/activate

# Check if model exists
if [ ! -d "weights/DotsOCR" ]; then
    echo "📥 Downloading DotsOCR model..."
    python tools/download_model.py
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🚀 Starting Services"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if vLLM server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use!"
    echo ""
    read -p "Kill existing process? (y/N): " kill_existing
    if [[ $kill_existing == "y" ]]; then
        kill $(lsof -t -i:8000)
        sleep 2
        echo "✓ Killed existing process"
    else
        echo "Using existing vLLM server on port 8000"
    fi
else
    echo "Starting vLLM server on port 8000..."
    echo "(This will take 1-2 minutes to load the model)"
    echo ""
    
    # Register model to vLLM
    hf_model_path=./weights/DotsOCR
    export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
    
    # Start vLLM in background
    CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.8 \
        --chat-template-content-format string \
        --served-model-name model \
        --trust-remote-code \
        --port 8000 \
        > logs/vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    echo "✓ vLLM server started (PID: $VLLM_PID)"
    
    # Wait for server to be ready
    echo "Waiting for vLLM server to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ vLLM server is ready!"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
fi

echo ""
echo "Starting Gradio demo on port 7860..."
echo ""

# Start Gradio
python demo/demo_gradio.py 7860

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Demo stopped. To kill vLLM server:"
echo "  kill \$(lsof -t -i:8000)"
echo "═══════════════════════════════════════════════════════"

