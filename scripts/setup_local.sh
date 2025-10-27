#!/bin/bash
# Complete local setup script

set -e

echo "═══════════════════════════════════════════════════════"
echo "🚀 dots.ocr-ultimate Local Setup"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo "⚠️  CUDA not detected (will use CPU - slow!)"
fi

echo ""
echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

source venv/bin/activate

echo ""
echo "Step 2: Installing dependencies..."
echo "This may take 5-10 minutes..."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements-unified.txt

echo "✓ Dependencies installed"

echo ""
echo "Step 3: Downloading model..."
if [ ! -d "weights/DotsOCR" ]; then
    python3 tools/download_model.py
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

echo ""
echo "Step 4: Testing installation..."
python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Quick start:"
echo "  1. Simple API:      ./scripts/start_simple_api.sh"
echo "  2. Test API:        ./scripts/test_api.sh"
echo ""
echo "Documentation:"
echo "  - docs/DEPLOYMENT_LOCAL.md"
echo "  - docs/ARCHITECTURE.md"
echo ""
