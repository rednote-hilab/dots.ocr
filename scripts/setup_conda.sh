#!/bin/bash
# Setup with Conda instead of venv

set -e

echo "═══════════════════════════════════════════════════════"
echo "🚀 dots.ocr-ultimate Setup with Conda"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo ""
    echo "Install Miniconda:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    exit 1
fi

echo "✓ Conda found: $(conda --version)"
echo ""

# Create conda environment
ENV_NAME="dots-ocr-ultimate"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "✓ Conda environment '$ENV_NAME' exists"
    read -p "Recreate? (y/N): " recreate
    if [[ $recreate == "y" ]]; then
        conda env remove -n $ENV_NAME -y
        echo "✓ Old environment removed"
    fi
fi

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
    echo "✓ Environment created"
fi

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "✓ Environment activated: $ENV_NAME"
echo ""

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

echo "✓ PyTorch installed"
echo ""

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements-unified.txt

echo "✓ Dependencies installed"
echo ""

# Download model
echo "Downloading model..."
if [ ! -d "weights/DotsOCR" ]; then
    python tools/download_model.py
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

echo ""
echo "Testing installation..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}'); print(f'✓ CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "To activate environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To start API:"
echo "  conda activate $ENV_NAME"
echo "  python api/simple/ocr_api_server.py"
echo ""
echo "To deactivate:"
echo "  conda deactivate"
echo ""

