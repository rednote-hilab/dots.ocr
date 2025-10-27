#!/bin/bash
# Start Simple API with Conda

set -e

ENV_NAME="dots-ocr-ultimate"

# Check if conda environment exists
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "❌ Conda environment '$ENV_NAME' not found!"
    echo ""
    echo "Run setup first:"
    echo "  ./scripts/setup_conda.sh"
    echo ""
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo "🚀 Starting Simple API with Conda"
echo "═══════════════════════════════════════════════════════"
echo ""

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "✓ Environment activated: $ENV_NAME"
echo ""

# Start API
cd "$(dirname "$0")/.."
export FLASK_APP=api/simple/ocr_api_server.py
export FLASK_ENV=development

echo "Starting API on http://localhost:5000..."
echo ""

python api/simple/ocr_api_server.py

