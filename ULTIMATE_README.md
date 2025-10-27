# 🚀 ULTIMATE dots.ocr - The Complete Package

<div align="center">

**The most comprehensive dots.ocr fork - combining the best features from all community forks**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Original](https://img.shields.io/badge/Based%20on-rednote--hilab/dots.ocr-orange.svg)](https://github.com/rednote-hilab/dots.ocr)

</div>

---

## 🎯 What is this?

**ULTIMATE dots.ocr** is a unified fork that combines the best components from the entire dots.ocr ecosystem:

| Component | Source | Description |
|-----------|--------|-------------|
| 🔧 **Simple API** | [am009](https://github.com/am009/dots.ocr) | Flask REST API with auto GPU detection |
| 🏢 **Enterprise API** | [AKCqhzdy](https://github.com/AKCqhzdy/dots.ocr) | FastAPI with PostgreSQL, Redis, Tracing |
| 🎓 **Training Suite** | [wjbmattingly](https://github.com/wjbmattingly/dots.ocr) | Full fine-tuning + LoRA support |
| 🐳 **Docker** | [am009](https://github.com/am009/dots.ocr) | Production-ready containers |
| 📄 **Structured PDF** | [AKCqhzdy/zihao](https://github.com/AKCqhzdy/dots.ocr/tree/zihao) | PP-DocLayout + text extraction |
| 💎 **Core Model** | [rednote-hilab](https://github.com/rednote-hilab/dots.ocr) | Original SOTA 1.7B model |

---

## 📦 Repository Structure

```
dots.ocr-ultimate/
├── api/
│   ├── simple/              # 🔧 Flask REST API (am009)
│   │   ├── ocr_api_server.py
│   │   ├── API_Documentation_en.md
│   │   └── test_ocr_api.py
│   └── enterprise/          # 🏢 FastAPI Enterprise (AKCqhzdy)
│       └── app/
│           ├── dotsocr_service.py
│           └── utils/       # PostgreSQL, Redis, Tracing
│
├── training/                # 🎓 Training Suite (wjbmattingly)
│   ├── train_simple.py
│   ├── train_dotsocr.py     # LoRA support
│   ├── create_training_data.py
│   └── README_model_training.md
│
├── deployment/              # 🐳 Deployment configs
│   └── docker/              # Docker from am009
│       ├── Dockerfile
│       └── start.sh
│
├── dots_ocr/                # 💎 Core model
│   ├── model/
│   │   ├── inference.py
│   │   └── layout_service.py  # Structured PDF (zihao)
│   ├── parser.py
│   └── utils/
│       └── pdf_extractor.py   # Text extraction (zihao)
│
└── requirements-unified.txt  # All dependencies
```

---

## 🚀 Quick Start Guide

### Option 1: Simple API (Quick Prototyping)

**Best for:** Small projects, rapid development, single GPU

```bash
# Install
pip install -r requirements-unified.txt

# Run Simple API server
python api/simple/ocr_api_server.py

# Use
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "/path/to/image.jpg",
    "image_format": "path",
    "prompt_type": "prompt_layout_all_en"
  }'
```

**Features:**
- ✅ RTX 20xx support (float32)
- ✅ RTX 30xx+ support (bfloat16)
- ✅ Base64, URL, file path input
- ✅ Streaming responses

---

### Option 2: Enterprise API (Production)

**Best for:** Large-scale deployments, enterprises, high-load systems

```bash
# Setup environment
export POSTGRES_URL_NO_SSL_DEV="postgresql://user:pass@localhost/db"
export API_KEY="your-openai-api-key"
export OCR_INFERENCE_HOST="localhost"
export OCR_INFERENCE_PORT="8000"

# Run Enterprise service
python api/enterprise/app/dotsocr_service.py

# API endpoints
# - POST /parse - Parse documents
# - GET /token_usage - Token statistics
# - GET /status - Job status
```

**Features:**
- ✅ PostgreSQL + PGVector
- ✅ Redis caching
- ✅ OpenTelemetry tracing
- ✅ Job queue with retry
- ✅ Token tracking
- ✅ OSS storage integration

---

### Option 3: Docker (One-Click Deploy)

**Best for:** Quick deployment, testing, CI/CD

```bash
# For RTX 20xx (Turing or older)
docker run --name dots-ocr -d \
  --runtime=nvidia --gpus=all \
  -p 5000:5000 \
  docker.io/am009/dots.ocr:latest

# For RTX 30xx and newer
docker run --name dots-ocr -d \
  --runtime=nvidia --gpus=all \
  --entrypoint bash \
  -p 5000:5000 \
  docker.io/am009/dots.ocr:latest \
  -c "sed -i 's/bf16=False/bf16=True/' /DotsOCR/weights/DotsOCR/modeling_dots_vision.py; /DotsOCR/start.sh"
```

---

### Option 4: Gradio Demo (Interactive UI)

**Best for:** Testing, visualization, quick document processing

```bash
# Setup (one-time)
bash scripts/setup_local.sh

# Start demo server
bash scripts/start_demo.sh

# Access at: http://0.0.0.0:7860
```

**Features:**
- 🎨 Interactive web interface
- 📊 Real-time GPU memory monitoring with timestamp logging
- ⏱️ Processing time tracking
- 🎮 Configurable GPU memory limit (default: 80% = 6.4GB for 8GB cards)
- 📝 Visual layout analysis output
- 🔍 Detailed process logs

**GPU Configuration:**
The demo automatically manages GPU memory:
- Model loading: ~5.5GB VRAM
- Peak inference: ~6GB VRAM
- Memory limit: Adjustable in `demo/demo_simple.py` (line 19)
- Logs: Saved to `logs/demo_YYYYMMDD_HHMMSS.log` with timestamps

---

### Option 5: Training (Fine-tune on Your Data)

**Best for:** Custom domains, specialized documents, research

```bash
# 1. Prepare training data
python training/create_training_data.py \
  --input_dir /path/to/pagexml+images \
  --output_file training_data.jsonl

# 2. Simple training
python training/train_simple.py \
  --data training_data.jsonl \
  --epochs 3 \
  --batch_size 1

# 3. LoRA training (memory efficient)
python training/train_dotsocr.py \
  --train_data training_data.jsonl \
  --lora_training \
  --lora_rank 8 \
  --learning_rate 1e-4 \
  --bf16

# 4. With W&B monitoring
python training/train_simple.py \
  --data training_data.jsonl \
  --wandb
```

**Training Strategies:**
- 🔥 **Full Fine-tuning**: Best performance (~24GB GPU)
- 💚 **LoRA**: Memory efficient (~12GB GPU)
- ⚡ **Freeze Vision**: Fast, text-focused

---

## 🌟 Unique Features

### 1. Structured PDF Support (zihao branch)

For "clean" PDFs with embedded text layer:

```python
from dots_ocr.parser import DotsOCRParser

parser = DotsOCRParser(use_layout_detection=True)
result = parser.parse("structured_document.pdf")
```

**How it works:**
- 📄 PP-DocLayout-L detects layout
- 📝 PyMuPDF extracts text directly
- ⚡ 10x faster than OCR (no inference)

---

### 2. Multiple API Modes

```python
# Full parsing (layout + text)
"prompt_layout_all_en"

# Layout detection only
"prompt_layout_only_en"

# OCR only (no headers/footers)
"prompt_ocr"

# Grounding OCR (by coordinates)
"prompt_grounding_ocr"
```

---

### 3. Enterprise Features

**PostgreSQL Vector Search:**
- Document deduplication
- Vector-based content search
- Processing history

**OpenTelemetry Tracing:**
- Distributed tracing
- Performance monitoring
- Error tracking

**Token Usage API:**
```bash
curl http://localhost:8000/token_usage
# {
#   "total_tokens": 125000,
#   "cost_usd": 0.25,
#   "requests": 45
# }
```

---

## 📊 Comparison Table

| Mode | Use Case | GPU | Complexity | Production |
|------|----------|-----|-----------|-----------|
| **Simple API** | Prototypes, small projects | 1x RTX 2080+ | ⭐ | ⭐⭐⭐ |
| **Enterprise API** | Corporations, high-load | 2x+ GPU | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Docker** | Quick deploy | 1x GPU | ⭐ | ⭐⭐⭐⭐ |
| **Training** | Custom models | 1x A100/H100 | ⭐⭐⭐⭐ | ⭐⭐ |

---

## 🔧 Configuration

### Simple API

```bash
# Auto-detect GPU type
TORCH_DTYPE=auto  # float32/float16/bfloat16
```

### Enterprise API

```bash
# Required
POSTGRES_URL_NO_SSL_DEV=postgresql://...
API_KEY=sk-...

# Optional
OSS_ENDPOINT=https://...
OSS_ACCESS_KEY_ID=...
OSS_ACCESS_KEY_SECRET=...
NUM_WORKERS=4
CONCURRENT_OCR_TASK_LIMIT=2
API_TIMEOUT=300
```

---

## 📚 Documentation

- **Simple API**: `api/simple/API_Documentation_en.md`
- **Training**: `training/README_model_training.md`
- **Enterprise**: See `api/enterprise/app/dotsocr_service.py` comments
- **Original**: `README.md`

---

## 🎁 Credits

This ultimate fork combines work from:

- **[rednote-hilab](https://github.com/rednote-hilab/dots.ocr)** - Original SOTA model
- **[AKCqhzdy](https://github.com/AKCqhzdy/dots.ocr)** - Enterprise API, Structured PDF
- **[am009](https://github.com/am009/dots.ocr)** - Simple API, Docker
- **[wjbmattingly](https://github.com/wjbmattingly/dots.ocr)** - Training Suite

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 💬 Support & Contributing

Found a bug? Have a feature request?
- Open an issue on GitHub
- Check individual component documentation
- Refer to original repos for specific questions

---

**Version**: Ultimate v1.0  
**Date**: October 2025  
**Status**: 🚀 Production Ready  
**Maintainer**: Community-driven fork

