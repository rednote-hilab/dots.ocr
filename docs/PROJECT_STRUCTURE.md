# 📁 PROJECT STRUCTURE - dots.ocr-ultimate

Complete file structure with descriptions

## 🗂️ Directory Tree

```
dots.ocr-ultimate/
│
├── 📦 api/                          # API implementations
│   ├── simple/                      # Simple Flask API (am009)
│   │   ├── ocr_api_server.py       # Main Flask server (7.7KB)
│   │   ├── forward_exec.py         # Execution wrapper (9.5KB)
│   │   ├── test_ocr_api.py         # API tests (4.1KB)
│   │   ├── API_Documentation.md    # Chinese docs (6.9KB)
│   │   └── API_Documentation_en.md # English docs (7.1KB)
│   │
│   └── enterprise/                  # Enterprise FastAPI (akcqhzdy)
│       └── app/
│           ├── dotsocr_service.py  # Main service (30KB)
│           └── utils/
│               ├── configs.py           # Configuration (3.1KB)
│               ├── hash.py              # MD5 utilities
│               ├── redis.py             # Redis client
│               ├── storage.py           # OSS storage (7.8KB)
│               ├── tracing.py           # OpenTelemetry (8.3KB)
│               ├── executor/
│               │   ├── __init__.py
│               │   ├── job_executor_pool.py   # Job queue
│               │   ├── task_executor_pool.py  # Task queue
│               │   └── ocr_task.py           # OCR tasks
│               └── pg_vector/
│                   ├── __init__.py
│                   ├── pg_vector.py          # PG client
│                   └── table.py              # ORM models
│
├── 📚 docs/                         # Documentation
│   ├── ARCHITECTURE.md             # Architecture guide
│   ├── PROJECT_STRUCTURE.md        # This file
│   ├── API_GUIDE.md                # API usage (to be created)
│   └── TRAINING_GUIDE.md           # Training guide (to be created)
│
├── 💎 dots_ocr/                     # Core model (original)
│   ├── __init__.py
│   ├── parser.py                   # Main parser interface
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── inference.py           # VLM inference
│   │   └── layout_service.py      # PP-DocLayout (zihao)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── consts.py              # Constants
│       ├── image_utils.py         # Image processing
│       ├── page_parser.py         # Page parsing
│       ├── pdf_extractor.py       # PDF text extraction (zihao)
│       ├── prompts.py             # Prompt templates
│       └── demo_utils/
│
├── 🎓 training/                     # Training suite (wjbmattingly)
│   ├── train_simple.py             # Simple training (14.7KB)
│   ├── train_dotsocr.py            # Advanced training (20.5KB)
│   ├── create_training_data.py    # Data prep (19.4KB)
│   ├── example_usage.py            # Examples (3.4KB)
│   ├── run_training.sh             # Automation script
│   ├── test_training.py            # Tests (7.9KB)
│   ├── test_training_script.py    # Test scripts (8.2KB)
│   ├── config_training.yaml        # Config template
│   ├── training_requirements.txt   # Training deps
│   ├── README_model_training.md    # Main guide (7.0KB)
│   └── README_training.md          # Quick start (3.5KB)
│
├── 🐳 deployment/                   # Deployment configs
│   └── docker/                     # Docker (am009)
│       ├── Dockerfile              # Container def (1.2KB)
│       ├── start.sh                # Start script
│       ├── daemon-start.sh         # Daemon start
│       └── stop.sh                 # Stop script
│
├── 🎨 assets/                       # Images and resources
│   ├── logo.png
│   ├── chart.png
│   ├── showcase/                   # Example outputs
│   └── showcase_origin/
│
├── 🎮 demo/                         # Demo applications
│   ├── demo_hf.py                  # HuggingFace demo
│   ├── demo_vllm.py                # vLLM demo
│   ├── demo_gradio.py              # Gradio UI
│   ├── demo_gradio_annotion.py    # Annotation demo
│   ├── demo_image1.jpg             # Test images
│   └── demo_pdf1.pdf               # Test PDFs
│
├── 🛠️ tools/                        # Utility scripts
│   └── download_model.py           # Model downloader
│
├── 📦 app/                          # Enterprise app (duplicate for compat)
│   └── [same as api/enterprise/app/]
│
├── 📋 Root Files
│   ├── README.md                   # Original README (30KB)
│   ├── ULTIMATE_README.md          # Ultimate README (new)
│   ├── PUSH_INSTRUCTIONS.md        # Git push guide
│   │
│   ├── requirements.txt            # Original deps
│   ├── requirements-unified.txt    # All deps combined
│   ├── setup.py                    # Package setup
│   │
│   ├── LICENSE                     # MIT License
│   ├── NOTICE                      # Attribution (115KB)
│   ├── dots.ocr LICENSE AGREEMENT  # License details
│   │
│   └── .gitignore                  # Git ignore rules
│
└── 🐳 docker/                       # Original docker (kept for compat)
    └── ...
```

---

## 📊 File Statistics

### By Category

| Category | Files | Lines | Size |
|----------|-------|-------|------|
| **Core Model** | ~15 | ~3,000 | ~50KB |
| **APIs** | ~20 | ~7,000 | ~100KB |
| **Training** | ~11 | ~2,500 | ~100KB |
| **Documentation** | ~10 | ~2,000 | ~100KB |
| **Tests** | ~8 | ~1,000 | ~30KB |
| **Total** | **~70+** | **~13,000+** | **~400KB+** |

### By Language

| Language | Files | Percentage |
|----------|-------|------------|
| Python | ~60 | 85% |
| Markdown | ~10 | 10% |
| Shell | ~4 | 3% |
| YAML | ~2 | 2% |

---

## 🔑 Key Files Explained

### Core Model

**`dots_ocr/parser.py`**
- Main entry point for document parsing
- Handles images and PDFs
- Coordinates between OCR and layout detection
- Outputs JSON, Markdown, and visualizations

**`dots_ocr/model/inference.py`**
- VLM inference implementation
- Qwen2.5-VL model loading
- Prompt processing
- Token generation

**`dots_ocr/model/layout_service.py`** *(zihao)*
- PP-DocLayout-L model integration
- Layout element detection
- Bounding box prediction
- Category classification

**`dots_ocr/utils/pdf_extractor.py`** *(zihao)*
- PyMuPDF-based text extraction
- Detects structured vs scanned PDFs
- Direct text extraction for structured PDFs
- Fallback to OCR for scanned

---

### API Layer

**`api/simple/ocr_api_server.py`**
- Flask REST API server
- `/health` and `/ocr` endpoints
- Auto GPU type detection
- Temp file management
- Processing lock

**`api/enterprise/app/dotsocr_service.py`**
- FastAPI service
- Job queue system
- PostgreSQL integration
- Redis caching
- OpenTelemetry tracing
- Multiple endpoints

**`api/enterprise/app/utils/executor/job_executor_pool.py`**
- Async job queue
- Concurrency control
- Retry logic
- Job history tracking

**`api/enterprise/app/utils/pg_vector/pg_vector.py`**
- PostgreSQL client
- Vector embedding storage
- MD5-based deduplication
- Query interface

---

### Training Suite

**`training/train_simple.py`**
- Beginner-friendly training
- CLI interface
- W&B integration
- Auto checkpointing
- Progress bars

**`training/train_dotsocr.py`**
- Advanced training
- LoRA support
- Parameter freezing
- Mixed precision
- DeepSpeed compatible

**`training/create_training_data.py`**
- PAGEXML parser
- Image-text pairing
- JSONL generation
- Validation checks

**`training/config_training.yaml`**
- Training hyperparameters
- Model configuration
- Data paths
- Hardware settings

---

### Deployment

**`deployment/docker/Dockerfile`**
- NVIDIA CUDA base
- Python 3.10
- Dependencies installation
- Model weights
- API server

**`deployment/docker/start.sh`**
- Environment setup
- Model loading
- Server startup
- Port binding

---

## 📦 Important Directories

### `api/` - API Implementations

Two complete API implementations for different use cases:
- **simple/**: Quick setup, single GPU, Flask-based
- **enterprise/**: Production-ready, scalable, FastAPI-based

### `training/` - Training Infrastructure

Everything needed to fine-tune the model:
- Training scripts (simple + advanced)
- Data preparation tools
- Configuration templates
- Documentation
- Tests

### `docs/` - Documentation

Comprehensive documentation:
- Architecture guide
- API reference
- Training tutorials
- Deployment guides

### `dots_ocr/` - Core Model

Original dots.ocr implementation + enhancements:
- Model loading and inference
- Parser interface
- Utilities for image/PDF processing
- Structured PDF support (zihao)

---

## 🔄 Duplicate Files Explained

Some files appear in multiple locations for compatibility:

### `app/` vs `api/enterprise/app/`
- `api/enterprise/app/`: New organized location
- `app/`: Root level for backward compatibility
- Both are identical copies

### Root Level Scripts
- `Dockerfile`, `start.sh`, etc. at root
- Also in `deployment/docker/`
- Root level for Docker builds
- `deployment/` for organization

### Requirements Files
- `requirements.txt`: Original dependencies
- `requirements-unified.txt`: All forks combined
- `training/training_requirements.txt`: Training-specific

---

## 📝 Configuration Files

### Python Configuration
- `setup.py`: Package installation
- `requirements.txt`: Core dependencies
- `requirements-unified.txt`: All dependencies

### Training Configuration
- `training/config_training.yaml`: Training hyperparams
- `training/training_requirements.txt`: Training deps

### Docker Configuration
- `deployment/docker/Dockerfile`: Container definition
- `.dockerignore`: Docker ignore rules (if present)

### Git Configuration
- `.gitignore`: Git ignore rules
- `.git/`: Git repository (hidden)

---

## 🎯 Entry Points

### API Servers
```bash
# Simple API
python api/simple/ocr_api_server.py

# Enterprise API
python api/enterprise/app/dotsocr_service.py
```

### Training
```bash
# Simple training
python training/train_simple.py --data data.jsonl

# Advanced training
python training/train_dotsocr.py --train_data data.jsonl
```

### Parser (Direct Usage)
```bash
# Parse document
python dots_ocr/parser.py input.pdf
```

### Demos
```bash
# Gradio UI
python demo/demo_gradio.py

# HuggingFace
python demo/demo_hf.py

# vLLM
python demo/demo_vllm.py
```

---

## 🔍 Finding Files

### By Functionality

**OCR/Parsing:**
- `dots_ocr/parser.py`
- `dots_ocr/model/inference.py`

**Layout Detection:**
- `dots_ocr/model/layout_service.py`
- `dots_ocr/utils/pdf_extractor.py`

**Simple API:**
- `api/simple/ocr_api_server.py`

**Enterprise API:**
- `api/enterprise/app/dotsocr_service.py`

**Training:**
- `training/train_simple.py`
- `training/train_dotsocr.py`

**Tests:**
- `api/simple/test_ocr_api.py`
- `training/test_training.py`

**Documentation:**
- `docs/ARCHITECTURE.md`
- `ULTIMATE_README.md`
- `training/README_model_training.md`

---

## 📈 Growth Path

### Current Structure (v1.0)
```
71 files
13,152 lines
~400KB code
```

### Future Additions (Planned)
```
docs/
├── API_GUIDE.md           # Detailed API docs
├── TRAINING_GUIDE.md      # Training best practices
├── DEPLOYMENT_GUIDE.md    # Production deployment
├── CONTRIBUTING.md        # Contribution guidelines
└── CHANGELOG.md           # Version history

tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
└── e2e/                   # End-to-end tests

examples/
├── notebooks/             # Jupyter notebooks
├── scripts/               # Example scripts
└── datasets/              # Sample datasets
```

---

## 🛠️ Maintenance

### Keeping Organized

1. **New features** → appropriate subdirectory
2. **Documentation** → `docs/`
3. **Tests** → alongside code or in `tests/`
4. **Examples** → `examples/`
5. **Tools** → `tools/`

### File Naming Conventions

- **Python**: `snake_case.py`
- **Markdown**: `UPPERCASE.md` or `Title_Case.md`
- **Shell**: `kebab-case.sh` or `snake_case.sh`
- **Config**: `lowercase.yaml` or `lowercase.json`

---

**Last Updated**: October 2025  
**Version**: Ultimate v1.0


