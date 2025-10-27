# 🏗️ ARCHITECTURE - dots.ocr-ultimate

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [API Layer](#api-layer)
- [Training Infrastructure](#training-infrastructure)
- [Deployment](#deployment)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)

---

## 🎯 Overview

**dots.ocr-ultimate** is a unified fork combining the best features from multiple community forks of the original dots.ocr project. It provides a complete ecosystem for document OCR with multiple deployment options and training capabilities.

### Design Principles

1. **Modularity**: Each component can be used independently
2. **Flexibility**: Multiple API options for different use cases
3. **Scalability**: From single GPU to enterprise deployments
4. **Extensibility**: Easy to add new features and integrate updates

---

## 📁 Project Structure

```
dots.ocr-ultimate/
│
├── 📦 Core Model Layer
│   └── dots_ocr/              # Original dots.ocr implementation
│       ├── model/             # Model implementations
│       │   ├── inference.py   # Inference logic
│       │   └── layout_service.py  # PP-DocLayout integration (zihao)
│       ├── parser.py          # Main parser interface
│       └── utils/
│           ├── pdf_extractor.py   # Text extraction (zihao)
│           └── demo_utils/        # Utilities
│
├── 🌐 API Layer
│   ├── simple/                # Flask REST API (am009)
│   │   ├── ocr_api_server.py  # Simple API server
│   │   ├── forward_exec.py    # Execution wrapper
│   │   ├── test_ocr_api.py    # API tests
│   │   └── API_Documentation*.md
│   │
│   └── enterprise/            # FastAPI Enterprise (akcqhzdy)
│       └── app/
│           ├── dotsocr_service.py  # Main service
│           └── utils/
│               ├── configs.py         # Configuration
│               ├── executor/          # Task execution
│               │   ├── job_executor_pool.py
│               │   ├── task_executor_pool.py
│               │   └── ocr_task.py
│               ├── pg_vector/         # PostgreSQL integration
│               │   ├── pg_vector.py
│               │   └── table.py
│               ├── storage.py         # OSS storage
│               ├── tracing.py         # OpenTelemetry
│               ├── redis.py           # Redis cache
│               └── hash.py            # MD5 utilities
│
├── 🎓 Training Infrastructure
│   └── training/              # Training suite (wjbmattingly)
│       ├── train_simple.py          # Beginner-friendly training
│       ├── train_dotsocr.py         # Advanced training + LoRA
│       ├── create_training_data.py  # Data preparation
│       ├── example_usage.py         # Examples
│       ├── run_training.sh          # Automation scripts
│       ├── test_training*.py        # Tests
│       ├── config_training.yaml     # Config templates
│       └── README_*.md              # Documentation
│
├── 🐳 Deployment
│   └── deployment/
│       └── docker/            # Docker configs (am009)
│           ├── Dockerfile
│           ├── start.sh
│           ├── daemon-start.sh
│           └── stop.sh
│
├── 📚 Documentation
│   ├── docs/
│   │   ├── ARCHITECTURE.md    # This file
│   │   ├── API_GUIDE.md       # API usage guide
│   │   └── TRAINING_GUIDE.md  # Training guide
│   │
│   ├── ULTIMATE_README.md     # Main README
│   ├── requirements-unified.txt
│   └── PUSH_INSTRUCTIONS.md
│
├── 🛠️ Tools & Utilities
│   ├── tools/                 # Helper scripts
│   ├── demo/                  # Demo applications
│   └── assets/                # Images and resources
│
└── 📋 Configuration
    ├── requirements.txt       # Original requirements
    ├── requirements-unified.txt
    ├── setup.py
    └── .gitignore
```

---

## 💎 Core Components

### 1. Model Layer (`dots_ocr/`)

**Purpose**: Core OCR and layout parsing functionality

#### Key Files:

**`model/inference.py`**
- VLM inference using transformers
- Qwen2.5-VL based implementation
- Multi-modal input processing (images + text prompts)

**`model/layout_service.py`** *(from zihao branch)*
- PP-DocLayout-L integration
- Layout detection for structured PDFs
- Faster processing for text-based documents

**`parser.py`**
- Main interface for document parsing
- Handles images and PDFs
- Coordinates between OCR and layout detection

**`utils/pdf_extractor.py`** *(from zihao branch)*
- PyMuPDF (fitz) based text extraction
- Detects structured vs scanned PDFs
- Direct text extraction when available

#### Data Flow:

```
Input Document
    ↓
Parser (parser.py)
    ↓
├─ Structured PDF? → layout_service.py + pdf_extractor.py
│                     (Fast path - no OCR needed)
└─ Scanned/Image? → inference.py (VLM OCR)
                     (Full OCR inference)
    ↓
Output (JSON/Markdown)
```

---

## 🌐 API Layer

### Option 1: Simple API (`api/simple/`)

**Source**: am009 fork  
**Technology**: Flask + Transformers  
**Best for**: Quick prototyping, single GPU, simple deployments

#### Architecture:

```
Client Request
    ↓
ocr_api_server.py (Flask)
    ↓
forward_exec.py (Execution wrapper)
    ↓
Transformers Model (dots_ocr)
    ↓
Response (JSON/Streaming)
```

#### Key Features:
- **Auto GPU Detection**: Automatically selects float32/float16/bfloat16
- **Multiple Input Formats**: Path, URL, Base64
- **Streaming Support**: For large documents
- **Processing Lock**: Single request at a time
- **Temporary File Management**: Auto-cleanup

#### Endpoints:
- `GET /health` - Health check
- `POST /ocr` - OCR processing

#### Configuration:
```python
# Environment variables
TORCH_DTYPE=auto  # or float32, float16, bfloat16

# Supports RTX 20xx (Turing) and newer
```

---

### Option 2: Enterprise API (`api/enterprise/`)

**Source**: akcqhzdy fork  
**Technology**: FastAPI + PostgreSQL + Redis + OpenTelemetry  
**Best for**: Production, high-load, enterprise deployments

#### Architecture:

```
Client Request
    ↓
FastAPI (dotsocr_service.py)
    ↓
Job Queue (job_executor_pool.py)
    ↓
├─ Task Executor (task_executor_pool.py)
│  ├─ OCR Task (ocr_task.py)
│  └─ Picture Description Task
│      ↓
├─ PostgreSQL (pg_vector/)
│  └─ Vector embeddings + metadata
│      ↓
├─ Redis (redis.py)
│  └─ Caching layer
│      ↓
├─ OSS Storage (storage.py)
│  └─ Distributed file storage
│      ↓
└─ OpenTelemetry (tracing.py)
   └─ Distributed tracing

Response + Metrics
```

#### Components:

**1. Job Executor Pool**
```python
# Manages concurrent jobs
- Max concurrent jobs: configurable
- Queue management
- Retry logic with exponential backoff
- Job status tracking
```

**2. Task Executor Pool**
```python
# Manages OCR inference tasks
- Concurrent task limit
- Async execution
- Timeout management
- Performance metrics
```

**3. PostgreSQL + PGVector**
```python
# Document storage and search
- OCR results storage
- Vector embeddings
- MD5-based deduplication
- Query by similarity
```

**4. Redis Cache**
```python
# Performance optimization
- Result caching
- Session management
- Rate limiting support
```

**5. OpenTelemetry Tracing**
```python
# Monitoring and debugging
- Request tracing
- Performance profiling
- Error tracking
- SQLAlchemy instrumentation
- OpenAI API instrumentation
```

#### Endpoints:
- `POST /parse` - Parse documents
- `GET /token_usage` - Token statistics
- `GET /status` - Job status
- `GET /health` - Health check

#### Configuration:
```bash
# Required
POSTGRES_URL_NO_SSL_DEV=postgresql://user:pass@host/db
API_KEY=sk-your-openai-key

# Optional
OSS_ENDPOINT=https://oss.example.com
OSS_ACCESS_KEY_ID=your-key
OSS_ACCESS_KEY_SECRET=your-secret
OCR_INFERENCE_HOST=localhost
OCR_INFERENCE_PORT=8000
INTERN_VL_HOST=localhost
INTERN_VL_PORT=8001
NUM_WORKERS=4
CONCURRENT_OCR_TASK_LIMIT=2
CONCURRENT_DESCRIBE_PICTURE_TASK_LIMIT=1
API_TIMEOUT=300
DPI=200
TASK_RETRY_COUNT=3
```

---

## 🎓 Training Infrastructure

**Source**: wjbmattingly fork  
**Purpose**: Fine-tune dots.ocr on custom datasets

### Components:

#### 1. Data Preparation
```
PAGEXML + JPEG Files
    ↓
create_training_data.py
    ↓
Training JSONL Format
```

#### 2. Training Scripts

**`train_simple.py`** - Beginner-friendly
```python
# Features:
- Sensible defaults
- Simple CLI interface
- Progress tracking
- Weights & Biases integration
- Automatic checkpointing
```

**`train_dotsocr.py`** - Advanced
```python
# Features:
- Full parameter control
- LoRA support (parameter-efficient)
- Vision encoder freezing
- Mixed precision training (bf16/fp16)
- Custom learning rate scheduling
- Gradient accumulation
- DeepSpeed integration (optional)
```

#### 3. Training Strategies

**Strategy 1: Full Fine-tuning**
```bash
# Best results, requires ~24GB GPU
python training/train_simple.py \
  --data training_data.jsonl \
  --epochs 3 \
  --batch_size 1 \
  --learning_rate 2e-5
```

**Strategy 2: LoRA (Memory Efficient)**
```bash
# Good results, requires ~12GB GPU
python training/train_dotsocr.py \
  --train_data training_data.jsonl \
  --lora_training \
  --lora_rank 8 \
  --learning_rate 1e-4
```

**Strategy 3: Freeze Vision Encoder**
```bash
# Fast training, text-focused
python training/train_dotsocr.py \
  --train_data training_data.jsonl \
  --freeze_vision_encoder \
  --learning_rate 5e-5
```

### Training Pipeline:

```
Data Preparation
    ↓
create_training_data.py
    ↓
Training Data (JSONL)
    ↓
train_simple.py / train_dotsocr.py
    ↓
├─ Model Loading (dots_ocr)
├─ Data Loading & Processing
├─ Training Loop
│  ├─ Forward Pass
│  ├─ Loss Calculation
│  ├─ Backward Pass
│  └─ Optimizer Step
├─ Validation (optional)
├─ Checkpointing
└─ Metrics Logging (W&B)
    ↓
Fine-tuned Model
```

---

## 🐳 Deployment

### Docker Configuration

**Source**: am009 fork  
**Purpose**: Containerized deployment

```dockerfile
# Base: NVIDIA CUDA image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python + dependencies
RUN apt-get update && apt-get install -y python3.10

# Copy application
COPY . /DotsOCR

# Install dependencies
RUN pip install -r requirements-unified.txt

# Expose API port
EXPOSE 5000

# Start service
CMD ["/DotsOCR/start.sh"]
```

### Deployment Options:

#### 1. Docker Single Container
```bash
docker run -d --runtime=nvidia --gpus=all \
  -p 5000:5000 \
  dots.ocr-ultimate:latest
```

#### 2. Docker Compose (Multi-Service)
```yaml
version: '3.8'
services:
  dots-ocr:
    image: dots.ocr-ultimate
    runtime: nvidia
    environment:
      - TORCH_DTYPE=auto
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=dotsocr
  
  redis:
    image: redis:7
```

#### 3. Kubernetes (Scalable)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dots-ocr
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dots-ocr
  template:
    spec:
      containers:
      - name: dots-ocr
        image: dots.ocr-ultimate
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## 🔄 Data Flow Diagrams

### Simple API Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ HTTP POST /ocr
     │ {image, format, prompt}
     ↓
┌─────────────────┐
│  Flask Server   │
│ (port 5000)     │
└────┬────────────┘
     │
     ├─ Health Check?
     │  └→ Return status
     │
     ├─ Image Processing
     │  ├─ Load from path/URL/base64
     │  └─ Create temp file if needed
     │
     ↓
┌──────────────────────┐
│ Transformers Model   │
│ (dots_ocr)           │
└────┬─────────────────┘
     │
     ├─ Vision Encoder
     ├─ Language Model
     └─ Generate Output
     │
     ↓
┌──────────────────┐
│ Response         │
│ - JSON/Stream    │
│ - Cleanup temps  │
└──────────────────┘
```

### Enterprise API Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ HTTP POST /parse
     ↓
┌─────────────────────┐
│ FastAPI Service     │
│ (dotsocr_service)   │
└────┬────────────────┘
     │
     ├─ Check Redis Cache
     │  └─ Hit? → Return cached
     │
     ├─ Check PostgreSQL
     │  └─ MD5 exists? → Return stored
     │
     ├─ Create Job
     │  └─ Add to job_executor_pool
     │
     ↓
┌────────────────────┐
│ Job Executor Pool  │
│ (async workers)    │
└────┬───────────────┘
     │
     ├─ Acquire processing lock
     ├─ Download from OSS (if needed)
     ├─ Create OCR tasks
     │
     ↓
┌──────────────────────┐
│ Task Executor Pool   │
│ (OCR + Description)  │
└────┬─────────────────┘
     │
     ├─ OCR Task
     │  └─ Call dots_ocr model
     │
     ├─ Picture Description
     │  └─ Call InternVL model
     │
     └─ Retry on failure
     │
     ↓
┌────────────────────┐
│ Result Processing  │
└────┬───────────────┘
     │
     ├─ Save to PostgreSQL
     ├─ Cache in Redis
     ├─ Upload to OSS
     ├─ Log to OpenTelemetry
     │
     ↓
┌──────────────┐
│   Response   │
│   + Metrics  │
└──────────────┘
```

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **ML Framework** | PyTorch | 2.0+ | Deep learning |
| **Model** | Qwen2.5-VL | 1.7B | Vision-Language Model |
| **Transformers** | HuggingFace | 4.54 | Model loading & inference |
| **Vision** | torchvision | latest | Image processing |
| **Acceleration** | flash-attn | 2.8.0 | Attention optimization |

### API Layer

| Component | Technology | Use Case |
|-----------|-----------|----------|
| **Simple API** | Flask 3.0+ | Quick prototyping |
| **Enterprise API** | FastAPI 0.100+ | Production |
| **Async Runtime** | uvicorn | ASGI server |
| **HTTP Client** | httpx | Async requests |

### Data Storage

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | PostgreSQL + PGVector | Document embeddings |
| **Cache** | Redis 5.0+ | Performance |
| **Object Storage** | S3-compatible (OSS) | Distributed files |
| **ORM** | SQLAlchemy 2.0 | Database abstraction |

### Monitoring

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Tracing** | OpenTelemetry | Distributed tracing |
| **Logging** | Loguru | Structured logging |
| **Metrics** | Custom + OTLP | Performance tracking |
| **Training** | Weights & Biases | Experiment tracking |

### Document Processing

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF** | PyMuPDF (fitz) | Text extraction |
| **Layout** | PP-DocLayout-L | Structure detection |
| **OCR** | dots.ocr (VLM) | Image-to-text |

---

## 📊 Performance Characteristics

### Simple API
- **Latency**: ~2-5 seconds per page
- **Throughput**: 1 request at a time
- **Memory**: 6-12GB VRAM (depending on dtype)
- **Scalability**: Single instance

### Enterprise API
- **Latency**: ~2-5 seconds per page (parallel)
- **Throughput**: Configurable (NUM_WORKERS)
- **Memory**: Depends on concurrent tasks
- **Scalability**: Horizontal (multiple instances)

### Structured PDF (zihao)
- **Latency**: ~0.5-1 second per page (10x faster)
- **Throughput**: Limited by layout detection
- **Memory**: Minimal VRAM usage
- **Best for**: Born-digital PDFs

---

## 🔐 Security Considerations

### API Security
- API key authentication (Enterprise)
- Rate limiting (via Redis)
- Input validation
- Temporary file cleanup

### Data Privacy
- No persistent storage of input images (Simple API)
- Optional OSS encryption
- PostgreSQL SSL support
- Audit logging

---

## 📈 Scalability Patterns

### Horizontal Scaling
```
Load Balancer
    ↓
├─ dots-ocr-1 (GPU 0)
├─ dots-ocr-2 (GPU 1)
└─ dots-ocr-3 (GPU 2)
    ↓
Shared PostgreSQL + Redis
```

### Vertical Scaling
```
Single Server
├─ Multiple GPU (CUDA_VISIBLE_DEVICES)
├─ Concurrent workers
└─ Task queues
```

---

## 🔄 Update Strategy

### Staying Synchronized

```bash
# Fetch updates from original
git fetch upstream-original master

# Fetch from forks
git fetch am009 master
git fetch akcqhzdy master
git fetch wjbmattingly master

# Review changes
git log HEAD..upstream-original/master

# Merge selectively
git cherry-pick <commit-hash>
```

---

## 📚 Further Reading

- [API Guide](API_GUIDE.md) - Detailed API documentation
- [Training Guide](TRAINING_GUIDE.md) - Training best practices
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment
- [Contributing](../CONTRIBUTING.md) - How to contribute

---

**Last Updated**: October 2025  
**Version**: Ultimate v1.0  
**Maintainer**: Community Fork


