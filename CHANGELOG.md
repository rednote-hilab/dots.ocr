# Changelog

All notable changes to ULTIMATE dots.ocr will be documented in this file.

## [Unreleased] - 2025-10-27

### Added 🎉
- **Gradio Interactive Demo** (`demo/demo_simple.py`)
  - Web-based UI for document processing
  - Real-time GPU memory monitoring with detailed metrics
  - Timestamp logging for all operations
  - Configurable GPU memory limits (default: 80%)
  - Visual layout analysis output
  - Process logs with performance metrics

- **GPU Memory Management**
  - Automatic memory limit configuration (80% default for 8GB cards)
  - Peak memory tracking during inference
  - Memory usage statistics (before/after processing)
  - CUDA memory fragmentation handling with `expandable_segments`

- **Enhanced Logging System**
  - Timestamp-based log files (`logs/demo_YYYYMMDD_HHMMSS.log`)
  - Real-time process monitoring
  - GPU memory tracking per request
  - Processing time metrics

- **Setup Scripts**
  - `scripts/setup_local.sh` - Automated local environment setup with CUDA detection
  - `scripts/start_demo.sh` - Demo launcher with timestamp logging
  - `scripts/setup_conda.sh` - Conda environment setup alternative

- **Documentation Updates**
  - Added Gradio Demo section to README
  - Updated DEPLOYMENT_LOCAL.md with GPU configuration guide
  - Added GPU memory requirements and optimization tips
  - Enhanced Quick Start guide with demo option

### Changed 🔄
- **Model Loading**
  - Implemented proper `video_processor` initialization for Qwen2_5_VL compatibility
  - Fixed `DotsVLProcessor` initialization to accept `video_processor` parameter
  - Updated both cached and source `configuration_dots.py` files

- **Demo Configuration**
  - Set default network binding to `0.0.0.0` for remote access
  - Increased GPU memory limit from 50% to 80% for stable inference
  - Added automatic `temp/` directory creation

- **.gitignore**
  - Added log file patterns (keeping recent logs only)
  - Added model weight file patterns (*.bin, *.safetensors, etc.)
  - Added training artifact directories
  - Added system files (nohup.out, etc.)

### Fixed 🐛
- **GPU Out of Memory Issues**
  - Fixed memory allocation by setting limits before model loading
  - Added `PYTORCH_CUDA_ALLOC_CONF` environment variable configuration
  - Implemented proper CUDA cache clearing

- **Module Import Errors**
  - Fixed `video_processor` TypeError in `DotsVLProcessor`
  - Resolved Python module caching issues with `__pycache__` cleanup
  - Fixed PYTHONPATH configuration in launch scripts

- **File System Errors**
  - Fixed `FileNotFoundError` for temp directory
  - Added automatic directory creation where needed

### Removed 🗑️
- Cleaned up old/duplicate log files
- Removed Python cache files (`__pycache__`, `*.pyc`)
- Removed temporary processing directories

### Technical Details 🔧
- **GPU Memory Profile**:
  - Model loading: ~5.5-5.7GB VRAM
  - Peak inference: ~6.0-6.1GB VRAM
  - Recommended: 8GB+ VRAM
  - Processing time: ~40s per document (varies by complexity)

- **Supported Hardware**:
  - NVIDIA RTX 4060 (8GB) - Tested ✅
  - NVIDIA RTX 3060+ - Supported
  - NVIDIA RTX 2080+ - Supported (with float32 fallback)

### Performance Metrics 📊
- Initial model load: ~9-10 seconds
- Average document processing: 40 seconds
- GPU memory overhead: +0.01-0.1GB per request
- Memory leak: None detected

---

## Integration History

This ultimate fork combines features from:
- **[rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr)** - Core model
- **[am009/dots.ocr](https://github.com/am009/dots.ocr)** - Simple API + Docker
- **[AKCqhzdy/dots.ocr](https://github.com/AKCqhzdy/dots.ocr)** - Enterprise API
- **[wjbmattingly/dots.ocr](https://github.com/wjbmattingly/dots.ocr)** - Training suite

---

**Date Format**: YYYY-MM-DD  
**Version**: Based on semantic versioning (when released)


