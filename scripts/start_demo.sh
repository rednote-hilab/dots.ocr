#!/bin/bash
# Start Gradio Demo with timestamp logging

cd /srv/dots.ocr/dots.ocr-ultimate

source venv/bin/activate

export PYTHONPATH=/srv/dots.ocr/dots.ocr-ultimate:$PYTHONPATH

# Создаем директорию для логов если её нет
mkdir -p logs

# Timestamp функция
log_with_timestamp() {
    while IFS= read -r line; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
    done
}

echo "═══════════════════════════════════════════════════════"
echo "🚀 Starting dots.ocr Gradio Demo at $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════"
echo ""

# Запускаем с логированием через timestamp
python -u demo/demo_simple.py 7860 2>&1 | log_with_timestamp | tee logs/demo_$(date '+%Y%m%d_%H%M%S').log

