# 🚀 Local Deployment Guide - dots.ocr-ultimate

Полное руководство по локальному развертыванию для интеграции с другими проектами.

---

## 📋 Table of Contents
- [Системные требования](#системные-требования)
- [Варианты развертывания](#варианты-развертывания)
- [Quick Start (5 минут)](#quick-start-5-минут)
- [Production Setup](#production-setup)
- [Интеграция с DeepSeekOCR](#интеграция-с-deepseekocr)
- [API для внешних проектов](#api-для-внешних-проектов)
- [Troubleshooting](#troubleshooting)

---

## 💻 Системные требования

### Минимальные (Simple API)
```
CPU: 4+ cores
RAM: 16GB
GPU: NVIDIA RTX 2080 (8GB VRAM) или выше
Disk: 50GB свободного места
OS: Ubuntu 20.04+ / WSL2
```

### Рекомендуемые (Enterprise API)
```
CPU: 8+ cores
RAM: 32GB
GPU: NVIDIA RTX 3090 (24GB VRAM) или A100
Disk: 100GB SSD
OS: Ubuntu 22.04
```

### Программное обеспечение
```bash
# Обязательно
- Python 3.10+
- CUDA 11.8+
- Docker 24.0+ (опционально)
- Git

# Для Enterprise API
- PostgreSQL 15+
- Redis 7+
```

---

## 🎯 Варианты развертывания

### Вариант 1: Simple API (Flask) 🔧
**Лучший для**: Быстрая интеграция, один GPU, простые задачи

- **Порт**: 5000
- **Протокол**: HTTP
- **Endpoint**: `/ocr`
- **Время запуска**: ~1 минута
- **Память**: 8-12GB VRAM

### Вариант 2: Enterprise API (FastAPI) 🏢
**Лучший для**: Production, высокие нагрузки, масштабирование

- **Порт**: 8000 (настраивается)
- **Протокол**: HTTP
- **Endpoints**: `/parse`, `/token_usage`, `/status`
- **Время запуска**: ~2 минуты
- **Память**: 12-24GB VRAM

### Вариант 3: Gradio Demo 🎨
**Лучший для**: Тестирование, визуализация, демонстрация

- **Порт**: 7860
- **Протокол**: HTTP
- **Интерфейс**: Web UI (Gradio)
- **Время запуска**: ~10 секунд
- **Память**: 5.5-6GB VRAM (с ограничением 80%)
- **Мониторинг**: Real-time GPU tracking с timestamp

### Вариант 4: Docker 🐳
**Лучший для**: Изолированная среда, CI/CD, быстрое развертывание

- **Порт**: 5000
- **Протокол**: HTTP
- **Готов за**: 30 секунд
- **Память**: Зависит от конфигурации

---

## ⚡ Quick Start (5 минут)

### Шаг 1: Установка зависимостей

```bash
cd /srv/dots.ocr/dots.ocr-ultimate

# Создать виртуальное окружение
python3.10 -m venv venv
source venv/bin/activate

# Установить PyTorch с CUDA
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu118

# Установить все зависимости
pip install -r requirements-unified.txt
```

### Шаг 2: Скачать модель

```bash
# Вариант A: HuggingFace (рекомендуется)
python3 tools/download_model.py

# Вариант B: ModelScope (для Китая)
python3 tools/download_model.py --type modelscope

# Модель будет в: ./weights/DotsOCR/
```

### Шаг 3: Выбрать способ запуска

#### Вариант A: Gradio Demo (рекомендуется для начала)

```bash
# Запуск интерактивного демо
bash scripts/start_demo.sh

# Вывод:
# [2025-10-27 22:52:05] ✓ GPU memory limit set to 80% (~6.4GB)
# [2025-10-27 22:52:14] ✓ Model loaded (HuggingFace mode)
# [2025-10-27 22:52:14] ✓ Current allocated: 5.66GB
# [2025-10-27 22:52:15] * Running on local URL:  http://0.0.0.0:7860

# Откройте в браузере: http://localhost:7860
# Или по IP: http://192.168.1.115:7860
```

**Мониторинг:**
- Логи с timestamp: `logs/demo_YYYYMMDD_HHMMSS.log`
- GPU память отображается в реальном времени
- Время обработки и использование памяти для каждого документа

#### Вариант B: Simple API

```bash
# Запуск REST API
python api/simple/ocr_api_server.py

# Вывод:
# * Running on http://0.0.0.0:5000
# * Model loaded: dots-ocr
# * GPU: NVIDIA RTX 4060 (bfloat16)
```

### Шаг 4: Проверить работу

```bash
# Проверка здоровья
curl http://localhost:5000/health

# Тестовый запрос
curl -X POST http://localhost:5000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "demo/demo_image1.jpg",
    "image_format": "path",
    "prompt_type": "prompt_layout_all_en"
  }'
```

✅ **Готово!** API работает на порту 5000

---

## 🏢 Production Setup

### Enterprise API с PostgreSQL + Redis

#### 1. Установить зависимости

```bash
# PostgreSQL
sudo apt update
sudo apt install postgresql-15 postgresql-contrib

# Redis
sudo apt install redis-server

# Python зависимости
pip install -r requirements-unified.txt
```

#### 2. Настроить PostgreSQL

```bash
# Создать базу данных
sudo -u postgres psql

postgres=# CREATE DATABASE dotsocr;
postgres=# CREATE USER dotsocr_user WITH PASSWORD 'your_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE dotsocr TO dotsocr_user;
postgres=# \q

# Включить pgvector (для embeddings)
sudo -u postgres psql -d dotsocr -c "CREATE EXTENSION vector;"
```

#### 3. Настроить Redis

```bash
# Запустить Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Проверить
redis-cli ping
# PONG
```

#### 4. Создать .env файл

```bash
# Создать /srv/dots.ocr/dots.ocr-ultimate/.env

cat > .env << 'EOF'
# Database
POSTGRES_URL_NO_SSL_DEV=postgresql://dotsocr_user:your_password@localhost:5432/dotsocr

# API Keys
API_KEY=sk-your-openai-key-here

# OCR Model (vLLM inference server)
OCR_INFERENCE_HOST=localhost
OCR_INFERENCE_PORT=8001

# InternVL Model (для описания картинок)
INTERN_VL_HOST=localhost
INTERN_VL_PORT=8002

# Object Storage (опционально)
# OSS_ENDPOINT=https://oss.example.com
# OSS_ACCESS_KEY_ID=your-key
# OSS_ACCESS_KEY_SECRET=your-secret

# Workers
NUM_WORKERS=4
CONCURRENT_OCR_TASK_LIMIT=2
CONCURRENT_DESCRIBE_PICTURE_TASK_LIMIT=1

# Timeouts
API_TIMEOUT=300

# PDF Settings
DPI=200
TASK_RETRY_COUNT=3
EOF
```

#### 5. Запустить vLLM inference server

```bash
# Terminal 1: vLLM для dots.ocr
vllm serve ./weights/DotsOCR \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --port 8001

# Модель загружается ~30 секунд
# Слушает на: http://localhost:8001
```

#### 6. Запустить Enterprise API

```bash
# Terminal 2: Enterprise API
cd /srv/dots.ocr/dots.ocr-ultimate
source venv/bin/activate
python api/enterprise/app/dotsocr_service.py

# API запускается на порту 8000
# Endpoints:
#   POST http://localhost:8000/parse
#   GET  http://localhost:8000/token_usage
#   GET  http://localhost:8000/status
```

---

## 🔄 Интеграция с DeepSeekOCR

Если рядом есть DeepSeekOCR, можно использовать его как альтернативную модель.

### Архитектура

```
Ваш проект
    ↓ HTTP Request
┌─────────────────────┐
│  dots.ocr API       │ (Port 5000/8000)
│  (Simple/Enterprise)│
└──────┬──────────────┘
       │
       ├─→ dots.ocr model      (./weights/DotsOCR)
       │
       └─→ DeepSeekOCR model   (../DeepSeekOCR)
```

### Вариант 1: Запустить DeepSeekOCR как отдельный сервис

```bash
# Terminal 1: DeepSeekOCR на порту 8003
cd ../DeepSeekOCR
python api_server.py --port 8003

# Terminal 2: dots.ocr на порту 5000
cd /srv/dots.ocr/dots.ocr-ultimate
python api/simple/ocr_api_server.py
```

### Вариант 2: Создать роутер API

Создадим умный роутер, который выбирает модель:

```bash
# Создать /srv/dots.ocr/dots.ocr-ultimate/api/router_api.py
```

```python
# api/router_api.py
from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Конфигурация моделей
MODELS = {
    "dots-ocr": {
        "url": "http://localhost:5000/ocr",
        "best_for": ["layout", "tables", "formulas", "multilingual"]
    },
    "deepseek-ocr": {
        "url": "http://localhost:8003/ocr",
        "best_for": ["text", "stamps", "seals", "handwriting"]
    }
}

@app.route('/ocr', methods=['POST'])
def route_ocr():
    data = request.json
    
    # Выбор модели на основе задачи
    task_type = data.get('task_type', 'general')
    model = data.get('model', 'auto')
    
    if model == 'auto':
        # Автовыбор модели
        if task_type in ['stamps', 'seals', 'handwriting']:
            model = 'deepseek-ocr'
        else:
            model = 'dots-ocr'
    
    # Отправить запрос в выбранную модель
    target_url = MODELS[model]['url']
    response = requests.post(target_url, json=data)
    
    return jsonify({
        'model_used': model,
        'result': response.json()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
```

Запуск:
```bash
# Terminal 3: Router API на порту 9000
python api/router_api.py

# Использование
curl -X POST http://localhost:9000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "image": "document.jpg",
    "image_format": "path",
    "task_type": "stamps",
    "model": "auto"
  }'
```

---

## 🔌 API для внешних проектов

### Формат запроса для вашего проекта

```python
# Пример клиента для вашего проекта
import requests
import base64
import json

class DotsOCRClient:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
    
    def ocr_file(self, file_path, detect_stamps=True):
        """
        Распознать документ и найти печати/штампы
        
        Returns:
            {
                'layout': [...],  # Все элементы с координатами
                'text': '...',    # Весь текст
                'stamps': [...],  # Найденные печати
                'tables': [...],  # Таблицы
                'formulas': [...] # Формулы
            }
        """
        # Прочитать файл
        with open(file_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        # Отправить запрос
        response = requests.post(
            f"{self.api_url}/ocr",
            json={
                "image": image_b64,
                "image_format": "base64",
                "prompt_type": "prompt_layout_all_en"
            }
        )
        
        result = response.json()
        
        # Парсить результат
        parsed = self._parse_response(result)
        
        # Найти печати (если нужно)
        if detect_stamps:
            parsed['stamps'] = self._detect_stamps(parsed['layout'])
        
        return parsed
    
    def _parse_response(self, response):
        """Парсинг JSON ответа"""
        data = json.loads(response['response'])
        
        layout_elements = []
        text_parts = []
        tables = []
        formulas = []
        
        for element in data:
            bbox = element.get('bbox', [])
            category = element.get('category', '')
            text = element.get('text', '')
            
            layout_elements.append({
                'bbox': bbox,  # [x1, y1, x2, y2]
                'category': category,
                'text': text
            })
            
            if category == 'Table':
                tables.append({
                    'bbox': bbox,
                    'html': text
                })
            elif category == 'Formula':
                formulas.append({
                    'bbox': bbox,
                    'latex': text
                })
            elif category not in ['Picture', 'Page-header', 'Page-footer']:
                text_parts.append(text)
        
        return {
            'layout': layout_elements,
            'text': '\n'.join(text_parts),
            'tables': tables,
            'formulas': formulas
        }
    
    def _detect_stamps(self, layout_elements):
        """
        Найти печати/штампы
        
        Эвристика:
        - Круглые/овальные области
        - Красный/синий цвет (нужна дополнительная обработка)
        - Содержат специфичные слова (ПЕЧАТЬ, УТВЕРЖДЕНО и т.д.)
        """
        stamps = []
        
        for element in layout_elements:
            text = element.get('text', '').upper()
            
            # Ключевые слова для печатей
            stamp_keywords = [
                'ПЕЧАТЬ', 'УТВЕРЖДЕНО', 'СОГЛАСОВАНО',
                'STAMP', 'APPROVED', 'М.П.', 'SEAL'
            ]
            
            # Проверить текст
            is_stamp = any(keyword in text for keyword in stamp_keywords)
            
            # Проверить форму (соотношение сторон близко к 1:1)
            bbox = element['bbox']
            if len(bbox) == 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 0
                
                # Круглая/квадратная форма (0.8 < ratio < 1.2)
                if 0.8 < aspect_ratio < 1.2 and (is_stamp or width < 200):
                    stamps.append({
                        'bbox': bbox,
                        'text': element['text'],
                        'confidence': 'high' if is_stamp else 'medium'
                    })
        
        return stamps

# Использование
if __name__ == '__main__':
    client = DotsOCRClient("http://localhost:5000")
    
    result = client.ocr_file("document.pdf")
    
    print(f"Текст: {result['text'][:200]}...")
    print(f"Найдено печатей: {len(result['stamps'])}")
    print(f"Найдено таблиц: {len(result['tables'])}")
    
    # Вывести координаты печатей
    for i, stamp in enumerate(result['stamps']):
        print(f"Печать {i+1}: {stamp['bbox']}, текст: {stamp['text'][:50]}")
```

### Продвинутое определение печатей

Для более точного определения печатей, добавим обработку изображений:

```python
# api/stamp_detector.py
import cv2
import numpy as np
from PIL import Image

class StampDetector:
    """Детектор печатей/штампов на документах"""
    
    def __init__(self):
        pass
    
    def detect_stamps_visual(self, image_path):
        """
        Визуальное определение печатей по цвету и форме
        
        Returns:
            List[dict]: [{bbox, color, confidence}, ...]
        """
        # Загрузить изображение
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        stamps = []
        
        # Поиск красных областей (красные печати)
        red_stamps = self._find_colored_regions(img_hsv, 'red')
        stamps.extend(red_stamps)
        
        # Поиск синих областей (синие печати)
        blue_stamps = self._find_colored_regions(img_hsv, 'blue')
        stamps.extend(blue_stamps)
        
        # Фильтрация по форме
        stamps = self._filter_by_shape(stamps, img)
        
        return stamps
    
    def _find_colored_regions(self, img_hsv, color):
        """Найти области определенного цвета"""
        
        if color == 'red':
            # Красный цвет в HSV (два диапазона)
            lower1 = np.array([0, 100, 100])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([160, 100, 100])
            upper2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(img_hsv, lower1, upper1)
            mask2 = cv2.inRange(img_hsv, lower2, upper2)
            mask = mask1 | mask2
            
        elif color == 'blue':
            # Синий цвет в HSV
            lower = np.array([100, 100, 100])
            upper = np.array([130, 255, 255])
            mask = cv2.inRange(img_hsv, lower, upper)
        
        # Найти контуры
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Минимальный размер печати (30x30 пикселей)
            if w > 30 and h > 30:
                regions.append({
                    'bbox': [x, y, x+w, y+h],
                    'color': color,
                    'area': w * h
                })
        
        return regions
    
    def _filter_by_shape(self, stamps, img):
        """Фильтрация по форме (круглые/овальные)"""
        filtered = []
        
        for stamp in stamps:
            x1, y1, x2, y2 = stamp['bbox']
            w = x2 - x1
            h = y2 - y1
            
            # Соотношение сторон
            aspect_ratio = w / h if h > 0 else 0
            
            # Печати обычно круглые/квадратные (0.7 < ratio < 1.3)
            if 0.7 < aspect_ratio < 1.3:
                stamp['confidence'] = 'high'
                filtered.append(stamp)
            # Или прямоугольные штампы (2:1 или 3:1)
            elif 1.5 < aspect_ratio < 3.5 or 0.3 < aspect_ratio < 0.7:
                stamp['confidence'] = 'medium'
                filtered.append(stamp)
        
        return filtered

# Интеграция с DotsOCRClient
class EnhancedDotsOCRClient(DotsOCRClient):
    def __init__(self, api_url="http://localhost:5000"):
        super().__init__(api_url)
        self.stamp_detector = StampDetector()
    
    def ocr_file(self, file_path, detect_stamps=True):
        # Получить результат OCR
        result = super().ocr_file(file_path, detect_stamps=False)
        
        if detect_stamps:
            # Комбинировать текстовые и визуальные методы
            text_stamps = self._detect_stamps(result['layout'])
            visual_stamps = self.stamp_detector.detect_stamps_visual(file_path)
            
            # Объединить результаты
            result['stamps'] = self._merge_stamp_detections(
                text_stamps, visual_stamps
            )
        
        return result
    
    def _merge_stamp_detections(self, text_stamps, visual_stamps):
        """Объединить результаты текстового и визуального детектирования"""
        merged = []
        
        # Добавить визуальные печати
        for vstamp in visual_stamps:
            merged.append({
                'bbox': vstamp['bbox'],
                'color': vstamp.get('color'),
                'confidence': vstamp['confidence'],
                'detection_method': 'visual',
                'text': ''
            })
        
        # Добавить текстовые печати
        for tstamp in text_stamps:
            # Проверить, не пересекается ли с визуальными
            overlaps = False
            for vstamp in visual_stamps:
                if self._bbox_overlap(tstamp['bbox'], vstamp['bbox']) > 0.5:
                    overlaps = True
                    # Обновить текст визуальной печати
                    for m in merged:
                        if m['bbox'] == vstamp['bbox']:
                            m['text'] = tstamp['text']
                            m['confidence'] = 'very_high'
                    break
            
            if not overlaps:
                merged.append({
                    'bbox': tstamp['bbox'],
                    'text': tstamp['text'],
                    'confidence': tstamp['confidence'],
                    'detection_method': 'text'
                })
        
        return merged
    
    def _bbox_overlap(self, bbox1, bbox2):
        """Вычислить IoU (Intersection over Union) двух bbox"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Пересечение
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Площади
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
```

---

## 🎯 Полный пример для вашего проекта

```python
# your_project/ocr_integration.py

from enhanced_dots_ocr_client import EnhancedDotsOCRClient
import json

def process_document(file_path, output_path=None):
    """
    Обработать документ через dots.ocr
    
    Args:
        file_path: Путь к файлу (PDF/Image)
        output_path: Путь для сохранения результата (опционально)
    
    Returns:
        dict: Полный результат распознавания
    """
    # Создать клиент
    client = EnhancedDotsOCRClient("http://localhost:5000")
    
    # Распознать
    print(f"Обработка: {file_path}")
    result = client.ocr_file(file_path, detect_stamps=True)
    
    # Вывести статистику
    print(f"✓ Распознано символов: {len(result['text'])}")
    print(f"✓ Найдено элементов: {len(result['layout'])}")
    print(f"✓ Найдено печатей: {len(result['stamps'])}")
    print(f"✓ Найдено таблиц: {len(result['tables'])}")
    print(f"✓ Найдено формул: {len(result['formulas'])}")
    
    # Сохранить результат
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ Результат сохранен: {output_path}")
    
    return result

if __name__ == '__main__':
    # Пример использования
    result = process_document(
        file_path="../documents/contract.pdf",
        output_path="../results/contract_ocr.json"
    )
    
    # Получить только печати для проверки подписи
    stamps = result['stamps']
    for i, stamp in enumerate(stamps):
        x1, y1, x2, y2 = stamp['bbox']
        print(f"Печать {i+1}: ({x1},{y1})-({x2},{y2})")
        print(f"  Текст: {stamp.get('text', 'Н/Д')}")
        print(f"  Цвет: {stamp.get('color', 'Н/Д')}")
        print(f"  Уверенность: {stamp['confidence']}")
```

---

## 📊 Таблица портов

| Сервис | Порт | Протокол | Описание |
|--------|------|----------|----------|
| **Simple API** | 5000 | HTTP | Flask REST API |
| **Enterprise API** | 8000 | HTTP | FastAPI с очередями |
| **vLLM (dots.ocr)** | 8001 | HTTP | Inference server |
| **InternVL** | 8002 | HTTP | Image descriptions |
| **DeepSeekOCR** | 8003 | HTTP | Альтернативная модель |
| **Router API** | 9000 | HTTP | Умный роутер |
| **PostgreSQL** | 5432 | TCP | База данных |
| **Redis** | 6379 | TCP | Кэш |

---

## 🔧 Troubleshooting

### Проблема: Out of Memory (OOM)

```bash
# Решение 1: Уменьшить batch size
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Решение 2: Использовать CPU (медленно)
export TORCH_DTYPE=float32
export CUDA_VISIBLE_DEVICES=-1

# Решение 3: Использовать quantization
python api/simple/ocr_api_server.py --quantization int8
```

### Проблема: Model not found

```bash
# Переска�ать модель
python tools/download_model.py --force

# Проверить путь
ls -la ./weights/DotsOCR/
```

### Проблема: PostgreSQL connection error

```bash
# Проверить статус
sudo systemctl status postgresql

# Перезапустить
sudo systemctl restart postgresql

# Проверить подключение
psql -h localhost -U dotsocr_user -d dotsocr
```

### Проблема: Медленная работа

```bash
# Проверить GPU использование
nvidia-smi

# Увеличить workers
export NUM_WORKERS=8

# Включить Redis кэш (Enterprise)
export REDIS_URL=redis://localhost:6379
```

---

## 📝 Следующие шаги

1. ✅ Развернуть локально
2. ✅ Протестировать API
3. ✅ Интегрировать с вашим проектом
4. ✅ Настроить детектор печатей
5. ⬜ Оптимизировать производительность
6. ⬜ Добавить мониторинг

---

**Last Updated**: October 2025  
**Version**: v1.0

