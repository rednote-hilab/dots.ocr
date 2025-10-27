# 🚀 QUICKSTART - dots.ocr-ultimate

Быстрый старт за 5 минут!

## ⚡ Автоматическая установка

```bash
# 1. Клонировать (если еще не клонировали)
git clone https://github.com/ansidenko/dots.ocr-ultimate.git
cd dots.ocr-ultimate

# 2. Автоматическая настройка
chmod +x scripts/setup_local.sh
./scripts/setup_local.sh

# 3. Запустить API
./scripts/start_simple_api.sh
```

**Готово!** API работает на `http://localhost:5000`

## 🧪 Тестирование

```bash
# В другом терминале
./scripts/test_api.sh
```

## 📖 Использование из Python

```python
from api.client import DotsOCRClient

client = DotsOCRClient("http://localhost:5000")
result = client.ocr_file("document.pdf", detect_stamps=True)

print(f"Текст: {result['text'][:200]}...")
print(f"Печатей найдено: {len(result['stamps'])}")
```

## 📚 Полная документация

- 📋 [Архитектура](docs/ARCHITECTURE.md)
- 🚀 [Локальное развертывание](docs/DEPLOYMENT_LOCAL.md)
- 📁 [Структура проекта](docs/PROJECT_STRUCTURE.md)
- 📖 [Главный README](ULTIMATE_README.md)

## 🔧 Порты

| Сервис | Порт | Описание |
|--------|------|----------|
| Simple API | 5000 | Flask REST API |
| Enterprise API | 8000 | FastAPI Production |
| vLLM | 8001 | Inference server |

## ❓ Проблемы?

См. [Troubleshooting](docs/DEPLOYMENT_LOCAL.md#troubleshooting)
