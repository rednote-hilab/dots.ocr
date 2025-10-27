# CUDA Setup Guide

## Проверка текущей CUDA

```bash
# Проверить версию CUDA
nvcc --version

# Проверить GPU
nvidia-smi

# Проверить, видит ли PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## Три сценария установки

### 1️⃣ У вас УЖЕ ЕСТЬ нативная CUDA (рекомендуется)

**Используйте venv + pip:**

```bash
# Проверить CUDA
nvcc --version  # Должно показать версию (например, 11.8 или 12.1)

# Установить с правильной версией PyTorch
./scripts/setup_local.sh

# Или вручную:
python3 -m venv venv
source venv/bin/activate

# Для CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-unified.txt
```

**Преимущества:**
- ✅ Использует системную CUDA (без дубликатов)
- ✅ Быстрая установка
- ✅ Меньше места

---

### 2️⃣ У вас НЕТ CUDA (или хотите изолировать)

**Используйте conda:**

```bash
# Conda установит CUDA автоматически
./scripts/setup_conda.sh
```

**Преимущества:**
- ✅ Автоматическая установка CUDA
- ✅ Полная изоляция
- ✅ Нет зависимости от системы

**Недостатки:**
- ❌ Дубликат CUDA (~2GB)
- ❌ Может конфликтовать с nvidia-smi

---

### 3️⃣ CPU-only (без GPU)

```bash
python3 -m venv venv
source venv/bin/activate

# PyTorch без CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements-unified.txt
```

---

## Проверка после установки

```bash
# Активировать окружение
source venv/bin/activate  # для venv
# ИЛИ
conda activate dots-ocr-ultimate  # для conda

# Проверить PyTorch
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

**Ожидаемый вывод с GPU:**
```
PyTorch version: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 3090
```

---

## Troubleshooting

### ❌ CUDA available: False (но GPU есть)

**Причина:** Несовпадение версий CUDA и PyTorch

**Решение:**
```bash
# Узнать версию системной CUDA
nvcc --version  # Например, 11.8

# Переустановить PyTorch с правильной версией
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ❌ CUDA out of memory

**Решение:** Уменьшить batch size в конфиге
```python
# В api/simple/ocr_api_server.py
# Найти: max_batch_size = 8
# Заменить на: max_batch_size = 2
```

### ❌ RuntimeError: CUDA error: no kernel image is available

**Причина:** GPU не поддерживается этой версией PyTorch

**Решение:** Обновить PyTorch или использовать CPU
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Рекомендации

| Сценарий | Метод | Команда |
|----------|-------|---------|
| 🖥️ Есть нативная CUDA | venv + pip | `./scripts/setup_local.sh` |
| 🌐 Нет CUDA | conda | `./scripts/setup_conda.sh` |
| 💻 CPU-only | venv + pip (cpu) | Manual install |
| 🐳 Docker | venv + pip | Use Dockerfile |
| 🚀 Production | venv + pip | `./scripts/setup_local.sh` |

**Вывод:** Если у вас уже есть нативная CUDA → используйте venv + pip!
