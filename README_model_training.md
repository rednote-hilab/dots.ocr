# Model Training for dots.ocr

This directory contains complete training scripts to fine-tune the dots.ocr model on your PAGEXML + HTR data.

## 🚀 Quick Start

### 1. Install Training Dependencies
```bash
pip install -r training_requirements.txt
```

### 2. Prepare Your Training Data
First, convert your PAGEXML + JPEG data using the data preparation script:
```bash
python create_training_data.py --input_dir /path/to/data --output_file training_data.jsonl
```

### 3. Download the Base Model
```bash
python tools/download_model.py
```

### 4. Start Training
```bash
# Simple training (recommended for beginners)
python train_simple.py --data training_data.jsonl --epochs 3 --batch_size 1

# Or use the shell script
chmod +x run_training.sh
./run_training.sh
```

## 📁 Training Scripts

### 1. `train_simple.py` - Beginner-Friendly
- **Easy to use** with minimal configuration
- **Good defaults** for most use cases
- **Clear logging** and progress tracking

```bash
python train_simple.py \
    --data training_data.jsonl \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --wandb
```

### 2. `train_dotsocr.py` - Advanced Training
- **Full control** over training parameters
- **LoRA support** for memory-efficient training
- **Parameter freezing** strategies
- **Advanced evaluation** and checkpointing

```bash
python train_dotsocr.py \
    --train_data training_data.jsonl \
    --output_dir ./checkpoints \
    --num_train_epochs 3 \
    --lora_training \
    --bf16
```

## 🎯 Training Strategies

### Strategy 1: Full Fine-tuning (Best Results)
```bash
python train_simple.py \
    --data training_data.jsonl \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5
```
- **Pros**: Best performance, adapts entire model
- **Cons**: Requires more GPU memory (~24GB+)

### Strategy 2: LoRA Fine-tuning (Memory Efficient)
```bash
python train_dotsocr.py \
    --train_data training_data.jsonl \
    --lora_training \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-4
```
- **Pros**: Memory efficient (~12GB), faster training
- **Cons**: Slightly lower performance than full fine-tuning

### Strategy 3: Freeze Vision Encoder
```bash
python train_dotsocr.py \
    --train_data training_data.jsonl \
    --freeze_vision_encoder \
    --learning_rate 5e-5
```
- **Pros**: Faster training, good for text-focused tasks
- **Cons**: Won't adapt vision understanding

## 📊 Recommended Training Configurations

### Small Dataset (< 1K samples)
```bash
python train_simple.py \
    --data training_data.jsonl \
    --epochs 5 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --max_length 4096
```

### Medium Dataset (1K - 10K samples)
```bash
python train_dotsocr.py \
    --train_data training_data.jsonl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --lora_training
```

### Large Dataset (10K+ samples)
```bash
python train_dotsocr.py \
    --train_data training_data.jsonl \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --bf16
```

## 🔧 Hardware Requirements

### Minimum Requirements
- **GPU**: 12GB VRAM (RTX 3060, RTX 4070)
- **RAM**: 32GB
- **Storage**: 50GB free space

### Recommended
- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A100)
- **RAM**: 64GB
- **Storage**: 100GB SSD

### Multi-GPU Training
```bash
# Using accelerate for multi-GPU
accelerate config  # Run once to configure
accelerate launch train_dotsocr.py --train_data training_data.jsonl
```

## 📈 Monitoring Training

### Weights & Biases (Recommended)
```bash
pip install wandb
wandb login

python train_simple.py --data training_data.jsonl --wandb --run_name my-experiment
```

### TensorBoard
```bash
python train_dotsocr.py --train_data training_data.jsonl --report_to tensorboard
tensorboard --logdir ./checkpoints/runs
```

## 🎛️ Key Training Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `learning_rate` | How fast the model learns | `1e-5` to `5e-5` |
| `batch_size` | Samples per batch | `1` (limited by memory) |
| `gradient_accumulation_steps` | Effective batch size | `4-16` |
| `max_length` | Max sequence length | `4096-8192` |
| `epochs` | Training iterations | `2-5` |
| `warmup_steps` | Learning rate warmup | `100-500` |

## 🔍 Training Tips

### 1. Start Small
- Begin with a subset of your data
- Use shorter sequences (`max_length=2048`)
- Test with 1 epoch first

### 2. Monitor Overfitting
- Use evaluation data if available
- Watch for train/eval loss divergence
- Save checkpoints regularly

### 3. Memory Optimization
```bash
# Enable gradient checkpointing
python train_dotsocr.py --gradient_checkpointing

# Use LoRA for large models
python train_dotsocr.py --lora_training --lora_rank 4

# Reduce sequence length
python train_simple.py --max_length 4096
```

### 4. Debugging
```bash
# Enable detailed logging
export TRANSFORMERS_VERBOSITY=info

# Check data loading
python -c "
from train_simple import SimpleDotsOCRDataset
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('./weights/DotsOCR', trust_remote_code=True)
ds = SimpleDotsOCRDataset('training_data.jsonl', processor)
print(f'Dataset size: {len(ds)}')
print(f'Sample: {ds[0]}')
"
```

## 🚨 Common Issues

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 16

# Use LoRA
--lora_training

# Reduce sequence length
--max_length 2048
```

### Slow Training
```bash
# Use mixed precision
--bf16

# Increase workers
--num_workers 4

# Use flash attention (already enabled by default)
```

### Bad Results
- Check your training data quality
- Ensure proper reading order in PAGEXML conversion
- Try different learning rates
- Increase training epochs
- Use evaluation data to monitor progress

## 📝 Example Training Session

```bash
# 1. Prepare data
python create_training_data.py --input_dir ./pagexml_data --output_file training.jsonl

# 2. Split into train/eval (optional)
head -n 800 training.jsonl > train.jsonl
tail -n 200 training.jsonl > eval.jsonl

# 3. Start training
python train_simple.py \
    --data train.jsonl \
    --eval_data eval.jsonl \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --wandb \
    --run_name "pagexml-htr-experiment-1"

# 4. Model will be saved to ./simple_checkpoints/final_model/
```

## 🔄 Using Your Trained Model

After training, use your model like this:
```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "./simple_checkpoints/final_model/",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(
    "./simple_checkpoints/final_model/",
    trust_remote_code=True
)

# Use with dots_ocr parser
from dots_ocr.parser import DotsOCRParser
parser = DotsOCRParser(model_path="./simple_checkpoints/final_model/", use_hf=True)
```

Happy training! 🚀