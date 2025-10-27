#!/bin/bash
# Training script for dots.ocr model

# Configuration
DATA_DIR="./your_data"
TRAIN_DATA="${DATA_DIR}/training_data.jsonl"
EVAL_DATA="${DATA_DIR}/eval_data.jsonl"  # Optional
MODEL_PATH="./weights/DotsOCR"
OUTPUT_DIR="./checkpoints"

# Training parameters
EPOCHS=3
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=2e-5
MAX_LENGTH=8192

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Starting dots.ocr training..."
echo "Data: ${TRAIN_DATA}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"

# Method 1: Simple training script (recommended for beginners)
echo "Using simple training script..."
python train_simple.py \
    --data "${TRAIN_DATA}" \
    --eval_data "${EVAL_DATA}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --max_length ${MAX_LENGTH} \
    --wandb \
    --run_name "dots-ocr-$(date +%Y%m%d-%H%M%S)"

# Method 2: Advanced training script (for more control)
# python train_dotsocr.py \
#     --train_data "${TRAIN_DATA}" \
#     --eval_data "${EVAL_DATA}" \
#     --model_name_or_path "${MODEL_PATH}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --num_train_epochs ${EPOCHS} \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
#     --learning_rate ${LEARNING_RATE} \
#     --max_seq_length ${MAX_LENGTH} \
#     --bf16 \
#     --report_to wandb \
#     --run_name "dots-ocr-advanced-$(date +%Y%m%d-%H%M%S)"

# Method 3: LoRA training (memory efficient)
# python train_dotsocr.py \
#     --train_data "${TRAIN_DATA}" \
#     --model_name_or_path "${MODEL_PATH}" \
#     --output_dir "${OUTPUT_DIR}/lora" \
#     --lora_training \
#     --lora_rank 8 \
#     --num_train_epochs ${EPOCHS} \
#     --per_device_train_batch_size 2 \
#     --learning_rate 1e-4 \
#     --bf16

echo "Training completed!"
echo "Model saved to: ${OUTPUT_DIR}"