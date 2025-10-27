#!/usr/bin/env python3
"""
Training script for dots.ocr model
Fine-tunes the vision-language model on document layout parsing tasks

Usage:
    python train_dotsocr.py --train_data training_data.jsonl --output_dir ./checkpoints
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DotsOCRTrainingArguments:
    """Training arguments specific to dots.ocr"""
    
    # Data arguments
    train_data: str = field(metadata={"help": "Path to training JSONL file"})
    eval_data: Optional[str] = field(default=None, metadata={"help": "Path to evaluation JSONL file"})
    max_seq_length: int = field(default=32768, metadata={"help": "Maximum sequence length"})
    max_image_pixels: int = field(default=11289600, metadata={"help": "Maximum image pixels (11M default)"})
    
    # Model arguments
    model_name_or_path: str = field(default="./weights/DotsOCR", metadata={"help": "Model path"})
    trust_remote_code: bool = field(default=True, metadata={"help": "Trust remote code"})
    
    # Training strategy
    freeze_vision_encoder: bool = field(default=False, metadata={"help": "Freeze vision encoder during training"})
    freeze_llm: bool = field(default=False, metadata={"help": "Freeze language model during training"})
    lora_training: bool = field(default=False, metadata={"help": "Use LoRA for efficient training"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    
    # Training arguments
    output_dir: str = field(default="./checkpoints", metadata={"help": "Output directory"})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Training batch size per device"})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Evaluation batch size per device"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps"})
    learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    warmup_steps: int = field(default=100, metadata={"help": "Warmup steps"})
    logging_steps: int = field(default=10, metadata={"help": "Logging steps"})
    save_steps: int = field(default=500, metadata={"help": "Save steps"})
    eval_steps: int = field(default=500, metadata={"help": "Evaluation steps"})
    save_total_limit: int = field(default=3, metadata={"help": "Maximum number of checkpoints to keep"})
    
    # Advanced training
    fp16: bool = field(default=False, metadata={"help": "Use FP16"})
    bf16: bool = field(default=True, metadata={"help": "Use BF16"})
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of dataloader workers"})
    report_to: Optional[str] = field(default=None, metadata={"help": "Reporting service (wandb, tensorboard)"})
    run_name: Optional[str] = field(default=None, metadata={"help": "Run name for logging"})


class DotsOCRDataset(Dataset):
    """Dataset for dots.ocr training data"""
    
    def __init__(
        self, 
        data_path: str, 
        processor: Any,
        max_seq_length: int = 32768,
        max_image_pixels: int = 11289600
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.max_image_pixels = max_image_pixels
        
        # Load training data
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    if self._validate_sample(sample):
                        self.samples.append(sample)
                    else:
                        logger.warning(f"Invalid sample at line {line_num}, skipping")
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(self.samples)} training samples from {data_path}")
    
    def _validate_sample(self, sample: Dict) -> bool:
        """Validate training sample format"""
        if not isinstance(sample, dict):
            return False
        if 'messages' not in sample:
            return False
        messages = sample['messages']
        if len(messages) != 2:
            return False
        if messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            return False
        return True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        try:
            # Process the sample
            messages = sample['messages']
            
            # Extract image and text from user message
            user_content = messages[0]['content']
            image_path = None
            prompt_text = None
            
            for content_item in user_content:
                if content_item['type'] == 'image':
                    image_path = content_item['image']
                elif content_item['type'] == 'text':
                    prompt_text = content_item['text']
            
            if image_path is None or prompt_text is None:
                raise ValueError("Missing image or text in user message")
            
            # Load and process image
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                logger.warning(f"Image not found: {image_path}, using dummy image")
                image = Image.new('RGB', (224, 224), color='white')
            
            # Resize image if too large
            image = self._resize_image_if_needed(image)
            
            # Get target response
            target_text = messages[1]['content']
            
            # Create conversation for tokenization
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": target_text
                }
            ]
            
            # Apply chat template
            text_input = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # Process vision inputs
            image_inputs, video_inputs = process_vision_info(conversation)
            
            # Tokenize
            model_inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            
            # Extract tensors and remove batch dimension
            input_ids = model_inputs['input_ids'].squeeze(0)
            attention_mask = model_inputs['attention_mask'].squeeze(0)
            
            # Handle images
            if 'pixel_values' in model_inputs:
                pixel_values = model_inputs['pixel_values'].squeeze(0)
            else:
                pixel_values = None
            
            # Create labels (same as input_ids for causal LM)
            labels = input_ids.clone()
            
            # Mask prompt tokens in labels (only train on assistant response)
            labels = self._mask_prompt_tokens(labels, text_input, target_text)
            
            result = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
            
            if pixel_values is not None:
                result['pixel_values'] = pixel_values
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return dummy sample to avoid crashing
            return self._get_dummy_sample()
    
    def _resize_image_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds max pixels"""
        width, height = image.size
        total_pixels = width * height
        
        if total_pixels > self.max_image_pixels:
            # Calculate new dimensions
            scale_factor = (self.max_image_pixels / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        return image
    
    def _mask_prompt_tokens(self, labels: torch.Tensor, full_text: str, target_text: str) -> torch.Tensor:
        """Mask prompt tokens so we only train on the assistant response"""
        # Find where the assistant response starts
        assistant_start = full_text.find(target_text)
        if assistant_start == -1:
            return labels  # Fallback: train on everything
        
        # Tokenize just the prompt part to find where to start unmasking
        prompt_part = full_text[:assistant_start]
        prompt_tokens = self.processor.tokenizer(prompt_part, add_special_tokens=False)['input_ids']
        prompt_length = len(prompt_tokens)
        
        # Mask prompt tokens (set to -100)
        labels[:prompt_length] = -100
        
        return labels
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample in case of errors"""
        return {
            'input_ids': torch.tensor([1, 2, 3]),  # Dummy tokens
            'attention_mask': torch.tensor([1, 1, 1]),
            'labels': torch.tensor([-100, -100, 3]),  # Only train on last token
        }


class DotsOCRTrainer(Trainer):
    """Custom trainer for dots.ocr"""
    
    def __init__(self, freeze_vision_encoder: bool = False, freeze_llm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_llm = freeze_llm
    
    def _setup_model_freezing(self):
        """Setup parameter freezing based on training strategy"""
        if not hasattr(self, '_freezing_setup_done'):
            model = self.model
            
            if self.freeze_vision_encoder:
                logger.info("Freezing vision encoder parameters")
                for name, param in model.named_parameters():
                    if 'visual' in name or 'vision' in name or 'vit' in name.lower():
                        param.requires_grad = False
            
            if self.freeze_llm:
                logger.info("Freezing language model parameters")
                for name, param in model.named_parameters():
                    if not ('visual' in name or 'vision' in name or 'vit' in name.lower()):
                        param.requires_grad = False
            
            # Count trainable parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                       f"({trainable_params/total_params*100:.2f}%)")
            
            self._freezing_setup_done = True
    
    def training_step(self, model, inputs):
        """Custom training step"""
        self._setup_model_freezing()
        return super().training_step(model, inputs)


def setup_lora(model, args: DotsOCRTrainingArguments):
    """Setup LoRA for efficient training"""
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Define LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with rank {args.lora_rank}, alpha {args.lora_alpha}")
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
        
    except ImportError:
        logger.error("PEFT library not found. Install with: pip install peft")
        raise


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    # Find max length in batch
    max_length = max(item['input_ids'].size(0) for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_mask = []
    labels = []
    pixel_values = []
    
    for item in batch:
        # Pad input_ids
        pad_length = max_length - item['input_ids'].size(0)
        padded_input_ids = torch.cat([
            item['input_ids'], 
            torch.full((pad_length,), 0)  # Pad with 0 (typically pad token)
        ])
        input_ids.append(padded_input_ids)
        
        # Pad attention_mask
        padded_attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_length)
        ])
        attention_mask.append(padded_attention_mask)
        
        # Pad labels
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_length,), -100)  # Pad labels with -100 (ignore index)
        ])
        labels.append(padded_labels)
        
        # Handle pixel values
        if 'pixel_values' in item and item['pixel_values'] is not None:
            pixel_values.append(item['pixel_values'])
    
    result = {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
    }
    
    if pixel_values:
        result['pixel_values'] = torch.stack(pixel_values)
    
    return result


def evaluate_model(model, eval_dataloader, processor):
    """Simple evaluation function"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'eval_loss': avg_loss,
        'eval_perplexity': perplexity
    }


def main():
    parser = argparse.ArgumentParser(description='Train dots.ocr model')
    
    # Add arguments
    parser.add_argument('--train_data', required=True, help='Path to training JSONL file')
    parser.add_argument('--eval_data', default=None, help='Path to evaluation JSONL file')
    parser.add_argument('--model_name_or_path', default='./weights/DotsOCR', help='Model path')
    parser.add_argument('--output_dir', default='./checkpoints', help='Output directory')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Gradient accumulation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=500, help='Save steps')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--max_seq_length', type=int, default=32768, help='Max sequence length')
    parser.add_argument('--freeze_vision_encoder', action='store_true', help='Freeze vision encoder')
    parser.add_argument('--freeze_llm', action='store_true', help='Freeze language model')
    parser.add_argument('--lora_training', action='store_true', help='Use LoRA training')
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--report_to', default=None, help='Logging service (wandb, tensorboard)')
    parser.add_argument('--run_name', default=None, help='Run name')
    parser.add_argument('--resume_from_checkpoint', default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.report_to == 'wandb':
        wandb.init(project='dots-ocr-training', name=args.run_name)
    
    # Load model and processor
    logger.info(f"Loading model from {args.model_name_or_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    
    # Setup LoRA if requested
    if args.lora_training:
        model = setup_lora(model, args)
    
    # Create datasets
    logger.info("Loading training data...")
    train_dataset = DotsOCRDataset(
        args.train_data, 
        processor, 
        max_seq_length=args.max_seq_length
    )
    
    eval_dataset = None
    if args.eval_data:
        logger.info("Loading evaluation data...")
        eval_dataset = DotsOCRDataset(
            args.eval_data,
            processor,
            max_seq_length=args.max_seq_length
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps if eval_dataset else None,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        run_name=args.run_name,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        dataloader_pin_memory=False,  # Can cause issues with vision models
    )
    
    # Create trainer
    trainer = DotsOCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        freeze_vision_encoder=args.freeze_vision_encoder,
        freeze_llm=args.freeze_llm,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    
    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()