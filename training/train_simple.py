#!/usr/bin/env python3
"""
Simplified training script for dots.ocr model
Easy-to-use interface for fine-tuning on your data

Usage:
    python train_simple.py --data training_data.jsonl --epochs 3 --batch_size 2
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm
import wandb
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDotsOCRDataset(Dataset):
    """Simplified dataset for dots.ocr training"""
    
    def __init__(self, data_path: str, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length
        
        # Load data
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        sample = json.loads(line.strip())
                        self.samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        
        # Extract user message components
        user_content = messages[0]['content']
        image_path = None
        prompt_text = None
        
        for item in user_content:
            if item['type'] == 'image':
                image_path = item['image']
            elif item['type'] == 'text':
                prompt_text = item['text']
        
        # Load image
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # Create dummy image if file not found
                image = Image.new('RGB', (224, 224), color='white')
        except Exception:
            image = Image.new('RGB', (224, 224), color='white')
        
        # Get target text
        target_text = messages[1]['content']
        
        # Create conversation for vision processing
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        # Apply chat template to get text input
        text_input = self.processor.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process vision inputs using qwen_vl_utils
        image_inputs, video_inputs = process_vision_info(conversation)
        
        # Process with processor including vision inputs
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_inputs = self.processor.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length // 2,  # Reserve space for prompt
            return_tensors="pt"
        )
        
        # Combine input and target
        input_ids = inputs['input_ids'].squeeze(0)
        target_ids = target_inputs['input_ids'].squeeze(0)
        
        # Create full sequence (input + target)
        full_input_ids = torch.cat([input_ids, target_ids], dim=0)
        
        # Create attention mask
        attention_mask = torch.ones_like(full_input_ids)
        
        # Create labels (mask the prompt part)
        labels = full_input_ids.clone()
        labels[:len(input_ids)] = -100  # Don't train on prompt
        
        # Handle all possible vision-related inputs
        result = {
            'input_ids': full_input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        # Add vision inputs if they exist
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            result['pixel_values'] = inputs['pixel_values'].squeeze(0)
        
        if 'image_grid_thw' in inputs and inputs['image_grid_thw'] is not None:
            result['image_grid_thw'] = inputs['image_grid_thw'].squeeze(0)
            
        if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
            result['video_grid_thw'] = inputs['video_grid_thw'].squeeze(0)
        
        return result


def collate_fn(batch):
    """Collate function for batching"""
    # Get max length
    max_len = max([item['input_ids'].size(0) for item in batch])
    
    # Pad sequences
    padded_batch = {}
    
    for key in ['input_ids', 'attention_mask', 'labels']:
        padded_sequences = []
        for item in batch:
            seq = item[key]
            pad_length = max_len - seq.size(0)
            
            if key == 'labels':
                # Pad labels with -100
                padded_seq = torch.cat([seq, torch.full((pad_length,), -100)])
            else:
                # Pad others with 0
                padded_seq = torch.cat([seq, torch.zeros(pad_length, dtype=seq.dtype)])
            
            padded_sequences.append(padded_seq)
        
        padded_batch[key] = torch.stack(padded_sequences)
    
    # Handle vision-related inputs
    vision_keys = ['pixel_values', 'image_grid_thw', 'video_grid_thw']
    
    for key in vision_keys:
        # Check if any item in batch has this key
        if any(key in item and item[key] is not None for item in batch):
            values = []
            for item in batch:
                if key in item and item[key] is not None:
                    values.append(item[key])
                else:
                    # Skip items that don't have this key or have None values
                    continue
            
            if values:
                padded_batch[key] = torch.stack(values)
    
    return padded_batch


def train_model(
    model, 
    train_dataloader, 
    eval_dataloader=None,
    num_epochs=3,
    learning_rate=2e-5,
    warmup_steps=100,
    save_dir="./checkpoints",
    log_wandb=False,
    gradient_accumulation_steps=4  # Add gradient accumulation
):
    """Simple training loop with memory optimization"""
    
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update model every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Clear cache to save memory
                torch.cuda.empty_cache()
            
            # Logging
            epoch_loss += loss.item() * gradient_accumulation_steps  # Unscale for logging
            num_batches += 1
            
            # Update progress bar
            avg_loss = epoch_loss / num_batches
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}' if scheduler.get_last_lr() else '0'
            })
            
            # Log to wandb
            if log_wandb and (batch_idx + 1) % gradient_accumulation_steps == 0:
                wandb.log({
                    'train_loss': loss.item() * gradient_accumulation_steps,
                    'learning_rate': scheduler.get_last_lr()[0] if scheduler.get_last_lr() else 0,
                    'epoch': epoch,
                    'step': global_step
                })
        
        # Handle remaining gradients at end of epoch
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        
        # Evaluation
        if eval_dataloader:
            eval_loss = evaluate_model(model, eval_dataloader, device)
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            
            if log_wandb:
                wandb.log({'eval_loss': eval_loss, 'epoch': epoch})
        else:
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(save_dir, f"checkpoint-epoch-{epoch + 1}")
        model.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()


def evaluate_model(model, eval_dataloader, device):
    """Simple evaluation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0


def main():
    parser = argparse.ArgumentParser(description='Simple dots.ocr training')
    parser.add_argument('--data', required=True, help='Training data JSONL file')
    parser.add_argument('--eval_data', default=None, help='Evaluation data JSONL file')
    parser.add_argument('--model_path', default='./weights/DotsOCR', help='Model path')
    parser.add_argument('--output_dir', default='./simple_checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=4096, help='Max sequence length (reduced default)')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--wandb', action='store_true', help='Log to wandb')
    parser.add_argument('--run_name', default='dots-ocr-simple', help='Run name')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 training')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project='dots-ocr-training', name=args.run_name)
    
    # Load model and processor
    logger.info(f"Loading model from {args.model_path}")
    
    # Determine dtype based on arguments
    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = SimpleDotsOCRDataset(args.data, processor, args.max_length)
    
    eval_dataset = None
    if args.eval_data:
        eval_dataset = SimpleDotsOCRDataset(args.eval_data, processor, args.max_length)
    
    # Create dataloaders with reduced num_workers to save memory
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Reduced to save memory
        pin_memory=False  # Disable pin_memory to save memory
    )
    
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False
        )
    
    # Train
    logger.info("Starting training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        save_dir=args.output_dir,
        log_wandb=args.wandb
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    
    logger.info(f"Training completed! Final model saved to {final_model_path}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()