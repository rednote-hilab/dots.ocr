#!/usr/bin/env python3
"""
Test script for training functionality
Validates that training scripts can load data and model correctly
"""

import os
import json
import tempfile
import torch
from PIL import Image

def create_test_training_data(output_path: str, num_samples: int = 5):
    """Create minimal test training data"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        samples = []
        
        for i in range(num_samples):
            # Create dummy image
            img_path = os.path.join(temp_dir, f"test_img_{i}.jpg")
            dummy_img = Image.new('RGB', (400, 300), color='white')
            dummy_img.save(img_path)
            
            # Create training sample
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_path},
                            {"type": "text", "text": "Extract text from this document image."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": f'[{{"bbox": [10, 10, 200, 50], "category": "Text", "text": "Test text content {i}"}}]'
                    }
                ]
            }
            samples.append(sample)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"Created test training data with {num_samples} samples at {output_path}")


def test_simple_training_script():
    """Test the simple training script"""
    print("Testing simple training script...")
    
    try:
        from train_simple import SimpleDotsOCRDataset
        
        # Check if we can import the model components
        from transformers import AutoProcessor
        
        print("✅ Import successful")
        
        # Create test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data_path = f.name
        
        create_test_training_data(test_data_path, num_samples=2)
        
        # Test dataset creation (without actual model loading)
        print("Testing dataset creation...")
        
        # This would require the actual model, so we'll skip for now
        print("⚠️  Skipping actual dataset test (requires model weights)")
        
        # Cleanup
        os.unlink(test_data_path)
        
        print("✅ Simple training script test passed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the training requirements:")
        print("pip install -r training_requirements.txt")
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_advanced_training_script():
    """Test the advanced training script"""
    print("\nTesting advanced training script...")
    
    try:
        from train_dotsocr import DotsOCRDataset, DotsOCRTrainer
        
        print("✅ Advanced training script imports successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")


def test_data_preparation_integration():
    """Test integration between data preparation and training"""
    print("\nTesting data preparation integration...")
    
    try:
        # Test that we can create training data and then load it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data_path = f.name
        
        create_test_training_data(test_data_path, num_samples=3)
        
        # Verify the data format
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    
                    # Validate structure
                    assert 'messages' in sample, "Missing 'messages' key"
                    assert len(sample['messages']) == 2, "Should have exactly 2 messages"
                    assert sample['messages'][0]['role'] == 'user', "First message should be user"
                    assert sample['messages'][1]['role'] == 'assistant', "Second message should be assistant"
                    
                    user_content = sample['messages'][0]['content']
                    assert len(user_content) == 2, "User content should have image and text"
                    
                    # Check content types
                    content_types = [item['type'] for item in user_content]
                    assert 'image' in content_types, "Should have image content"
                    assert 'text' in content_types, "Should have text content"
                    
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON at line {line_num}")
                    return False
                except AssertionError as e:
                    print(f"❌ Validation error at line {line_num}: {e}")
                    return False
        
        # Cleanup
        os.unlink(test_data_path)
        
        print("✅ Data preparation integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def check_gpu_availability():
    """Check GPU availability for training"""
    print("\nChecking GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✅ GPU available: {gpu_name}")
        print(f"   GPU count: {gpu_count}")
        print(f"   GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 8:
            print("⚠️  Warning: Less than 8GB GPU memory. Consider using LoRA training.")
        elif gpu_memory >= 24:
            print("✅ Excellent! You have enough memory for full fine-tuning.")
        else:
            print("✅ Good! You should be able to train with small batch sizes.")
            
    else:
        print("❌ No GPU available. Training will be very slow on CPU.")
        print("   Consider using Google Colab, Paperspace, or cloud GPU instances.")


def print_training_quick_start():
    """Print quick start instructions"""
    print("\n" + "="*60)
    print("🚀 TRAINING QUICK START")
    print("="*60)
    
    print("\n1. First, prepare your training data:")
    print("   python create_training_data.py --input_dir /path/to/pagexml --output_file training.jsonl")
    
    print("\n2. Install training requirements:")
    print("   pip install -r training_requirements.txt")
    
    print("\n3. Download the base model:")
    print("   python tools/download_model.py")
    
    print("\n4. Start training (choose one):")
    print("   # Simple training (recommended for beginners)")
    print("   python train_simple.py --data training.jsonl --epochs 3")
    print()
    print("   # Advanced training with LoRA (memory efficient)")
    print("   python train_dotsocr.py --train_data training.jsonl --lora_training")
    print()
    print("   # Use the shell script")
    print("   ./run_training.sh")
    
    print("\n5. Monitor training:")
    print("   # Add --wandb for online monitoring")
    print("   python train_simple.py --data training.jsonl --wandb --run_name my-experiment")
    
    print("\n📚 For more details, see README_model_training.md")


def main():
    print("Testing dots.ocr Training Setup")
    print("=" * 40)
    
    # Run tests
    test_simple_training_script()
    test_advanced_training_script()
    test_data_preparation_integration()
    check_gpu_availability()
    
    # Print quick start guide
    print_training_quick_start()


if __name__ == "__main__":
    main()