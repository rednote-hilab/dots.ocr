import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import json
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.image_utils import fetch_image
from PIL import Image
import tempfile
import base64
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables for model and processor
model = None
processor = None
processing_lock = threading.Lock()

def load_model():
    """Load the OCR model and processor"""
    global model, processor

    TORCH_DTYPE = 'float32'
    if 'TORCH_DTYPE' in os.environ:
        TORCH_DTYPE = os.environ['TORCH_DTYPE']
    elif torch.cuda.get_device_capability()[0] > 7: # 30 series
        print("30-series or newer is detected! use bfloat16!")
        TORCH_DTYPE = 'bfloat16'

    torch_dtype = torch.float32
    if TORCH_DTYPE == 'float32':
        torch_dtype = torch.float32
    elif TORCH_DTYPE == 'float16':
        torch_dtype = torch.float16
    elif TORCH_DTYPE == 'bfloat16':
        torch_dtype = torch.bfloat16
    else:
        print(f"Unknown TORCH_DTYPE: {TORCH_DTYPE}, default to float32")

    model_path = "./weights/DotsOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("Model loaded successfully")

def inference_with_stream(image_data, prompt, max_new_tokens=24000):
    """Run inference with streaming output using TextIteratorStreamer"""
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image_data},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to("cuda")
    
    # Create streamer
    streamer = TextIteratorStreamer(
        processor.tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True
    )
    
    # Generation parameters
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "streamer": streamer,
        "pad_token_id": processor.tokenizer.eos_token_id
    }
    
    # Start generation in a separate thread
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()
    
    return streamer

def inference_non_stream(image_data, prompt, max_new_tokens=12000):
    """Run inference without streaming"""
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image_data},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/ocr', methods=['POST'])
def ocr():
    """OCR endpoint that processes images and returns text"""
    # Check if model is loaded
    if model is None or processor is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    # Parse request data
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Extract parameters
    image_data = data.get('image')
    prompt_type = data.get('prompt_type', 'prompt_layout_all_en')
    max_new_tokens = data.get('max_new_tokens', 24000)
    stream = data.get('stream', False)
    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400
    
    if prompt_type not in dict_promptmode_to_prompt:
        return jsonify({"error": f"Invalid prompt_type. Must be one of: {list(dict_promptmode_to_prompt.keys())}"}), 400
    
    # Use lock to ensure single request processing
    if not processing_lock.acquire(blocking=False):
        return jsonify({"error": "Server is busy processing another request"}), 429

    try:
        # Get prompt
        prompt = dict_promptmode_to_prompt[prompt_type]
        
        if stream:
            # Return streaming response using TextIteratorStreamer
            def generate_stream():
                try:
                    streamer = inference_with_stream(image_data, prompt, max_new_tokens)
                    
                    accumulated_text = ""
                    for new_text in streamer:
                        accumulated_text += new_text
                        response_data = {
                            "model": "dots-ocr",
                            "created_at": datetime.now().isoformat(),
                            "response": new_text,
                            "done": False
                        }
                        print(new_text, end='')
                        yield f"{json.dumps(response_data)}\n"
                    
                    # Final response
                    final_response = {
                        "model": "dots-ocr",
                        "created_at": datetime.now().isoformat(),
                        "response": '',
                        "done": True
                    }
                    yield f"{json.dumps(final_response)}\n"
                    
                except Exception as e:
                    error_response = {
                        "model": "dots-ocr",
                        "error": str(e),
                        "done": True
                    }
                    yield f"{json.dumps(error_response)}\n"
            
            return Response(generate_stream(), mimetype='application/x-ndjson')

        else:
            # Non-streaming response
            result = inference_non_stream(image_data, prompt, max_new_tokens)
            

            return jsonify({
                "model": "dots-ocr",
                "response": result,
                "prompt_type": prompt_type
            })

    except Exception as e:

        return jsonify({"error": str(e)}), 500
    
    finally:
        processing_lock.release()

if __name__ == '__main__':
    print("Loading OCR model...")
    load_model()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)