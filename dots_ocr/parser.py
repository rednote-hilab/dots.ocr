import os
import json
import concurrent.futures
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse

import torch

from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


class DotsOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self, 
            ip='localhost',
            port=8000,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=64,
            dpi = 200, 
            output_dir="./output", 
            min_pixels=None,
            max_pixels=None,
            use_hf=False,
            use_mps=False,
        ):
        self.dpi = dpi

        # default args for vllm server
        self.ip = ip
        self.port = port
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.use_hf = use_hf
        self.use_mps = use_mps
        
        # Set environment variables for offline caches when using MPS
        if self.use_mps:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            
        if self.use_hf:
            self._load_hf_model()
            print(f"use hf model, num_thread will be set to 1")
        elif self.use_mps:
            self._load_hf_model()
            print(f"use mps model, num_thread will be set to 1")
        else:
            print(f"use vllm model, num_thread will be set to {self.num_thread}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    def _load_hf_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig
        from qwen_vl_utils import process_vision_info
        import types as _types

        model_path = "./weights/DotsOCR"
        
        # MPS-specific configuration
        if self.use_mps:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            
            # Force SDPA & disable SWA for MPS compatibility
            config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
            if getattr(config, "vision_config", None) is not None:
                config.vision_config.attn_implementation = "sdpa"
            else:
                setattr(config, "attn_implementation", "sdpa")
            if hasattr(config, "use_sliding_window"):
                config.use_sliding_window = False
            if hasattr(config, "sliding_window"):
                config.sliding_window = None
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                attn_implementation="sdpa",
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
            if device == "mps":
                self.model.to("mps")
                # CRITICAL: force fp16 to avoid dtype mismatch on MPS
                orig_forward_func = self.model.vision_tower.forward.__func__  # unbound function
                def _forward_no_bf16(self, hidden_states, grid_thw, bf16=True):
                    return orig_forward_func(self, hidden_states, grid_thw, bf16=False)
                self.model.vision_tower.forward = _types.MethodType(_forward_no_bf16, self.model.vision_tower)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        self.process_vision_info = process_vision_info

    def _inference_with_hf(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if self.use_mps:
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device == "mps" else torch.float32
            
            # Move to device + force FP16 on floats for MPS
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if torch.is_floating_point(v):
                        v = v.to(dtype=dtype)
                    inputs[k] = v
        else:
            inputs = inputs.to("cuda")

        # Use streaming inference with IteratorStreamer
        from transformers import TextIteratorStreamer
        import threading
        import time
        
        # Create iterator streamer for streaming output
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=100,
        )
        generated_text = []

        # Start generation in a separate thread
        def generate():
            gen_kwargs = dict(
                max_new_tokens=24000,   
                do_sample=False,
                temperature=None,
                # Performance optimizations
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Use greedy decoding for speed
                early_stopping=True,  # Stop early if EOS token is generated
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=[self.processor.tokenizer.eos_token_id]
            )
            with torch.inference_mode():
                self.model.generate(**inputs, streamer=streamer, **gen_kwargs)

        # Start generation thread
        generation_thread = threading.Thread(target=generate)
        generation_thread.start()
        
        print("Generating content:")
        buf = []
        last_flush = time.time()
        try:
            for new_text in streamer:
                generated_text.append(new_text)
                buf.append(new_text)
                
                if len("".join(buf)) >= 1024 or (time.time() - last_flush) > 0.2:
                    print("".join(buf), end="", flush=False)
                    buf.clear()
                    last_flush = time.time()
        except Exception as e:
            print(f"\nStreaming error: {e}")
        
        if buf:
            print("".join(buf), end="")
        # Wait for generation to complete with timeout
        generation_thread.join(timeout=5.0)  # 5 second timeout
        if generation_thread.is_alive():
            print("\nWarning: Generation thread still running, continuing...")
         
        print()  # New line after generation completes
        
        # Convert collected tokens to tensor for final decode
        response = "".join(generated_text)

        
        # Clear torch cache to prevent memory overflow
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
        return response

    def _inference_with_vllm(self, image, prompt):
        response = inference_with_vllm(
            image,
            prompt, 
            model_name=self.model_name,
            ip=self.ip,
            port=self.port,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <+ {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True) # used for clean output or metric of omnidocbench、olmbench 
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
        total_pages = len(images_origin)
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        print(f"Parsing PDF with {total_pages} pages...")

        results = []
        if self.use_hf:
            # Sequential processing for use_hf case
            with tqdm(total=total_pages, desc="Processing PDF pages (sequential)") as pbar:
                for task in tasks:
                    result = _execute_task(task)
                    results.append(result)
                    pbar.update(1)
        else:
            # Parallel processing for vLLM case
            num_thread = min(total_pages, self.num_thread)
            print(f"Using {num_thread} threads for parallel processing...")
            with ThreadPoolExecutor(max_workers=num_thread) as executor:
                with tqdm(total=total_pages, desc="Processing PDF pages (parallel)") as pbar:
                    future_to_task = {executor.submit(_execute_task, task): task for task in tasks}
                    for future in concurrent.futures.as_completed(future_to_task):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path
        return results

    def parse_file(self, 
        input_path, 
        output_dir="", 
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False
        ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")
        
        print(f"Parsing finished, results saving to {save_dir}")
        with open(os.path.join(output_dir, os.path.basename(filename)+'.jsonl'), 'w', encoding="utf-8") as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        return results



def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input PDF/image file path"
    )
    
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=16,
        help=""
    )
    parser.add_argument(
        "--no_fitz_preprocess", action='store_true',
        help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--use_hf", type=bool, default=False,
        help=""
    )
    parser.add_argument(
        "--use_mps", type=bool, default=False,
        help=""
    )
    args = parser.parse_args()

    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=args.use_hf,
        use_mps=args.use_mps,
    )

    fitz_preprocess = not args.no_fitz_preprocess
    if fitz_preprocess:
        print(f"Using fitz preprocess for image input, check the change of the image pixels")
    result = dots_ocr_parser.parse_file(
        args.input_path, 
        prompt_mode=args.prompt,
        bbox=args.bbox,
        fitz_preprocess=fitz_preprocess,
        )
    


if __name__ == "__main__":
    main()
