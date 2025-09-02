from concurrent.futures import ThreadPoolExecutor
import os
import json
import asyncio
import httpx
from dots_ocr.model.inference import inference_with_vllm

from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md


class PageParser:
    """
    Asynchronous parser for image or PDF files.
    """
    def __init__(self, ip='localhost', port=8000, model_name='model', temperature=0.1, top_p=1.0,
                 max_completion_tokens=32768, concurrency_limit=16, dpi=200, min_pixels=None, max_pixels=None):
        self.dpi = dpi
        self.ip = ip
        self.port = port
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)
        self.http_client = httpx.AsyncClient(timeout=300.0)
        self.cpu_executor = ThreadPoolExecutor(max_workers=os.cpu_count())

        print(f"Async parser initialized with concurrency limit: {self.concurrency_limit}")
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    async def _inference_with_vllm(self, image, prompt):
        response = await inference_with_vllm(
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
            
    def _prepare_image_and_prompt(self, origin_image, prompt_mode, source, fitz_preprocess, bbox):
        """Synchronous, CPU-bound part of image preparation."""
        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bbox = pre_process_bboxes(origin_image, [bbox], input_width=image.width, input_height=image.height)[0]
            prompt += str(bbox)
        
        return image, prompt
    
    def _process_and_save_results(self, response, prompt_mode, save_dir, save_name, origin_image, image):
        """Synchronous, CPU/IO-bound part of post-processing and saving."""
        os.makedirs(save_dir, exist_ok=True)
        result = {}
        cells, _ = post_process_output(response, prompt_mode, origin_image, image)
        width, height = origin_image.size
        cells_with_size = {
            "width": width,
            "height": height,
            "full_layout_info": cells
        }

        try:
            image_with_layout = draw_layout_on_image(origin_image, cells)
        except Exception as e:
            print(f"Error drawing layout on image: {e}")
            image_with_layout = origin_image
            
        json_path = os.path.join(save_dir, f"{save_name}.json")
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(cells_with_size, f, ensure_ascii=False, indent=4)
        result['layout_info_path'] = json_path

        md_content = layoutjson2md(origin_image, cells, text_key='text')
        md_content_nohf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)
        
        md_path = os.path.join(save_dir, f"{save_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        result['md_content_path'] = md_path

        md_nohf_path = os.path.join(save_dir, f"{save_name}_nohf.md")
        with open(md_nohf_path, "w", encoding="utf-8") as f:
            f.write(md_content_nohf)
        result['md_content_nohf_path'] = md_nohf_path

        return result

    def _process_results(self, response, prompt_mode, origin_image, image, page_idx = None):
        """Synchronous, CPU/IO-bound part of post-processing and saving."""
        result = {}
        cells, _ = post_process_output(response, prompt_mode, origin_image, image)
        width, height = origin_image.size
        cells_with_size = {
            "width": width,
            "height": height,
            "full_layout_info": cells
        }
        if page_idx is not None:
            cells_with_size['page_no'] = page_idx
        return cells_with_size

    async def _save_results(self, cell, save_dir, save_name, image_origin):

        result = {}
        if cell['page_no'] is not None:
            result['page_no'] = cell['page_no']
        json_path = os.path.join(save_dir, f"{save_name}.json")
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(cell, f, ensure_ascii=False, indent=4)
        result['layout_info_path'] = json_path

        md_content = layoutjson2md(image_origin, cell["full_layout_info"], text_key='text')
        md_content_nohf = layoutjson2md(image_origin, cell["full_layout_info"], text_key='text', no_page_hf=True)
        
        md_path = os.path.join(save_dir, f"{save_name}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        result['md_content_path'] = md_path

        md_nohf_path = os.path.join(save_dir, f"{save_name}_nohf.md")
        with open(md_nohf_path, "w", encoding="utf-8") as f:
            f.write(md_content_nohf)
        result['md_content_nohf_path'] = md_nohf_path

        return result


    async def _parse_single_image_do_not_save(self, origin_image, prompt_mode, source="image", page_idx=0, bbox=None, fitz_preprocess=False):
        """Asynchronous pipeline for a single image."""
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            
            # 1. Run CPU-bound image prep in executor
            image, prompt = await loop.run_in_executor(
                self.cpu_executor, self._prepare_image_and_prompt, origin_image, prompt_mode, source, fitz_preprocess, bbox
            )
            
            # 2. Make non-blocking network call for inference
            response = await self._inference_with_vllm(image, prompt)
            
            # 3. Run CPU/IO-bound post-processing and saving in executor
            cells = await loop.run_in_executor(
                self.cpu_executor, self._process_results, response, prompt_mode, origin_image, image, page_idx
            )
            
            return cells

    async def _parse_single_image(self, origin_image, prompt_mode, save_dir, save_name, source="image", page_idx=0, bbox=None, fitz_preprocess=False):
        """Asynchronous pipeline for a single image."""
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            
            # 1. Run CPU-bound image prep in executor
            image, prompt = await loop.run_in_executor(
                self.cpu_executor, self._prepare_image_and_prompt, origin_image, prompt_mode, source, fitz_preprocess, bbox
            )
            
            # 2. Make non-blocking network call for inference
            response = await self._inference_with_vllm(image, prompt)
            
            # 3. Run CPU/IO-bound post-processing and saving in executor
            save_name_page = f"{save_name}_page_{page_idx}" if source == 'pdf' else save_name
            result = await loop.run_in_executor(
                self.cpu_executor, self._process_and_save_results, response, prompt_mode, save_dir, save_name_page, origin_image, image
            )
            
            result['page_no'] = page_idx
            return result