import os
import json
import asyncio
import httpx
from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import base64
from io import BytesIO
from dots_ocr.model.inference import inference_with_vllm

from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import load_images_from_pdf, iter_images_from_pdf, get_pdf_page_count_fitz
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md

# A ThreadPoolExecutor for CPU-bound tasks
CPU_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count())

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class DotsOCRParser:
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
        
        try:
            image_with_layout = draw_layout_on_image(origin_image, cells)
        except Exception as e:
            print(f"Error drawing layout on image: {e}")
            image_with_layout = origin_image
            
        json_path = os.path.join(save_dir, f"{save_name}.json")
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(cells, f, ensure_ascii=False, indent=4)
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

    async def _parse_single_image(self, origin_image, prompt_mode, save_dir, save_name, source="image", page_idx=0, bbox=None, fitz_preprocess=False):
        """Asynchronous pipeline for a single image."""
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            
            # 1. Run CPU-bound image prep in executor
            image, prompt = await loop.run_in_executor(
                CPU_EXECUTOR, self._prepare_image_and_prompt, origin_image, prompt_mode, source, fitz_preprocess, bbox
            )
            
            # 2. Make non-blocking network call for inference
            response = await self._inference_with_vllm(image, prompt)
            
            # 3. Run CPU/IO-bound post-processing and saving in executor
            save_name_page = f"{save_name}_page_{page_idx}" if source == 'pdf' else save_name
            result = await loop.run_in_executor(
                CPU_EXECUTOR, self._process_and_save_results, response, prompt_mode, save_dir, save_name_page, origin_image, image
            )
            
            result['page_no'] = page_idx
            return result
        
    async def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        loop = asyncio.get_running_loop()
        origin_image = await loop.run_in_executor(CPU_EXECUTOR, fetch_image, input_path)
        result = await self._parse_single_image(
            origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess
        )
        return [result]
        
    async def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        loop = asyncio.get_running_loop()
        
        print(f"Loading PDF: {input_path}")
        # Run blocking PDF loading in executor
        images_origin = await loop.run_in_executor(CPU_EXECUTOR, load_images_from_pdf, input_path)
        
        total_pages = len(images_origin)
        print(f"Parsing PDF with {total_pages} pages using concurrency of {self.concurrency_limit}...")

        tasks = [
            self._parse_single_image(
                origin_image=image,
                prompt_mode=prompt_mode,
                save_dir=save_dir,
                save_name=filename,
                source="pdf",
                page_idx=i,
            ) for i, image in enumerate(images_origin)
        ]

        results = await tqdm.gather(*tasks, desc="Processing PDF pages")
        
        results.sort(key=lambda x: x["page_no"])
        return results

    async def parse_pdf_stream(self, input_path, filename, prompt_mode, save_dir):
        loop = asyncio.get_running_loop()

        print(f"Loading PDF for streaming: {input_path}")
        images_origin = await loop.run_in_executor(CPU_EXECUTOR, load_images_from_pdf, input_path)
        
        total_pages = len(images_origin)
        if total_pages == 0:
            print("Warning: PDF has no pages or no images could be extracted.")
            return

        print(f"Streaming parse for PDF with {total_pages} pages using concurrency of {self.concurrency_limit}...")

        tasks = [
            asyncio.create_task(self._parse_single_image(
                origin_image=image,
                prompt_mode=prompt_mode,
                save_dir=save_dir,
                save_name=filename,
                source="pdf",
                page_idx=i,
            )) for i, image in enumerate(images_origin)
        ]

        for future in tqdm(asyncio.as_completed(tasks), total=total_pages, desc="Processing PDF pages (stream)"):
            try:
                result = await future
                yield result
            except Exception as e:
                print(f"An error occurred while processing a page: {e}")
    
    async def parse_pdf_stream2(self, input_path, filename, prompt_mode, save_dir, batch_size=32):

        loop = asyncio.get_running_loop()
        
        total_pages = get_pdf_page_count_fitz(input_path)

        tasks = []
        with tqdm(total=total_pages, desc="Processing PDF pages (stream)") as pbar:
            for page_idx, image in iter_images_from_pdf(input_path, dpi=200):
                tasks.append(asyncio.create_task(
                    self._parse_single_image(
                        origin_image=image,
                        prompt_mode=prompt_mode,
                        save_dir=save_dir,
                        save_name=filename,
                        source="pdf",
                        page_idx=page_idx,
                    )
                ))
                if len(tasks) >= batch_size:
                    for future in asyncio.as_completed(tasks):
                        yield await future
                        pbar.update(1)
                    tasks.clear()

            for future in asyncio.as_completed(tasks):
                yield await future
                pbar.update(1)