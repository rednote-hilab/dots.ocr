import os
import json
import asyncio
import httpx
from tqdm.asyncio import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import base64
from io import BytesIO

from dots_ocr.utils.image_utils import fetch_image
from dots_ocr.utils.doc_utils import load_images_from_pdf, iter_images_from_pdf, get_pdf_page_count_fitz
from dots_ocr.utils.directory_cleaner import DirectoryCleaner
from dots_ocr.utils.page_parser import PageParser

def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL image to a base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

class DotsOCRParser:
    def __init__(self, ip='localhost', port=8000, model_name='model', temperature=0.1, top_p=1.0,
                 max_completion_tokens=32768, concurrency_limit=16, dpi=200, min_pixels=None, max_pixels=None):
        self.parser = PageParser(
            ip=ip,
            port=port,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            concurrency_limit=concurrency_limit,
            dpi=dpi,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        self.directory_cleaner = None
        
    async def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        loop = asyncio.get_running_loop()
        origin_image = await loop.run_in_executor(self.parser.cpu_executor, fetch_image, input_path)
        result = await self.parser._parse_single_image(
            origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess
        )
        return [result]
        
    async def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        loop = asyncio.get_running_loop()
        
        print(f"Loading PDF: {input_path}")
        # Run blocking PDF loading in executor
        images_origin = await loop.run_in_executor(self.parser.cpu_executor, load_images_from_pdf, input_path)
        
        total_pages = len(images_origin)
        print(f"Parsing PDF with {total_pages} pages using concurrency of {self.parser.concurrency_limit}...")

        tasks = [
            self.parser._parse_single_image(
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
    
    async def rebuild_directory(self, cells_list, images_origin):
        if self.directory_cleaner is None:
            self.directory_cleaner = DirectoryCleaner()

        await self.directory_cleaner.reset_header_level(cells_list, images_origin)
    
    async def parse_pdf_rebuild_directory(self, input_path, filename, prompt_mode, save_dir):
        loop = asyncio.get_running_loop()
        
        print(f"Loading PDF: {input_path}")
        # Run blocking PDF loading in executor
        images_origin = await loop.run_in_executor(self.parser.cpu_executor, load_images_from_pdf, input_path)
        
        total_pages = len(images_origin)
        print(f"Parsing PDF with {total_pages} pages using concurrency of {self.parser.concurrency_limit}...")

        tasks = [
            self.parser._parse_single_image_do_not_save(
                origin_image=image,
                prompt_mode=prompt_mode,
                source="pdf",
                page_idx=i,
            ) for i, image in enumerate(images_origin)
        ]

        cells_list = await tqdm.gather(*tasks, desc="Processing PDF pages")
        cells_list.sort(key=lambda x: x["page_no"])

        await self.rebuild_directory(cells_list, images_origin)
    
        results = []
        for cell in cells_list:
            save_name_page = f"{filename}_page_{cell["page_no"]}"
            result = await self.parser._save_results(cell, save_dir, save_name_page, images_origin[cell["page_no"]])
            results.append(result)
    
        return results

    async def parse_pdf_stream(self, input_path, filename, prompt_mode, save_dir, batch_size=16, existing_pages=set()):

        loop = asyncio.get_running_loop()
        
        total_pages = get_pdf_page_count_fitz(input_path) - len(existing_pages)

        semaphore = asyncio.Semaphore(batch_size)
        tasks = []
        with tqdm(total=total_pages, desc="Processing PDF pages (stream)") as pbar:
            async def worker(page_idx, image):
                async with semaphore:
                    result = await self.parser._parse_single_image(
                        origin_image=image,
                        prompt_mode=prompt_mode,
                        save_dir=save_dir,
                        save_name=filename,
                        source="pdf",
                        page_idx=page_idx,
                    )
                    pbar.update(1)
                    return result

            for page_idx, image in iter_images_from_pdf(input_path, dpi=200, existing_pages=existing_pages):
                task = asyncio.create_task(worker(page_idx, image))
                tasks.append(task)

            for future in asyncio.as_completed(tasks):
                yield await future