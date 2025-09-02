import asyncio
import os
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dots_ocr.utils.doc_utils import load_images_from_pdf
from PIL import Image
from dots_ocr.utils.page_parser import PageParser
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.directory_cleaner import CropImage, SectionHeader, DirectoryStructure, Reranker, DirectoryCleaner

dots_parser = PageParser()

def extract_single_directory_headers(json_path):
    dir_structure = DirectoryStructure()
    if os.path.exists(json_path):
        dir_structure.load_from_json_path(json_path)
    return dir_structure

def extract_and_print_headers_with_bbox():
    """Extract section headers with bbox from JSON files"""
    directorys = []
    
    for i in range(14):
        json_path = f"/dots.ocr/test/test_merge/test{i}.json"
        if os.path.exists(json_path):
            dir_structure = DirectoryStructure()
            dir_structure.load_from_json_path(json_path)
            directorys.append(dir_structure)
            
            print(f"File {i}: {dir_structure}")
            for header in dir_structure.get_all_headers():
                print(f"  {header}")
            print('----------------------')
    
    return directorys


CPU_EXECUTOR = ProcessPoolExecutor(max_workers=os.cpu_count())
async def get_image(input_path):

    loop = asyncio.get_running_loop()
    
    print(f"Loading PDF: {input_path}")
    
    images = await loop.run_in_executor(CPU_EXECUTOR, load_images_from_pdf, input_path)

    return images


def concat_images(images, mode="vertical"):
    if mode == "vertical":
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        new_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        y_offset = 0
        for img in images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height
    elif mode == "horizontal":
        total_width = sum(img.width for img in images)
        total_height = max(img.height for img in images)
        new_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
    
    return new_img

    
async def gen_page_output():
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    images = await get_image(input_pdf_path)

    tasks = []
    for i in range(len(images)):
        tasks.append(
            dots_parser._parse_single_image(
                images[i],
                prompt_mode="prompt_layout_all_en",
                save_dir=f"/dots.ocr/test/test_merge",
                save_name=f"test{i}"
            )
        )

    await asyncio.gather(*tasks)


async def gen_concat_output():
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    images = await get_image(input_pdf_path)

    tasks = []
    for i in range(len(images)-1):
        new_image = concat_images([images[i], images[i+1]], mode="vertical")
        tasks.append(
            dots_parser._parse_single_image(
                new_image,
                prompt_mode="prompt_layout_all_en",
                save_dir=f"/dots.ocr/test/test_merge{i}",
                save_name="test"
            )
        )

    await asyncio.gather(*tasks)

async def extract_single_page_headers(page_index, save_crops=True):
    """Extract headers from a single page"""
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    
    images = await get_image(input_pdf_path)
    
    if page_index >= len(images):
        print(f"Page {page_index} not found in PDF")
        return None
    
    json_path = f"/dots.ocr/test/test_merge/test{page_index}.json"
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return None
    
    # Load directory structure
    dir_structure = extract_single_directory_headers(json_path)
    
    if save_crops:
        # Extract and save header crops
        save_dir = f"/dots.ocr/test/header_crops/page_{page_index}"
        crops = dir_structure.extract_all_header_crops(images[page_index], save_dir)
    else:
        # Just extract crops without saving
        crops = []
        for header in dir_structure.headers:
            crops.append(header.crop_from_image(images[page_index]))
    
    return dir_structure, crops


async def extract_header_images_from_pdf():
    """Extract header images from PDF pages"""
    input_pdf_path = "/dots.ocr/test/data/PGhandbook.pdf"
    
    images = await get_image(input_pdf_path)
    
    # Process each page
    cells_list = []
    for i in range(len(images)):
        json_path = f"/dots.ocr/test/test_merge/test{i}.json"
        if os.path.exists(json_path):
            print(f"Processing page {i}...")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                cells_list.append(data)

    return cells_list, images






if __name__ == "__main__":

    # asyncio.run(gen_concat_output())

    # directorys = extract_and_print_headers_with_bbox()



    asyncio.run(gen_page_output())
    cells_list, images = asyncio.run(extract_header_images_from_pdf())
    dc = DirectoryCleaner()
    asyncio.run(dc.reset_header_level(cells_list, images))
