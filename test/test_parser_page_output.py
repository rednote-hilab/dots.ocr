import asyncio
import os
import json
import re
from concurrent.futures import ProcessPoolExecutor
from dots_ocr.utils.doc_utils import load_images_from_pdf
from PIL import Image
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.directory_cleaner import CropImage, SectionHeader, DirectoryStructure, Reranker

async def main():
    dots_parser = DotsOCRParser()
    await dots_parser.parse_pdf_rebuild_directory(
        "/dots.ocr/test/data/PGhandbook.pdf",
        "prompt_layout_all_en"
    )

if __name__ == "__main__":
    asyncio.run(main())