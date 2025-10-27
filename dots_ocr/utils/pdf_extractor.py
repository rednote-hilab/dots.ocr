import fitz
import os
import json
from PIL import Image
import re

class PdfExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_document = fitz.open(pdf_path)

    @property
    def is_structured(self):
        return len(self.pdf_document.get_toc()) > 0
    @property
    def num_pages(self):
        return self.pdf_document.page_count
    def page_size(self, page_no: int) -> int:
        rect = self.pdf_document[page_no].rect
        return rect.width, rect.height

    #TODO(zihao): fitz can get more infomation about the pdf structure, explore later
    # italic / bold / font size etc.
    def extract_text(
        self,
        page_no,
        bbox : tuple = None
    ) -> str:
        page = self.pdf_document[page_no]
        if bbox:
            rect = fitz.Rect(bbox)
            text = page.get_text("text", clip=rect)
        else:
            text = page.get_text("text")
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text) 
        return text.strip()

    def page_to_image(
        self,
        page_no: int,
        dpi: int = 72, # 72 is pdf default. dpi can only be 72 if layoutjson2md(used in save_result) we save crops instead of the descriptions.
    ) -> Image.Image:
        if page_no < 0 or page_no >= self.num_pages:
            raise ValueError(f"Page number {page_no} out of range [1, {self.num_pages}]")
    
        page = self.pdf_document[page_no]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        return img