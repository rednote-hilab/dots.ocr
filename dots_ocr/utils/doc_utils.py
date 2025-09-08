import fitz
import numpy as np
import enum
from pydantic import BaseModel, Field
from PIL import Image


class SupportedPdfParseMethod(enum.Enum):
    OCR = 'ocr'
    TXT = 'txt'


class PageInfo(BaseModel):
    """The width and height of page
    """
    w: float = Field(description='the width of page')
    h: float = Field(description='the height of page')

def get_pdf_page_count_fitz(pdf_path: str) -> int:
    try:
        with fitz.open(pdf_path) as doc:
            return doc.page_count
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return 0

def fitz_doc_to_image(doc, target_dpi=200, origin_dpi=None) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        doc (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    from PIL import Image
    mat = fitz.Matrix(target_dpi / 72, target_dpi / 72)
    pm = doc.get_pixmap(matrix=mat, alpha=False)

    if pm.width > 4500 or pm.height > 4500:
        mat = fitz.Matrix(72 / 72, 72 / 72)  # use fitz default dpi
        pm = doc.get_pixmap(matrix=mat, alpha=False)

    image = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    return image, target_dpi / 72


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    images = []
    scale_factors = []
    with fitz.open(pdf_file) as doc:
        pdf_page_num = doc.page_count
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            print('end_page_id is out of range, use images length')
            end_page_id = pdf_page_num - 1

        for index in range(0, doc.page_count):
            if start_page_id <= index <= end_page_id:
                page = doc[index]
                img, scale_factor = fitz_doc_to_image(page, target_dpi=dpi)
                images.append(img)
                scale_factors.append(scale_factor)
    return images, scale_factors

def iter_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None, existing_pages=set()):
    with fitz.open(pdf_file) as doc:
        pdf_page_num = doc.page_count
        end_page_id = (
            end_page_id
            if end_page_id is not None and end_page_id >= 0
            else pdf_page_num - 1
        )
        if end_page_id > pdf_page_num - 1:
            print("end_page_id is out of range, use max page index")
            end_page_id = pdf_page_num - 1

        for index in range(start_page_id, end_page_id + 1):
            if index in existing_pages:
                continue
            page = doc[index]
            img, scale_factor = fitz_doc_to_image(page, target_dpi=dpi)
            yield index, img, scale_factor