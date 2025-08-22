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


def fitz_doc_to_image(page, target_dpi=200, max_side=4500, max_pixels=None) -> dict:
    """Convert a PyMuPDF page to a NumPy-compatible image array.

    This function renders a single `fitz.Page` object to an image with a
    target DPI, while ensuring constraints on maximum side length and
    maximum pixel count are respected.

    Args:
        page (fitz.Page): A PyMuPDF page object to render.
        target_dpi (int, optional): Desired resolution in DPI.
            Defaults to 200.
        max_side (int, optional): Maximum allowed width or height (in pixels)
            of the rendered image. Defaults to 4500.
        max_pixels (int, optional): Maximum allowed total number of pixels
            in the rendered image. If provided, the image will be scaled down
            to respect this constraint. Defaults to None.

    Returns:
        PIL.Image.Image: The rendered page as a PIL Image object.

    Raises:
        ValueError: If the input `page` is not a `fitz.Page`.
    """
    from PIL import Image
    # base zoom for requested DPI
    zoom = target_dpi / 72.0

    # predict size at requested DPI
    w0 = page.rect.width * zoom
    h0 = page.rect.height * zoom

    # compute an extra scale factor s to stay within limits
    s = 1.0
    if max_side:
        s = min(s, max_side / max(w0, h0))
    if max_pixels:
        s = min(s, (max_pixels / (w0 * h0)) ** 0.5)

    # don’t upscale beyond requested dpi
    s = min(s, 1.0)

    mat = fitz.Matrix(zoom * s, zoom * s)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)


def load_images_from_pdf(pdf_file, dpi=200, start_page_id=0, end_page_id=None) -> list:
    images = []
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
                img = fitz_doc_to_image(page, target_dpi=dpi)
                images.append(img)
    return images
