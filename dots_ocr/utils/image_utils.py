import math
import base64
from PIL import Image
from typing import Tuple
import os
from dots_ocr.utils.consts import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.doc_utils import fitz_doc_to_image
from io import BytesIO
import fitz
import requests
import copy


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 11289600,
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
        if h_bar * w_bar > max_pixels:  # max_pixels first to control the token length 
            beta = math.sqrt((h_bar * w_bar) / max_pixels)
            h_bar = max(factor, floor_by_factor(h_bar / beta, factor))
            w_bar = max(factor, floor_by_factor(w_bar / beta, factor))
    return h_bar, w_bar



def PILimage_to_base64(image, format='PNG'):
    buffered = BytesIO()
    image.save(buffered, format=format)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{base64_str}"


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


# copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
def fetch_image(
        image, 
        min_pixels=None,
        max_pixels=None,
        resized_height=None,
        resized_width=None,
    ) -> Image.Image:
    assert image is not None, f"image not found, maybe input format error: {image}"
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=IMAGE_FACTOR,
        )
        assert resized_height>0 and resized_width>0, f"resized_height: {resized_height}, resized_width: {resized_width}, min_pixels: {min_pixels}, max_pixels:{max_pixels}, width: {width}, height:{height}, "
        image = image.resize((resized_width, resized_height))
    elif min_pixels or max_pixels:
        width, height = image.size
        if not min_pixels:
            min_pixels = MIN_PIXELS
        if not max_pixels:
            max_pixels = MAX_PIXELS
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        assert resized_height>0 and resized_width>0, f"resized_height: {resized_height}, resized_width: {resized_width}, min_pixels: {min_pixels}, max_pixels:{max_pixels}, width: {width}, height:{height}, "
        image = image.resize((resized_width, resized_height))

    return image

def get_input_dimensions(
    image: Image.Image,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28
) -> Tuple[int, int]:
    """
    Gets the resized dimensions of the input image.
    
    Args:
        image: The original image.
        min_pixels: The minimum number of pixels.
        max_pixels: The maximum number of pixels.
        factor: The resizing factor.
        
    Returns:
        The resized (width, height).
    """
    input_height, input_width = smart_resize(
        image.height, 
        image.width,
        factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
    return input_width, input_height


def get_image_by_fitz_doc(image, target_dpi=200):
    # get image through fitz, to get target dpi image, mainly for higher image
    if not isinstance(image, Image.Image):
        assert isinstance(image, str)
        _, file_ext = os.path.splitext(image)
        assert file_ext in {'.jpg', '.jpeg', '.png'}

        if image.startswith("http://") or image.startswith("https://"):
            with requests.get(image, stream=True) as response:
                response.raise_for_status()
                data_bytes = response.content
        else:
            with open(image, 'rb') as f:
                data_bytes = f.read()

        image = Image.open(BytesIO(data_bytes))
    else:
        data_bytes = BytesIO()
        image.save(data_bytes, format='PNG')

    origin_dpi = image.info.get('dpi', None)
    pdf_bytes = fitz.open(stream=data_bytes).convert_to_pdf()
    doc = fitz.open('pdf', pdf_bytes)
    page = doc[0]
    image_fitz, scale_factor = fitz_doc_to_image(page, target_dpi=target_dpi, origin_dpi=origin_dpi)

    return image_fitz
