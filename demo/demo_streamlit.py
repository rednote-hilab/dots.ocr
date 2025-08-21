"""
Layout Inference Web Application

A Streamlit-based layout inference tool that supports image uploads and multiple backend inference engines.
"""

import streamlit as st
import json
import os
import io
import tempfile
from PIL import Image
import requests

# Local utility imports

# from utils import infer

from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.format_transformer import layoutjson2md
from dots_ocr.utils.layout_utils import draw_layout_on_image, post_process_cells
from dots_ocr.utils.image_utils import get_input_dimensions, get_image_by_fitz_doc
from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS

import os
from PIL import Image
from dots_ocr.utils.demo_utils.display import read_image



# ==================== Configuration ====================
DEFAULT_CONFIG = {
    'ip': "127.0.0.1",
    'port_vllm': 8000,
    'min_pixels': MIN_PIXELS,
    'max_pixels': MAX_PIXELS,
    'test_images_dir': "./assets/showcase_origin",
}

# ==================== Utility Functions ====================


@st.cache_resource
def read_image_v2(img: str):
    if img.startswith(("http://", "https://")):
        with requests.get(img, stream=True) as response:
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))

    if isinstance(img, str):
        # img = transform_image_path(img)
        img, _, _ = read_image(img, use_native=True)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError(f"Invalid image type: {type(img)}")
    return img


# ==================== UI Components ====================
def create_config_sidebar():
    """Create configuration sidebar"""
    st.sidebar.header("Configuration Parameters")
    
    config = {}
    config['prompt_key'] = st.sidebar.selectbox("Prompt Mode", list(dict_promptmode_to_prompt.keys()))
    config['ip'] = st.sidebar.text_input("Server IP", DEFAULT_CONFIG['ip'])
    config['port'] = st.sidebar.number_input("Port", min_value=1000, max_value=9999, value=DEFAULT_CONFIG['port_vllm'])
    # config['eos_word'] = st.sidebar.text_input("EOS Word", DEFAULT_CONFIG['eos_word'])
    
    # Image configuration
    st.sidebar.subheader("Image Configuration")
    config['min_pixels'] = st.sidebar.number_input("Min Pixels", value=DEFAULT_CONFIG['min_pixels'])
    config['max_pixels'] = st.sidebar.number_input("Max Pixels", value=DEFAULT_CONFIG['max_pixels'])
    
    return config

def get_image_input():
    """Get image input"""
    st.markdown("#### Image Input")
    
    input_mode = st.pills(label="Select input method", options=["Upload Image", "Enter Image URL/Path", "Select Test Image"], key="input_mode", label_visibility="collapsed")

    if input_mode == "Upload Image":
        # File uploader
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
    elif input_mode == 'Enter Image URL/Path':
        # URL input
        img_url_input = st.text_input("Enter Image URL/Path")
        return img_url_input

    elif input_mode == 'Select Test Image':
        # Test image selection
        test_images = []
        test_dir = DEFAULT_CONFIG['test_images_dir']
        if os.path.exists(test_dir):
            test_images = [os.path.join(test_dir, name) for name in os.listdir(test_dir)]
        img_url_test = st.selectbox("Select Test Image", [""] + test_images)
        return img_url_test
    else:
        raise ValueError(f"Invalid input mode: {input_mode}")

    return None



def process_and_display_results(output: dict, image: Image.Image, config: dict):
    """Process and display inference results in a way that matches parser's refined logic"""
    prompt, response = output['prompt'], output['response']
    prompt_key = output.get('prompt_key')

    # Determine if this is a layout prompt
    layout_prompts = ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']
    is_layout = prompt_key in layout_prompts

    # Always show original image
    st.markdown('---')
    st.markdown('##### Original Image')
    st.image(image, width=image.width if image.width < 1000 else 1000)

    # Show server input dimensions for reference
    input_width, input_height = get_input_dimensions(
        image,
        min_pixels=config['min_pixels'],
        max_pixels=config['max_pixels']
    )
    st.write(f'Input Dimensions: {input_width} x {input_height}')

    if not is_layout:
        # Non-layout prompts: show raw text output only
        st.text_area('Model Output', response, height=300)
        return

    # Layout prompts: try to parse JSON and visualize only when there are cells
    try:
        cells = json.loads(response)
        cells = post_process_cells(
            image, cells,
            image.width, image.height,
            min_pixels=config['min_pixels'],
            max_pixels=config['max_pixels']
        )

        st.text_area('Original Model Output (JSON)', response, height=200)
        st.text_area('Post-processed Cells', str(cells), height=200)

        if isinstance(cells, list) and len(cells) > 0:
            col1, col2 = st.columns(2)
            with col1:
                new_image = draw_layout_on_image(
                    image, cells,
                    resized_height=None, resized_width=None,
                    fill_bbox=True, draw_bbox=True
                )
                st.markdown('##### Layout Visualization')
                st.image(new_image, width=new_image.width if new_image.width < 1000 else 1000)
            with col2:
                md_code = layoutjson2md(image, cells, text_key='text')
                st.markdown('##### Markdown Format')
                st.markdown(md_code, unsafe_allow_html=True)
        else:
            st.info('No layout detected. Skipping visualization.')
    except json.JSONDecodeError:
        # JSON invalid => align with parser: no layout image should be created
        st.warning('Model output is not valid JSON for a layout prompt. Skipping visualization.')
        st.text_area('Original Model Output', response, height=300)
    except Exception as e:
        st.error(f"Error processing results: {e}")

# ==================== Main Application ====================
def main():
    """Main application function"""
    st.set_page_config(page_title="Layout Inference Tool", layout="wide")
    st.title("🔍 Layout Inference Tool")
    
    # Configuration
    config = create_config_sidebar()
    prompt = dict_promptmode_to_prompt[config['prompt_key']]
    st.sidebar.info(f"Current Prompt: {prompt}")
    
    # Image input
    img_url = get_image_input()
    start_button = st.button('🚀 Start Inference', type="primary")
    
    if img_url is not None and img_url.strip() != "":
        try:
            # processed_image = read_image_v2(img_url)
            origin_image = read_image_v2(img_url)
            st.write(f"Original Dimensions: {origin_image.width} x {origin_image.height}")
            # processed_image = get_image_by_fitz_doc(origin_image, target_dpi=200)
            processed_image = origin_image
        except Exception as e:
            st.error(f"Failed to read image: {e}")
            return
    else:
        st.info("Please enter an image URL/path or upload an image")
        return

    output = None
    # Inference button
    if start_button:
        with st.spinner(f"Inferring... Server: {config['ip']}:{config['port']}"):
            
            response = inference_with_vllm(
                processed_image, prompt, config['ip'], config['port'],
            )
            output = {
                'prompt': prompt,
                'prompt_key': config['prompt_key'],
                'response': response,
            }
    else:
        st.image(processed_image, width=500)

    # Process results
    if output:
        process_and_display_results(output, processed_image, config)

if __name__ == "__main__":
    main()
