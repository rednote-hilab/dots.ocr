"""
Layout Inference Web Application with Gradio - Annotation Version

A Gradio-based layout inference tool that supports image uploads and multiple backend inference engines.
This version adds an image annotation feature, allowing users to draw bounding boxes on an image and send both the image and the boxes to the model.
"""

import gradio as gr
import json
import os
import io
import tempfile
import base64
import zipfile
import uuid
import re
from pathlib import Path
from PIL import Image
import requests
from gradio_image_annotation import image_annotator

# Local utility imports
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.demo_utils.display import read_image
from dots_ocr.utils.doc_utils import load_images_from_pdf

# Add DotsOCRParser import
from dots_ocr.parser import DotsOCRParser

# ==================== Configuration ====================
DEFAULT_CONFIG = {
    'ip': "127.0.0.1",
    'port_vllm': 8000,
    'min_pixels': MIN_PIXELS,
    'max_pixels': MAX_PIXELS,
    'test_images_dir': "./assets/showcase_origin",
}

# ==================== Global Variables ====================
# Store the current configuration
current_config = DEFAULT_CONFIG.copy()

# Create a DotsOCRParser instance
dots_parser = DotsOCRParser(
    ip=DEFAULT_CONFIG['ip'],
    port=DEFAULT_CONFIG['port_vllm'],
    dpi=200,
    min_pixels=DEFAULT_CONFIG['min_pixels'],
    max_pixels=DEFAULT_CONFIG['max_pixels']
)

# Store processing results
processing_results = {
    'original_image': None,
    'processed_image': None,
    'layout_result': None,
    'markdown_content': None,
    'cells_data': None,
    'temp_dir': None,
    'session_id': None,
    'result_paths': None,
    'annotation_data': None  # Store annotation data
}

# ==================== Utility Functions ====================
def read_image_v2(img):
    """Reads an image, supporting URLs and local paths."""
    if isinstance(img, str) and img.startswith(("http://", "https://")):
        with requests.get(img, stream=True) as response:
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
    elif isinstance(img, str):
        img, _, _ = read_image(img, use_native=True)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise ValueError(f"Invalid image type: {type(img)}")
    return img

def get_test_images():
    """Gets the list of test images."""
    test_images = []
    test_dir = current_config['test_images_dir']
    if os.path.exists(test_dir):
        test_images = [os.path.join(test_dir, name) for name in os.listdir(test_dir) 
                      if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return test_images

def create_temp_session_dir():
    """Creates a unique temporary directory for each processing request."""
    session_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(tempfile.gettempdir(), f"dots_ocr_demo_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir, session_id

def parse_image_with_bbox(parser, image, prompt_mode, bbox=None, fitz_preprocess=False):
    """
    Processes an image using DotsOCRParser, with support for the bbox parameter.
    """
    # Create a temporary session directory
    temp_dir, session_id = create_temp_session_dir()
    
    try:
        # Save the PIL Image to a temporary file
        temp_image_path = os.path.join(temp_dir, f"input_{session_id}.png")
        image.save(temp_image_path, "PNG")
        
        # Use the high-level parse_image interface, passing the bbox parameter
        filename = f"demo_{session_id}"
        results = parser.parse_image(
            input_path=temp_image_path,
            filename=filename, 
            prompt_mode=prompt_mode,
            save_dir=temp_dir,
            bbox=bbox,
            fitz_preprocess=fitz_preprocess
        )
        
        # Parse the results
        if not results:
            raise ValueError("No results returned from parser")
        
        result = results[0]  # parse_image returns a list with a single result
        
        # Read the result files
        layout_image = None
        cells_data = None
        md_content = None
        filtered = False
        
        # Read the layout image
        if 'layout_image_path' in result and os.path.exists(result['layout_image_path']):
            layout_image = Image.open(result['layout_image_path'])
        
        # Read the JSON data
        if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
            with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                cells_data = json.load(f)
        
        # Read the Markdown content
        if 'md_content_path' in result and os.path.exists(result['md_content_path']):
            with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                md_content = f.read()
        
        # Check for the original response file (if JSON parsing fails)
        if 'filtered' in result:
            filtered = result['filtered']
        
        return {
            'layout_image': layout_image,
            'cells_data': cells_data,
            'md_content': md_content,
            'filtered': filtered,
            'temp_dir': temp_dir,
            'session_id': session_id,
            'result_paths': result
        }
        
    except Exception as e:
        # Clean up the temporary directory on error
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def process_annotation_data(annotation_data):
    """Processes annotation data, converting it to the format required by the model."""
    if not annotation_data or not annotation_data.get('boxes'):
        return None, None
    
    # Get image and box data
    image = annotation_data.get('image')
    boxes = annotation_data.get('boxes', [])
    
    if not boxes:
        return image, None
    
    # Ensure the image is in PIL Image format
    if image is not None:
        import numpy as np
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            # If it's another format, try to convert it
            try:
                image = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
            except Exception as e:
                print(f"Image format conversion failed: {e}")
                return None, None
    
    # Get the coordinate information of the box (only one box)
    box = boxes[0]
    bbox = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
    
    return image, bbox

# ==================== Core Processing Function ====================
def process_image_inference_with_annotation(annotation_data, test_image_input,
                          prompt_mode, server_ip, server_port, min_pixels, max_pixels,
                          fitz_preprocess=False
                          ):
    """Core function for image inference, supporting annotation data."""
    global current_config, processing_results, dots_parser
    
    # First, clean up previous processing results
    if processing_results.get('temp_dir') and os.path.exists(processing_results['temp_dir']):
        import shutil
        try:
            shutil.rmtree(processing_results['temp_dir'], ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up previous temporary directory: {e}")
    
    # Reset processing results
    processing_results = {
        'original_image': None,
        'processed_image': None,
        'layout_result': None,
        'markdown_content': None,
        'cells_data': None,
        'temp_dir': None,
        'session_id': None,
        'result_paths': None,
        'annotation_data': annotation_data
    }
    
    # Update configuration
    current_config.update({
        'ip': server_ip,
        'port_vllm': server_port,
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    })
    
    # Update parser configuration
    dots_parser.ip = server_ip
    dots_parser.port = server_port
    dots_parser.min_pixels = min_pixels
    dots_parser.max_pixels = max_pixels
    
    # Determine the input source and process annotation data
    image = None
    bbox = None
    
    # Prioritize processing annotation data
    if annotation_data and annotation_data.get('image') is not None:
        image, bbox = process_annotation_data(annotation_data)
        if image is not None:
            # If there's a bbox, force the use of 'prompt_grounding_ocr' mode
            assert bbox is not None
            prompt_mode = "prompt_grounding_ocr"
    
    # If there's no annotation data, check the test image input
    if image is None and test_image_input and test_image_input != "":
        try:
            image = read_image_v2(test_image_input)
        except Exception as e:
            return None, f"Failed to read test image: {e}", "", "", gr.update(value=None), ""
    
    if image is None:
        return None, "Please select a test image or add an image in the annotation component", "", "", gr.update(value=None), ""
    if bbox is None:
        return None, "Please select a bounding box by mouse", "Please select a bounding box by mouse", "", "", gr.update(value=None)
    
    try:
        # Process using DotsOCRParser, passing the bbox parameter
        original_image = image
        parse_result = parse_image_with_bbox(dots_parser, image, prompt_mode, bbox, fitz_preprocess)
        
        # Extract parsing results
        layout_image = parse_result['layout_image']
        cells_data = parse_result['cells_data']
        md_content = parse_result['md_content']
        filtered = parse_result['filtered']
        
        # Store the results
        processing_results.update({
            'original_image': original_image,
            'processed_image': None,
            'layout_result': layout_image,
            'markdown_content': md_content,
            'cells_data': cells_data,
            'temp_dir': parse_result['temp_dir'],
            'session_id': parse_result['session_id'],
            'result_paths': parse_result['result_paths'],
            'annotation_data': annotation_data
        })
        
        # Handle the case where parsing fails
        if filtered:
            info_text = f"""
**Image Information:**
- Original Dimensions: {original_image.width} x {original_image.height}
- Processing Mode: {'Region OCR' if bbox else 'Full Image OCR'}
- Processing Status: JSON parsing failed, using cleaned text output
- Server: {current_config['ip']}:{current_config['port_vllm']}
- Session ID: {parse_result['session_id']}
- Box Coordinates: {bbox if bbox else 'None'}
            """
            
            return (
                md_content or "No markdown content generated",
                info_text,
                md_content or "No markdown content generated",
                md_content or "No markdown content generated",
                gr.update(visible=False),
                ""
            )
        
        # Handle the case where JSON parsing succeeds
        num_elements = len(cells_data) if cells_data else 0
        info_text = f"""
**Image Information:**
- Original Dimensions: {original_image.width} x {original_image.height}
- Processing Mode: {'Region OCR' if bbox else 'Full Image OCR'}
- Server: {current_config['ip']}:{current_config['port_vllm']}
- Detected {num_elements} layout elements
- Session ID: {parse_result['session_id']}
- Box Coordinates: {bbox if bbox else 'None'}
        """
        
        # Current page JSON output
        current_json = ""
        if cells_data:
            try:
                current_json = json.dumps(cells_data, ensure_ascii=False, indent=2)
            except:
                current_json = str(cells_data)
        
        # Create a downloadable ZIP file
        download_zip_path = None
        if parse_result['temp_dir']:
            download_zip_path = os.path.join(parse_result['temp_dir'], f"layout_results_{parse_result['session_id']}.zip")
            try:
                with zipfile.ZipFile(download_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(parse_result['temp_dir']):
                        for file in files:
                            if file.endswith('.zip'):
                                continue
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, parse_result['temp_dir'])
                            zipf.write(file_path, arcname)
            except Exception as e:
                print(f"Failed to create download ZIP: {e}")
                download_zip_path = None
        
        return (
            md_content or "No markdown content generated",
            info_text,
            md_content or "No markdown content generated",
            md_content or "No markdown content generated",
            gr.update(value=download_zip_path, visible=True) if download_zip_path else gr.update(visible=False),
            current_json
        )
        
    except Exception as e:
        return f"An error occurred during processing: {e}", f"An error occurred during processing: {e}", "", "", gr.update(value=None), ""

def load_image_to_annotator(test_image_input):
    """Loads an image into the annotation component."""
    image = None
    
    # Check the test image input
    if test_image_input and test_image_input != "":
        try:
            image = read_image_v2(test_image_input)
        except Exception as e:
            return None
    
    if image is None:
        return None
    
    # Return the format required by the annotation component
    return {
        "image": image,
        "boxes": []
    }

def clear_all_data():
    """Clears all data."""
    global processing_results
    
    # Clean up the temporary directory
    if processing_results.get('temp_dir') and os.path.exists(processing_results['temp_dir']):
        import shutil
        try:
            shutil.rmtree(processing_results['temp_dir'], ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up temporary directory: {e}")
    
    # Reset processing results
    processing_results = {
        'original_image': None,
        'processed_image': None,
        'layout_result': None,
        'markdown_content': None,
        'cells_data': None,
        'temp_dir': None,
        'session_id': None,
        'result_paths': None,
        'annotation_data': None
    }
    
    return (
        "",    # Clear test image selection
        None,  # Clear annotation component
        "Waiting for processing results...",  # Reset info display
        "## Waiting for processing results...",  # Reset Markdown display
        "🕐 Waiting for parsing results...",    # Clear raw Markdown text
        gr.update(visible=False),  # Hide download button
        "🕐 Waiting for parsing results..."     # Clear JSON
    )

def update_prompt_display(prompt_mode):
    """Updates the displayed prompt content."""
    return dict_promptmode_to_prompt[prompt_mode]

# ==================== Gradio Interface ====================
def create_gradio_interface():
    """Creates the Gradio interface."""
    
    # CSS styling to match the reference style
    css = """
    footer {
        visibility: hidden;
    }
    
    #info_box {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
        font-size: 14px;
    }
    
    #markdown_tabs {
        height: 100%;
    }
    
    #annotation_component {
        border-radius: 8px;
    }
    """
    
    with gr.Blocks(theme="ocean", css=css, title='dots.ocr - Annotation') as demo:
        
        # Title
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2em;">🔍 dots.ocr - Annotation Version</h1>
            </div>
            <div style="text-align: center; margin-bottom: 10px;">
                <em>Supports image annotation, drawing boxes, and sending box information to the model for OCR.</em>
            </div>
        """)
        
        with gr.Row():
            # Left side: Input and Configuration
            with gr.Column(scale=1, variant="compact"):
                gr.Markdown("### 📁 Select Example")
                test_images = get_test_images()
                test_image_input = gr.Dropdown(
                    label="Select Example",
                    choices=[""] + test_images,
                    value="",
                    show_label=True
                )
                
                # Button to load image into the annotation component
                load_btn = gr.Button("📷 Load Image to Annotation Area", variant="secondary")
                
                prompt_mode = gr.Dropdown(
                    label="Select Prompt",
                    # choices=["prompt_layout_all_en", "prompt_layout_only_en", "prompt_ocr", "prompt_grounding_ocr"],
                    choices=["prompt_grounding_ocr"],
                    value="prompt_grounding_ocr",
                    show_label=True,
                    info="If a box is drawn, 'prompt_grounding_ocr' mode will be used automatically."
                )
                
                # Display the current prompt content
                prompt_display = gr.Textbox(
                    label="Current Prompt Content",
                    # value=dict_promptmode_to_prompt[list(dict_promptmode_to_prompt.keys())[0]],
                    value=dict_promptmode_to_prompt["prompt_grounding_ocr"],
                    lines=4,
                    max_lines=8,
                    interactive=False,
                    show_copy_button=True
                )
                
                gr.Markdown("### ⚙️ Actions")
                process_btn = gr.Button("🔍 Parse", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("### 🛠️ Configuration")

                fitz_preprocess = gr.Checkbox(
                    label="Enable fitz_preprocess", 
                    value=False,
                    info="Performs fitz preprocessing on the image input, converting the image to a PDF and then to a 200dpi image."
                )
                
                with gr.Row():
                    server_ip = gr.Textbox(
                        label="Server IP",
                        value=DEFAULT_CONFIG['ip']
                    )
                    server_port = gr.Number(
                        label="Port",
                        value=DEFAULT_CONFIG['port_vllm'],
                        precision=0
                    )
                
                with gr.Row():
                    min_pixels = gr.Number(
                        label="Min Pixels",
                        value=DEFAULT_CONFIG['min_pixels'],
                        precision=0
                    )
                    max_pixels = gr.Number(
                        label="Max Pixels", 
                        value=DEFAULT_CONFIG['max_pixels'],
                        precision=0
                    )
            
            # Right side: Result Display
            with gr.Column(scale=6, variant="compact"):
                with gr.Row():
                    # Image Annotation Area
                    with gr.Column(scale=3):
                        gr.Markdown("### 🎯 Image Annotation Area")
                        gr.Markdown("""
                        **Instructions:**
                        - Method 1: Select an example image on the left and click "Load Image to Annotation Area".
                        - Method 2: Upload an image directly in the annotation area below (drag and drop or click to upload).
                        - Use the mouse to draw a box on the image to select the region for recognition.
                        - Only one box can be drawn. To draw a new one, please delete the old one first.
                        - **Hotkey: Press the Delete key to remove the selected box.**
                        - After drawing a box, clicking Parse will automatically use the Region OCR mode.
                        """)
                        
                        annotator = image_annotator(
                            value=None,
                            label="Image Annotation",
                            height=600,
                            show_label=False,
                            elem_id="annotation_component",
                            single_box=True,  # Only allow one box; a new box will replace the old one
                            box_min_size=10,
                            interactive=True,
                            disable_edit_boxes=True,  # Disable the edit dialog
                            label_list=["OCR Region"],  # Set the default label
                            label_colors=[(255, 0, 0)],  # Set color to red
                            use_default_label=True,  # Use the default label
                            image_type="pil"  # Ensure it returns a PIL Image format
                        )
                        
                        # Information Display
                        info_display = gr.Markdown(
                            "Waiting for processing results...",
                            elem_id="info_box"
                        )
                    
                    # Result Display Area
                    with gr.Column(scale=3):
                        gr.Markdown("### ✅ Results")
                        
                        with gr.Tabs(elem_id="markdown_tabs"):
                            with gr.TabItem("Markdown Rendered View"):
                                md_output = gr.Markdown(
                                    "## Please upload an image and click the Parse button for recognition...",
                                    label="Markdown Preview",
                                    max_height=1000,
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False},
                                    ],
                                    show_copy_button=False,
                                    elem_id="markdown_output"
                                )
                            
                            with gr.TabItem("Markdown Raw Text"):
                                md_raw_output = gr.Textbox(
                                    value="🕐 Waiting for parsing results...",
                                    label="Markdown Raw Text",
                                    max_lines=100,
                                    lines=38,
                                    show_copy_button=True,
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                            
                            with gr.TabItem("JSON Result"):
                                json_output = gr.Textbox(
                                    value="🕐 Waiting for parsing results...",
                                    label="JSON Result",
                                    max_lines=100,
                                    lines=38,
                                    show_copy_button=True,
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                
                # Download Button
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        "⬇️ Download Results",
                        visible=False
                    )
        
        # Event Binding
        
        # When the prompt mode changes, update the displayed content
        prompt_mode.change(
            fn=update_prompt_display,
            inputs=prompt_mode,
            outputs=prompt_display,
            show_progress=False
        )
        
        # Load image into the annotation component
        load_btn.click(
            fn=load_image_to_annotator,
            inputs=[test_image_input],
            outputs=annotator,
            show_progress=False
        )
        
        # Process Inference
        process_btn.click(
            fn=process_image_inference_with_annotation,
            inputs=[
                annotator, test_image_input,
                prompt_mode, server_ip, server_port, min_pixels, max_pixels, 
                fitz_preprocess
            ],
            outputs=[
                md_output, info_display, md_raw_output, md_raw_output,
                download_btn, json_output
            ],
            show_progress=True
        )
        
        # Clear Data
        clear_btn.click(
            fn=clear_all_data,
            outputs=[
                test_image_input, annotator,
                info_display, md_output, md_raw_output,
                download_btn, json_output
            ],
            show_progress=False
        )
    
    return demo

# ==================== Main Program ====================
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7861,  # Use a different port to avoid conflicts
        debug=True
    )
