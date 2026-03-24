"""
Layout Inference Web Application with Gradio

A Gradio-based layout inference tool that supports image uploads and multiple backend inference engines.
It adopts a reference-style interface design while preserving the original inference logic.
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
import shutil # Import shutil for cleanup

# Local tool imports
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
    'test_images_dir': "./assets/showcase/origin",
}

# ==================== Multi-Model Server Configuration ====================
MODEL_SERVERS = {
    "dots.mocr": {
        'ip': "127.0.0.1",
        'port_vllm': 8000,
        'description': "dots.mocr"
    },
    "dots.mocr-svg": {
        'ip': "127.0.0.1",
        'port_vllm': 8000,  # 请根据实际情况修改端口
        'description': "dots.mocr-svg"
    },
}



#每个prompt的预处理写死
PROMPT_TO_FITZ_PREPROCESS = {
    "prompt_layout_all_en": True,      # 文档布局分析 - 启用预处理
    "prompt_layout_only_en": True,    # 仅布局检测 - 启用预处理
    "prompt_ocr": True,               # 仅文字识别 - 启用预处理
    "prompt_web_parsing": False,       # 网页解析 - 禁用预处理
    "prompt_scene_spotting": False,   # 场景检测 - 禁用预处理
    "prompt_image_to_svg": False,     # SVG 转换 - 禁用预处理
    "prompt_general": False,          # 自由问答 - 禁用预处理
}

#不同任务需要不同temperature
PROMPT_TO_TEMPERATURE = {
    "prompt_layout_all_en": 0.1,      # 文档布局分析 - 低温度，更确定性
    "prompt_layout_only_en": 0.1,     # 仅布局检测 - 低温度
    "prompt_ocr": 0.1,                # OCR 识别 - 低温度
    "prompt_web_parsing": 0.1,        # 网页解析 - 稍高一点
    "prompt_scene_spotting": 0.1,     # 场景检测 - 中等温度
    "prompt_image_to_svg": 0.9,       # SVG 转换 - 较低温度
    "prompt_general": 0.1,            # 自由问答 - 高温度，更有创造性
}

# 不同prompt_mode对应的模型
PROMPT_TO_MODEL = {
    "prompt_image_to_svg": "dots.mocr-svg",  # SVG任务使用SVG模型
}

# ==================== Demo Case Configuration ====================
# 根据文件名自动选择 prompt_mode 和预设的 custom_prompt
DEMO_CASE_CONFIG = {
    # 格式: "文件名关键字": {"prompt_mode": "xxx", "custom_prompt": "xxx"}
    
    # 布局分析类
    "doc": {"prompt_mode": "prompt_layout_all_en"},
    "formula": {"prompt_mode": "prompt_layout_all_en"},
    "table": {"prompt_mode": "prompt_layout_all_en"},
    
    # 仅布局检测
    "detect": {"prompt_mode": "prompt_layout_only_en"},
    # OCR 识别
    "ocr": {"prompt_mode": "prompt_ocr"},

    # 网页解析
    "webpage": {"prompt_mode": "prompt_web_parsing"},

    # 场景文字检测
    "scene": {"prompt_mode": "prompt_scene_spotting"},

    # SVG 转换
    "svg": {"prompt_mode": "prompt_image_to_svg"},

    # QA 任务（带预设 prompt）
    "general_qa": {
        "prompt_mode": "prompt_general", 
        "custom_prompt": "Across panels 1-12 plotting against clean accuracy, which variable appears most positively correlated with clean accuracy?"
    },


}

# 默认配置（找不到匹配时使用）
DEFAULT_DEMO_CONFIG = {"prompt_mode": "prompt_layout_all_en"}

def get_config_for_file(file_path):
    """
    根据文件名自动匹配 prompt_mode 和 custom_prompt
    支持部分匹配（文件名包含关键字即可）
    """
    if not file_path:
        return DEFAULT_DEMO_CONFIG.copy()
    
    filename = os.path.basename(file_path).lower()
    
    # 遍历配置字典，查找匹配的关键字
    for keyword, config in DEMO_CASE_CONFIG.items():
        if keyword.lower() in filename:
            return config.copy()
    
    # 没有匹配则返回默认配置
    return DEFAULT_DEMO_CONFIG.copy()

# ==================== Global Variables ====================
# Store current configuration
current_config = DEFAULT_CONFIG.copy()

# Parser cache for multiple models
_parser_cache = {}

def get_parser(model_name: str, min_pixels: int = None, max_pixels: int = None) -> DotsMOCRParser:
    """
    Get or create a parser instance for the specified model.
    Uses cache to avoid recreating parsers for the same model.
    """
    if model_name not in MODEL_SERVERS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = MODEL_SERVERS[model_name]
    
    # Create cache key based on model and pixel settings
    cache_key = model_name
    
    # If parser exists in cache, update its settings and return
    if cache_key in _parser_cache:
        parser = _parser_cache[cache_key]
        parser.min_pixels = min_pixels or DEFAULT_CONFIG['min_pixels']
        parser.max_pixels = max_pixels or DEFAULT_CONFIG['max_pixels']
        return parser
    
    # Create new parser instance
    parser = DotsMOCRParser(
        ip=model_config['ip'],
        port=model_config['port_vllm'],
        dpi=200,
        min_pixels=min_pixels or DEFAULT_CONFIG['min_pixels'],
        max_pixels=max_pixels or DEFAULT_CONFIG['max_pixels']
    )
    _parser_cache[cache_key] = parser
    return parser

def get_initial_session_state():
    return {
        'processing_results': {
            'original_image': None,
            'processed_image': None,
            'layout_result': None,
            'markdown_content': None,
            'cells_data': None,
            'temp_dir': None,
            'session_id': None,
            'result_paths': None,
            'pdf_results': None
        },
        'pdf_cache': {
            "images": [],
            "current_page": 0,
            "total_pages": 0,
            "file_type": None,
            "is_parsed": False,
            "results": []
        },
        'auto_custom_prompt': None,
    }

def read_image_v2(img):
    """Reads an image, supports URLs and local paths"""
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

def load_file_for_preview(file_path, session_state):
    """Loads a file for preview, supports PDF and image files"""
    pdf_cache = session_state['pdf_cache']
    
    if not file_path or not os.path.exists(file_path):
        return None, "<div id='page_info_box'>0 / 0</div>", session_state
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            pages = load_images_from_pdf(file_path)
            pdf_cache["file_type"] = "pdf"
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            image = Image.open(file_path)
            pages = [image]
            pdf_cache["file_type"] = "image"
        else:
            return None, "<div id='page_info_box'>Unsupported file format</div>", session_state
    except Exception as e:
        return None, f"<div id='page_info_box'>PDF loading failed: {str(e)}</div>", session_state
    
    pdf_cache["images"] = pages
    pdf_cache["current_page"] = 0
    pdf_cache["total_pages"] = len(pages)
    pdf_cache["is_parsed"] = False
    pdf_cache["results"] = []
    
    return pages[0], f"<div id='page_info_box'>1 / {len(pages)}</div>", session_state

def on_test_image_select(file_path, session_state):
    """选择测试图片时的回调：加载预览 + 自动设置 prompt_mode + 自动切换模型"""
    preview_image, page_info, session_state = load_file_for_preview(file_path, session_state)
    
    if not file_path:
        return (
            preview_image, 
            page_info, 
            session_state,
            gr.update(), 
            gr.update(), 
            gr.update()
        )
    
    auto_config = get_config_for_file(file_path)
    prompt_mode_value = auto_config["prompt_mode"]
    custom_prompt_value = auto_config.get("custom_prompt", "")
    
    session_state['auto_custom_prompt'] = custom_prompt_value if custom_prompt_value else None
    
    is_free_qa = prompt_mode_value == 'prompt_general'
    if is_free_qa and custom_prompt_value:
        prompt_text = custom_prompt_value
    else:
        prompt_text = update_prompt_display(prompt_mode_value)
    
    # 根据prompt_mode自动选择模型
    auto_model = PROMPT_TO_MODEL.get(prompt_mode_value, list(MODEL_SERVERS.keys())[0])
    
    return (
        preview_image, 
        page_info, 
        session_state,
        gr.update(value=prompt_mode_value),
        gr.update(value=prompt_text, interactive=is_free_qa),
        gr.update(value=auto_model),
    )



def turn_page(direction, session_state):
    """Page turning function"""
    pdf_cache = session_state['pdf_cache']
    
    if not pdf_cache["images"]:
        return None, "<div id='page_info_box'>0 / 0</div>", "", session_state

    if direction == "prev":
        pdf_cache["current_page"] = max(0, pdf_cache["current_page"] - 1)
    elif direction == "next":
        pdf_cache["current_page"] = min(pdf_cache["total_pages"] - 1, pdf_cache["current_page"] + 1)

    index = pdf_cache["current_page"]
    current_image = pdf_cache["images"][index]  # Use the original image by default
    page_info = f"<div id='page_info_box'>{index + 1} / {pdf_cache['total_pages']}</div>"
    
    current_json = ""
    if pdf_cache["is_parsed"] and index < len(pdf_cache["results"]):
        result = pdf_cache["results"][index]
        if 'cells_data' in result and result['cells_data']:
            try:
                current_json = json.dumps(result['cells_data'], ensure_ascii=False, indent=2)
            except:
                current_json = str(result.get('cells_data', ''))
        if 'layout_image' in result and result['layout_image']:
            current_image = result['layout_image']
    
    return current_image, page_info, current_json, session_state

def get_test_images():
    """Gets the list of test images"""
    test_images = []
    test_dir = current_config['test_images_dir']
    if os.path.exists(test_dir):
        test_images = sorted([
            os.path.join(test_dir, name) 
            for name in os.listdir(test_dir) 
            if name.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))
        ])
    return test_images

def create_temp_session_dir():
    """Creates a unique temporary directory for each processing request"""
    session_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(tempfile.gettempdir(), f"dots_mocr_demo_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir, session_id

def parse_image_with_high_level_api(parser, image, prompt_mode, fitz_preprocess=False, custom_prompt=None, temperature=None):
    """
    Processes using the high-level API parse_image from DotsMOCRParser
    """
    # Create a temporary session directory
    temp_dir, session_id = create_temp_session_dir()
    
    try:
        # Save the PIL Image as a temporary file
        temp_image_path = os.path.join(temp_dir, f"input_{session_id}.png")
        image.save(temp_image_path, "PNG")
        
        # Use the high-level API parse_image
        filename = f"demo_{session_id}"
        results = parser.parse_image(
            input_path=image,
            filename=filename, 
            prompt_mode=prompt_mode,
            save_dir=temp_dir,
            fitz_preprocess=fitz_preprocess,
            custom_prompt=custom_prompt,
            temperature=temperature,
        )
        
        # Parse the results
        if not results:
            raise ValueError("No results returned from parser")
        
        result = results[0]  # parse_image returns a list with a single result
        
        layout_image = None
        if 'layout_image_path' in result and os.path.exists(result['layout_image_path']):
            layout_image = Image.open(result['layout_image_path'])
        
        cells_data = None
        if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
            with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                cells_data = json.load(f)
        
        md_content = None
        if 'md_content_path' in result and os.path.exists(result['md_content_path']):
            with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                md_content = f.read()
        
        return {
            'layout_image': layout_image,
            'cells_data': cells_data,
            'md_content': md_content,
            'filtered': result.get('filtered', False),
            'temp_dir': temp_dir,
            'session_id': session_id,
            'result_paths': result,
            'input_width': result.get('input_width', 0),
            'input_height': result.get('input_height', 0),
        }
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def parse_pdf_with_high_level_api(parser, pdf_path, prompt_mode):
    """
    Processes using the high-level API parse_pdf from DotsMOCRParser
    """
    # Create a temporary session directory
    temp_dir, session_id = create_temp_session_dir()
    
    try:
        # Use the high-level API parse_pdf
        filename = f"demo_{session_id}"
        results = parser.parse_pdf(
            input_path=pdf_path,
            filename=filename,
            prompt_mode=prompt_mode,
            save_dir=temp_dir
        )
        
        # Parse the results
        if not results:
            raise ValueError("No results returned from parser")
        
        # Handle multi-page results
        parsed_results = []
        all_md_content = []
        all_cells_data = []
        
        for i, result in enumerate(results):
            page_result = {
                'page_no': result.get('page_no', i),
                'layout_image': None,
                'cells_data': None,
                'md_content': None,
                'filtered': False
            }
            
            # Read the layout image
            if 'layout_image_path' in result and os.path.exists(result['layout_image_path']):
                page_result['layout_image'] = Image.open(result['layout_image_path'])
            
            # Read the JSON data
            if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
                with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                    page_result['cells_data'] = json.load(f)
                    all_cells_data.extend(page_result['cells_data'])
            
            # Read the Markdown content
            if 'md_content_path' in result and os.path.exists(result['md_content_path']):
                with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                    page_content = f.read()
                    page_result['md_content'] = page_content
                    all_md_content.append(page_content)
            page_result['filtered'] = result.get('filtered', False)
            parsed_results.append(page_result)
        
        combined_md = "\n\n---\n\n".join(all_md_content) if all_md_content else ""
        return {
            'parsed_results': parsed_results,
            'combined_md_content': combined_md,
            'combined_cells_data': all_cells_data,
            'temp_dir': temp_dir,
            'session_id': session_id,
            'total_pages': len(results)
        }
        
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

# ==================== Core Processing Function ====================
def process_image_inference(session_state, test_image_input, file_input,
                          prompt_mode, model_selector,  # Changed: use model_selector instead of server_ip/port
                          min_pixels, max_pixels,
                          fitz_preprocess=False, custom_prompt=""
                          ):
    """Core function to handle image/PDF inference"""
    # Use session_state instead of global variables
    processing_results = session_state['processing_results']
    pdf_cache = session_state['pdf_cache']
    
    if processing_results.get('temp_dir') and os.path.exists(processing_results['temp_dir']):
        try:
            shutil.rmtree(processing_results['temp_dir'], ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up previous temporary directory: {e}")
    
    # Reset processing results for the current session
    session_state['processing_results'] = get_initial_session_state()['processing_results']
    processing_results = session_state['processing_results']
    
    fitz_preprocess = PROMPT_TO_FITZ_PREPROCESS.get(prompt_mode, True)
    temperature = PROMPT_TO_TEMPERATURE.get(prompt_mode, 0.1) 
    print(temperature)
    # Get the selected model configuration
    model_config = MODEL_SERVERS[model_selector]
    current_config.update({
        'ip': model_config['ip'],
        'port_vllm': model_config['port_vllm'],
        'min_pixels': min_pixels,
        'max_pixels': max_pixels
    })
    
    # Get parser for the selected model
    try:
        dots_parser = get_parser(model_selector, min_pixels, max_pixels)
    except ValueError as e:
        return None, f"Error: {str(e)}", "", "", gr.update(value=None), None, "", session_state
    
    input_file_path = file_input if file_input else test_image_input
    
    if not input_file_path:
        return None, "Please upload image/PDF file or select test image", "", "", gr.update(value=None), None, "", session_state
    
    file_ext = os.path.splitext(input_file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            # MINIMAL CHANGE: The `process_pdf_file` function is now inlined and uses session_state.
            preview_image, page_info, session_state = load_file_for_preview(input_file_path, session_state)
            pdf_result = parse_pdf_with_high_level_api(dots_parser, input_file_path, prompt_mode)
            
            session_state['pdf_cache']["is_parsed"] = True
            session_state['pdf_cache']["results"] = pdf_result['parsed_results']
            
            processing_results.update({
                'markdown_content': pdf_result['combined_md_content'],
                'cells_data': pdf_result['combined_cells_data'],
                'temp_dir': pdf_result['temp_dir'],
                'session_id': pdf_result['session_id'],
                'pdf_results': pdf_result['parsed_results']
            })
            
            total_elements = len(pdf_result['combined_cells_data'])
            info_text = f"**PDF Information:**\n- Total Pages: {pdf_result['total_pages']}\n- Model: {model_selector}\n- Server: {model_config['ip']}:{model_config['port_vllm']}\n- Total Detected Elements: {total_elements}\n- Session ID: {pdf_result['session_id']}"
            
            current_page_layout_image = preview_image
            current_page_json = ""
            if session_state['pdf_cache']["results"]:
                first_result = session_state['pdf_cache']["results"][0]
                if 'layout_image' in first_result and first_result['layout_image']:
                    current_page_layout_image = first_result['layout_image']
                if first_result.get('cells_data'):
                    try:
                        current_page_json = json.dumps(first_result['cells_data'], ensure_ascii=False, indent=2)
                    except:
                        current_page_json = str(first_result['cells_data'])

            download_zip_path = None
            if pdf_result['temp_dir']:
                download_zip_path = os.path.join(pdf_result['temp_dir'], f"layout_results_{pdf_result['session_id']}.zip")
                with zipfile.ZipFile(download_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(pdf_result['temp_dir']):
                        for file in files:
                            if not file.endswith('.zip'): zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), pdf_result['temp_dir']))

            return (
                current_page_layout_image, info_text, pdf_result['combined_md_content'] or "No markdown content generated",
                pdf_result['combined_md_content'] or "No markdown content generated",
                gr.update(value=download_zip_path, visible=bool(download_zip_path)), page_info, current_page_json, session_state
            )
        
        else: # Image processing
            image = read_image_v2(input_file_path)
            session_state['pdf_cache'] = get_initial_session_state()['pdf_cache']
            
            original_image = image
            effective_custom_prompt = custom_prompt if prompt_mode == 'prompt_general' else None
            parse_result = parse_image_with_high_level_api(dots_parser, image, prompt_mode, fitz_preprocess, effective_custom_prompt, temperature)
            
            if parse_result['filtered']:
                 info_text = f"**Image Information:**\n- Original Size: {original_image.width} x {original_image.height}\n- Model: {model_selector}\n- Processing: JSON parsing failed, using cleaned text output\n- Server: {model_config['ip']}:{model_config['port_vllm']}\n- Session ID: {parse_result['session_id']}"
                 processing_results.update({
                     'original_image': original_image, 'markdown_content': parse_result['md_content'],
                     'temp_dir': parse_result['temp_dir'], 'session_id': parse_result['session_id'],
                     'result_paths': parse_result['result_paths']
                 })
                 return original_image, info_text, parse_result['md_content'], parse_result['md_content'], gr.update(visible=False), None, "", session_state
            
            md_content_raw = parse_result['md_content'] or "No markdown content generated"
            processing_results.update({
                'original_image': original_image, 'layout_result': parse_result['layout_image'],
                'markdown_content': parse_result['md_content'], 'cells_data': parse_result['cells_data'],
                'temp_dir': parse_result['temp_dir'], 'session_id': parse_result['session_id'],
                'result_paths': parse_result['result_paths']
            })
            
            num_elements = len(parse_result['cells_data']) if parse_result['cells_data'] else 0
            info_text = f"**Image Information:**\n- Original Size: {original_image.width} x {original_image.height}\n- Model Input Size: {parse_result['input_width']} x {parse_result['input_height']}\n- Model: {model_selector}\n- Server: {model_config['ip']}:{model_config['port_vllm']}\n- Detected {num_elements} layout elements\n- Session ID: {parse_result['session_id']}"
            
            current_json = json.dumps(parse_result['cells_data'], ensure_ascii=False, indent=2) if parse_result['cells_data'] else ""
            
            download_zip_path = None
            if parse_result['temp_dir']:
                download_zip_path = os.path.join(parse_result['temp_dir'], f"layout_results_{parse_result['session_id']}.zip")
                with zipfile.ZipFile(download_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(parse_result['temp_dir']):
                        for file in files:
                            if not file.endswith('.zip'): zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), parse_result['temp_dir']))
            
            return (
                parse_result['layout_image'], info_text, parse_result['md_content'] or "No markdown content generated",
                md_content_raw, gr.update(value=download_zip_path, visible=bool(download_zip_path)),
                None, current_json, session_state
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during processing: {e}", "", "", gr.update(value=None), None, "", session_state

# MINIMAL CHANGE: Functions now take `session_state` as an argument.
def clear_all_data(session_state):
    """Clears all data"""
    processing_results = session_state['processing_results']
    
    if processing_results.get('temp_dir') and os.path.exists(processing_results['temp_dir']):
        try:
            shutil.rmtree(processing_results['temp_dir'], ignore_errors=True)
        except Exception as e:
            print(f"Failed to clean up temporary directory: {e}")
    
    # Reset the session state by returning a new initial state
    new_session_state = get_initial_session_state()
    
    return (
        None,  # Clear file input
        "",    # Clear test image selection
        None,  # Clear result image
        "Waiting for processing results...",  # Reset info display
        "## Waiting for processing results...",  # Reset Markdown display
        "🕐 Waiting for parsing result...",    # Clear raw Markdown text
        gr.update(visible=False),  # Hide download button
        "<div id='page_info_box'>0 / 0</div>",  # Reset page info
        "🕐 Waiting for parsing result...",     # Clear current page JSON
        new_session_state
    )

def update_prompt_display(prompt_mode):
    """Updates the prompt display content"""
    if prompt_mode == 'prompt_general':
        return ""  # free_qa 模式下清空，让用户输入
    return dict_promptmode_to_prompt[prompt_mode]

# ==================== Gradio Interface ====================
def create_gradio_interface():
    """Creates the Gradio interface"""
    
    # CSS styles, matching the reference style
    css = """

    #parse_button {
        background: #FF576D !important; /* !important 确保覆盖主题默认样式 */
        border-color: #FF576D !important;
    }
    /* 鼠标悬停时的颜色 */
    #parse_button:hover {
        background: #F72C49 !important;
        border-color: #F72C49 !important;
    }
    
    #page_info_html {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        margin: 0 12px;
    }

    #page_info_box {
        padding: 8px 20px;
        font-size: 16px;
        border: 1px solid #bbb;
        border-radius: 8px;
        background-color: #f8f8f8;
        text-align: center;
        min-width: 80px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    #markdown_output {
        min-height: 800px;
        overflow: auto;
    }

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
    
    #result_image {
        border-radius: 8px;
    }
    
    #markdown_tabs {
        height: 100%;
    }

    #model_selector_box {
        margin-bottom: 8px;
    }
    """
    
    with gr.Blocks(theme="ocean", css=css, title='dots.mocr') as demo:
        session_state = gr.State(value=get_initial_session_state())
        
        # Title
        gr.HTML("""
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 2em;">🔍 dots.mocr</h1>
            </div>
            <div style="text-align: center; margin-bottom: 10px;">
                <em>Recognize Any Human Scripts and Symbols</em>
            </div>
        """)
        
        with gr.Row():
            # Left side: Input and Configuration
            with gr.Column(scale=1, elem_id="left-panel"):
                gr.Markdown("### 📥 Upload & Select")
                file_input = gr.File(
                    label="Upload PDF/Image", 
                    type="filepath", 
                    file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                )
                
                # ============ NEW: Model Selector ============
                model_selector = gr.Dropdown(
                    label="🤖 Select Model",
                    choices=list(MODEL_SERVERS.keys()),
                    value=list(MODEL_SERVERS.keys())[0],
                    elem_id="model_selector_box",
                    info="Switch between different model servers"
                )
                
                test_images = get_test_images()
                test_image_input = gr.Dropdown(
                    label="Or Select an Example",
                    choices=[""] + test_images,
                    value="",
                )

                gr.Markdown("### ⚙️ Prompt & Actions")
                prompt_mode = gr.Dropdown(
                    label="Select Prompt",
                        choices=[
                            "prompt_layout_all_en", 
                            "prompt_web_parsing",
                            "prompt_scene_spotting",         
                            "prompt_image_to_svg",       
                            "prompt_general",
                            "prompt_layout_only_en", 
                            "prompt_ocr",            
                        ],
                    value="prompt_layout_all_en",
                )
                
                # Display current prompt content
                prompt_display = gr.Textbox(
                    label="Current Prompt Content",
                    value=dict_promptmode_to_prompt[list(dict_promptmode_to_prompt.keys())[0]],
                    lines=4,
                    max_lines=8,
                    interactive=False,  # 默认不可编辑，free_qa 模式下改为可编辑
                )
                
                with gr.Row():
                    process_btn = gr.Button("🔍 Parse", variant="primary", scale=2, elem_id="parse_button")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                with gr.Accordion("🛠️ Advanced Configuration", open=False):
                    fitz_preprocess = gr.Checkbox(
                        label="Enable fitz_preprocess for images", 
                        value=True,
                        info="Processes image via a PDF-like pipeline (image->pdf->200dpi image). Recommended if your image DPI is low.",
                        visible=False, ###直接隐藏，调用模型前根据prompt mode 写死
                    )
                    with gr.Row():
                        min_pixels = gr.Number(label="Min Pixels", value=DEFAULT_CONFIG['min_pixels'], precision=0)
                        max_pixels = gr.Number(label="Max Pixels", value=DEFAULT_CONFIG['max_pixels'], precision=0)
            # Right side: Result Display
            with gr.Column(scale=6, variant="compact"):
                with gr.Row():
                    # Result Image
                    with gr.Column(scale=3):
                        gr.Markdown("### 👁️ File Preview")
                        result_image = gr.Image(
                            label="Layout Preview",
                            visible=True,
                            height=800,
                            show_label=False,
                        )
                        
                        # Page navigation (shown during PDF preview)
                        with gr.Row():
                            prev_btn = gr.Button("⬅ Previous", size="sm")
                            page_info = gr.HTML(
                                value="<div id='page_info_box'>0 / 0</div>", 
                                elem_id="page_info_html"
                            )
                            next_btn = gr.Button("Next ➡", size="sm")
                        
                        # Info Display
                        info_display = gr.Markdown(
                            "Waiting for processing results...",
                            elem_id="info_box"
                        )
                    
                    # Markdown Result
                    with gr.Column(scale=3):
                        gr.Markdown("### ✔️ Result Display")
                        
                        with gr.Tabs(elem_id="markdown_tabs"):
                            with gr.TabItem("Markdown Render Preview"):
                                md_output = gr.Markdown(
                                    "## Please click the parse button to parse or select for single-task recognition...",
                                    max_height=600,
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False}
                                    ],
                                    elem_id="markdown_output"
                                )
                            
                            with gr.TabItem("Markdown Raw Text"):
                                md_raw_output = gr.Textbox(
                                    value="🕐 Waiting for parsing result...",
                                    label="Markdown Raw Text",
                                    max_lines=100,
                                    lines=38,
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                            
                            with gr.TabItem("Current Page JSON"):
                                current_page_json = gr.Textbox(
                                    value="🕐 Waiting for parsing result...",
                                    label="Current Page JSON",
                                    max_lines=100,
                                    lines=38,
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                
                # Download Button
                with gr.Row():
                    download_btn = gr.DownloadButton(
                        "⬇️ Download Results",
                        visible=False
                    )

        def update_prompt_and_interactive(prompt_mode, session_state):
            """更新 prompt_display 并自动切换模型"""
            is_free_qa = prompt_mode == 'prompt_general'
            auto_custom_prompt = session_state.get('auto_custom_prompt')
            
            if is_free_qa and auto_custom_prompt:
                prompt_text = auto_custom_prompt
                interactive = True
            else:
                prompt_text = update_prompt_display(prompt_mode)
                interactive = is_free_qa
            
            # 根据prompt_mode自动选择模型
            auto_model = PROMPT_TO_MODEL.get(prompt_mode, list(MODEL_SERVERS.keys())[0])
            
            return (
                gr.update(value=prompt_text, interactive=interactive), 
                session_state,
                gr.update(value=auto_model),
            )
                

        prompt_mode.change(
            fn=update_prompt_and_interactive,
            inputs=[prompt_mode, session_state],
            outputs=[prompt_display, session_state, model_selector],
        )
        
        # Show preview on file upload
        file_input.upload(
            # fn=lambda file_data, state: load_file_for_preview(file_data, state),
            fn=load_file_for_preview,
            inputs=[file_input, session_state],
            outputs=[result_image, page_info, session_state]
        )
        
        # Also handle test image selection
        test_image_input.change(
            fn=on_test_image_select,
            inputs=[test_image_input, session_state],
            outputs=[result_image, page_info, session_state, prompt_mode, prompt_display, model_selector],
        )

        prev_btn.click(
            fn=lambda s: turn_page("prev", s),
            inputs=[session_state], 
            outputs=[result_image, page_info, current_page_json, session_state]
        )
        
        next_btn.click(
            fn=lambda s: turn_page("next", s),
            inputs=[session_state], 
            outputs=[result_image, page_info, current_page_json, session_state]
        )
        
        # ============ MODIFIED: process_btn.click with model_selector ============
        process_btn.click(
            fn=process_image_inference,
            inputs=[
                session_state, test_image_input, file_input,
                prompt_mode, model_selector,  # Changed: model_selector instead of server_ip/port
                min_pixels, max_pixels, 
                fitz_preprocess, prompt_display
            ],
            outputs=[
                result_image, info_display, md_output, md_raw_output,
                download_btn, page_info, current_page_json, session_state
            ]
        )
        
        clear_btn.click(
            fn=clear_all_data,
            inputs=[session_state],
            outputs=[
                file_input, test_image_input,
                result_image, info_display, md_output, md_raw_output,
                download_btn, page_info, current_page_json, session_state
            ]
        )
    
    return demo

# ==================== Main Program ====================
if __name__ == "__main__":
    import sys
    port = int(sys.argv[1])
    demo = create_gradio_interface()
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=port, 
        debug=True
    )
