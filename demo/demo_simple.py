"""
Simple Demo без vLLM - использует модель напрямую через transformers
"""
import os
# КРИТИЧНО: Устанавливаем переменные окружения ДО импорта PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Асинхронный режим для лучшей производительности

import gradio as gr
import torch
from PIL import Image
from pathlib import Path

print(f"✓ PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

# ВАЖНО: Ограничиваем GPU память сразу после импорта torch
if torch.cuda.is_available():
    # Ограничиваем до 80% (6.4GB из 8GB) - модель требует ~5.5GB
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    # Очищаем кеш перед загрузкой модели
    torch.cuda.empty_cache()
    print(f"✓ GPU memory limit set to 80% (~6.4GB)")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")

# Импортируем необходимые компоненты
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils import dict_promptmode_to_prompt

print("Loading model...")
print(f"⏳ GPU memory before loading: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB allocated")
model_path = "./weights/DotsOCR"

# Создаем парсер (используем HuggingFace модель напрямую, без vLLM)
parser = DotsOCRParser(
    use_hf=True  # Ключевой параметр!
)

print(f"✓ GPU memory after loading: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB allocated")
print(f"✓ GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB")

print(f"✓ Model loaded (HuggingFace mode)")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    print(f"✓ Memory limit: {torch.cuda.get_device_properties(0).total_memory*0.8/1024**3:.2f}GB (80%)")
    print(f"✓ Current allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
    print(f"✓ Current reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB")

def process_image(image, prompt_mode):
    """Process image with DotsOCR"""
    if image is None:
        return None, "Please upload an image", "", ""
    
    try:
        # Получаем промпт
        prompt = dict_promptmode_to_prompt.get(prompt_mode, dict_promptmode_to_prompt["prompt_layout_all_en"])
        
        # Создаем директорию temp если её нет
        import os
        os.makedirs("./temp", exist_ok=True)
        
        # Логируем начало обработки
        import time
        start_time = time.time()
        
        # GPU мониторинг
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_mem_reserved_before = torch.cuda.memory_reserved(0) / 1024**3  # GB
        
        print(f"\n{'='*60}")
        print(f"🔍 НАЧАЛО ОБРАБОТКИ")
        print(f"{'='*60}")
        print(f"📄 Тип изображения: {type(image)}")
        print(f"📐 Размер изображения: {image.size if hasattr(image, 'size') else 'Unknown'}")
        print(f"🎯 Prompt mode: {prompt_mode}")
        print(f"📝 Prompt: {prompt[:100]}...")
        if torch.cuda.is_available():
            print(f"🎮 GPU Memory (before): {gpu_mem_before:.2f}GB allocated, {gpu_mem_reserved_before:.2f}GB reserved")
        print(f"{'='*60}\n")
        
        # Обработка изображения
        result = parser.parse_image(
            input_path=image,
            filename="demo",
            prompt_mode=prompt_mode,
            save_dir="./temp"
        )
        
        processing_time = time.time() - start_time
        
        # GPU мониторинг после обработки
        if torch.cuda.is_available():
            gpu_mem_after = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_mem_reserved_after = torch.cuda.memory_reserved(0) / 1024**3  # GB
            gpu_mem_used = gpu_mem_after - gpu_mem_before
        
        print(f"\n{'='*60}")
        print(f"✅ РЕЗУЛЬТАТ ОБРАБОТКИ")
        print(f"{'='*60}")
        print(f"⏱️  Время обработки: {processing_time:.2f}s")
        if torch.cuda.is_available():
            print(f"🎮 GPU Memory (after): {gpu_mem_after:.2f}GB allocated, {gpu_mem_reserved_after:.2f}GB reserved")
            print(f"📊 GPU Memory used: {gpu_mem_used:+.2f}GB")
            print(f"📈 Peak memory: {torch.cuda.max_memory_allocated(0)/1024**3:.2f}GB")
        print(f"📊 Результатов: {len(result) if result else 0}")
        
        if result and len(result) > 0:
            result_data = result[0]
            
            print(f"🔑 Ключи результата: {list(result_data.keys())}")
            
            # Получаем результаты
            md_content = result_data.get('md_content', '')
            layout_image = result_data.get('layout_image', image)
            cells_data = result_data.get('cells_data', [])
            
            print(f"📝 Markdown длина: {len(md_content) if md_content else 0} символов")
            print(f"🔢 Обнаружено элементов: {len(cells_data) if cells_data else 0}")
            print(f"{'='*60}\n")
            
            # Создаем детальное описание процесса
            process_log = f"""### 🔍 Процесс обработки:

**1. Входные данные:**
- Размер изображения: {image.size if hasattr(image, 'size') else 'Unknown'}
- Prompt mode: `{prompt_mode}`
- Prompt: `{prompt[:80]}...`

**2. Обработка:**
- Время: {processing_time:.2f}s
- Обнаружено элементов: {len(cells_data) if cells_data else 0}

**3. Результат:**
- Markdown: {len(md_content) if md_content else 0} символов
- Layout image: {'✅ Создан' if layout_image else '❌ Не создан'}
"""
            
            info = f"""### ✅ Processing Complete!
**Prompt Mode:** {prompt_mode}
**Status:** Success
**Time:** {processing_time:.2f}s
**Elements detected:** {len(cells_data) if cells_data else 0}
**Mode:** HuggingFace (No vLLM)
"""
            
            if not md_content:
                md_content = "⚠️ Markdown content is empty. Check process log."
            
            return layout_image, info, md_content, process_log
        else:
            print(f"⚠️  Нет результатов!")
            print(f"{'='*60}\n")
            return image, "⚠️  No results returned", "", "❌ No results from parser"
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        print(f"\n{'='*60}")
        print(f"❌ ОШИБКА:")
        print(f"{'='*60}")
        print(error_trace)
        print(f"{'='*60}\n")
        
        error_msg = f"""### ❌ Error occurred:
```
{str(e)}
```
"""
        error_log = f"""### Traceback:
```
{error_trace}
```
"""
        return image, error_msg, "", error_log

# Создаем Gradio интерфейс
with gr.Blocks(theme="ocean", title="dots.ocr Simple Demo") as demo:
    gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🔍 dots.ocr Simple Demo</h1>
            <p><em>Document OCR with Layout Analysis (No vLLM required)</em></p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")
            image_input = gr.Image(type="pil", label="Upload Image")
            
            prompt_mode = gr.Dropdown(
                label="Prompt Mode",
                choices=list(dict_promptmode_to_prompt.keys()),
                value="prompt_layout_all_en"
            )
            
            process_btn = gr.Button("🔍 Process", variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
            
            info_display = gr.Markdown("Waiting for input...")
        
        with gr.Column(scale=1):
            gr.Markdown("### 👁️  Layout Result")
            layout_output = gr.Image(label="Layout Analysis")
            
        with gr.Column(scale=1):
            gr.Markdown("### ✔️ Results")
            
            with gr.Tabs():
                with gr.TabItem("📝 Markdown Output"):
                    md_output = gr.Markdown("Waiting for processing...")
                
                with gr.TabItem("🔍 Process Log"):
                    process_log = gr.Markdown("Waiting for processing...")
    
    # Event handlers
    process_btn.click(
        fn=process_image,
        inputs=[image_input, prompt_mode],
        outputs=[layout_output, info_display, md_output, process_log]
    )
    
    clear_btn.click(
        fn=lambda: (None, "Waiting for input...", "Waiting for processing...", "Waiting for processing..."),
        inputs=[],
        outputs=[layout_output, info_display, md_output, process_log]
    )

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    
    print("\n═══════════════════════════════════════════════════════")
    print(f"🚀 Starting Gradio on 0.0.0.0:{port}")
    print(f"   Accessible from: http://192.168.1.115:{port}")
    print(f"   Or any local IP on port {port}")
    print("═══════════════════════════════════════════════════════\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )

