import os
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

def inference(image_path, prompt, model, processor):
    # image_path = "demo/demo_image1.jpg"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=24000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)



if __name__ == "__main__":
    # We recommend enabling flash_attention_2 or flash_attention_3 for better acceleration and memory saving, especially in multi-image and video scenarios.
    model_path = "./weights/DotsMOCR"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # device_map="cpu",  # ve里默认使用flash-attn，无法直接运行
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True)

    image_path = "demo/demo_image1.jpg"
    for prompt_mode, prompt in dict_promptmode_to_prompt.items():
        print(f"prompt: {prompt}")
        inference(image_path, prompt, model, processor)
    