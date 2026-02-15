import argparse

from openai import OpenAI
from transformers.utils.versions import require_version
from PIL import Image
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.model.inference import inference_with_vllm


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="localhost")
parser.add_argument("--port", type=str, default="8000")
parser.add_argument("--model_name", type=str, default="rednote-hilab/dots.ocr")
parser.add_argument("--image_path", type=str, default="demo/demo_image1.jpg")
parser.add_argument("--prompt_mode", type=str, default="prompt_layout_all_en",help=(
        "Choose a task prompt: "
        "prompt_layout_all_en=full document layout+OCR to JSON/MD; "
        "prompt_layout_only_en=layout detection only; "
        "prompt_grounding_ocr=OCR within a given bbox; "
        "prompt_web_parsing=parse webpage screenshot layout into JSON; "
        "prompt_scene_spotting=detect+recognize scene text (OCR boxes+texts); "
        "prompt_image_to_svg=generate SVG code to reconstruct the image.")
)

args = parser.parse_args()

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    addr = f"http://{args.ip}:{args.port}/v1"
    image_path = args.image_path
    prompt = dict_promptmode_to_prompt[args.prompt_mode]
    image = Image.open(image_path)
    response = inference_with_vllm(
        image,
        prompt, 
        ip=args.ip,
        port=args.port,
        temperature=0.1,
        top_p=0.9,
        model_name=args.model_name,
    )
    print(f"response: {response}")


if __name__ == "__main__":
    main()
