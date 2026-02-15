import argparse

from openai import OpenAI
from transformers.utils.versions import require_version
from PIL import Image
from dots_ocr.utils import dict_promptmode_to_prompt
from dots_ocr.model.inference import inference_with_vllm


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="localhost")
parser.add_argument("--port", type=str, default="8000")
parser.add_argument("--model_name", type=str, default="rednote-hilab/dots.ocr-1.5")
parser.add_argument("--custom_prompt", type=str, default="Please describe the content of this image.")

args = parser.parse_args()

require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")


def main():
    addr = f"http://{args.ip}:{args.port}/v1"
    image_path = "demo/demo_image3.jpg"
    prompt = args.custom_prompt
    image = Image.open(image_path)
    response = inference_with_vllm(
        image,
        prompt, 
        ip=args.ip,
        port=args.port,
        temperature=0.1,
        top_p=0.9,
        model_name=args.model_name,
        system_prompt="You are a helpful assistant.", #general tasks need system_prompt
    )
    print(f"response: {response}")


if __name__ == "__main__":
    main()
