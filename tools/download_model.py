from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="huggingface")
    parser.add_argument('--name', '-n', type=str, default="rednote-hilab/dots.ocr-1.5")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Attention: The model save dir dots.ocr should be replace by a name without `.` like DotsOCR, util we merge our code to transformers.")
    model_dir = os.path.join(script_dir, "weights/DotsOCR_1_5")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.type == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
    elif args.type == "modelscope":
        from modelscope import snapshot_download
        snapshot_download(repo_id=args.name, local_dir=model_dir)
    else:
        raise ValueError(f"Invalid type: {args.type}")
    
    print(f"model downloaded to {model_dir}")
