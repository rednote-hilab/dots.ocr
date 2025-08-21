# dots_mps_parse.py
# macOS/M1-friendly runner for dots.ocr (local, offline, SDPA on MPS)
# - Uses repo's built-in prompts via --prompt-mode (JSON-ready)
# - Harmless flash_attn shim (with valid __spec__) to avoid import crashes
# - Forces SDPA attention (not Flash-Attn)
# - Disables Sliding-Window Attention for SDPA/Qwen2
# - Avoids BF16 by forcing FP16 everywhere and monkey-patching the vision tower
# - Processes PDFs page-by-page with pixel caps to avoid OOM on 16 GB
# - Parses JSON output flexibly (array OR object) and can drop headers/footers

# --- harmless flash_attn shim with a valid __spec__ --------------------------
import sys, types, importlib.machinery as _machinery
if "flash_attn" not in sys.modules:
    _m = types.ModuleType("flash_attn")
    _m.__spec__ = _machinery.ModuleSpec(name="flash_attn", loader=None)
    _m.__path__ = []
    def flash_attn_varlen_func(*args, **kwargs):
        raise RuntimeError(
            "flash_attn was requested but isn't available on this platform. "
            "Use attn_implementation='sdpa'."
        )
    _m.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = _m
# ---------------------------------------------------------------------------

import os, gc, argparse, fitz, json, re
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from dots_ocr.utils.prompts import dict_promptmode_to_prompt  # repo prompts
from dots_ocr.utils.try_parse_json_object import try_parse_ast_to_json

def pil_from_page(page, dpi=144):
    # Render the PDF page to an RGB PIL.Image at a modest DPI to cap memory.
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

# ---- JSON helpers -----------------------------------------------------------

def parse_json_flex(s: str):
    """
    Accepts:
      - a full JSON object: {...}
      - a full JSON array:  [...]
      - messy text with one JSON object/array embedded: ...{...}... or ...[...]...
    Returns the parsed Python value.
    """
    t = s.strip()
    # Fast paths (trim trailing junk if the model echoes EOS tokens etc.)
    if t.startswith("{"):
        return json.loads(t[: t.rfind("}") + 1])
    if t.startswith("["):
        return json.loads(t[: t.rfind("]") + 1])

    # Slow path: extract first {...} or [...] block
    m_obj = re.search(r"\{[\s\S]*\}", s)
    m_arr = re.search(r"\[[\s\S]*\]", s)
    m = None
    if m_obj and m_arr:
        m = m_obj if m_obj.start() < m_arr.start() else m_arr
    else:
        m = m_obj or m_arr
    if not m:
        raise ValueError("No JSON object/array found in model output.")
    return json.loads(m.group(0))

def maybe_filter_blocks(obj, drop_headers_footers: bool):
    if not drop_headers_footers:
        return obj
    obj["blocks"] = [
        b for b in obj.get("blocks", [])
        if b.get("category") not in ("Page-header", "Page-footer")
    ]
    return obj

# ----------------------------------------------------------------------------

def main(args):
    # Offline caches: safe to set in-code
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # If you hit MPS-missing ops, set this in your shell before running:
    #   export PYTORCH_ENABLE_MPS_FALLBACK=1

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    # Cap visual tokens to avoid OOM on 16 GB (raise max_pix later if you have headroom)
    min_pix = 256 * 28 * 28
    max_pix = 640 * 28 * 28

    # Force SDPA & disable SWA (SWA not implemented for SDPA/Qwen2)
    config = AutoConfig.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    if getattr(config, "vision_config", None) is not None:
        config.vision_config.attn_implementation = "sdpa"
    else:
        setattr(config, "attn_implementation", "sdpa")
    if hasattr(config, "use_sliding_window"):
        config.use_sliding_window = False
    if hasattr(config, "sliding_window"):
        config.sliding_window = None

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        min_pixels=min_pix,
        max_pixels=max_pix,
        trust_remote_code=True,
        # use_fast=True,  # uncomment to quiet the "slow processor" warning
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        config=config,                     # keep SDPA + SWA=off
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if device == "mps":
        model.to("mps")

    # CRITICAL: their vision tower defaults to bf16; force fp16 to avoid dtype mismatch on MPS
    import types as _types
    orig_forward_func = model.vision_tower.forward.__func__  # unbound function
    def _forward_no_bf16(self, hidden_states, grid_thw, bf16=True):
        return orig_forward_func(self, hidden_states, grid_thw, bf16=False)
    model.vision_tower.forward = _types.MethodType(_forward_no_bf16, model.vision_tower)

    attn_impl = getattr(
        getattr(model.config, "vision_config", model.config), "attn_implementation", None
    )
    print(f"[info] device={device}, dtype={dtype}, vision_attn={attn_impl}, "
          f"use_sliding_window={getattr(model.config, 'use_sliding_window', None)}")

    doc = fitz.open(args.pdf)
    for i, page in enumerate(doc, start=1):
        img = pil_from_page(page, dpi=args.dpi)

        # Choose prompt text: repo prompt-mode first; --prompt overrides if provided
        prompt_text = dict_promptmode_to_prompt.get(args.prompt_mode, "")
        if args.prompt is not None:
            prompt_text = args.prompt

        # Qwen-style chat with user (image+text). Repo prompt already asks for JSON.
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

        # Move to device + force FP16 on floats
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(device)
                if torch.is_floating_point(v):
                    v = v.to(dtype=dtype)
                inputs[k] = v

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=None,  # deterministic
                do_sample=False,
            )

        trimmed = [o[len(iids):] for iids, o in zip(inputs["input_ids"], out_ids)]
        out_text = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if args.print_raw:
            print("\n--- RAW MODEL OUTPUT ---\n")
            print(out_text)

        # Parse JSON flexibly (array OR object) 
        parsed, result = try_parse_ast_to_json(out_text)

        obj = {"blocks": parsed} if isinstance(parsed, list) else parsed

        # Ensure page metadata exists
        w, h = img.size
        obj.setdefault("page", i)
        obj.setdefault("width", w)
        obj.setdefault("height", h)

        obj = maybe_filter_blocks(obj, args.drop_headers_footers)

        # Print for debugging
        print(f"\n===== Page {i}/{len(doc)} JSON =====\n")
        print(json.dumps(obj, ensure_ascii=False, indent=2))

        # Persist to disk if requested
        if args.json_out:
            out_line = json.dumps(obj, ensure_ascii=False)
            if args.json_out.lower().endswith(".jsonl"):
                with open(args.json_out, "a", encoding="utf-8") as f:
                    f.write(out_line + "\n")
            else:
                # If a plain .json was given, write alongside as .jsonl
                with open(args.json_out + ".jsonl", "a", encoding="utf-8") as f:
                    f.write(out_line + "\n")

        # Free per-page memory
        del img, inputs, out_ids, trimmed, obj, parsed
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()  # release unoccupied cached VRAM

if __name__ == "__main__":
    # Build argparse choices from repo prompts at runtime
    PROMPT_KEYS = tuple(dict_promptmode_to_prompt.keys())
    DEFAULT_PROMPT_MODE = "prompt_layout_all_en" if "prompt_layout_all_en" in PROMPT_KEYS else (PROMPT_KEYS[0] if PROMPT_KEYS else None)

    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--model-dir", default="./weights/DotsOCR")
    ap.add_argument("--dpi", type=int, default=144)        # drop to 120 if memory spikes
    ap.add_argument("--max-new-tokens", type=int, default=1536)  # 512–2048 is realistic on 16 GB
    ap.add_argument("--prompt", default=None, help="Custom user prompt (overrides --prompt-mode).")
    ap.add_argument("--prompt-mode",
                    default=DEFAULT_PROMPT_MODE,
                    choices=list(PROMPT_KEYS),
                    help="Use a built-in dots.ocr prompt (e.g. prompt_layout_all_en outputs JSON).")
    ap.add_argument("--json-out", default=None,
                    help="Path to write JSONL (.jsonl) or base name (.json -> will write .jsonl). "
                         "Appends one JSON object per page.")
    ap.add_argument("--drop-headers-footers", action="store_true",
                    help="Drop Page-header/Page-footer blocks from JSON.")
    ap.add_argument("--print-raw", action="store_true",
                    help="Print raw model output before JSON parsing.")
    args = ap.parse_args()
    main(args)