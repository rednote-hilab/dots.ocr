# Make flash-attn optional via extras — 2025-08-28 19:22:00

Change summary:
- Update `setup.py` to exclude `flash-attn` and `accelerate` from default `install_requires`.
- Add `extras_require["gpu"]` so GPU users can explicitly opt in.

Why:
- `flash-attn` often fails to build on CPU-only or mismatched CUDA/Torch setups during metadata generation.
- Making it optional prevents install failures on non-GPU environments.

How to install:
- CPU/default (no GPU extras):
  - `pip install -e .`
- GPU with flash-attn:
  - `pip install -e .[gpu]`
  - Ensure a compatible CUDA-enabled PyTorch and toolchain are present before installing.

Notes:
- `requirements.txt` still lists `flash-attn==2.8.0.post2`, but `setup.py` filters it out for default installs.
- The root orchestrator (`../run.sh`) performs a CPU-safe install by skipping `flash-attn` automatically.
