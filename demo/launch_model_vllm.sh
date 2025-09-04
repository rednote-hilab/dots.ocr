# download model to /path/to/model
if [ -z "$NODOWNLOAD" ]; then
    python3 tools/download_model.py
fi

# register model to vllm
hf_model_path=./weights/DotsOCR  # Path to your downloaded model weights
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`

# launch vllm server
model_name=dotsocr
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name ${model_name} --trust-remote-code

# # run python demo after launch vllm server
# python demo/demo_vllm.py