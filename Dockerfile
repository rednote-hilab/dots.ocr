FROM rednotehilab/dots.ocr:vllm-openai-v0.9.1

# docker run --name dots-ocr-container -it -v .:/DotsOCR --runtime=nvidia --gpus=all --privileged --entrypoint bash rednotehilab/dots.ocr:vllm-openai-v0.9.1
# docker run --name dots-ocr-container -p 51234:5000 --restart=always -it -v /sn640/ai-apps/dots.ocr:/workspace --runtime=nvidia --gpus=all --privileged --entrypoint bash rednotehilab/dots.ocr:vllm-openai-v0.9.1 /workspace/start.sh
# docker stop dots-ocr-container ; docker start dots-ocr-container
# docker logs -f dots-ocr-container
# docker stop dots-ocr-container ; docker start dots-ocr-container

# docker build . -t docker.io/am009/dots.ocr:latest
# docker run --name dots-ocr-container -d --runtime=nvidia --gpus=all -p 51234:5000 --privileged docker.io/am009/dots.ocr:latest

COPY . /DotsOCR

RUN cd /DotsOCR && \
    pip install flask flask_cors --ignore-installed && \
    pip install -e .

ENTRYPOINT ["/bin/bash"]
CMD ["/DotsOCR/start.sh"]

# sed -i 's/bf16=True/bf16=False/' weights/DotsOCR/modeling_dots_vision.py


# pip install flash_attn_triton==0.1.1
# rm -rf /usr/local/lib/python3.12/dist-packages/flash_attn
