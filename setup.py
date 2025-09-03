from setuptools import setup, find_packages

# 从requirements.txt文件读取依赖（默认排除 flash-attn/accelerate，作为可选GPU加速安装）
def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.read().splitlines()]
    # 过滤空行、注释，以及GPU相关的可选依赖
    filtered = []
    for ln in lines:
        if not ln or ln.startswith('#'):
            continue
        lower = ln.lower()
        if lower.startswith('flash-attn') or lower.startswith('flash_attn') or lower.startswith('accelerate'):
            # 这些在 extras_require['gpu'] 中提供
            continue
        filtered.append(ln)
    return filtered

setup(
    name='dots_ocr',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        # 使用 GPU 时，可以通过 `pip install -e .[gpu]` 安装以下可选依赖
        'gpu': [
            'flash-attn==2.8.0.post2',
            'accelerate',
        ],
    },
    description='dots.ocr: Multilingual Document Layout Parsing in one Vision-Language Model',
    url="https://github.com/rednote-hilab/dots.ocr",
    python_requires=">=3.10",
)
