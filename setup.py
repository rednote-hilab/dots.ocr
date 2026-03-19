from setuptools import setup, find_packages

# 从requirements.txt文件读取依赖
def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().splitlines()
        
setup(
    name='dots_ocr',  
    version='1.0', 
    packages=find_packages(),  
    include_package_data=True,  
    install_requires=parse_requirements('requirements.txt'),  
    description='dots.ocr: Multilingual Document Layout Parsing in one Vision-Language Model',
    url="https://github.com/rednote-hilab/dots.ocr",
    python_requires=">=3.10",
)
conda create -n dots_ocr python=3.12
conda activate dots_ocr

git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr

# Install pytorch, see https://pytorch.org/get-started/previous-versions/ for your cuda version
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .