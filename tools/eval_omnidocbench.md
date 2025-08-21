Here is the step-by-step doc to reproduce OmniDocBench benchmark results.

## 1. Model Env
Here, we use [Docker Image](https://hub.docker.com/r/rednotehilab/dots.ocr) for setup.

## 2. Model Launch
```shell
# dots.ocr parser env
git clone https://github.com/rednote-hilab/dots.ocr.git
cd dots.ocr
pip install -e .
 
# model setup and register
python3 tools/download_model.py
export hf_model_path=./weights/DotsOCR  # Path to your downloaded model weights, Please use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) for the model save path. This is a temporary workaround pending our integration with Transformers.
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from DotsOCR import modeling_dots_ocr_vllm' `which vllm`  # If you downloaded model weights by yourself, please replace `DotsOCR` by your model saved directory name, and remember to use a directory name without periods (e.g., `DotsOCR` instead of `dots.ocr`) 
 
# launch vllm server
CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95  --chat-template-content-format string --served-model-name model --trust-remote-code
```


## 3. Model Inference

```python

from tqdm import tqdm
import json
import argparse
from multiprocessing.pool import ThreadPool, Pool
import shutil
import os


if __name__=="__main__":
    from dots_ocr import DotsOCRParser
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=128,
        help=""
    )
    # parser.add_argument(
    #     "--fitz_preprocess", type=bool, default=False,
    #     help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    # )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    args = parser.parse_args()

    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        # output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    filepath = "/path/to/OmniDocBench.jsonl"  # download OmniDocBench datasets from https://huggingface.co/datasets/opendatalab/OmniDocBench, reformat it to input the images into model
    with open(filepath, 'r') as f:
        list_items = [json.loads(line) for line in f]
    
    results = []
    output_path = "./output_omni.jsonl"
    f_out = open(output_path, 'w')

    tasks = [[item['file_path'], f_out] for item in list_items]

    def _excute(task):
        image_path, f_out = task
        result = dots_ocr_parser.parse_file(
            image_path, 
            prompt_mode="prompt_layout_all_en",
            # prompt_mode="prompt_ocr",
            fitz_preprocess=True,
            )
        results.append(result)
        f_out.write(f"{json.dumps(result, ensure_ascii=False)}\n")
        f_out.flush()

    with ThreadPool(128) as pool:
        with tqdm(total=len(tasks)) as pbar:
            for result in pool.imap(_excute, tasks):
                pbar.update(1)
        pool.close()
        pool.join()   

    f_out.close()

    eval_result_save_dir = "./output_omni/"
    os.makedirs(eval_result_save_dir, exist_ok=True)

    with open(output_path, "r") as f:
        for line in f.readlines():
            item = json.loads(line)[0]
            if 'md_content_nohf_path' in item:
                file_name = os.path.basename(item['md_content_nohf_path']).replace("_nohf", "")
                shutil.copy2(item['md_content_nohf_path'], os.path.join(eval_result_save_dir, file_name))
            else:
                shutil.copy2(item['md_content_path'], eval_result_save_dir)

    print(f"md results saved to {eval_result_save_dir}")
```


## 4. Evaluation
We use [OmniDocBench](https://github.com/opendatalab/OmniDocBench) to evaluate the performance. Prepare the omnidocbench env by yourself, follow the official steps.

```shell
git clone https://github.com/opendatalab/OmniDocBench.git
cd /OmniDocBench

# Follow https://github.com/opendatalab/OmniDocBench?tab=readme-ov-file#environment-setup-and-running
conda create -n omnidocbench python=3.10
conda activate omnidocbench
pip install -r requirements.txt

# Eval. Change the gt&pred path in end2end.yaml to your own, here by our inference steps, prediction data_path set as: /path/to/dots.ocr/output_omni/
python pdf_validation.py --config ./end2end.yaml > evaluation_output.log
 
cat evaluation_output.log
```

./end2end.yaml like:
```yaml
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
    display_formula:
      metric:
        - Edit_dist
        - CDM
    table:
      metric:
        - TEDS
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./OmniDocBench.json  # change to omnidocbench official gt
    prediction:
      data_path: /path/to/dots.ocr/output_omni/  # change to your own output dir
    match_method: quick_match
```

Eval results as follow:
```shell
DATASET_REGISTRY:  ['recogition_text_dataset', 'omnidocbench_single_module_dataset', 'recogition_formula_dataset', 'recogition_table_dataset', 'end2end_dataset', 'recogition_end2end_base_dataset', 'recogition_end2end_table_dataset', 'detection_dataset', 'detection_dataset_simple_format', 'md2md_dataset']
METRIC_REGISTRY:  ['TEDS', 'BLEU', 'METEOR', 'Edit_dist', 'CDM']
EVAL_TASK_REGISTRY:  ['detection_eval', 'end2end_eval', 'recogition_eval']
###### Process:  _quick_match
【Overall】
display_formula CDM is not found
display_formula CDM is not found
----------------------------  --------------------
text_block_Edit_dist_EN       0.031039851583834904
text_block_Edit_dist_CH       0.0643426705744435
display_formula_Edit_dist_EN  0.32843522681423176
display_formula_Edit_dist_CH  0.42557920974720154
display_formula_CDM_EN        -
display_formula_CDM_CH        -
table_TEDS_EN                 88.91012727754615
table_TEDS_CH                 89.0531009325606
table_Edit_dist_EN            0.0943878061222165
table_Edit_dist_CH            0.09173810770062703
reading_order_Edit_dist_EN    0.04079293927450415
reading_order_Edit_dist_CH    0.06625588944827145
overall_EN                    0.12366395594869685
overall_CH                    0.16197896936763587
----------------------------  --------------------


【PDF types】
--------------------------------  ---------
data_source: book                 0.0183191
data_source: PPT2PDF              0.0470068
data_source: research_report      0.0107441
data_source: colorful_textbook    0.0710044
data_source: exam_paper           0.0763102
data_source: magazine             0.0278807
data_source: academic_literature  0.0279
data_source: note                 0.112103
data_source: newspaper            0.0787516
ALL                               0.055183
--------------------------------  ---------


【Layout】
Layout                      Mean         Var
---------------------  ---------  ----------
layout: single_column  0.0267498  0.0187775
layout: double_column  0.0789817  0.0283393
layout: three_column   0.0738766  0.00154036
layout: other_layout   0.115941   0.0336075


【Text Attribute】
--------------------------------------  ---------
text_language: text_english             0.0296053
text_language: text_simplified_chinese  0.106577
text_language: text_en_ch_mixed         0.0888106
text_background: white                  0.0880222
text_background: single_colored         0.0752833
text_background: multi_colored          0.0723353
--------------------------------------  ---------


【Table Attribute】
----------------------------------  --------
language: table_en                  0.876843
language: table_simplified_chinese  0.872133
language: table_en_ch_mixed         0.941139
line: full_line                     0.895071
line: less_line                     0.897401
line: fewer_line                    0.842987
line: wireless_line                 0.847398
with_span: True                     0.865542
with_span: False                    0.881582
include_equation: True              0.839543
include_equation: False             0.884461
include_background: True            0.886555
include_background: False           0.8707
table_layout: vertical              0.884036
table_layout: horizontal            0.875826
----------------------------------  --------

```

> **Notes:** 
> - The metrics reported in the README.md are the average of 5 runs. Each run may show a variance of ±0.005 for the overall_EN and overall_ZH scores.