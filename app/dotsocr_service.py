from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel
import os
from pathlib import Path
import tempfile
import uuid
import json

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
import uvicorn
import boto3
import logging
import asyncio
from botocore.config import Config
import httpx
import re

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(
    title="dotsOCR API",
    description="API for PDF and image text recognition using dotsOCR by Grant",
    version="1.0.0"
)


s3_client = boto3.client('s3')

endpoint = os.getenv('OSS_ENDPOINT')
access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
secret_access_key = os.getenv('OSS_ACCESS_KEY_SECRET')
bucket_name = os.getenv('OSS_BUCKET_NAME')

s3_oss_client = boto3.client(
    's3',
    aws_access_key_id=access_key_id,
    aws_secret_access_key=secret_access_key,
    endpoint_url="https://oss-cn-hongkong.aliyuncs.com",
    config=Config(s3={"addressing_style": "virtual"},
                  signature_version='s3'))



BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

GLOBAL_LOCK_MANAGER = asyncio.Lock() 
PROCESSING_INPUT_LOCKS = {} 
PROCESSING_OUTPUT_LOCKS = {} 

# Initialize parser with default config
class ParseRequest(BaseModel):
    prompt_mode: str = "prompt_layout_all_en"
    fitz_preprocess: bool = False


dots_parser = DotsOCRParser(
    ip="localhost",
    port=8000,
    dpi=200,
    concurrency_limit=16,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)


def _get_existing_page_indices_s3_sync(bucket: str, prefix: str) -> set:
    print(bucket, prefix)
    existing_indices = set()
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)({re.escape('.json')}|{re.escape('.md')}|{re.escape('_nohf.md')})$")

    count = {}
    for page in pages:
        if "Contents" in page:
            for obj in page['Contents']:
                key = obj['Key']
                match = pattern.match(key)
                if match:
                    num = int(match.group(1))
                    count[num] = count.get(num, 0) + 1
                    if count[num] == 3:
                        existing_indices.add(num)
    print(existing_indices)
    return existing_indices

async def _get_existing_page_indices_s3(bucket: str, prefix: str) -> set:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _get_existing_page_indices_s3_sync, bucket, prefix
    )

async def parse(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False,
    parse_type: str = "pdf"  # or "image", default is "pdf"
):
    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and input_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")
    try:
        def parse_s3_path(s3_path: str):
            if is_s3:
                s3_path = s3_path.replace("s3://", "")
            else:
                s3_path = s3_path.replace("oss://", "")
            bucket, *key_parts = s3_path.split("/")
            return bucket, "/".join(key_parts)

        file_bucket, file_key = parse_s3_path(input_s3_path)
        input_file_path = INPUT_DIR / file_bucket / file_key
        input_file_path.parent.mkdir(parents=True, exist_ok=True)


        async with GLOBAL_LOCK_MANAGER:
            if input_s3_path not in PROCESSING_INPUT_LOCKS:
                PROCESSING_INPUT_LOCKS[input_s3_path] = asyncio.Lock()
            if output_s3_path not in PROCESSING_OUTPUT_LOCKS:
                PROCESSING_OUTPUT_LOCKS[output_s3_path] = asyncio.Lock()
        input_lock = PROCESSING_INPUT_LOCKS[input_s3_path]
        output_lock = PROCESSING_OUTPUT_LOCKS[output_s3_path]
        
        async with input_lock:
            async with output_lock:
                # download file from S3
                try:
                    if is_s3:
                        s3_client.download_file(
                            Bucket=file_bucket,
                            Key=file_key,
                            Filename=str(input_file_path)
                        )
                    else:
                        s3_oss_client.download_file(
                            Bucket=file_bucket,
                            Key=file_key,
                            Filename=str(input_file_path)
                        )
                    logging.info(f"download from s3/oss successfully: {input_s3_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from s3/oss: {str(e)}") from e

                output_bucket, output_key = parse_s3_path(output_s3_path)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                # print(output_file_path)
                # print(output_file_name)
                # print(output_md_path)
                # try:
                if parse_type == "image":
                    results = await dots_parser.parse_image(
                        input_path=input_file_path,
                        filename=output_file_name,
                        prompt_mode=prompt_mode,
                        save_dir=output_file_path,
                        fitz_preprocess=fitz_preprocess
                    )
                else:
                    results = await dots_parser.parse_pdf(
                        input_path=input_file_path,
                        filename=output_file_name,
                        prompt_mode=prompt_mode,
                        save_dir=output_file_path
                    )


                # Format results for all pages
                formatted_results = []
                all_md_content = []
                all_md_nohf_content = []
                for result in results:
                    layout_info_path = result.get('layout_info_path')
                    full_layout_info = {}
                    if layout_info_path and os.path.exists(layout_info_path):
                        try:
                            with open(layout_info_path, 'r', encoding='utf-8') as f:
                                full_layout_info = json.load(f)
                        except Exception as e:
                            print(f"WARNING: Failed to read layout info file: {str(e)}")
                    formatted_results.append({
                        "page_no": result.get('page_no'),
                        "full_layout_info": full_layout_info
                    })

                    md_content_path = result.get('md_content_path')
                    md_content = ""
                    if md_content_path and os.path.exists(md_content_path):
                        try:
                            with open(md_content_path, 'r', encoding='utf-8') as f:
                                md_content = f.read()
                        except Exception as e:
                            print(f"WARNING: Failed to read markdown file: {str(e)}")
                    all_md_content.append(md_content)

                    md_content_nohf_path = result.get('md_content_nohf_path')
                    md_content_nohf = ""
                    if md_content_nohf_path and os.path.exists(md_content_nohf_path):
                        try:
                            with open(md_content_nohf_path, 'r', encoding='utf-8') as f:
                                md_content_nohf = f.read()
                        except Exception as e:
                            print(f"WARNING: Failed to read markdown file: {str(e)}")
                    all_md_nohf_content.append(md_content_nohf)

                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_results, f, indent=4, ensure_ascii=False)
                with open(output_md_path, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(all_md_content))
                with open(output_md_nohf_path, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(all_md_nohf_content))


                # upload output files to S3
                try:
                    if is_s3:
                        s3_client.upload_file(
                            Bucket=output_bucket,
                            Key=f"{output_key}/{output_file_name}.md",
                            Filename=str(output_md_path)
                        )
                        s3_client.upload_file(
                            Bucket=output_bucket,
                            Key=f"{output_key}/{output_file_name}_nohf.md",
                            Filename=str(output_md_nohf_path)
                        )
                        s3_client.upload_file(
                            Bucket=output_bucket,
                            Key=f"{output_key}/{output_file_name}.json",
                            Filename=str(output_json_path)
                        )
                    else:
                        s3_oss_client.upload_file(
                            Bucket=output_bucket,
                            Key=f"{output_key}/{output_file_name}.md",
                            Filename=str(output_md_path)
                        )
                        s3_oss_client.upload_file(
                            Bucket=output_bucket,
                            Key=f"{output_key}/{output_file_name}_nohf.md",
                            Filename=str(output_md_nohf_path)
                        )
                        s3_oss_client.upload_file(
                            Bucket=output_bucket,
                            Key=f"{output_key}/{output_file_name}.json",
                            Filename=str(output_json_path)
                        )
                    logging.info(f"upload from s3/oss successfully: {output_file_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to upload file from s3/oss: {str(e)}") from e

        return {
            "success": True,
            "total_pages": len(results),
            "output_s3_path": output_s3_path
        }

        # finally:
        #     # Ensure cleanup even if parser fails
        #     if os.path.exists(temp_path):
        #         os.remove(temp_path)
        #     if os.path.exists(temp_dir):
        #         os.rmdir(temp_dir)
        #     if os.path.exists(output_dir):
        #         for f in os.listdir(output_dir):
        #             os.remove(os.path.join(output_dir, f))
        #         os.rmdir(output_dir)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/parse/image")
async def parse_image(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except TypeError:
        raise HTTPException(status_code=400, detail="Invalid filename format")
    if file_ext not in ['.jpg', '.jpeg', '.png']:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Supported: .jpg, .jpeg, .png")
    
    return await parse(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, parse_type="image")

@app.post("/parse/pdf")
async def parse_pdf(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except TypeError:
        raise HTTPException(status_code=400, detail="Invalid filename format")
    if file_ext not in ['.pdf']:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Supported: .pdf")

    return await parse(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, parse_type="pdf")


@app.post("/parse/file")
async def parse_file(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    try:
        try:
            file_ext = Path(input_s3_path).suffix.lower()
        except TypeError:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        if file_ext == '.pdf':
            return await parse_pdf(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return await parse_image(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#------------------------------------stream-----------------------------------------#

async def upload_file(bucket, key, local_path, is_s3):
    if not local_path or not os.path.exists(local_path):
        return None
    try:
        if is_s3:
            s3_client.upload_file(Bucket=bucket, Key=key, Filename=local_path)
        else:
            s3_oss_client.upload_file(Bucket=bucket, Key=key, Filename=local_path)
        s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
        logging.info(f"Successfully uploaded {local_path} to {s3_full_path}")
        # os.remove(local_path)
        return s3_full_path
    except Exception as e:
        logging.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
        return None
        
async def download_file(bucket, key, local_path, is_s3):
    if not bucket or not key or not local_path:
        logging.warning("Bucket, key, and local_path must be specified.")
        return None
    try:
        if is_s3:
            s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)
        else:
            s3_oss_client.download_file(Bucket=bucket, Key=key, Filename=local_path)
        s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
        logging.info(f"Successfully downloaded {local_path} to {s3_full_path}")
        return s3_full_path
    except Exception as e:
        logging.error(f"Failed to download s3://{bucket}/{key} to {local_path}: {e}")
        return None

async def stream_page_by_page_upload_generator(
    input_s3_path: str,
    output_s3_path: str,
    prompt_mode: str
):
    """
    Parses a PDF file, and streams the output files for each page directly to S3/OSS
    as they are completed.
    """
    
    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and input_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")
    
    try:
        def parse_s3_path(s3_path: str):
            if is_s3:
                s3_path = s3_path.replace("s3://", "")
            else:
                s3_path = s3_path.replace("oss://", "")
            bucket, *key_parts = s3_path.split("/")
            return bucket, "/".join(key_parts)

        file_bucket, file_key = parse_s3_path(input_s3_path)
        input_file_path = INPUT_DIR / file_bucket / file_key
        input_file_path.parent.mkdir(parents=True, exist_ok=True)

        async with GLOBAL_LOCK_MANAGER:
            if input_s3_path not in PROCESSING_INPUT_LOCKS:
                PROCESSING_INPUT_LOCKS[input_s3_path] = asyncio.Lock()
            if output_s3_path not in PROCESSING_OUTPUT_LOCKS:
                PROCESSING_OUTPUT_LOCKS[output_s3_path] = asyncio.Lock()
        input_lock = PROCESSING_INPUT_LOCKS[input_s3_path]
        output_lock = PROCESSING_OUTPUT_LOCKS[output_s3_path]

        async with input_lock:
            async with output_lock:

                # download file from S3
                await download_file(
                    bucket=file_bucket, key=file_key, local_path=str(input_file_path), is_s3 = is_s3
                )
                
                # prepare local path
                output_bucket, output_key = parse_s3_path(output_s3_path)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                output_file_path.mkdir(parents=True, exist_ok=True)
            
                # print(output_bucket, output_key)
                # print(output_file_name)
                # print(output_file_path)
                # print(output_md_path)
                s3_prefix = f"{output_key}/{output_file_name}_page_"
                existing_pages = await _get_existing_page_indices_s3(output_bucket, s3_prefix)

                # parse the PDF file and upload each page's output files
                all_paths_to_upload = []
                async for result in dots_parser.parse_pdf_stream(
                    input_path=input_file_path,
                    filename=Path(input_file_path).stem,
                    prompt_mode=prompt_mode,
                    save_dir=output_file_name,
                    existing_pages=existing_pages
                ):
                    page_no = result.get('page_no', -1)
                    
                    page_upload_tasks = []
                    paths_to_upload = {
                        'md': result.get('md_content_path'),
                        'md_nohf': result.get('md_content_nohf_path'),
                        'json': result.get('layout_info_path')
                    }
                    for file_type, local_path in paths_to_upload.items():
                        if local_path:
                            file_name = Path(local_path).name
                            s3_key = f"{output_key}/{file_name}"
                            task = asyncio.create_task(
                                upload_file(output_bucket, s3_key, local_path, is_s3)
                            )
                            page_upload_tasks.append(task)
                    uploaded_paths_for_page = await asyncio.gather(*page_upload_tasks)                    

                    paths_to_upload['page_no'] = page_no
                    all_paths_to_upload.append(paths_to_upload)
                    page_response = {
                        "success": True,
                        "message": "parse success",
                        "page_no": page_no,
                        "uploaded_files": [path for path in uploaded_paths_for_page if path]
                    }
                    yield json.dumps(page_response) + "\n"


                # combine all page to upload
                ## download exist pages
                for page_no in existing_pages:
                    json_path = f"{output_file_path}/{output_file_name}_page_{page_no}.json"
                    md_path = f"{output_file_path}/{output_file_name}_page_{page_no}.md"
                    md_nohf_path = f"{output_file_path}/{output_file_name}_page_{page_no}_nohf.md"
                    paths_to_download = {
                        'md': md_path,
                        'md_nohf': md_nohf_path,
                        'json': json_path
                    }
                    downloaded_paths_for_page = []
                    if not (os.path.exists(json_path) and os.path.exists(md_nohf_path) and os.path.exists(md_path)):
                        page_download_tasks = []
                        for file_type, local_path in paths_to_download.items():
                            if local_path:
                                file_name = Path(local_path).name
                                s3_key = f"{output_key}/{file_name}"
                                task = asyncio.create_task(
                                    download_file(output_bucket, s3_key, local_path, is_s3)
                                )
                                page_download_tasks.append(task)
                        downloaded_paths_for_page = await asyncio.gather(*page_download_tasks)
                        
                        
                    paths_to_download['page_no'] = page_no
                    all_paths_to_upload.append(paths_to_download)
                    page_response = {
                        "success": True,
                        "message": "parse output already exists in s3/oss",
                        "page_no": page_no,
                        "downloaded_files": [path for path in downloaded_paths_for_page if path]
                    }
                    yield json.dumps(page_response) + "\n"
                        
                ## combine output
                all_paths_to_upload.sort(key=lambda item: item['page_no'])
                output_files = {}
                output_files['md'] = open(output_md_path, 'w', encoding='utf-8')
                output_files['json'] = open(output_json_path, 'w', encoding='utf-8')
                output_files['md_nohf'] = open(output_md_nohf_path, 'w', encoding='utf-8')
                for p in all_paths_to_upload:
                    page_no = p.pop('page_no')
                    for file_type, local_path in p.items():
                        with open(local_path, 'r', encoding='utf-8') as input_file:
                            file_content = input_file.read()
                        output_files[file_type].write(file_content)
                        output_files[file_type].write("\n\n")

                await upload_file(output_bucket, f"{output_key}/{output_file_name}.md", str(output_md_path), is_s3)
                await upload_file(output_bucket, f"{output_key}/{output_file_name}_nohf.md", str(output_md_nohf_path), is_s3)
                await upload_file(output_bucket, f"{output_key}/{output_file_name}.json", str(output_json_path), is_s3)

                final_response = {
                    "success": True,
                    "total_pages": len(all_paths_to_upload),
                    "output_s3_path": output_s3_path
                }
                yield json.dumps(final_response) + '\n'
                        
    except Exception as e:
        error_msg = json.dumps({"success": False, "detail": str(e)})
        yield error_msg + "\n"

@app.post("/parse/pdf_stream")
async def parse_pdf_stream(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
):
    if not input_s3_path.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Invalid file format. This endpoint only supports .pdf")

    generator = stream_page_by_page_upload_generator(
        input_s3_path=input_s3_path,
        output_s3_path=output_s3_path,
        prompt_mode=prompt_mode,
    )
    return StreamingResponse(generator, media_type="application/x-ndjson")

#---------------------------directly send file to parser---------------------------

@app.post("/directly_parse/image")
async def parse_image_old(
    file: UploadFile = File(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    """Parse a single image file"""
    try:
        # Validate upload file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        try:
            file_ext = Path(file.filename).suffix.lower()
        except TypeError:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        if file_ext not in ['.jpg', '.jpeg', '.png']:
            raise HTTPException(
                status_code=400, detail="Invalid image format. Supported: .jpg, .jpeg, .png")

        # Verify file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        await file.seek(0)  # Reset file pointer after reading

        # Create temp file with debug logging
        print(f"DEBUG: Creating temp file for {file.filename}")
        temp_dir = tempfile.mkdtemp()
        print(f"DEBUG: Created temp dir {temp_dir}")

        temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}{file_ext}")
        print(f"DEBUG: Temp file path will be {temp_path}")

        # Save uploaded file with validation
        file_content = await file.read()
        print(f"DEBUG: Read {len(file_content)} bytes from upload")

        if not isinstance(temp_path, (str, bytes, os.PathLike)):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid temp path type: {type(temp_path)}"
            )

        with open(temp_path, "wb") as buffer:
            buffer.write(file_content)
        print(f"DEBUG: Saved {len(file_content)} bytes to {temp_path}")

        # Verify file was written
        if not os.path.exists(temp_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to create temp file"
            )
        print(f"DEBUG: Temp file exists at {temp_path}")

        # Process the image with debug logging
        print(f"DEBUG: Calling parser with: {temp_path}")

        # Get absolute path and verify file exists
        abs_temp_path = os.path.abspath(temp_path)
        if not os.path.exists(abs_temp_path):
            raise HTTPException(
                status_code=500,
                detail=f"Temp file not found at {abs_temp_path}"
            )

        # Create and clean output directory
        output_dir = tempfile.mkdtemp()
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

        try:
            results = dots_parser.parse_image(
                input_path=abs_temp_path,
                filename="api_image",
                prompt_mode=prompt_mode,
                save_dir=output_dir,
                fitz_preprocess=fitz_preprocess
            )
            print(f"DEBUG: Parser completed successfully=={results}")

            # Extract and return the relevant data
            result = results[0]  # Single result for image
            layout_info_path = result.get('layout_info_path')
            full_layout_info = {}
            if layout_info_path and os.path.exists(layout_info_path):
                try:
                    with open(layout_info_path, 'r', encoding='utf-8') as f:
                        full_layout_info = json.load(f)
                except Exception as e:
                    print(f"WARNING: Failed to read layout info file: {str(e)}")

        except Exception as e:
            print(f"DEBUG: Parser error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Parser error: {str(e)}"
            )
        finally:
            # Ensure cleanup even if parser fails
            if os.path.exists(abs_temp_path):
                os.remove(abs_temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, f))
                os.rmdir(output_dir)

        return {
            "success": True,
            "total_pages": len(results),
            "results": [{"page_no": 0,
                         "full_layout_info": full_layout_info}]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/directly_parse/pdf")
async def parse_pdf_old(
    file: UploadFile = File(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    """Parse a PDF file (multi-page)"""
    try:
        # Validate upload file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        try:
            if Path(file.filename).suffix.lower() != '.pdf':
                raise HTTPException(
                    status_code=400, detail="Invalid PDF format. Only .pdf files accepted")
        except TypeError:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        # Verify file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        await file.seek(0)  # Reset file pointer after reading

        # Create temp file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}.pdf")

        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        # Create and clean output directory
        output_dir = tempfile.mkdtemp()
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

        try:
            # Process the PDF
            results = dots_parser.parse_pdf(
                input_path=temp_path,
                filename="api_pdf",
                prompt_mode=prompt_mode,
                save_dir=output_dir
            )
            print(f"DEBUG: Parser completed successfully=={results}")
            # Format results for all pages
            formatted_results = []
            for result in results:
                layout_info_path = result.get('layout_info_path')
                full_layout_info = {}
                if layout_info_path and os.path.exists(layout_info_path):
                    try:
                        with open(layout_info_path, 'r', encoding='utf-8') as f:
                            full_layout_info = json.load(f)
                    except Exception as e:
                        print(f"WARNING: Failed to read layout info file: {str(e)}")

                formatted_results.append({
                    "page_no": result.get('page_no'),
                    "full_layout_info": full_layout_info
                })

            return {
                "success": True,
                "total_pages": len(results),
                "results": formatted_results
            }

        finally:
            # Ensure cleanup even if parser fails
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            if os.path.exists(output_dir):
                for f in os.listdir(output_dir):
                    os.remove(os.path.join(output_dir, f))
                os.rmdir(output_dir)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/directly_parse/file")
async def parse_file_old(
    file: UploadFile = File(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    """Automatically detect file type and parse accordingly"""
    try:
        # Validate upload file
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        try:
            file_ext = Path(file.filename).suffix.lower()
        except TypeError:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        # Verify file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        await file.seek(0)  # Reset file pointer after reading

        if file_ext == '.pdf':
            return await parse_pdf_old(file, prompt_mode, fitz_preprocess)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return await parse_image_old(file, prompt_mode, fitz_preprocess)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

TARGET_URL = "http://localhost:8000/health"
async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(TARGET_URL, timeout=5.0)

        headers_to_exclude = {
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "connection",
        }
        proxied_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in headers_to_exclude
        }

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=proxied_headers,
            media_type=response.headers.get("content-type")
        )
    except httpx.ConnectError as e:
        return JSONResponse(
            status_code=503,
            content={"success": False, "status": 503, "detail": f"Health check failed: Unable to connect to DotsOCR service. Error: {e}"}
        )
    except httpx.TimeoutException as e:
        return JSONResponse(
            status_code=504,
            content={"success": False, "status": 504,"detail": f"Health check failed: Request to DotsOCR service timed out. Error: {e}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "status": 500,"detail": f"An unexpected error occurred during health check. Error: {e}"}
        )
    

@app.get("/health")
async def health():
    return await health_check()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6008)


