from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Response
from fastapi.responses import JSONResponse
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
    max_pixels=MAX_PIXELS
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
        raise RuntimeError(f"must use s3 or oss. {str(e)}") from e
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
                output_key = output_key
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


