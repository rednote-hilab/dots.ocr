from fastapi import FastAPI, Form, HTTPException, UploadFile, File, Response
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Literal, List
from collections import deque
import os
from pathlib import Path
import tempfile
import uuid
import json
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from contextlib import asynccontextmanager 

from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import MIN_PIXELS, MAX_PIXELS
import uvicorn
import logging
import asyncio
import httpx
import re
from app.utils.storage import StorageManager
from app.utils.redis import RedisConnector
from app.utils.hash import compute_md5

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


NUM_WORKERS = 4 
WORKER_TASKS: List[asyncio.Task] = []
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Starting up {NUM_WORKERS} worker tasks...")
    for i in range(NUM_WORKERS):
        task = asyncio.create_task(worker(f"Worker-{i}"))
        WORKER_TASKS.append(task)
    
    yield

    logging.info("Shutting down and canceling worker tasks...")
    for task in WORKER_TASKS:
        task.cancel()
    
    await asyncio.gather(*WORKER_TASKS, return_exceptions=True)
    logging.info("All worker tasks have been canceled.")

app = FastAPI(
    title="dotsOCR API",
    description="API for PDF and image text recognition using dotsOCR by Grant",
    version="1.0.0",
    lifespan=lifespan 
)



BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)

GLOBAL_LOCK_MANAGER = asyncio.Lock() 
PROCESSING_INPUT_LOCKS = {} 
PROCESSING_OUTPUT_LOCKS = {} 

dots_parser = DotsOCRParser(
    ip="localhost",
    port=8000,
    dpi=200,
    concurrency_limit=16,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)


storage_manager = StorageManager()
redis_connector = RedisConnector()

def parse_s3_path(s3_path: str, is_s3):
    if is_s3:
        s3_path = s3_path.replace("s3://", "")
    else:
        s3_path = s3_path.replace("oss://", "")
    bucket, *key_parts = s3_path.split("/")
    return bucket, "/".join(key_parts)

class JobResponseModel(BaseModel):
    job_id: str
    knowledgebase_id: str
    workspace_id: str
    status: Literal["pending", "retrying", "processing", "completed", "failed", "canceled"] # canceled havn't implemented
    message: str
    is_s3: bool = True

    input_s3_path: str
    output_s3_path: str
    page_prefix: str = None
    json_url: str = None
    md_url: str = None
    md_nohf_url: str = None

    prompt_mode: str = "prompt_layout_all_en"
    fitz_preprocess: bool = False

    def transform_to_map(self):
        mapping = {
            "url": self.output_s3_path,
            "knowledgebaseId": self.knowledgebase_id,
            "workspaceId": self.workspace_id,
            "markdownUrl": self.md_url,
            "jsonUrl": self.json_url,
            "status": self.status
        }
        return {k: (v if v is not None else "") for k, v in mapping.items()}
        
async def update_redis(job: JobResponseModel):
    redis_connector.hset(f"OCRJobId:{job.job_id}", mapping=job.transform_to_map())


JobResponseDict: Dict[str, JobResponseModel] = {}
JobLocks: Dict[str, asyncio.Lock] = {}
JobQueue = asyncio.Queue()

@app.post("/status")
async def status_check(OCRJobId: str = Form(...)):
    if OCRJobId not in JobResponseDict:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    async with JobLocks[OCRJobId]:
        JobResponse = JobResponseDict[OCRJobId]
        status = JobResponse.status
        message = JobResponse.message
        page_prefix = JobResponse.page_prefix

        if status == "completed":
            return JobResponse
        elif status == "failed":
            return JSONResponse({"OCRJobId": OCRJobId, "status": status, "message": message, "page_prefix": page_prefix}, status_code=500)
        return JSONResponse({"OCRJobId": OCRJobId, "status": status, "message": message, "page_prefix": page_prefix}, status_code=202)

async def stream_and_upload_generator(
    JobResponse: JobResponseModel
):
    input_s3_path = JobResponse.input_s3_path
    output_s3_path = JobResponse.output_s3_path
    is_s3 = JobResponse.is_s3

    try:

        file_bucket, file_key = parse_s3_path(input_s3_path, is_s3)
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
                    await storage_manager.download_file(
                        bucket=file_bucket, key=file_key, local_path=str(input_file_path), is_s3 = is_s3
                    )
                    logging.info(f"download from s3/oss successfully: {input_s3_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from s3/oss: {str(e)}") from e
                
                # compute MD5 hash of the input file
                try:
                    file_md5 = compute_md5(str(input_file_path))
                    logging.info(f"MD5 hash of input file {input_s3_path}: {file_md5}")
                except Exception as e:
                    logging.error(f"Failed to compute MD5 hash for {input_s3_path}: {str(e)}")
                    raise RuntimeError(f"Failed to compute MD5 hash: {str(e)}") from e
                
                # prepare local path
                output_bucket, output_key = parse_s3_path(output_s3_path, is_s3)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md5_path = output_md_path.with_suffix(".md5")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                output_file_path.mkdir(parents=True, exist_ok=True)
            
                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = await storage_manager.check_existing_results_sync(
                    bucket=output_bucket, prefix=f"{output_key}/{output_file_name}", is_s3=is_s3
                )
            
                # If so, download md5 file and compare hashes
                if md5_exists:
                    try:
                        await storage_manager.download_file(
                            bucket=output_bucket,
                            key=f"{output_key}/{output_file_name}.md5",
                            local_path=str(output_md5_path),
                            is_s3=is_s3
                        )
                        with open(output_md5_path, 'r') as f:
                            existing_md5 = f.read().strip()
                        if existing_md5 == file_md5:
                            if all_files_exist:
                                logging.info(f"Output files already exist in S3 and MD5 matches for {input_s3_path}. Skipping processing.")
                                skip_response = {
                                    "success": True,
                                    "total_pages": 0,
                                    "output_s3_path": output_s3_path,
                                    "message": "Output files already exist and MD5 matches. Skipped processing."
                                }
                                yield json.dumps(skip_response) + "\n"
                                return
                            logging.info(f"MD5 matches for {input_s3_path}, but some output files are missing. Reprocessing the file.")
                        else:
                            # clean the whole output directory in S3
                            print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                            logging.info(f"MD5 mismatch for {input_s3_path}. Reprocessing the file.")
                            await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    except Exception as e:
                        logging.warning(f"Failed to verify existing MD5 hash for {input_s3_path}: {str(e)}. Reprocessing the file.")
                else:
                    # clean the whole output directory in S3 for safety
                    print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                    logging.info(f"No MD5 hash found for {input_s3_path}. Cleaning output directory.")
                    await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)

                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                with open(output_md5_path, 'w') as f:
                    f.write(file_md5)
                logging.info(f"Saved MD5 hash to {output_md5_path}")
                
                # Upload MD5 hash file to S3/OSS
                try:
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md5", str(output_md5_path), is_s3
                    )
                except Exception as e:
                    logging.warning(f"Failed to upload MD5 hash file to s3/oss: {str(e)}")

            
                # print(output_bucket, output_key)
                # print(output_file_name)
                # print(output_file_path)
                # print(output_md_path)

                JobResponse.json_url = f"{output_s3_path}/{output_file_name}.json"
                JobResponse.md_url = f"{output_s3_path}/{output_file_name}.md"
                JobResponse.md_nohf_url = f"{output_s3_path}/{output_file_name}_nohf.md"
                JobResponse.page_prefix = f"{output_s3_path}/{output_file_name}_page_"
                
                s3_prefix = f"{output_key}/{output_file_name}_page_"
                existing_pages = await storage_manager._get_existing_page_indices_s3(output_bucket, s3_prefix)
 
                # parse the PDF file and upload each page's output files
                all_paths_to_upload = []
                async for result in dots_parser.parse_pdf_stream(
                    input_path=input_file_path,
                    filename=Path(input_file_path).stem,
                    prompt_mode=JobResponse.prompt_mode,
                    save_dir=output_file_path,
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
                                storage_manager.upload_file(output_bucket, s3_key, local_path, is_s3)
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
                                    storage_manager.download_file(
                                        bucket=output_bucket, 
                                        key=s3_key, 
                                        local_path=local_path, 
                                        is_s3=is_s3
                                    )
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
                try:
                    output_files['md'] = open(output_md_path, 'w', encoding='utf-8')
                    output_files['json'] = open(output_json_path, 'w', encoding='utf-8')
                    output_files['md_nohf'] = open(output_md_nohf_path, 'w', encoding='utf-8')
                    all_json_data = []
                    for p in all_paths_to_upload:
                        page_no = p.pop('page_no')
                        for file_type, local_path in p.items():
                            if file_type == 'json':
                                try:
                                    with open(local_path, 'r', encoding='utf-8') as input_file:
                                        data = json.load(input_file)
                                    data = {"page_no": page_no, **data}
                                    all_json_data.append(data)
                                except Exception as e:
                                    print(f"WARNING: Failed to read layout info file {local_path}: {str(e)}")
                                    all_json_data.append({"page_no": page_no})
                            else:
                                try:
                                    with open(local_path, 'r', encoding='utf-8') as input_file:
                                        file_content = input_file.read()
                                    output_files[file_type].write(file_content)
                                    output_files[file_type].write("\n\n")
                                except Exception as e:
                                    print(f"WARNING: Failed to read {file_type} file {local_path}: {str(e)}")
                    json.dump(all_json_data, output_files['json'], indent=4, ensure_ascii=False)
                finally:
                    # Ensure all file handles are properly closed
                    for file_handle in output_files.values():
                        if hasattr(file_handle, 'close'):
                            file_handle.close()
                
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}.md", str(output_md_path), is_s3)
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}_nohf.md", str(output_md_nohf_path), is_s3)
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}.json", str(output_json_path), is_s3)

                final_response = {
                    "success": True,
                    "total_pages": len(all_paths_to_upload),
                    "output_s3_path": output_s3_path
                }
                yield json.dumps(final_response) + '\n'
                        
    except Exception as e:
        error_msg = json.dumps({"success": False, "detail": str(e)})
        yield error_msg + "\n"


RETRY_TIMES = 3
@retry(
    stop=stop_after_attempt(RETRY_TIMES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True
)
async def attempt_to_process_job(job: JobResponseModel):
    attempt_num = attempt_to_process_job.retry.statistics.get('attempt_number', 0) + 1
    if attempt_num == 1:
        job.status = "processing"
        await update_redis(job)
    else:
        job.status = "retrying"
        await update_redis(job)
    job.message = f"Processing job, attempt number: {attempt_num}"

    try:
        async for page_result in stream_and_upload_generator(job):
            pass
    except Exception as e:
        logging.error(f"Job {job.job_id} failed on attempt {attempt_num} with error: {e}")
        raise 

async def worker(worker_id: str):
    print(f"{worker_id} started")
    while True:
        try:
            job_id = await JobQueue.get()
            
            JobResponse = JobResponseDict.get(job_id)
            if not JobResponse:
                logging.error(f"{worker_id}: Job ID '{job_id}' found in queue but not in JobResponseDict. Discarding task.")
                JobQueue.task_done()
                continue

            try:
                await attempt_to_process_job(JobResponse)
                logging.info(f"Job {JobResponse.job_id} successfully processed.")
            except Exception as e:
                logging.error(f"Job {JobResponse.job_id} failed after 5 attempts. Final error: {e}", exc_info=True)
                JobResponse.status = "failed"
                JobResponse.message = f"Job failed after multiple retries. Final error: {str(e)}"
                await update_redis(JobResponse)

            JobResponse.status = "completed"
            JobResponse.message = "Job completed successfully"
            await update_redis(JobResponse)
            JobQueue.task_done()
            
        except asyncio.CancelledError:
            logging.info(f"{worker_id} is shutting down.")
            break
        except Exception as e:
            logging.error(f"Unexpected error in {worker_id}: {e}")

@app.post("/parse/file")
async def parse_file(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    knowledgebaseId: str = Form(...),
    workspaceId: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except (TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Invalid input_s3_path format")

    supported_formats = ['.pdf', '.jpg', '.jpeg', '.png']
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats are: {', '.join(supported_formats)}"
        )
    
    OCRJobId = knowledgebaseId + workspaceId + str(uuid.uuid4())
    
    
    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")

    JobResponse = JobResponseModel(
        job_id=OCRJobId,
        status="pending",
        knowledgebase_id=knowledgebaseId,
        workspace_id=workspaceId,
        message="Job is pending",
        is_s3=is_s3,
        input_s3_path=input_s3_path,
        output_s3_path=output_s3_path,
        prompt_mode=prompt_mode,
        fitz_preprocess=fitz_preprocess
    )
    JobResponseDict[OCRJobId] = JobResponse
    JobLocks[OCRJobId] = asyncio.Lock()
    await update_redis(JobResponse)
    await JobQueue.put(OCRJobId)

    return JSONResponse({"OCRJobId": OCRJobId}, status_code=200)

#------------------------------------stream-----------------------------------------


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
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")
    
    try:
        file_bucket, file_key = parse_s3_path(input_s3_path, is_s3)
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
                    await storage_manager.download_file(
                        bucket=file_bucket, key=file_key, local_path=str(input_file_path), is_s3 = is_s3
                    )
                    logging.info(f"download from s3/oss successfully: {input_s3_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from s3/oss: {str(e)}") from e
                
                # compute MD5 hash of the input file
                try:
                    file_md5 = compute_md5(str(input_file_path))
                    logging.info(f"MD5 hash of input file {input_s3_path}: {file_md5}")
                except Exception as e:
                    logging.error(f"Failed to compute MD5 hash for {input_s3_path}: {str(e)}")
                    raise RuntimeError(f"Failed to compute MD5 hash: {str(e)}") from e
                
                # prepare local path
                output_bucket, output_key = parse_s3_path(output_s3_path, is_s3)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md5_path = output_md_path.with_suffix(".md5")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                output_file_path.mkdir(parents=True, exist_ok=True)
            
                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = await storage_manager.check_existing_results_sync(
                    bucket=output_bucket, prefix=f"{output_key}/{output_file_name}", is_s3=is_s3
                )
                
                # If so, download md5 file and compare hashes
                if md5_exists:
                    try:
                        await storage_manager.download_file(
                            bucket=output_bucket,
                            key=f"{output_key}/{output_file_name}.md5",
                            local_path=str(output_md5_path),
                            is_s3=is_s3
                        )
                        with open(output_md5_path, 'r') as f:
                            existing_md5 = f.read().strip()
                        if existing_md5 == file_md5:
                            if all_files_exist:
                                logging.info(f"Output files already exist in S3 and MD5 matches for {input_s3_path}. Skipping processing.")
                                skip_response = {
                                    "success": True,
                                    "total_pages": 0,
                                    "output_s3_path": output_s3_path,
                                    "message": "Output files already exist and MD5 matches. Skipped processing."
                                }
                                yield json.dumps(skip_response) + "\n"
                                return
                            logging.info(f"MD5 matches for {input_s3_path}, but some output files are missing. Reprocessing the file.")
                        else:
                            # clean the whole output directory in S3
                            print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                            logging.info(f"MD5 mismatch for {input_s3_path}. Reprocessing the file.")
                            await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    except Exception as e:
                        logging.warning(f"Failed to verify existing MD5 hash for {input_s3_path}: {str(e)}. Reprocessing the file.")
                else:
                    # clean the whole output directory in S3 for safety
                    print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                    logging.info(f"No MD5 hash found for {input_s3_path}. Cleaning output directory.")
                    await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    
                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                with open(output_md5_path, 'w') as f:
                    f.write(file_md5)
                logging.info(f"Saved MD5 hash to {output_md5_path}")
                
                # Upload MD5 hash file to S3/OSS
                try:
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md5", str(output_md5_path), is_s3
                    )
                except Exception as e:
                    logging.warning(f"Failed to upload MD5 hash file to s3/oss: {str(e)}")
            
                # print(output_bucket, output_key)
                # print(output_file_name)
                # print(output_file_path)
                # print(output_md_path)
                s3_prefix = f"{output_key}/{output_file_name}_page_"
                existing_pages = await storage_manager._get_existing_page_indices_s3(output_bucket, s3_prefix)

                # parse the PDF file and upload each page's output files
                all_paths_to_upload = []
                try:
                    logging.info("About to start parse_pdf_stream async iteration")
                    async for result in dots_parser.parse_pdf_stream(
                        input_path=input_file_path,
                        filename=Path(input_file_path).stem,
                        prompt_mode=prompt_mode,
                        save_dir=output_file_path,
                        existing_pages=existing_pages
                    ):
                        logging.info(f"Received result from parse_pdf_stream: {result} (type: {type(result)})")
                        
                        try:
                            page_no = result.get('page_no', -1)
                            logging.info(f"Successfully got page_no: {page_no}")
                        except Exception as e:
                            logging.error(f"Error getting page_no: {str(e)}")
                            logging.error(f"Result keys: {list(result.keys()) if hasattr(result, 'keys') else 'No keys method'}")
                            raise
                        
                        page_upload_tasks = []
                        try:
                            md_path = result.get('md_content_path')
                            md_nohf_path = result.get('md_content_nohf_path')
                            json_path = result.get('layout_info_path')
                            logging.info(f"Got paths: md={md_path}, md_nohf={md_nohf_path}, json={json_path}")
                            
                            paths_to_upload = {
                                'md': md_path,
                                'md_nohf': md_nohf_path,
                                'json': json_path
                            }
                            logging.info(f"Successfully created paths_to_upload: {paths_to_upload}")
                        except Exception as e:
                            logging.error(f"Error creating paths_to_upload: {str(e)}")
                            raise
                            
                        for file_type, local_path in paths_to_upload.items():
                            if local_path:
                                file_name = Path(local_path).name
                                s3_key = f"{output_key}/{file_name}"
                                task = asyncio.create_task(
                                    storage_manager.upload_file(output_bucket, s3_key, local_path, is_s3)
                                )
                                page_upload_tasks.append(task)
                        
                        logging.info(f"About to gather {len(page_upload_tasks)} upload tasks for page {page_no}")
                        try:
                            uploaded_paths_for_page = await asyncio.gather(*page_upload_tasks)
                            logging.info(f"Upload gather returned: {uploaded_paths_for_page} (type: {type(uploaded_paths_for_page)}, length: {len(uploaded_paths_for_page)})")
                        except Exception as e:
                            logging.error(f"Error in upload gather: {str(e)}")
                            raise

                        paths_to_upload['page_no'] = page_no
                        all_paths_to_upload.append(paths_to_upload)
                        page_response = {
                            "success": True,
                            "message": "parse success",
                            "page_no": page_no,
                            "uploaded_files": [path for path in uploaded_paths_for_page if path]
                        }
                        yield json.dumps(page_response) + "\n"
                except Exception as e:
                    logging.error(f"Error in parse_pdf_stream loop: {str(e)}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    raise


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
                                    storage_manager.download_file(
                                        bucket=output_bucket, 
                                        key=s3_key, 
                                        local_path=local_path, 
                                        is_s3=is_s3
                                    )
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
                try:
                    output_files['md'] = open(output_md_path, 'w', encoding='utf-8')
                    output_files['json'] = open(output_json_path, 'w', encoding='utf-8')
                    output_files['md_nohf'] = open(output_md_nohf_path, 'w', encoding='utf-8')
                    all_json_data = []
                    for p in all_paths_to_upload:
                        page_no = p.pop('page_no')
                        for file_type, local_path in p.items():
                            if file_type == 'json':
                                try:
                                    with open(local_path, 'r', encoding='utf-8') as input_file:
                                        data = json.load(input_file)
                                    data = {"page_no": page_no, **data}
                                    all_json_data.append(data)
                                except Exception as e:
                                    print(f"WARNING: Failed to read layout info file {local_path}: {str(e)}")
                                    all_json_data.append({"page_no": page_no})
                            else:
                                try:
                                    with open(local_path, 'r', encoding='utf-8') as input_file:
                                        file_content = input_file.read()
                                    output_files[file_type].write(file_content)
                                    output_files[file_type].write("\n\n")
                                except Exception as e:
                                    print(f"WARNING: Failed to read {file_type} file {local_path}: {str(e)}")
                    json.dump(all_json_data, output_files['json'], indent=4, ensure_ascii=False)
                finally:
                    # Ensure all file handles are properly closed
                    for file_handle in output_files.values():
                        if hasattr(file_handle, 'close'):
                            file_handle.close()

                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}.md", str(output_md_path), is_s3)
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}_nohf.md", str(output_md_nohf_path), is_s3)
                await storage_manager.upload_file(output_bucket, f"{output_key}/{output_file_name}.json", str(output_json_path), is_s3)

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






#---------------------------not streamming---------------------------

async def parse(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False,
    parse_type: str = "pdf",  # or "image", default is "pdf"
    rebuild_directory: bool = False,
    describe_picture: bool = False
):
    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")
    try:
        file_bucket, file_key = parse_s3_path(input_s3_path, is_s3)
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
                    await storage_manager.download_file(
                        bucket=file_bucket, key=file_key, local_path=str(input_file_path), is_s3 = is_s3
                    )
                    logging.info(f"download from s3/oss successfully: {input_s3_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to download file from s3/oss: {str(e)}") from e
                
                # compute MD5 hash of the input file
                try:
                    file_md5 = compute_md5(str(input_file_path))
                    logging.info(f"MD5 hash of input file {input_s3_path}: {file_md5}")
                except Exception as e:
                    logging.error(f"Failed to compute MD5 hash for {input_s3_path}: {str(e)}")
                    raise RuntimeError(f"Failed to compute MD5 hash: {str(e)}") from e

                output_bucket, output_key = parse_s3_path(output_s3_path, is_s3)
                output_file_name = output_s3_path.rstrip("/").split("/")[-1]
                output_file_path = OUTPUT_DIR / output_bucket / output_key
                output_md_path = output_file_path / output_file_name
                output_json_path = output_md_path.with_suffix(".json")
                output_md_nohf_path = output_md_path.with_name(output_md_path.stem + "_nohf").with_suffix(".md")
                output_md_path = output_md_path.with_suffix(".md")
                output_md5_path = output_md_path.with_suffix(".md5")
                output_md_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = await storage_manager.check_existing_results_sync(
                    bucket=output_bucket, prefix=f"{output_key}/{output_file_name}", is_s3=is_s3
                )
                
                # If so, download md5 file and compare hashes
                if md5_exists:
                    try:
                        await storage_manager.download_file(
                            bucket=output_bucket,
                            key=f"{output_key}/{output_file_name}.md5",
                            local_path=str(output_md5_path),
                            is_s3=is_s3
                        )
                        with open(output_md5_path, 'r') as f:
                            existing_md5 = f.read().strip()
                        if existing_md5 == file_md5:
                            if all_files_exist:
                                logging.info(f"Output files already exist in S3 and MD5 matches for {input_s3_path}. Skipping processing.")
                                return {
                                    "success": True,
                                    "total_pages": 0,
                                    "output_s3_path": output_s3_path,
                                    "message": "Output files already exist and MD5 matches. Skipped processing."
                                }
                            logging.info(f"MD5 matches for {input_s3_path}, but some output files are missing. Reprocessing the file.")
                        else:
                            # clean the whole output directory in S3
                            print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                            logging.info(f"MD5 mismatch for {input_s3_path}. Reprocessing the file.")
                            await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)
                    except Exception as e:
                        logging.warning(f"Failed to verify existing MD5 hash for {input_s3_path}: {str(e)}. Reprocessing the file.")
                else:
                    # clean the whole output directory in S3 for safety
                    print(f"Cleaning output directory in S3: {output_bucket}/{output_key}/")
                    logging.info(f"No MD5 hash found for {input_s3_path}. Cleaning output directory.")
                    await storage_manager.delete_files_in_directory(output_bucket, f"{output_key}/", is_s3)

                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                with open(output_md5_path, 'w') as f:
                    f.write(file_md5)
                logging.info(f"Saved MD5 hash to {output_md5_path}")
                
                # Upload MD5 hash file to S3/OSS
                try:
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md5", str(output_md5_path), is_s3
                    )
                except Exception as e:
                    logging.warning(f"Failed to upload MD5 hash file to s3/oss: {str(e)}")

                # print(output_file_path)
                # print(output_file_name)
                # print(output_md_path)
                # try:
                if parse_type == "image":
                    results = await dots_parser.parse_image(
                        input_path=str(input_file_path),
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
                        save_dir=output_file_path,
                        rebuild_directory=rebuild_directory,
                        describe_picture=describe_picture,
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
                    full_layout_info = {"page_no": result.get('page_no', -1), **full_layout_info}
                    formatted_results.append(full_layout_info)

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
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.md", str(output_md_path), is_s3
                    )
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}_nohf.md", str(output_md_nohf_path), is_s3
                    )
                    await storage_manager.upload_file(
                        output_bucket, f"{output_key}/{output_file_name}.json", str(output_json_path), is_s3
                    )
                    logging.info(f"upload from s3/oss successfully: {output_file_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to upload files to s3/oss: {str(e)}") from e

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

@app.post("/parse/image_old")
async def parse_image_old(
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

@app.post("/parse/pdf_old")
async def parse_pdf_old(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = Form(False),
    rebuild_directory: bool = Form(False),
    describe_picture: bool = Form(False)
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except TypeError:
        raise HTTPException(status_code=400, detail="Invalid filename format")
    if file_ext not in ['.pdf']:
        raise HTTPException(
            status_code=400, detail="Invalid image format. Supported: .pdf")

    return await parse(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, parse_type="pdf", rebuild_directory=rebuild_directory, describe_picture=describe_picture)


@app.post("/parse/file_old")
async def parse_file_old(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = False,
    rebuild_directory: bool = False,
    describe_picture: bool = False
):
    try:
        try:
            file_ext = Path(input_s3_path).suffix.lower()
        except TypeError:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        if file_ext == '.pdf':
            return await parse_pdf_old(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess, rebuild_directory, describe_picture)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return await parse_image_old(input_s3_path, output_s3_path, prompt_mode, fitz_preprocess)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# #---------------------------directly send file to parser---------------------------

# @app.post("/directly_parse/image")
# async def directly_parse_image(
#     file: UploadFile = File(...),
#     prompt_mode: str = "prompt_layout_all_en",
#     fitz_preprocess: bool = False
# ):
#     """Parse a single image file"""
#     try:
#         # Validate upload file
#         if not file:
#             raise HTTPException(status_code=400, detail="No file uploaded")
#         if not file.filename:
#             raise HTTPException(status_code=400, detail="Missing filename")

#         try:
#             file_ext = Path(file.filename).suffix.lower()
#         except TypeError:
#             raise HTTPException(status_code=400, detail="Invalid filename format")

#         if file_ext not in ['.jpg', '.jpeg', '.png']:
#             raise HTTPException(
#                 status_code=400, detail="Invalid image format. Supported: .jpg, .jpeg, .png")

#         # Verify file content
#         file_content = await file.read()
#         if not file_content:
#             raise HTTPException(status_code=400, detail="Uploaded file is empty")
#         await file.seek(0)  # Reset file pointer after reading

#         # Create temp file with debug logging
#         print(f"DEBUG: Creating temp file for {file.filename}")
#         temp_dir = tempfile.mkdtemp()
#         print(f"DEBUG: Created temp dir {temp_dir}")

#         temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}{file_ext}")
#         print(f"DEBUG: Temp file path will be {temp_path}")

#         # Save uploaded file with validation
#         file_content = await file.read()
#         print(f"DEBUG: Read {len(file_content)} bytes from upload")

#         if not isinstance(temp_path, (str, bytes, os.PathLike)):
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Invalid temp path type: {type(temp_path)}"
#             )

#         with open(temp_path, "wb") as buffer:
#             buffer.write(file_content)
#         print(f"DEBUG: Saved {len(file_content)} bytes to {temp_path}")

#         # Verify file was written
#         if not os.path.exists(temp_path):
#             raise HTTPException(
#                 status_code=500,
#                 detail="Failed to create temp file"
#             )
#         print(f"DEBUG: Temp file exists at {temp_path}")

#         # Process the image with debug logging
#         print(f"DEBUG: Calling parser with: {temp_path}")

#         # Get absolute path and verify file exists
#         abs_temp_path = os.path.abspath(temp_path)
#         if not os.path.exists(abs_temp_path):
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Temp file not found at {abs_temp_path}"
#             )

#         # Create and clean output directory
#         output_dir = tempfile.mkdtemp()
#         for f in os.listdir(output_dir):
#             os.remove(os.path.join(output_dir, f))

#         try:
#             results = dots_parser.parse_image(
#                 input_path=abs_temp_path,
#                 filename="api_image",
#                 prompt_mode=prompt_mode,
#                 save_dir=output_dir,
#                 fitz_preprocess=fitz_preprocess
#             )
#             print(f"DEBUG: Parser completed successfully=={results}")

#             # Extract and return the relevant data
#             result = results[0]  # Single result for image
#             layout_info_path = result.get('layout_info_path')
#             full_layout_info = {}
#             if layout_info_path and os.path.exists(layout_info_path):
#                 try:
#                     with open(layout_info_path, 'r', encoding='utf-8') as f:
#                         full_layout_info = json.load(f)
#                 except Exception as e:
#                     print(f"WARNING: Failed to read layout info file: {str(e)}")

#         except Exception as e:
#             print(f"DEBUG: Parser error: {str(e)}")
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Parser error: {str(e)}"
#             )
#         finally:
#             # Ensure cleanup even if parser fails
#             if os.path.exists(abs_temp_path):
#                 os.remove(abs_temp_path)
#             if os.path.exists(temp_dir):
#                 os.rmdir(temp_dir)
#             if os.path.exists(output_dir):
#                 for f in os.listdir(output_dir):
#                     os.remove(os.path.join(output_dir, f))
#                 os.rmdir(output_dir)

#         return {
#             "success": True,
#             "total_pages": len(results),
#             "results": [{"page_no": 0,
#                          "full_layout_info": full_layout_info}]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/directly_parse/pdf")
# async def directly_parse_pdf(
#     file: UploadFile = File(...),
#     prompt_mode: str = "prompt_layout_all_en",
#     fitz_preprocess: bool = False
# ):
#     """Parse a PDF file (multi-page)"""
#     try:
#         # Validate upload file
#         if not file:
#             raise HTTPException(status_code=400, detail="No file uploaded")
#         if not file.filename:
#             raise HTTPException(status_code=400, detail="Missing filename")

#         try:
#             if Path(file.filename).suffix.lower() != '.pdf':
#                 raise HTTPException(
#                     status_code=400, detail="Invalid PDF format. Only .pdf files accepted")
#         except TypeError:
#             raise HTTPException(status_code=400, detail="Invalid filename format")

#         # Verify file content
#         file_content = await file.read()
#         if not file_content:
#             raise HTTPException(status_code=400, detail="Uploaded file is empty")
#         await file.seek(0)  # Reset file pointer after reading

#         # Create temp file
#         temp_dir = tempfile.mkdtemp()
#         temp_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}.pdf")

#         # Save uploaded file
#         with open(temp_path, "wb") as buffer:
#             buffer.write(await file.read())

#         # Create and clean output directory
#         output_dir = tempfile.mkdtemp()
#         for f in os.listdir(output_dir):
#             os.remove(os.path.join(output_dir, f))

#         try:
#             # Process the PDF
#             results = dots_parser.parse_pdf(
#                 input_path=temp_path,
#                 filename="api_pdf",
#                 prompt_mode=prompt_mode,
#                 save_dir=output_dir
#             )
#             print(f"DEBUG: Parser completed successfully=={results}")
#             # Format results for all pages
#             formatted_results = []
#             for result in results:
#                 layout_info_path = result.get('layout_info_path')
#                 full_layout_info = {}
#                 if layout_info_path and os.path.exists(layout_info_path):
#                     try:
#                         with open(layout_info_path, 'r', encoding='utf-8') as f:
#                             full_layout_info = json.load(f)
#                     except Exception as e:
#                         print(f"WARNING: Failed to read layout info file: {str(e)}")

#                 formatted_results.append({
#                     "page_no": result.get('page_no'),
#                     "full_layout_info": full_layout_info
#                 })

#             return {
#                 "success": True,
#                 "total_pages": len(results),
#                 "results": formatted_results
#             }

#         finally:
#             # Ensure cleanup even if parser fails
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)
#             if os.path.exists(temp_dir):
#                 os.rmdir(temp_dir)
#             if os.path.exists(output_dir):
#                 for f in os.listdir(output_dir):
#                     os.remove(os.path.join(output_dir, f))
#                 os.rmdir(output_dir)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/directly_parse/file")
# async def directly_parse_file(
#     file: UploadFile = File(...),
#     prompt_mode: str = "prompt_layout_all_en",
#     fitz_preprocess: bool = False
# ):
#     """Automatically detect file type and parse accordingly"""
#     try:
#         # Validate upload file
#         if not file:
#             raise HTTPException(status_code=400, detail="No file uploaded")
#         if not file.filename:
#             raise HTTPException(status_code=400, detail="Missing filename")

#         try:
#             file_ext = Path(file.filename).suffix.lower()
#         except TypeError:
#             raise HTTPException(status_code=400, detail="Invalid filename format")

#         # Verify file content
#         file_content = await file.read()
#         if not file_content:
#             raise HTTPException(status_code=400, detail="Uploaded file is empty")
#         await file.seek(0)  # Reset file pointer after reading

#         if file_ext == '.pdf':
#             return await directly_parse_pdf(file, prompt_mode, fitz_preprocess)
#         elif file_ext in ['.jpg', '.jpeg', '.png']:
#             return await directly_parse_image(file, prompt_mode, fitz_preprocess)
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file format")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

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
