"""
Environment variables:
- Required:
  - POSTGRES_URL_NO_SSL_DEV: for establishing connection to the PostgreSQL database
  - API_KEY: the API key used in OpenAI API to access LLM services
- Optional:
  - OSS_ENDPOINT: the endpoint for accessing the OSS storage
  - OSS_ACCESS_KEY_ID: the access key for accessing the OSS storage
  - OSS_ACCESS_KEY_SECRET: the secret key for accessing the OSS storage

File resources:
- app/input: the directory for storing the input files. Created on startup if not exists.
- app/output: the directory for storing the output files. Created on startup if not exists.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from sys import stderr
import shutil

import httpx
import uvicorn
from fastapi import FastAPI, Form, HTTPException, Response
from fastapi.responses import JSONResponse
from loguru import logger
from openai.types import CompletionUsage

from app.utils.configs import INPUT_DIR, OUTPUT_DIR, Configs
from app.utils.executor import Job, JobExecutorPool, JobResponseModel, TaskExecutorPool
from app.utils.hash import compute_md5_file, compute_md5_string
from app.utils.pg_vector import OCRTable, PGVector
from app.utils.storage import StorageManager
from app.utils.tracing import get_tracer, setup_tracing, trace_span_async, traced
from dots_ocr.model.inference import InferenceTaskOptions
from dots_ocr.parser import DotsOCRParser
from dots_ocr.utils.consts import MAX_PIXELS, MIN_PIXELS
from dots_ocr.utils.page_parser import PageParser, ParseOptions


######################################## Resources ########################################

configs = Configs()
logger.info(f"Configs: {configs}")

global_lock_manager = asyncio.Lock()
pgvector_lock = asyncio.Lock()
# In production, an input path corresponds to a certain output path.
# The same input path will be mapped to different output paths only in testing codes.
processing_input_locks = {}
processing_output_locks = {}
storage_manager = StorageManager()
pg_vector_manager = PGVector()
job_executor_pool = JobExecutorPool(
    concurrent_job_limit=configs.NUM_WORKERS, max_queue_size=configs.JOB_QUEUE_MAX_SIZE
)
ocr_task_executor_pool = TaskExecutorPool(
    concurrent_task_limit=configs.CONCURRENT_OCR_INFERENCE_TASK_LIMIT,
    max_queue_size=configs.OCR_INFERENCE_TASK_QUEUE_MAX_SIZE,
    name="OcrInferenceTask",
)
describe_picture_task_executor_pool = TaskExecutorPool(
    concurrent_task_limit=configs.CONCURRENT_DESCRIBE_PICTURE_TASK_LIMIT,
    max_queue_size=configs.DESCRIBE_PICTURE_TASK_QUEUE_MAX_SIZE,
    name="PictureDescriptionTask",
)
page_parser = PageParser(
    ocr_inference_task_options=InferenceTaskOptions(
        model_name="dotsocr",
        model_host=configs.OCR_INFERENCE_HOST,
        model_port=configs.OCR_INFERENCE_PORT,
        temperature=0.1,
        top_p=1.0,
        max_completion_tokens=32768,
        timeout=configs.API_TIMEOUT,
    ),
    describe_picture_task_options=InferenceTaskOptions(
        model_name="InternVL3_5-2B",
        model_host=configs.INTERN_VL_HOST,
        model_port=configs.INTERN_VL_PORT,
        temperature=0.1,
        top_p=1.0,
        max_completion_tokens=8192,
        timeout=configs.API_TIMEOUT,
    ),
    parse_options=ParseOptions(
        dpi=configs.DPI,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        task_retry_count=configs.TASK_RETRY_COUNT,
    ),
    concurrency_limit=configs.CONCURRENT_OCR_TASK_LIMIT,
)
dots_parser = DotsOCRParser(
    ocr_task_executor_pool=ocr_task_executor_pool,
    describe_picture_task_executor_pool=describe_picture_task_executor_pool,
    page_parser=page_parser,
    storage_manager=storage_manager,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    logger.remove(0)
    logger.add(stderr, level=configs.LOG_LEVEL)

    await pg_vector_manager.ensure_table_exists()

    job_executor_pool.start()
    ocr_task_executor_pool.start()
    describe_picture_task_executor_pool.start()

    yield

    job_executor_pool.stop()
    ocr_task_executor_pool.stop()
    describe_picture_task_executor_pool.stop()
    await pg_vector_manager.flush()
    await pg_vector_manager.close()


app = FastAPI(
    title="dotsOCR API",
    description="API for PDF and image text recognition using dotsOCR by Grant",
    version="1.0.0",
    lifespan=lifespan,
)

setup_tracing(
    app,
    configs.DOTSOCR_OTEL_SERVICE_NAME,
    configs.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
    configs.OTEL_EXPORTER_OTLP_TRACES_TIMEOUT,
)


@traced()
async def update_pgvector(job: JobResponseModel):
    """Insert a new job or update an existing job in the PG database.

    Args:
        job (JobResponseModel): The job to be inserted or updated.
    """
    job.updated_at = datetime.now(UTC)
    # TODO(tatiana): Why do we need to use a lock here? Consider measuring the performance impact.
    async with pgvector_lock:
        # await pg_vector_manager.ensure_table_exists()
        record = job.get_table_record()
        await pg_vector_manager.upsert_record(record)

        # If there is a problem, it may not be reported immediately after the operation.
        # Use flush to force the operation to be committed, so that the error can be
        # reported immediately if exists.
        # await pg_vector_manager.flush()


@traced(record_return=True)
async def get_record_pgvector(job_id: str) -> OCRTable:
    async with pgvector_lock:
        # await pg_vector_manager.ensure_table_exists()
        record = await pg_vector_manager.get_record_by_id(job_id)
        return record


def sum_token_usage(
    sum_usage: dict[str, CompletionUsage], update: dict[str, CompletionUsage]
) -> dict[str, CompletionUsage]:
    for model_name, usage in update.items():
        if model_name not in sum_usage:
            sum_usage[model_name] = usage
        else:
            sum_usage[model_name].completion_tokens += usage.completion_tokens
            sum_usage[model_name].prompt_tokens += usage.prompt_tokens
            sum_usage[model_name].total_tokens += usage.total_tokens


@app.post("/status")
async def status_check(ocr_job_id: str = Form(alias="OCRJobId")):
    job_response = job_executor_pool.get_job_response(ocr_job_id)
    if job_response is None:
        record = await get_record_pgvector(ocr_job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"status": record.status}

    return {"status": job_response.status, "task_stats": job_response.task_stats}


@traced()
async def stream_and_upload_generator(job_response: JobResponseModel):
    input_s3_path = job_response.input_s3_path
    output_s3_path = job_response.output_s3_path
    is_s3 = job_response.is_s3

    try:
        async with global_lock_manager:
            if input_s3_path not in processing_input_locks:
                processing_input_locks[input_s3_path] = asyncio.Lock()
            if output_s3_path not in processing_output_locks:
                processing_output_locks[output_s3_path] = asyncio.Lock()
        input_lock = processing_input_locks[input_s3_path]
        output_lock = processing_output_locks[output_s3_path]

        async with input_lock:
            async with output_lock:
                # prepare local path
                job_files = job_response.get_job_local_files()
                job_files.input_file_path.parent.mkdir(parents=True, exist_ok=True)
                job_files.output_md_path.parent.mkdir(parents=True, exist_ok=True)
                job_files.output_dir_path.mkdir(parents=True, exist_ok=True)

                download_input_from_s3 = True
                md5_match = False

                logger.debug(
                    f"Checking existing remote files for job {job_response.job_id}"
                )
                # Check if 4 required files already exist in S3
                md5_exists, all_files_exist = (
                    await storage_manager.check_existing_results_sync(
                        bucket=job_files.remote_output_bucket,
                        prefix=f"{job_files.remote_output_file_key}/{job_response.output_file_name}",
                        is_s3=is_s3,
                    )
                )

                # Try to reuse existing input if overwrite is disabled
                if not job_response.overwrite and md5_exists:
                    logger.debug(
                        f"Downloading MD5 hash file for job {job_response.job_id}"
                    )
                    await storage_manager.download_file(
                        bucket=job_files.remote_output_bucket,
                        key=f"{job_files.remote_output_file_key}/{job_response.output_file_name}.md5",
                        local_path=str(job_files.output_md5_path),
                        is_s3=is_s3,
                    )
                    with open(job_files.output_md5_path, "r", encoding="utf-8") as f:
                        existing_md5 = f.read().strip()

                    # If input file already exists locally and overwrite is not set,
                    # skip downloading if md5 matches
                    if job_files.input_file_path.exists():
                        file_md5 = f"{job_response.job_id}:" + compute_md5_file(
                            str(job_files.input_file_path)
                        )
                        if file_md5 == existing_md5:
                            download_input_from_s3 = False
                            md5_match = True

                # download file from S3
                if download_input_from_s3:
                    try:
                        logger.debug(
                            f"Downloading input S3 file for job {job_response.job_id}"
                        )
                        await storage_manager.download_file(
                            bucket=job_files.remote_input_bucket,
                            key=job_files.remote_input_file_key,
                            local_path=str(job_files.input_file_path),
                            is_s3=is_s3,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to download file from s3/oss: {str(e)}"
                        ) from e

                    # compute MD5 hash of the input file
                    try:
                        file_md5 = (
                            job_response.job_id
                            + ":"
                            + compute_md5_file(str(job_files.input_file_path))
                        )
                        logger.info(
                            f"MD5 hash of input file {input_s3_path}: {file_md5}",
                        )
                        if not job_response.overwrite:
                            md5_match = md5_exists and file_md5 == existing_md5
                    except Exception as e:
                        logger.error(
                            f"Failed to compute MD5 hash for {input_s3_path}: {str(e)}",
                        )
                        raise RuntimeError(
                            f"Failed to compute MD5 hash: {str(e)}"
                        ) from e

                # If overwrite is disabled and md5 matches, try to reuse existing output
                if not job_response.overwrite and md5_match and all_files_exist:
                    logger.info(
                        f"Output files already exist in S3 and MD5 matches for {input_s3_path}. ",
                        "Skipping processing.",
                    )
                    if job_response.status != "completed":
                        job_response.status = "completed"
                        job_response.message = "Output files already exist and MD5 matches. Recover pg vector record."
                        job_response.token_usage = {
                            "dotsocr": {
                                "completion_tokens": 0,
                                "prompt_tokens": 0,
                                "total_tokens": 0,
                            },
                            "InternVL3_5-2B": {
                                "completion_tokens": 0,
                                "prompt_tokens": 0,
                                "total_tokens": 0,
                            },
                        }
                        logger.warning(
                            f"Job {job_response.job_id} not found in pgvector but output files exist and MD5 matches."
                            "Updating to completed but token usage can't be recovered and is set to zero."
                        )
                        await update_pgvector(job_response)
                    skip_response = {
                        "success": True,
                        "total_pages": 0,
                        "output_s3_path": output_s3_path,
                        "message": "Output files already exist and MD5 matches."
                        " Skipped processing.",
                    }
                    return skip_response

                if md5_exists or all_files_exist:
                    logger.info(
                        f"Reprocessing {input_s3_path}. "
                        f"md5 exists {md5_exists}, all files exist {all_files_exist}",
                        f" overwrite {job_response.overwrite}, md5 match {md5_match}.",
                    )
                # clean the whole output directory in S3 for safety
                # logger.info(
                #     f"No MD5 hash found for {input_s3_path}. "
                #     f"Cleaning output directory {job_local_files.remote_output_bucket}/{output_key}/."
                # )
                # await storage_manager.delete_files_in_directory(
                #     job_local_files.remote_output_bucket, f"{output_key}/", is_s3
                # )

                # Mismatch or no existing MD5 hash found, save new MD5 hash to a file
                # TODO(tatiana): use put_object API to directly write the data to S3/OSS
                # without extra IO to disk.
                if not md5_match:
                    with open(job_files.output_md5_path, "w", encoding="utf-8") as f:
                        f.write(file_md5)
                    logger.info("Saved MD5 hash to {job_files.output_md5_path}")

                    # Upload MD5 hash file to S3/OSS
                    try:
                        await storage_manager.upload_file(
                            job_files.remote_output_bucket,
                            f"{job_files.remote_output_file_key}/{job_files.output_file_name}.md5",
                            str(job_files.output_md5_path),
                            is_s3,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to upload MD5 hash file to s3/oss: {str(e)}",
                        )

                # parse the PDF file and upload each page's output files
                all_paths_to_upload = []

                try:
                    if job_response.parse_type == "image":
                        result, total_token_usage = await dots_parser.parse_image(
                            job_response
                        )
                        all_paths_to_upload.append(result)
                    else:
                        total_token_usage = {}
                        async with trace_span_async("schedule_pdf_tasks") as span:
                            async for (
                                result,
                                status,
                                token_usage,
                            ) in dots_parser.schedule_pdf_tasks(job_response):
                                sum_token_usage(total_token_usage, token_usage)
                                if status in ["fallback", "timeout", "failed"]:
                                    # TODO(tatiana): save failed/fallback task to OCRTable and
                                    # allow partial rerun after fix
                                    if status == "failed" or status == "timeout":
                                        job_response.task_stats.failed_task_count += 1
                                        continue
                                    job_response.task_stats.fallback_task_count += 1

                                all_paths_to_upload.append(result)
                except Exception as e:
                    logger.exception(f"Error during parsing pages: {e}")
                    raise

                # TODO(tatiana): if job is rerun (e.g., due to overwrite), we need to accumulate
                # this measure or differentiate job runs
                job_response.token_usage = {
                    model_name: usage.model_dump(
                        include={"completion_tokens", "prompt_tokens", "total_tokens"}
                    )
                    for model_name, usage in total_token_usage.items()
                }

                if job_response.parse_type == "pdf":
                    if (
                        job_response.task_stats.failed_task_count
                        / job_response.task_stats.total_task_count
                        > configs.TASK_FAIL_THRESHOLD
                    ):
                        final_response = {
                            "success": False,
                            "total_pages": job_response.task_stats.total_task_count,
                            "output_s3_path": output_s3_path,
                            "detail": f"Failed to parse {job_response.task_stats.failed_task_count} pages "
                            f"out of {job_response.task_stats.total_task_count} pages.",
                        }
                        return final_response
                    # combine all page to upload
                    all_paths_to_upload.sort(key=lambda item: item["page_no"])
                    output_files = {}
                    try:
                        output_files["md"] = open(
                            job_files.output_md_path, "w", encoding="utf-8"
                        )
                        output_files["md_nohf"] = open(
                            job_files.output_md_nohf_path, "w", encoding="utf-8"
                        )
                        all_json_data = []
                        for p in all_paths_to_upload:
                            page_no = p.pop("page_no")
                            for file_type, local_path in p.items():
                                if file_type == "json":
                                    try:
                                        with open(
                                            local_path, "r", encoding="utf-8"
                                        ) as input_file:
                                            data = json.load(input_file)[0]
                                        data = {"page_no": page_no, **data}
                                        all_json_data.append(data)
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to read layout info file {local_path}: {str(e)}"
                                        )
                                        all_json_data.append({"page_no": page_no})
                                else:
                                    try:
                                        with open(
                                            local_path, "r", encoding="utf-8"
                                        ) as input_file:
                                            file_content = input_file.read()
                                        output_files[file_type].write(file_content)
                                        output_files[file_type].write("\n\n")
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to read {file_type} file {local_path}: {str(e)}"
                                        )
                        with open(
                            job_files.output_json_path, "w", encoding="utf-8"
                        ) as json_output:
                            json.dump(
                                all_json_data,
                                json_output,
                                indent=4,
                                ensure_ascii=False,
                            )
                    finally:
                        # Ensure all file handles are properly closed
                        for file_handle in output_files.values():
                            if hasattr(file_handle, "close"):
                                file_handle.close()

                await storage_manager.upload_file(
                    job_files.remote_output_bucket,
                    f"{job_files.remote_output_file_key}/{job_files.output_file_name}.md",
                    str(job_files.output_md_path),
                    is_s3,
                )
                await storage_manager.upload_file(
                    job_files.remote_output_bucket,
                    f"{job_files.remote_output_file_key}/{job_files.output_file_name}_nohf.md",
                    str(job_files.output_md_nohf_path),
                    is_s3,
                )
                await storage_manager.upload_file(
                    job_files.remote_output_bucket,
                    f"{job_files.remote_output_file_key}/{job_files.output_file_name}.json",
                    str(job_files.output_json_path),
                    is_s3,
                )

                final_response = {
                    "success": True,
                    "total_pages": len(all_paths_to_upload),
                    "output_s3_path": output_s3_path,
                }
                return final_response

    except Exception as e:
        return {"success": False, "detail": str(e)}


    finally:
        if configs.CLEANUP_LOCAL:
            input_file_path = job_files.input_file_path
            output_dir_path = job_files.output_dir_path
            if input_file_path and input_file_path.exists():
                logger.info(f"Cleaning up local input file for job {job_response.job_id}: {input_file_path}")
                try:
                    input_file_path.unlink()
                except Exception as e:
                    logger.error(
                        f"An error occurred during file cleanup for job {job_response.job_id}: {e}",
                        exc_info=True
                    )

            if output_dir_path and output_dir_path.exists():
                logger.info(f"Cleaning up local root directory for job {job_response.job_id}: {output_dir_path}")
                try:
                    shutil.rmtree(output_dir_path)
                except Exception as e:
                    logger.error(
                        f"An error occurred during directory cleanup for job {job_response.job_id}: {e}",
                        exc_info=True
                    )


@traced()
async def run_job(job: JobResponseModel):
    result = await stream_and_upload_generator(job)
    if not result.get("success", False):
        logger.error(f"Job {job.job_id} failed with: {result.get('detail', 'error')}")
        raise RuntimeError(
            f"Job {job.job_id} failed with: {result.get('detail', 'error')}"
        )

    logger.success(f"Job {job.job_id} successfully processed.")


@app.post("/parse/file")
async def parse_file(
    input_s3_path: str = Form(...),
    output_s3_path: str = Form(...),
    knowledgebase_id: str = Form(alias="knowledgebaseId"),
    workspace_id: str = Form(alias="workspaceId"),
    prompt_mode: str = "prompt_layout_all_en",
    fitz_preprocess: bool = Form(False),
    rebuild_directory: bool = Form(False),
    describe_picture: bool = Form(True),
    overwrite: bool = Form(False),
):
    try:
        file_ext = Path(input_s3_path).suffix.lower()
    except (TypeError, AttributeError) as err:
        raise HTTPException(
            status_code=400, detail="Invalid input_s3_path format"
        ) from err

    supported_formats = [".pdf", ".jpg", ".jpeg", ".png"]
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. "
            f"Supported formats are: {', '.join(supported_formats)}",
        )

    # Current logic: only if input, output, knowledgebaseId, workspaceId are all the same,
    # we consider it as the same job, and add job_id to the md5 file
    ocr_job_id = "job-" + compute_md5_string(
        f"{input_s3_path}_{output_s3_path}_{knowledgebase_id}_{workspace_id}"
    )

    is_s3 = False
    if input_s3_path.startswith("s3://") and output_s3_path.startswith("s3://"):
        is_s3 = True
    elif input_s3_path.startswith("oss://") and output_s3_path.startswith("oss://"):
        is_s3 = False
    else:
        raise RuntimeError("Input and output paths must both be s3:// or oss://")

    # Get the existing job status from pgvector
    existing_record = await get_record_pgvector(ocr_job_id)
    if existing_record and job_executor_pool.is_job_waiting(ocr_job_id):
        if existing_record.status in ["pending", "processing"]:
            return JSONResponse(
                {
                    "OCRJobId": ocr_job_id,
                    "status": existing_record.status,
                    "message": "Job is already in progress",
                },
                status_code=202,
            )
        if existing_record.status in ["completed", "failed", "canceled"]:
            # allow re-process but check md5 first in the worker
            pass

    job_response = JobResponseModel(
        job_id=ocr_job_id,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        status="pending",
        knowledgebase_id=knowledgebase_id,
        workspace_id=workspace_id,
        message="Job is pending",
        is_s3=is_s3,
        input_s3_path=input_s3_path,
        output_s3_path=output_s3_path,
        parse_type="pdf" if file_ext == ".pdf" else "image",
        prompt_mode=prompt_mode,
        fitz_preprocess=fitz_preprocess,
        rebuild_directory=rebuild_directory,
        describe_picture=describe_picture,
        overwrite=overwrite,
    )

    logger.info(
        f"Job {ocr_job_id} created. {job_response}"
    )
    await update_pgvector(job_response)
    await job_executor_pool.add_job(
        Job(
            get_tracer().start_span("async_job"),
            job_response,
            execute=run_job,
            on_status_change=update_pgvector,
        )
    )
    return JSONResponse({"OCRJobId": ocr_job_id}, status_code=200)


@app.post("/parse/image_old")
async def parse_image_old(**kwargs):
    raise HTTPException(
        status_code=400, detail="Deprecated API, please use /parse/file instead"
    )


@app.post("/parse/pdf_old")
async def parse_pdf_old(**kwargs):
    raise HTTPException(
        status_code=400, detail="Deprecated API, please use /parse/file instead"
    )


@app.post("/parse/file_old")
async def parse_file_old(**kwargs):
    raise HTTPException(
        status_code=400, detail="Deprecated API, please use /parse/file instead"
    )


async def health_check():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(configs.OCR_HEALTH_CHECK_URL, timeout=5.0)

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
            media_type=response.headers.get("content-type"),
        )
    except httpx.ConnectError as e:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "status": 503,
                "detail": f"Health check failed: Unable to connect to DotsOCR service. Error: {e}",
            },
        )
    except httpx.TimeoutException as e:
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "status": 504,
                "detail": f"Health check failed: Request to DotsOCR service timed out. Error: {e}",
            },
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": 500,
                "detail": f"An unexpected error occurred during health check. Error: {e}",
            },
        )


@app.get("/health")
async def health():
    return await health_check()


@app.get("/token_usage/{ocr_job_id}")
async def token_usage(ocr_job_id: str):
    logger.info(f"get token usage for {ocr_job_id}")
    job_response = job_executor_pool.get_job_response(ocr_job_id)
    if job_response is None:
        record = await get_record_pgvector(ocr_job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return record.tokenUsage
    return job_response.token_usage


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6008, reload=True)
