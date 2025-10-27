import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Awaitable, Callable, Dict, List, Literal, Optional

from loguru import logger
from opentelemetry import trace
from pydantic import BaseModel

from app.utils.configs import INPUT_DIR, OUTPUT_DIR
from app.utils.pg_vector import JobStatusType, OCRTable, is_job_terminated
from app.utils.storage import parse_s3_path


class JobLocalFiles(BaseModel):
    remote_input_bucket: str
    remote_input_file_key: str

    remote_output_bucket: str
    remote_output_file_key: str

    output_file_name: str

    @property
    def input_file_path(self):
        return INPUT_DIR / self.remote_input_bucket / self.remote_input_file_key

    @property
    def output_dir_path(self):
        return OUTPUT_DIR / self.remote_output_bucket / self.remote_output_file_key

    @property
    def output_json_path(self):
        output_file_path = self.output_dir_path / self.output_file_name
        return output_file_path.with_suffix(".json")

    @property
    def output_md_path(self):
        output_file_path = self.output_dir_path / self.output_file_name
        return output_file_path.with_suffix(".md")

    @property
    def output_md_nohf_path(self):
        return self.output_dir_path / f"{self.output_file_name}_nohf.md"

    @property
    def output_md5_path(self):
        output_file_path = self.output_dir_path / self.output_file_name
        return output_file_path.with_suffix(".md5")


class JobTaskStats(BaseModel):
    total_task_count: int = 0
    finished_task_count: int = 0
    failed_task_count: int = 0
    fallback_task_count: int = 0


class JobResponseModel(BaseModel):
    job_id: str
    created_by: str = "system"
    updated_by: str = "system"
    created_at: datetime = None
    updated_at: datetime = None
    knowledgebase_id: str
    workspace_id: str
    status: JobStatusType
    message: str
    parse_type: Literal["pdf", "image"] = "pdf"

    is_s3: bool = True
    input_s3_path: str
    output_s3_path: str

    prompt_mode: str = "prompt_layout_all_en"
    fitz_preprocess: bool = False
    rebuild_directory: bool = False
    describe_picture: bool = False
    overwrite: bool = False

    # model_name: usage_json from openai.types.CompletionUsage
    token_usage: dict[str, dict[str, int]] = {}

    task_stats: JobTaskStats = JobTaskStats()

    _job_local_files: JobLocalFiles = None

    def get_job_local_files(self):
        if self._job_local_files is None:
            input_bucket, input_file_key = parse_s3_path(self.input_s3_path, self.is_s3)
            output_bucket, output_file_key = parse_s3_path(
                self.output_s3_path, self.is_s3
            )
            output_file_name = self.output_s3_path.rstrip("/").rsplit("/", 1)[-1]
            self._job_local_files = JobLocalFiles(
                remote_input_bucket=input_bucket,
                remote_input_file_key=input_file_key,
                remote_output_bucket=output_bucket,
                remote_output_file_key=output_file_key,
                output_file_name=output_file_name,
            )
        return self._job_local_files

    @property
    def output_file_name(self):
        return self.get_job_local_files().output_file_name

    @property
    def json_url(self):
        return f"{self.output_s3_path}/{self.output_file_name}.json"

    @property
    def md_url(self):
        return f"{self.output_s3_path}/{self.output_file_name}.md"

    @property
    def md_nohf_url(self):
        return f"{self.output_s3_path}/{self.output_file_name}_nohf.md"

    @property
    def page_prefix(self):
        return f"{self.output_s3_path}/{self.output_file_name}_page_"

    def __str__(self):
        return self.model_dump_json()

    def transform_to_map(self):
        mapping = {
            "url": self.output_s3_path,
            "knowledgebaseId": self.knowledgebase_id,
            "workspaceId": self.workspace_id,
            "markdownUrl": self.md_url,
            "jsonUrl": self.json_url,
            "status": self.status,
        }
        return {k: (v if v is not None else "") for k, v in mapping.items()}

    def get_table_record(self) -> OCRTable:
        return OCRTable(
            id=self.job_id,
            url=self.input_s3_path,
            markdownUrl=self.md_url,
            jsonUrl=self.json_url,
            status=self.status,
            createdBy=self.created_by,
            updatedBy=self.updated_by,
            createdAt=self.created_at.replace(tzinfo=None),
            updatedAt=self.updated_at.replace(tzinfo=None),
            tokenUsage=self.token_usage,
        )


class Job:
    """
    Not thread-safe. Separate job states from job execution.
    - Job states are stored in JobResponseModel and requires persistence.
    - Job execution logic is implemented in this class.
    """

    def __init__(
        self,
        span: trace.Span,
        job_response: JobResponseModel,
        execute: Callable[[JobResponseModel], Awaitable[None]],
        on_status_change: Callable[[JobResponseModel], Awaitable[None]],
    ):
        self._span = span
        self.job_response = job_response
        self._on_status_change = on_status_change
        self._execute = execute
        self._cancel_requested = False

    async def process(self):
        # TODO(tatiana): handle the cancellation logic here. Now just do a trivial cancellation.
        if self._cancel_requested:
            logger.info(f"Job {self.job_response.job_id} is cancelled.")
            await self._set_cancelled()
            return

        with trace.use_span(self._span, end_on_exit=False):
            try:
                logger.info(f"Job {self.job_response.job_id} starts execution now.")
                await self._set_processing()
                await self._execute(self.job_response)
                logger.success(
                    f"Job {self.job_response.job_id} successfully processed."
                )
                await self._set_finished()
            except Exception as e:
                logger.error(
                    f"Job {self.job_response.job_id} failed. Final error: {e}",
                    exc_info=True,
                )
                await self._set_failed(e)

    def cancel(self):
        # TODO(tatiana): handle the cancellation logic here. Now just do a trivial cancellation.
        self._cancel_requested = True

    async def restore(self):
        # TODO(tatiana): support failure recovery and resume processing
        #                in the middle of job execution.
        raise NotImplementedError(
            "Failure recovery is not supported yet. "
            f"Job {self.job_response.job_id} cannot be restored."
        )

    async def _set_processing(self):
        self.job_response.status = "processing"
        self.job_response.message = "Processing job"
        await self._on_status_change(self.job_response)

    async def _set_failed(self, error):
        self.job_response.status = "failed"
        self.job_response.message = (
            f"Job failed after multiple retries. Final error: {str(error)}"
        )
        await self._on_status_change(self.job_response)
        self._span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        self._span.end()

    async def _set_finished(self):
        self.job_response.status = "completed"
        self.job_response.message = "Job completed successfully"
        await self._on_status_change(self.job_response)
        self._span.set_status(trace.Status(trace.StatusCode.OK))
        self._span.end()

    async def _set_cancelled(self):
        self.job_response.status = "canceled"
        self.job_response.message = "Job is cancelled"
        await self._on_status_change(self.job_response)
        self._span.set_status(trace.Status(trace.StatusCode.ERROR, "Job is cancelled"))
        self._span.end()


# TODO(tatiana): Make jobs run in parallel?
# TODO(tatiana): failure recovery. recover job from persistent storage
class JobExecutorPool(BaseModel):
    max_queue_size: int = 4
    concurrent_job_limit: int = 4
    terminated_job_retention_seconds: int = 24 * 60 * 60  # 24 hours
    job_clean_interval_seconds: int = 10

    _workers: List[asyncio.Task] = []
    _job_queue: asyncio.Queue = None
    _job_dict: Dict[str, Job] = {}

    def start(self):
        logger.info(f"Starting up {self.concurrent_job_limit} workers...")
        self._job_queue = asyncio.Queue(self.max_queue_size)

        for i in range(self.concurrent_job_limit):
            task = asyncio.create_task(self._worker_loop(f"Worker-{i}"))
            self._workers.append(task)
        self._workers.append(asyncio.create_task(self._clean_old_terminated_jobs()))

    def stop(self):
        logger.info("Shutting down and canceling worker tasks...")
        for worker in self._workers:
            worker.cancel()
        asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("All worker tasks have been canceled.")

    def is_job_waiting(self, job_id: str) -> bool:
        return job_id in self._job_dict

    async def add_job(self, job: Job):
        """Add a job for async execution."""
        self._job_dict[job.job_response.job_id] = job
        await self._job_queue.put(job.job_response.job_id)

    async def cancel_job(self, job_id: str):
        """Cancel a job.
        If the job is already terminated or does not exist, there will be no effect.
        If the job is currently running, it will be asynchronously cancelled.
        """
        job = self._job_dict.get(job_id)
        if job:
            job.cancel()

    def get_job_status(self, job_id: str) -> Optional[JobStatusType]:
        job = self._job_dict.get(job_id, None)
        if job is None:
            return None
        return job.job_response.status

    def get_job_response(self, job_id: str) -> Optional[JobResponseModel]:
        job = self._job_dict.get(job_id, None)
        if job is None:
            return None
        return job.job_response

    async def _clean_old_terminated_jobs(self):
        logger.debug(f"Starting job clean task")
        while True:
            try:
                for job_id, job in self._job_dict.items():
                    # IF job is terminated
                    if is_job_terminated(job.job_response.status):
                        # And it is terminated for a longer period than the retention period
                        diff = (datetime.now(UTC) - job.job_response.updated_at).seconds
                        if diff > self.terminated_job_retention_seconds:
                            # Remove the job from history
                            logger.debug(
                                f"Remove job {job_id} from history, "
                                f"job status: {job.job_response.status}, "
                                f"updated at: {job.job_response.updated_at}"
                            )
                            self._job_dict.pop(job_id)
                await asyncio.sleep(self.job_clean_interval_seconds)
            except asyncio.CancelledError:
                logger.info("Job clean task is shutting down.")
                break

    async def _worker_loop(self, worker_id: str):
        logger.debug(f"{worker_id} started")
        while True:
            try:
                job_id = await self._job_queue.get()
                job = self._job_dict.get(job_id)

                # This is unlikely since add_job should always save job_response
                # to self.job_response_dict
                if not job:
                    logger.error(
                        f"Worker {worker_id}: Job ID '{job_id}' found in queue "
                        "but not in JobResponseDict. Discarding task."
                    )
                    self._job_queue.task_done()
                    continue

                await job.process()
                self._job_queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"{worker_id} is shutting down.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in {worker_id}: {e}")
