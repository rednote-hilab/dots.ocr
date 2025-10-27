from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Configs(BaseSettings):
    # Number of concurrent jobs that can run.
    NUM_WORKERS: int = 16

    # The max number of concurrent OCR inference requests that can be sent to the ocr model.
    # Increasing this may improve GPU utilization to some extent but at the cost of the model
    # server memory usage.
    CONCURRENT_OCR_INFERENCE_TASK_LIMIT: int = 16

    # The max number of concurrent picture description requests that can be sent to the internVL
    # model. Increasing this may improve GPU utilization to some extent but at the cost of the
    # model server memory usage.
    CONCURRENT_DESCRIBE_PICTURE_TASK_LIMIT: int = 16

    # The max number of concurrent OCR tasks that can be run. Increasing this may improve overall
    # resource overlapping, but at the cost of memory for buffering the extracted images from docs,
    # i.e, around CONCURRENT_OCR_TASK_LIMIT images (one for each page) can be buffered in memory.
    CONCURRENT_OCR_TASK_LIMIT: int = 64

    # DPI of images extracted from documents that are used for OCR.
    DPI: int = 200

    # The max number of jobs that can be queued. 0 means unlimited.
    JOB_QUEUE_MAX_SIZE: int = 0

    # The number of OCR inference tasks that can be queued. Increase this may improve resource
    # overlapping, but at the cost of memory for buffering the extracted images from docs,
    OCR_INFERENCE_TASK_QUEUE_MAX_SIZE: int = 2 * CONCURRENT_OCR_INFERENCE_TASK_LIMIT

    # The number of OCR inference tasks that can be queued. Increase this may improve resource
    # overlapping, but at the cost of memory for buffering the picture blocks identified from
    # the documents.
    DESCRIBE_PICTURE_TASK_QUEUE_MAX_SIZE: int = 24

    OCR_INFERENCE_HOST: str = "localhost"
    OCR_INFERENCE_PORT: int = 8000
    OCR_HEALTH_CHECK_URL: str = (
        f"http://{OCR_INFERENCE_HOST}:{OCR_INFERENCE_PORT}/health"
    )

    INTERN_VL_HOST: str = "internvl3-5"
    INTERN_VL_PORT: int = 6008

    # TODO(tatiana): need to check the timeout semantics in OpenAI API.
    # Exclude queuing time from the timeout.
    API_TIMEOUT: List[int] = [60, 120, 120]

    TASK_RETRY_COUNT: int = 3

    # If the number of failed tasks is greater than this threshold, the job will be considered failed.
    TASK_FAIL_THRESHOLD: float = 0.1

    LOG_LEVEL: str = "INFO"

    DOTSOCR_OTEL_SERVICE_NAME: str = "dots.ocr"
    # Endpoint URL for trace data only, with an optionally-specified port number.
    # If not provided, the tracing is disabled.
    # Example value:
    #   gRPC: "http://localhost:4317"
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Optional[str] = None
    # The timeout value for all outgoing traces in milliseconds. Default is 10 seconds.
    OTEL_EXPORTER_OTLP_TRACES_TIMEOUT: int = 10_000  # 10 seconds

    # Whether to delete local result files after each job is completed.
    CLEANUP_LOCAL: bool = True

BASE_DIR: Path = Path(__file__).resolve().parent.parent
INPUT_DIR: Path = BASE_DIR / "input"
OUTPUT_DIR: Path = BASE_DIR / "output"
