import asyncio
from loguru import logger
import os
import re
from functools import partial

import boto3
from botocore.config import Config

from app.utils.tracing import trace_span_async, traced


def parse_s3_path(s3_path: str, is_s3: bool):
    if is_s3:
        s3_path = s3_path.replace("s3://", "")
    else:
        s3_path = s3_path.replace("oss://", "")
    bucket, *key_parts = s3_path.split("/")
    return bucket, "/".join(key_parts)


class StorageManager:

    def __init__(self):
        self.s3_client = boto3.client("s3")

        endpoint = os.getenv("OSS_ENDPOINT")
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("OSS_ACCESS_KEY_SECRET")
        if not all([endpoint, access_key_id, secret_access_key]):
            logger.warning(
                "OSS environment variables (OSS_ENDPOINT, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET) are not fully set. OSS client may not work."
            )

        self.s3_oss_client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint,
            config=Config(s3={"addressing_style": "virtual"}, signature_version="s3"),
        )

    @traced()
    async def upload_file(self, bucket, key, local_path, is_s3):
        if not local_path or not os.path.exists(local_path):
            return None
        try:
            if is_s3:
                self.s3_client.upload_file(Bucket=bucket, Key=key, Filename=local_path)
            else:
                self.s3_oss_client.upload_file(
                    Bucket=bucket, Key=key, Filename=local_path
                )
            s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
            logger.info(f"Successfully uploaded {local_path} to {s3_full_path}")
            # os.remove(local_path)
            return s3_full_path
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            return None

    @traced(record_return=True)
    async def download_file(self, bucket, key, local_path, is_s3):
        if not bucket or not key or not local_path:
            logger.warning("Bucket, key, and local_path must be specified.")
            return None
        try:
            if is_s3:
                self.s3_client.download_file(
                    Bucket=bucket, Key=key, Filename=local_path
                )
            else:
                self.s3_oss_client.download_file(
                    Bucket=bucket, Key=key, Filename=local_path
                )
            s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
            logger.info(f"Successfully downloaded {local_path} to {s3_full_path}")
            return s3_full_path
        except Exception as e:
            logger.error(
                f"Failed to download s3://{bucket}/{key} to {local_path}: {e}"
            )
            return None

    @traced()
    async def delete_file(self, bucket, key, is_s3):
        if not bucket or not key:
            logger.warning("Bucket and key must be specified.")
            return False
        try:
            if is_s3:
                self.s3_client.delete_object(Bucket=bucket, Key=key)
            else:
                self.s3_oss_client.delete_object(Bucket=bucket, Key=key)
            s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
            logger.info(f"Successfully deleted {s3_full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete s3://{bucket}/{key}: {e}")
            return False

    @traced()
    async def delete_files_in_directory(self, bucket, prefix, is_s3):
        if not bucket or not prefix:
            logger.warning("Bucket and prefix must be specified.")
            return False

        def delete_objects_sync(client, bucket, prefix):
            try:
                paginator = client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

                objects_to_delete = []
                for page in pages:
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            objects_to_delete.append({"Key": obj["Key"]})

                if objects_to_delete:
                    for i in range(0, len(objects_to_delete), 1000):
                        chunk = objects_to_delete[i : i + 1000]
                        client.delete_objects(Bucket=bucket, Delete={"Objects": chunk})
                    return True, len(objects_to_delete)
                else:
                    return True, 0
            except Exception as e:
                raise e

        try:
            client = self.s3_client if is_s3 else self.s3_oss_client
            loop = asyncio.get_event_loop()
            success, count = await loop.run_in_executor(
                None, partial(delete_objects_sync, client, bucket, prefix)
            )

            if count > 0:
                logger.info(
                    f"Successfully deleted {count} files under s3://{bucket}/{prefix}"
                )
            else:
                logger.info(f"No files found under s3://{bucket}/{prefix} to delete.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete files under s3://{bucket}/{prefix}: {e}")
            return False

    @traced()
    async def check_existing_results_sync(
        self, bucket: str, prefix: str, is_s3: bool
    ) -> tuple[bool, bool]:
        """
        Check if the 4 output files (.json, .md, _nohf.md, .md5) exist in the S3 bucket with the given prefix.
        """
        required_extensions = [".json", ".md", "_nohf.md", ".md5"]
        existing_files = 0
        md5_exists = False

        @traced(record_return=True)
        def check_file_exists(client, bucket, key, is_s3):
            try:
                if is_s3:
                    client.head_object(Bucket=bucket, Key=key)
                else:
                    client.head_object(bucket, key)
                return True
            except Exception:
                return False

        client = self.s3_client if is_s3 else self.s3_oss_client

        for ext in required_extensions:
            key = f"{prefix}{ext}"
            exists = check_file_exists(client, bucket, key, is_s3)
            if exists:
                existing_files += 1
                if ext == ".md5":
                    md5_exists = True

        return md5_exists, existing_files == len(required_extensions)

    def _get_existing_page_indices_s3_sync(self, bucket: str, prefix: str) -> set:
        print(bucket, prefix)
        existing_indices = set()

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        pattern = re.compile(
            rf"^{re.escape(prefix)}(\d+)({re.escape('.json')}|{re.escape('.md')}|{re.escape('_nohf.md')})$"
        )

        count = {}
        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    match = pattern.match(key)
                    if match:
                        num = int(match.group(1))
                        count[num] = count.get(num, 0) + 1
                        if count[num] == 3:
                            existing_indices.add(num)
        print(existing_indices)
        return existing_indices

    async def _get_existing_page_indices_s3(self, bucket: str, prefix: str) -> set:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._get_existing_page_indices_s3_sync, bucket, prefix
        )
