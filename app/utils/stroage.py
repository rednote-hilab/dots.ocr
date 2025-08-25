import boto3
from botocore.config import Config
import os
import logging
import re
import asyncio
from functools import partial
class StorageManager:

    def __init__(self):        
        self.s3_client = boto3.client('s3')

        endpoint = os.getenv('OSS_ENDPOINT')
        access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        secret_access_key = os.getenv('OSS_ACCESS_KEY_SECRET')
        if not all([endpoint, access_key_id, secret_access_key]):
            logging.warning("OSS environment variables (OSS_ENDPOINT, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET) are not fully set. OSS client may not work.")

        self.s3_oss_client = boto3.client(
            's3',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            endpoint_url=endpoint,
            config=Config(s3={"addressing_style": "virtual"},
                        signature_version='s3'))
                        
    async def upload_file(self, bucket, key, local_path, is_s3):
        if not local_path or not os.path.exists(local_path):
            return None
        try:
            if is_s3:
                self.s3_client.upload_file(Bucket=bucket, Key=key, Filename=local_path)
            else:
                self.s3_oss_client.upload_file(Bucket=bucket, Key=key, Filename=local_path)
            s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
            logging.info(f"Successfully uploaded {local_path} to {s3_full_path}")
            # os.remove(local_path)
            return s3_full_path
        except Exception as e:
            logging.error(f"Failed to upload {local_path} to s3://{bucket}/{key}: {e}")
            return None
            
    async def download_file(self, bucket, key, local_path, is_s3):
        if not bucket or not key or not local_path:
            logging.warning("Bucket, key, and local_path must be specified.")
            return None
        try:
            if is_s3:
                self.s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)
            else:
                self.s3_oss_client.download_file(Bucket=bucket, Key=key, Filename=local_path)
            s3_full_path = f"{'s3' if is_s3 else 'oss'}://{bucket}/{key}"
            logging.info(f"Successfully downloaded {local_path} to {s3_full_path}")
            return s3_full_path
        except Exception as e:
            logging.error(f"Failed to download s3://{bucket}/{key} to {local_path}: {e}")
            return None
    
    def _get_existing_page_indices_s3_sync(self, bucket: str, prefix: str) -> set:
        print(bucket, prefix)
        existing_indices = set()
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
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

    async def _get_existing_page_indices_s3(self, bucket: str, prefix: str) -> set:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._get_existing_page_indices_s3_sync, bucket, prefix
        )
