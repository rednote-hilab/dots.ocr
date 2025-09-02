import boto3
import os
from botocore.config import Config

# 配置
BUCKET_NAME = 'monkeyocr'
OBJECT_KEY = 'test/input/test_pdf/small.pdf'
FILE_NAME = 'test.pdf'

BUCKET_NAME_OSS = 'ofnil-ml-test'
OBJECT_KEY_OSS = 'ocr/test/Guide-to-U.S.-Healthcare-System.pdf'
FILE_NAME_OSS = 'test.pdf'


def download_file_oss(bucket_name, object_key, file_name):
    endpoint = os.getenv('OSS_ENDPOINT')
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
    secret_access_key = os.getenv('OSS_ACCESS_KEY_SECRET')
    s3_oss_client = boto3.client(
        's3',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        endpoint_url="https://oss-cn-hongkong.aliyuncs.com",
        config=Config(s3={"addressing_style": "virtual"},
                        signature_version='v4'))
    
    print(bucket_name, object_key, file_name)
    s3_oss_client.download_file(bucket_name, object_key, file_name)


def download_file(bucket_name, object_key, file_name):
    s3 = boto3.client('s3')

    try:
        s3.download_file(bucket_name, object_key, file_name)
        print(f"文件 '{object_key}' 已成功从 '{bucket_name}' 下载到 '{file_name}'")
    except Exception as e:
        print(f"下载文件时发生错误：{e}")

def upload_file_oss(bucket_name, object_key, file_name):
    endpoint = os.getenv('OSS_ENDPOINT')
    access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
    secret_access_key = os.getenv('OSS_ACCESS_KEY_SECRET')
    s3_oss_client = boto3.client(
        's3',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        endpoint_url="https://oss-cn-hongkong.aliyuncs.com",
        config=Config(s3={"addressing_style": "virtual"},
                        signature_version='v4'))
    
    print(bucket_name, object_key, file_name)
    s3_oss_client.upload_file(file_name, bucket_name, object_key)


def upload_file(bucket_name, object_key, file_name):
    s3 = boto3.client('s3')

    try:
        s3.upload_file(file_name, bucket_name, object_key)
        print(f"文件 '{object_key}' 已成功从 '{file_name}' upload到 '{object_key}'")
    except Exception as e:
        print(f"upload文件时发生错误：{e}")



if __name__ == '__main__':
    # download_file(BUCKET_NAME, OBJECT_KEY, FILE_NAME)
    # download_file_oss(BUCKET_NAME_OSS, OBJECT_KEY_OSS, FILE_NAME_OSS)

    upload_file("monkeyocr", "test/input/test_pdf/output1.jpg", "/dots.ocr/test/output/output1.jpg")