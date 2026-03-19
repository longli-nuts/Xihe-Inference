import os
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError


def get_s3_endpoint_url_with_protocol():
    # Return the S3 endpoint URL with https:// prefix if missing.
    endpoint = os.environ.get("AWS_S3_ENDPOINT", "minio.dive.edito.eu")
    if not endpoint.startswith("http"):
        return f"https://{endpoint}"
    return endpoint


def get_s3_client():
    # Create and return a boto3 S3 client using environment credentials.
    return boto3.client(
        "s3",
        endpoint_url=get_s3_endpoint_url_with_protocol(),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def save_file_to_s3(bucket_name, local_file_path, object_key):
    # Upload a single local file to S3 using 500MB multipart chunks for MinIO compatibility.
    s3_client  = get_s3_client()
    local_path = Path(local_file_path)

    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_file_path}")

    config = TransferConfig(
        multipart_threshold=1024 * 1024 * 500,
        max_concurrency=1,
        multipart_chunksize=1024 * 1024 * 500,
        use_threads=False,
    )

    print(f"Uploading: {local_path.name} ({local_path.stat().st_size / 1e6:.2f} MB)")
    s3_client.upload_file(str(local_path), bucket_name, object_key, Config=config)

    s3_url = f"s3://{bucket_name}/{object_key}"
    print(f"[OK] {s3_url}")
    return s3_url


def upload_bytes_to_s3(bucket_name, data_bytes, object_key, content_type="application/octet-stream"):
    # Upload raw bytes directly to S3 (used for thumbnails).
    s3_client = get_s3_client()
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=data_bytes,
        ContentType=content_type,
    )
    return f"s3://{bucket_name}/{object_key}"


def download_from_s3(bucket_name, object_key, local_file_path):
    # Download a single file from S3 to a local path.
    s3_client = get_s3_client()
    local_path = Path(local_file_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {bucket_name}/{object_key}")
    s3_client.download_file(bucket_name, object_key, str(local_path))
    print(f"[OK] {local_path}")
    return str(local_path)
