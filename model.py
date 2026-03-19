import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


MODELS_S3_BUCKET = "project-moi-ai"
MODELS_S3_PREFIX = "Xihe"

MODEL_FILES = (
    [f"xihe_1to22_{day}day.onnx"  for day in range(1, 11)] +
    [f"xihe_23to33_{day}day.onnx" for day in range(1, 11)]
)


def _get_s3_client():
    # Create and return a boto3 S3 client using environment credentials.
    endpoint = os.environ.get("AWS_S3_ENDPOINT", "minio.dive.edito.eu")
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def download_xihe_models(output_dir):
    # Download all 20 XIHE ONNX models from S3, skip files already present locally.
    local_models_dir = os.environ.get("XIHE_MODELS_DIR")

    if local_models_dir:
        print(f"Using local models from: {local_models_dir}")
        local_path = Path(local_models_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local models directory not found: {local_models_dir}")
        missing = [f for f in MODEL_FILES if not (local_path / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} model files in {local_models_dir}")
        print(f"[OK] Found all {len(MODEL_FILES)} models locally")
        return {f: str(local_path / f) for f in MODEL_FILES}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    s3 = _get_s3_client()

    print(f"Downloading XIHE models from S3:")
    print(f"   Bucket: s3://{MODELS_S3_BUCKET}/{MODELS_S3_PREFIX}/")
    print(f"   Local:  {output_dir}")
    print(f"   Total:  {len(MODEL_FILES)} ONNX files")

    downloaded = {}
    total_size  = 0

    for model_file in MODEL_FILES:
        s3_key     = f"{MODELS_S3_PREFIX}/{model_file}"
        local_path = output_path / model_file

        if local_path.exists():
            size = local_path.stat().st_size
            total_size += size
            downloaded[model_file] = str(local_path)
            print(f"   [cached] {model_file} ({size / 1e6:.1f} MB)")
            continue

        try:
            print(f"   Downloading {model_file}...", end=" ", flush=True)
            s3.download_file(MODELS_S3_BUCKET, s3_key, str(local_path))
            size = local_path.stat().st_size
            total_size += size
            downloaded[model_file] = str(local_path)
            print(f"[OK] ({size / 1e6:.1f} MB)")
        except ClientError as e:
            raise RuntimeError(f"Failed to download {model_file}: {e}")

    print(f"[OK] All models ready ({total_size / 1e6:.1f} MB total)")
    return downloaded


def verify_models(models_dir):
    # Check that all 20 model files are present in the given directory.
    models_path = Path(models_dir)
    missing = [f for f in MODEL_FILES if not (models_path / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} model files:\n" + "\n".join(f"  - {m}" for m in missing))
    return True
