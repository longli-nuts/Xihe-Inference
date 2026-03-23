#!/usr/bin/env python3
import os
import sys
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from model           import download_xihe_models
from get_inits_cmems import fetch_marine_data, preprocess_to_npy
from get_inits_wind  import get_ifs_wind
from xihe_forecast   import run_inference
from utilities       import download_assets, cleanup_assets, npy_to_zarr, zarr_to_zip
from s3_upload       import download_from_s3, save_file_to_s3
from generate_thumbnails import generate_thumbnails


LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "/tmp/xihe")
MARINE_INIT_FILE_NAME = "marine_init.nc"
WIND_INIT_FILE_NAME = "wind_init.nc"


def validate_environment(use_custom_init: bool):
    # Check that all required environment variables are set.
    required = [
        "AWS_BUCKET_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]
    if not use_custom_init:
        required.extend(
            [
                "COPERNICUSMARINE_SERVICE_USERNAME",
                "COPERNICUSMARINE_SERVICE_PASSWORD",
                "ECMWF_API_KEY",
                "ECMWF_API_EMAIL",
            ]
        )
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print("[ERROR] Missing required environment variables:")
        for v in missing:
            print(f"   - {v}")
        sys.exit(1)


def download_init_file(init_url: str, local_dir: Path) -> str:
    # Download a custom init file from S3 into the local work directory.
    if not init_url.startswith("s3://"):
        raise ValueError(f"Only s3:// URLs are supported for custom init files, got: {init_url}")

    bucket_and_key = init_url[len("s3://"):]
    bucket_name, object_key = bucket_and_key.split("/", 1)
    local_path = local_dir / Path(object_key).name
    return download_from_s3(bucket_name, object_key, str(local_path))


def build_s3_file_url(folder_url: str, file_name: str) -> str:
    # Build a file URL from a custom init folder URL and a fixed file name.
    return folder_url.rstrip("/") + "/" + file_name


def main():
    print("=" * 70)
    print(" " * 20 + "XIHE OCEAN FORECASTING")
    print("=" * 70)

    init_folder_url = os.environ.get("INIT_FILES_FOLDER_URL")
    use_custom_init = bool(init_folder_url)

    validate_environment(use_custom_init)

    default_date_str  = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_date_str = os.environ.get("FORECAST_DATE", default_date_str)
    try:
        forecast_date = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERROR] FORECAST_DATE '{forecast_date_str}' must be YYYY-MM-DD")
        sys.exit(1)

    print(f"Mode: {'CUSTOM' if use_custom_init else 'AUTO'}")
    print(f"Forecast Date: {forecast_date}")

    bucket_name      = os.environ.get("AWS_BUCKET_NAME")
    s3_output_folder = os.environ.get("S3_OUTPUT_FOLDER", "xihe-forecasts")
    forecast_start   = forecast_date + timedelta(days=1)
    forecast_end     = forecast_date + timedelta(days=10)
    zarr_name        = f"XIHE_MOI_{forecast_start}_{forecast_end}.zarr"

    print(f"Output: s3://{bucket_name}/{s3_output_folder}/{forecast_date}/")

    work_dir        = Path(LOCAL_WORK_DIR)
    raw_data_dir    = work_dir / "raw_data"
    input_data_dir  = work_dir / "input_data"
    output_data_dir = work_dir / "output_data"
    zarr_path       = work_dir / zarr_name

    for d in [raw_data_dir, input_data_dir, output_data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        print("\n" + "=" * 70)
        print("STEP 1: Downloading assets")
        print("=" * 70)
        download_assets()

        print("\n" + "=" * 70)
        print("STEP 2: XIHE Models")
        print("=" * 70)
        local_models_dir = os.environ.get("XIHE_MODELS_DIR")
        if local_models_dir:
            models_dir = Path(local_models_dir)
            print(f"[OK] Using local models: {models_dir}")
        else:
            models_dir = work_dir / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            download_xihe_models(str(models_dir))

        if use_custom_init:
            print("\n" + "=" * 70)
            print("STEP 3: Using custom init files")
            print("=" * 70)
            custom_init_dir = raw_data_dir / "custom_init"
            custom_init_dir.mkdir(parents=True, exist_ok=True)
            marine_init_url = build_s3_file_url(init_folder_url, MARINE_INIT_FILE_NAME)
            wind_init_url = build_s3_file_url(init_folder_url, WIND_INIT_FILE_NAME)
            marine_file = download_init_file(marine_init_url, custom_init_dir)
            wind_file = download_init_file(wind_init_url, custom_init_dir)
            print(f"[OK] Marine init: {marine_file}")
            print(f"[OK] Wind init:   {wind_file}")
        else:
            print("\n" + "=" * 70)
            print("STEP 3: Fetching Copernicus Marine (GLO12)")
            print("=" * 70)
            marine_file = fetch_marine_data(
                forecast_date=forecast_date,
                output_dir=str(raw_data_dir / "mkt_data"),
            )
            print(f"[OK] {marine_file}")

            print("\n" + "=" * 70)
            print("STEP 4: Fetching IFS wind forecast (ECMWF)")
            print("=" * 70)
            wind_file = get_ifs_wind(
                forecast_date=forecast_date,
                output_dir=str(raw_data_dir / "ifs_wind"),
            )
            print(f"[OK] {wind_file}")

        print("\n" + "=" * 70)
        print("STEP 5: Preprocessing to NPY")
        print("=" * 70)
        surface_files = list((input_data_dir / "input_surface_data").glob("*.npy"))
        deep_files    = list((input_data_dir / "input_deep_data").glob("*.npy"))
        if surface_files and deep_files:
            print(f"[OK] Cached - skipping preprocessing")
        else:
            preprocess_to_npy(
                marine_file=marine_file,
                wind_file=wind_file,
                save_path=str(input_data_dir),
            )

        print("\n" + "=" * 70)
        print("STEP 6: Inference (10 days)")
        print("=" * 70)
        output_surface = output_data_dir / "output_surface_data"
        output_deep    = output_data_dir / "output_deep_data"

        for lead_day in range(1, 11):
            print(f"\n  Day {lead_day}/10...")
            for f in output_surface.glob("*.npy"):
                f.unlink()
            for f in output_deep.glob("*.npy"):
                f.unlink()

            run_inference(
                input_path=str(input_data_dir),
                save_path=str(output_data_dir),
                lead_day=lead_day,
                models_dir=str(models_dir),
            )
            npy_to_zarr(
                output_surface_path=str(output_surface),
                output_deep_path=str(output_deep),
                zarr_path=str(zarr_path),
                lead_day=lead_day,
                forecast_date=forecast_date,
            )
            print(f"  [OK] Day {lead_day}")

        print("\n" + "=" * 70)
        print("STEP 7: Converting Zarr to zip")
        print("=" * 70)
        zip_path = str(zarr_path) + ".zip"
        zarr_to_zip(str(zarr_path), zip_path)

        print("\n" + "=" * 70)
        print("STEP 8: Uploading to S3")
        print("=" * 70)
        zip_name   = f"{zarr_name}.zip"
        file_key   = f"{s3_output_folder}/{forecast_date}/{zip_name}"
        result_url = save_file_to_s3(
            bucket_name=bucket_name,
            local_file_path=zip_path,
            object_key=file_key,
        )
        print(f"[OK] {result_url}")

        print("\n" + "=" * 70)
        print("STEP 9: Generating thumbnails")
        print("=" * 70)
        try:
            thumb_prefix   = f"{s3_output_folder}/{forecast_date}/thumbnails"
            thumbnail_urls = generate_thumbnails(
                zarr_path=zip_path,
                bucket_name=bucket_name,
                s3_prefix=thumb_prefix,
            )
            print(f"[OK] {len(thumbnail_urls)} thumbnails")
        except Exception as e:
            print(f"[WARNING] Thumbnail generation failed: {e}")

        cleanup_assets()
        shutil.rmtree(work_dir)
        print(f"\n[OK] Cleaned up: {work_dir}")

        print("\n" + "=" * 70)
        print("[OK] XIHE FORECAST COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Date:     {forecast_date}")
        print(f"Forecast: {forecast_start} to {forecast_end} (10 days)")
        print(f"Result:   {result_url}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
