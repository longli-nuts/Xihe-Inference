#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from get_inits_cmems import fetch_marine_data
from get_inits_wind import get_ifs_wind
from s3_upload import save_file_to_s3


LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "/tmp/xihe_test_init")
MARINE_INIT_FILE_NAME = "marine_init.nc"
WIND_INIT_FILE_NAME = "wind_init.nc"


def main():
    bucket_name = os.environ.get("AWS_BUCKET_NAME")
    s3_prefix = os.environ.get("TEST_INIT_S3_PREFIX", "xihe-test-init")
    test_date_str = os.environ.get(
        "FORECAST_DATE",
        (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    if not bucket_name:
        print("[ERROR] AWS_BUCKET_NAME is required")
        sys.exit(1)

    try:
        test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERROR] TEST_INIT_DATE '{test_date_str}' must be YYYY-MM-DD")
        sys.exit(1)

    work_dir = Path(LOCAL_WORK_DIR)
    marine_dir = work_dir / "marine"
    wind_dir = work_dir / "wind"
    marine_dir.mkdir(parents=True, exist_ok=True)
    wind_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test init for {test_date}...")
    marine_file = fetch_marine_data(
        forecast_date=test_date,
        output_dir=str(marine_dir),
    )
    wind_file = get_ifs_wind(
        forecast_date=test_date,
        output_dir=str(wind_dir),
    )

    custom_init_prefix = f"{s3_prefix}/{test_date}"
    fixed_marine_file = work_dir / MARINE_INIT_FILE_NAME
    fixed_wind_file = work_dir / WIND_INIT_FILE_NAME
    Path(marine_file).replace(fixed_marine_file)
    Path(wind_file).replace(fixed_wind_file)

    marine_key = f"{custom_init_prefix}/{MARINE_INIT_FILE_NAME}"
    wind_key = f"{custom_init_prefix}/{WIND_INIT_FILE_NAME}"

    save_file_to_s3(
        bucket_name=bucket_name,
        local_file_path=str(fixed_marine_file),
        object_key=marine_key,
    )
    save_file_to_s3(
        bucket_name=bucket_name,
        local_file_path=str(fixed_wind_file),
        object_key=wind_key,
    )

    print("\nUse these for CUSTOM mode:")
    print(f"  export FORECAST_DATE={test_date}")
    print(f"  export INIT_FILES_FOLDER_URL=s3://{bucket_name}/{custom_init_prefix}")


if __name__ == "__main__":
    main()
