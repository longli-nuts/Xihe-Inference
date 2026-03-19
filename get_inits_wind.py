import os
import sys
import subprocess
from pathlib import Path
from ecmwfapi import ECMWFService


def _get_mars_client():
    # Create MARS client from ECMWF_API_KEY and ECMWF_API_EMAIL environment variables.
    api_key   = os.environ.get("ECMWF_API_KEY")
    api_email = os.environ.get("ECMWF_API_EMAIL")
    if not api_key or not api_email:
        print("[ERROR] Missing ECMWF environment variables:")
        if not api_key:
            print("   - ECMWF_API_KEY  (https://api.ecmwf.int/v1/key/)")
        if not api_email:
            print("   - ECMWF_API_EMAIL")
        sys.exit(1)
    os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
    return ECMWFService("mars")


def fetch_ifs_wind(forecast_date, output_dir):
    # Download IFS forecast wind (u10/v10) via MARS for a given date and return the GRIB file path.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = forecast_date.strftime("%Y%m%d")
    raw_file = output_path / f"ifs_wind_{date_str}.grib"

    if raw_file.exists():
        print(f"[OK] IFS wind already exists: {raw_file}")
        return str(raw_file)

    print(f"Fetching IFS wind forecast via MARS:")
    print(f"   Date:      {forecast_date}")
    print(f"   Variables: u10 (165), v10 (166)")
    print(f"   Steps:     0 to 240h (10 days, every 6h)")

    server = _get_mars_client()
    server.execute(
        {
            "class":   "od",
            "date":    forecast_date.strftime("%Y-%m-%d"),
            "expver":  "1",
            "levtype": "sfc",
            "param":   "165/166",
            "step":    "0/to/240/by/6",
            "stream":  "oper",
            "time":    "00",
            "type":    "fc",
            "grid":    "0.25/0.25",
            "area":    "90/-180/-80/180",
        },
        str(raw_file),
    )

    print(f"[OK] IFS wind downloaded: {raw_file} ({raw_file.stat().st_size / 1e6:.2f} MB)")
    return str(raw_file)


def upsample_wind(raw_file, output_dir, date_str):
    # Upsample IFS wind from 0.25deg to 0.083deg using CDO bilinear interpolation.
    output_path    = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    upsampled_file = output_path / f"ifs_wind_0.083deg_{date_str}.nc"

    if upsampled_file.exists():
        print(f"[OK] Already upsampled: {upsampled_file}")
        return str(upsampled_file)

    print(f"Upsampling IFS wind to 0.083deg...")
    cdo_cmd = (
        f"cdo -P 16 sellonlatbox,-180,180,-80,90 "
        f"-remapbil,r4320x2161 {raw_file} {upsampled_file}"
    )
    result = subprocess.run(cdo_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CDO upsampling failed:\n{result.stderr}")

    print(f"[OK] Upsampled: {upsampled_file} ({upsampled_file.stat().st_size / 1e6:.2f} MB)")
    return str(upsampled_file)


def get_ifs_wind(forecast_date, output_dir):
    # Fetch IFS wind from MARS, upsample to XIHE grid, delete raw GRIB and return final NetCDF path.
    date_str = forecast_date.strftime("%Y%m%d")
    raw_dir  = Path(output_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_file       = fetch_ifs_wind(forecast_date, str(raw_dir))
    upsampled_file = upsample_wind(raw_file, output_dir, date_str)

    Path(raw_file).unlink(missing_ok=True)
    print(f"Removed raw GRIB: {raw_file}")
    return upsampled_file


if __name__ == "__main__":
    from datetime import datetime, timedelta
    test_date = datetime.now().date() - timedelta(days=1)
    result = get_ifs_wind(test_date, "/tmp/xihe_test/wind")
    print(f"Wind file ready: {result}")
