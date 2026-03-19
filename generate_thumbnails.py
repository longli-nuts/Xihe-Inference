import io
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from s3_upload import upload_bytes_to_s3


def generate_thumbnails(zarr_path, bucket_name, s3_prefix):
    # Generate PNG thumbnails for all ocean variables at day 1 and upload them to S3.
    print(f"Generating thumbnails from: {Path(zarr_path).name}")

    ds = xr.open_zarr(zarr_path)

    variables = {
        "zos":    {"title": "Sea Surface Height",           "cmap": "RdBu_r",   "units": "m"},
        "thetao": {"title": "Temperature (Surface)",        "cmap": "RdYlBu_r", "units": "°C"},
        "so":     {"title": "Salinity (Surface)",           "cmap": "viridis",  "units": "PSU"},
        "uo":     {"title": "Eastward Velocity (Surface)",  "cmap": "RdBu_r",   "units": "m/s"},
        "vo":     {"title": "Northward Velocity (Surface)", "cmap": "RdBu_r",   "units": "m/s"},
    }

    thumbnail_urls = {}

    for var_name, config in variables.items():
        if var_name not in ds:
            print(f"  [WARNING] {var_name} not found, skipping")
            continue

        try:
            data = ds[var_name].isel(time=0)
            if "depth" in data.dims:
                data = data.isel(depth=0)

            lat = data.latitude if "latitude" in data.dims else data.lat
            lon = data.longitude if "longitude" in data.dims else data.lon

            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            vmin, vmax = np.nanpercentile(data.values, [2, 98])

            im = ax.pcolormesh(lon, lat, data.values, cmap=config["cmap"],
                               vmin=vmin, vmax=vmax, shading="auto")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"{config['title']} - Day 1 Forecast", fontsize=14, fontweight="bold")

            cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40)
            cbar.set_label(config["units"], fontsize=10)
            ax.set_aspect("auto")
            plt.tight_layout()

            buffer = io.BytesIO()
            plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            buffer.seek(0)
            plt.close(fig)

            object_key = f"{s3_prefix}/{var_name}.png"
            s3_url = upload_bytes_to_s3(
                bucket_name=bucket_name,
                data_bytes=buffer.getvalue(),
                object_key=object_key,
                content_type="image/png",
            )
            thumbnail_urls[var_name] = s3_url
            print(f"  [OK] {var_name}.png -> {s3_url}")

        except Exception as e:
            print(f"  [WARNING] Failed for {var_name}: {e}")

    ds.close()
    print(f"[OK] {len(thumbnail_urls)} thumbnails generated")
    return thumbnail_urls
