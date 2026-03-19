import os
import shutil
from pathlib import Path

import numpy as np
import yaml
import xarray as xr
import torch
from torchvision.transforms import transforms


ASSETS_BUCKET  = "project-moi-ai"
ASSETS_S3_KEYS = {
    "data.yaml":              "src/data.yaml",
    "normalize_mean_50.npz":  "src/normalize_mean_50.npz",
    "normalize_std_50.npz":   "src/normalize_std_50.npz",
    "mask_surface.npy":       "src/mask_surface.npy",
    "mask_deep.npy":          "src/mask_deep.npy",
    "mercator_lat.npy":       "data_get/mercator_lat.npy",
    "mercator_lon.npy":       "data_get/mercator_lon.npy",
}

CACHE_DIR = Path(os.environ.get("LOCAL_WORK_DIR", "/tmp/xihe")) / "assets_cache"

DEPTH_LEVELS = np.array([
    0.4940,  2.6457,  5.0782,  7.9296, 11.4050, 15.8101, 21.5988, 29.4447,
    40.3441, 55.7643, 77.8539, 92.3261, 109.7293, 130.6660, 155.8507, 186.1256,
    222.4752, 266.0403, 318.1274, 380.2130, 453.9377, 541.0889, 643.5668,
])

LAYER_INDEX = {
    "thetao": [2,  6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90],
    "so":     [3,  7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91],
    "uo":     [4,  8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92],
    "vo":     [5,  9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93],
    "zos":    [0],
}


def _get_s3_client():
    # Create and return a boto3 S3 client using environment credentials.
    import boto3
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


def download_assets():
    # Download all config assets from S3 to local cache, skip if already present.
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s3 = _get_s3_client()
    for filename, s3_key in ASSETS_S3_KEYS.items():
        local_path = CACHE_DIR / filename
        if not local_path.exists():
            print(f"Downloading asset: {filename} ...")
            s3.download_file(ASSETS_BUCKET, s3_key, str(local_path))
            print(f"[OK] {filename}")
    print(f"[OK] All assets in cache: {CACHE_DIR}")
    return str(CACHE_DIR)


def cleanup_assets():
    # Remove the local asset cache directory.
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        print(f"[OK] Cleaned up asset cache: {CACHE_DIR}")


def get_asset(filename):
    # Return local path to a cached asset, raise if not found.
    path = CACHE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Asset not found in cache: {filename}. Call download_assets() first.")
    return str(path)


class ProcessData:
    # Handles normalization and denormalization of XIHE input/output tensors.

    def __init__(self, origin_input, input_key, output_key):
        yaml_path = get_asset("data.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.content = yaml.safe_load(f)["data"]
        self.origin_input   = origin_input
        self.input_key      = input_key
        self.output_key     = output_key
        self.transforms     = self._get_normalize(self.content[input_key])
        self.out_transforms = self._get_normalize(self.content[output_key])

    def _get_normalize(self, variables):
        # Build a Normalize transform from the npz stats files.
        mean_npz = dict(np.load(get_asset("normalize_mean_50.npz")))
        std_npz  = dict(np.load(get_asset("normalize_std_50.npz")))
        mean = []
        for var in variables:
            if var != "sst":
                mean.append(mean_npz[var])
            else:
                mean.append(mean_npz[var] - 273.15)
        normalize_mean = np.concatenate(mean)
        normalize_std  = np.concatenate([std_npz[var] for var in variables])
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_denormalize(self):
        # Return the inverse normalize transform for post-processing.
        norm = self.out_transforms
        mean_norm, std_norm = norm.mean, norm.std
        mean_denorm = -mean_norm / std_norm
        std_denorm  = 1 / std_norm
        return transforms.Normalize(mean_denorm, std_denorm)

    def _create_var_map(self):
        # Map variable names to their index in the origin input.
        variables = self.content[self.origin_input]
        return {var: idx for idx, var in enumerate(variables)}

    def read_data(self, x):
        # Select and normalize the input channels for the model.
        var_map  = self._create_var_map()
        index_in = [var_map[var] for var in self.content[self.input_key]]
        data_x   = x[:, index_in, :, :]
        data_x[np.isnan(data_x).bool()] = -32767
        mask   = data_x < -30000
        data_x = self.transforms(data_x)
        data_x[mask] = 0
        return data_x.numpy().astype(np.float32)


def npy_to_zarr(output_surface_path, output_deep_path, zarr_path, lead_day, forecast_date):
    # Convert surface + deep NPY files for one lead day into the Zarr store, appending along time.
    lat = np.load(get_asset("mercator_lat.npy"))
    lon = np.load(get_asset("mercator_lon.npy"))

    surface_files = sorted(Path(output_surface_path).glob("*.npy"))
    deep_files    = sorted(Path(output_deep_path).glob("*.npy"))

    if not surface_files or not deep_files:
        raise RuntimeError(f"No NPY files found for day {lead_day}")

    file     = surface_files[0].name
    time_str = file[10:18]
    stime    = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:]}"
    vtime    = np.datetime64(stime, "h")
    ref_date = np.datetime64("1950-01-01", "h")
    vtime_int = int((vtime - ref_date) / np.timedelta64(1, "h"))

    npy_surface = np.load(str(surface_files[0]))
    npy_deep    = np.load(str(deep_files[0]))
    npy_data    = np.concatenate([npy_surface, npy_deep], axis=1)

    data_vars = {
        "thetao": (["time", "depth", "latitude", "longitude"], npy_data[:, LAYER_INDEX["thetao"], :, :]),
        "so":     (["time", "depth", "latitude", "longitude"], npy_data[:, LAYER_INDEX["so"],     :, :]),
        "uo":     (["time", "depth", "latitude", "longitude"], npy_data[:, LAYER_INDEX["uo"],     :, :]),
        "vo":     (["time", "depth", "latitude", "longitude"], npy_data[:, LAYER_INDEX["vo"],     :, :]),
        "zos":    (["time", "latitude", "longitude"],          np.squeeze(npy_data[:, LAYER_INDEX["zos"], :, :], axis=0)),
    }
    coords = {
        "time":      [vtime],
        "depth":     DEPTH_LEVELS,
        "latitude":  lat,
        "longitude": lon,
    }
    ds = xr.Dataset(data_vars, coords=coords)

    mask_nan  = np.isnan(ds["uo"][0, 0, :, :])
    ds["zos"] = ds["zos"].where(~mask_nan)

    import zarr
    from numcodecs import Blosc
    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    encoding   = {var: {"compressor": compressor} for var in ds.data_vars}

    zarr_store = Path(zarr_path)
    if not zarr_store.exists():
        ds.to_zarr(str(zarr_store), mode="w", encoding=encoding)
    else:
        ds.to_zarr(str(zarr_store), mode="a", append_dim="time")

    print(f"[OK] Day {lead_day} written to Zarr: {zarr_path}")


def zarr_to_zip(zarr_path, zip_path):
    # Convert a Zarr directory store into a single zip file and remove the original directory.
    import zarr
    print(f"Converting Zarr store to zip...")
    print(f"   From: {zarr_path}")
    print(f"   To:   {zip_path}")

    source = zarr.open(str(zarr_path), mode="r")
    with zarr.ZipStore(str(zip_path), mode="w") as zip_store:
        zarr.copy_store(source.store, zip_store)

    size_mb = Path(zip_path).stat().st_size / 1e6
    print(f"[OK] Zarr zip: {zip_path} ({size_mb:.2f} MB)")

    shutil.rmtree(zarr_path)
    print(f"[OK] Removed temporary Zarr: {zarr_path}")
    return str(zip_path)
