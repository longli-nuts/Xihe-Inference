import os
import time
import subprocess
import pathlib
import multiprocessing
import psutil
import numpy as np
import xarray as xr
import copernicusmarine
from pathlib import Path


DATASETS = {
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m": ["thetao"],
    "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m":     ["so"],
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m":    ["uo", "vo"],
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m":        ["zos"],
}

DEPTH_MIN = 0.49402499198913574
DEPTH_MAX = 5727.9169921875
LON_MIN   = -180.0
LON_MAX   =  180.0
LAT_MIN   = -80.0
LAT_MAX   =  90.0

SURFACE_LAYERS = list(range(0, 22, 2)) + [21]
DEEP_LAYERS    = list(range(22, 33))

_MARINE_FILE       = None
_U10               = None
_V10               = None
_SAVE_SURFACE_PATH = None
_SAVE_DEEP_PATH    = None


def fetch_marine_data(forecast_date, output_dir):
    # Download GLO12 ocean variables for the given date and merge into a single NetCDF.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_date  = forecast_date.strftime("%Y-%m-%d")
    output_file = output_path / f"mercatorglorys12v1_gl12_mean_{forecast_date.strftime('%Y%m%d')}.nc"

    if output_file.exists():
        print(f"[OK] Already exists: {output_file}")
        return str(output_file)

    print(f"Fetching Copernicus Marine data (4 datasets):")
    print(f"   Date: {start_date}")

    tmp_files = []
    for dataset_id, variables in DATASETS.items():
        tmp_file  = output_path / f"tmp_{dataset_id}_{forecast_date.strftime('%Y%m%d')}.nc"
        depth_min = None if variables == ["zos"] else DEPTH_MIN
        depth_max = None if variables == ["zos"] else DEPTH_MAX

        print(f"   Downloading {dataset_id} -> {variables}...")
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=LON_MIN,
            maximum_longitude=LON_MAX,
            minimum_latitude=LAT_MIN,
            maximum_latitude=LAT_MAX,
            start_datetime=start_date,
            end_datetime=start_date,
            minimum_depth=depth_min,
            maximum_depth=depth_max,
            output_filename=str(tmp_file.name),
            output_directory=str(output_path),
        )
        tmp_files.append(str(tmp_file))
        print(f"   [OK] {variables}")

    print(f"Merging {len(tmp_files)} files...")
    merge_cmd = ["cdo", "-O", "merge", *tmp_files, str(output_file)]
    result = subprocess.run(merge_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CDO merge failed:\n{result.stderr}")

    for f in tmp_files:
        Path(f).unlink(missing_ok=True)

    print(f"[OK] {output_file} ({output_file.stat().st_size / 1e6:.2f} MB)")
    return str(output_file)


def _task_surface(i):
    # Worker: build and save surface NPY array from GLO12 + IFS wind data.
    print("Processing surface layers...")
    df_mkt = xr.open_dataset(_MARINE_FILE)
    dtstr  = str(df_mkt["time"].values)[2:12].replace("-", "")

    zos    = np.expand_dims(df_mkt["zos"].values, axis=0)
    u      = _U10[0, :, :].reshape(1, 1, 2041, 4320)
    v      = _V10[0, :, :].reshape(1, 1, 2041, 4320)
    sst    = df_mkt["thetao"].values[0, 0, :, :].reshape(1, 1, 2041, 4320)
    thetao = np.expand_dims(df_mkt["thetao"].values[0, SURFACE_LAYERS, :, :], axis=0)
    so     = np.expand_dims(df_mkt["so"].values[0, SURFACE_LAYERS, :, :], axis=0)
    uo     = np.expand_dims(df_mkt["uo"].values[0, SURFACE_LAYERS, :, :], axis=0)
    vo     = np.expand_dims(df_mkt["vo"].values[0, SURFACE_LAYERS, :, :], axis=0)

    parts = [zos, u, v, sst]
    for idx in range(len(SURFACE_LAYERS)):
        parts += [
            thetao[:, idx, :, :].reshape(1, 1, 2041, 4320),
            so[:, idx, :, :].reshape(1, 1, 2041, 4320),
            uo[:, idx, :, :].reshape(1, 1, 2041, 4320),
            vo[:, idx, :, :].reshape(1, 1, 2041, 4320),
        ]

    data = np.concatenate(parts, axis=1).astype(np.float16)
    np.save(os.path.join(_SAVE_SURFACE_PATH, f"mra5_{dtstr}.npy"), data)
    print(f"[OK] Surface saved: mra5_{dtstr}.npy")


def _task_deep(i):
    # Worker: build and save deep NPY array from GLO12 + IFS wind data.
    print("Processing deep layers...")
    df_mkt = xr.open_dataset(_MARINE_FILE)
    dtstr  = str(df_mkt["time"].values)[2:12].replace("-", "")

    zos    = np.expand_dims(df_mkt["zos"].values, axis=0)
    u      = _U10.mean(axis=0).reshape(1, 1, 2041, 4320)
    v      = _V10.mean(axis=0).reshape(1, 1, 2041, 4320)
    sst    = df_mkt["thetao"].values[0, 0, :, :].reshape(1, 1, 2041, 4320)
    thetao = np.expand_dims(df_mkt["thetao"].values[0, DEEP_LAYERS, :, :], axis=0)
    so     = np.expand_dims(df_mkt["so"].values[0, DEEP_LAYERS, :, :], axis=0)
    uo     = np.expand_dims(df_mkt["uo"].values[0, DEEP_LAYERS, :, :], axis=0)
    vo     = np.expand_dims(df_mkt["vo"].values[0, DEEP_LAYERS, :, :], axis=0)

    parts = [zos, u, v, sst]
    for idx in range(len(DEEP_LAYERS)):
        parts += [
            thetao[:, idx, :, :].reshape(1, 1, 2041, 4320),
            so[:, idx, :, :].reshape(1, 1, 2041, 4320),
            uo[:, idx, :, :].reshape(1, 1, 2041, 4320),
            vo[:, idx, :, :].reshape(1, 1, 2041, 4320),
        ]

    data = np.concatenate(parts, axis=1).astype(np.float16)
    np.save(os.path.join(_SAVE_DEEP_PATH, f"mra5_{dtstr}.npy"), data)
    print(f"[OK] Deep saved: mra5_{dtstr}.npy")


def preprocess_to_npy(marine_file, wind_file, save_path):
    # Load GLO12 and IFS wind data, then run surface and deep preprocessing workers in parallel.
    global _MARINE_FILE, _U10, _V10, _SAVE_SURFACE_PATH, _SAVE_DEEP_PATH

    save_surface_path = os.path.join(save_path, "input_surface_data")
    save_deep_path    = os.path.join(save_path, "input_deep_data")
    pathlib.Path(save_surface_path).mkdir(exist_ok=True, parents=True)
    pathlib.Path(save_deep_path).mkdir(exist_ok=True, parents=True)

    df_wind = xr.open_dataset(wind_file, engine="cfgrib")
    u10     = df_wind["u10"].values
    v10     = df_wind["v10"].values
    print(f"IFS wind shape: u10={u10.shape}, v10={v10.shape}")

    _MARINE_FILE       = marine_file
    _U10               = u10
    _V10               = v10
    _SAVE_SURFACE_PATH = save_surface_path
    _SAVE_DEEP_PATH    = save_deep_path

    start     = time.time()
    n_workers = max(1, psutil.cpu_count() - 1)
    print(f"Number of processes: {n_workers}")

    pool = multiprocessing.Pool(n_workers)
    pool.map(_task_surface, [0])
    pool.map(_task_deep, [0])
    pool.close()
    pool.join()

    print(f"Preprocessing done in {time.time() - start:.1f}s")
