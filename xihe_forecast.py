import os
import time
import pathlib
from datetime import datetime, timedelta

import numpy as np
import torch
import onnxruntime
import torchdata.datapipes as dp

from utilities import ProcessData, get_asset


def run_inference(input_path, save_path, lead_day, models_dir):
    # Run ONNX inference for surface and deep layers for a given lead day and save NPY outputs.
    if not (1 <= lead_day <= 10):
        raise ValueError("lead_day must be between 1 and 10")

    input_surface_path  = os.path.join(input_path,  "input_surface_data")
    input_deep_path     = os.path.join(input_path,  "input_deep_data")
    output_surface_path = os.path.join(save_path,   "output_surface_data")
    output_deep_path    = os.path.join(save_path,   "output_deep_data")

    pathlib.Path(output_surface_path).mkdir(exist_ok=True, parents=True)
    pathlib.Path(output_deep_path).mkdir(exist_ok=True, parents=True)

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers.insert(0, "CUDAExecutionProvider")

    depth_configs = [
        {
            "input_path":  input_surface_path,
            "output_path": output_surface_path,
            "mask_asset":  "mask_surface.npy",
            "layer":       "1to22",
        },
        {
            "input_path":  input_deep_path,
            "output_path": output_deep_path,
            "mask_asset":  "mask_deep.npy",
            "layer":       "23to33",
        },
    ]

    for depth in depth_configs:
        data_path_list = list(dp.iter.FileLister(depth["input_path"]))
        mask_file      = np.load(get_asset(depth["mask_asset"]))

        onnx_path = os.path.join(models_dir, f"xihe_{depth['layer']}_{lead_day}day.onnx")
        print(f"Loading model: {onnx_path}")

        processor      = ProcessData(
            origin_input=f"variables_{depth['layer']}",
            input_key=f"input_{depth['layer']}",
            output_key=f"output_{depth['layer']}",
        )
        out_transforms = processor.get_denormalize()
        ort_session    = onnxruntime.InferenceSession(onnx_path, providers=providers)
        print("Model loaded.")

        for path in data_path_list:
            x = np.load(path).astype(np.float32)

            file_name   = path.split("/")[-1]
            date_string = file_name.split("_")[1].split(".")[0]
            print(f"cur_date: {date_string}")

            date      = datetime.strptime(date_string, "%Y%m%d")
            save_date = date + timedelta(days=lead_day)
            pred_date = save_date.strftime("%Y%m%d")
            print(f"pred_date: {pred_date}")

            x = torch.tensor(x)
            x = processor.read_data(x)

            start      = time.time()
            ort_output = ort_session.run(None, {"input": x})[0]
            y          = out_transforms(torch.from_numpy(ort_output))
            print(f"Inference time: {time.time() - start:.2f}s")

            y = y.cpu().numpy()
            y[mask_file] = np.nan

            out_file = os.path.join(depth["output_path"], f"pred_mra5_{pred_date}.npy")
            np.save(out_file, y)
            print(f"[OK] Saved: {out_file}")
