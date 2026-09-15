# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Step 7b: HDF5 Exporter (optional output format).

Convert a LeRobot v3.0 dataset directory to standard HDF5:
  <output>.hdf5
    ├── data/ep_<i>/observations/state      (T, D) float32
    │                └──────── images    (T, H, W, 3) uint8   (with --images)
    │              actions                    (T, D) float32
    │              attrs: task
    └── attrs: robot / fps / episodes

Usage:
  python cli.py export --input <lerobot_dataset_dir> --output <out.hdf5> [--images]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def export_hdf5(input_dir: str, output: str, with_images: bool = False,
                verbose: bool = True) -> str:
    """Export a LeRobot v3.0 dataset to HDF5 and return the output path."""
    import h5py
    import pyarrow.parquet as pq
    import cv2 as cv

    inp = Path(input_dir)
    info = json.load(open(inp / "meta" / "info.json"))
    eps_rows = pq.read_table(
        inp / "meta" / "episodes/chunk-000/file-000.parquet").to_pylist()
    tasks_py = pq.read_table(inp / "meta" / "tasks.parquet").to_pydict()
    task_by_idx = {int(t): str(n)
                   for t, n in zip(tasks_py["task_index"], tasks_py["task"])}
    if with_images:
        H = info["features"]["observation.images.ego"]["shape"][0]
        W = info["features"]["observation.images.ego"]["shape"][1]

    # Frame-level data (single parquet chunk)
    table = pq.read_table(inp / "data/chunk-000/file-000.parquet")
    cols = {name: table.column(name).to_numpy()
            for name in ("action", "observation.state", "index")}

    cap = None
    n_vid = 0
    if with_images:
        cap = cv.VideoCapture(str(inp / "videos/observation.images.ego/chunk-000/file-000.mp4"))
        n_vid = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0

    f = h5py.File(output, "w")
    f.attrs["robot"] = str(info.get("robot_type", "panda_dual"))
    f.attrs["fps"] = int(info.get("fps", 30))
    f.attrs["episodes"] = len(eps_rows)
    f.attrs["source"] = "ego2robot pipeline (EgoVerse -> LeRobot v3.0 converted)"

    t0 = time.time()
    for row in eps_rows:
        ep = int(row["episode_index"])
        n = int(row["length"])
        if n <= 0:
            continue
        i0 = int(row["dataset_from_index"])
        for k in row:
            if k.startswith("videos/observation.images.ego/from_"):
                v_from = int(row[k])
                break
        else:
            v_from = 0
        tk = int(row.get("task_index", 0)) if (row.get("tasks") is None) else 0
        task_name = task_by_idx.get(tk, "manipulation")

        g = f.create_group(f"data/ep_{ep:03d}")
        g.create_dataset("state", data=np.stack(cols["observation.state"][i0:i0 + n]),
                         compression="gzip")
        g.create_dataset("actions", data=np.stack(cols["action"][i0:i0 + n]),
                         compression="gzip")
        g.attrs["task"] = task_name
        if with_images and cap is not None and n_vid > 0:
            imgs = np.empty((n, H, W, 3), np.uint8)
            for t in range(n):
                cap.set(cv.CAP_PROP_POS_FRAMES, v_from + t)
                ok, fr = cap.read()
                if ok:
                    imgs[t] = fr
            g.create_dataset("observations/images", data=imgs,
                             compression="gzip", shuffle=True)
        if verbose:
            print(f"  ep_{ep:03d}: {n} frames, task='{task_name}' "
                  f"({time.time()-t0:.1f}s)", flush=True)
    if cap is not None:
        cap.release()
    f.close()
    return output


def build_arg_parser():
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(description="Step 7b: Dataset format export")
    ap.add_argument("--input", required=True, help="LeRobot v3.0 dataset directory")
    ap.add_argument("--format", default="hdf5",
                    help="Target format: currently supports hdf5 (extensible later)")
    ap.add_argument("--output", required=True, help="Output file path (*.hdf5)")
    ap.add_argument("--images", action="store_true",
                    help="Decode the ego video into HDF5 (large); by default write only state/action")
    return ap


def run(args):
    """Run the export workflow."""
    if args.format != "hdf5":
        raise ValueError(f"Unsupported format: {args.format} (currently only hdf5)")
    out = export_hdf5(args.input, args.output, with_images=args.images)
    print(f"✓ HDF5 -> {out}")


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
