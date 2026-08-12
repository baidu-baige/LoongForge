#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from https://github.com/thu-ml/Motus under the Apache-2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Online-vs-online VAE-input parity check (DATA link).

Compares the LATENT_DUMP dumps of TWO separate online training runs (both
launched with the SAME PARITY_DATA_SEED so the sampler visits identical anchors
in identical order). For each matching rankR_stepN.pt it bit-compares:

  - first_frame        (decode + normalization + H2D)
  - video_frames       (decode + normalization + H2D)
  - clean_full_latent  (end-to-end: data + VAE compute)

A `torch.equal` pass on first_frame / video_frames proves the data pipeline
(torchcodec decode + transform + collate) is bit-deterministic ACROSS PROCESSES
-- i.e. re-fetching the same (episode, condition_frame) anchor in the offline
precompute will get the exact same input pixels. Combined with
verify_offline_latents.py (compute link), a pass on both scripts means the
offline latent cache is end-to-end bit-identical to the online encode.

Usage
-----
  python examples/embodied/motus/diff_online_dumps.py \
      --dir-a /tmp/latent_dump_run1 --dir-b /tmp/latent_dump_run2
"""
from __future__ import annotations

import argparse
import glob
import os

import torch


def _cmp(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"SHAPE-MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}"
    if torch.equal(a, b):
        return "equal"
    diff = (a.float() - b.float()).abs().max().item()
    return f"DIFFERS max|abs|={diff:.3e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True, help="dump dir of online run #1")
    ap.add_argument("--dir-b", required=True, help="dump dir of online run #2")
    args = ap.parse_args()

    files_a = sorted(glob.glob(os.path.join(args.dir_a, "rank*_step*.pt")))
    if not files_a:
        print(f"[FAIL] no dumps in {args.dir_a}")
        return 2

    all_ok = True
    for fa in files_a:
        name = os.path.basename(fa)
        fb = os.path.join(args.dir_b, name)
        if not os.path.exists(fb):
            print(f"{name}: MISSING in dir-b")
            all_ok = False
            continue
        da = torch.load(fa, map_location="cpu")
        db = torch.load(fb, map_location="cpu")
        ff = _cmp(da["first_frame"], db["first_frame"])
        vf = _cmp(da["video_frames"], db["video_frames"])
        lat = _cmp(da["clean_full_latent"], db["clean_full_latent"])
        ok = ff == "equal" and vf == "equal"
        all_ok = all_ok and ok
        print(f"{name}: first_frame={ff} | video_frames={vf} | latent={lat}")

    print()
    if all_ok:
        print("[PASS] data pipeline is bit-deterministic across processes.")
        print("       -> offline precompute will get identical VAE inputs per anchor.")
        return 0
    print("[WARN] VAE inputs differ across runs -> decode/transform is not")
    print("       cross-process deterministic; offline cache keys must pin the")
    print("       exact decoded frames (store inputs, not just anchor ids).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
