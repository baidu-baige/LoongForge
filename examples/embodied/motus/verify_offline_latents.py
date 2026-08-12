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

"""Offline VAE-latent parity check (COMPUTE link).

Loads the batches dumped by the ONLINE training run (LATENT_DUMP=1 hook in
MotusTrainer._prefetch_batch), rebuilds ONLY the frozen Wan2.2 VAE in a FRESH
process, replays the EXACT `encode_video_latents` path (same normalization +
`vae.encode`) at the SAME batch size, and bit-compares the recomputed latent
against the dumped online latent.

This isolates the risky link: "same input bytes -> same latent bytes, in a
different process (fresh cuDNN context / algo selection)". A `torch.equal` pass
proves the offline latent cache would be bit-identical to the online encode
(given: same GPU arch, same torch/cuDNN version, same batch size, same flags).

It does NOT test the data link (same anchor -> same input pixels); use
diff_online_dumps.py across two online runs for that.

Usage
-----
  PYTHONPATH=$LOONGFORGE_PATH python examples/embodied/motus/verify_offline_latents.py \
      --dump-dir /tmp/latent_dump \
      --vae-path /workspace/motus/models/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth

Run this on the SAME GPU type (A800) and SAME torch/cuDNN version as training.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch


def _add_loongforge_to_path() -> None:
    lf = os.environ.get(
        "LOONGFORGE_PATH",
        "/workspace/AIAK-Training-Omni",
    )
    if lf not in sys.path:
        sys.path.insert(0, lf)


def _encode_like_online(vae, first_frame, video_frames):
    """Replay motus.py:encode_video_latents EXACTLY (normalization + vae.encode).

    first_frame  : [B, C, H, W]      (compute dtype, e.g. bf16 -- as dumped online)
    video_frames : [B, F, C, H, W]   (compute dtype)
    Returns clean_full_latent [B, 48, T', H', W'].
    """
    first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)            # [B,C,1,H,W]
    video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)  # [B,C,F,H,W]
    full_video = torch.cat([first_frame_norm, video_normalized], dim=2)   # [B,C,F+1,H,W]
    with torch.no_grad():
        # vae.encode() wraps the model.encode in autocast(fp32); vae.dtype == fp32.
        return vae.encode(full_video.to(vae.dtype))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/tmp/latent_dump")
    ap.add_argument(
        "--vae-path",
        default="/workspace/motus/models/hf/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--rank", type=int, default=0,
        help="which online rank's dumps to verify (default 0).",
    )
    args = ap.parse_args()

    _add_loongforge_to_path()
    from loongforge.embodied.model.motus.motus_impl.wan.modules.vae2_2 import Wan2_2_VAE

    # Mirror the training process's cuDNN flags (train/utils/utils.py:63-64).
    # These + same GPU arch + same batch size are what make the TF32 conv
    # bit-reproducible across processes.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    files = sorted(glob.glob(os.path.join(args.dump_dir, f"rank{args.rank}_step*.pt")))
    if not files:
        print(f"[FAIL] no dump files matching rank{args.rank}_step*.pt in {args.dump_dir}")
        return 2

    print(f"[info] building frozen Wan2.2 VAE from {args.vae_path} on {args.device}")
    vae = Wan2_2_VAE(vae_pth=args.vae_path, device=args.device)  # dtype defaults to fp32
    print(f"[info] vae.dtype={vae.dtype}  |  verifying {len(files)} dumped batch(es)\n")

    all_ok = True
    for f in files:
        d = torch.load(f, map_location="cpu")
        ff = d["first_frame"].to(args.device)
        vf = d["video_frames"].to(args.device)
        online = d["clean_full_latent"].to(args.device)
        bsz = d.get("batch_size", ff.shape[0])

        offline = _encode_like_online(vae, ff, vf)

        equal = torch.equal(offline, online)
        diff = (offline.float() - online.float()).abs()
        max_abs = diff.max().item()
        denom = online.float().abs().clamp_min(1e-12)
        max_rel = (diff / denom).max().item()
        # per-sample bit-equality (isolates any single-sample divergence)
        per_sample = [
            bool(torch.equal(offline[i], online[i])) for i in range(offline.shape[0])
        ]

        status = "BIT-IDENTICAL" if equal else "DIFFERS"
        all_ok = all_ok and equal
        print(
            f"{os.path.basename(f)}: {status}  batch={bsz}  "
            f"latent={tuple(online.shape)}  compute_dtype={d.get('compute_dtype')}"
        )
        print(f"    max|abs|={max_abs:.3e}  max|rel|={max_rel:.3e}  "
              f"per_sample_equal={per_sample}")

    print()
    if all_ok:
        print("[PASS] offline recompute is BIT-IDENTICAL to online for all batches.")
        print("       -> offline latent cache is safe for step-1 bit-parity tracking.")
        return 0
    print("[WARN] offline latent differs at the TF32 rounding level.")
    print("       If max|abs| ~1e-6 and per-sample only, it is 'parity within base's")
    print("       run-to-run band' (video-loss spread ~8e-3), NOT bit-identical.")
    print("       Check: same GPU arch, same torch/cuDNN version, same batch size (8).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
