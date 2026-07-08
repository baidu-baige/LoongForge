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

"""Offline VAE-latent precompute (SKELETON).

Precomputes `clean_full_latent` for every valid (episode, condition_frame)
anchor of the Motus dataset and writes it to disk, so online training can skip
the ~500ms/step frozen-VAE encode by loading the cached latent instead.

Parity contract (proven bit-identical to the online encode by
verify_offline_latents.py + diff_online_dumps.py, 2026-07-29):
  1. same input pixels     -> reuse the SAME dataset decode+transform per anchor
  2. same VAE weights+prec  -> build Wan2_2_VAE(vae_pth=model_cfg.vae_path) (fp32/autocast/TF32)
  3. same GPU arch + versions -> RUN THIS ON A800 with the training torch/cuDNN
  4. same batch SIZE = 8    -> encode in groups of EXACTLY 8 (composition is irrelevant;
                               only the size drives cuDNN algo/split-K/TF32 rounding).
                               The tail batch is padded to 8 and the pad outputs discarded.

Config comes from `parse_train_args()`, so launch this with the SAME CLI args
as training (same --model-name/--dataset-path/--per-device-batch-size 8/...).

Usage (single process; no torchrun needed -- enumerates the full anchor set):
  PYTHONPATH=$LOONGFORGE_PATH python examples/embodied/motus/precompute_latent_cache.py \
      --model-name motus \
      --dataset-format lerobot_datasets --dataset-strategy motus \
      --dataset-path /workspace/motus/data/aloha_mobile_cabinet \
      --video-backend torchcodec \
      --per-device-batch-size 8 \
      --cache-dir /workspace/motus/data/latent_cache/aloha_mobile_cabinet

STATUS: SKELETON. The enumerate/encode/store control flow is complete, but two
decisions are marked TODO and should be confirmed before a production run:
  (A) on-disk layout + dtype (default: one fp32 .pt per anchor; ~34GB for aloha).
  (B) how the dataset/trainer READ this cache (separate change points #2/#3; not here).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


# --- config batch size mandated by the parity contract (do not change) --------
PARITY_BATCH_SIZE = 8


def _parse_cli():
    """Split our own flags off argv, leaving the rest for parse_train_args()."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cache-dir", required=True,
                    help="output dir for the latent cache")
    ap.add_argument("--cache-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="fp32 = bit-identical to online (default); bf16 halves "
                         "storage but truncates -> NOT bit-identical.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit-anchors", type=int, default=0,
                    help="debug: cap number of anchors (0 = all).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the anchor set into this many contiguous shards "
                         "for multi-GPU concurrency (run one process per shard, "
                         "each with its own --shard-id + --device). Batch=8 is "
                         "preserved within each shard, so outputs stay bit-identical.")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="which shard [0, num-shards) this process encodes.")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-encode + overwrite even if the .pt already exists "
                         "(default: RESUME -- skip any batch whose outputs all "
                         "exist, avoiding the costly re-decode).")
    ours, rest = ap.parse_known_args()
    if not (0 <= ours.shard_id < ours.num_shards):
        ap.error(f"--shard-id must be in [0, {ours.num_shards}); got {ours.shard_id}")
    # Hand the remaining args to the framework parser (reads sys.argv).
    sys.argv = [sys.argv[0]] + rest
    return ours


def _enumerate_anchors(dataset):
    """Yield every valid (episode_id, condition_frame_idx, flat_idx) anchor.

    Mirrors index_samplers._motus_*: for each episode with relative bounds
    (from_, to_), the condition frame may range over
    [0, (to_-from_) - physical_chunk_size - 1]; the flat index fed to
    dataset.__getitem__ is from_ + condition_frame_idx.

    episode_id is the REAL episode index (dataset.episodes sorted ascending),
    matching the sorted-ascending accumulation used by _single_relative_bounds.
    """
    from loongforge.embodied.data.datasets.motus.motus_dataset import (
        _compute_relative_episode_bounds,
    )

    bounds = _compute_relative_episode_bounds(dataset)
    phys = int(dataset._physical_chunk_size)  # 48 = num_video_frames*vfr*gdr

    # Map bounds position -> real episode_id (sorted ascending, single-repo).
    # TODO(multi): for MultiLeRobotV3Dataset the episode_id namespace spans
    # sub-datasets; extend this mapping before running task_mode="multi".
    if dataset.episodes is not None:
        episode_ids = sorted(int(e) for e in dataset.episodes)
    else:
        episode_ids = list(range(len(bounds)))

    for pos, (from_, to_) in enumerate(bounds):
        ep_id = episode_ids[pos] if pos < len(episode_ids) else pos
        max_cf = (to_ - from_) - phys - 1
        for cf in range(0, max_cf + 1):
            yield ep_id, cf, from_ + cf


def _fetch_sample(dataset, flat_idx):
    """Fetch ONE anchor's per-sample dict by feeding the raw flat index.

    _index_map_fn is disabled by the caller so `dataset[flat_idx]` runs
    super().__getitem__(flat_idx) + transform directly -> same first_frame /
    video_frames the online random sampler would have produced for that anchor.
    """
    s = dataset[flat_idx]
    return s["first_frame"], s["video_frames"]  # [C,H,W], [F,C,H,W]


def _encode_batch(vae, first_frames, video_frames):
    """Replay motus.encode_video_latents EXACTLY at batch=len(first_frames).

    first_frames : [B, C, H, W]   video_frames : [B, F, C, H, W]
    Returns clean_full_latent [B, 48, T', H', W'].
    """
    first_frame_norm = (first_frames * 2.0 - 1.0).unsqueeze(2)             # [B,C,1,H,W]
    video_normalized = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)   # [B,C,F,H,W]
    full_video = torch.cat([first_frame_norm, video_normalized], dim=2)    # [B,C,F+1,H,W]
    with torch.no_grad():
        return vae.encode(full_video.to(vae.dtype))                        # autocast(fp32) inside


def _store_latent(cache_dir, flat_idx, latent, cache_dtype):
    """Write one anchor's latent, keyed by the flat frame index.

    Layout: cache_dir/{flat_idx:08d}.pt holding just the latent tensor. The key
    is the exact value the online index sampler returns (from_idx +
    condition_frame_idx), so the training read hook (_make_latent_cache_fn) loads
    it back by the same flat index. fp32 keeps bit-parity (~276KB/anchor, ~34GB
    total for aloha_mobile_cabinet); bf16 halves it but is NOT bit-identical.
    """
    out = latent.detach().to("cpu")
    if cache_dtype == "bf16":
        out = out.to(torch.bfloat16)
    path = os.path.join(cache_dir, f"{int(flat_idx):08d}.pt")
    torch.save(out, path)


def main() -> int:
    ours = _parse_cli()

    # Match the training process's cuDNN flags (train/utils/utils.py:63-64) --
    # these + A800 + batch=8 are what make the TF32 conv bit-reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from loongforge.embodied.train.parser import parse_train_args
    from loongforge.embodied.data.datasets.motus.motus_dataset import (
        build_motus_lerobot_dataset,
    )
    from loongforge.embodied.data.datasets.transforms.pipeline import (
        build_transforms_from_args,
    )
    from loongforge.embodied.model.motus.motus_impl.wan.modules.vae2_2 import (
        Wan2_2_VAE,
    )

    training_args, model_cfg, data_cfg = parse_train_args()

    # 1. Build the SAME dataset (same decode/transform/geometry as training).
    dataset = build_motus_lerobot_dataset(model_cfg, data_cfg, training_args)
    # build_motus_lerobot_dataset does NOT attach the per-sample transform; the
    # dataloader normally does that (dataloader.py:93/119). Replicate it here so
    # dataset[flat_idx] yields the assembled first_frame/video_frames dict (the
    # motus builder ignores dataset_stats -> None is fine; it loads stat.json).
    transform = build_transforms_from_args(
        model_cfg, data_cfg, training_args, dataset, dataset_stats=None
    )
    dataset._transform = transform
    # Disable the random index-map so raw flat indices address anchors directly.
    dataset._index_map_fn = None

    # 2. Build the SAME frozen VAE (same checkpoint as the online model).
    if not model_cfg.vae_path:
        print("[FAIL] model_cfg.vae_path is empty; cannot build VAE.")
        return 2
    print(f"[info] building frozen Wan2.2 VAE from {model_cfg.vae_path}")
    vae = Wan2_2_VAE(vae_pth=model_cfg.vae_path, device=ours.device)  # fp32 default
    print(f"[info] vae.dtype={vae.dtype}  cache_dtype={ours.cache_dtype}")

    os.makedirs(ours.cache_dir, exist_ok=True)

    # 3. Enumerate all anchors, group into EXACT batches of 8, then take this
    #    shard's contiguous slice of batches. Grouping happens on the FULL list
    #    first, so every batch has the same 8-anchor composition a single full
    #    run would produce -> per-anchor latents are bit-identical regardless of
    #    --num-shards (the only thing that matters for parity is batch SIZE=8).
    anchors = list(_enumerate_anchors(dataset))
    if ours.limit_anchors > 0:
        anchors = anchors[: ours.limit_anchors]
    total = len(anchors)
    all_batches = [
        anchors[i : i + PARITY_BATCH_SIZE]
        for i in range(0, total, PARITY_BATCH_SIZE)
    ]
    n_batches = len(all_batches)

    # Contiguous shard split over BATCHES (keeps each batch intact = parity-safe).
    per_shard = (n_batches + ours.num_shards - 1) // ours.num_shards
    b_lo = ours.shard_id * per_shard
    b_hi = min(b_lo + per_shard, n_batches)
    my_batches = list(range(b_lo, b_hi))
    my_anchor_count = sum(len(all_batches[bi]) for bi in my_batches)
    print(
        f"[info] {total} anchors / {n_batches} batches total; "
        f"shard {ours.shard_id}/{ours.num_shards} owns batches [{b_lo},{b_hi}) "
        f"= {my_anchor_count} anchors  (resume={'off' if ours.overwrite else 'on'})"
    )

    done = 0
    skipped = 0
    for bi in my_batches:
        group = all_batches[bi]
        real = len(group)
        # RESUME: if every real output in this batch already exists, skip the
        # whole batch (avoids the expensive re-decode + re-encode). A partial
        # batch is re-encoded in full so the missing outputs are produced at
        # batch=8; overwriting the present ones is a bit-identical no-op.
        if not ours.overwrite and all(
            os.path.exists(os.path.join(ours.cache_dir, f"{int(flat):08d}.pt"))
            for _ep, _cf, flat in group
        ):
            skipped += real
            continue

        ffs, vfs = [], []
        for _ep, _cf, flat in group:
            ff, vf = _fetch_sample(dataset, flat)
            ffs.append(ff)
            vfs.append(vf)
        # Pad the tail batch to EXACTLY 8 so cuDNN sees batch=8 (parity req #4);
        # composition is irrelevant, so repeat the last sample. Pad outputs are
        # sliced off below.
        while len(ffs) < PARITY_BATCH_SIZE:
            ffs.append(ffs[-1])
            vfs.append(vfs[-1])

        first_frames = torch.stack(ffs).to(ours.device)
        video_frames = torch.stack(vfs).to(ours.device)
        latents = _encode_batch(vae, first_frames, video_frames)  # [8,48,T',H',W']

        for i in range(real):
            _ep_id, _cf_idx, flat_idx = group[i]
            _store_latent(ours.cache_dir, flat_idx, latents[i], ours.cache_dtype)
        done += real
        if (bi - b_lo) % 50 == 0:
            print(f"[info] shard {ours.shard_id}: {done} encoded / {skipped} skipped "
                  f"/ {my_anchor_count} owned")

    print(f"[DONE] shard {ours.shard_id}/{ours.num_shards}: wrote {done} + skipped "
          f"{skipped} = {done + skipped}/{my_anchor_count} latents to {ours.cache_dir}")
    # TODO(A): if a manifest/index is desired (anchor -> path, geometry, dtype),
    #          emit it here so the dataset read-path (change point #2) can load
    #          without re-deriving keys.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
