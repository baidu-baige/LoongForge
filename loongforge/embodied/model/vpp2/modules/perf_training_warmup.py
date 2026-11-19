# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare fixed LIBERO graphs before the first measured training iteration."""
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def warmup(model, args):
    cfg = model.config
    if (cfg.training_stage != "action" or cfg.perf_action_execution != "native_graph"
            or int(args.per_device_batch_size) != 2 or cfg.action_horizon != 32
            or cfg.proprio_dim != 8 or cfg.dtype != "bfloat16"):
        raise ValueError("Startup warmup supports the validated LIBERO B2/action32/BF16 graph recipe")
    device = torch.device("cuda", torch.cuda.current_device())
    start = time.time()
    logger.info("VPP2 startup warmup begins; no data is consumed and no optimizer update is performed")
    with torch.random.fork_rng(devices=[device.index]):
        rgb = torch.zeros(2, 3, 1, 224, 448, device=device, dtype=torch.bfloat16)
        # Stabilize encoder capture/compile states, including the per-sample
        # VAE loop. Original encoders use the same preparation when disabled.
        with torch.no_grad():
            for _ in range(3):
                condition = model._encode_video_latents(rgb)
                clip = model._encode_clip_fea_from_video(rgb)
        context = torch.zeros(2, 129, 4096, device=device, dtype=torch.bfloat16,
                              requires_grad=True)
        context_mask = torch.ones(2, 129, device=device, dtype=torch.bool)
        timestep = torch.zeros(2, device=device, dtype=torch.bfloat16)
        latents = torch.zeros(2, 16, 5, 28, 56, device=device, dtype=torch.bfloat16)
        cache, mask, video_length = model._build_frozen_video_cache(
            latents=latents,
            timestep_video=torch.full_like(timestep, model.train_video_scheduler.num_train_timesteps),
            context=context, context_mask=context_mask, condition_latents=condition,
            clip_fea=clip, action_seq_len=32, fuse_vae_embedding_in_latents=False,
        )
        from .perf_action_graph import replay_full_action

        prediction = replay_full_action(
            model.mot, model.action_expert,
            torch.zeros(2, 32, 7, device=device, dtype=torch.bfloat16),
            timestep, context, context_mask, cache, mask,
            visible_video_tokens=(min(video_length, cfg.action_visible_video_frames * 392)
                                  if (
                                      cfg.perf_action_graph_visible_kv
                                      or cfg.perf_action_visible_kv_sdpa
                                  ) else None),
            exact_visible_kv=bool(
                cfg.perf_action_visible_kv_sdpa
                and not cfg.perf_action_graph_visible_kv
            ),
        )
        torch.cuda.synchronize()
        del prediction, cache, mask, latents, context, context_mask, condition, clip, rgb, timestep
    # Release the unused live autograd edges before actual training. The
    # captured compute remains resident; no backward/optimizer call was made.
    gc.collect()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    record = dict(start_unix=start, end_unix=time.time(), data_batches_consumed=0,
                  optimizer_updates=0, rng_restored=True, video_prefills=1,
                  action_graph_replays=1, encoder_batches=3,
                  shape_scope="LIBERO RGB224x448, latent5x28x56, action32, B2")
    root = os.environ.get("OUTPUT_DIR")
    if root:
        (Path(root) / ("startup_warmup_rank" + os.environ.get("RANK", "0") + ".json")).write_text(
            json.dumps(record, indent=2) + "\n")
    logger.info("VPP2 startup warmup completed seconds=%.4f; training step1 timer starts afterwards",
                record["end_unix"] - start)
