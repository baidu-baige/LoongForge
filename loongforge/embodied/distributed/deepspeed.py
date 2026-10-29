# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Incremental DeepSpeed integration for embodied trainers.

The helpers in this module keep DeepSpeed-specific engine, optimizer, and
checkpoint behavior out of the existing DDP/FSDP implementation.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

_DEEPSPEED_META_FILE = "resume_meta.json"
_DEEPSPEED_STATE_DIR = "deepspeed"


def is_deepspeed_engine(model) -> bool:
    """Return whether *model* is a DeepSpeed engine without importing eagerly."""
    return (
        model is not None
        and type(model).__name__ == "DeepSpeedEngine"
        and type(model).__module__.startswith("deepspeed.")
    )


def load_deepspeed_config(path: str, training_args, world_size: int) -> dict:
    """Load the JSON and resolve batch/precision fields like Accelerate."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    config = deepcopy(config)

    micro_batch = int(training_args.per_device_batch_size)
    gradient_accumulation = int(training_args.gradient_accumulation_steps)
    global_batch = micro_batch * gradient_accumulation * int(world_size)
    resolved = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "train_batch_size": global_batch,
    }
    for key, expected in resolved.items():
        current = config.get(key, "auto")
        if current == "auto":
            config[key] = expected
        elif int(current) != expected:
            raise ValueError(
                f"DeepSpeed {key}={current} does not match the effective embodied "
                f"value {expected}."
            )

    clipping = config.get("gradient_clipping", 0.0)
    requested_clip = float(training_args.clip_grad)
    if clipping == "auto":
        config["gradient_clipping"] = requested_clip
    elif float(clipping) != requested_clip:
        raise ValueError(
            "--clip-grad must match DeepSpeed gradient_clipping "
            f"(CLI={requested_clip}, JSON={clipping}; absent JSON means 0)."
        )
    dtype = str(training_args.dtype).lower()
    config.setdefault("bf16", {})["enabled"] = dtype in {"bf16", "bfloat16"}
    config.setdefault("fp16", {})["enabled"] = dtype in {"fp16", "float16"}
    config.setdefault("steps_per_print", float("inf"))
    # A client torch optimizer is intentional and matches the VPP2 reference.
    config.setdefault("zero_allow_untested_optimizer", True)
    return config


def initialize_deepspeed(
    model, optimizer, scheduler, training_args, ctx
):
    """Create the engine around an eagerly loaded model and client optimizer."""
    try:
        import deepspeed
    except ImportError as exc:  # pragma: no cover - dependency is in project extras
        raise ImportError(
            "--distributed-strategy=deepspeed requires the deepspeed package."
        ) from exc

    # This backend was validated with 0.18.9. Keep its conservative version
    # guard local; do not upgrade the entire project's dependency for VPP2.
    if Version(deepspeed.__version__) < Version("0.18.9"):
        raise RuntimeError(
            "Embodied DeepSpeed requires deepspeed>=0.18.9 for correct ZeRO-2 "
            "gradient accumulation; older versions can discard earlier "
            f"microbatch gradients (installed: {deepspeed.__version__})."
        )

    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError(
            "DeepSpeed requires the embodied model to be fully materialized before "
            "engine initialization."
        )
    config = load_deepspeed_config(
        training_args.deepspeed_config, training_args, ctx.world_size
    )
    engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config,
        dist_init_required=False,
    )
    if ctx.is_main:
        with (Path(training_args.output_dir) / "effective_deepspeed.json").open("w", encoding="utf-8") as handle:
            json.dump(engine._config._param_dict, handle, indent=2)
        logger.info(
            "DeepSpeed initialized: stage=%s world=%d micro_batch=%d "
            "gradient_accumulation=%d global_batch=%d bf16=%s",
            config["zero_optimization"]["stage"],
            ctx.world_size,
            config["train_micro_batch_size_per_gpu"],
            config["gradient_accumulation_steps"],
            config["train_batch_size"],
            config["bf16"]["enabled"],
        )
    return engine, ds_optimizer, ds_scheduler, config


def _rank_aux_path(path: str, kind: str, rank: int) -> str:
    return os.path.join(path, kind, f"rank_{rank}.pt")


def save_deepspeed_checkpoint(
    engine,
    *,
    checkpoint_dir: str,
    step: int,
    epoch: int,
    dataloader_state: dict,
    ctx,
) -> str:
    """Save engine/model/optimizer/scheduler plus exact rank-local state."""
    from loongforge.embodied.distributed.checkpoint import capture_rank_rng_state

    path = os.path.join(checkpoint_dir, f"steps_{int(step)}")
    state_dir = os.path.join(path, _DEEPSPEED_STATE_DIR)
    if ctx.is_main:
        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(os.path.join(path, "rng"), exist_ok=True)
        os.makedirs(os.path.join(path, "dataloader"), exist_ok=True)
    ctx.barrier()

    torch.save(
        capture_rank_rng_state(ctx),
        _rank_aux_path(path, "rng", ctx.rank),
    )
    torch.save(
        dataloader_state or {},
        _rank_aux_path(path, "dataloader", ctx.rank),
    )
    engine.save_checkpoint(
        state_dir,
        tag=f"global_step{int(step)}",
        client_state={"completed_steps": int(step), "epoch": int(epoch)},
        save_latest=True,
    )
    ctx.barrier()
    if ctx.is_main:
        meta_path = os.path.join(path, _DEEPSPEED_META_FILE)
        temp_meta_path = f"{meta_path}.tmp"
        with open(temp_meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "completed_steps": int(step),
                    "epoch": int(epoch),
                    "ckpt_format": "deepspeed",
                    "world_size": int(ctx.world_size),
                },
                handle,
                indent=2,
            )
        os.replace(temp_meta_path, meta_path)
    ctx.barrier()
    return path


def resume_deepspeed_checkpoint(engine, checkpoint_path: str, ctx):
    """Strictly restore engine and return epoch/dataloader/RNG state."""
    from loongforge.embodied.distributed.checkpoint import gather_rng_for_deferred_restore

    meta_path = os.path.join(checkpoint_path, _DEEPSPEED_META_FILE)
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    saved_world = int(meta.get("world_size", ctx.world_size))
    if saved_world != ctx.world_size:
        raise RuntimeError(
            "DeepSpeed resume requires the original world size: "
            f"checkpoint={saved_world}, current={ctx.world_size}."
        )

    state_dir = os.path.join(checkpoint_path, _DEEPSPEED_STATE_DIR)
    tag = f"global_step{int(meta['completed_steps'])}"
    load_path, client_state = engine.load_checkpoint(
        state_dir,
        tag=tag,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )
    if load_path is None:
        raise RuntimeError(f"DeepSpeed failed to load checkpoint {state_dir}/{tag}")

    rng_path = _rank_aux_path(checkpoint_path, "rng", ctx.rank)
    dataloader_path = _rank_aux_path(checkpoint_path, "dataloader", ctx.rank)
    local_rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    dataloader_state = torch.load(
        dataloader_path, map_location="cpu", weights_only=False
    )
    rng_per_rank = gather_rng_for_deferred_restore(local_rng, ctx)
    saved_epoch = int((client_state or {}).get("epoch", meta.get("epoch", 0)))
    return saved_epoch, dataloader_state or {}, rng_per_rank


def get_latest_deepspeed_checkpoint(checkpoint_dir: str):
    """Return the newest resumable LoongForge DeepSpeed checkpoint."""
    if not os.path.isdir(checkpoint_dir):
        return None, 0, 0
    candidates = [
        (int(name[6:]), name)
        for name in os.listdir(checkpoint_dir)
        if name.startswith("steps_") and name[6:].isdigit()
    ]
    if not candidates:
        return None, 0, 0
    step, name = max(candidates)
    path = os.path.join(checkpoint_dir, name)
    meta_path = os.path.join(path, _DEEPSPEED_META_FILE)
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("ckpt_format") != "deepspeed" or int(meta["completed_steps"]) != step:
        raise ValueError(f"Invalid DeepSpeed checkpoint metadata: {meta_path}")
    state_dir = os.path.join(path, _DEEPSPEED_STATE_DIR)
    tag = os.path.join(state_dir, f"global_step{step}")
    required = [Path(tag), *[
        Path(_rank_aux_path(path, kind, rank))
        for kind in ("rng", "dataloader")
        for rank in range(int(meta["world_size"]))
    ]]
    if not Path(tag).is_dir() or not any(Path(tag).glob("*model_states.pt")):
        raise FileNotFoundError(f"Incomplete DeepSpeed model state: {tag}")
    if not any(Path(tag).glob("*optim_states.pt")):
        raise FileNotFoundError(f"Incomplete DeepSpeed optimizer state: {tag}")
    for item in required:
        if not item.exists():
            raise FileNotFoundError(f"Incomplete DeepSpeed checkpoint: {item}")
    return path, step, int(meta.get("epoch", 0))
