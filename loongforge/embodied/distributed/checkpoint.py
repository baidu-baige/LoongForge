# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Distributed checkpoint save/load/resume.

Supports three formats selected via ``--save-format``:

* ``safetensors`` (default, legacy) — rank0 consolidates full state dict and writes
  ``model.safetensors`` + ``training_state.pt``.
* ``pt`` (legacy) — same as above but writes ``pytorch_model.pt``.
* ``dcp`` — every rank writes its own shard via
  ``torch.distributed.checkpoint`` into ``dcp/``. Avoids rank0 OOM and supports
  model resharding when resuming on a different world size. Rank-local ZeRO
  optimizer state requires the original world size.

``resume_training_state`` auto-detects the format on disk so old checkpoints
keep working unchanged.
"""

import gc
import json
import logging
import os
import random
import shutil
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FSDPModule
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor

from .context import DistributedContext
from .utils import unwrap_model

logger = logging.getLogger(__name__)

# Canonical file names for each checkpoint format.
# Update these constants when adding new formats rather than editing each call site.
_DCP_METADATA_FILE = "dcp/.metadata"  # relative to checkpoint step dir
_SAFETENSORS_FILE = "model.safetensors"
_PT_FILE = "pytorch_model.pt"
_ADAPTER_FILE = "adapter_model.safetensors"
_ADAPTER_CONFIG_FILE = "adapter_config.json"
_ADAPTER_META_FILE = "adapter_meta.json"
_ZERO_OPTIMIZER_DIR = "zero_optimizer"
_ZERO_OPTIMIZER_METADATA_FILE = f"{_ZERO_OPTIMIZER_DIR}/metadata.json"
_VALID_FORMATS = ("safetensors", "pt", "dcp")

# Module-level state for async DCP save. At most one in-flight save at a time;
# subsequent saves (or shutdown) wait on the previous future before launching a
# new one to avoid contending CPU-staging buffers and to prevent
# ``get_latest_checkpoint`` from picking up an incomplete directory.
_pending_async_save: Optional[dict] = None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_dir: str,
    ctx: DistributedContext,
    training_args,
    epoch: int = 0,
    dataloader_state: Optional[Dict] = None,
    model_cfg=None,
):
    """Save a full checkpoint or a consolidated LoRA adapter checkpoint."""
    # Make sure any previous async save has finalized before we touch a new
    # ``steps_{N}`` dir or stage another state-dict snapshot.
    flush_pending_save(ctx)

    path = os.path.join(checkpoint_dir, f"steps_{step}")
    use_lora = bool(training_args.use_lora)
    save_format = "safetensors" if use_lora else training_args.save_format
    async_save = (
        save_format == "dcp"
        and training_args.async_save
        and hasattr(dcp, "async_save")
    )
    if training_args.async_save and not async_save and ctx.is_main:
        if use_lora:
            logger.warning(
                "--async-save ignored: LoRA checkpoints use adapter safetensors."
            )
        elif save_format != "dcp":
            logger.warning("--async-save ignored: only effective with --save-format=dcp.")
        elif not hasattr(dcp, "async_save"):
            logger.warning(
                "--async-save ignored: torch.distributed.checkpoint.async_save "
                "unavailable (requires PyTorch >= 2.4)."
            )

    if ctx.is_main:
        os.makedirs(path, exist_ok=True)
    ctx.barrier()

    meta = {
        "completed_steps": step,
        "epoch": epoch,
        "ckpt_format": save_format,
        "world_size": ctx.world_size,
        "use_lora": use_lora,
    }

    if save_format == "dcp":
        future = _save_dcp(
            model, optimizer, scheduler, path, ctx, training_args,
            dataloader_state, async_save=async_save,
        )
    else:
        future = None
        _save_legacy(
            model, optimizer, scheduler, path, ctx, training_args, epoch,
            dataloader_state, save_format, model_cfg=model_cfg,
        )

    if future is not None:
        # Defer ``resume_meta.json`` until the async write has flushed —
        # otherwise a crash mid-write leaves a "valid-looking" dir.
        global _pending_async_save
        _pending_async_save = {
            "future": future,
            "path": path,
            "meta": meta,
            "is_main": ctx.is_main,
            "rank": ctx.rank,
            "save_format": save_format,
        }
        if ctx.is_main:
            logger.info(f"Async checkpoint launched ({save_format}): {path}")
        # No barrier here: ranks must be free to keep training. Cross-rank
        # synchronization happens inside the next flush via ctx.barrier().
        return

    if ctx.is_main:
        _write_resume_meta(path, meta)
        logger.info(f"Checkpoint saved ({save_format}): {path}")

    ctx.barrier()


def flush_pending_save(ctx: Optional[DistributedContext] = None):
    """Wait on any in-flight async DCP save and finalize its ``resume_meta.json``.

    Safe to call repeatedly. Should be invoked before the process exits and
    before launching another save (the latter is handled internally by
    ``save_checkpoint``).
    """
    global _pending_async_save
    if _pending_async_save is None:
        return
    pending = _pending_async_save
    _pending_async_save = None

    local_ok = True
    local_err: Optional[Exception] = None
    try:
        pending["future"].result()
        if pending["is_main"]:
            _write_resume_meta(pending["path"], pending["meta"])
    except Exception as e:  # noqa: BLE001 - propagated below after global vote
        local_ok = False
        local_err = e
        logger.exception(
            "Async DCP save failed on rank %d for %s",
            pending["rank"], pending["path"],
        )

    global_ok = local_ok
    if ctx is not None and ctx.is_distributed:
        flag = torch.tensor(
            [1 if local_ok else 0], dtype=torch.int, device=ctx.device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        global_ok = bool(flag.item() == 1)

    if not global_ok:
        if pending["is_main"]:
            # Remove the partially-written dir so failed shards don't accumulate on
            # disk. Only rank0 deletes, and only after the global vote — so no rank
            # is still writing into it. Best-effort: a cleanup failure must not mask
            # the original save error raised below.
            try:
                shutil.rmtree(pending["path"])
                logger.info("Removed failed checkpoint dir: %s", pending["path"])
            except Exception:
                logger.exception(
                    "Failed to remove incomplete checkpoint dir %s", pending["path"]
                )
        if ctx is not None:
            ctx.barrier()

        raise RuntimeError(
            f"Aborting training because async checkpoint commit failed for "
            f"{pending['path']} (see per-rank logs for the original save error)."
        ) from local_err     

    if pending["is_main"]:
        logger.info(
            f"Async checkpoint finalized ({pending['save_format']}): {pending['path']}"
        )
        

def _write_resume_meta(path: str, meta: dict):
    with open(os.path.join(path, "resume_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_pretrained(model: nn.Module, checkpoint_path: str, ctx: DistributedContext) -> nn.Module:
    """Load pretrained weights (call BEFORE DDP/FSDP wrapping).

    Only consumes consolidated single-file weights (``model.safetensors`` /
    ``pytorch_model.pt``). To use a DCP checkpoint as pretrained source, run
    ``tools/dcp_to_safetensors.py`` first to produce a single-file version.
    """
    if not checkpoint_path:
        return model

    resolved = _resolve_file(checkpoint_path)
    sd = _load_sd(resolved)

    # Filter out shape mismatches
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape != v.shape:
            skipped.append(k)
        else:
            filtered[k] = v

    if skipped and ctx.is_main:
        logger.warning(f"Skipped {len(skipped)} shape-mismatched keys")
        for k in skipped[:5]:
            logger.warning(f"  {k}")

    model.load_state_dict(filtered, strict=False)
    if ctx.is_main:
        logger.info(f"Loaded pretrained: {checkpoint_path}")
    return model


def get_latest_checkpoint(
    checkpoint_dir: str, require_training_state: bool = True
) -> Tuple[Optional[str], int, int]:
    """Find the latest *resumable* checkpoint directory.

    Validates the latest ``steps_N`` only — no fallback to older dirs. A
    checkpoint is considered resumable if either:
      * DCP optimizer state exists in ``dcp/``,
      * rank-local ZeRO optimizer state exists in ``zero_optimizer/``, or
      * ``training_state.pt`` exists (legacy format)
    plus ``resume_meta.json``.
    """
    if not os.path.isdir(checkpoint_dir):
        return None, 0, 0

    steps = []
    for d in os.listdir(checkpoint_dir):
        if d.startswith("steps_") and d[len("steps_"):].isdigit():
            steps.append((int(d[len("steps_"):]), d))
    if not steps:
        return None, 0, 0
    _, latest_d = max(steps)

    ckpt_dir = os.path.join(checkpoint_dir, latest_d)
    meta_path = os.path.join(ckpt_dir, "resume_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"resume_meta.json not found in {ckpt_dir}")
    dcp_dir = os.path.join(ckpt_dir, "dcp")
    has_dcp_training_state = (
        os.path.exists(os.path.join(ckpt_dir, _DCP_METADATA_FILE))
        and _dcp_has_key(dcp_dir, "optim")
    )
    has_zero_training_state = os.path.exists(
        os.path.join(ckpt_dir, _ZERO_OPTIMIZER_METADATA_FILE)
    )
    has_training_state = os.path.exists(os.path.join(ckpt_dir, "training_state.pt"))
    if require_training_state and not (
        has_dcp_training_state or has_zero_training_state or has_training_state
    ):
        raise FileNotFoundError(
            f"No resumable training state in {ckpt_dir}: expected optimizer state "
            f"in dcp/, {_ZERO_OPTIMIZER_DIR}/, or training_state.pt. "
            f"Re-save with --save-training-state."
        )

    with open(meta_path) as f:
        meta = json.load(f)
    return ckpt_dir, int(meta["completed_steps"]), int(meta.get("epoch", 0))


def resume_training_state(
    model,
    optimizer,
    scheduler,
    checkpoint_path,
    ctx,
    restore_rng: bool = True,
) -> Tuple[Optional[int], Dict, Optional[list]]:
    """Restore optimizer/scheduler/RNG state. Auto-detects ckpt format.

    Returns ``(saved_epoch, dataloader_state, rng_state_per_rank)``. When
    ``restore_rng`` is True, RNG is restored in-place and ``rng_state_per_rank``
    is None; when False, the caller is responsible for invoking
    ``restore_rank_rng_state`` later (e.g., after dataloader iter init).
    """
    fmt = detect_checkpoint_format(checkpoint_path)
    if fmt == "dcp":
        return _resume_dcp(
            model, optimizer, scheduler, checkpoint_path, ctx, restore_rng,
        )
    return _resume_legacy(
        model, optimizer, scheduler, checkpoint_path, ctx, restore_rng,
    )


def restore_rank_rng_state(rng_per_rank, ctx: DistributedContext, source: str = "checkpoint"):
    """Validate per-rank RNG payload and restore this rank's stream."""
    if rng_per_rank is None:
        raise KeyError(
            f"RNG state not present in {source} (older checkpoint format). "
            f"Re-save with --save-training-state."
        )
    if len(rng_per_rank) != ctx.world_size:
        raise RuntimeError(
            f"RNG state was saved for world_size={len(rng_per_rank)} but current "
            f"world_size={ctx.world_size}."
        )
    _set_rank_rng(rng_per_rank[ctx.rank], ctx)


def detect_checkpoint_format(path: str) -> str:
    """Identify checkpoint format by metadata, falling back to directory sniff.

    Returns one of ``"safetensors"``, ``"pt"``, ``"dcp"``.
    """
    meta_path = os.path.join(path, "resume_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            fmt = json.load(f).get("ckpt_format")
        if fmt in _VALID_FORMATS:
            return fmt
    if os.path.exists(os.path.join(path, _DCP_METADATA_FILE)):
        return "dcp"
    if os.path.exists(os.path.join(path, _SAFETENSORS_FILE)):
        return "safetensors"
    if os.path.exists(os.path.join(path, _PT_FILE)):
        return "pt"
    raise FileNotFoundError(f"No recognizable checkpoint in {path}")


def is_lora_adapter_checkpoint(path: str) -> bool:
    """Return whether a checkpoint directory contains a LoRA adapter."""
    return os.path.isfile(adapter_file_path(path)) and os.path.isfile(
        os.path.join(path, _ADAPTER_CONFIG_FILE)
    )


def read_adapter_meta(path: str) -> Optional[dict]:
    """Read adapter metadata, returning ``None`` when it is absent."""
    meta_path = os.path.join(path, _ADAPTER_META_FILE)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as file:
        return json.load(file)


def adapter_file_path(path: str) -> str:
    """Return the canonical adapter safetensors path."""
    return os.path.join(path, _ADAPTER_FILE)


def _save_legacy(
    model, optimizer, scheduler, path, ctx, training_args, epoch,
    dataloader_state, save_format, model_cfg=None,
):
    """Save a rank0-consolidated full model or trainable LoRA tensors."""
    use_lora = bool(training_args.use_lora)
    state_dict = _get_full_state_dict(
        model,
        ctx,
        ignore_frozen_params=use_lora,
        cpu_offload=use_lora,
    )

    if ctx.is_main:
        if use_lora:
            from loongforge.embodied.train.lora import (
                get_adapter_state_dict,
                save_adapter_config,
            )

            adapter_state = get_adapter_state_dict(
                state_dict,
                unwrap_model(model),
            )
            _save_state_dict_safetensors(
                adapter_state,
                adapter_file_path(path),
                metadata={"format": "pt"},
            )
            save_adapter_config(unwrap_model(model), path)
            _write_adapter_meta(path, training_args, model_cfg)
            logger.info(
                "Saved LoRA adapter with %d tensors to %s",
                len(adapter_state),
                adapter_file_path(path),
            )
        elif save_format == "safetensors":
            torch.cuda.empty_cache()
            gc.collect()
            _save_state_dict_safetensors(state_dict, os.path.join(path, "model.safetensors"))
            gc.collect()
            torch.cuda.empty_cache()
        else:
            torch.save(state_dict, os.path.join(path, "pytorch_model.pt"))

    if training_args.save_training_state:
        _save_training_state(
            model, optimizer, scheduler, epoch, path, ctx, training_args,
            dataloader_state=dataloader_state,
        )


def _write_adapter_meta(path: str, training_args, model_cfg) -> None:
    """Persist LoongForge metadata alongside the standard PEFT adapter."""
    from loongforge.embodied.train.lora import adapter_meta

    with open(
        os.path.join(path, _ADAPTER_META_FILE),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(adapter_meta(training_args, model_cfg), file, indent=2)

    if model_cfg is None:
        return
    config_payload = _config_to_json_value(model_cfg)
    if not isinstance(config_payload, dict):
        return
    with open(
        os.path.join(path, "model_config.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config_payload, file, indent=2)


def _config_to_json_value(value):
    """Convert config values to JSON data without ``deepcopy`` side effects."""
    if is_dataclass(value):
        return {
            field.name: _config_to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _config_to_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_config_to_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _config_to_json_value(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _fill_missing_optimizer_state(optim_state_dict: dict) -> tuple[dict, int]:
    """Add empty state for parameters not updated before a legacy save."""
    state = optim_state_dict.get("state")
    param_groups = optim_state_dict.get("param_groups")
    if not isinstance(state, dict) or not isinstance(param_groups, list):
        return optim_state_dict, 0
    if state and all(isinstance(key, int) for key in state):
        return optim_state_dict, 0

    param_names = {
        param_name
        for param_group in param_groups
        for param_name in param_group.get("params", [])
    }
    missing = param_names.difference(state)
    if not missing:
        return optim_state_dict, 0

    patched_state = dict(state)
    patched_state.update({param_name: {} for param_name in missing})
    patched = dict(optim_state_dict)
    patched["state"] = patched_state
    return patched, len(missing)


def _resume_legacy(model, optimizer, scheduler, checkpoint_path, ctx, restore_rng):
    """Legacy resume — load training_state.pt and dispatch optim state via FSDP API."""
    state_file = os.path.join(checkpoint_path, "training_state.pt")
    if not os.path.exists(state_file):
        raise FileNotFoundError(
            f"No training_state.pt in {checkpoint_path}; cannot resume training. "
            f"Re-save the checkpoint with --save-training-state (default on)."
        )

    state = torch.load(state_file, map_location="cpu", weights_only=False)

    if isinstance(model, FSDPModule):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        optim_state, missing_state_count = _fill_missing_optimizer_state(
            state["optimizer"]
        )
        # ``get_state_dict`` returns the canonical state/param_groups mapping
        # for one optimizer. Wrapping it by optimizer index makes current DCP
        # interpret it as flattened state and fail during unflattening.
        set_state_dict(
            model, optimizers=[optimizer],
            model_state_dict={}, optim_state_dict=optim_state,
            options=options,
        )
        if missing_state_count and ctx.is_main:
            logger.info(
                "Initialized empty optimizer state for %d parameters absent "
                "from the checkpoint",
                missing_state_count,
            )
    else:
        optimizer.load_state_dict(state["optimizer"])
    if ctx.is_main:
        logger.info("optimizer resumed successfully")

    if "scheduler" in state and state["scheduler"] is not None and scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
        if ctx.is_main:
            logger.info("scheduler resumed successfully")

    rng_per_rank = None
    if restore_rng:
        restore_rank_rng_state(state.get("rng_state_per_rank"), ctx, source=state_file)
        if ctx.is_main:
            logger.info("RNG state resumed successfully")
    else:
        rng_per_rank = state.get("rng_state_per_rank")

    dataloader_state = {}
    dataloader_state_per_rank = state.get("dataloader_state_per_rank")
    if dataloader_state_per_rank is not None:
        if len(dataloader_state_per_rank) != ctx.world_size:
            raise RuntimeError(
                f"Dataloader state was saved for world_size={len(dataloader_state_per_rank)} "
                f"but current world_size={ctx.world_size}."
            )
        dataloader_state = dataloader_state_per_rank[ctx.rank] or {}

    saved_epoch = state.get("epoch")
    saved_epoch = int(saved_epoch) if saved_epoch is not None else None
    return saved_epoch, dataloader_state, rng_per_rank


def _save_dcp(
    model, optimizer, scheduler, path, ctx, training_args, dataloader_state,
    *, async_save: bool = False,
):
    """Sharded save: each rank writes its own DCP shard.

    When ``async_save`` is True, returns the future from ``dcp.async_save``;
    the caller is responsible for waiting on it before the next save or
    process exit (handled by ``save_checkpoint`` / ``flush_pending_save``).
    """
    save_training_state = training_args.save_training_state
    save_local_zero_state = save_training_state and _is_zero_optimizer(optimizer)
    optimizers = [optimizer] if save_training_state and not save_local_zero_state else []

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_sd, optim_sd = get_state_dict(
        model, optimizers=optimizers, options=options
    )
    state = {"model": model_sd}
    if optimizers:
        state["optim"] = optim_sd

    dcp_dir = os.path.join(path, "dcp")
    if ctx.is_main:
        os.makedirs(dcp_dir, exist_ok=True)
    ctx.barrier()

    # Aux files (scheduler / per-rank RNG / per-rank dataloader) are tiny and
    # written synchronously regardless of ``async_save`` — they must be on
    # disk before we return, since training will mutate the RNG and
    # dataloader state immediately after.
    if save_training_state:
        if ctx.is_main and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        _save_aux_per_rank(path, ctx, dataloader_state)
        if save_local_zero_state:
            _save_zero_optimizer_state(path, optimizer, ctx)

    writer = dcp.FileSystemWriter(dcp_dir)
    if async_save:
        # ``dcp.async_save`` stages a CPU snapshot of ``state`` synchronously,
        # then writes shards in a background thread and returns a future.
        # After return, training is free to modify the live tensors.
        return dcp.async_save(state, storage_writer=writer)

    dcp.save(state, storage_writer=writer)
    return None


def _resume_dcp(model, optimizer, scheduler, checkpoint_path, ctx, restore_rng):
    """Sharded resume: each rank reads its own DCP shard (with reshard support).

    Model and standard optimizer state are loaded in separate ``dcp.load`` calls:

    * **Model** uses the default planner — schema mismatches are real bugs and
      should fail loudly.
    * **Optimizer** uses ``DefaultLoadPlanner(allow_partial_load=True)`` — it
      tolerates load-side keys that are absent in the checkpoint (e.g. AdamW
      ``step`` saved as a Python scalar but expected as a tensor, params that
      had not yet been ``optimizer.step()``-ed at save time, or freeze/unfreeze
      schema differences across runs). Missing entries keep their freshly
      initialized values (zero momentum / step=0), which is mathematically
      equivalent to "this param has not been optimized yet". Any hard failure
      raises rather than silently degrading to a fresh optimizer.

    ZeRO optimizer state uses same-world-size rank-local files because calling
    its global ``state_dict`` would first consolidate every shard onto one rank.
    """
    dcp_dir = os.path.join(checkpoint_path, "dcp")
    if not os.path.exists(os.path.join(checkpoint_path, _DCP_METADATA_FILE)):
        raise FileNotFoundError(f"No dcp/ metadata in {checkpoint_path}")

    has_optim = _dcp_has_key(dcp_dir, "optim")
    has_local_zero_optim = os.path.exists(
        os.path.join(checkpoint_path, _ZERO_OPTIMIZER_METADATA_FILE)
    )
    if has_local_zero_optim and not _is_zero_optimizer(optimizer):
        raise RuntimeError(
            "Checkpoint contains rank-local ZeRO optimizer state, but the current "
            "optimizer is not a ZeroRedundancyOptimizer."
        )
    optimizers = [optimizer] if has_optim else []

    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    model_sd, optim_sd = get_state_dict(model, optimizers=optimizers, options=options)

    # 1) Model — strict.
    model_state = {"model": model_sd}
    dcp.load(model_state, storage_reader=dcp.FileSystemReader(dcp_dir))
    set_state_dict(
        model, optimizers=[],
        model_state_dict=model_state["model"],
        optim_state_dict={},
        options=options,
    )
    if ctx.is_main:
        logger.info("model resumed via DCP (strict)")

    # 2) Optimizer — lenient. Missing keys are expected; full failure is not.
    if has_optim:
        optim_state = {"optim": optim_sd}
        try:
            dcp.load(
                optim_state,
                storage_reader=dcp.FileSystemReader(dcp_dir),
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            set_state_dict(
                model, optimizers=[optimizer],
                model_state_dict={},
                optim_state_dict=optim_state["optim"],
                options=options,
            )
            if ctx.is_main:
                logger.info("optimizer resumed via DCP (allow_partial_load=True)")
        except Exception as e:
            raise RuntimeError(
                f"Failed to restore optimizer state from DCP checkpoint {checkpoint_path}; "
                "refusing to continue with a fresh optimizer during --resume."
            ) from e
    elif has_local_zero_optim:
        _load_zero_optimizer_state(checkpoint_path, optimizer, ctx)
        if ctx.is_main:
            logger.info("optimizer resumed from rank-local ZeRO state")

    sched_path = os.path.join(checkpoint_path, "scheduler.pt")
    if scheduler is not None and os.path.exists(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=False))
        if has_local_zero_optim:
            _restore_zero_optimizer_lrs(optimizer, scheduler.get_last_lr())
        if ctx.is_main:
            logger.info("scheduler resumed successfully")

    rng_local = _load_rank_rng(checkpoint_path, ctx)
    rng_per_rank = None
    if restore_rng:
        if rng_local is not None:
            _set_rank_rng(rng_local, ctx)
            if ctx.is_main:
                logger.info("RNG state resumed (per-rank file)")
        elif ctx.is_main:
            logger.warning(
                "No per-rank RNG file for rank %d; continuing with current RNG",
                ctx.rank,
            )
    else:
        # Caller wants deferred restore — synthesize a world-sized list.
        rng_per_rank = _gather_rng_for_deferred(rng_local, ctx)

    dl_state = _load_rank_dataloader(checkpoint_path, ctx)

    meta_path = os.path.join(checkpoint_path, "resume_meta.json")
    saved_epoch = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            saved_epoch = json.load(f).get("epoch")
        saved_epoch = int(saved_epoch) if saved_epoch is not None else None

    return saved_epoch, dl_state, rng_per_rank


def _dcp_has_key(dcp_dir: str, key: str) -> bool:
    """Probe DCP metadata to check whether a top-level key was saved.

    Reads the ``.metadata`` file written by ``torch.distributed.checkpoint``
    and scans the top-level state-dict keys for an exact match or a dotted
    prefix match.

    Examples
    --------
    A checkpoint saved with ``{"model": ..., "optim": ...}`` will produce
    metadata keys like ``"model.encoder.weight"``, ``"optim.state.0.exp_avg"``,
    etc.  Calling ``_dcp_has_key(dcp_dir, "optim")`` returns ``True`` because
    at least one key starts with ``"optim."``.

    Calling ``_dcp_has_key(dcp_dir, "scheduler")`` returns ``False`` when
    the scheduler was not included in the saved state dict.

    Raises
    ------
    OSError
        If the checkpoint directory or ``.metadata`` file cannot be read
        (e.g. path does not exist, permission denied, corrupted file).
    RuntimeError
        If the DCP reader raises an internal error while parsing metadata.

    Notes
    -----
    Only ``OSError`` and ``RuntimeError`` are caught and re-raised explicitly.
    All other exceptions (e.g. ``KeyboardInterrupt``, unexpected bugs in the
    DCP library) are intentionally **not** caught and will propagate normally.
    In particular, this function never silently returns ``True`` on failure —
    any probe error surfaces immediately rather than masking a broken checkpoint.
    """
    try:
        reader = FileSystemReader(dcp_dir)
        meta = reader.read_metadata()
        for k in meta.state_dict_metadata.keys():
            if k == key or k.startswith(key + "."):
                return True
        return False
    except (OSError, RuntimeError):
        raise


def _save_aux_per_rank(path, ctx, dataloader_state):
    """Each rank writes its own RNG and dataloader state files."""
    rng_dir = os.path.join(path, "rng")
    dl_dir = os.path.join(path, "dataloader")
    if ctx.is_main:
        os.makedirs(rng_dir, exist_ok=True)
        os.makedirs(dl_dir, exist_ok=True)
    ctx.barrier()

    torch.save(_capture_rank_rng(ctx), os.path.join(rng_dir, f"rng_rank{ctx.rank}.pt"))

    if dataloader_state:
        torch.save(dataloader_state, os.path.join(dl_dir, f"dl_rank{ctx.rank}.pt"))


def _capture_rank_rng(ctx) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(ctx.device) if torch.cuda.is_available() else None,
    }


def _set_rank_rng(rng: dict, ctx):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"], device=ctx.device)


def _load_rank_rng(checkpoint_path, ctx) -> Optional[dict]:
    f = os.path.join(checkpoint_path, "rng", f"rng_rank{ctx.rank}.pt")
    if not os.path.exists(f):
        return None
    return torch.load(f, map_location="cpu", weights_only=False)


def _load_rank_dataloader(checkpoint_path, ctx) -> dict:
    f = os.path.join(checkpoint_path, "dataloader", f"dl_rank{ctx.rank}.pt")
    if not os.path.exists(f):
        return {}
    return torch.load(f, map_location="cpu", weights_only=False) or {}


def _gather_rng_for_deferred(rng_local, ctx):
    """Build a world-sized RNG list for callers that defer restoration."""
    if rng_local is None:
        return None
    if not ctx.is_distributed:
        return [rng_local]
    out = [None] * ctx.world_size
    dist.all_gather_object(out, rng_local)
    return out


def _get_full_state_dict(
    model: nn.Module,
    ctx: DistributedContext,
    *,
    ignore_frozen_params: bool = False,
    cpu_offload: bool = False,
) -> dict:
    """Get full state dict handling FSDP1/FSDP2/DDP."""
    if isinstance(model, FSDPModule):
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=cpu_offload,
            ignore_frozen_params=ignore_frozen_params,
        )
        model_sd, _ = get_state_dict(model, optimizers=[], options=options)
        return model_sd
    else:
        return unwrap_model(model).state_dict()


def _is_zero_optimizer(optimizer) -> bool:
    """Check if optimizer is a ZeroRedundancyOptimizer."""
    from torch.distributed.optim import ZeroRedundancyOptimizer
    return isinstance(optimizer, ZeroRedundancyOptimizer) or getattr(
        optimizer, "_is_multi_dtype_zero_optimizer", False
    )


def _zero_optimizer_children(optimizer) -> list:
    """Return the ZeRO optimizers represented by an optimizer wrapper."""
    from torch.distributed.optim import ZeroRedundancyOptimizer

    if isinstance(optimizer, ZeroRedundancyOptimizer):
        return [optimizer]
    if getattr(optimizer, "_is_multi_dtype_zero_optimizer", False):
        children = optimizer._optimizers
        if all(isinstance(child, ZeroRedundancyOptimizer) for child in children):
            return children
    raise TypeError(f"Unsupported ZeRO optimizer type: {type(optimizer).__name__}")


def _local_optimizer_state_dict(optimizer) -> dict:
    """Build a complete checkpoint state for one rank-local optimizer."""
    factory = getattr(optimizer, "local_checkpoint_state_dict", None)
    return factory() if factory is not None else optimizer.state_dict()


def _load_local_optimizer_state_dict(optimizer, state_dict: dict) -> None:
    """Restore a complete checkpoint state for one rank-local optimizer."""
    loader = getattr(optimizer, "load_local_checkpoint_state_dict", None)
    if loader is not None:
        loader(state_dict)
    else:
        optimizer.load_state_dict(state_dict)


def _save_zero_optimizer_state(path, optimizer, ctx) -> None:
    """Save each ZeRO rank's local optimizer shard without consolidation."""
    state_dir = os.path.join(path, _ZERO_OPTIMIZER_DIR)
    if ctx.is_main:
        os.makedirs(state_dir, exist_ok=True)
    ctx.barrier()

    children = _zero_optimizer_children(optimizer)
    local_state = []
    for child in children:
        # LR schedulers update ZeRO's public groups. The rank-local optimizer
        # can retain the previous step's LR until the next ZeRO update, so
        # checkpoint the public options that the next update will consume.
        _copy_param_group_options(child.param_groups, child.optim.param_groups)
        local_state.append(_local_optimizer_state_dict(child.optim))
    torch.save(
        local_state,
        os.path.join(state_dir, f"rank_{ctx.rank}.pt"),
    )
    ctx.barrier()

    if ctx.is_main:
        with open(
            os.path.join(path, _ZERO_OPTIMIZER_METADATA_FILE),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "format_version": 1,
                    "world_size": ctx.world_size,
                    "optimizer_count": len(children),
                },
                file,
                indent=2,
            )
    ctx.barrier()


def _load_zero_optimizer_state(checkpoint_path, optimizer, ctx) -> None:
    """Load the ZeRO optimizer shard owned by the current rank."""
    metadata_path = os.path.join(
        checkpoint_path, _ZERO_OPTIMIZER_METADATA_FILE
    )
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)
    format_version = int(metadata["format_version"])
    if format_version != 1:
        raise RuntimeError(
            f"Unsupported rank-local ZeRO checkpoint version: {format_version}."
        )
    saved_world_size = int(metadata["world_size"])
    if saved_world_size != ctx.world_size:
        raise RuntimeError(
            "Rank-local ZeRO optimizer state requires the original world size: "
            f"checkpoint={saved_world_size}, current={ctx.world_size}."
        )

    children = _zero_optimizer_children(optimizer)
    expected_count = int(metadata["optimizer_count"])
    if expected_count != len(children):
        raise RuntimeError(
            "ZeRO optimizer shard count mismatch: "
            f"checkpoint={expected_count}, current={len(children)}."
        )

    rank_path = os.path.join(
        checkpoint_path, _ZERO_OPTIMIZER_DIR, f"rank_{ctx.rank}.pt"
    )
    if not os.path.exists(rank_path):
        raise FileNotFoundError(
            f"ZeRO optimizer state for rank {ctx.rank} not found: {rank_path}"
        )
    local_state = torch.load(rank_path, map_location="cpu", weights_only=False)
    if len(local_state) != len(children):
        raise RuntimeError(
            "ZeRO optimizer state count mismatch in rank file: "
            f"checkpoint={len(local_state)}, current={len(children)}."
        )
    for child, state_dict in zip(children, local_state):
        _load_local_optimizer_state_dict(child.optim, state_dict)
        _copy_param_group_options(child.optim.param_groups, child.param_groups)


def _restore_zero_optimizer_lrs(optimizer, last_lrs) -> None:
    """Restore scheduler-owned LRs on ZeRO's public and local param groups."""
    if len(last_lrs) != len(optimizer.param_groups):
        raise RuntimeError(
            "Scheduler LR count does not match ZeRO parameter groups: "
            f"scheduler={len(last_lrs)}, optimizer={len(optimizer.param_groups)}."
        )
    for param_group, lr in zip(optimizer.param_groups, last_lrs):
        param_group["lr"] = lr
    for child in _zero_optimizer_children(optimizer):
        _copy_param_group_options(child.param_groups, child.optim.param_groups)


def _copy_param_group_options(source_groups, target_groups) -> None:
    """Copy optimizer options without replacing parameter lists."""
    if len(source_groups) != len(target_groups):
        raise RuntimeError(
            "Optimizer parameter-group count mismatch: "
            f"source={len(source_groups)}, target={len(target_groups)}."
        )
    for source, target in zip(source_groups, target_groups):
        for key, value in source.items():
            if key != "params":
                target[key] = value


def _save_training_state(
    model,
    optimizer,
    scheduler,
    epoch,
    path,
    ctx,
    training_args=None,
    dataloader_state=None,
):
    """Legacy path: rank0-aggregated optimizer + scheduler + per-rank RNG."""
    if isinstance(model, FSDPModule):
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        _, optim_sd = get_state_dict(model, optimizers=[optimizer], options=options)
    elif _is_zero_optimizer(optimizer):
        optimizer.consolidate_state_dict()
        optim_sd = optimizer.state_dict() if ctx.is_main else None
    else:
        optim_sd = optimizer.state_dict()

    local_rng = _capture_rank_rng(ctx)
    if ctx.is_distributed:
        rng_per_rank = [None] * ctx.world_size
        dist.all_gather_object(rng_per_rank, local_rng)
    else:
        rng_per_rank = [local_rng]

    dataloader_state_per_rank = None
    if dataloader_state:
        if ctx.is_distributed:
            dataloader_state_per_rank = [None] * ctx.world_size
            dist.all_gather_object(dataloader_state_per_rank, dataloader_state)
        else:
            dataloader_state_per_rank = [dataloader_state]

    if ctx.is_main:
        os.makedirs(path, exist_ok=True)
        torch.save(
            {
                "optimizer": optim_sd,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "epoch": epoch,
                "dataloader_state_per_rank": dataloader_state_per_rank,
                "rng_state_per_rank": rng_per_rank,
            },
            os.path.join(path, "training_state.pt"),
        )


def _save_state_dict_safetensors(
    state_dict: dict,
    filepath: str,
    metadata: Optional[dict[str, str]] = None,
):
    """Save state dict with safetensors, creating fresh tensors to avoid storage issues."""
    clean_sd = {}
    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            v = v.full_tensor()
        clean_sd[k] = v.detach().cpu().clone()

    save_file(clean_sd, filepath, metadata=metadata)


def _resolve_file(checkpoint_path: str) -> str:
    if os.path.isdir(checkpoint_path):
        for name in (_SAFETENSORS_FILE, _PT_FILE):
            f = os.path.join(checkpoint_path, name)
            if os.path.exists(f):
                return f
        raise FileNotFoundError(
            f"No single-file model in {checkpoint_path} "
            f"(if this is a DCP checkpoint, run tools/dcp_to_safetensors.py first)."
        )
    return checkpoint_path


def _load_sd(path: str) -> dict:
    if path.endswith(".safetensors"):
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)
