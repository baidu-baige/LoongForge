"""Megatron SFT entrypoint for VLA models like groot."""

from __future__ import annotations

from functools import partial
from copy import deepcopy
from dataclasses import fields, replace
import os
from pathlib import Path
from collections import Counter
from pprint import pformat
import json
import hashlib
from threading import Thread
from queue import Queue
from typing import Optional
import random
from typing import Any, Optional

import numpy as np

import torch
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector
from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.core import parallel_state as mpu
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Sampler, default_collate

from loongforge.models import get_model_family, get_model_provider
from loongforge.train.megatron_trainer import MegatronTrainer
from loongforge.train.sft.utils import _build_cylic_iterator, _cyclic_iter
from loongforge.train.trainer_builder import register_model_trainer
from loongforge.utils import constants, get_args, print_rank_0
from loongforge.utils.global_vars import get_model_config
from loongforge.models.embodied.groot_n1_6.configuration_groot import (
    Gr00tN1d6OmniConfig as LrGr00tN1d6Config,
)

stimer = StragglerDetector()


class BatchInterleavedDistributedSampler(Sampler):
    """DP sampler: round-robin batches across ranks with tail padding."""

    def __init__(self, dataset, batch_size, num_replicas, rank, source_sampler=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.source_sampler = source_sampler

    def __iter__(self):
        indices = list(self.source_sampler) if self.source_sampler else list(range(len(self.dataset)))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        batches += batches[:(-len(batches) % self.num_replicas)]
        return iter([idx for i, b in enumerate(batches) if i % self.num_replicas == self.rank for idx in b])

    def __len__(self):
        total = len(self.source_sampler) if self.source_sampler else len(self.dataset)
        n = (total + self.batch_size - 1) // self.batch_size
        per_rank = (n + (-n % self.num_replicas)) // self.num_replicas
        last = total % self.batch_size or self.batch_size
        if (n - 1) % self.num_replicas == self.rank:
            return (per_rank - 1) * self.batch_size + last
        return per_rank * self.batch_size


def _ensure_megatron_defaults(train_args):
    """Backfill Megatron-required args for the VLA sanity path."""
    defaults = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "num_layers": 1,
        "hidden_size": 1,
        "num_attention_heads": 1,
        "seq_length": 1,
        "max_position_embeddings": 1,
    }
    for key, value in defaults.items():
        if not hasattr(train_args, key) or getattr(train_args, key) is None:
            setattr(train_args, key, value)

    world_size = getattr(train_args, "world_size", int(os.environ.get("WORLD_SIZE", 1)))
    total_model_size = (
        train_args.tensor_model_parallel_size
        * train_args.pipeline_model_parallel_size
        * train_args.context_parallel_size
        * train_args.expert_model_parallel_size
    )
    if not hasattr(train_args, "ffn_hidden_size") or train_args.ffn_hidden_size is None:
        hs = getattr(train_args, "hidden_size", 1) or 1
        try:
            hs_int = int(hs)
        except Exception:  # noqa: BLE001
            hs_int = 1
        setattr(train_args, "ffn_hidden_size", max(1, hs_int * 4))
    if not hasattr(train_args, "kv_channels") or train_args.kv_channels is None:
        cfg = get_model_config()
        if cfg is not None and getattr(cfg, "hidden_size", None) and getattr(cfg, "num_attention_heads", None):
            try:
                train_args.kv_channels = max(1, int(cfg.hidden_size) // int(cfg.num_attention_heads))
            except Exception:  # noqa: BLE001
                train_args.kv_channels = None
    if not hasattr(train_args, "data_parallel_size") or train_args.data_parallel_size is None:
        dp_size = max(1, world_size // max(1, total_model_size))
        setattr(train_args, "data_parallel_size", dp_size)
    if not hasattr(train_args, "distributed_timeout_minutes") or train_args.distributed_timeout_minutes is None:
        setattr(train_args, "distributed_timeout_minutes", 30)
    # Training length defaults: run a tiny sanity loop if unset so Megatron schedulers don't error.
    if not hasattr(train_args, "train_iters") or train_args.train_iters is None:
        setattr(train_args, "train_iters", 100)
    if not hasattr(train_args, "train_samples") or train_args.train_samples is None:
        micro_bs = getattr(train_args, "micro_batch_size", 1) or 1
        global_bs = getattr(train_args, "global_batch_size", None)
        if global_bs is None:
            global_bs = micro_bs * getattr(train_args, "data_parallel_size", 1)
        setattr(train_args, "train_samples", int(global_bs * train_args.train_iters))
    # LR/WD scheduler defaults derived from config if absent on args.
    cfg = get_model_config() or getattr(train_args, "model_config", None)
    if cfg is not None:
        if not hasattr(train_args, "lr") or train_args.lr is None:
            train_args.lr = float(getattr(cfg, "optimizer_lr", 1e-4))
        if not hasattr(train_args, "min_lr") or train_args.min_lr is None:
            train_args.min_lr = float(getattr(cfg, "scheduler_decay_lr", 0.0))
        if not hasattr(train_args, "lr_decay_style") or train_args.lr_decay_style is None:
            train_args.lr_decay_style = "cosine"
        if not hasattr(train_args, "start_weight_decay") or train_args.start_weight_decay is None:
            train_args.start_weight_decay = float(getattr(cfg, "optimizer_weight_decay", 0.0))
        if not hasattr(train_args, "end_weight_decay") or train_args.end_weight_decay is None:
            train_args.end_weight_decay = float(getattr(cfg, "optimizer_weight_decay", 0.0))
        if not hasattr(train_args, "weight_decay_incr_style") or train_args.weight_decay_incr_style is None:
            train_args.weight_decay_incr_style = "constant"

        # Warmup behavior:
        # 1) Prefer Megatron-native --lr-warmup-fraction path when provided.
        # 2) Otherwise keep LeRobot-style auto warmup fallback for --warmup-ratio.
        if not hasattr(train_args, "lr_warmup_init") or train_args.lr_warmup_init is None:
            lr_warmup_fraction = getattr(train_args, "lr_warmup_fraction", None)
            if lr_warmup_fraction is None:
                if not hasattr(train_args, "lr_warmup_iters") or train_args.lr_warmup_iters is None:
                    # Calculate warmup_iters from warmup_ratio if not specified
                    # Priority: CLI args > config file > default
                    warmup_ratio = getattr(train_args, "warmup_ratio", None)
                    if warmup_ratio is None:
                        warmup_ratio = getattr(cfg, "warmup_ratio", 0.05)
                    lr_decay_iters = getattr(train_args, "lr_decay_iters", None)
                    train_iters = getattr(train_args, "train_iters", None)

                    # Auto-scale warmup steps based on training duration (LeRobot behavior)
                    if lr_decay_iters is not None and train_iters is not None:
                        if train_iters < lr_decay_iters:
                            # Scale warmup ratio proportionally
                            scale_factor = train_iters / lr_decay_iters
                            warmup_iters = int(int(lr_decay_iters * warmup_ratio) * scale_factor)
                        else:
                            warmup_iters = int(lr_decay_iters * warmup_ratio)
                    else:
                        warmup_iters = 0

                    train_args.lr_warmup_iters = warmup_iters

                # Calculate init_lr using LeRobot formula: init_lr = peak_lr / (warmup_steps + 1)
                warmup_iters = train_args.lr_warmup_iters
                if warmup_iters > 0 and train_args.lr is not None:
                    train_args.lr_warmup_init = train_args.lr / (warmup_iters + 1)
                else:
                    train_args.lr_warmup_init = 0.0
            else:
                # Keep warmup_iters untouched for fraction mode (should stay 0),
                # and use Megatron default warmup init behavior.
                train_args.lr_warmup_init = 0.0
    # Final safety net in case cfg was None.
    if not hasattr(train_args, "lr") or train_args.lr is None:
        train_args.lr = 1e-4
    if not hasattr(train_args, "min_lr") or train_args.min_lr is None:
        train_args.min_lr = 0.0
    if not hasattr(train_args, "lr_warmup_fraction") or train_args.lr_warmup_fraction is None:
        train_args.lr_warmup_fraction = 0.05
    if not hasattr(train_args, "lr_decay_style") or train_args.lr_decay_style is None:
        train_args.lr_decay_style = "constant"
    if not hasattr(train_args, "start_weight_decay") or train_args.start_weight_decay is None:
        train_args.start_weight_decay = 0.0
    if not hasattr(train_args, "end_weight_decay") or train_args.end_weight_decay is None:
        train_args.end_weight_decay = train_args.start_weight_decay
    if not hasattr(train_args, "weight_decay_incr_style") or train_args.weight_decay_incr_style is None:
        train_args.weight_decay_incr_style = "constant"
    # Eval defaults: groot pipeline has no val/test datasets, so force-disable eval to avoid None iterators.
    train_args.eval_iters = 0
    # Megatron expects eval_interval > 0 when computing sample counts; set to 1 to avoid div-by-zero.
    train_args.eval_interval = max(1, getattr(train_args, "eval_interval", 0) or 0)
    if not hasattr(train_args, "eval_batch_size") or train_args.eval_batch_size is None:
        train_args.eval_batch_size = getattr(train_args, "micro_batch_size", 1) or 1
    if not hasattr(train_args, "eval_seq_length") or train_args.eval_seq_length is None:
        train_args.eval_seq_length = getattr(train_args, "seq_length", 1) or 1
    if not hasattr(train_args, "eval_micro_batch_size") or train_args.eval_micro_batch_size is None:
        train_args.eval_micro_batch_size = getattr(train_args, "micro_batch_size", 1) or 1
    if not hasattr(train_args, "eval_max_tokens") or train_args.eval_max_tokens is None:
        train_args.eval_max_tokens = 0
    if not hasattr(train_args, "multiple_validation_sets"):
        train_args.multiple_validation_sets = False
    if not hasattr(train_args, "full_validation"):
        train_args.full_validation = False
    if not hasattr(train_args, "sft"):
        train_args.sft = True
    # Data loader/scheduler bookkeeping defaults.
    if not hasattr(train_args, "consumed_train_samples") or train_args.consumed_train_samples is None:
        train_args.consumed_train_samples = 0
    if not hasattr(train_args, "consumed_valid_samples") or train_args.consumed_valid_samples is None:
        train_args.consumed_valid_samples = 0
    if not hasattr(train_args, "skipped_train_samples") or train_args.skipped_train_samples is None:
        train_args.skipped_train_samples = 0
    # Ensure optimizer precision defaults are safe when precision-aware optimizer is disabled.
    if not hasattr(train_args, "use_precision_aware_optimizer") or train_args.use_precision_aware_optimizer is None:
        train_args.use_precision_aware_optimizer = False
    if not train_args.use_precision_aware_optimizer:
        # Force fp32 dtypes to satisfy OptimizerConfig assertions when precision-aware mode is off.
        for attr in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
            setattr(train_args, attr, torch.float32)


def get_lerobot_dataset_stats(dataset: Any) -> dict[str, Any] | None:
    """Return LeRobot dataset stats if available."""

    meta = getattr(dataset, "meta", None)
    stats = getattr(meta, "stats", None) if meta is not None else None
    return stats


def model_provider(pre_process=True, post_process=True, vp_stage: int | None = None):
    """Build the groot model through the standard provider registry."""
    args = get_args()
    model_family = get_model_family(args.model_name)
    provider = get_model_provider(model_family)
   # Debug hook to inspect args before model construction.
    assert provider is not None, f"model provider for {args.model_name} not found"

    config = get_model_config()
    if config is None:
        raise ValueError("groot config was not initialized; pass --config-file configs/models/groot/groot_n1_6.yaml")

    # Megatron's Float16Module expects fp16/bf16 flags on the config.
    if not hasattr(config, "fp16"):
        config.fp16 = bool(getattr(args, "fp16", False))
    if not hasattr(config, "bf16"):
        config.bf16 = bool(getattr(args, "bf16", False))
    if not hasattr(config, "fp8"):
        config.fp8 = getattr(args, "fp8", None)
    if not hasattr(config, "fp4"):
        config.fp4 = getattr(args, "fp4", None)
    if not hasattr(config, "enable_autocast"):
        config.enable_autocast = bool(getattr(args, "enable_autocast", False))
    if not hasattr(config, "calculate_per_token_loss"):
        config.calculate_per_token_loss = bool(getattr(args, "calculate_per_token_loss", False))
    if not hasattr(config, "init_model_with_meta_device"):
        config.init_model_with_meta_device = bool(getattr(args, "init_model_with_meta_device", False))
    if not hasattr(config, "barrier_with_L1_time"):
        config.barrier_with_L1_time = bool(getattr(args, "barrier_with_L1_time", False))
    if not hasattr(config, "timers") or config.timers is None:
        config.timers = get_timers()
    if not hasattr(config, "fine_grained_activation_offloading"):
        config.fine_grained_activation_offloading = bool(
            getattr(args, "fine_grained_activation_offloading", False)
        )
    if not hasattr(config, "no_sync_func"):
        config.no_sync_func = None
    if not hasattr(config, "overlap_moe_expert_parallel_comm"):
        config.overlap_moe_expert_parallel_comm = bool(
            getattr(args, "overlap_moe_expert_parallel_comm", False)
        )
    if not hasattr(config, "deallocate_pipeline_outputs"):
        # Megatron pipeline scheduler expects this flag; default to False.
        config.deallocate_pipeline_outputs = False

    if getattr(config, "device", None) is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mirror lerobot: only seed when cfg.seed is explicitly set
    cfg_seed = getattr(args, "seed", None)
    if cfg_seed is not None:
        run_seed = int(cfg_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer_path = getattr(args, "hf_tokenizer_path", None)
    if tokenizer_path and Path(tokenizer_path).exists():
        config.tokenizer_assets_repo = tokenizer_path

    # LeRobot factory compatibility fallback for environments where gr00t_n1d6 is
    # not merged yet and policy config misses delta index accessors.
    if not hasattr(config, "observation_delta_indices"):
        config.observation_delta_indices = None
    if not hasattr(config, "action_delta_indices"):
        horizon = int(getattr(config, "action_horizon", 16) or 16)
        config.action_delta_indices = list(range(min(horizon, 16)))
    if not hasattr(config, "reward_delta_indices"):
        config.reward_delta_indices = None

    # Ensure trainable params stay in fp32 after Float16Module wraps the model.
    # In lerobot, the top LLM layers and entire action_head are kept in fp32 (with
    # bf16 autocast for compute). Float16Module calls .bfloat16() on the whole model,
    # so we must tell training_utils to restore these params to fp32 afterwards.
    if getattr(args, "use_fp32_dtype_for_param_pattern", None) is None:
        select_layer = int(getattr(config, "select_layer", 16))
        tune_top = int(getattr(config, "tune_top_llm_layers", 4))
        llm_layer_patterns = []
        for i in range(select_layer - tune_top, select_layer):
            llm_layer_patterns.append(f"language_model.model.layers.{i}.")
            llm_layer_patterns.append(f"language_model.layers.{i}.")
        args.use_fp32_dtype_for_param_pattern = [
            "action_head",
        ] + llm_layer_patterns

    return provider(pre_process, post_process, vp_stage, config=config)


def get_batch(data_iterator):
    """Generate a batch and move it to the active device."""
    from torch.utils._pytree import tree_map

    batch = next(data_iterator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return tree_map(lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x, batch)


# Module-level flag: when True, loss_func skips NCCL all_reduce (used during
# CUDA graph capture/replay where NCCL ops are unreliable).
_SKIP_LOSS_ALLREDUCE = False


def loss_func(local_loss_dict: dict, output_tensor: torch.Tensor):
    """Reduce loss across data-parallel ranks and surface useful metrics."""
    loss = output_tensor.float()

    if _SKIP_LOSS_ALLREDUCE:
        # During graph capture/replay, store raw per-GPU loss without NCCL.
        # The graph wrapper will all_reduce after replay.
        loss_reduced = {"loss": loss.clone().detach()}
    else:
        averaged_loss = average_losses_across_data_parallel_group([loss])
        loss_reduced = {"loss": averaged_loss[0]}

    if local_loss_dict and "loss_per_dim" in local_loss_dict:
        # Megatron's reduction expects scalar or 2-element tensors. Collapse per-dim losses to a scalar mean.
        loss_reduced["groot loss_per_dim"] = torch.tensor(
            local_loss_dict["loss_per_dim"], device=loss.device
        ).mean()

    # Pass total_inputs to training_utils for throughput calculation
    if local_loss_dict and "total_inputs" in local_loss_dict:
        loss_reduced["total_inputs"] = local_loss_dict["total_inputs"]

    return loss, loss_reduced

def _count_tokens_in_batch(batch: dict[str, Any]) -> int:
    """Count total tokens (VLM + DiT) for throughput calculation.

    For GROOT VLA models the compute has two stages:
      1. VLM backbone  — processes ``input_ids`` (text + image tokens)
      2. DiT action head — processes ``sa_embs = cat(state, action)`` tokens

    Returns:
        total_tokens: Sum of VLM tokens and DiT tokens (padding included).
    """
    if "input_ids" in batch:
        batch_size = batch["input_ids"].shape[0]
    elif "action" in batch:
        batch_size = batch["action"].shape[0]
    elif "state" in batch:
        batch_size = batch["state"].shape[0]
    else:
        batch_size = 1

    if "input_ids" in batch:
        vlm_seq_len = batch["input_ids"].shape[1]
        vlm_tokens = batch_size * vlm_seq_len
    else:
        vlm_tokens = 0

    # Get action_horizon from model config (default: 50)
    config = get_model_config()
    action_horizon = int(getattr(config, "action_horizon", 50) or 50)
    dit_tokens = batch_size * (1 + action_horizon)

    total_tokens = vlm_tokens + dit_tokens

    return total_tokens

def forward_step(data_iterator, model):
    """Forward training step."""
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        batch = get_batch(data_iterator=data_iterator)
    timers("batch-generator").stop()

    total_tokens = _count_tokens_in_batch(batch)
    with stimer:
        # GrooT model expects inputs dict with keys:
        # - state, action, embodiment_id, action_mask for action input
        # - vlm_content with text and images for VLM input
        output = model(batch)
        # The model returns a dict with 'loss' key
        output_loss = output["loss"]

    # Scale by data parallel size so total_inputs reflects global batch tokens.
    # Use CPU tensor to avoid CUDA allocation during CUDA graph capture.
    dp_world_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    output["total_inputs"] = torch.tensor(total_tokens * dp_world_size, dtype=torch.float)
    return output_loss, partial(loss_func, output)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/valid/test datasets."""
    try:
        from loongforge.models.embodied.groot_n1_6.processor_groot import (
            make_gr00t_n1d6_pre_post_processors,
        )
        from loongforge.models.embodied.groot_n1_6.utils import EMBODIMENT_STAT_CONFIGS
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.datasets.sampler import EpisodeAwareSampler
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.datasets.factory import make_dataset
        # from lerobot.policies.gr00t_n1d6.configuration_gr00t_n1d6 import (
        #     Gr00tN1d6Config as LrGr00tN1d6Config,
        # )
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(
            "groot_n1_6 SFT trainer requires lerobot + gr00t_n1d6 policy dependencies. "
            "Please ensure the runtime environment has these packages available."
        ) from error

    args = get_args()
    config = get_model_config()
    if config is None:
        raise ValueError("groot config was not initialized; pass --config-file configs/models/groot/groot_n1_6.yaml")

    if getattr(config, "device", None) is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_samples = train_val_test_num_samples[0] if train_val_test_num_samples else None
    if train_samples is None:
        train_samples = getattr(args, "train_iters", 1) * getattr(args, "global_batch_size", 1)
    consumed = getattr(args, "consumed_train_samples", 0) or 0

    # Build the LeRobot-backed dataset via shared helper to mirror lerobot_train.
    repo_id = None
    if isinstance(getattr(args, "data_path", None), (list, tuple)):
        repo_id = args.data_path[0] if args.data_path else None
    elif isinstance(getattr(args, "data_path", None), str):
        repo_id = args.data_path

    if repo_id is None:
        raise ValueError(
            "groot SFT requires --data-path to point to a LeRobot dataset repo_id or local path"
        )

    repo_path = Path(str(repo_id))
    is_local_repo = repo_path.exists()
    ds_root = repo_path if is_local_repo else getattr(args, "data_cache_dir", None)
    repo_id_value = repo_path.name if is_local_repo else str(repo_id)

    # Build dataset via lerobot's factory to match lerobot_train.py path exactly.
    ds_cfg = DatasetConfig(
        repo_id=repo_id_value,
        root=str(ds_root) if ds_root is not None else None,
        episodes=None,
        revision=None,
        use_imagenet_stats=True,
        streaming=getattr(args, "sft_data_streaming", False),
    )
    

    # Instantiate lerobot-native config so training presets (optimizer/scheduler) are populated.
    lr_field_names = {f.name for f in fields(LrGr00tN1d6Config)}
    lr_kwargs = {k: v for k, v in config.__dict__.items() if k in lr_field_names}
    lr_policy_cfg = LrGr00tN1d6Config(**lr_kwargs)
    for k, v in config.__dict__.items():
        if k not in lr_field_names:
            setattr(lr_policy_cfg, k, v)
    
    tp_cfg = TrainPipelineConfig(dataset=ds_cfg, policy=lr_policy_cfg)
    if getattr(tp_cfg, "optimizer", None) is None:
        optimizer_preset = getattr(lr_policy_cfg, "get_optimizer_preset", None)
        if callable(optimizer_preset):
            tp_cfg.optimizer = optimizer_preset()
    if getattr(tp_cfg, "scheduler", None) is None:
        scheduler_preset = getattr(lr_policy_cfg, "get_scheduler_preset", None)
        if callable(scheduler_preset):
            tp_cfg.scheduler = scheduler_preset()
    base_dataset = make_dataset(tp_cfg)
    # Debug hook to inspect the raw dataset object before training.
   

    visual_feature_keys = [
        key
        for key, feat in getattr(base_dataset.meta, "features", {}).items()
        if getattr(feat, "type", None) is FeatureType.VISUAL and key.startswith("observation.images.")
    ]
    required_feature_keys = set(visual_feature_keys) | {"observation.state", "action"}
    missing_required = [
        key for key in required_feature_keys
        if key not in getattr(base_dataset.meta, "features", {})
    ]
    if missing_required:
        print_rank_0(message=f"[sft_groot] Missing required dataset features: {missing_required}")

    if hasattr(base_dataset, "video_backend"):
        print_rank_0(message=f"[sft_groot] video_backend = {base_dataset.video_backend}")

    dataset_stats = get_lerobot_dataset_stats(base_dataset)

    # Auto-fill config features from dataset metadata if the caller didn't set them,
    # mirroring lerobot's factory logic so camera keys align with the dataset.
    ds_features = dataset_to_policy_features(base_dataset.meta.features)
    config.output_features = {k: ft for k, ft in ds_features.items() if ft.type is FeatureType.ACTION}
    if not config.input_features:
        config.input_features = {k: ft for k, ft in ds_features.items() if k not in config.output_features}
    else:
        # Backfill any missing inputs (especially visual keys) from the dataset metadata.
        for k, ft in ds_features.items():
            if k not in config.output_features and k not in config.input_features:
                config.input_features[k] = ft
    missing_visuals = [
        k
        for k, ft in ds_features.items()
        if ft.type is FeatureType.VISUAL and k not in config.input_features
    ]
    if missing_visuals:
        print_rank_0(message=f"[sft_groot] Warning: missing visual keys were not added: {missing_visuals}")

    # Keep policy features synchronized on both configs.
    lr_policy_cfg.output_features = config.output_features
    lr_policy_cfg.input_features = config.input_features
    lr_policy_cfg.device = "cpu"

    preprocessor_cfg = replace(lr_policy_cfg,
        max_state_dim=29,
        max_action_dim=29,
        action_horizon=16,
    )

    preprocessor, _postprocessor = make_gr00t_n1d6_pre_post_processors(
        policy_cfg=preprocessor_cfg,
        dataset_stats=dataset_stats,
        dataset=base_dataset,
        max_length=getattr(args, "cuda_graph_pad_length", None),
    )

    # Optional debug hook: force the first sample index to match a reference run (e.g., lerobot).
    sampler = None
    if hasattr(lr_policy_cfg, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            base_dataset.meta.episodes["dataset_from_index"],
            base_dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=base_dataset.episodes,
            drop_n_last_frames=lr_policy_cfg.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    num_workers = getattr(args, "num_workers", 0) or 0
    streaming = bool(getattr(tp_cfg.dataset, "streaming", False))

    dp_world_size = mpu.get_data_parallel_world_size()
    if dp_world_size > 1:
        sampler = BatchInterleavedDistributedSampler(
            dataset=base_dataset,
            batch_size=args.micro_batch_size,
            num_replicas=dp_world_size,
            rank=mpu.get_data_parallel_rank(),
            source_sampler=sampler,
        )
        shuffle = False

    # Seed for reproducible dataloader ordering.
    run_seed = args.seed if args.seed is not None else 42
    data_loader_generator = torch.Generator().manual_seed(run_seed)

    def worker_init_fn(worker_id):
        """Set per-worker seed derived from the run seed."""
        worker_seed = run_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    dataloader = torch.utils.data.DataLoader(
        base_dataset,
        num_workers=args.num_workers,
        batch_size=args.micro_batch_size,
        shuffle=shuffle and not streaming,
        sampler=sampler,
        pin_memory=config.device == "cuda",
        drop_last=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=data_loader_generator,
    )

    def _prefetch_preprocess_iter(dl_iter, prefetch_count=2):
        """Async prefetch wrapper: preprocess next batch(es) in a background
        thread so that CPU preprocessing overlaps with GPU forward/backward."""
        q = Queue(maxsize=prefetch_count)
    
        def _worker():
            try:
                for batch in dl_iter:
                    q.put(preprocessor(batch))
            except Exception as e:
                q.put(e)
    
        t = Thread(target=_worker, daemon=True)
        t.start()
    
        while True:
            item = q.get()
            if isinstance(item, Exception):
                raise item
            yield item
    train_iter = _prefetch_preprocess_iter(_cyclic_iter(dataloader), prefetch_count=2)

    return train_iter, None, None


@register_model_trainer(
    model_family=constants.VisionLanguageActionModelFamilies.GROOT_N1_6,
    training_phase=constants.TrainingPhase.SFT,
)
def default_sft_trainer(train_args):
    """Megatron-FSDP trainer for groot SFT."""
    _ensure_megatron_defaults(train_args)

    # Validate that --cuda-graph-pad-length is set when using full_iteration scope.
    # Without padding, _graph_safe_unpad_and_attend may access packed tensors OOB during capture.
    if (getattr(train_args, 'cuda_graph_impl', 'none') == 'local' and
            getattr(train_args, 'cuda_graph_scope', 'full') == 'full_iteration'):
        if getattr(train_args, 'cuda_graph_pad_length', None) is None:
            raise ValueError(
                "--cuda-graph-pad-length must be set when using "
                "--cuda-graph-scope=full_iteration. Without padding tokens, "
                "_graph_safe_unpad_and_attend may access packed tensors out of "
                "bounds during CUDA graph capture."
            )

    # Register full-iteration CUDA graph wrapper for this model family.
    from loongforge.train.training_utils import (
        register_full_iteration_graph_wrapper,
        register_per_microbatch_graph_wrapper,
    )
    from loongforge.models.embodied.groot_n1_6.modules.groot_full_iteration_graph import (
        GrootFullIterationGraph,
    )
    from loongforge.models.embodied.groot_n1_6.modules.groot_per_microbatch_graph import (
        GrootPerMicrobatchGraph,
    )
    register_full_iteration_graph_wrapper("groot_n1_6", GrootFullIterationGraph)
    register_per_microbatch_graph_wrapper("groot_n1_6", GrootPerMicrobatchGraph)

    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )

    return trainer


