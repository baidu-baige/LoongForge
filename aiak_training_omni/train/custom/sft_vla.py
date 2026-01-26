"""Megatron SFT entrypoint for VLA models like pi05."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import os
from pathlib import Path

import torch
from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector
from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group
from torch.utils.data import Dataset, default_collate

from aiak_training_omni.data.lerobot import (
    LeRobotDatasetConfig,
    build_lerobot_dataset,
    get_lerobot_dataset_stats,
    make_pi05_pre_post_processors,
)
from aiak_training_omni.models import get_model_family, get_model_provider
from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.sft.utils import _build_cylic_iterator
from aiak_training_omni.train.trainer_builder import register_model_trainer
from aiak_training_omni.utils import constants, get_args, print_rank_0
from aiak_training_omni.utils.global_vars import get_model_config
from lerobot.configs.types import FeatureType
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


stimer = StragglerDetector()


def _strip_leading_batch_dim(sample: dict):
    """Remove the singleton batch dimension added by the processor."""
    cleaned = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            cleaned[key] = value.squeeze(0)
        elif isinstance(value, list) and len(value) == 1:
            cleaned[key] = value[0]
        else:
            cleaned[key] = value
    return cleaned


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
        if not hasattr(train_args, "lr_warmup_init") or train_args.lr_warmup_init is None:
            train_args.lr_warmup_init = 0.0
        if not hasattr(train_args, "lr_warmup_iters") or train_args.lr_warmup_iters is None:
            train_args.lr_warmup_iters = int(getattr(cfg, "scheduler_warmup_steps", 0))
        if not hasattr(train_args, "lr_decay_style") or train_args.lr_decay_style is None:
            train_args.lr_decay_style = "cosine"
        if not hasattr(train_args, "start_weight_decay") or train_args.start_weight_decay is None:
            train_args.start_weight_decay = float(getattr(cfg, "optimizer_weight_decay", 0.0))
        if not hasattr(train_args, "end_weight_decay") or train_args.end_weight_decay is None:
            train_args.end_weight_decay = float(getattr(cfg, "optimizer_weight_decay", 0.0))
        if not hasattr(train_args, "weight_decay_incr_style") or train_args.weight_decay_incr_style is None:
            train_args.weight_decay_incr_style = "constant"
    # Final safety net in case cfg was None.
    if not hasattr(train_args, "lr") or train_args.lr is None:
        train_args.lr = 1e-4
    if not hasattr(train_args, "min_lr") or train_args.min_lr is None:
        train_args.min_lr = 0.0
    if not hasattr(train_args, "lr_warmup_init") or train_args.lr_warmup_init is None:
        train_args.lr_warmup_init = 0.0
    if not hasattr(train_args, "lr_warmup_iters") or train_args.lr_warmup_iters is None:
        train_args.lr_warmup_iters = 0
    if not hasattr(train_args, "lr_decay_style") or train_args.lr_decay_style is None:
        train_args.lr_decay_style = "constant"
    if not hasattr(train_args, "start_weight_decay") or train_args.start_weight_decay is None:
        train_args.start_weight_decay = 0.0
    if not hasattr(train_args, "end_weight_decay") or train_args.end_weight_decay is None:
        train_args.end_weight_decay = train_args.start_weight_decay
    if not hasattr(train_args, "weight_decay_incr_style") or train_args.weight_decay_incr_style is None:
        train_args.weight_decay_incr_style = "constant"
    # Eval defaults: pi05 pipeline has no val/test datasets, so force-disable eval to avoid None iterators.
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



class Pi05PreprocessedDataset(Dataset):
    """Wrap a LeRobot dataset and run the pi05 processor per-sample."""

    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        processed = self.preprocessor(sample)
        return _strip_leading_batch_dim(processed)


def model_provider(pre_process=True, post_process=True, vp_stage: int | None = None):
    """Build the pi05 model through the standard provider registry."""
    args = get_args()
    model_family = get_model_family(args.model_name)
    provider = get_model_provider(model_family)
    assert provider is not None, f"model provider for {args.model_name} not found"

    config = get_model_config()
    if config is None:
        raise ValueError("pi05 config was not initialized; pass --config-file configs/models/pi05/pi05.yaml")

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

    # Debug snapshot to verify Megatron-facing flags are present.
    dbg = {
        "fp16": config.fp16,
        "bf16": config.bf16,
        "fp8": config.fp8,
        "fp4": config.fp4,
        "enable_autocast": config.enable_autocast,
        "barrier_with_L1_time": config.barrier_with_L1_time,
        "fine_grained_activation_offloading": config.fine_grained_activation_offloading,
        "no_sync_func": config.no_sync_func,
        "overlap_moe_expert_parallel_comm": config.overlap_moe_expert_parallel_comm,
        "calculate_per_token_loss": config.calculate_per_token_loss,
        "init_model_with_meta_device": config.init_model_with_meta_device,
    }
    print_rank_0(f"[pi05] config debug snapshot: {dbg}")

    return provider(pre_process, post_process, vp_stage, config=config)


def get_batch(data_iterator):
    """Generate a batch and move it to the active device."""
    batch = next(data_iterator)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_to_device(value):
        return value.to(device=target_device, non_blocking=True) if torch.is_tensor(value) else value

    for key, val in batch.items():
        batch[key] = _move_to_device(val)
    return batch


def loss_func(local_loss_dict: dict, output_tensor: torch.Tensor):
    """Reduce loss across data-parallel ranks and surface useful metrics."""
    loss = output_tensor.float()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_reduced = {"lm loss": averaged_loss[0]}

    if local_loss_dict and "loss_per_dim" in local_loss_dict:
        # Megatron's reduction expects scalar or 2-element tensors. Collapse per-dim losses to a scalar mean.
        loss_reduced["pi05 loss_per_dim"] = torch.tensor(
            local_loss_dict["loss_per_dim"], device=loss.device
        ).mean()

    return loss, loss_reduced


def forward_step(data_iterator, model):
    """Forward training step."""
    timers = get_timers()

    timers("batch-generator", log_level=2).start()
    with stimer(bdata=True):
        batch = get_batch(data_iterator)
    timers("batch-generator").stop()

    with stimer:
        output_loss, loss_dict = model(batch)

    return output_loss, partial(loss_func, loss_dict)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train/valid/test datasets."""
    args = get_args()
    config = get_model_config()
    if config is None:
        raise ValueError("pi05 config was not initialized; pass --config-file configs/models/pi05/pi05.yaml")

    if getattr(config, "device", None) is None:
        config.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_samples = train_val_test_num_samples[0] if train_val_test_num_samples else None
    if train_samples is None:
        train_samples = getattr(args, "train_iters", 1) * getattr(args, "global_batch_size", 1)
    consumed = getattr(args, "consumed_train_samples", 0) or 0

    # Build the LeRobot-backed dataset mirroring the pretrain pipeline.
    repo_id = None
    if isinstance(getattr(args, "data_path", None), (list, tuple)):
        repo_id = args.data_path[0] if args.data_path else None
    elif isinstance(getattr(args, "data_path", None), str):
        repo_id = args.data_path

    if repo_id is None:
        raise ValueError(
            "pi05 SFT requires --data-path to point to a LeRobot dataset repo_id or local path"
        )

    repo_path = Path(str(repo_id))
    ds_root = repo_path if repo_path.exists() else getattr(args, "data_cache_dir", None)

    ds_cfg = LeRobotDatasetConfig(
        repo_id=repo_id,
        root=str(ds_root) if ds_root is not None else None,
        episodes=None,
        revision=None,
        use_imagenet_stats=True,
        streaming=getattr(args, "sft_data_streaming", False),
        tolerance_s=getattr(config, "tolerance_s", 1e-4),
    )

    base_dataset = build_lerobot_dataset(ds_cfg, policy=config)

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
        print_rank_0(message=f"[sft_vla] Warning: missing visual keys were not added: {missing_visuals}")

    dataset_stats = get_lerobot_dataset_stats(base_dataset)

    preprocess_config = deepcopy(config)
    preprocess_config.device = "cpu"
    preprocessor, _ = make_pi05_pre_post_processors(
        config=preprocess_config, dataset_stats=dataset_stats
    )
    processed_dataset = Pi05PreprocessedDataset(base_dataset, preprocessor)

    train_iter = _build_cylic_iterator(processed_dataset, consumed, default_collate)

    print_rank_0(f"> finished creating {args.model_name} sft datasets ...")

    # Validation/test are not wired for this pipeline.
    return train_iter, None, None


@register_model_trainer(
    model_family=constants.VisionLanguageActionModelFamilies.PI05,
    training_phase=constants.TrainingPhase.SFT,
)
def default_sft_trainer(train_args):
    """Megatron-FSDP trainer for pi05 SFT."""
    _ensure_megatron_defaults(train_args)

    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )

    return trainer
