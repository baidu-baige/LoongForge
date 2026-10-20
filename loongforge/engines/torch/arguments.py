# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backend arguments for the Torch engine.

Each mixin here describes a capability of this backend: how a step is executed,
which runtime switches are applied at initialization, and in which on-disk
format a checkpoint is written. ``TrainingArgs`` in
``loongforge.training.torch.args`` aggregates these mixins with the training
recipe mixins; at runtime it is still one flat frozen dataclass, so engine code
keeps reading ``training_args.<field>``.

The CLI parsers below are shared with the training layer because parsed fields
such as ``--dtype`` and ``--activation-checkpoint-module-patterns`` are
consumed here.
"""

import argparse
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional

import torch


# ---------------------------------------------------------------------------
# Custom CLI value parsers (referenced by field metadata below)
# ---------------------------------------------------------------------------
def parse_reshard_after_forward(value: str):
    """Parse FSDP2 reshard_after_forward from CLI text: true|false|none|int>1."""
    normalized = value.strip().lower()
    if normalized in {"true", "t"}:
        return True
    if normalized in {"false", "f"}:
        return False
    if normalized in {"none", "null"}:
        return None
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected one of: true, false, none, or an integer greater than 1"
        ) from exc
    if int_value <= 1:
        raise argparse.ArgumentTypeError(
            "integer reshard_after_forward must be greater than 1"
        )
    return int_value


def parse_reshard_after_forward_map(raw_map: str):
    """Parse comma-separated ClassName=value pairs for per-module FSDP reshard_after_forward.

    Each pair maps a module class name to a reshard strategy accepted by
    ``parse_reshard_after_forward``.  Unknown keys are passed through as-is.

    Example::

        "TransformerLayer=True,EmbeddingLayer=False"
        # -> {"TransformerLayer": True, "EmbeddingLayer": False}
    """
    class_to_reshard = {}
    for pair in raw_map.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                "expected comma-separated ClassName=value pairs"
            )
        class_name, reshard_value = pair.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise argparse.ArgumentTypeError("empty class name in reshard map")
        class_to_reshard[class_name] = parse_reshard_after_forward(reshard_value)
    return class_to_reshard


def parse_positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if int_value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int_value


def parse_non_negative_int(value: str) -> int:
    """Parse a non-negative integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a non-negative integer"
        ) from exc
    if int_value < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return int_value


def parse_module_key_patterns(
    value: str | list[str] | None,
    *,
    option_name: str,
) -> list[str]:
    """Parse comma-separated qualified module-key patterns.

    Accepts a raw CLI string or an already-split list so the same helper can
    serve both argparse ``type=`` and the later wrap-time re-parse.
    """
    if isinstance(value, (list, tuple)):
        value = ",".join(value)
    normalized_patterns = (value or "").strip()
    if not normalized_patterns:
        return []

    patterns = [
        pattern.strip()
        for pattern in normalized_patterns.split(",")
        if pattern.strip()
    ]
    if any(
        any(not segment for segment in pattern.split("."))
        for pattern in patterns
    ):
        raise ValueError(f"{option_name} cannot contain empty segments")
    return patterns


def parse_class_names(value: str) -> list[str]:
    """Parse a comma-separated list of class names."""
    return [name.strip() for name in (value or "").split(",") if name.strip()]


def parse_dtype_from_str(dtype_name: str) -> torch.dtype:
    """Map a canonical FSDP CLI dtype name to ``torch.dtype``."""
    DTYPE_MAP_BY_STR = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }

    try:
        return DTYPE_MAP_BY_STR[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported FSDP dtype {dtype_name!r}") from exc


def parse_optional_int_list(raw_value: str | None) -> list[int] | None:
    """Parse a comma-separated string of integers into ``list[int]``.

    Returns ``None`` for empty or whitespace-only input so optional kwargs
    remain unset.

    Example::

        parse_optional_int_list("25,50,100")
        # -> [25, 50, 100]

        parse_optional_int_list(None)
        # -> None
    """
    if not raw_value or not raw_value.strip():
        return None
    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    return [int(item) for item in items] if items else None


# ---------------------------------------------------------------------------
# TrainingArgs - single source of truth for generic training params
#
# The parameters are split into concern-scoped mixin dataclasses (_XxxArgs)
# below; ``TrainingArgs`` aggregates them via multiple inheritance and remains a
# single flat frozen dataclass. Attribute access stays flat
# (``training_args.lr_base``) and the CLI/serialization behavior is unchanged.
# To add or change a generic parameter, edit the matching _XxxArgs mixin.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StepArgs:
    """Per-micro-batch step execution and gradient accumulation."""

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of micro-batches accumulated before each optimizer "
                    "step; effective batch = per_device_batch_size * world_size "
                    "* this value."
        },
    )

    loss_spike_threshold: float = field(
        default=100.0,
        metadata={
            "help": "Loss spike guard threshold. The scaled backward loss "
                    "(loss / gradient_accumulation_steps) is checked each "
                    "micro-batch; if it is NaN/Inf or greater than this value, "
                    "that loss contribution is zeroed before backward and the "
                    "optimizer iteration is counted as spiked/skipped."
        },
    )

    check_for_nan_in_loss_and_grad: bool = field(
        default=True,
        metadata={
            "help": "Run host-side NaN/Inf checks on loss and gradients each step."
        },
    )


@dataclass(frozen=True)
class _RuntimeArgs:
    """Backend runtime switches applied at initialization."""

    deterministic_mode: bool = field(
        default=False,
        metadata={
            "help": "Force cuDNN deterministic algorithms. Improves "
                    "reproducibility at some throughput cost; requires "
                    "CUBLAS_WORKSPACE_CONFIG to be set."
        },
    )

    disable_tf32: bool = field(
        default=False,
        metadata={
            "help": "disable"
                    "torch.backends.cudnn.allow_tf32"
                    "torch.backends.cuda.matmul.allow_tf32"
        },
    )

    cudnn_benchmark: bool = field(
        default=False,
        metadata={
            "help": "Let cuDNN autotune convolution algorithms for the shapes it "
                    "sees (torch.backends.cudnn.benchmark). Worth it for models "
                    "with a fixed conv shape per step, e.g. the FastWAM VAE "
                    "encoder; costs a slower first step per new shape. Mutually "
                    "exclusive with --deterministic-mode."
        },
    )


@dataclass(frozen=True)
class _CheckpointIOArgs:
    """On-disk checkpoint format and save behavior."""

    save_format: str = field(
        default="safetensors",
        metadata={
            "choices": ["safetensors", "pt", "dcp"],
            "help": "On-disk checkpoint format: safetensors, raw torch .pt, or "
                    "distributed checkpoint (dcp).",
        },
    )

    save_training_state: bool = field(
        default=True,
        metadata={
            "help": "Also save optimizer, LR scheduler, and RNG state (needed to "
                    "resume)."
        },
    )

    async_save: bool = field(
        default=False,
        metadata={
            "help": "Save checkpoints asynchronously in the background (dcp "
                    "format only)."
        },
    )


@dataclass(frozen=True)
class _OptimizerArgs:
    """Optimizer selection, gradient clipping, and weight decay."""

    optimizer: str = field(
        default="AdamW",
        metadata={
            "help": "Optimizer name. Supported: AdamW, TorchFusedAdamW, "
                    "TEFusedAdamW, ApexFusedAdamW, Adam, SGD, Dmuon. TEFusedAdamW "
                    "requires TransformerEngine; ApexFusedAdamW requires Apex."
        },
    )
    clip_grad: float = field(
        default=1.0,
        metadata={
            "help": "Max global gradient norm for clipping; <=0 disables clipping."
        },
    )
    weight_decay: float = field(
        default=0.01,
        metadata={
            "help": "Decoupled weight decay coefficient (AdamW, and DMuon's "
                    "AdamW route; the Muon route uses --dmuon-muon-weight-decay)."
        },
    )
    weight_decay_grouping: str = field(
        default="all",
        metadata={
            "choices": ["all", "bias_norm"],
            "help": "How weight decay is applied: 'all' applies --weight-decay to every "
                    "trainable parameter; 'bias_norm' excludes bias and norm parameters.",
        },
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={
            "help": "Adam beta1 — exponential decay rate for the first moment "
                    "(mean). Also used by DMuon's AdamW route."
        },
    )
    adam_beta2: float = field(
        default=0.95,
        metadata={
            "help": "Adam beta2 — exponential decay rate for the second moment "
                    "(variance). Also used by DMuon's AdamW route."
        },
    )
    adam_eps: float = field(
        default=1e-8,
        metadata={
            "help": "Adam epsilon added to the denominator for numerical "
                    "stability. Also used by DMuon's AdamW route."
        },
    )
    dmuon_muon_lr: float = field(
        default=0.02,
        metadata={"help": "DMuon Muon-route learning rate."},
    )
    dmuon_momentum: float = field(
        default=0.95,
        metadata={"help": "DMuon momentum for Muon-route parameters."},
    )
    dmuon_ns_steps: int = field(
        default=5,
        metadata={"help": "Number of Newton-Schulz iterations used by DMuon."},
    )
    dmuon_muon_weight_decay: float = field(
        default=0.0,
        metadata={"help": "DMuon decoupled weight decay for Muon-route parameters."},
    )
    dmuon_adamw_lr: float = field(
        default=1e-3,
        metadata={
            "help": "DMuon AdamW-route learning rate for non-Muon parameters. The "
                    "remaining AdamW hyper-parameters are shared with "
                    "--adam-beta1, --adam-beta2, --adam-eps and --weight-decay."
        },
    )
    dmuon_ns_backend: str = field(
        default="gram",
        metadata={
            "choices": ["gram", "direct"],
            "help": "DMuon Newton-Schulz backend.",
        },
    )
    dmuon_ns_coefficients: str = field(
        default="default",
        metadata={
            "choices": ["default", "wallx_muon"],
            "help": "DMuon Newton-Schulz coefficient set.",
        },
    )
    dmuon_nesterov: bool = field(
        default=True,
        metadata={"help": "Enable DMuon Nesterov momentum."},
    )
    dmuon_forward_prefetch_depth: int = field(
        default=1,
        metadata={
            "help": "Number of subsequent DMuon parameter groups to prefetch "
                    "during forward execution."
        },
    )
    dmuon_adamw_foreach: bool = field(
        default=False,
        metadata={
            "help": "Batch the DMuon AdamW update over the FSDP2 symmetric "
                    "shards into multi-tensor (foreach) kernels instead of "
                    "updating one tensor at a time. Numerically identical to the "
                    "scalar path; the DMuon-dedicated parameters keep using it."
        },
    )
    dmuon_adamw_foreach_bucket_mib: float = field(
        default=64.0,
        metadata={
            "help": "Upper bound in MiB on the temporary buffers one foreach "
                    "AdamW bucket may allocate. Larger buckets mean fewer kernel "
                    "launches and more peak memory. Requires "
                    "--dmuon-adamw-foreach."
        },
    )


@dataclass(frozen=True)
class _LoraArgs:
    """LoRA/PEFT fine-tuning configuration."""

    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Enable generic PEFT LoRA fine-tuning before distributed "
                    "wrapping. The model supplies default target modules."
        },
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA scaling alpha."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability inside LoRA adapters."},
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated target module names. Overrides the "
                    "model-provided defaults."
        },
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated modules trained and saved in full. "
                    "Overrides the model-provided defaults."
        },
    )
    lora_bias: str = field(
        default="none",
        metadata={
            "choices": ["none", "all", "lora_only"],
            "help": "PEFT LoraConfig.bias policy.",
        },
    )
    lora_init: str = field(
        default="true",
        metadata={
            "help": "PEFT init_lora_weights mode, such as true, gaussian, or pissa."
        },
    )


@dataclass(frozen=True)
class _CudaGraphArgs:
    """CUDA graph capture and manual gradient all-reduce."""

    cuda_graph_impl: str = field(
        default="none",
        metadata={
            "choices": ["none", "local"],
            "help": "CUDA graph capture backend: 'none' disables, 'local' "
                    "captures the training step to cut per-step launch overhead.",
        },
    )
    cuda_graph_scope: str = field(
        default="full_iteration",
        metadata={
            "choices": ["full_iteration", "per_microbatch"],
            "help": "What to capture into the graph: the whole iteration or "
                    "each micro-batch.",
        },
    )
    cuda_graph_warmup_steps: int = field(
        default=3,
        metadata={
            "help": "Number of eager (uncaptured) warmup iterations before "
                    "graph capture."
        },
    )
    cuda_graph_pad_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Fixed token sequence length to pad to; required so "
                    "captured shapes stay static across steps."
        },
    )
    cuda_graph_ddp_sync_in_graph: bool = field(
        default=False,
        metadata={
            "help": "Capture DDP gradient all-reduce inside the graph instead "
                    "of running it eagerly."
        },
    )
    cuda_graph_grad_sync_bucket_mb: float = field(
        default=200.0,
        metadata={
            "help": "Bucket size (MiB) for the manual gradient all-reduce used "
                    "with CUDA graphs."
        },
    )
    cuda_graph_grad_sync_impl: str = field(
        default="coalesced",
        metadata={
            "choices": ["flat", "coalesced"],
            "help": "Manual gradient all-reduce implementation: single flat "
                    "buffer or coalesced buckets.",
        },
    )
    cuda_graph_grad_sync_dtype: str = field(
        default="fp32",
        metadata={
            "choices": ["fp32", "float32", "bf16", "bfloat16"],
            "help": "Communication dtype for the manual gradient all-reduce "
                    "(bf16 halves comm volume).",
        },
    )


@dataclass(frozen=True)
class _ActivationCheckpointArgs:
    """activation-checkpoint module selection."""
    activation_checkpoint_module_patterns: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": partial(
                parse_module_key_patterns,
                option_name="--activation-checkpoint-module-patterns",
            ),
            "help": "Comma-separated qualified module-key patterns to wrap "
                    "with activation checkpointing. '*' matches one module-key "
                    "segment.",
        },
    )
    activation_checkpoint_skip_modules: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": partial(
                parse_module_key_patterns,
                option_name="--activation-checkpoint-skip-modules",
            ),
            "help": "Optional comma-separated qualified module-key patterns to "
                    "exclude from --activation-checkpoint-module-patterns. Same "
                    "syntax: '*' matches one module-key segment. Every pattern "
                    "must match at least one selected module."
        },
    )


@dataclass(frozen=True)
class _FP8Args:
    """FP8 configuration shared by, or specific to, each backend."""

    # Backend-independent FP8 switches and module selection.
    fp8: bool = field(
        default=False,
        metadata={
            "help": "Convert selected nn.Linear modules for FP8 training with "
                    "the backend selected by --fp8-backend.",
        },
    )
    fp8_backend: str = field(
        default="te",
        metadata={
            "choices": ["te", "torchao"],
            "help": "FP8 implementation: TransformerEngine (te) or native "
                    "PyTorch TorchAO (torchao).",
        },
    )

    fp8_module_patterns: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": partial(
                parse_module_key_patterns,
                option_name="--fp8-module-patterns",
            ),
            "help": "Comma-separated qualified module-key patterns whose "
                    "nn.Linear descendants are converted for FP8. Unlike "
                    "--activation-checkpoint-module-patterns, a pattern selects "
                    "a subtree, not one module. Unset falls back to the model's "
                    "default_fp8_targets().",
        },
    )
    fp8_skip_modules: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": partial(
                parse_module_key_patterns,
                option_name="--fp8-skip-modules",
            ),
            "help": "Optional comma-separated qualified module-key patterns to "
                    "exclude from --fp8-module-patterns. Use this to keep "
                    "numerically sensitive layers (vision towers, action "
                    "projections) out of FP8. Every pattern must match at least "
                    "one selected layer.",
        },
    )
    fp8_min_dim: int = field(
        default=2048,
        metadata={
            "cli_type": parse_positive_int,
            "help": "Only convert layers whose max(in_features, out_features) "
                    "reaches this size. Smaller GEMMs are slower in FP8 than in "
                    "bf16, so converting them costs throughput.",
        },
    )

    # TransformerEngine-only recipe and scaling parameters.
    fp8_te_recipe: str = field(
        default="blockwise",
        metadata={
            "choices": ["blockwise", "current", "delayed", "mxfp8"],
            "help": "TransformerEngine FP8 scaling recipe: blockwise "
                    "(Float8BlockScaling), "
                    "current (Float8CurrentScaling), delayed (DelayedScaling), "
                    "or mxfp8 (MXFP8BlockScaling microscaling).",
        },
    )
    fp8_te_format: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["e4m3", "e5m2", "hybrid"],
            "help": "TransformerEngine-only FP8 data format override. Unset preserves each "
                    "TransformerEngine recipe's native default: E4M3 for "
                    "blockwise and HYBRID for current/delayed/mxfp8. E5M2 "
                    "requires a TransformerEngine recipe that supports pure "
                    "E5M2 and is not supported by MXFP8.",
        },
    )
    fp8_te_margin: int = field(
        default=0,
        metadata={
            "cli_type": parse_non_negative_int,
            "help": "TransformerEngine DelayedScaling only: power-of-two safety "
                    "margin used when "
                    "computing scale = FP8_MAX / (amax * 2**margin).",
        },
    )
    fp8_te_amax_history_len: int = field(
        default=1024,
        metadata={
            "cli_type": parse_positive_int,
            "help": "TransformerEngine DelayedScaling only: number of historical "
                    "amax values "
                    "retained for scale computation.",
        },
    )
    fp8_te_amax_compute_algo: str = field(
        default="max",
        metadata={
            "choices": ["max", "most_recent"],
            "help": "TransformerEngine DelayedScaling only: choose the largest "
                    "historical amax "
                    "or the most recently observed value.",
        },
    )
    fp8_te_reduce_amax: bool = field(
        default=True,
        metadata={
            "help": "TransformerEngine DelayedScaling only: reduce amax across "
                    "the fp8 process "
                    "group so data-parallel ranks use synchronized scales.",
        },
    )
    fp8_te_current_use_power_2_scales: bool = field(
        default=False,
        metadata={
            "help": "TransformerEngine Float8CurrentScaling only: constrain "
                    "scaling factors to "
                    "powers of two.",
        },
    )
    fp8_te_block_use_f32_scales: bool = field(
        default=False,
        metadata={
            "help": "TransformerEngine Float8BlockScaling only: allow "
                    "unconstrained FP32 scales "
                    "instead of the default power-of-two scales.",
        },
    )
    fp8_te_block_backward_override: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["high_precision", "dequantized"],
            "help": "TransformerEngine Float8BlockScaling only: override "
                    "backward precision. Unset preserves the default behavior; "
                    "high_precision keeps high-precision backward operands; "
                    "dequantized dequantizes saved operands before backward.",
        },
    )

    # TorchAO-only recipe and FSDP/shape parameters.
    fp8_torchao_recipe: str = field(
        default="tensorwise",
        metadata={
            "choices": ["tensorwise", "rowwise", "rowwise_with_gw_hp"],
            "help": "TorchAO Float8Linear recipe. Tensorwise is fastest; "
                    "rowwise improves outlier handling; rowwise_with_gw_hp "
                    "keeps grad-weight GEMM in high precision.",
        },
    )
    fp8_torchao_pad_inner_dim: bool = field(
        default=True,
        metadata={
            "help": "TorchAO only: zero-pad unaligned GEMM inner dimensions. "
                    "Enabled by default so Linear K/hidden dimensions that are "
                    "not 16-aligned can use the scaled GEMM. This does not pad "
                    "the flattened token/batch M dimension.",
        },
    )
    fp8_torchao_fsdp_float8_all_gather: bool = field(
        default=False,
        metadata={
            "help": "TorchAO tensorwise + FSDP only: cast weight shards to FP8 "
                    "for all-gather to reduce communication bandwidth.",
        },
    )


@dataclass(frozen=True)
class _DistributedArgs:
    """Parallelism strategy (FSDP/DDP), dtype, ZeRO, and meta-device init."""

    init_on_meta: bool = field(
        default=False,
        metadata={"help": "Allocate all params on 'meta' device."
                          "Then weights are loaded into the sharded DTensors."
        }
    )

    distributed_strategy: str = field(
        default="fsdp",
        metadata={
            "choices": ["ddp", "fsdp"],
            "help": "Parallelism strategy: DDP (replicate) or FSDP2 "
                    "(fully sharded).",
        },
    )
    hsdp_shard_size: Optional[int] = field(
        default=None,
        metadata={
            "cli_type": parse_positive_int,
            "help": "Enable HSDP and set the second 2D mesh dimension size. "
                    "The first mesh dimension replicates parameters across "
                    "groups, and this dimension shards parameters within "
                    "each group. Must divide the distributed world size. "
                    "Unset uses regular 1D FSDP.",
        },
    )
    fsdp_reshard_default: Any = field(
        default=None,
        metadata={
            "cli_type": parse_reshard_after_forward,
            "help": "Default FSDP2 reshard_after_forward policy: "
                    "true|false|none|int>1. Controls whether params are "
                    "re-sharded after forward to save memory. Does not apply to "
                    "the root group, which FSDP2 always keeps unsharded after "
                    "forward.",
        },
    )
    fsdp_reshard_module_overrides: Any = field(
        default=None,
        metadata={
            "cli_type": parse_reshard_after_forward_map,
            "help": "Comma-separated ClassName=value overrides for FSDP "
                    "reshard_after_forward, e.g. GemmaMLP=false,Linear=true.",
        },
    )
    fsdp_ignored_param_names: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Parameter-name substrings excluded from FSDP sharding; "
                    "matched params stay replicated on every rank. Frozen params "
                    "only. Example: q_proj lm_head",
        },
    )
    fsdp_wrap_modules: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": parse_class_names,
            "help": "Comma-separated module class names that define exact FSDP "
                    "units. Activation-checkpoint wrappers are matched by their "
                    "wrapped module class. When set, automatic unit selection "
                    "is disabled."
        },
    )
    fsdp_no_wrap_modules: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": parse_class_names,
            "help": "Comma-separated module class names that must never become "
                    "FSDP units. Their parameters are still sharded by the "
                    "closest enclosing FSDP group."
        },
    )
    fsdp_ignore_frozen_module_classes: Optional[list[str]] = field(
        default=None,
        metadata={
            "cli_type": parse_class_names,
            "help": "Comma-separated frozen module class names whose parameters "
                    "FSDP leaves replicated instead of sharding. This can avoid "
                    "unused parameter all-gathers at the cost of higher per-rank "
                    "memory. Every matched parameter must have requires_grad=False."
        },
    )
    fsdp_ignored_frozen_param_dtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
            "help": "Optional storage and compute dtype for parameters selected "
                    "by --fsdp-ignore-frozen-module-classes. When set, it must "
                    "match --dtype; unset preserves their original dtype."
        },
    )
    fsdp_min_param_num: int = field(
        default=1_000_000,
        metadata={
            "help": "Minimum parameter count for auto-wrapping repeated "
                    "transformer layers."
        },
    )
    fsdp_original_param_dtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
            "help": "Dtype the sharded parameters are stored (and optimizer-stepped) "
                    "in. Unset follows --dtype. Set this to fp32 while --dtype is "
                    "bf16 to keep fp32 master weights with bf16 compute; "
                    "--fsdp-unshard-param-dtype then defaults to --dtype instead of "
                    "'no cast'. Ignored when the model is authored with mixed "
                    "parameter dtypes, which are preserved as-is.",
        },
    )
    fsdp_unshard_param_dtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
            "help": "Optional dtype of all-gathered FSDP parameters used for "
                    "forward/backward. Unset follows --dtype for uniform-dtype "
                    "models (and whenever --fsdp-original-param-dtype is set). "
                    "Authored mixed-dtype models keep 'no extra cast' so each "
                    "group all-gathers in its original dtype.",
        },
    )
    fsdp_reduce_dtype: str = field(
        default="fp32",
        metadata={
            "choices": ["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
            "help": "FSDP gradient reduction dtype.",
        },
    )
    fsdp_cast_forward_inputs: bool = field(
        default=True,
        metadata={
            "help": "Cast FSDP unit forward inputs to its parameter dtype."
        },
    )
    fsdp_output_dtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
            "help": "Dtype for casting floating-point forward outputs of each FSDP unit. "
                    "Useful when different modules have different mixed precision policies. "
                    "If unset, forward outputs are not cast.",
        },
    )
    fsdp_forward_prefetch_distance: int = field(
        default=0,
        metadata={
            "help": "Number of subsequent configured FSDP units to prefetch "
                    "during forward. Supports only containers that execute each "
                    "child once in registration order."
        },
    )
    fsdp_backward_prefetch_distance: int = field(
        default=0,
        metadata={
            "help": "Number of preceding configured FSDP units to prefetch "
                    "during backward. Supports only containers that execute each "
                    "child once in registration order."
        },
    )
    fsdp_delta_fp8_allgather: bool = field(
        default=False,
        metadata={
            "help": "Replace FSDP2 foreach_all_gather with a delta-FP8 path: "
                    "communicate per-block FP8 deltas against a persistent "
                    "unsharded BF16 reference instead of the full weight. "
                    "Default off; FSDP launchers may opt in."
        },
    )
    fsdp_delta_fp8_block: int = field(
        default=256,
        metadata={
            "help": "Elements per FP8 scale block for --fsdp-delta-fp8-allgather."
        },
    )
    fsdp_delta_fp8_prime_steps: int = field(
        default=1,
        metadata={
            "help": "Full BF16 all-gathers used to prime each FSDP unit's "
                    "delta-FP8 reference before switching to quantized deltas."
        },
    )
    fsdp_delta_fp8_reprime_interval: int = field(
        default=0,
        metadata={
            "help": "Force a full BF16 all-gather every N unshards to re-anchor "
                    "the delta-FP8 reference after a discontinuous parameter "
                    "jump such as checkpoint resume. 0 disables re-priming; "
                    "error feedback already prevents in-run drift."
        },
    )
    fsdp_root_optimizer_prefetch: bool = field(
        default=False,
        metadata={
            "help": "Update the root FSDP AdamW shard first, asynchronously "
                    "unshard it, then update the remaining optimizer groups."
        },
    )
    ddp_broadcast_buffers: bool = field(
        default=True,
        metadata={
            "help": "Broadcast module buffers (e.g. BN stats) from rank 0 each "
                    "forward."
        },
    )
    ddp_init_sync: bool = field(
        default=True,
        metadata={
            "help": "Synchronize parameters and buffers across ranks at "
                    "initialization."
        },
    )
    ddp_bucket_cap_mb: Optional[int] = field(
        default=None,
        metadata={"help": "Gradient all-reduce bucket size (MiB) for DDP."},
    )
    ddp_find_unused_parameters: bool = field(
        default=True,
        metadata={
            "help": "Detect parameters unused in the forward graph (needed for "
                    "conditional branches; adds overhead)."
        },
    )
    ddp_gradient_as_bucket_view: bool = field(
        default=False,
        metadata={
            "help": "Expose gradients as views into DDP communication buckets to "
                    "save memory."
        },
    )
    ddp_static_graph: bool = field(
        default=False,
        metadata={
            "help": "Assume a static graph across iterations to enable DDP "
                    "optimizations."
        },
    )
    ddp_skip_all_reduce_unused_params: bool = field(
        default=False,
        metadata={
            "help": "Skip the gradient all-reduce for parameters detected as "
                    "unused."
        },
    )
    ddp_bucket_cap_mb_list: Optional[list[int]] = field(
        default=None,
        metadata={
            "cli_type": parse_optional_int_list,
            "help": "Comma-separated per-bucket sizes (MiB) for fine-grained DDP "
                    "bucketing.",
        },
    )
    ddp_batched_grad_copy: bool = field(
        default=False,
        metadata={
            "help": "Batch gradient copies into buckets to reduce kernel launches."
        },
    )
    ddp_comm_hook: Optional[str] = field(
        default=None,
        metadata={
            "choices": [
                "allreduce_hook",
                "fp16_compress_hook",
                "bf16_compress_hook",
                "fp8_a2a_allgather_hook",
            ],
            "help": "DDP gradient communication hook.",
        },
    )
    ddp_comm_hook_logging: bool = field(
        default=False,
        metadata={
            "help": "Wrap the DDP comm hook with rank-0 logging of bucket info "
                    "before/after each all-reduce."
        },
    )
    ddp_comm_hook_fp8_block: int = field(
        default=256,
        metadata={
            "help": "fp8_a2a_allgather_hook only. Elements per fp8 quantization "
                    "block, i.e. per fp32 scale. Must be a power of two in "
                    "[1, 1024]. Smaller tracks the local dynamic "
                    "range more tightly but costs 4/block extra bytes on the wire "
                    "(1.6% at 256)."
        },
    )
    ddp_comm_hook_fp8_min_mib: float = field(
        default=8.0,
        metadata={
            "help": "fp8_a2a_allgather_hook only. Buckets smaller than this fall "
                    "back to plain AllReduce, since two collectives plus four "
                    "kernels do not pay for themselves on a few MiB. 0 quantizes "
                    "every bucket."
        },
    )
    ddp_comm_hook_fp8_max_scratch_gb: float = field(
        default=24.0,
        metadata={
            "help": "fp8_a2a_allgather_hook only. Total resident comm scratch "
                    "across all buckets, roughly 1.15x each bucket. Buckets that "
                    "do not fit degrade to full-precision AllReduce rather than "
                    "failing the run, so 0 degrades every bucket and disables the "
                    "hook. Raise it to quantize more buckets; there is no value "
                    "that removes the cap."
        },
    )
    ddp_comm_hook_fp8_error_feedback: str = field(
        default="none",
        metadata={
            "choices": ["none", "allgather", "both"],
            "help": "fp8_a2a_allgather_hook only. Carry a quantization point's "
                    "residual into the next step, which bounds the accumulated "
                    "quantization bias at the cost of ~40% more per-step error. "
                    "'allgather' keeps only the shard residual (0.5 extra bytes "
                    "of resident scratch per element) and covers the larger of the "
                    "two error contributions; 'both' also keeps the bucket-sized "
                    "residual, for 4.5 bytes per element. Residuals are fp32: a "
                    "lower-precision residual reintroduces the systematic drift "
                    "error feedback exists to remove."
        },
    )
    ddp_comm_hook_fp8_exempt: str = field(
        default="1d,embed",
        metadata={
            "help": "fp8_a2a_allgather_hook only. Comma list of parameter classes "
                    "kept at the exact fp32 cross-rank mean instead of being fp8 "
                    "quantized (selective-precision exemption). Tokens: '1d' = "
                    "every 1-D tensor (norm scales, biases -- tiny and highly "
                    "loss-sensitive), 'embed' = params owned by an nn.Embedding "
                    "module, 'head' = params owned by an output-head module "
                    "(leaf named lm_head/output/score/classifier). Detection is "
                    "structural, not name-substring, so it is model-agnostic. "
                    "Default '1d,embed' recovers most of the fp8 loss gap at "
                    "negligible cost; set '' to disable. Note: exempting a large "
                    "trainable vocab embedding (e.g. tune_llm=True) can make the "
                    "exempt set big -- use '1d' then; the per-bucket exempt "
                    "fraction is logged at INFO so this is visible."
        },
    )
    dynamo_optimize_ddp: bool = field(
        default=True,
        metadata={
            "help": "Set torch._dynamo.config.optimize_ddp. When True, TorchDynamo "
                    "is allowed to optimize across DDP bucket boundaries. Disable "
                    "(False) if you hit graph-break errors with DDP + torch.compile."
        },
    )
    dtype: str = field(
        default="bfloat16",
        metadata={
            "choices": ["bfloat16", "float16", "float32"],
            "help": "Target training dtype. Uniform-dtype models are cast to "
                    "this dtype; mixed-original-dtype models may preserve some "
                    "parameter dtypes for dtype-sensitive modules.",
        },
    )
    zero_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Wrap optimizer with ZeroRedundancyOptimizer (ZeRO Stage-1). "
                    "Shards optimizer states across ranks. Only effective with DDP."
        },
    )
    zero_parameters_as_bucket_view: bool = field(
        default=False,
        metadata={
            "help": "Pass parameters_as_bucket_view=True to "
                    "ZeroRedundancyOptimizer. Reduces peak memory by reusing "
                    "gradient buffers as parameter storage, but may conflict "
                    "with torch.compile + DDP reducer assumptions. Only "
                    "effective when --zero-optimizer is set."
        },
    )
    zero_master_param_dtype: str = field(
        default="none",
        metadata={
            "choices": ["none", "fp32"],
            "help": "Optional DDP ZeRO-1 master parameter dtype. 'fp32' keeps rank-local fp32 "
                    "master parameters and broadcasts updated model shards after each step.",
        },
    )
