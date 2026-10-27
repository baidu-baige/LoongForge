# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training arguments for the Torch engine (recipe, CLI, and parsing).

This module owns the parameters that describe *how* to train: model routing, run
length, output location, and initialization source (pretrained weights or
resume), the learning-rate schedule, the data loader, logging, profiling, and
freezing. All Torch CLI definitions live here; ``loongforge.engines.torch.arg_types``
keeps only value parsers. :class:`TrainingArgs` combines the mixins into one
flat frozen dataclass, so attribute access, the CLI, and the
YAML schema are unchanged.

``parse_train_args`` also lives here: it reflects the dataclasses into an
argparse CLI, routes ``--model-name`` through the model catalog, and merges the
YAML config. None of that depends on the backend.
"""

import argparse
import dataclasses
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, List, Optional, Union, get_args, get_origin

from omegaconf import OmegaConf

from loongforge.engines.torch.arg_types import (
    parse_class_names,
    parse_module_key_patterns,
    parse_non_negative_int,
    parse_optional_int_list,
    parse_positive_int,
    parse_reshard_after_forward,
    parse_reshard_after_forward_map,
)
from loongforge.engines.torch.validators import validate_engine_args
from loongforge.models.catalog import get_config_path, get_config_types
from loongforge.training.embodied.args import _EmbodiedDataArgs, _EmbodiedRoutingArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recipe mixins
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ModelRoutingArgs:
    """Model routing: which YAML / trainer / tokenizer to select."""

    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model identifier (e.g. 'pi05', 'gr00t_n1_6'). Selects the "
                    "ModelConfig/DataConfig classes and default YAML via "
                    "loongforge.models.catalog. Required."
        },
    )

    config_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Explicit path to a model YAML config. Overrides the default "
                    "YAML resolved from --model-name; --model-name is still "
                    "required to pick the config classes."
        },
    )

    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory or HF repo id of the tokenizer. Exported to the "
                    "TOKENIZER_PATH env var so the model/data tokenizer loaders "
                    "can pick it up."
        },
    )


@dataclass(frozen=True)
class _RunArgs:
    """Run length, output location, initialization source, seeding, and manual GC."""

    pretrained_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained weights to initialize the model from "
                    "(fine-tuning)."
        },
    )

    resume: bool = field(
        default=False,
        metadata={
            "help": "Resume training (weights + optimizer/scheduler/RNG state) "
                    "from the latest checkpoint in --output-dir."
        },
    )

    train_iters: int = field(
        default=150000,
        metadata={
            "help": "Total number of optimizer update steps to run before stopping."
        },
    )

    save_interval: int = field(
        default=10000,
        metadata={"help": "Write a checkpoint every N iterations; 0 disables saving."},
    )

    seed: int = field(
        default=3047,
        metadata={
            "help": "Global RNG seed for Python/NumPy/PyTorch and data shuffling "
                    "(reproducibility)."
        },
    )

    set_seed_by_rank: bool = field(
        default=False,
        metadata={
            "help": "If True, seed += torch.distributed.get_rank()"
        },
    )

    output_dir: str = field(
        default="outputs/default",
        metadata={
            "help": "Root directory for checkpoints, logs, and other run artifacts."
        },
    )

    manual_gc: bool = field(
        default=False,
        metadata={
            "help": "Disable automatic Python GC and collect explicitly after optimizer steps."
        },
    )

    manual_gc_interval: int = field(
        default=0,
        metadata={
            "help": "Manual GC cadence in steps when --manual-gc is enabled; "
                    "0 disables periodic collection."
        },
    )


@dataclass(frozen=True)
class _LearningRateArgs:
    """Learning rate, LR groups, and LR schedules."""

    lr_base: float = field(
        default=2.5e-5,
        metadata={
            "help": "Base learning rate applied to all parameters not matched by "
                    "--lr-group."
        },
    )

    lr_group: Optional[str] = field(
        default=None,
        metadata={
            "help": "Per-module LR overrides in 'module.path=lr' format, "
                    "comma-separated. Order matters: parameters are assigned to "
                    "the first matching entry and excluded from all later entries. "
                    "Child module paths must appear before their parent paths, "
                    "otherwise the child rule is silently ignored because its "
                    "parameters have already been consumed by the parent. "
                    "Example: 'model.paligemma_with_expert.gemma_expert=1e-4,"
                    "model.paligemma_with_expert=1e-5'. The final catch-all "
                    "group uses --lr-base."
        },
    )
    # ============================================================
    # Learning Rate Schedules (relative LR vs optimizer step)
    #
    # Notation:
    #   W   : warmup steps
    #   T   : total training steps
    #   C   : cycle length (per-cycle span)
    #   peak: maximum LR (after warmup)
    #   min : minimum LR
    #   lr_end: final LR for polynomial decay
    #
    # Axes:
    #   y-axis: lr (relative scale)
    #   x-axis: step →
    #
    # ------------------------------------------------------------
    # linear (warmup + linear decay to 0):
    #
    #   lr ^
    #      |                 /\
    #      |                /  \
    #      |               /    \
    #      |______________/      \____________ 0
    #      +---------------W------T-----------> step
    #
    #   warmup: 0 → peak (linear)
    #   decay : peak → 0 (linear)
    #
    # ------------------------------------------------------------
    # cosine (warmup + cosine decay to 0):
    #
    #   lr ^
    #      |                 /\
    #      |                /  `-.
    #      |               /      `-.
    #      |______________/          `______ 0
    #      +---------------W-----------T----> step
    #
    #   decay follows: 0.5 * (1 + cos(pi * t))
    #
    # ------------------------------------------------------------
    # cosine_with_restarts (periodic cosine decay):
    #
    #   lr ^
    #      |            /\      /\      /\
    #      |           /  \    /  \    /  \
    #      |          /    \  /    \  /    \
    #      |_________/      \/      \/      \___ 0
    #      +-----------W-----------------------> step
    #
    #   each cycle: cosine decay from peak → 0
    #   cycles repeat with period C
    #
    # ------------------------------------------------------------
    # polynomial (warmup + polynomial decay):
    #
    #   lr ^
    #      |                 /\
    #      |                /  `.
    #      |               /     `.
    #      |______________/        `_______ lr_end
    #      +---------------W--------T------> step
    #
    #   decay: (1 - t/T)^p
    #
    # ------------------------------------------------------------
    # constant:
    #
    #   lr ^
    #      | ============================== peak
    #      |
    #      +------------------------------------> step
    #
    # ------------------------------------------------------------
    # constant_with_warmup:
    #
    #   lr ^
    #      |                /=================== peak
    #      |               /
    #      |______________/
    #      +---------------W--------------------> step
    #
    # ------------------------------------------------------------
    # inverse_sqrt (Transformer-style):
    #
    #   lr ^
    #      |                /\
    #      |               /  `--.
    #      |              /       `--.
    #      |_____________/            `--...   ~1/sqrt(step)
    #      +---------------W------------------> step
    #
    #   warmup: linear
    #   decay : ∝ step^(-0.5)
    #
    # ------------------------------------------------------------
    # cosine_with_min_lr:
    #
    #   lr ^
    #      |                 /\
    #      |                /  `-.
    #      |               /      `-.
    #   min|______________/          `------ min
    #      +---------------W-----------T----> step
    #
    # ------------------------------------------------------------
    # cosine_warmup_with_min_lr:
    #
    #   lr ^
    #      |             ./\
    #      |            /   `-.
    #      |           /       `-.
    #   min|__________/           `------ min
    #      +---------------W-----------T--> step
    #
    #   warmup start ≈ peak / W (if not explicitly set)
    #
    # ------------------------------------------------------------
    # lambda_linear (multi-cycle linear warmup + linear decay):
    #
    #   lr ^
    #      |            /\            /\            /\
    #      |           /  \          /  \          /  \
    #      |          /    \        /    \        /    \
    #   min|_________/      \______/      \______/      \______
    #      |        W      C      W      C      W      C
    #      +--------------------------------------------------> step
    #
    #   Per cycle:
    #     warmup: f_start → f_max (linear, over W)
    #     decay : f_max   → f_min (linear, over C - W)
    #
    #   Global step is partitioned into consecutive cycles of length C
    #
    # ============================================================

    lr_decay_style: str = field(
        default="cosine_with_min_lr",
        metadata={
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
                "inverse_sqrt",
                "cosine_with_min_lr",
                "cosine_warmup_with_min_lr",
                "lambda_linear",
                "linear_warmup_cosine_annealing",
            ],
            "help": "Learning-rate scheduler name. Most values are passed to "
                    "transformers.get_scheduler; lambda_linear uses the custom "
                    "LambdaLinearScheduler; linear_warmup_cosine_annealing uses "
                    "torch LinearLR followed by CosineAnnealingLR."
        },
    )

    lr_warmup_iters: int = field(
        default=2000,
        metadata={
            "help": "Number of iterations to linearly warm up the LR from 0 to "
                    "its peak."
        },
    )

    lr_decay_iters: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of scheduler decay steps. Defaults to --train-iters when unset."
        },
    )

    lr_decay_start_iters: int = field(
        default=0,
        metadata={
            "cli_type": parse_non_negative_int,
            "help": "Optimizer step at which custom cosine decay begins, holding "
                    "the peak LR after warmup until then. 0 starts decay immediately "
                    "after warmup. Requires --custom-lr-lambda."
        },
    )

    min_lr: float = field(
        default=1e-6,
        metadata={"help": "Lower bound the LR schedule decays to (floor)."},
    )

    custom_lr_lambda: bool = field(
        default=False,
        metadata={
            "help": "Use the custom cosine LR lambda (warmup + cosine decay to "
                    "--min-lr) instead of the built-in HF scheduler. Only applies "
                    "when --lr-decay-style is cosine_with_min_lr or "
                    "cosine_warmup_with_min_lr."
        },
    )
    # ── lambda_linear scheduler params ──

    lambda_f_max: float = field(
        default=0.4,
        metadata={
            "help": "lambda_linear: peak LR multiplier (applied after warmup). "
                    "Only used when --lr-decay-style=lambda_linear."
        },
    )

    lambda_f_min: float = field(
        default=0.0,
        metadata={
            "help": "lambda_linear: minimum LR multiplier (floor of each cycle). "
                    "Only used when --lr-decay-style=lambda_linear."
        },
    )

    lambda_f_start: float = field(
        default=0.0,
        metadata={
            "help": "lambda_linear: LR multiplier at the start of warmup. "
                    "Only used when --lr-decay-style=lambda_linear."
        },
    )

    lambda_cycle_length: Optional[int] = field(
        default=10000,
        metadata={
            "help": "lambda_linear: number of steps per cycle. Defaults to "
                    "--train-iters when unset. "
                    "Only used when --lr-decay-style=lambda_linear."
        },
    )
    # ── polynomial scheduler params ──

    lr_end: float = field(
        default=1e-7,
        metadata={
            "help": "polynomial: final LR value at the end of decay. "
                    "Only used when --lr-decay-style=polynomial."
        },
    )

    polynomial_power: float = field(
        default=1.0,
        metadata={
            "help": "polynomial: power factor of the decay curve. "
                    "Only used when --lr-decay-style=polynomial."
        },
    )
    # ── cosine_with_restarts scheduler params ──

    num_cycles: float = field(
        default=1.0,
        metadata={
            "help": "cosine_with_restarts: number of hard restart cycles. "
                    "Only used when --lr-decay-style=cosine_with_restarts."
        },
    )


@dataclass(frozen=True)
class _FreezeArgs:
    """Parameter freezing by module path prefix."""

    freeze_modules: str = field(
        default="",
        metadata={
            "help": "Comma-separated module path prefixes whose parameters are "
                    "frozen (requires_grad=False)."
        },
    )


@dataclass(frozen=True)
class _LoggingArgs:
    """Logging cadence, W&B, and TensorBoard."""

    log_interval: int = field(
        default=1,
        metadata={
            "help": "Log scalar metrics (loss, LR, throughput) every N iterations."
        },
    )

    detail_log_interval: int = field(
        default=20,
        metadata={
            "help": "Log detailed per-stage timing breakdown every N iterations."
        },
    )

    timing_log_level: int = field(
        default=0,
        metadata={
            "choices": [0, 1],
            "help": "Verbosity of per-stage timing logs: 0 = summary, "
                    "1 = detailed.",
        },
    )

    loss_log_rank: List[int] = field(
        default_factory=lambda: [-1],
        metadata={
            "help": "Ranks whose loss is logged; -1 logs the all-reduced mean "
                    "across ranks."
        },
    )

    wandb_project: str = field(
        default="loongforge-vla",
        metadata={"help": "Weights & Biases project name."},
    )

    wandb_mode: str = field(
        default="disabled",
        metadata={
            "choices": ["online", "offline", "disabled"],
            "help": "W&B logging mode: stream online, buffer offline, or disable.",
        },
    )

    tensorboard_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory for TensorBoard event files; unset disables "
                    "TensorBoard."
        },
    )

    tensorboard_queue_size: int = field(
        default=1000,
        metadata={
            "help": "Max pending events buffered before the async TensorBoard "
                    "writer flushes."
        },
    )


@dataclass(frozen=True)
class _ProfilerArgs:
    """torch.profiler / Nsight profiling capture window."""

    use_pytorch_profiler: bool = field(
        default=False,
        metadata={"help": "Enable torch.profiler to capture CPU/GPU op traces."},
    )

    use_nsys_profiler: bool = field(
        default=False,
        metadata={
            "help": "Enable NVIDIA Nsight Systems (nsys) profiling range markers."
        },
    )

    profile_step_start: int = field(
        default=10,
        metadata={"help": "Iteration at which profiling capture starts."},
    )

    profile_step_end: int = field(
        default=12,
        metadata={"help": "Iteration at which profiling capture stops."},
    )

    profile_ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "Ranks on which the profiler is active."},
    )

    profile_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to write profiler traces to."},
    )


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
            "help": "Disable TF32 for cuDNN convolutions and CUDA matmuls "
                    "(torch.backends.cudnn.allow_tf32, "
                    "torch.backends.cuda.matmul.allow_tf32)."
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
        metadata={"help": "Allocate all params on the 'meta' device. "
                          "Weights are then loaded into the sharded DTensors."
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


@dataclass(frozen=True)
class TrainingArgs(
    _ModelRoutingArgs,
    _RunArgs,
    _LearningRateArgs,
    _OptimizerArgs,
    _EmbodiedRoutingArgs,
    _EmbodiedDataArgs,
    _CheckpointIOArgs,
    _FreezeArgs,
    _LoraArgs,
    _LoggingArgs,
    _ProfilerArgs,
    _CudaGraphArgs,
    _ActivationCheckpointArgs,
    _FP8Args,
    _DistributedArgs,
    _StepArgs,
    _RuntimeArgs,
):
    """Generic training args (single source of truth). Frozen after construction.

    The fields come from the mixins in this module and
    ``loongforge.training.embodied.args``. At runtime this is one flat frozen
    dataclass, so access stays flat (``training_args.lr_base``) and
    ``dataclasses.fields(TrainingArgs)`` / ``OmegaConf.structured(TrainingArgs)``
    see every field at the top level.
    """

    def __post_init__(self):
        # Deterministic mode needs a seeded dataloader (sampler shuffle +
        # per-worker RNG); otherwise data ordering and augmentation are not
        # reproducible run-to-run. Force it on and warn when left off.
        if self.deterministic_mode and not self.dataloader_seed_workers:
            object.__setattr__(self, "dataloader_seed_workers", True)
            logger.warning(
                "deterministic_mode=True forces dataloader_seed_workers=True "
                "(seeding the sampler shuffle and per-worker RNG for reproducibility)."
            )


# ---------------------------------------------------------------------------
# CLI generation
# ---------------------------------------------------------------------------


def _base_type(field_type):
    """Resolve Optional[X] / Union[X, None] to X; leave others unchanged."""
    origin = get_origin(field_type)
    if origin is Union:
        non_none = [t for t in get_args(field_type) if t is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return field_type


def add_args_from_dataclass(parser: argparse.ArgumentParser, cls, prefix: str = ""):
    """Register one argparse argument per dataclass field.

    - ``bool`` fields use ``BooleanOptionalAction`` (``--flag`` / ``--no-flag``).
    - ``list`` fields use ``nargs='+'`` with the element type.
    - ``metadata['cli_type']`` overrides the parser for special types.
    - ``metadata['choices']`` / ``metadata['help']`` are forwarded.
    - Every arg uses ``default=SUPPRESS`` so only user-provided values appear.
    """
    for f in dataclasses.fields(cls):
        name = f"--{(prefix + f.name).replace('_', '-')}"
        meta = f.metadata
        kwargs = {
            "default": argparse.SUPPRESS,
            "help": meta.get("help", ""),
            "dest": prefix + f.name,
        }
        if "choices" in meta:
            kwargs["choices"] = meta["choices"]

        ftype = _base_type(f.type)

        if "cli_type" in meta:
            kwargs["type"] = meta["cli_type"]
        elif ftype is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif get_origin(ftype) in (list, tuple) or ftype in (list, tuple):
            elem_types = [t for t in get_args(ftype) if t is not Ellipsis]
            kwargs["type"] = elem_types[0] if elem_types else str
            kwargs["nargs"] = "+"
        else:
            kwargs["type"] = ftype

        parser.add_argument(name, **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the training arg parser: TrainingArgs flags + YAML dotlist overrides."""
    parser = argparse.ArgumentParser(
        description="LoongForge Embodied Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_args_from_dataclass(parser, TrainingArgs)
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        help="YAML overrides in dotlist format: model.action_horizon=64 data.image_size=448",
    )
    return parser


def validate(training_args, model_cfg, data_cfg):
    """Validate the training recipe, then the backend options.

    Recipe checks cover the learning rate, step counts, manual GC, data loading,
    the tokenizer, and profiling. Backend checks (CUDA graph, FSDP, DDP ZeRO,
    DMuon, LoRA, FP8) run last.
    """
    # ── Learning rate ──
    if training_args.lr_base <= 0:
        raise ValueError(f"--lr-base must be positive, got {training_args.lr_base}")
    if training_args.min_lr < 0:
        raise ValueError(f"--min-lr must be >= 0, got {training_args.min_lr}")
    if training_args.min_lr >= training_args.lr_base:
        logger.warning(
            f"--min-lr ({training_args.min_lr}) >= --lr-base ({training_args.lr_base}); "
            f"cosine decay will have no effect."
        )

    # ── Steps ──
    if training_args.train_iters <= 0:
        raise ValueError(f"--train-iters must be positive, got {training_args.train_iters}")
    if training_args.save_interval < 0:
        raise ValueError(
            f"--save-interval must be >= 0, got {training_args.save_interval}"
        )
    if training_args.lr_decay_iters is not None and training_args.lr_decay_iters <= 0:
        raise ValueError(f"--lr-decay-iters must be positive, got {training_args.lr_decay_iters}")
    lr_schedule_steps = int(training_args.lr_decay_iters or training_args.train_iters)
    if training_args.lr_decay_start_iters < 0:
        raise ValueError("--lr-decay-start-iters must be non-negative")
    if training_args.lr_decay_start_iters:
        if not training_args.custom_lr_lambda or training_args.lr_decay_style not in {
            "cosine_with_min_lr", "cosine_warmup_with_min_lr"
        }:
            raise ValueError("--lr-decay-start-iters requires a custom cosine LR scheduler")
        if not training_args.lr_warmup_iters <= training_args.lr_decay_start_iters < lr_schedule_steps:
            raise ValueError(
                "--lr-decay-start-iters must be at least --lr-warmup-iters "
                "and less than the scheduler steps"
            )
    if training_args.lr_warmup_iters >= lr_schedule_steps:
        logger.warning(
            f"--lr-warmup-iters ({training_args.lr_warmup_iters}) >= scheduler steps ({lr_schedule_steps})"
        )
    if training_args.manual_gc_interval < 0:
        raise ValueError(
            f"--manual-gc-interval must be >= 0, got {training_args.manual_gc_interval}"
        )

    # ── Data loading ──
    if training_args.dataloader_prefetch_factor <= 0:
        raise ValueError(
            "--dataloader-prefetch-factor must be positive, got "
            f"{training_args.dataloader_prefetch_factor}"
        )

    # ── Tokenizer ──
    if training_args.tokenizer_path is None and not os.environ.get("TOKENIZER_PATH"):
        logger.warning(
            "Neither --tokenizer-path nor TOKENIZER_PATH env var is set. "
            "Model initialization may fail if a tokenizer is required."
        )

    # ── Profiler mutual exclusion ──
    if training_args.use_pytorch_profiler and training_args.use_nsys_profiler:
        raise ValueError(
            "--use-pytorch-profiler and --use-nsys-profiler are mutually exclusive."
        )
    if training_args.use_pytorch_profiler or training_args.use_nsys_profiler:
        if training_args.profile_step_end < training_args.profile_step_start:
            raise ValueError(
                f"--profile-step-end ({training_args.profile_step_end}) must be greater than "
                f"--profile-step-start ({training_args.profile_step_start})."
            )

    # ── Model config sanity ──
    if not model_cfg.model_type:
        raise ValueError("ModelConfig.model_type must be set (from YAML model.model_type).")

    validate_engine_args(training_args, model_cfg)


def parse_train_args():
    """Parse CLI + YAML into (TrainingArgs, ModelConfig, DataConfig) instances."""
    parser = build_arg_parser()
    raw = parser.parse_args()
    raw_dict = vars(raw)
    overrides = raw_dict.pop("overrides", [])

    # ── 1. TrainingArgs: structured defaults ← CLI overrides ──
    base = OmegaConf.structured(TrainingArgs)
    cli = OmegaConf.create(dict(raw_dict))
    training_dc = OmegaConf.merge(base, cli)

    # Propagate tokenizer path to env var (model builder reads TOKENIZER_PATH)
    if training_dc.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = training_dc.tokenizer_path

    # ── 2. Route model_name / config_file → schema + YAML ──
    if training_dc.config_file:
        config_path = training_dc.config_file
        if not training_dc.model_name:
            raise ValueError(
                "--model-name is required to select ModelConfig/DataConfig classes "
                "even when --config-file is provided."
            )
    elif training_dc.model_name:
        config_path = get_config_path(training_dc.model_name)
    else:
        raise ValueError("Must specify --model-name (and optionally --config-file).")

    model_config_cls, data_config_cls = get_config_types(training_dc.model_name)
    yaml_cfg = OmegaConf.load(config_path)

    # ── 3. ModelConfig / DataConfig: structured schema ← YAML section ──
    model_dc = OmegaConf.merge(
        OmegaConf.structured(model_config_cls),
        yaml_cfg.get("model", {}),
    )
    data_dc = OmegaConf.merge(
        OmegaConf.structured(data_config_cls),
        yaml_cfg.get("data", {}),
    )

    # ── 4. Shell dotlist overrides (model.* / data.*) ──
    if overrides:
        ov = OmegaConf.from_dotlist(list(overrides))
        unknown = set(ov.keys()) - {"model", "data"}
        if unknown:
            raise ValueError(
                "YAML dotlist overrides must be prefixed with 'model.' or 'data.'. "
                f"Unknown top-level override keys: {sorted(unknown)}"
            )
        if "model" in ov:
            model_dc = OmegaConf.merge(model_dc, ov.model)
        if "data" in ov:
            data_dc = OmegaConf.merge(data_dc, ov.data)

    # ── 5. Validate (DictConfig stage) ──
    validate(training_dc, model_dc, data_dc)

    # ── 6. Instantiate frozen dataclasses ──
    training_args = OmegaConf.to_object(training_dc)
    model_cfg = OmegaConf.to_object(model_dc)
    data_cfg = OmegaConf.to_object(data_dc)
    return training_args, model_cfg, data_cfg


# Alias used by the composition root.
parse = parse_train_args


__all__ = [
    "TrainingArgs",
    "add_args_from_dataclass",
    "build_arg_parser",
    "parse",
    "parse_train_args",
    "validate",
]
