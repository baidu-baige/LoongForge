# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generic training args + CLI front-end (single module).

This module is the single source of truth for the generic, model-independent
training parameters (``TrainingArgs``, a frozen dataclass instantiated from the
CLI) plus the tooling that turns that dataclass into an argparse CLI.

Usage rules (must follow)
-------------------------
1. Always read fields via direct attribute access: ``training_args.lr_base``.
2. Never use ``getattr(training_args, "x", default)`` or ``cfg.get("x", default)``:
   - a default supplied there creates a second source of truth and hides the real one;
   - a misspelled field should raise ``AttributeError`` immediately, not silently return
     a fallback.
3. To add or change a generic parameter, edit only the matching concern-scoped
   ``_XxxArgs`` mixin dataclass (still one authoritative definition per field);
   ``TrainingArgs`` aggregates the mixins via multiple inheritance and stays a
   single flat frozen dataclass, so the CLI flags, ``--help``, and the parameter
   summary are all generated from it by reflection and attribute access stays
   flat (``training_args.lr_base``).

Boundary: model-structure switches (freeze_vision_encoder, train_expert_only,
compile_model, ...) live in the per-model ModelConfig (YAML model:). Generic
runtime behavior, including framework-managed activation checkpoint selection,
lives here. Data-processing params (image_size, normalization_mode, ...) live in
the per-model DataConfig (YAML data:).
"""

import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional, get_args, get_origin, Union

import torch


_FSDP_CONCRETE_DTYPE_CHOICES = ("fp32", "bf16", "fp16")
_FSDP_DTYPE_BY_NAME = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


# ---------------------------------------------------------------------------
# Custom CLI value parsers (referenced by field metadata below)
# ---------------------------------------------------------------------------


def parse_bool(value: str) -> bool:
    """Parse bool from CLI text: true|false|1|0|yes|no."""
    if value.strip().lower() in {"true", "t", "1", "yes"}:
        return True
    if value.strip().lower() in {"false", "f", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value!r}")


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


def parse_reshard_after_forward_map(value: str):
    """Parse comma-separated ClassName=value pairs for per-module FSDP reshard."""
    result = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "expected comma-separated ClassName=value pairs"
            )
        class_name, raw_value = item.split("=", 1)
        class_name = class_name.strip()
        if not class_name:
            raise argparse.ArgumentTypeError("empty class name in reshard map")
        result[class_name] = parse_reshard_after_forward(raw_value)
    return result


def parse_positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    try:
        int_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if int_value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int_value


def parse_module_key_patterns(
    value: str | None,
    *,
    option_name: str,
) -> list[str]:
    """Parse comma-separated qualified module-key patterns."""
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
class _ModelRoutingArgs:
    """Model routing: which YAML / trainer / tokenizer to select."""

    # ── Model routing (which YAML / trainer / tokenizer) ──
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model identifier (e.g. 'pi05', 'groot_n1_6'). Selects the "
                    "ModelConfig/DataConfig classes and default YAML via "
                    "MODEL_SCHEMA. Required."
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
    trainer_type: str = field(
        default="FinetuneTrainer",
        metadata={
            "help": "Trainer class to instantiate (e.g. FinetuneTrainer, "
                    "GrootN1d6Trainer); resolved by the trainer builder registry."
        },
    )


@dataclass(frozen=True)
class _BasicTrainingArgs:
    """Basic training loop control, seeding, and output directory."""

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
    output_dir: str = field(
        default="outputs/default",
        metadata={
            "help": "Root directory for checkpoints, logs, and other run artifacts."
        },
    )
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
    check_for_nan_in_loss_and_grad: bool = field(
        default=True,
        metadata={
            "help": "Run host-side NaN/Inf checks on loss and gradients each step."
        },
    )


@dataclass(frozen=True)
class _LearningRateArgs:
    """Learning rate, LR groups, and LR schedules (see diagrams below)."""

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
            ],
            "help": "Learning-rate scheduler name. Most values are passed to "
                    "transformers.get_scheduler; lambda_linear uses the custom "
                    "LambdaLinearScheduler."
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
    min_lr: float = field(
        default=1e-6,
        metadata={"help": "Lower bound the LR schedule decays to (floor)."},
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
        default=None,
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
class _OptimizerArgs:
    """Optimizer selection, gradient clipping, and weight decay."""

    optimizer: str = field(
        default="AdamW",
        metadata={
            "help": "Optimizer name. Supported: AdamW, TorchFusedAdamW, "
                    "TEFusedAdamW, ApexFusedAdamW, Adam, SGD. TEFusedAdamW "
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
        metadata={"help": "Decoupled weight decay coefficient (AdamW)."},
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
                    "(mean)."
        },
    )
    adam_beta2: float = field(
        default=0.95,
        metadata={
            "help": "Adam beta2 — exponential decay rate for the second moment "
                    "(variance)."
        },
    )
    adam_eps: float = field(
        default=1e-8,
        metadata={
            "help": "Adam epsilon added to the denominator for numerical stability."
        },
    )


@dataclass(frozen=True)
class _DataArgs:
    """Data loading control (cross-model; per-model processing lives in DataConfig)."""

    dataset_format: str = field(
        default="lerobot_datasets",
        metadata={
            "help": "Dataset backend to use "
                    "(e.g. lerobot_datasets, hdf5_datasets, dummy_datasets)."
        },
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Filesystem path or repo id of the dataset to train on."},
    )
    dataset_strategy: Optional[str] = field(
        default="default",
        metadata={
            "help": "Under --dataset-format lerobot_datasets: the model-specific "
                    "dataset build strategy ('default', 'fastwam', "
                    "'cosmos3_droid', or 'dreamzero'); unknown values fall back "
                    "to 'default'."
        },
    )
    split: str = field(
        default="train",
        metadata={
            "help": "Dataset split to load (RLDS), e.g. 'train' or 'train[:95%%]'."
        },
    )
    lerobotdataset_version: str = field(
        default="v3.0",
        metadata={
            "choices": ["v2.0", "v2.1", "v3.0"],
            "help": "On-disk LeRobot dataset format version to parse.",
        },
    )
    video_backend: str = field(
        default="torchcodec",
        metadata={
            "choices": ["torchcodec", "decord", "opencv", "pyav", "torchvision_av"],
            "help": "Backend used to decode episode videos into frames.",
        },
    )
    streaming: bool = field(
        default=False,
        metadata={
            "help": "Use a streaming/iterable dataset instead of map-style random "
                    "access (lower memory, no global shuffle)."
        },
    )
    data_root_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Root directory containing the datasets referenced by "
                    "--dataset-mix."
        },
    )
    robot_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Robot embodiment type (e.g. libero_franka); selects "
                    "action/state layout."
        },
    )
    task_name: str = field(
        default="perform the task",
        metadata={
            "help": "Language instruction used as the prompt when the dataset has "
                    "none (HDF5)."
        },
    )
    per_device_batch_size: int = field(
        default=4,
        metadata={"help": "Micro-batch size processed per GPU per forward pass."},
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of DataLoader worker processes per rank."})
    dataloader_seed_workers: bool = field(
        default=False,
        metadata={"help": "Set DataLoader worker_init_fn and generator from --seed. "
                          "Default leaves both unset for baseline precision comparison."})
    dataloader_multiprocessing_context: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["fork", "spawn", "forkserver"],
            "help": "Multiprocessing start method for DataLoader workers.",
        },
    )
    distributed_sampler_mode: str = field(
        default="cyclic",
        metadata={
            "choices": ["cyclic", "block"],
            "help": "How the distributed sampler partitions indices across ranks: "
                    "'cyclic' (round-robin) or 'block' (contiguous shards).",
        },
    )
    batch_drop_last: bool = field(
        default=True,
        metadata={
            "cli_type": parse_bool,
            "help": (
                "If True, drop the last incomplete batch so every rank sees the same "
                "number of full-size batches. Applied to both sampler and DataLoader. "
                "Default True preserves training stability."
            ),
        },
    )
    num_samples: int = field(
        default=100,
        metadata={
            "help": "Number of synthetic samples to generate. Only effective "
                    "when --dataset-format=dummy_datasets."
        },
    )


@dataclass(frozen=True)
class _CheckpointArgs:
    """Checkpoint save/resume format and state."""

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
class _LoggingArgs:
    """Logging cadence, GC control, W&B, and TensorBoard."""

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
            "choices": ["fp32", "bf16"],
            "help": "Communication dtype for the manual gradient all-reduce "
                    "(bf16 halves comm volume).",
        },
    )


@dataclass(frozen=True)
class _ActivationCheckpointArgs:
    """activation-checkpoint module selection."""

    activation_checkpoint_module_patterns: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated qualified module-key patterns to wrap "
                    "with activation checkpointing. '*' matches one module-key "
                    "segment."
        },
    )
    activation_checkpoint_skip_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional comma-separated qualified module keys to exclude "
                    "from --activation-checkpoint-module-patterns."
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
                    "re-sharded after forward to save memory.",
        },
    )
    fsdp_reshard_root: Any = field(
        default=False,
        metadata={
            "cli_type": parse_reshard_after_forward,
            "help": "reshard_after_forward policy for the root FSDP group "
                    "specifically.",
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
    fsdp_wrap_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated module class names that define exact FSDP "
                    "units. Activation-checkpoint wrappers are matched by their "
                    "wrapped module class. When set, automatic unit selection "
                    "is disabled."
        },
    )
    fsdp_no_wrap_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated module class names to exclude from FSDP "
                    "wrapping."
        },
    )
    fsdp_min_num_params: int = field(
        default=1_000_000,
        metadata={
            "help": "Minimum parameter count for auto-wrapping repeated "
                    "transformer layers."
        },
    )
    fsdp_leftover_min_num_params: int = field(
        default=1_000_000,
        metadata={
            "help": "Minimum parameter count for auto-wrapping remaining "
                    "(leftover) modules."
        },
    )
    fsdp_original_param_dtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": _FSDP_CONCRETE_DTYPE_CHOICES,
            "help": "Optional model parameter dtype before FSDP sharding. "
                    "Unset preserves authored mixed dtypes and otherwise follows "
                    "--dtype."
        },
    )
    fsdp_unsharded_param_dtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": _FSDP_CONCRETE_DTYPE_CHOICES,
            "help": "Optional dtype of all-gathered FSDP parameters used for "
                    "forward/backward. Unset preserves authored mixed dtypes when "
                    "the original dtype is also unset; otherwise follows --dtype."
        },
    )
    fsdp_reduce_dtype: str = field(
        default="fp32",
        metadata={
            "choices": _FSDP_CONCRETE_DTYPE_CHOICES,
            "help": "FSDP gradient reduction dtype."
        },
    )
    fsdp_cast_forward_inputs: bool = field(
        default=True,
        metadata={
            "help": "Cast FSDP unit forward inputs to its parameter dtype."
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
    ddp_bucket_cap_mb_list: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated per-bucket sizes (MiB) for fine-grained DDP "
                    "bucketing."
        },
    )
    ddp_batched_grad_copy: bool = field(
        default=False,
        metadata={
            "help": "Batch gradient copies into buckets to reduce kernel launches."
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
                    "parameter dtypes for dtype-sensitive modules."
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


# ---------------------------------------------------------------------------
# TrainingArgs - aggregate of the grouped mixins (single flat frozen dataclass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingArgs(
    _ModelRoutingArgs,
    _BasicTrainingArgs,
    _LearningRateArgs,
    _OptimizerArgs,
    _DataArgs,
    _CheckpointArgs,
    _FreezeArgs,
    _LoraArgs,
    _LoggingArgs,
    _ProfilerArgs,
    _CudaGraphArgs,
    _ActivationCheckpointArgs,
    _DistributedArgs,
):
    """Generic training args (single source of truth). Frozen after construction.

    The field definitions live in the concern-scoped ``_XxxArgs`` mixins above;
    this class only aggregates them via multiple inheritance. At runtime it is
    still a single flat frozen dataclass, so access stays flat
    (``training_args.lr_base``) and ``dataclasses.fields(TrainingArgs)`` /
    ``OmegaConf.structured(TrainingArgs)`` see every field at the top level.
    To add or change a generic parameter, edit the matching ``_XxxArgs`` mixin.
    """


# ---------------------------------------------------------------------------
# FSDP dtype resolution
# ---------------------------------------------------------------------------


def resolve_fsdp_dtype(value: str) -> torch.dtype:
    """Map a canonical FSDP CLI dtype name to ``torch.dtype``."""
    try:
        return _FSDP_DTYPE_BY_NAME[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported FSDP dtype {value!r}") from exc


# ---------------------------------------------------------------------------
# CLI generation — reflect TrainingArgs into an argparse parser
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


__all__ = [
    "TrainingArgs",
    "add_args_from_dataclass",
    "build_arg_parser",
    "parse_module_key_patterns",
    "parse_reshard_after_forward",
    "parse_reshard_after_forward_map",
    "parse_positive_int",
    "resolve_fsdp_dtype",
]
