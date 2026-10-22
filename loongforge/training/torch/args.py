# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training arguments for the Torch engine (recipe, CLI, and parsing).

This module owns the parameters that describe *how* to train: model routing, run
length, output location, and initialization source (pretrained weights or
resume), the learning-rate schedule, the data loader, logging, profiling, and
freezing. Backend capabilities stay in
``loongforge.engines.torch.arguments``; :class:`TrainingArgs` combines both sets
of mixins into one flat frozen dataclass, so attribute access, the CLI, and the
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
from typing import List, Optional, Union, get_args, get_origin

from omegaconf import OmegaConf

from loongforge.engines.torch.arguments import (
    _ActivationCheckpointArgs,
    _CheckpointIOArgs,
    _CudaGraphArgs,
    _DistributedArgs,
    _FP8Args,
    _LoraArgs,
    _OptimizerArgs,
    _RuntimeArgs,
    _StepArgs,
)
from loongforge.engines.torch.arguments import parse_non_negative_int, parse_positive_int
from loongforge.engines.torch.validators import validate_engine_args
from loongforge.models.catalog import get_config_path, get_config_types

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
            "help": "Model identifier (e.g. 'pi05', 'groot_n1_6'). Selects the "
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

    trainer_type: str = field(
        default="FinetuneTrainer",
        metadata={
            "help": "Trainer class to instantiate (e.g. FinetuneTrainer, "
                    "GrootN1d6Trainer); resolved by the trainer builder registry."
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
                    "'lingbot_va', 'groot_n1_7', 'cosmos3_droid', "
                    "'dreamzero', or 'wall_oss_0_5'); unknown values raise "
                    "ValueError."
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

    per_device_batch_size: int = field(
        default=4,
        metadata={"help": "Micro-batch size processed per GPU per forward pass."},
    )

    num_workers: int = field(
        default=4,
        metadata={"help": "Number of DataLoader worker processes per rank."})

    dataloader_prefetch_factor: int = field(
        default=2,
        metadata={
            "cli_type": parse_positive_int,
            "help": "Number of batches prefetched by each DataLoader worker.",
        },
    )

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

    sampler_shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle sample order in the sampler."},
    )

    distributed_sampler_mode: str = field(
        default="cyclic",
        metadata={
            "choices": ["cyclic", "block"],
            "help": "How the distributed sampler partitions indices across ranks: "
                    "'cyclic' (round-robin) or 'block' (contiguous shards).",
        },
    )

    sampler_epoch_mode: str = field(
        default="advance",
        metadata={
            "choices": ["advance", "fixed"],
            "help": "Whether the sampler's epoch advances between epochs. "
                    "'advance' (default) uses seed+epoch, so every epoch reshuffles. "
                    "'fixed' pins the epoch to 0, so every epoch reuses the SAME "
                    "permutation -- this reproduces standalone FastWAM, whose "
                    "ResumableEpochSampler epoch is never advanced. Only used by "
                    "model types that register their own sampler builder (currently "
                    "fastwam); see data/datasets/<model>/samplers.py.",
        },
    )

    batch_drop_last: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, drop the last incomplete batch so every rank sees the same "
                "number of full-size batches. Applied to both sampler and DataLoader. "
                "Default False preserves all samples."
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
class TrainingArgs(
    _ModelRoutingArgs,
    _RunArgs,
    _LearningRateArgs,
    _OptimizerArgs,
    _DataArgs,
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

    The fields come from the recipe mixins above and the backend mixins in
    ``loongforge.engines.torch.arguments``. At runtime this is one flat frozen
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


__all__ = [
    "TrainingArgs",
    "add_args_from_dataclass",
    "build_arg_parser",
    "parse_train_args",
    "validate",
]
