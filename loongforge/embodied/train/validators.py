# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLA training config validation.

Called by parse_train_args() on the DictConfig stage (before to_object()), so it
benefits from OmegaConf missing-key / interpolation checks. Operates on the three
typed configs: training_args (TrainingArgs), model_cfg (ModelConfig),
data_cfg (DataConfig).
"""

import logging
import os

logger = logging.getLogger(__name__)


def _validate_fsdp_ignored_frozen_args(training_args):
    """Reject unsupported frozen-parameter replication combinations."""
    if (
        training_args.fsdp_ignored_frozen_param_dtype is not None
        and not training_args.fsdp_ignore_frozen_module_classes
    ):
        raise ValueError(
            "--fsdp-ignored-frozen-param-dtype requires "
            "--fsdp-ignore-frozen-module-classes"
        )
    if not training_args.fsdp_ignore_frozen_module_classes:
        return
    if training_args.distributed_strategy != "fsdp":
        raise ValueError(
            "--fsdp-ignore-frozen-module-classes requires "
            "--distributed-strategy fsdp"
        )
    if training_args.init_on_meta:
        raise ValueError(
            "--fsdp-ignore-frozen-module-classes is incompatible with "
            "--init-on-meta because ignored meta parameters cannot be "
            "materialized generically"
        )
    if training_args.fsdp_ignored_frozen_param_dtype is not None:
        dtype_aliases = {
            "fp32": "float32",
            "float32": "float32",
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "float16": "float16",
        }
        ignored_dtype = dtype_aliases.get(
            training_args.fsdp_ignored_frozen_param_dtype
        )
        compute_dtype = dtype_aliases.get(training_args.dtype)
        if (
            ignored_dtype is None
            or compute_dtype is None
            or ignored_dtype != compute_dtype
        ):
            raise ValueError(
                "--fsdp-ignored-frozen-param-dtype must match the training "
                f"compute dtype --dtype ({training_args.dtype}); got "
                f"{training_args.fsdp_ignored_frozen_param_dtype}"
            )


def validate(training_args, model_cfg, data_cfg):
    """Validate the combination of TrainingArgs + ModelConfig + DataConfig.

    Raises ValueError on hard errors, logs warnings for soft issues.
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
    if training_args.lr_warmup_iters >= lr_schedule_steps:
        logger.warning(
            f"--lr-warmup-iters ({training_args.lr_warmup_iters}) >= scheduler steps ({lr_schedule_steps})"
        )
    if training_args.manual_gc_interval < 0:
        raise ValueError(
            f"--manual-gc-interval must be >= 0, got {training_args.manual_gc_interval}"
        )
    if training_args.cuda_graph_warmup_steps <= 0:
        raise ValueError(
            f"--cuda-graph-warmup-steps must be positive, got {training_args.cuda_graph_warmup_steps}"
        )

    # ── CUDA graph ──
    if training_args.cuda_graph_impl == "local":
        logger.warning(
            "Host-side loss/grad NaN checks are disabled because CUDA graph mode "
            "is enabled."
        )

        if training_args.cuda_graph_pad_length is None:
            raise ValueError(
                "--cuda-graph-pad-length must be set when --cuda-graph-impl=local."
            )
        if training_args.cuda_graph_pad_length < 0:
            raise ValueError(
                f"--cuda-graph-pad-length must be non-negative, got {training_args.cuda_graph_pad_length}"
            )
        if training_args.cuda_graph_scope not in {"full_iteration", "per_microbatch"}:
            raise ValueError(
                f"Unsupported --cuda-graph-scope={training_args.cuda_graph_scope!r} in embodied trainer."
            )

    # ── Tokenizer ──
    if training_args.tokenizer_path is None and not os.environ.get("TOKENIZER_PATH"):
        logger.warning(
            "Neither --tokenizer-path nor TOKENIZER_PATH env var is set. "
            "Model initialization may fail if a tokenizer is required."
        )

    # ── FSDP ──
    for option_name, value in (
        ("--fsdp-min-param-num", training_args.fsdp_min_param_num),
        (
            "--fsdp-forward-prefetch-distance",
            training_args.fsdp_forward_prefetch_distance,
        ),
        (
            "--fsdp-backward-prefetch-distance",
            training_args.fsdp_backward_prefetch_distance,
        ),
        (
            "--fsdp-delta-fp8-prime-steps",
            training_args.fsdp_delta_fp8_prime_steps,
        ),
        (
            "--fsdp-delta-fp8-reprime-interval",
            training_args.fsdp_delta_fp8_reprime_interval,
        ),
    ):
        if value < 0:
            raise ValueError(f"{option_name} must be non-negative, got {value}")
    delta_fp8_block = training_args.fsdp_delta_fp8_block
    if delta_fp8_block <= 0 or delta_fp8_block & (delta_fp8_block - 1):
        raise ValueError(
            "--fsdp-delta-fp8-block must be a positive power of two, got "
            f"{delta_fp8_block}"
        )
    if delta_fp8_block > 1 << 20:
        raise ValueError(
            "--fsdp-delta-fp8-block must be <= 1048576 for Triton "
            f"tl.arange, got {delta_fp8_block}"
        )
    if (
        training_args.fsdp_delta_fp8_allgather
        and training_args.distributed_strategy != "fsdp"
    ):
        raise ValueError(
            "--fsdp-delta-fp8-allgather requires --distributed-strategy fsdp"
        )

    if (
        training_args.hsdp_shard_size is not None
        and training_args.hsdp_shard_size <= 0
    ):
        raise ValueError(
            f"--hsdp-shard-size must be positive, got {training_args.hsdp_shard_size}"
        )

    if training_args.distributed_strategy == "fsdp":
        wrap_conflict = set(training_args.fsdp_wrap_modules or []) & set(
            training_args.fsdp_no_wrap_modules or []
        )
        if wrap_conflict:
            raise ValueError(
                "Module classes appear in both --fsdp-wrap-modules and "
                f"--fsdp-no-wrap-modules: {', '.join(sorted(wrap_conflict))}."
            )
    elif training_args.fsdp_original_param_dtype is not None:
        logger.warning(
            "--fsdp-original-param-dtype only applies to --distributed-strategy "
            "fsdp; parameter storage follows --dtype under %s.",
            training_args.distributed_strategy,
        )
    _validate_fsdp_ignored_frozen_args(training_args)

    # ── ZeRO optimizer options ──
    if training_args.zero_parameters_as_bucket_view and not training_args.zero_optimizer:
        logger.warning(
            "--zero-parameters-as-bucket-view has no effect without --zero-optimizer."
        )
    if training_args.zero_master_param_dtype != "none" and not training_args.zero_optimizer:
        logger.warning(
            "--zero-master-param-dtype has no effect without --zero-optimizer."
        )
    if (
        training_args.zero_master_param_dtype != "none"
        and training_args.distributed_strategy != "ddp"
    ):
        logger.warning(
            "--zero-master-param-dtype is only effective with --distributed-strategy ddp."
        )

    # ── DMuon optimizer options ──
    if training_args.optimizer.lower() == "dmuon":
        if training_args.distributed_strategy != "fsdp":
            raise ValueError("--optimizer=dmuon requires --distributed-strategy=fsdp.")
        if training_args.save_format == "dcp":
            raise ValueError("--optimizer=dmuon currently supports only safetensors/pt checkpoints.")
        if training_args.zero_optimizer:
            raise ValueError("--optimizer=dmuon is incompatible with --zero-optimizer.")
        if training_args.dmuon_ns_coefficients == "wallx_muon" and training_args.dmuon_ns_backend != "direct":
            raise ValueError(
                "--dmuon-ns-coefficients=wallx_muon requires --dmuon-ns-backend=direct."
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

    # ── LoRA ──
    if training_args.lora_r <= 0:
        raise ValueError(f"--lora-r must be positive, got {training_args.lora_r}")
    if training_args.lora_alpha <= 0:
        raise ValueError(
            f"--lora-alpha must be positive, got {training_args.lora_alpha}"
        )
    if not 0.0 <= training_args.lora_dropout < 1.0:
        raise ValueError(
            "--lora-dropout must be in [0, 1), got "
            f"{training_args.lora_dropout}"
        )
    if training_args.use_lora:
        if training_args.init_on_meta:
            raise ValueError("--use-lora does not currently support --init-on-meta")
        if training_args.save_interval == 0:
            logger.warning(
                "--save-interval=0 disables both checkpoints and LoRA adapter saving."
            )

    # ── Model config sanity ──
    if not model_cfg.model_type:
        raise ValueError("ModelConfig.model_type must be set (from YAML model.model_type).")
