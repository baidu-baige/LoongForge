# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Backend option validation for the Torch engine.

Checks the combinations that only make sense for this backend: CUDA graph, FSDP,
DDP ZeRO, DMuon, LoRA, and FP8. Recipe checks (learning rate, step counts, data
loading, profiling) live in ``loongforge.training.torch.args.validate``, which
calls :func:`validate_engine_args` last.
"""

import logging

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


def validate_engine_args(training_args, model_cfg):
    """Validate backend options for the Torch engine.

    Raises ValueError on hard errors, logs warnings for soft issues.
    """
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
    _validate_fsdp_ignored_frozen_args(training_args)

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

    # ── FP8 ──
    if training_args.fp8:
        if training_args.init_on_meta:
            raise ValueError(
                "--fp8 does not currently support --init-on-meta in the "
                "embodied model wrapping lifecycle."
            )
        if training_args.use_lora:
            raise ValueError(
                "--fp8 and --use-lora are mutually exclusive. PEFT selects "
                "targets and replaces Linear modules before FP8 conversion; "
                "this combination has not been validated for either backend."
            )
        if training_args.fp8_skip_modules and not training_args.fp8_module_patterns:
            logger.warning(
                "--fp8-skip-modules is set without --fp8-module-patterns; it "
                "will be applied against the model's default_fp8_targets()."
            )
        if (
            training_args.fp8_backend == "te"
            and getattr(model_cfg, "compile_model", False)
        ):
            raise ValueError(
                "--fp8 conflicts with model.compile_model=true: te.Linear.forward "
                "carries torch.compiler.disable, so Dynamo raises "
                "'Skip inlining torch.compiler.disable()d function' instead of "
                "graph-breaking. Pass model.compile_model=false to run FP8."
            )
        if training_args.fp8_backend == "torchao":
            te_overrides = (
                training_args.fp8_te_format is not None
                or training_args.fp8_te_margin != 0
                or training_args.fp8_te_amax_history_len != 1024
                or training_args.fp8_te_amax_compute_algo != "max"
                or not training_args.fp8_te_reduce_amax
                or training_args.fp8_te_current_use_power_2_scales
                or training_args.fp8_te_block_use_f32_scales
                or training_args.fp8_te_block_backward_override is not None
            )
            if te_overrides:
                raise ValueError(
                    "TransformerEngine recipe options cannot be used with "
                    "--fp8-backend=torchao; configure --fp8-torchao-recipe "
                    "and TorchAO-specific options instead."
                )
            if (
                training_args.fp8_torchao_fsdp_float8_all_gather
                and training_args.distributed_strategy != "fsdp"
            ):
                raise ValueError(
                    "--fp8-torchao-fsdp-float8-all-gather requires "
                    "--distributed-strategy=fsdp."
                )
            if (
                training_args.fp8_torchao_fsdp_float8_all_gather
                and training_args.fp8_torchao_recipe != "tensorwise"
            ):
                raise ValueError(
                    "TorchAO FP8 FSDP all-gather currently supports only "
                    "--fp8-torchao-recipe=tensorwise."
                )
        elif (
            training_args.fp8_torchao_recipe != "tensorwise"
            or not training_args.fp8_torchao_pad_inner_dim
            or training_args.fp8_torchao_fsdp_float8_all_gather
        ):
            raise ValueError(
                "TorchAO-specific FP8 options require --fp8-backend=torchao."
            )
