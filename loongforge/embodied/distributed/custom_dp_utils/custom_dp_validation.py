# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Startup validation of the custom_dp (custom data parallel) strategy.

Called from ``train/validators.py``; every custom_dp-specific rule lives here so the
generic validator carries a single call.
"""

# custom_dp wraps the model itself instead of using DDP, so any explicitly set DDP knob
# would be silently ignored. Only knobs whose default is falsy are listed, i.e. a
# non-default value means the user asked for it.
_DDP_ONLY_ARGS = (
    "ddp_comm_hook",
    "ddp_comm_hook_logging",
    "ddp_bucket_cap_mb",
    "ddp_bucket_cap_mb_list",
    "ddp_static_graph",
    "ddp_gradient_as_bucket_view",
    "ddp_skip_all_reduce_unused_params",
    "ddp_batched_grad_copy",
)


def validate_custom_dp_args(training_args) -> None:
    """Reject option combinations custom_dp cannot honour; no-op for other strategies
    except that ``--optimizer muon`` needs custom_dp."""
    optimizer = training_args.optimizer.lower()
    if training_args.distributed_strategy != "custom_dp":
        # The Muon optimizer is built by the custom_dp trainer, not by
        # OPTIMIZER_REGISTRY, so without this check a wrong strategy surfaces as
        # "Unknown optimizer 'muon'" from build_optimizer, which points nowhere.
        if optimizer == "muon":
            raise ValueError("--optimizer=muon requires --distributed-strategy=custom_dp.")
        return

    # --zero-optimizer is what turns the mode on: the fp32 masters live on each
    # tensor's owner, so the DDP-side ZeRO knobs do not apply.
    if not training_args.zero_optimizer:
        raise ValueError("--distributed-strategy=custom_dp requires --zero-optimizer.")
    if training_args.zero_parameters_as_bucket_view:
        raise ValueError(
            "--zero-parameters-as-bucket-view is not supported by "
            "--distributed-strategy=custom_dp."
        )
    set_ddp_only = [
        name for name in _DDP_ONLY_ARGS
        if getattr(training_args, name) not in (None, False)
    ]
    if set_ddp_only:
        raise ValueError(
            "--distributed-strategy=custom_dp does not use DDP; remove: "
            + ", ".join("--" + name.replace("_", "-") for name in set_ddp_only)
        )
    if training_args.fp8:
        raise ValueError(
            "--fp8 is not supported with --distributed-strategy=custom_dp: custom_dp "
            "replaces wrap_model, which is where FP8 conversion happens."
        )
    # The rank-local masters only survive in the dcp layout. The trainer refuses
    # this at save time; failing here avoids finding out hours in.
    if (
        training_args.save_interval
        and training_args.save_training_state
        and (training_args.use_lora or training_args.save_format != "dcp")
    ):
        raise ValueError(
            "--distributed-strategy=custom_dp training-state checkpoints require "
            "--save-format=dcp and no --use-lora; pass "
            "--no-save-training-state to export weights only."
        )
