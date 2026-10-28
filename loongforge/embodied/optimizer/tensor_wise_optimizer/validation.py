# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Startup validation of the tensor_wise trainers (``TensorWiseOptimizer``).

Called from ``train/validators.py``; every tensor_wise-specific rule lives here so the
generic validator carries a single call.
"""

# --trainer-type values whose trainer drives TensorWiseOptimizer. Selecting one
# of them is what enables tensor_wise; a new integration adds its trainer here.
TENSOR_WISE_TRAINERS = ("TensorWiseTrainer", "LingbotVlaV2Trainer")

# tensor_wise takes over gradient communication instead of the DDP wrapper, so any
# explicitly set DDP knob would be silently ignored. Only knobs whose default is
# falsy are listed, i.e. a non-default value means the user asked for it.
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


def validate_tensor_wise_args(training_args) -> None:
    """Reject option combinations a tensor_wise trainer cannot honour; other
    trainers are not checked here."""
    trainer = training_args.trainer_type
    if trainer not in TENSOR_WISE_TRAINERS:
        return

    # Compute replicas are complete on every rank, i.e. data parallel; the
    # optimizer replaces the DDP wrapper rather than an FSDP sharding.
    if training_args.distributed_strategy != "ddp":
        raise ValueError(f"--trainer-type {trainer} requires --distributed-strategy=ddp.")
    # Both shard the optimizer state; they are alternatives, not layers.
    if training_args.zero_optimizer:
        raise ValueError(
            f"--trainer-type {trainer} shards the optimizer state with "
            "TensorWiseOptimizer and is incompatible with --zero-optimizer."
        )
    set_ddp_only = [
        name for name in _DDP_ONLY_ARGS
        if getattr(training_args, name) not in (None, False)
    ]
    if set_ddp_only:
        raise ValueError(
            f"--trainer-type {trainer} replaces the DDP wrapper; remove: "
            + ", ".join("--" + name.replace("_", "-") for name in set_ddp_only)
        )
    if training_args.fp8:
        raise ValueError(
            f"--fp8 is not supported with --trainer-type {trainer}: tensor_wise "
            "replaces wrap_model, which is where FP8 conversion happens."
        )
    # The parameter plan copies the masters while the trainer wraps the model,
    # and meta tensors are only materialized after that.
    if training_args.init_on_meta:
        raise ValueError(f"--init-on-meta is not supported with --trainer-type {trainer}.")
    # The rank-local masters only survive in the dcp layout. The trainer refuses
    # this at save time; failing here avoids finding out hours in.
    if (
        training_args.save_interval
        and training_args.save_training_state
        and (training_args.use_lora or training_args.save_format != "dcp")
    ):
        raise ValueError(
            f"--trainer-type {trainer} training-state checkpoints require "
            "--save-format=dcp and no --use-lora; pass "
            "--no-save-training-state to export weights only."
        )
