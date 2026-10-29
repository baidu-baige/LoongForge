# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-wise distributed optimizer, driven by a tensor_wise trainer
(``--trainer-type LingbotVlaV2Trainer``) on ``--distributed-strategy ddp``.

Compute replicas stay complete on every rank and no tensor is split: each
parameter is assigned whole to one owner, gradients are reduced to that owner,
and the owner publishes the updated parameter back. The fp32 masters and the
optimizer state that follows them live only on the owner. How this differs from
ZeRO-1 and from Megatron's ``LayerWiseDistributedOptimizer`` is in
``tensor_wise_optimizer.py``.

    config.py                 ``TensorWiseConfig``, the collective precision and
                              overlap knobs a model config holds as its
                              ``tensor_wise`` field
    precision_policy.py       the per-parameter precision contract a model
                              implements and the strategy consumes
    validation.py             startup checks of the optimizer's option combinations
    parameter_plan.py         ownership, capabilities, fp32 masters, checkpoint manifest
    gradient_reducer.py       reduce-to-owner, plus the bucket/buffer layout both paths share
    parameter_sync.py         owner-to-replica publication, including the fp8 delta path
    tensor_wise_optimizer.py  wires the above into one optimizer object

The inner Muon + AdamW optimizer built over the masters comes from
``optimizer/muon_builder.py``, which plain DDP runs share.

Import the modules directly; there are deliberately no re-exports.
"""
