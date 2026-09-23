# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared config fields for the replicated-sharded manager.

A model that trains under ``ReplicatedShardedManager`` supplies these
collective-precision and overlap knobs. Inherit this frozen dataclass into the
model's ``ModelConfig`` so the fields, their defaults and their rationale come
for free instead of being copied per model:

    @dataclass(frozen=True)
    class MyModelConfig(ReplicatedShardedConfig):
        ...

The manager reads every field by direct attribute access, so a model that
forgets one would fail at manager construction; inheriting removes that footgun.
Downcasting is per parameter (the parameter policy exempts the precision-critical
ones), never per collective.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ReplicatedShardedConfig:
    """Collective-precision and overlap knobs read by the manager."""

    # ── Collective precision ──
    # ``grad_reduce_dtype``: fp32 | bf16 | mixed | compute. The largest collective
    # in the step; bf16 is 1.229x end-to-end (1574.0 -> 1280.9 ms) for 0.792% max
    # loss deviation over 20 iterations. Single-step grad norm moves up to 8%, so
    # switch back to fp32 when bisecting numerics.
    grad_reduce_dtype: str = "bf16"
    #
    # ``param_sync_precision``: how the fp32 masters are published to the replicas
    # -- one mutually exclusive enum. bf16-compute towers and the critical fp32
    # tensors are unaffected. Values:
    #   fp32           -- publish the fp32 masters verbatim (no downcast).
    #   bf16           -- round to bf16 on the wire; a further 1.076x over fp32, but
    #                     rejected as a default: it rounds the state (not just the
    #                     increment, ~1e4 larger error) and drives the fp32-compute
    #                     params into a near-identical dead zone step to step.
    #   bf16_ef        -- bf16 with an owner-local fp32 residual folded into the next
    #                     publish, turning the persistent bias into zero-mean dither.
    #   bf16_ef_delta  -- broadcast the bf16 delta since the last publish and
    #                     accumulate it into the fp32 replica, which carries the
    #                     residual implicitly. Rounds the small per-step delta.
    #   fp8_e4m3_delta -- publish quantized (master - replica) E4M3 deltas with one
    #                     fp32 scale per block; replicas add the reconstructed delta.
    #                     ~1.10x on the broadcast path; the delta is re-derived
    #                     against the true fp32 master each step, so its
    #                     reconstruction error does not accumulate. Needs an
    #                     fp8-capable GPU (compute capability >= 8.9), so the library
    #                     default stays fp32 and production opts in via the YAML.
    # Override per run on the command line, e.g. model.param_sync_precision=bf16.
    param_sync_precision: str = "fp32"
    # Read only when param_sync_precision=fp8_e4m3_delta: block sizes the per-scale
    # group; reprime_interval>0 re-anchors replicas to a full BF16 copy every N
    # publishes (0 keeps a pure delta stream).
    param_sync_fp8_block: int = 256
    param_sync_fp8_reprime_interval: int = 0
    # Read only when param_sync_precision=fp8_e4m3_delta: FP8 whitelist. This is
    # the sole scope and a pure opt-in -- the FP8 delta applies only to eligible
    # (non-critical, fp32-compute action-expert, or bf16-compute under the shadow)
    # parameters whose name contains one of these substrings. Empty means NO
    # parameter is quantized (you must name them); a non-empty list restricts FP8
    # to that subset and everything else keeps its normal publish (bf16 for the
    # VLM, fp32 for the action expert), e.g. model.param_sync_fp8_include='[self_attn]'.
    param_sync_fp8_include: List[str] = field(default_factory=list)
    # Read only when param_sync_precision=fp8_e4m3_delta: also publish the bf16-compute
    # tower (VLM) as an FP8 delta, not just the fp32-compute action expert. A bf16
    # replica is too coarse to accumulate a small FP8 delta (it dead-zones at bf16
    # ULP), so each rank keeps a full fp32 shadow of the eligible bf16 parameters and
    # accumulates the delta there, casting shadow->bf16 into the compute replica for
    # the forward. That shadow costs an extra 4 bytes per eligible parameter resident
    # on every rank, so this is off by default and meant to be scoped with
    # param_sync_fp8_include while its loss/memory tradeoff is measured.
    param_sync_bf16_with_fp8: bool = False

    # ── Collective overlap ──
    # Declared here rather than read from the environment so the launcher cannot
    # disagree with the code about whether overlap is on: a run that does not go
    # through the launch script must get the same defaults as one that does.
    #   grad_overlap : start each gradient bucket's reduce as soon as backward has
    #                  produced it, instead of after the whole backward.
    #   param_overlap: start each owner's broadcast as soon as its master is
    #                  updated, instead of after the whole optimizer step.
    #   comm_bucket_mb: bucket size for both. Overlap wants finer buckets so
    #                  collectives start early; measured 1024 costs 0.94x with
    #                  overlap on. 0 means "pick by grad_overlap" (256 on, 1024 off).
    #   grad_inflight_mb: byte cap on gradient collectives in flight during
    #                  backward. A memory bound, not a throughput dial -- lower it
    #                  on cards with less memory than the validated configuration.
    grad_overlap: bool = True
    param_overlap: bool = True
    comm_bucket_mb: int = 256
    grad_inflight_mb: int = 3072


__all__ = ["ReplicatedShardedConfig"]
