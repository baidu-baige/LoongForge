# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collective-precision and overlap knobs for ``ReplicatedShardedManager``.

Inherit this frozen dataclass in a model's ``ModelConfig`` to get the fields and
their defaults; the manager reads every field directly:

    @dataclass(frozen=True)
    class MyModelConfig(ReplicatedShardedConfig):
        ...

Downcasting is per parameter: the parameter policy keeps precision-critical
parameters in fp32 regardless of these settings.
"""

from dataclasses import dataclass, field
from typing import List

# Single source of the defaults; the manager, parameter plan and synchronizer import
# these names instead of repeating the literals.
DEFAULT_GRAD_REDUCE_DTYPE = "bf16"
DEFAULT_PARAM_SYNC_PRECISION = "fp32"
DEFAULT_FP8_BLOCK = 256


@dataclass(frozen=True)
class ReplicatedShardedConfig:
    """Collective-precision and overlap knobs read by the manager."""

    # ── Collective precision ──
    # fp32 | bf16 | mixed | compute. bf16 is faster for a small loss deviation, but
    # single-step grad norm shifts by several percent: use fp32 when bisecting numerics.
    grad_reduce_dtype: str = DEFAULT_GRAD_REDUCE_DTYPE
    #
    # How the fp32 masters are published to the replicas (critical fp32 tensors and
    # bf16-compute towers are unaffected):
    #   fp32           -- publish masters as-is.
    #   bf16           -- round the state to bf16; fast but biased, so not the default.
    #   bf16_ef        -- bf16 plus an owner-local fp32 residual folded into the next
    #                     publish (error feedback).
    #   bf16_ef_delta  -- broadcast the bf16 delta since the last publish and
    #                     accumulate it into the fp32 replica.
    #   fp8_e4m3_delta -- broadcast (master - replica) as E4M3 with one fp32 scale
    #                     per block; fastest, needs compute capability >= 8.9, so
    #                     it is opt-in.
    # Override per run, e.g. model.param_sync_precision=bf16.
    param_sync_precision: str = DEFAULT_PARAM_SYNC_PRECISION
    # fp8_e4m3_delta only: elements per scale; reprime_interval>0 re-anchors replicas
    # to a full BF16 copy every N publishes (0 = pure delta stream).
    param_sync_fp8_block: int = DEFAULT_FP8_BLOCK
    param_sync_fp8_reprime_interval: int = 0
    # fp8_e4m3_delta only: name substrings selecting which eligible parameters are
    # quantized. Empty means none are, e.g. model.param_sync_fp8_include='[self_attn]'.
    param_sync_fp8_include: List[str] = field(default_factory=list)
    # fp8_e4m3_delta only: also quantize bf16-compute (VLM) parameters. A bf16 replica
    # cannot accumulate small deltas, so each rank keeps an fp32 shadow (+4 bytes per
    # eligible parameter); scope it with param_sync_fp8_include.
    #
    # Validated on lingbot_vla_v2 (8xA800, 20 steps): loss/grad_norm identical to the
    # baseline, step time 1.64s vs 1.82s. Opt in per run:
    #   model.param_sync_precision=fp8_e4m3_delta
    #   model.param_sync_bf16_with_fp8=true
    #   model.param_sync_fp8_include='[layers.,blocks.]'
    param_sync_bf16_with_fp8: bool = False

    # ── Collective overlap ──
    # Declared here, not read from the environment, so runs outside the launch
    # script get the same defaults.
    #   grad_overlap:     reduce each gradient bucket as soon as backward produces it.
    #   param_overlap:    broadcast each owner's master as soon as it is updated.
    #   comm_bucket_mb:   bucket size for both; 0 picks by grad_overlap (256 on, 1024 off).
    #   grad_inflight_mb: cap on in-flight gradient collectives; a memory bound, so
    #                     lower it on cards with less memory.
    grad_overlap: bool = True
    param_overlap: bool = True
    comm_bucket_mb: int = 256
    grad_inflight_mb: int = 3072


__all__ = [
    "DEFAULT_FP8_BLOCK",
    "DEFAULT_GRAD_REDUCE_DTYPE",
    "DEFAULT_PARAM_SYNC_PRECISION",
    "ReplicatedShardedConfig",
]
