# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collective-precision and overlap knobs for ``TensorWiseOptimizer``.

A model makes them tunable from its YAML by holding one ``tensor_wise`` field:

    @dataclass(frozen=True)
    class MyModelConfig:
        ...
        tensor_wise: TensorWiseConfig = field(default_factory=TensorWiseConfig)

and then sets ``model.tensor_wise.<knob>`` in the YAML or on the command line. A
model without the field runs with these defaults.

Not frozen on purpose: OmegaConf marks a nested frozen dataclass read-only, so
YAML values could not be merged into it. The enclosing model config is frozen.

Downcasting is per parameter: the parameter policy keeps precision-critical
parameters in fp32 regardless of these settings.
"""

from dataclasses import dataclass, field
from typing import List

# Single source of the defaults; the optimizer, parameter plan and synchronizer import
# these names instead of repeating the literals.
DEFAULT_GRAD_REDUCE_DTYPE = "bf16"
DEFAULT_PARAM_SYNC_PRECISION = "fp32"
# Elements per fp32 scale in the fp8_e4m3_delta wire format. TransformerEngine's
# blockwise quantizer fixes it at 128, so this is a codec constant, not a knob.
FP8_BLOCK = 128


@dataclass
class TensorWiseConfig:
    """Collective-precision and overlap knobs read by the optimizer."""

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
    # Override per run, e.g. model.tensor_wise.param_sync_precision=bf16.
    param_sync_precision: str = DEFAULT_PARAM_SYNC_PRECISION
    # fp8_e4m3_delta only: reprime_interval>0 re-anchors replicas to a full BF16 copy
    # every N publishes (0 = pure delta stream). The scale block is fixed at 128 by
    # TransformerEngine (FP8_BLOCK), so it is not configurable.
    param_sync_fp8_reprime_interval: int = 0
    # fp8_e4m3_delta only: name substrings selecting which eligible parameters are
    # quantized. Empty means none are, e.g. model.tensor_wise.param_sync_fp8_include='[self_attn]'.
    param_sync_fp8_include: List[str] = field(default_factory=list)
    # fp8_e4m3_delta only: also quantize bf16-compute (VLM) parameters. A bf16 replica
    # cannot accumulate small deltas, so each rank keeps an fp32 shadow (+4 bytes per
    # eligible parameter); scope it with param_sync_fp8_include.
    #
    # Validated on lingbot_vla_v2 (8xA800, 20 steps): loss/grad_norm identical to the
    # baseline, step time 1.64s vs 1.82s. Opt in per run:
    #   model.tensor_wise.param_sync_precision=fp8_e4m3_delta
    #   model.tensor_wise.param_sync_bf16_with_fp8=true
    #   model.tensor_wise.param_sync_fp8_include='[layers.,blocks.]'
    param_sync_bf16_with_fp8: bool = False

    # ── Collective overlap ──
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
    "FP8_BLOCK",
    "DEFAULT_GRAD_REDUCE_DTYPE",
    "DEFAULT_PARAM_SYNC_PRECISION",
    "TensorWiseConfig",
]
