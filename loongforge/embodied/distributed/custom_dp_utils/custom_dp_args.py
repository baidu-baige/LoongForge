# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CLI arguments of the custom_dp (custom data parallel) strategy.

``CustomDpArgs`` is mixed into ``TrainingArgs`` so the ``--custom-dp-*`` flags sit next to the
generic ones, but every field, default and help string is owned here.
"""

from dataclasses import dataclass, field
from typing import List

from loongforge.embodied.distributed.custom_dp_utils.custom_dp_config import CustomDpConfig
from loongforge.embodied.distributed.custom_dp_utils.precision_policy import (
    GRAD_REDUCE_MODES,
    PARAM_SYNC_PRECISIONS,
)


@dataclass(frozen=True)
class CustomDpArgs:
    """Collective knobs for ``--distributed-strategy custom_dp``.

    Used only when the model config does not inherit ``CustomDpConfig``; a model that
    does keeps reading them from its YAML ``model:`` section. Defaults come from
    ``CustomDpConfig`` so there is a single source for them.
    """

    custom_dp_grad_reduce_dtype: str = field(
        default=CustomDpConfig.grad_reduce_dtype,
        metadata={
            "choices": list(GRAD_REDUCE_MODES),
            "help": "custom_dp gradient-reduce precision (fp32, bf16, mixed, compute). "
                    "Parameters the policy marks precision-critical stay fp32.",
        },
    )
    custom_dp_param_sync_precision: str = field(
        default=CustomDpConfig.param_sync_precision,
        metadata={
            "choices": list(PARAM_SYNC_PRECISIONS),
            "help": "How the fp32 masters are published to the replicas.",
        },
    )
    custom_dp_grad_overlap: bool = field(
        default=CustomDpConfig.grad_overlap,
        metadata={
            "help": "Reduce each gradient bucket as soon as backward produces it."
        },
    )
    custom_dp_param_overlap: bool = field(
        default=CustomDpConfig.param_overlap,
        metadata={
            "help": "Broadcast each owner's parameters as soon as they are updated."
        },
    )
    custom_dp_comm_bucket_mb: int = field(
        default=CustomDpConfig.comm_bucket_mb,
        metadata={
            "help": "custom_dp communication bucket size in MiB; 0 picks by "
                    "--custom-dp-grad-overlap (256 on, 1024 off)."
        },
    )
    custom_dp_grad_inflight_mb: int = field(
        default=CustomDpConfig.grad_inflight_mb,
        metadata={
            "help": "Cap on in-flight gradient collectives in MiB; a memory "
                    "bound, so lower it on cards with less memory."
        },
    )
    custom_dp_param_sync_fp8_reprime_interval: int = field(
        default=CustomDpConfig.param_sync_fp8_reprime_interval,
        metadata={
            "help": "fp8_e4m3_delta only: re-anchor replicas to a full copy every "
                    "N publishes (0 = pure delta stream)."
        },
    )
    custom_dp_param_sync_fp8_include: List[str] = field(
        default_factory=list,
        metadata={
            "help": "fp8_e4m3_delta only: parameter-name substrings selecting "
                    "which eligible parameters are quantized; empty quantizes none.",
        },
    )
    custom_dp_param_sync_bf16_with_fp8: bool = field(
        default=CustomDpConfig.param_sync_bf16_with_fp8,
        metadata={
            "help": "fp8_e4m3_delta only: also quantize bf16-compute parameters "
                    "(keeps an fp32 shadow, +4 bytes per eligible parameter)."
        },
    )
