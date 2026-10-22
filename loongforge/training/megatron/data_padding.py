# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Sequence padding factors derived from backend parallelism and precision.

The data layer must not read TP/CP/SP/FP8 state. The training method computes the
factors here and passes the results into the collators.
"""

from dataclasses import dataclass
from math import gcd
from typing import Optional

from megatron.core.models.multimodal import context_parallel


def llm_pad_to_multiple_of(args) -> int:
    """Sequence multiple an LLM SFT batch is padded to before the model sees it."""
    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        return args.chunksize
    factor = args.tensor_model_parallel_size if args.sequence_parallel else 1
    if args.context_parallel_size > 1:
        factor *= 2 * args.context_parallel_size
    # TransformerEngine FP8 GEMMs want a multiple of 128.
    # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L151
    if args.fp8:
        factor *= 128
    return factor


@dataclass(frozen=True)
class VlmPackedPadding:
    """How packed VLM sequences are padded for the run's parallelism and precision.

    Each segment is padded so CP/SP can split it; the packed total is then padded to
    ``total_multiple``. Frozen and picklable so the collator survives DataLoader workers.
    """

    tp_size: int
    cp_size: int
    sequence_parallel: bool
    fp8_enabled: bool
    fp8_recipe: Optional[str]

    def segment_padding(self, length: int) -> int:
        """Tokens to append to one segment of ``length`` tokens."""
        return context_parallel.get_padding(length, self.cp_size, self.tp_size, self.sequence_parallel)

    @property
    def total_multiple(self) -> int:
        """Multiple the whole packed sequence is padded to."""
        if self.sequence_parallel and self.cp_size > 1:
            factor = self.tp_size * self.cp_size * 2
        elif self.cp_size > 1:
            factor = self.cp_size * 2
        elif self.sequence_parallel:
            factor = self.tp_size
        else:
            factor = 1
        if self.fp8_enabled:
            # TransformerEngine FP8 GEMM alignment per recipe.
            fp8_factor = {"mxfp8": 32, "blockwise": 128}.get(self.fp8_recipe, 16)
            if self.sequence_parallel:
                fp8_factor *= self.tp_size
            factor = factor * fp8_factor // gcd(factor, fp8_factor)
        return factor


def vlm_packed_padding(args) -> Optional[VlmPackedPadding]:
    """Padding for packed VLM batches, or ``None`` when the run needs no alignment.

    Alignment is needed under CP, SP, or blockwise FP8.
    """
    fp8_enabled = bool(getattr(args, "fp8", None))
    fp8_recipe = getattr(args, "fp8_recipe", None)
    if not (
        args.context_parallel_size > 1
        or args.sequence_parallel
        or (fp8_enabled and fp8_recipe == "blockwise")
    ):
        return None
    return VlmPackedPadding(
        tp_size=args.tensor_model_parallel_size,
        cp_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,
        fp8_enabled=fp8_enabled,
        fp8_recipe=fp8_recipe,
    )
