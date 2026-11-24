# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 hybrid pattern and native MCore stack specification."""

from typing import Iterable, List, Sequence, Union

from omegaconf import ListConfig

from megatron.core.models.hybrid.hybrid_layer_specs import (
    hybrid_dsv4_stack_spec as deepseek_v4_hybrid_stack_spec,
)


_COMPRESS_RATIO_SYMBOLS = {0: "W", 4: "C", 128: "H"}


def _decoder_layer_is_moe(
    layer_id: int,
    num_layers: int,
    moe_layer_freq: Union[int, Sequence[int], None],
) -> bool:
    """Return whether a decoder layer owns a MoE FFN."""
    if moe_layer_freq is None:
        moe_layer_freq = 1
    if isinstance(moe_layer_freq, (list, tuple, ListConfig)):
        if len(moe_layer_freq) != num_layers:
            raise ValueError(
                f"moe_layer_freq length ({len(moe_layer_freq)}) must equal "
                f"num_layers ({num_layers})."
            )
        value = moe_layer_freq[layer_id]
        if value not in (0, 1, False, True):
            raise ValueError(
                "DeepSeek-V4 hybrid patterns require binary moe_layer_freq entries."
            )
        return bool(value)
    if not isinstance(moe_layer_freq, int) or moe_layer_freq <= 0:
        raise ValueError("moe_layer_freq must be a positive integer or a binary layer list.")
    return layer_id % moe_layer_freq == 0


def _ratio_symbol(ratio: int) -> str:
    try:
        return _COMPRESS_RATIO_SYMBOLS[ratio]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DeepSeek-V4 compress ratio {ratio}; "
            f"expected one of {sorted(_COMPRESS_RATIO_SYMBOLS)}."
        ) from exc


def build_deepseek_v4_hybrid_layer_pattern(
    csa_compress_ratios: Iterable[int],
    num_layers: int,
    mtp_num_layers: int = 0,
    moe_layer_freq: Union[int, Sequence[int], None] = 1,
) -> str:
    """Build the unified HybridModel pattern for DeepSeek-V4.

    Each logical decoder layer is represented by an attention physical layer and
    an FFN physical layer. The final walker/select symbols become ``W``, ``C`` or
    ``H``; MoE and dense FFNs use ``E`` and ``-`` respectively.
    """
    ratios: List[int] = list(csa_compress_ratios)
    expected_length = num_layers + mtp_num_layers
    if len(ratios) != expected_length:
        raise ValueError(
            f"csa_compress_ratios length ({len(ratios)}) must equal "
            f"num_layers + mtp_num_layers ({expected_length})."
        )

    decoder_parts: List[str] = []
    for layer_id in range(num_layers):
        decoder_parts.append(_ratio_symbol(ratios[layer_id]))
        decoder_parts.append(
            "E" if _decoder_layer_is_moe(layer_id, num_layers, moe_layer_freq) else "-"
        )

    pattern = "".join(decoder_parts)
    if mtp_num_layers <= 0:
        return pattern

    mtp_ratios = ratios[num_layers:]
    if len(mtp_ratios) != mtp_num_layers:
        raise ValueError("MTP compress-ratio count does not match mtp_num_layers.")

    # MCore's unified MTP syntax repeats one pattern per prediction depth.
    mtp_symbol = _ratio_symbol(mtp_ratios[0])
    if any(_ratio_symbol(ratio) != mtp_symbol for ratio in mtp_ratios[1:]):
        raise ValueError(
            "MCore Hybrid MTP requires the same attention pattern at every MTP depth."
        )
    mtp_pattern = f"{mtp_symbol}E"
    return pattern + "/" + "/".join([mtp_pattern] * mtp_num_layers)


def expand_deepseek_v4_hybrid_compress_ratios(
    pattern: str,
    csa_compress_ratios: Iterable[int],
) -> List[int]:
    """Expand logical V4 attention ratios to one entry per physical Hybrid layer."""
    ratios = iter(csa_compress_ratios)
    expanded: List[int] = []
    for symbol in pattern:
        if symbol == "/":
            continue
        if symbol in _COMPRESS_RATIO_SYMBOLS.values():
            try:
                expanded.append(next(ratios))
            except StopIteration as exc:
                raise ValueError(
                    "csa_compress_ratios does not cover every attention symbol "
                    f"in hybrid pattern {pattern!r}."
                ) from exc
        else:
            expanded.append(0)
    try:
        next(ratios)
    except StopIteration:
        return expanded
    raise ValueError(
        f"csa_compress_ratios contains unused entries for hybrid pattern {pattern!r}."
    )


def get_deepseek_v4_hybrid_stack_spec(config=None, vp_stage=None):
    """Return MCore's native DeepSeek-V4 HybridStack specification."""
    return deepseek_v4_hybrid_stack_spec
