# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Action normalization helpers for Wall-OSS-0.5.

The model holds no normalization state. Training consumes actions that the
dataloader collator already mapped to ``[-1, 1]`` with q01/q99 bounds, and
inference receives the statistics explicitly through the shared eval contract
(``predict_action(..., dataset_stats=...)`` -> ``dataset_stats["action"]``),
mirroring pi05.

The stats mapping is the flat LeRobot form: ``{"q01": [...], "q99": [...]}``.
Its width defines the real action dims; anything past it is the virtual tail
that the collator zero-pads up to the configured ``action_dim``, so these
helpers zero the tail as well.
"""
import logging

import torch

logger = logging.getLogger(__name__)


def print_rank_last(message):
    """If distributed is initialized, log only on last rank."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1):
            logger.info(message)
    else:
        logger.info(message)


def _q99_bounds(action_stats, values):
    """Return ``(q01, delta, width)`` aligned to ``values`` dtype/device."""
    if "q01" not in action_stats or "q99" not in action_stats:
        raise ValueError(
            "Wall-OSS-0.5 action (un)normalization requires action stats with "
            f"'q01' and 'q99'; got keys {sorted(action_stats)}"
        )
    q01 = torch.as_tensor(
        action_stats["q01"], dtype=values.dtype, device=values.device
    ).flatten()
    q99 = torch.as_tensor(
        action_stats["q99"], dtype=values.dtype, device=values.device
    ).flatten()
    if q01.shape[-1] != q99.shape[-1]:
        raise ValueError(
            f"action stats q01 dim ({q01.shape[-1]}) != q99 dim ({q99.shape[-1]})"
        )
    width = min(int(q01.shape[-1]), int(values.shape[-1]))
    return q01[:width], q99[:width] - q01[:width], width


def normalize_actions_q99(values, action_stats):
    """Map physical actions to ``[-1, 1]``, zeroing the virtual tail dims.

    Raises when ``action_stats`` is missing: the normalized image of a physical
    action cannot be guessed, and silently using the wrong one would distort the
    flow-matching target for masked dims.
    """
    if action_stats is None:
        raise ValueError(
            "Wall-OSS-0.5 needs action stats (dataset_stats['action']) to "
            "normalize the padding action for masked dims; none were provided."
        )
    q01, delta, width = _q99_bounds(action_stats, values)
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    normalized = torch.clamp((values[..., :width] - q01) / delta * 2 - 1, -1, 1)
    if width == values.shape[-1]:
        return normalized
    out = torch.zeros_like(values)
    out[..., :width] = normalized
    return out


def unnormalize_actions_q99(values, action_stats):
    """Map ``[-1, 1]`` actions back to physical units, zeroing the virtual tail.

    ``action_stats is None`` returns ``values`` untouched so callers that opt out
    (server warmup, debugging) keep the raw normalized output, matching
    ``pi05._q99_unnormalize_actions``.
    """
    if action_stats is None:
        return values
    q01, delta, width = _q99_bounds(action_stats, values)
    torch.cuda.nvtx.range_push("unnormalize_actions_q99")
    unnormalized = (values[..., :width] + 1) / 2 * delta + q01
    torch.cuda.nvtx.range_pop()
    if width == values.shape[-1]:
        return unnormalized
    out = torch.zeros_like(values)
    out[..., :width] = unnormalized
    return out
