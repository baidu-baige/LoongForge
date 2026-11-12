# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch loss-mask / flow-horizon helpers used by the giga_brain07 training
forward.

Extracted verbatim from ``giga_brain_0_trainer`` so the route-B trainer can
import just these two functions without pulling the standalone giga Trainer (and
the giga_train framework behind it). No behavior change -- the function bodies are
identical to the originals.
"""

from __future__ import annotations

from typing import Any

import torch


def _apply_skip_loss_mask(
    skip_loss_mask: torch.Tensor | None,
    lang_loss_masks: torch.Tensor,
    action_loss_mask: torch.Tensor,
    action_dim_loss_mask: torch.Tensor | None,
    traj_loss_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if skip_loss_mask is None:
        return lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask

    skip_loss_mask = skip_loss_mask.to(device=lang_loss_masks.device, dtype=torch.bool).reshape(-1)
    if not skip_loss_mask.any():
        return lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask

    keep_loss_mask = ~skip_loss_mask
    lang_loss_masks = lang_loss_masks & keep_loss_mask[:, None]
    action_loss_mask = action_loss_mask & keep_loss_mask[:, None]

    if action_dim_loss_mask is not None:
        action_dim_loss_mask = action_dim_loss_mask & keep_loss_mask[:, None, None].to(
            device=action_dim_loss_mask.device
        )

    if traj_loss_mask is not None:
        expand_shape = [skip_loss_mask.shape[0]] + [1] * (traj_loss_mask.ndim - 1)
        traj_keep_loss_mask = keep_loss_mask.reshape(expand_shape).to(device=traj_loss_mask.device)
        traj_loss_mask = traj_loss_mask & traj_keep_loss_mask

    return lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask


def _get_flow_action_horizon(model: Any) -> int | None:
    for candidate in (model, getattr(model, 'module', None), getattr(model, '_orig_mod', None)):
        if candidate is None:
            continue
        flow_action_horizon = getattr(candidate, 'flow_action_horizon', None)
        if flow_action_horizon is not None:
            return int(flow_action_horizon)
        config = getattr(candidate, 'config', None)
        if isinstance(config, dict):
            flow_action_horizon = config.get('flow_action_horizon')
        else:
            flow_action_horizon = getattr(config, 'flow_action_horizon', None)
        if flow_action_horizon is not None:
            return int(flow_action_horizon)
    return None
