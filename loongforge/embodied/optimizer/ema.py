# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exponential Moving Average (EMA) of model weights.

LoongForge has no EMA support prior to this file; ported from
``giga_train/strategies/ema.py`` semantics (``giga-train`` — see
``Trainer.set_ema_models`` / ``Trainer.backward_step``'s EMA update call) so
GigaBrain-0's finetune run — which trains ``with_ema: true`` in the
reference config — can be reproduced. The reference checkpoint
``GigaBrain-0.1-3.5B-Base`` itself was produced by pretraining under this
scheme (its ``_name_or_path`` in ``config.json`` ends in ``.../model_ema``),
so this is not merely a logging nicety but part of the reference weight
lineage.

Update rule (standard EMA, applied once per optimizer step — i.e. only when
gradients are synced, not per gradient-accumulation micro-step):

    ema_param <- decay * ema_param + (1 - decay) * param
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EMAModel:
    """Maintains an exponential moving average of a model's parameters/buffers.

    Works with both plain ``nn.Module`` and FSDP2-wrapped models: callers pass
    a full (unsharded) ``state_dict`` — for FSDP2 this must already be
    gathered via ``torch.distributed.checkpoint.state_dict.get_model_state_dict``
    with ``full_state_dict=True`` (mirroring how
    ``giga_train.Trainer.backward_step`` gathers FSDP2 state before calling
    ``ema_model.step(state_dict)``). This class itself is distributed-agnostic:
    it only holds/updates a CPU or device dict of tensors.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor], decay: float = 0.9999, device: Optional[torch.device] = None):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().to(device=device) if device is not None else v.detach().clone()
            for k, v in state_dict.items()
        }

    @torch.no_grad()
    def step(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Update the shadow (EMA) weights from the current model ``state_dict``.

        Non-floating-point tensors (e.g. int buffers, batch-norm counters)
        are copied verbatim rather than averaged, matching common EMA
        implementations (averaging integer counters is not meaningful).
        """
        for key, param in state_dict.items():
            shadow_param = self.shadow.get(key)
            if shadow_param is None:
                # New parameter (e.g. added after construction) — adopt as-is.
                self.shadow[key] = param.detach().clone()
                continue
            if not torch.is_floating_point(param):
                shadow_param.copy_(param)
                continue
            param = param.detach().to(dtype=shadow_param.dtype, device=shadow_param.device)
            shadow_param.mul_(self.decay).add_(param, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the current EMA weights (for saving / applying to a model)."""
        return self.shadow

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Restore EMA weights (for resuming training)."""
        self.shadow = {k: v.detach().clone() for k, v in state_dict.items()}


def build_ema_model(model: nn.Module, decay: float, full_state_dict: Dict[str, torch.Tensor]) -> EMAModel:
    """Construct an :class:`EMAModel` initialized from ``full_state_dict``.

    ``full_state_dict`` must already be the model's *unsharded* state dict
    (see ``distributed/checkpoint.py``'s use of
    ``get_model_state_dict(..., options=StateDictOptions(full_state_dict=True, ...))``
    for the FSDP2 case), matching giga_train's EMA update, which always
    operates on a full (non-sharded) state dict.
    """
    del model  # kept in the signature for symmetry with other `build_*` factories
    logger.info("Initializing EMA model (decay=%.5f, %d tensors)", decay, len(full_state_dict))
    return EMAModel(full_state_dict, decay=decay)
