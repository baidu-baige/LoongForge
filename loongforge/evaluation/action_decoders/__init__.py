# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Eval-side ActionDecoder component.

Third core eval component (alongside per-benchmark ``adapters/`` and per-model
``payload_builders/``): converts a model's raw action chunk (its own encoding,
e.g. ee6d ``pos+rot6d+grip``) into the action space a benchmark env expects.

Single uniform interface — every decoder is an ``ActionDecoder`` subclass with
``__call__(actions, ctx) -> env_actions`` and an optional ``reset()`` (stateless
decoders inherit the base no-op). Concrete decoders are grouped by **source
encoding**:

* ``ee6d.py``  — models that output ee6d actions (LIBERO/CALVIN/SimplerEnv/
  ManiSkill + RoboTwin dual-arm)
* ``joint.py`` — models that output joint(-delta) actions (pi05 RoboTwin)

``build_action_decoder(key)`` returns an instance (empty key -> IdentityDecoder).
Kept separate from ``servers/predict_action_interface.py`` (the model author's
``predict_action`` contract): action decoding is an eval-side concern.
"""

from loongforge.evaluation.action_decoders.base import (
    ACTION_DECODER_REGISTRY,
    ActionDecoder,
    FunctionDecoder,
    IdentityDecoder,
    build_action_decoder,
    is_action_decoder_registered,
    register_action_decoder,
    register_action_fn,
)
from loongforge.evaluation.action_decoders.rotation import rot6d_interleaved_to_quat

__all__ = [
    "ActionDecoder",
    "FunctionDecoder",
    "IdentityDecoder",
    "ACTION_DECODER_REGISTRY",
    "build_action_decoder",
    "is_action_decoder_registered",
    "register_action_decoder",
    "register_action_fn",
    "rot6d_interleaved_to_quat",
]
