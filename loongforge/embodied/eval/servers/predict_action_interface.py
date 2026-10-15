# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared predict_action interface helpers for embodied eval model adapters.

This module defines the **model author's** ``predict_action`` contract and the
generic server-side caller. It deliberately contains no eval-side action
decoding logic — that lives in ``loongforge.embodied.eval.action_decoders``
(``ACTION_POSTPROCESS_REGISTRY`` / ``ACTION_DECODER_REGISTRY``).

A model that uses the generic eval policy should implement
``predict_action(images, instructions, state=None, dataset_stats=None)`` with
these responsibilities:

- Accept batched eval images and instructions.
- Accept optional model-ready state from the benchmark adapter.
- Accept optional dataset statistics passed through by eval.
- Apply any model-specific preprocessing needed before inference.
- Apply model-specific state normalization when the model consumes state.
- Apply model-specific action unnormalization when the model emits normalized actions.
- Return an action chunk in the action convention expected by the ActionDecoder.

This module only validates the interface and normalizes output shape. It does
not apply q01/q99, mean/std, min/max, or other model-specific normalization
rules; those belong inside the model's ``predict_action`` implementation.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class PredictActionModel(Protocol):
    """Protocol for models that can be reused by the generic eval policy path."""

    def predict_action(
        self,
        images: Any,
        instructions: Any,
        state: Optional[Any] = None,
        dataset_stats: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Return an adapter-ready action chunk for batched model inputs.

        Implementations may use ``dataset_stats`` for their own normalization or
        unnormalization logic; the generic caller only passes it through.
        """


def validate_predict_action_model(model: Any) -> None:
    """Validate that a model exposes the eval-compatible predict_action method."""
    predict_action = getattr(model, "predict_action", None)
    if not callable(predict_action):
        raise TypeError("model must expose a callable predict_action(images, instructions, state, dataset_stats)")

    signature = inspect.signature(predict_action)
    parameters = signature.parameters
    has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    required = ("images", "instructions")
    optional = ("state", "dataset_stats")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise TypeError(f"model.predict_action is missing required parameters: {missing}")
    unsupported = [name for name in optional if name not in parameters and not has_kwargs]
    if unsupported:
        raise TypeError(f"model.predict_action cannot accept eval keyword parameters: {unsupported}")


def _filter_supported_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the model's predict_action signature cannot accept.

    Emits a WARNING for each dropped kwarg so a misaligned PayloadBuilder
    (emitting fields the model silently ignores) is visible in logs instead
    of resulting in a wrong-but-non-erroring run.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    filtered: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            filtered[k] = v
        else:
            _LOGGER.warning(
                "predict_action dropped unsupported kwarg %r (value type=%s); "
                "PayloadBuilder emits a field the model does not consume.",
                k, type(v).__name__,
            )
    return filtered


def call_predict_action(
    model: PredictActionModel,
    images: Any,
    instructions: Any,
    state: Optional[Any],
    dataset_stats: Optional[Dict[str, Any]],
    action_dim: int,
    **kwargs: Any,
) -> np.ndarray:
    """Call a compatible predict_action implementation and normalize its output shape."""
    validate_predict_action_model(model)
    kwargs = _filter_supported_kwargs(model.predict_action, kwargs)
    result = model.predict_action(
        images=images,
        instructions=instructions,
        state=state,
        dataset_stats=dataset_stats,
        **kwargs,
    )
    actions = np.asarray(result, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3:
        actions = actions.reshape(-1, actions.shape[-1])
    elif actions.ndim != 2:
        raise ValueError(f"model.predict_action returned unsupported action shape: {actions.shape}")

    if actions.shape[-1] < action_dim:
        raise ValueError(
            f"model.predict_action returned action dim {actions.shape[-1]}, expected at least {action_dim}"
        )
    return actions[:, :action_dim]

