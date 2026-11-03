# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training recipe registration."""

from typing import Union, List, Callable
import logging

MODEL_FAMILY_TRAINER_FACTORY = {}
_TRAINERS_LOADED = False
_TRAINERS_LOADING = False
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def register_model_trainer(
    model_family: Union[str, List[str]],
    training_phase: str,
    training_func: Callable = None,
    override: bool = False,
):
    """
    register model training function

    Args:
        model_family: need to be consistent with the models.factory definition, otherwise it
                      cannot be retrieved correctly. (Case-insensitive)

        training_phase: need to be consistent with the --training-phase definition in training.megatron_args
        trainig_func: training function.
        override: Whether to overwrite existing training functions for the same model/phase.
                  Default: False (no overwrite).
    """
    _load_builtin_trainers()

    def _add_trainer(families, phase, func, override):
        if not isinstance(families, list):
            families = [families]

        for _family in families:
            _family = _family.lower()
            if _family not in MODEL_FAMILY_TRAINER_FACTORY:
                MODEL_FAMILY_TRAINER_FACTORY[_family] = {}

            if phase in MODEL_FAMILY_TRAINER_FACTORY[_family]:
                if not override:
                    continue
                else:
                    logger.info(f"Overriding existing trainer ({_family} family, {phase} phase)")

            MODEL_FAMILY_TRAINER_FACTORY[_family][phase] = func

    def _register_function(fn):
        _add_trainer(model_family, training_phase, fn, override)
        return fn

    if training_func is not None:
        return _add_trainer(model_family, training_phase, training_func, override)
    else:
        return _register_function


def _load_builtin_trainers():
    """Load defaults before custom registrations, allowing recipe import cycles."""
    global _TRAINERS_LOADED, _TRAINERS_LOADING
    if _TRAINERS_LOADED or _TRAINERS_LOADING:
        return

    _TRAINERS_LOADING = True
    try:
        from loongforge.training.llm import pretrain_llm  # noqa: F401
        from loongforge.training.vlm import pretrain_vlm  # noqa: F401
        from loongforge.training.llm import sft_llm  # noqa: F401
        from loongforge.training.vlm import sft_vlm, sft_internvl, sft_ernie  # noqa: F401
        from loongforge.training.diffusion import pretrain_wan, pretrain_qwen_image  # noqa: F401

        _TRAINERS_LOADED = True
    finally:
        _TRAINERS_LOADING = False
