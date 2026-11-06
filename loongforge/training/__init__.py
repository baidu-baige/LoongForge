# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Training recipe registration."""

import importlib
from typing import Union, List, Callable
import logging

MODEL_FAMILY_TRAINER_FACTORY = {}
_TRAINERS_LOADED = False
_TRAINERS_LOADING = False
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_TRAINER_CLASSES = {
    "FinetuneTrainer": "loongforge.training.embodied.sft.FinetuneTrainer",
    "TensorWiseTrainer": "loongforge.training.embodied.custom.tensor_wise_trainer.TensorWiseTrainer",
    "VLDAFinetuneTrainer": "loongforge.training.embodied.custom.vlda.VLDAFinetuneTrainer",
    "GrootN1d6Trainer": "loongforge.training.embodied.custom.gr00t_n1_6.GrootN1d6Trainer",
    "LingBotFinetuneTrainer": "loongforge.training.embodied.custom.lingbot_va.LingBotFinetuneTrainer",
    "GrootN1d7Trainer": "loongforge.training.embodied.custom.gr00t_n1_7.GrootN1d7Trainer",
    "LingbotVlaV2Trainer": "loongforge.training.embodied.custom.lingbot_vla_v2.LingbotVlaV2Trainer",
}


def resolve_torch_trainer(trainer_type: str):
    """Load the Torch trainer selected by ``--trainer-type``."""
    if not trainer_type:
        raise ValueError(
            "--trainer-type is required. "
            f"Available: {list(_TRAINER_CLASSES.keys())}"
        )

    entry = _TRAINER_CLASSES.get(trainer_type)
    if entry is None:
        raise ValueError(
            f"Unknown --trainer-type '{trainer_type}'. "
            f"Available: {list(_TRAINER_CLASSES.keys())}"
        )

    module_path, class_name = entry.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


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
