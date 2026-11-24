# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""all model trainer"""

from typing import Union, List, Callable
import importlib
import logging

from loongforge.utils.global_vars import get_hydra_config
from loongforge.utils import constants

MODEL_FAMILY_TRAINER_FACTORY = {}
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_trainer_registered(model_family: str, training_phase: str) -> None:
    """Import only the trainer module required by the selected family."""
    if model_family in MODEL_FAMILY_TRAINER_FACTORY and (
        training_phase in MODEL_FAMILY_TRAINER_FACTORY[model_family]
    ):
        return

    language_families = set(constants.LanguageModelFamilies.names())
    vision_families = set(constants.VisionLanguageModelFamilies.names())
    custom_families = set(constants.CustomModelFamilies.names())
    vla_families = set(constants.VisionLanguageActionModelFamilies.names())

    if model_family in language_families and training_phase == constants.TrainingPhase.PRETRAIN:
        module_name = "loongforge.train.pretrain.pretrain_llm"
    elif model_family in language_families and training_phase == constants.TrainingPhase.SFT:
        module_name = "loongforge.train.sft.sft_llm"
    elif model_family in vision_families and training_phase == constants.TrainingPhase.PRETRAIN:
        module_name = "loongforge.train.pretrain.pretrain_vlm"
    elif model_family == constants.VisionLanguageModelFamilies.INTERN_VL:
        module_name = "loongforge.train.sft.sft_internvl"
    elif model_family == constants.VisionLanguageModelFamilies.ERNIE4_5_VL:
        module_name = "loongforge.train.sft.sft_ernie"
    elif model_family in vision_families:
        module_name = "loongforge.train.sft.sft_vlm"
    elif model_family == constants.CustomModelFamilies.QWEN_IMAGE:
        module_name = "loongforge.train.diffusion.pretrain_qwen_image"
    elif model_family in custom_families:
        module_name = "loongforge.train.diffusion.pretrain_wan"
    elif model_family in vla_families:
        raise ValueError(
            f"LoongForge VLA trainer for {model_family} is not part of the core v0.20 migration"
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    importlib.import_module(module_name)


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

        training_phase: need to be consistent with the --training-phase definition in train.arguments
        trainig_func: training function.
        override: Whether to overwrite existing training functions for the same model/phase.
                  Default: False (no overwrite).
    """

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


def build_model_trainer(args):
    """create model trainer"""
    config = get_hydra_config()
    
    if hasattr(config, "model_type") and config.model_type in \
            (set(constants.LanguageModelFamilies.names()) |
             set(constants.CustomModelFamilies.names()) |
             set(constants.VisionLanguageActionModelFamilies.names())):
        model_family = config.model_type
    else:
        if not hasattr(config, 'model'):
            raise ValueError("Invalid model configuration structure")
        model_family = config.model.model_type

    _ensure_trainer_registered(model_family, args.training_phase)

    # get model family trainer
    if model_family not in MODEL_FAMILY_TRAINER_FACTORY:
        raise ValueError(
            f"Not found trainer for family: {model_family}"
        )

    if args.training_phase not in MODEL_FAMILY_TRAINER_FACTORY[model_family]:
        raise ValueError(
            f"Loongforge not support {args.training_phase} phase for {args.model_name} (family: {model_family})"
        )

    trainer = MODEL_FAMILY_TRAINER_FACTORY[model_family][
        args.training_phase
    ]
    return trainer(args)
