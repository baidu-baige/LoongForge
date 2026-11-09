# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""all model trainer"""

from typing import Union, List, Callable
import logging

MODEL_FAMILY_TRAINER_FACTORY = {}
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
    from loongforge.training.llm import pretrain_llm  # noqa: F401
    from loongforge.training.vlm import pretrain_vlm  # noqa: F401
    from loongforge.training.llm import sft_llm  # noqa: F401
    from loongforge.training.vlm import sft_vlm, sft_internvl, sft_ernie  # noqa: F401
    from loongforge.training.diffusion import pretrain_wan, pretrain_qwen_image  # noqa: F401

    model_family = args.model_family
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
