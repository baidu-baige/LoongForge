# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Trainer registration and lookup for every engine."""

import logging
from typing import Union, List, Callable

# Trainers selected by engine, model family, and training phase.
TRAINERS_BY_ENGINE = {"megatron": {}, "torch": {}}
# Explicit trainer-name overrides, currently supported by Torch.
TRAINERS_BY_NAME = {"torch": {}}
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def register_model_trainer(
    model_family: Union[str, List[str]],
    training_phase: str,
    training_func: Callable = None,
    override: bool = False,
    *,
    engine: str,
):
    """
    register model training function

    Args:
        model_family: the YAML ``model_type`` family. Megatron recipes and Torch
                      trainers both use it; an empty list registers a Torch trainer
                      for ``--trainer-type`` only. (Case-insensitive)
        training_phase: need to be consistent with the --training-phase definition in training.megatron_args
        training_func: training function. Torch trainer names must be unique within
                       a phase because ``--trainer-type`` selects them.
        override: Whether to overwrite existing training functions for the same model/phase.
                  Default: False (no overwrite).
        engine: key of ``ENGINES``.
    """
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Available: {sorted(ENGINES)}")

    def _add_trainer(families, phase, func, override):
        if not isinstance(families, list):
            families = [families]

        if engine == "torch":
            named = TRAINERS_BY_NAME[engine].setdefault(phase, {})
            existing = named.setdefault(func.__name__, func)
            if existing is not func:
                raise ValueError(
                    f"Duplicate trainer name '{func.__name__}' for {engine} {phase}: "
                    f"{existing.__module__} and {func.__module__}. Rename one of them."
                )
        engine_trainers = TRAINERS_BY_ENGINE[engine]
        for _family in families:
            _family = _family.lower()
            if _family not in engine_trainers:
                engine_trainers[_family] = {}

            if phase in engine_trainers[_family]:
                if not override:
                    continue
                else:
                    logger.info(
                        f"Overriding existing trainer ({engine}, {_family} family, {phase} phase)"
                    )

            engine_trainers[_family][phase] = func

    def _register_function(fn):
        _add_trainer(model_family, training_phase, fn, override)
        return fn

    if training_func is not None:
        return _add_trainer(model_family, training_phase, training_func, override)
    else:
        return _register_function


def _megatron():
    """Parse Megatron arguments and load the built-in Megatron recipes."""
    logging.basicConfig(level=logging.WARNING)
    from loongforge.engines.megatron.parser import parse_train_args

    args = parse_train_args()
    from loongforge.training.llm import pretrain_llm  # noqa: F401
    from loongforge.training.vlm import pretrain_vlm  # noqa: F401
    from loongforge.training.llm import sft_llm  # noqa: F401
    from loongforge.training.vlm import sft_vlm, sft_internvl, sft_ernie  # noqa: F401
    from loongforge.training.diffusion import pretrain_wan, pretrain_qwen_image  # noqa: F401
    return args.model_family, args.training_phase, None, (args,)


def _torch():
    """Parse Torch arguments, then load the built-in trainers."""
    from loongforge.data.constants import TrainingPhase
    from loongforge.training.torch_args import parse_train_args

    training_args, model_cfg, data_cfg = parse_train_args()

    # Shared trainer first: model-specific trainers override it for their family.
    from loongforge.training.embodied import sft  # noqa: F401
    from loongforge.training.embodied.custom import (  # noqa: F401
        gr00t_n1_6,
        gr00t_n1_7,
        lingbot_va,
        lingbot_vla_v2,
        tensor_wise_trainer,
        vlda,
    )

    return (
        model_cfg.model_type,
        TrainingPhase.SFT,
        training_args.trainer_type,
        (training_args, model_cfg, data_cfg),
    )


# Each engine parses its own arguments, loads its recipes, and returns the lookup key.
ENGINES = {"megatron": _megatron, "torch": _torch}


def build_model_trainer(engine):
    """Build the trainer registered for this run's engine, family, and phase."""
    family, phase, trainer_type, configs = ENGINES[engine]()
    raw_family = family
    family = family.lower()

    if trainer_type:
        named = TRAINERS_BY_NAME["torch"].get(phase, {})
        if trainer_type not in named:
            raise ValueError(
                f"Unknown --trainer-type '{trainer_type}'. "
                f"Available: {sorted(named)}"
            )
        return named[trainer_type](*configs)

    trainers = TRAINERS_BY_ENGINE[engine]
    if family not in trainers:
        raise ValueError(f"Not found trainer for family: {raw_family}")
    if phase not in trainers[family]:
        raise ValueError(f"Loongforge not support {phase} phase for family {raw_family}")
    return trainers[family][phase](*configs)
