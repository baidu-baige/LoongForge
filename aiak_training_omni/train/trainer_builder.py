"""all model trainer"""

from typing import Union, List, Callable

from aiak_training_omni.models import get_model_family
from aiak_training_omni.utils.global_vars import get_hydra_config

MODEL_FAMILY_TRAINER_FACTORY = {}


def register_model_trainer(
    model_family: Union[str, List[str]],
    training_phase: str,
    training_func: Callable = None,
):
    """
    register model training function

    Args:
        model_family: need to be consistent with the models.factory definition, otherwise it
                      cannot be retrieved correctly. (Case-insensitive)

        training_phase: need to be consistent with the --training-phase definition in train.arguments
        trainig_func: training function.
    """

    def _add_trainer(families, phase, func):
        if not isinstance(families, list):
            families = [families]

        for _family in families:
            _family = _family.lower()
            if _family not in MODEL_FAMILY_TRAINER_FACTORY:
                MODEL_FAMILY_TRAINER_FACTORY[_family] = {}

            if phase in MODEL_FAMILY_TRAINER_FACTORY[_family]:
                raise ValueError(
                    f"Cannot register duplicate trainer ({_family} family, {phase} phase)"
                )

            MODEL_FAMILY_TRAINER_FACTORY[_family][phase] = func

    def _register_function(fn):
        _add_trainer(model_family, training_phase, fn)
        return fn

    if training_func is not None:
        return _add_trainer(model_family, training_phase, training_func)
    else:
        return _register_function


def build_model_trainer(args):
    """create model trainer"""
    config = get_hydra_config()
    if not hasattr(config, 'model'):
        raise ValueError("Invalid model configuration structure")
    model_family = config.model.model_type
    # get model family trainer
    if model_family not in MODEL_FAMILY_TRAINER_FACTORY:
        raise ValueError(
            f"Not found trainer for family: {model_family}"
        )

    if args.training_phase not in MODEL_FAMILY_TRAINER_FACTORY[model_family]:
        raise ValueError(
            f"AIAK not support {args.training_phase} phase for {args.model_name} (family: {model_family})"
        )

    trainer = MODEL_FAMILY_TRAINER_FACTORY[model_family][
        args.training_phase
    ]
    return trainer(args)
