"""default pretrain for generative models like GPTS"""

import os

from megatron.core.enums import ModelType
from megatron.core.utils import StragglerDetector

from omni_training.utils import (
    constants
)
from omni_training.models.omni_models.omni_model_provider import (
    omni_model_provider,
)


from omni_training.train.megatron_trainer import MegatronTrainer
from omni_training.train.trainer_builder import register_model_trainer

stimer = StragglerDetector()


@register_model_trainer(model_family=constants.VisionLanguageModelFamilies.names(),
                        training_phase=constants.TrainingPhase.SFT)
def default_pretrain_trainer(train_args):
    """build trainer"""
    from omni_training.train.pretrain import pretrain_vlm
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=pretrain_vlm.train_valid_test_dataset_provider,
        model_provider=omni_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=pretrain_vlm.forward_step,
    )

    return trainer
