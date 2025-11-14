# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
"""moe layer for deepseek"""

from megatron.core.transformer.moe.moe_layer import (
    MoELayer as MegatronMoELayer,
    MoESubmodules,
)

from aiak_training_omni.models.deepseek.transformer.transformer_config import (
    DeepSeekTransformerConfig,
)
from aiak_training_omni.models.deepseek.transformer.router import TopKRouter


class MoELayer(MegatronMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronMoELayer): Base class for MoE layers
    """

    def __init__(
        self,
        config: DeepSeekTransformerConfig,
        submodules: MoESubmodules = None,
        layer_number: int = None,
    ):
        super(MoELayer, self).__init__(
            config=config, submodules=submodules, layer_number=layer_number
        )

        # overwrite router with mtp verison
        self.router = TopKRouter(config=self.config)
