# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge
# Modifications: Fix VLM LoRA to support VLM models.
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""defined lora classes"""

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import unwrap_model

from loongforge.models.base.peft.base import PEFT
from loongforge.models.base.peft.lora_layers import (
    LinearAdapter,
    LoRALinear,
    LoRATopKRouter,
    TEFusedLoRALinear,
    TELinearAdapter,
    patch_linear_module,
)
from loongforge.models.base.peft.module_matcher import ModuleMatcher
from loongforge.models.base.peft.utils import (
    ParallelLinearAdapter,
    get_adapter_attributes_from_linear,
    is_expert_linear,
)


logger = logging.getLogger(__name__)

try:
    import bitsandbytes

    HAVE_BNB = True
except ImportError:
    HAVE_BNB = False


@dataclass
class LoRA(PEFT, ModuleMatcher):
    """
    Implements the LoRA (Low-Rank Adaptation) module for parameter-efficient fine-tuning.

    LoRA uses a low-rank projection to adapt the weights of a pre-trained model to a new downstream task.
    This class facilitates the application of LoRA to specific modules within the model architecture.

    Args:
        target_modules (List[str], optional): A list of module names to apply LoRA to.
            Defaults to all linear layers ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2'].
                - 'linear_qkv': Apply LoRA to the fused linear layer used for query, key, and value projections
                                in self-attention.
                - 'linear_proj': Apply LoRA to the linear layer used for projecting the output of self-attention.
                - 'linear_fc1': Apply LoRA to the first fully-connected layer in MLP.
                - 'linear_fc2': Apply LoRA to the second fully-connected layer in MLP.
            Target modules can also contain wildcards. For example, you can specify
                target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv'] to add LoRA to only linear_qkv
                on the first two layers.
        exclude_modules (List[str], optional): A list of module names not to apply LoRa to. It will
            match all nn.Linear & nn.Linear-adjacent modules whose name does not match any string in
            exclude_modules. If used, will require target_modules to be empty list or None.
        dim (int): Dimension of the low-rank projection space. Defaults to 32.
        alpha (int): Weighting factor for the low-rank projection. Defaults to 32.
        dropout (float): Dropout rate for the low-rank projection. Defaults to 0.0.
        dropout_position (Literal['pre', 'post'], optional): Position for applying dropout.
            Can be 'pre' (before the low-rank projection) or 'post' (after). Defaults to 'pre'.
        a2a_experimental (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to False.
        lora_A_init_method (str): Initialization method for the low-rank matrix A. Defaults to "xavier".
        lora_B_init_method (str): Initialization method for the low-rank matrix B. Defaults to "zero".
        lora_dtype (torch.dtype): Parameter data type for LoRA weights. Default None (will use model's dtype).
    """

    target_modules: List[str] = field(
        default_factory=lambda: [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ]
    )
    dim: int = 32
    alpha: int = 32
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "pre"
    lora_A_init_method: str = "xavier"
    lora_B_init_method: str = "zero"
    a2a_experimental: bool = False
    lora_dtype: torch.dtype = None

    def transform(
        self,
        module: nn.Module,
        name: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> nn.Module:
        """
        Applies LoRA to a specific module within the model architecture.

        Args:
            m (nn.Module): The module to apply LoRA to.
            name (str, optional): Name of the module (if applicable). Defaults to None.
            prefix (str, optional): Prefix for the module name (if applicable). Defaults to None.

        Returns:
            nn.Module: The modified module with LoRA applied, or the original module if not a target.
        """
        # Skip already transformed modules
        adapter_types = (LinearAdapter, LoRALinear, LoRATopKRouter)
        adapter_types = adapter_types + (TELinearAdapter,)
        if isinstance(module, adapter_types):
            return module

        if (ans := self.match(module, name, prefix)) is not None:
            (match, full_name) = ans
            if isinstance(module, nn.Linear) or (module.__class__ == te.Linear):
                # Will use the `patch_linear_module` function if:
                # - is FSDP v1
                # - is DTensor (has _local_tensor attribute)
                # - has quant_state attribute
                if hasattr(module.weight.data, "_local_tensor") or (
                    HAVE_BNB
                    and getattr(module, "quant_state", None) is not None
                    and module.quant_state.__class__
                    == bitsandbytes.functional.QuantState
                ):
                    lora_cls = patch_linear_module
                elif module.__class__ == te.Linear:
                    lora_cls = TELinearAdapter
                else:
                    lora_cls = LinearAdapter

                return lora_cls(
                    module,
                    dim=self.dim,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    lora_A_init_method=self.lora_A_init_method,
                    lora_dtype=self.lora_dtype,
                )

            is_expert = is_expert_linear(full_name)
            attrs = get_adapter_attributes_from_linear(module, is_expert=is_expert)

            enable_op_fuser = (
                hasattr(module, "config")
                and getattr(module.config, "use_transformer_engine_op_fuser", False)
                # TP not yet supported
                and parallel_state.get_tensor_model_parallel_world_size() == 1
            )

            logging.info(f"Adding lora to: {full_name}")
            adapter = ParallelLinearAdapter(
                attrs.in_features,
                attrs.out_features,
                self.dim,
                base_linear_name=full_name,
                activation="identity",
                norm_type=None,
                column_init_method=self.lora_A_init_method,
                row_init_method=self.lora_B_init_method,
                gather_output=False,
                input_is_parallel=attrs.input_is_parallel,
                dropout=self.dropout,
                dropout_position=self.dropout_position,
                model_parallel_config=getattr(module, "config", None),
                alpha=self.alpha,
                is_expert=is_expert,
                a2a_experimental=self.a2a_experimental,
                disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                base_linear_is_parallel=attrs.base_linear_is_parallel,
            )
            if isinstance(module, TopKRouter):
                return LoRATopKRouter(module, adapter)
            if enable_op_fuser:
                return TEFusedLoRALinear(module, adapter)
            else:
                return LoRALinear(module, adapter)
        return module


@dataclass
class VLMLoRA(LoRA):
    """
    Implements the LoRA for Vision-Language Models.
    VLMLoRA additionally allows the user to specify whether the language or vision
    models should be frozen.
    For example, a common finetuning workload for multimodal models is to apply adapters to language model and fully
    finetune the vision model.

    """

    apply_to_foundation: bool = False
    apply_to_image_projector: bool = False
    apply_to_image_encoder: bool = False
    apply_to_video_projector: bool = False
    apply_to_video_encoder: bool = False

    def freeze_model(self, model: nn.Module, training: bool = True) -> None:
        """Freezes the specified parts of the model during training."""
        modules_to_freeze = []

        model = unwrap_model(model)[0]

        if (
            self.apply_to_foundation
            and hasattr(model, "foundation_model")
            and model.foundation_model is not None
        ):
            modules_to_freeze.append(model.foundation_model)

        if (
            self.apply_to_image_projector
            and hasattr(model.encoder_model, "image_projector")
            and model.encoder_model.image_projector is not None
        ):
            modules_to_freeze.append(model.encoder_model.image_projector)

        if (
            self.apply_to_image_encoder
            and hasattr(model.encoder_model, "image_encoder")
            and model.encoder_model.image_encoder is not None
        ):
            modules_to_freeze.append(model.encoder_model.image_encoder)

        if (
            self.apply_to_video_projector
            and hasattr(model.encoder_model, "video_projector")
            and model.encoder_model.video_projector is not None
        ):
            modules_to_freeze.append(model.encoder_model.video_projector)

        if (
            self.apply_to_video_encoder
            and hasattr(model.encoder_model, "video_encoder")
            and model.encoder_model.video_encoder is not None
        ):
            modules_to_freeze.append(model.encoder_model.video_encoder)

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)
