# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA Eagle3 under the Apache-2.0 License.
#
# Copyright 2024 NVIDIA. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Eagle backbone wrapper for Gr00tN1d6."""

from __future__ import annotations

import contextlib
import os

import torch
import transformers
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_utils import no_init_weights as _hf_no_init_weights
except ImportError:
    _hf_no_init_weights = None

from .configuration_eagle3_vl import Eagle3VLConfig, build_eagle_load_kwargs, resolve_eagle_local_path
from .modeling_eagle3_vl import load_eagle_model


_GROOT_EAGLE_TORCH_INIT_FUNCTIONS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "uniform",
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
)


def _get_transformers_major_version() -> int | None:
    try:
        return int(transformers.__version__.split(".", maxsplit=1)[0])
    except (AttributeError, ValueError):
        return None


_TRANSFORMERS_MAJOR_VERSION = _get_transformers_major_version()


@contextlib.contextmanager
def _groot_eagle_no_init_weights():
    """Match the Transformers 4 Eagle no-init side effects across HF versions."""
    # Transformers 4 is the base environment and exposes no_init_weights from
    # modeling_utils. Keep that exact path there; Transformers 5 moved no-init
    # to a broader implementation that changes the CPU RNG stream.
    if (
        _hf_no_init_weights is not None
        and (_TRANSFORMERS_MAJOR_VERSION is None or _TRANSFORMERS_MAJOR_VERSION < 5)
    ):
        with _hf_no_init_weights():
            yield
        return

    def _skip_init(*args, **kwargs):
        pass

    originals = {}
    original_init_weights = PreTrainedModel.init_weights
    try:
        for name in _GROOT_EAGLE_TORCH_INIT_FUNCTIONS:
            if hasattr(torch.nn.init, name):
                originals[name] = getattr(torch.nn.init, name)
                setattr(torch.nn.init, name, _skip_init)
        PreTrainedModel.init_weights = _skip_init
        yield
    finally:
        for name, init_func in originals.items():
            setattr(torch.nn.init, name, init_func)
        PreTrainedModel.init_weights = original_init_weights


def _use_graph_safe_eagle() -> bool:
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return True
    try:
        from loongforge.embodied.train.global_vars import get_training_args
        training_args = get_training_args()
        return (
            training_args.cuda_graph_impl == "local"
            and training_args.cuda_graph_scope in {"full_iteration", "per_microbatch"}
        )
    except (ImportError, RuntimeError, AssertionError):
        pass
    return (
        os.environ.get("CUDA_GRAPH_IMPL", "none") == "local"
        and os.environ.get("CUDA_GRAPH_SCOPE", "full_iteration") in {"full_iteration", "per_microbatch"}
    )


def _load_lerobot_eager_eagle(
    *,
    model_name: str,
    loading_kwargs: dict,
    use_flash_attention: bool,
    load_bf16: bool,
    current_file: str,
) -> torch.nn.Module:
    """Load Eagle through the same AutoModel remote-code path LeRobot uses."""
    if not use_flash_attention or not load_bf16:
        raise ValueError("GR00T-N1.6 Eagle eager parity expects flash_attention_2 and bf16 loading.")

    local_model_path = resolve_eagle_local_path(model_name, current_file)
    if not local_model_path or not os.path.exists(os.path.join(local_model_path, "config.json")):
        raise FileNotFoundError(
            f"Could not resolve local Eagle config for eager parity: model_name={model_name}, "
            f"resolved={local_model_path}"
        )

    config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    if hasattr(config, "text_config") and config.text_config is not None:
        config.text_config._attn_implementation = "flash_attention_2"
    if (
        hasattr(config, "vision_config")
        and config.vision_config is not None
        and config.vision_config.model_type in {"siglip_vision_model", "siglip2_vision_model"}
    ):
        config.vision_config._attn_implementation = "flash_attention_2"

    # LeRobot builds this backbone inside HF from_pretrained's no-init context.
    # Match that construction path here; weights are loaded by the outer GR00T loader.
    with _groot_eagle_no_init_weights():
        model = AutoModel.from_config(config, trust_remote_code=True)
    print(f"Created Eagle model via LeRobot eager AutoModel path: {local_model_path}")
    return model


class EagleBackbone(torch.nn.Module):
    """Backbone wrapper that keeps training/inference contract stable."""

    def __init__(
        self,
        model_name: str = "aravindhs-NV/eagle3-processor-groot-n1d6",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict | None = None,
    ):
        """Initialize EagleBackbone module.
        
        Args:
            model_name: Name of the pretrained model to load
            tune_llm: Whether to fine-tune language model parameters
            tune_visual: Whether to fine-tune visual model parameters
            select_layer: Which layer to select from language model
            reproject_vision: Whether to reproject visual features
            use_flash_attention: Whether to use flash attention
            projector_dim: Dimension of projection layer
            load_bf16: Whether to load model in bf16 precision
            tune_top_llm_layers: Number of top LLM layers to tune
            trainable_params_fp32: Whether to keep trainable params in fp32
            transformers_loading_kwargs: Additional kwargs for model loading
        """
        super().__init__()

        self._eagle_cfg = Eagle3VLConfig(
            model_name=model_name,
            use_flash_attention=use_flash_attention,
            load_bf16=load_bf16,
        )
        loading_kwargs, offline_mode = build_eagle_load_kwargs(
            transformers_loading_kwargs=transformers_loading_kwargs,
            use_flash_attention=use_flash_attention,
            load_bf16=load_bf16,
        )
        if _use_graph_safe_eagle():
            self.model = load_eagle_model(
                config=self._eagle_cfg,
                select_layer=select_layer,
                loading_kwargs=loading_kwargs,
                offline_mode=offline_mode,
                current_file=__file__,
            )
        else:
            self.model = _load_lerobot_eager_eagle(
                model_name=model_name,
                loading_kwargs=loading_kwargs,
                use_flash_attention=use_flash_attention,
                load_bf16=load_bf16,
                current_file=__file__,
            )

        # Handle layer selection for different model structures
        # The language model structure may vary based on how the model was loaded
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
                # Standard structure: model.language_model.model.layers
                while len(self.model.language_model.model.layers) > select_layer:
                    self.model.language_model.model.layers.pop(-1)
            elif hasattr(self.model.language_model, 'layers'):
                # Alternative structure: model.language_model.layers
                while len(self.model.language_model.layers) > select_layer:
                    self.model.language_model.layers.pop(-1)
            else:
                print(f"Warning: Could not find layers in language_model structure")
        else:
            print(f"Warning: model does not have language_model attribute")

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)

        if load_bf16 and trainable_params_fp32:
            for name, parameter in self.named_parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.float32)
                    print(f"Casting trainable parameter {name} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        """Set which parameters should be trainable.
        
        Args:
            tune_llm: Whether to tune language model parameters
            tune_visual: Whether to tune visual model parameters
            tune_top_llm_layers: Number of top LLM layers to tune
        """
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual

        for parameter in self.parameters():
            parameter.requires_grad = True

        if hasattr(self.model, 'language_model') and not tune_llm:
            self.model.language_model.requires_grad_(False)
        if hasattr(self.model, 'vision_model') and not tune_visual:
            self.model.vision_model.requires_grad_(False)
        if hasattr(self.model, 'mlp1') and not tune_visual:
            self.model.mlp1.requires_grad_(False)

        if tune_top_llm_layers > 0 and hasattr(self.model, 'language_model'):
            # Handle different layer structures
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
                layers = self.model.language_model.model.layers
            elif hasattr(self.model.language_model, 'layers'):
                layers = self.model.language_model.layers
            else:
                layers = []

            for layer in layers[-tune_top_llm_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")

    def set_frozen_modules_to_eval_mode(self):
        """Set frozen modules to evaluation mode."""
        if self.training:
            if hasattr(self.model, 'language_model') and self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if hasattr(self.model, 'vision_model') and self.model.vision_model and not self.tune_visual:
                self.model.vision_model.eval()
            if hasattr(self.model, 'mlp1') and self.model.mlp1 and not self.tune_visual:
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input for model processing.
        
        Args:
            batch: Input dictionary containing model inputs
            
        Returns:
            BatchFeature: Processed input features
        """
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """Forward pass of the model.
        
        Args:
            vl_input: Input features for the model
            
        Returns:
            BatchFeature: Model outputs
        """
        self.set_frozen_modules_to_eval_mode()

        keys_to_use = ["input_ids", "attention_mask", "pixel_values"]
        vl_input = {key: vl_input[key] for key in keys_to_use}

        outputs = self.model(**vl_input, output_hidden_states=True)

        # Handle different output structures from different model implementations
        if isinstance(outputs, dict):
            if "hidden_states" in outputs and outputs["hidden_states"]:
                outputs = outputs["hidden_states"][-1]
            elif "last_hidden_state" in outputs:
                outputs = outputs["last_hidden_state"]
            else:
                raise ValueError(f"Unexpected output structure: {outputs.keys()}")
        else:
            # Assume output has hidden_states attribute
            outputs = outputs.hidden_states[-1]

        # Get image token index from config
        image_token_index = self.model.config.image_token_index
        image_mask = vl_input["input_ids"] == image_token_index
        attention_mask = vl_input["attention_mask"] == 1

        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )
