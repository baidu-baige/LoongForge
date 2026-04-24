"""Eagle backbone wrapper for Gr00tN1d6."""

from __future__ import annotations

import torch
from transformers.feature_extraction_utils import BatchFeature

from .configuration_eagle3_vl import Eagle3VLConfig, build_eagle_load_kwargs
from .modeling_eagle3_vl import load_eagle_model


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
        self.model = load_eagle_model(
            config=self._eagle_cfg,
            select_layer=select_layer,
            loading_kwargs=loading_kwargs,
            offline_mode=offline_mode,
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
        image_token_index = getattr(self.model, "image_token_index", 0)
        if hasattr(self.model, "config"):
            image_token_index = getattr(self.model.config, "image_token_index", image_token_index)
        image_mask = vl_input["input_ids"] == image_token_index
        attention_mask = vl_input["attention_mask"] == 1

        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )