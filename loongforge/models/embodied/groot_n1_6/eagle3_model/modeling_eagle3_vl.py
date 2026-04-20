"""Eagle3 VLM model implementation - adapted from lerobot for LoongForge.

This provides the Eagle3_VLForConditionalGeneration model that is needed
for properly creating the Eagle model structure without requiring pretrained weights.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, GenerationConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers.models.siglip.modeling_siglip import SiglipVisionModel
from transformers.utils import logging

from .configuration_eagle3_vl import (
    Eagle3VLConfig,
    resolve_eagle_local_path,
)
from .modeling_siglip2 import Siglip2VisionModel

logger = logging.get_logger(__name__)


# =============================================================================
# Eagle3_VLForConditionalGeneration Model Implementation
# =============================================================================


class Eagle3VlPreTrainedModel(PreTrainedModel):
    """Base class for Eagle3_VL models."""

    config_class = Eagle3VLConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Qwen2DecoderLayer",
        "LlamaDecoderLayer",
        "Siglip2EncoderLayer",
        "SiglipEncoderLayer",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True
    _supports_static_cache = True
    _supports_quantized_cache = True
    _supports_sdpa = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Eagle3VlForConditionalGeneration(Eagle3VlPreTrainedModel, GenerationMixin):
    """Eagle3 VLM model for conditional generation.

    This model combines a vision encoder (Siglip2) with a language model (Qwen3)
    through a projection layer (mlp1). It is used for vision-language tasks.
    """

    config_class = Eagle3VLConfig

    def __init__(self, config: Eagle3VLConfig, vision_model=None, language_model=None):
        super().__init__(config)

        self.select_layer = config.select_layer
        self.template = getattr(config, "template", None)
        self.downsample_ratio = config.downsample_ratio
        self.loss_version = getattr(config, "loss_version", "v1")
        self.mlp_checkpoint = getattr(config, "mlp_checkpoint", False)

        logger.info(f"mlp_checkpoint: {self.mlp_checkpoint}")

        # Initialize vision model
        if config.vision_config.model_type == "siglip_vision_model":
            config.vision_config._attn_implementation = "flash_attention_2"
            self.vision_model = SiglipVisionModel(config.vision_config)
        elif config.vision_config.model_type == "siglip2_vision_model":
            config.vision_config._attn_implementation = "flash_attention_2"
            self.vision_model = Siglip2VisionModel(config.vision_config)
        else:
            raise ValueError(f"Unsupported vision model type: {config.vision_config.model_type}")

        # Initialize language model
        text_arch = config.text_config.architectures[0]
        if text_arch == "LlamaForCausalLM":
            self.language_model = LlamaForCausalLM(config.text_config)
        elif text_arch == "Phi3ForCausalLM":
            from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM
            self.language_model = Phi3ForCausalLM(config.text_config)
        elif text_arch == "Qwen2ForCausalLM":
            if getattr(config.text_config, "_attn_implementation", None) != "flash_attention_2":
                logger.warning(
                    "Qwen2 attention implementation is %s; overriding to flash_attention_2.",
                    getattr(config.text_config, "_attn_implementation", None),
                )
                config.text_config._attn_implementation = "flash_attention_2"
            self.language_model = Qwen2ForCausalLM(config.text_config)
        elif text_arch == "Qwen3ForCausalLM":
            if getattr(config.text_config, "_attn_implementation", None) != "flash_attention_2":
                logger.warning(
                    "Qwen3 attention implementation is %s; overriding to flash_attention_2.",
                    getattr(config.text_config, "_attn_implementation", None),
                )
                config.text_config._attn_implementation = "flash_attention_2"

            # if getattr(config.text_config, "_attn_implementation", None) != "eager":
            #     logger.warning(
            #         "Qwen2 attention implementation is %s; overriding to eager.",
            #         getattr(config.text_config, "_attn_implementation", None),
            #     )
            #     config.text_config._attn_implementation = "eager"
            self.language_model = Qwen3ForCausalLM(config.text_config)
        else:
            raise NotImplementedError(f"{text_arch} is not implemented.")

        # Initialize projection layer
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.image_token_index = config.image_token_index
        self.neftune_alpha = None

        self.use_backbone_lora = getattr(config, "use_backbone_lora", 0) != 0
        self.use_llm_lora = getattr(config, "use_llm_lora", 0) != 0

        if self.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if self.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        """Wrap vision model with LoRA."""
        lora_config = LoraConfig(
            r=r,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        """Wrap language model with LoRA."""
        lora_config = LoraConfig(
            r=r,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
        self.use_llm_lora = True

    def forward(
        self,
        pixel_values: list[torch.FloatTensor],
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_flags: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        """Forward pass of the model."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if image_flags is not None:
            image_flags = image_flags.view(-1)

        vit_embeds = self.extract_feature(pixel_values, image_flags)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.image_token_index
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds
        except Exception as e:
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = (
                input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            )

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle_back(self, vit_embeds, spatial_shapes):
        """Apply pixel shuffle to vision embeddings."""
        B, N, C = vit_embeds.shape
        shapes = spatial_shapes.tolist()

        # Split at once
        lengths = [h * w for (h, w) in shapes]
        slices = torch.split(vit_embeds.view(-1, C), lengths, dim=0)

        # Convert to [C, H, W]
        features = [
            sl.transpose(0, 1).reshape(C, h, w) 
            for sl, (h, w) in zip(slices, shapes, strict=True)
        ]

        # Group by scale and batch unshuffle
        down_feats = [None] * len(features)
        grouped: dict = defaultdict(list)
        for idx, (h, w) in enumerate(shapes):
            grouped[(h, w)].append(idx)

        for (_h, _w), idxs in grouped.items():
            # Stack features of same scale
            grp = torch.stack([features[i] for i in idxs], dim=0)
            # Pixel Unshuffle at once
            out = F.pixel_unshuffle(
                grp, downscale_factor=int(1 / self.downsample_ratio)
            )
            out = out.flatten(start_dim=2).transpose(1, 2)
            # Split back to respective positions
            for i, feat in zip(idxs, out, strict=True):
                down_feats[i] = feat

        down_feats = torch.cat(down_feats, dim=0).unsqueeze(0)
        return down_feats, (spatial_shapes * self.downsample_ratio).to(torch.int32)

    def mask_valid_tokens(self, vit_embeds, spatial_shapes, image_flags):
        """Mask out invalid tokens from vision embeddings."""
        """
        Args:
            vit_embeds: Vision embeddings tensor
            spatial_shapes: Spatial shapes of the features
            image_flags: Flags indicating valid image tokens
        """
        lengths = spatial_shapes[:, 0] * spatial_shapes[:, 1]
        valid_mask = []
        for flag, length in zip(image_flags, lengths, strict=True):
            valid_mask.extend([flag] * length)

        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=vit_embeds.device)
        valid_tokens = vit_embeds[valid_mask]

        return valid_tokens

    def extract_feature(self, pixel_values, image_flags=None):
        """Extract vision features and apply projection."""
        """
        Args:
            pixel_values: Input pixel values
            image_flags: Optional flags indicating valid image tokens
        """
        if self.select_layer == -1:
            vision_model_output = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            )
            if hasattr(vision_model_output, "last_hidden_state"):
                vit_embeds = vision_model_output.last_hidden_state
            if hasattr(vision_model_output, "spatial_shapes"):
                spatial_shapes = vision_model_output.spatial_shapes
            else:
                spatial_shapes = None
        else:
            vision_model_output = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            )
            vit_embeds = vision_model_output.hidden_states[self.select_layer]
            if hasattr(vision_model_output, "spatial_shapes"):
                spatial_shapes = vision_model_output.spatial_shapes
            else:
                spatial_shapes = None

        vit_embeds, spatial_shapes = self.pixel_shuffle_back(vit_embeds, spatial_shapes)

        if self.mlp_checkpoint and vit_embeds.requires_grad:
            vit_embeds = cp.checkpoint(self.mlp1, vit_embeds)
        else:
            vit_embeds = self.mlp1(vit_embeds)

        B, N, C = vit_embeds.shape
        vit_embeds = vit_embeds.reshape(B * N, C)

        if image_flags is not None and any(image_flags == 0):
            vit_embeds = self.mask_valid_tokens(vit_embeds, spatial_shapes, image_flags)

        return vit_embeds

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        visual_features: torch.FloatTensor | None = None,
        generation_config: GenerationConfig | None = None,
        output_hidden_states: bool | None = None,
        image_sizes: list[tuple[int, int]] | None = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """Generate text conditioned on visual input."""
        """
        Args:
            pixel_values: Input pixel values
            input_ids: Input token IDs
            attention_mask: Attention mask
            visual_features: Precomputed visual features
            generation_config: Generation configuration
            output_hidden_states: Whether to output hidden states
            image_sizes: Sizes of input images
            **generate_kwargs: Additional generation arguments
        """
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                pixel_values = [each.to(self.device) for each in pixel_values]

                torch.cuda.synchronize()
                for _ in range(10):
                    vit_embeds = self.extract_feature(pixel_values)
                torch.cuda.synchronize()

            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.image_token_index
            if selected.sum() == 0:
                raise ValueError("No image tokens found in input_ids")
            input_embeds[selected] = vit_embeds.to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        if "use_cache" not in generate_kwargs:
            generate_kwargs["use_cache"] = True

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            **generate_kwargs,
        )

        return outputs

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embeddings."""
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings."""
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        """Set decoder."""
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        """Get decoder."""
        return self.language_model.get_decoder()
# =============================================================================
# Model Loading Helpers
# =============================================================================


def load_eagle_model(
    config: Eagle3VLConfig,
    select_layer: int,
    loading_kwargs: dict,
    offline_mode: bool,
    current_file: str,
) -> torch.nn.Module:
    """Load Eagle model from config or hub, following lerobot's approach.

    This function uses AutoConfig to load the configuration and then creates
    the model structure using AutoModel.from_config, which creates the
    model without requiring pretrained weights. This is similar to lerobot's approach.

    Args:
        config: Eagle3VLConfig runtime config
        select_layer: Layer to select from the language model
        loading_kwargs: Additional kwargs for transformers loading
        offline_mode: Whether to use offline mode
        current_file: Path to the current file for resolving vendor paths

    Returns:
        Loaded Eagle model
    """
    # Try to load config from local path or HuggingFace Hub
    canonical_models = {
        "nvidia/Eagle-Block2A-2B-v2",
        "aravindhs-NV/eagle3-processor-groot-n1d6",
    }
    if config.model_name in canonical_models or "Eagle-Block2A-2B-v2" in config.model_name:
        local_model_path = resolve_eagle_local_path(config.model_name, current_file)
        if local_model_path and os.path.exists(os.path.join(local_model_path, "config.json")):
            eagle_path = str(local_model_path)
            print(f"Using local Eagle path: {eagle_path}")
        else:
            raise FileNotFoundError(
                "Local Eagle path not found for canonical model. "
                "Please set EAGLE_LOCAL_PATH or place model files under "
                "/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6."
            )

        # Load config from the cache directory
        try:
            model_config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
        except ValueError as exc:
            config_path = os.path.join(eagle_path, "config.json")
            if not os.path.exists(config_path):
                raise RuntimeError(
                    f"Eagle config not found at {config_path}. "
                    "Please provide a local model config/weights directory or disable offline mode."
                ) from exc

            with open(config_path, encoding="utf-8") as file_obj:
                raw_cfg = json.load(file_obj)

            if "vision_config" not in raw_cfg or "text_config" not in raw_cfg:
                fallback_config = Eagle3VLConfig(
                    select_layer=select_layer,
                    downsample_ratio=raw_cfg.get("downsample_ratio", 0.5),
                    image_token_index=raw_cfg.get("image_token_index", 151667),
                    initializer_range=raw_cfg.get("initializer_range", 0.02),
                )
                model = Eagle3VlForConditionalGeneration(fallback_config)
                print(
                    "Warning: Eagle config missing vision/text sections; "
                    f"using built-in defaults from Eagle3VLConfig: {config_path}"
                )
                return model

            raw_cfg = dict(raw_cfg)
            vision_config_dict = raw_cfg.pop("vision_config")
            text_config_dict = raw_cfg.pop("text_config")

            if raw_cfg.get("select_layer", None) is None:
                raw_cfg["select_layer"] = select_layer

            eagle_full_config = Eagle3VLConfig(
                vision_config=vision_config_dict,
                text_config=text_config_dict,
                **raw_cfg,
            )
            model = Eagle3VlForConditionalGeneration(eagle_full_config)
            print(
                "Loaded Eagle config via manual parsing "
                f"(missing model_type in AutoConfig path): {config_path}"
            )
            return model

        # Set attention implementation for text_config (required for Qwen2/Qwen3)
        if hasattr(model_config, "text_config") and model_config.text_config is not None:
            model_config.text_config._attn_implementation = "flash_attention_2"

        # Set attention implementation for vision_config if needed
        if (
            hasattr(model_config, "vision_config")
            and model_config.vision_config is not None
            and hasattr(model_config.vision_config, "model_type")
            and model_config.vision_config.model_type in ["siglip_vision_model", "siglip2_vision_model"]
        ):
            model_config.vision_config._attn_implementation = "flash_attention_2"

        # Create model from config (no weights loaded initially)
        # This is the key difference - creates model structure without requiring weights
        # IMPORTANT: For proper loading, we should load pretrained weights first,
        # then pop extra layers. This matches lerobot's behavior.
        try:
            # Prefer loading pretrained weights first, then popping layers
            # This ensures layers 12-15 have the pretrained weights before being popped
            print(f"Attempting to load pretrained weights from: {eagle_path}")
            try:
                # Try to load with pretrained weights (like lerobot does)
                model = AutoModel.from_pretrained(
                    eagle_path,
                    trust_remote_code=True,
                    **loading_kwargs
                )
                print(f"Loaded pretrained model with weights from: {eagle_path}")
                return model
            except Exception as e:
                print(f"Failed to load pretrained weights (will load from config): {e}")
                # Fall back to loading from config if pretrained loading fails
                pass

            # Fallback: create model from config only
            if hasattr(model_config, "to_dict"):
                model_config_dict = model_config.to_dict()
            else:
                model_config_dict = {}

            vision_config_dict = model_config_dict.pop(
                "vision_config",
                model_config.vision_config.to_dict() if hasattr(model_config, "vision_config") else None,
            )
            text_config_dict = model_config_dict.pop(
                "text_config",
                model_config.text_config.to_dict() if hasattr(model_config, "text_config") else None,
            )

            # Only fall back to runtime select_layer when missing in config.json
            if model_config_dict.get("select_layer", None) is None:
                model_config_dict["select_layer"] = select_layer

            eagle_full_config = Eagle3VLConfig(
                vision_config=vision_config_dict,
                text_config=text_config_dict,
                **model_config_dict,
            )

            # Align attention implementation with lerobot defaults
            if hasattr(eagle_full_config, "text_config"):
                eagle_full_config.text_config._attn_implementation = "flash_attention_2"
            if hasattr(eagle_full_config, "vision_config"):
                eagle_full_config.vision_config._attn_implementation = "flash_attention_2"
            eagle_full_config._attn_implementation = "flash_attention_2"

            model = Eagle3VlForConditionalGeneration(eagle_full_config)
            print(f"Created Eagle3 model from config (no pretrained weights): {eagle_path}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to create custom Eagle model from local config at {eagle_path}."
            ) from e

    # For other models, try the standard loading approach
    local_model_path = resolve_eagle_local_path(config.model_name, current_file)
    print(f"Resolved Eagle local path: {local_model_path}")

    if local_model_path is None or not os.path.exists(local_model_path):
        if offline_mode:
            raise FileNotFoundError(
                f"Eagle local model path not found: model_name={config.model_name}, resolved={local_model_path}"
            )
        # Try to load from HF Hub
        try:
            model_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True, **loading_kwargs)
            # Set attention implementations
            if hasattr(model_config, "text_config") and model_config.text_config is not None:
                model_config.text_config._attn_implementation = loading_kwargs.get("attn_implementation", "eager")
            if hasattr(model_config, "vision_config") and model_config.vision_config is not None:
                model_config.vision_config._attn_implementation = loading_kwargs.get("attn_implementation", "eager")
            # Create model from config
            model = AutoModel.from_config(model_config, trust_remote_code=True)
            print(f"Created Eagle model from hub config: {config.model_name}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Eagle model from HF Hub for '{config.model_name}'."
            ) from e

    has_model_files = (
        os.path.exists(os.path.join(local_model_path, "config.json"))
        and (
            os.path.exists(os.path.join(local_model_path, "model.safetensors"))
            or os.path.exists(os.path.join(local_model_path, "pytorch_model.bin"))
            or any(
                file_name.startswith("model-")
                for file_name in os.listdir(local_model_path)
                if file_name.endswith(".safetensors") or file_name.endswith(".bin")
            )
        )
    )

    if has_model_files:
        # Load from local with weights
        model = AutoModel.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,
            **loading_kwargs,
        )
        print(f"Loaded Eagle from local path with weights: {local_model_path}")
        return model
    else:
        # Load from config only (no weights)
        try:
            model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
            # Set attention implementations
            if hasattr(model_config, "text_config") and model_config.text_config is not None:
                model_config.text_config._attn_implementation = loading_kwargs.get("attn_implementation", "eager")
            if hasattr(model_config, "vision_config") and model_config.vision_config is not None:
                model_config.vision_config._attn_implementation = loading_kwargs.get("attn_implementation", "eager")
            model = AutoModel.from_config(model_config, trust_remote_code=True)
            print(f"Loaded Eagle from local config (no weights): {local_model_path}")
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to build Eagle model from local config path: {local_model_path}"
            ) from e
