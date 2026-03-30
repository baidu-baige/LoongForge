"""Configuration helpers for Eagle3 VLM loading."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch


# Full PretrainedConfig-based Eagle3_VLConfig class
import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.utils import logging
from .modeling_siglip2 import Siglip2VisionConfig

logger = logging.get_logger(__name__)


class Eagle3VLConfig(PretrainedConfig):
    """Configuration class for Eagle3 VLM model.

    This configuration inherits from PretrainedConfig and can be used to control the model outputs.
    """

    model_type = "eagle_3_vl"
    is_composition = True
    sub_configs = {"vision_config": SiglipVisionConfig, "text_config": Qwen2Config}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,
        select_layer=-4,
        downsample_ratio=0.5,
        template=None,
        loss_version="v1",
        mlp_checkpoint=False,
        image_token_index=151667,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"model_type": "siglip_vision_model"}
            logger.info("vision_config is None. Initializing with SiglipVisionModel default.")

        if text_config is None:
            text_config = {"architectures": ["Qwen3ForCausalLM"]}
            logger.info("text_config is None. Initializing with Qwen3ForCausalLM default.")

        # Initialize vision config
        if isinstance(vision_config, dict):
            if vision_config["model_type"] == "siglip_vision_model":
                self.vision_config = SiglipVisionConfig(**vision_config)
            elif vision_config["model_type"] == "siglip2_vision_model":
                self.vision_config = Siglip2VisionConfig(**vision_config)
            else:
                raise ValueError(f"Unsupported vision model_type: {vision_config['model_type']}")
        else:
            self.vision_config = vision_config

        # Initialize text config
        if isinstance(text_config, dict):
            text_arch = text_config["architectures"][0]
            if text_arch == "Qwen2ForCausalLM":
                self.text_config = Qwen2Config(**text_config)
            elif text_arch == "Qwen3ForCausalLM":
                self.text_config = Qwen3Config(**text_config)
            else:
                raise ValueError(f"Unsupported text architecture: {text_arch}")
        else:
            self.text_config = text_config

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.mlp_checkpoint = mlp_checkpoint
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.tie_word_embeddings = getattr(self.text_config, "tie_word_embeddings", False)
        self.image_token_index = image_token_index
        self.initializer_range = initializer_range

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["select_layer"] = self.select_layer
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["image_token_index"] = self.image_token_index
        output["initializer_range"] = self.initializer_range
        output["_attn_implementation"] = getattr(self, "_attn_implementation", None)
        output["_attn_implementation_autoset"] = getattr(self, "_attn_implementation_autoset", None)
        return output

def build_eagle_load_kwargs(
    transformers_loading_kwargs: dict | None,
    use_flash_attention: bool,
    load_bf16: bool,
) -> tuple[dict, bool]:
    """Build kwargs for HF model loading and detect offline mode."""
    loading_kwargs = dict(transformers_loading_kwargs or {})

    if use_flash_attention:
        loading_kwargs["attn_implementation"] = "flash_attention_2"
    if load_bf16:
        loading_kwargs["torch_dtype"] = torch.bfloat16

    offline_mode = (
        str(os.environ.get("HF_HUB_OFFLINE", "")).lower() in {"1", "true", "yes"}
        or str(os.environ.get("TRANSFORMERS_OFFLINE", "")).lower() in {"1", "true", "yes"}
        or bool(loading_kwargs.get("local_files_only", False))
    )

    return loading_kwargs, offline_mode

def resolve_eagle_local_path(model_name: str, current_file: str) -> str | None:
    """Resolve best-effort local Eagle path from env/vendor/cache."""
    # Workspace conventions
    env_local_path = os.environ.get("EAGLE_LOCAL_PATH")
    hardcoded_local_path = "/workspace/huggingface.co/aravindhs-NV/eagle3-processor-groot-n1d6"

    # In-tree vendor fallback (kept for compatibility)
    vendor_eagle_path = (
        Path(current_file).resolve().parent.parent.parent
        / "vendor"
        / "gr00t"
        / "model"
        / "modules"
        / "nvidia"
        / "Eagle-Block2A-2B-v2"
    )

    if env_local_path:
        return env_local_path
    if os.path.exists(hardcoded_local_path):
        return hardcoded_local_path
    if os.path.exists(model_name):
        return model_name
    if vendor_eagle_path.exists():
        return str(vendor_eagle_path)

    cache_root = os.environ.get("TRANSFORMERS_CACHE")
    if cache_root:
        cache_candidate = os.path.join(cache_root, "eagle3-processor-groot-n1d6")
        if os.path.exists(cache_candidate):
            return cache_candidate
        if os.path.exists(cache_root):
            return cache_root

    return None
