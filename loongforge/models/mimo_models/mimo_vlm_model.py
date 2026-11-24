# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge wrapper around the native MCore MiMo multimodal model."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import torch

from megatron.core.distributed import DistributedDataParallel
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import MultiModuleProcessGroupCollection
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import unwrap_model

from .qwen25_vl import split_image_inputs_for_fan_in


_LEGACY_OMNI_PREFIX_MAP = {
    "foundation_model.": "language_model.",
    "encoder_model.image_encoder.": (
        "modality_submodules.images.encoders.image_encoder."
    ),
    "encoder_model.image_projector.": (
        "modality_submodules.images.input_projections.0."
    ),
}
_LEGACY_OMNI_ALIASES = {
    # This weight aliases foundation_model.embedding.word_embeddings.weight in
    # OmniCombinationModel; MiMo has only the language model copy.
    "encoder_model.text_encoder.word_embeddings.weight",
}


def remap_legacy_omni_state_dict(state_dict):
    """Map a legacy Omni checkpoint into the native MiMo module tree."""
    if not any(
        key.startswith(tuple(_LEGACY_OMNI_PREFIX_MAP))
        or key in _LEGACY_OMNI_ALIASES
        for key in state_dict
    ):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        if key in _LEGACY_OMNI_ALIASES:
            continue
        for source_prefix, target_prefix in _LEGACY_OMNI_PREFIX_MAP.items():
            if key.startswith(source_prefix):
                key = target_prefix + key[len(source_prefix) :]
                break
        remapped[key] = value
    return remapped


class LoongForgeMimoVLMModel(torch.nn.Module):
    """Expose the existing LoongForge VLM forward contract on top of MiMo."""

    def __init__(
        self,
        *,
        mimo_model: MimoModel,
        topology,
        model_type: str,
        pre_process: bool,
        post_process: bool,
        image_spatial_merge_size: int,
    ) -> None:
        super().__init__()
        self.mimo_model = mimo_model
        self.topology = topology
        self.config = mimo_model.config
        self.model_type = model_type
        self.add_encoder = len(mimo_model.modality_submodules) > 0
        self.add_decoder = mimo_model.language_model is not None
        self.disable_param_offloading = True
        self.pre_process = pre_process
        self.post_process = post_process
        self.image_spatial_merge_size = image_spatial_merge_size
        self.pg_collection = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
        language_model = unwrap_model(mimo_model.language_model)
        self.share_embeddings_and_output_weights = getattr(
            language_model, "share_embeddings_and_output_weights", False
        )
        self.vp_stage = mimo_model.mimo_config.language_model_spec.params.get(
            "vp_stage"
        )
        active_module_names = [MIMO_LANGUAGE_MODULE_KEY]
        active_module_names.extend(mimo_model.modality_submodules.keys())
        active_module_pgs = {
            name: topology.module_pgs[name] for name in active_module_names
        }
        self.module_pg_collection = MultiModuleProcessGroupCollection(
            module_pgs=active_module_pgs,
            language_model_module_name=MIMO_LANGUAGE_MODULE_KEY,
        )

    def set_input_tensor(self, input_tensor) -> None:
        self.mimo_model.set_input_tensor(input_tensor)

    def enable_forward_pre_hook(self):
        return self.mimo_model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync: bool = True):
        return self.mimo_model.disable_forward_pre_hook(param_sync=param_sync)

    def start_param_sync(self, *args, **kwargs):
        return self.mimo_model.start_param_sync(*args, **kwargs)

    def start_grad_sync(self, *args, **kwargs):
        return self.mimo_model.start_grad_sync(*args, **kwargs)

    def free_overlap_buffers(self):
        return self.mimo_model.free_overlap_buffers()

    def zero_grad_buffer(self):
        return self.mimo_model.zero_grad_buffer()

    @property
    def remove_forward_pre_hook_handles(self):
        return self.mimo_model.remove_forward_pre_hook_handles

    def shared_embedding_or_output_weight(self):
        language = unwrap_model(self.mimo_model.language_model)
        return language.shared_embedding_or_output_weight()

    def state_dict(self, *args, **kwargs):
        return self.mimo_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load unwrapped MiMo or legacy Omni weights into the active module tree."""
        state_dict = remap_legacy_omni_state_dict(state_dict)
        roles = {"language_model": self.mimo_model.language_model}
        roles.update(
            (f"modality_submodules.{name}", module)
            for name, module in self.mimo_model.modality_submodules.items()
        )
        prefixes = {}
        for role, module in roles.items():
            target = role
            while isinstance(module, (DistributedDataParallel, Float16Module)):
                target += ".module"
                module = module.module
            if target != role:
                prefixes[role + "."] = target + "."

        if prefixes:
            def resolve_key(key):
                for source, target in prefixes.items():
                    if key.startswith(source) and not key.startswith(target):
                        return target + key[len(source):]
                return key

            resolved = OrderedDict()
            for key, value in state_dict.items():
                target_key = resolve_key(key)
                if target_key in resolved:
                    raise RuntimeError(f"Duplicate MiMo weight key: {target_key}")
                resolved[target_key] = value
            if hasattr(state_dict, "_metadata"):
                resolved._metadata = OrderedDict(
                    (resolve_key(name + ".")[:-1], value)
                    for name, value in state_dict._metadata.items()
                )
            state_dict = resolved

        return self.mimo_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return self.mimo_model.sharded_state_dict(
            prefix=prefix,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
        )

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.mimo_model.state_dict(prefix=prefix, keep_vars=keep_vars)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mimo_model.train(mode)
        return self

    def forward(
        self,
        image_inputs: Optional[dict[str, torch.Tensor]] = None,
        video_inputs: Optional[dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[dict[str, torch.Tensor]] = None,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Any = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if video_inputs is not None:
            raise NotImplementedError("Phase 3 Qwen2.5-VL MiMo smoke supports images only")
        if audio_inputs is not None:
            raise NotImplementedError("Qwen2.5-VL does not support audio inputs")
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Phase 3 MiMo smoke supports SBHD only; packed sequence is disabled"
            )

        modality_fan_in_sizes = None
        image_bridge = self.mimo_model.colocated_comms.get(("images", MIMO_LANGUAGE_MODULE_KEY))
        if image_bridge is not None and image_bridge.is_fan_out():
            raise NotImplementedError("Qwen2.5-VL MiMo needs vision DP >= language DP")
        if image_inputs is not None and image_bridge is not None and image_bridge.is_fan_in():
            slot = image_bridge.get_slice_info(image_bridge.scale).start
            image_inputs, fan_in_sizes = split_image_inputs_for_fan_in(
                image_inputs, slot, image_bridge.scale, self.image_spatial_merge_size
            )
            modality_fan_in_sizes = {"images": fan_in_sizes}

        modality_inputs = None
        if image_inputs is not None:
            modality_inputs = {
                "images": {
                    "image_encoder": {
                        "images": image_inputs["images"],
                        "image_grid_thw": image_inputs["image_grid_thw"],
                    }
                }
            }

        output, output_loss_mask = self.mimo_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            modality_inputs=modality_inputs,
            modality_fan_in_sizes=modality_fan_in_sizes,
        )
        if labels is not None:
            return output
        return output, output_loss_mask
