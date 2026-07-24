#!/usr/bin/env python
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from NVIDIA GR00T under the Apache-2.0 License.

"""GR00T-N1.7 native model implementation for LoongForge embodied trainer."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.model.registry import register_model

from .model_configuration_groot_n1_7 import GrootN1d7Config
from .modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from .modules.embodiment_mlp import CategorySpecificMLP, MultiEmbodimentActionEncoder
from .modules.qwen3_backbone import Qwen3Backbone

logger = logging.getLogger(__name__)


def _module_parameter_dtype(module: nn.Module) -> torch.dtype | None:
    """Return the dtype of the first floating point parameter, if any."""
    for parameter in module.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype
    return None


class Gr00tN1d7ActionHead(nn.Module):
    """Flow-matching action head used by GR00T-N1.7."""

    supports_gradient_checkpointing = True

    def __init__(self, config: GrootN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for GR00T-N1.7 action head")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for GR00T-N1.7 action head")

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()

        vl_self_attention_cfg = config.vl_self_attention_cfg if config.use_vl_self_attention else None
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.state_dropout_prob = config.state_dropout_prob
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self._split_noise_buf = None
        self._split_time_buf = None
        self._split_state_dropout_buf = None
        self._split_supports_state_dropout_buf = True
        self._split_record_shape = False
        self._split_actions_shape = None
        self._split_actions_device = None
        self._split_actions_dtype = None
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            config.tune_vlln,
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool,
    ) -> None:
        """Apply trainability flags to projector, DiT, and VL layer norms."""
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for parameter in self.parameters():
            parameter.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        """Keep frozen modules in eval mode when outer Trainer calls train()."""
        if not self.training:
            return
        if not self.tune_projector:
            self.state_encoder.eval()
            self.action_encoder.eval()
            self.action_decoder.eval()
            if self.config.add_pos_embed:
                self.position_embedding.eval()
        if not self.tune_diffusion_model:
            self.model.eval()
        if not self.tune_vlln:
            self.vlln.eval()
            self.vl_self_attention.eval()

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Sample flow-matching timesteps."""
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (1 - sample) * self.config.noise_s

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply VLM layer normalization and optional self-attention."""
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        return BatchFeature(
            data={
                **dict(backbone_output),
                "backbone_features": backbone_features,
            }
        )

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Compute masked flow-matching MSE loss."""
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id
        state = action_input.state
        actions = action_input.action

        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape[1] != self.config.state_history_length:
            raise ValueError(
                f"state history length mismatch: got {state.shape[1]}, "
                f"expected {self.config.state_history_length}"
            )
        state = state.view(state.shape[0], 1, -1)

        state_features = self.state_encoder(state, embodiment_id)
        if self.training and self.state_dropout_prob > 0:
            split_dropout = self._split_state_dropout_buf
            if split_dropout is None:
                do_dropout = (
                    torch.rand(state_features.shape[0], device=state_features.device)
                    < self.state_dropout_prob
                )
            else:
                _ = torch.rand(state_features.shape[0], device=state_features.device)
                do_dropout = split_dropout
            state_features = state_features * (1 - do_dropout[:, None, None].to(state_features.dtype))

        split_noise = self._split_noise_buf
        if split_noise is None:
            if self._split_record_shape:
                self._split_actions_shape = actions.shape
                self._split_actions_device = actions.device
                self._split_actions_dtype = actions.dtype
            noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]
        else:
            _ = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            noise = split_noise
            t = self._split_time_buf[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask
        if self.config.use_alternate_vl_dit:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return BatchFeature(
            data={
                "loss": loss,
                "action_loss": action_loss,
                "action_mask": action_mask,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @property
    def device(self) -> torch.device:
        """Return action-head device."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """Return action-head parameter dtype."""
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare action-head input."""
        return BatchFeature(data=batch)


class Gr00tN1d7(nn.Module):
    """GR00T-N1.7 VLA model with Qwen3/Cosmos backbone and action head."""

    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: GrootN1d7Config,
        transformers_loading_kwargs: dict | None = None,
    ):
        super().__init__()
        self.config = config
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}

        self.backbone = Qwen3Backbone(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.action_head = Gr00tN1d7ActionHead(config)

    def set_input_tensor(self, input_tensor) -> None:
        """Megatron compatibility no-op."""
        self._input_tensor = input_tensor

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Split LoongForge batch dict into backbone and action inputs."""
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)
        backbone_dtype = _module_parameter_dtype(self.backbone)
        action_dtype = _module_parameter_dtype(self.action_head)

        def to_device_with_dtype(value, dtype: torch.dtype | None):
            if torch.is_tensor(value):
                if torch.is_floating_point(value):
                    if dtype is None:
                        return value.to(self.device)
                    return value.to(self.device, dtype=dtype)
                return value.to(self.device)
            if isinstance(value, dict):
                return {
                    key: to_device_with_dtype(item, dtype) for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                converted = [to_device_with_dtype(item, dtype) for item in value]
                return type(value)(converted)
            return value

        backbone_dict = backbone_inputs.data if isinstance(backbone_inputs, BatchFeature) else backbone_inputs
        action_dict = action_inputs.data if isinstance(action_inputs, BatchFeature) else action_inputs
        return (
            BatchFeature(
                data={
                    key: to_device_with_dtype(value, backbone_dtype)
                    for key, value in backbone_dict.items()
                }
            ),
            BatchFeature(
                data={
                    key: to_device_with_dtype(value, action_dtype)
                    for key, value in action_dict.items()
                }
            ),
        )

    def forward(self, inputs: dict) -> BatchFeature:
        """Forward through backbone and action head."""
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        return self.action_head(backbone_outputs, action_inputs)

    @property
    def device(self) -> torch.device:
        """Return model parameter device."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """Return model parameter dtype."""
        return next(iter(self.parameters())).dtype


@register_model("Gr00tN1d7")
class GrootN1d7Policy(nn.Module):
    """GR00T-N1.7 policy wrapper for LoongForge FinetuneTrainer."""

    preserve_param_dtype = True

    def __init__(self, config: GrootN1d7Config):
        super().__init__()
        self.config = config
        self._pretrained_checkpoint_path: str | None = None
        self._reload_pretrained_once_after_precision_cast = False
        self._restoring_after_apply = False
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.model = Gr00tN1d7(
            config,
            transformers_loading_kwargs={
                "trust_remote_code": True,
                "local_files_only": True,
            },
        )

    def reset_data_iterator_rng(self, seed: int) -> None:
        """Align DataLoader worker base seeds with Isaac's finetune path."""
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def from_pretrained(cls, cfg) -> "GrootN1d7Policy":
        """Instantiate from config; weights are loaded by trainer via load_pretrained."""
        return cls(GrootN1d7Config.from_config(cfg))

    def forward(self, batch) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return LoongForge trainer contract: loss plus log-loss dict."""
        try:
            batch_inputs = batch.to_model_inputs()
        except AttributeError as exc:
            raise TypeError(
                "GrootN1d7Policy.forward expects a batch with to_model_inputs(), "
                f"got {type(batch).__name__}"
            ) from exc
        outputs = self.model(batch_inputs)
        loss = outputs.get("loss", None)
        if loss is None:
            action_loss = outputs["action_loss"]
            action_mask = outputs.get("action_mask", torch.ones_like(action_loss))
            loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return loss, {"action_loss": loss.detach()}

    @property
    def device(self) -> torch.device:
        """Return policy device."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        """Return policy dtype."""
        return next(iter(self.parameters())).dtype

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        if not self._restoring_after_apply:
            self._restore_precision_after_apply()
        return result

    def restore_trainable_params_fp32(self) -> None:
        """Restore trainable parameters to fp32 after framework dtype casts."""
        with self._precision_restore_guard():
            self._restore_trainable_params_fp32_impl()

    def _restore_trainable_params_fp32_impl(self) -> None:
        if self.config.backbone_trainable_params_fp32:
            _restore_trainable_params_fp32(self.model.backbone)
        _restore_trainable_params_fp32(self.model.action_head)
        _restore_rotary_buffers_fp32(self.model)

    def _restore_precision_after_apply(self) -> None:
        if os.environ.get("GROOT_ALLOW_TRAINABLE_PARAM_BF16", "0") == "1":
            with self._precision_restore_guard():
                _restore_rotary_buffers_fp32(self.model)
            return
        needs_checkpoint_reload = self._has_trainable_param_below_fp32()
        with self._precision_restore_guard():
            self._restore_trainable_params_fp32_impl()
            if (
                needs_checkpoint_reload
                and self._reload_pretrained_once_after_precision_cast
                and self._pretrained_checkpoint_path
            ):
                self._reload_pretrained_once_after_precision_cast = False
                self.load_pretrained(
                    self._pretrained_checkpoint_path,
                    device=self.device,
                    remember_path=False,
                )
                self._restore_trainable_params_fp32_impl()

    @contextlib.contextmanager
    def _precision_restore_guard(self):
        previous = self._restoring_after_apply
        self._restoring_after_apply = True
        try:
            yield
        finally:
            self._restoring_after_apply = previous

    def _has_trainable_param_below_fp32(self) -> bool:
        modules = [self.model.action_head]
        if self.config.backbone_trainable_params_fp32:
            modules.append(self.model.backbone)
        return any(
            parameter.requires_grad and parameter.dtype != torch.float32
            for module in modules
            for parameter in module.parameters()
        )

    def load_pretrained(
        self,
        path: str,
        device: torch.device | None = None,
        *,
        remember_path: bool = True,
    ) -> None:
        """Load GR00T-N1.7 safetensors checkpoint into native LoongForge model."""
        state_dict = _load_groot_n1_7_state_dict(path, device=device)
        model_sd = self.model.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            candidate_keys = (
                key.removeprefix("model."),
                key,
                f"model.{key}",
            )
            target_key = next(
                (
                    candidate
                    for candidate in candidate_keys
                    if candidate in model_sd and model_sd[candidate].shape == value.shape
                ),
                None,
            )
            if target_key is not None:
                filtered[target_key] = value
            else:
                skipped.append(key)

        missing, unexpected = self.model.load_state_dict(filtered, strict=False)
        logger.info(
            "Loaded GR00T-N1.7 checkpoint from %s: %d tensors, %d skipped, %d missing, %d unexpected",
            path,
            len(filtered),
            len(skipped),
            len(missing),
            len(unexpected),
        )
        if skipped:
            logger.warning("Skipped %d GR00T-N1.7 tensors; first keys: %s", len(skipped), skipped[:5])
        if remember_path:
            self._pretrained_checkpoint_path = path
            self._reload_pretrained_once_after_precision_cast = True


def _load_groot_n1_7_state_dict(
    path: str,
    device: torch.device | None = None,
) -> Dict[str, torch.Tensor]:
    """Load single-file or sharded safetensors/PT checkpoint."""
    map_location = str(device) if device is not None else "cpu"
    checkpoint = Path(path)
    if checkpoint.is_dir():
        index_path = checkpoint / "model.safetensors.index.json"
        single_safetensors = checkpoint / "model.safetensors"
        single_pt = checkpoint / "pytorch_model.pt"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as file_obj:
                index = json.load(file_obj)
            merged: Dict[str, torch.Tensor] = {}
            for shard_name in sorted(set(index["weight_map"].values())):
                merged.update(load_file(str(checkpoint / shard_name), device=map_location))
            return merged
        if single_safetensors.exists():
            return load_file(str(single_safetensors), device=map_location)
        if single_pt.exists():
            return torch.load(single_pt, map_location=map_location)
        raise FileNotFoundError(f"No GR00T-N1.7 checkpoint weights found in {checkpoint}")

    if str(checkpoint).endswith(".safetensors"):
        return load_file(str(checkpoint), device=map_location)
    return torch.load(checkpoint, map_location=map_location)


def _restore_trainable_params_fp32(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.float32)


def _restore_rotary_buffers_fp32(module: nn.Module) -> None:
    """Keep Qwen rotary buffers fp32 after framework casts."""
    for submodule in module.modules():
        if type(submodule).__name__ not in {"Qwen2RotaryEmbedding", "Qwen3RotaryEmbedding"}:
            continue
        try:
            inv_freq = submodule.inv_freq
            rope_init_fn = submodule.rope_init_fn
            config = submodule.config
        except AttributeError:
            continue
        if inv_freq.device.type == "meta" or not callable(rope_init_fn):
            continue
        new_inv_freq, attention_scaling = rope_init_fn(config, device=inv_freq.device)
        new_inv_freq = new_inv_freq.to(device=inv_freq.device, dtype=torch.float32)
        submodule.register_buffer("inv_freq", new_inv_freq, persistent=False)
        submodule.original_inv_freq = new_inv_freq
        submodule.attention_scaling = attention_scaling
