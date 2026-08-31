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
import re
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.data.datasets.groot_n1_6.transforms.processor_groot_n1_6 import (
    StateActionProcessor,
)
from loongforge.embodied.data.datasets.groot_n1_7.transforms.groot_collator import (
    Gr00tN1d7DataCollator,
)
from loongforge.embodied.data.datasets.groot_n1_7.transforms.groot_transform import (
    EMBODIMENT_STAT_CONFIGS,
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    MODALITY_CONFIGS,
    convert_lerobot_stats_to_groot_n1d7_format,
)
from loongforge.embodied.data.datasets.groot_n1_7.transforms.image_augmentations import (
    build_image_transformations_albumentations,
)
from loongforge.embodied.model.registry import register_model
from .model_configuration_groot_n1_7 import GrootN1d7Config
from .modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from .modules.embodiment_mlp import CategorySpecificMLP, MultiEmbodimentActionEncoder
from .modules.qwen3_backbone import Qwen3Backbone

logger = logging.getLogger(__name__)

_DEFAULT_PREDICT_ACTION_EMBODIMENT_TAG = "libero_sim"


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
        else:
            _ = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
            noise = split_noise

        split_time = self._split_time_buf
        if split_time is None:
            t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
            t = t[:, None, None]
        else:
            t = split_time[:, None, None]
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
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                # The action head consumes only the final projection; retaining
                # every intermediate state adds graph bookkeeping without
                # changing the training result.
                return_all_hidden_states=False,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=False,
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
        self.collator: Gr00tN1d7DataCollator | None = None
        self.embodiment_tag = _DEFAULT_PREDICT_ACTION_EMBODIMENT_TAG
        self.modality_config = MODALITY_CONFIGS[self.embodiment_tag]
        self.modality_meta = EMBODIMENT_STAT_CONFIGS[self.embodiment_tag]["modality_meta"]
        self.state_keys = list(self.modality_config["state"].modality_keys)
        self.action_keys = list(self.modality_config["action"].modality_keys)
        self.embodiment_id = int(EMBODIMENT_TAG_TO_PROJECTOR_INDEX[self.embodiment_tag])
        self.raw_state_dim = max(meta["end"] for meta in self.modality_meta["state"].values())
        self.model_action_horizon = int(config.action_horizon)
        self.action_horizon = len(self.modality_config["action"].delta_indices)
        self.action_dim = int(
            max(meta["end"] for meta in self.modality_meta["action"].values())
        )
        self._predict_action_initialized = False
        self._predict_action_validation_zero_state = False
        self._predict_action_use_bf16 = bool(config.use_bf16)
        self._predict_action_statistics: Dict[str, Any] | None = None
        self._predict_action_use_percentiles = False
        self._predict_action_clip_outliers = True
        self._predict_action_default_processor: StateActionProcessor | None = None
        self._predict_action_processor_cache: dict[str, StateActionProcessor] = {}
        self._predict_action_eval_image_transform = None

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
        action_head_inputs = getattr(batch, "to_action_head_inputs", None)
        if callable(action_head_inputs):
            backbone_output, action_input = action_head_inputs()
            outputs = self.model.action_head(backbone_output, action_input)
        else:
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
            # HF Trainer keeps the frozen Qwen backbone parameters in FP32 and
            # relies on autocast only for eligible kernels.  Keeping the
            # parameters permanently in bf16 changes the FP32 residual stream
            # through every vision and language block.
            _restore_module_params_fp32(self.model.backbone)
        _restore_trainable_params_fp32(self.model.action_head)
        _restore_rotary_buffers_fp32(self.model)

    def _restore_precision_after_apply(self) -> None:
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

    def configure_predict_action(
        self,
        *,
        checkpoint_statistics: Dict[str, Any],
        embodiment_tag: str = _DEFAULT_PREDICT_ACTION_EMBODIMENT_TAG,
        use_bf16: bool | None = None,
        validation_zero_state: bool = False,
        processor_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        """Configure eval-side resources used by ``predict_action``.

        Args:
            checkpoint_statistics: Per-embodiment or raw-LeRobot state/action stats.
            embodiment_tag: Which embodiment's modality config / projector to activate.
            use_bf16: Override the config's bf16 autocast setting.
            validation_zero_state: Allow ``state=None`` to fall back to a zero
                proprio. Only for interface validation / server warmup.
            processor_kwargs: The checkpoint's ``processor_config.json``
                ``processor_kwargs``. The image pipeline params are read from it
                verbatim because the released values are not all derivable from
                the model config: ``crop_fraction`` is 0.95, not the
                ``image_crop_size / image_target_size`` ratio (0.898), and
                ``letter_box_transform`` is False even though the training launch
                config says otherwise. Falls back to the released GR00T-N1.7
                values when omitted.
        """
        if embodiment_tag not in MODALITY_CONFIGS:
            raise ValueError(f"Unsupported GR00T-N1.7 embodiment_tag={embodiment_tag!r}")
        if embodiment_tag not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
            raise ValueError(f"No projector id registered for embodiment_tag={embodiment_tag!r}")

        self.embodiment_tag = embodiment_tag
        self.modality_config = MODALITY_CONFIGS[embodiment_tag]
        self.modality_meta = EMBODIMENT_STAT_CONFIGS[embodiment_tag]["modality_meta"]
        self.state_keys = list(self.modality_config["state"].modality_keys)
        self.action_keys = list(self.modality_config["action"].modality_keys)
        self.embodiment_id = int(EMBODIMENT_TAG_TO_PROJECTOR_INDEX[embodiment_tag])
        self.raw_state_dim = max(meta["end"] for meta in self.modality_meta["state"].values())
        self.action_horizon = len(self.modality_config["action"].delta_indices)
        self.model_action_horizon = int(self.config.action_horizon)
        self.action_dim = int(
            max(meta["end"] for meta in self.modality_meta["action"].values())
        )
        self._predict_action_validation_zero_state = bool(validation_zero_state)
        self._predict_action_use_bf16 = bool(self.config.use_bf16 if use_bf16 is None else use_bf16)
        self._predict_action_statistics = self._coerce_statistics(checkpoint_statistics)
        self._predict_action_processor_cache = {}
        pk = processor_kwargs or {}
        # State/action normalization mode comes from the checkpoint, not from the
        # StateActionProcessor defaults: the released LIBERO checkpoints set
        # use_percentiles=true (q01/q99), while the class defaults to min/max.
        # Getting this wrong silently rescales both the proprio fed to
        # state_encoder and the actions returned by unapply_action.
        self._predict_action_use_percentiles = bool(pk.get("use_percentiles", False))
        self._predict_action_clip_outliers = bool(pk.get("clip_outliers", True))
        self._predict_action_default_processor = self._build_state_action_processor(
            self._predict_action_statistics
        )
        self.collator = Gr00tN1d7DataCollator(
            model_name=self.config.model_name,
            model_type=self.config.backbone_model_type,
            transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
        )
        # Official GrootN1d7FeatureTransform eval pipeline: letterbox ->
        # SmallestMaxSize(shortest_image_edge) -> crop_fraction center crop ->
        # SmallestMaxSize. predict_action() applies this before building
        # vlm_content so callers only need to supply the env's native camera image.
        image_target_size = list(pk.get("image_target_size") or [256, 256])
        image_crop_size = list(pk.get("image_crop_size") or [230, 230])
        crop_fraction = pk.get("crop_fraction", 0.95)
        shortest_image_edge = pk.get("shortest_image_edge", 256)
        _, self._predict_action_eval_image_transform = build_image_transformations_albumentations(
            image_target_size,
            image_crop_size,
            int(pk.get("random_rotation_angle") or 0),
            None,  # color_jitter_params: train-only augmentation
            shortest_image_edge,
            crop_fraction,
            # Not read from processor_config.json: the checkpoint stores
            # letter_box_transform=false, but official Gr00tN1d7Processor marks
            # that field "stored but not actively used" and builds LetterBoxPad
            # unconditionally. Padding is a no-op on already-square frames.
            letter_box_transform=True,
        )
        logger.info(
            "GR00T-N1.7 eval image pipeline: target=%s crop=%s crop_fraction=%s "
            "shortest_edge=%s letterbox=True; state/action norm: percentiles=%s clip=%s",
            image_target_size,
            image_crop_size,
            crop_fraction,
            shortest_image_edge,
            self._predict_action_use_percentiles,
            self._predict_action_clip_outliers,
        )
        self._predict_action_initialized = True
        self.eval()

    @torch.no_grad()
    def predict_action(
        self,
        images,
        instructions,
        state=None,
        dataset_stats=None,
    ) -> np.ndarray:
        """Infer an action chunk for the shared eval ``predict_action`` interface."""
        image_batch = self._normalize_image_batch(images, instructions)
        image_batch = self._apply_eval_image_transform(image_batch)
        batch_size = len(image_batch)
        instruction_batch = self._normalize_instruction_batch(instructions, batch_size)
        processor = self._get_predict_action_processor(dataset_stats)
        raw_state_batch = self._coerce_state_batch(state, batch_size)
        normalized_state_batch = self._normalize_state_batch(processor, raw_state_batch)

        vlm_content = [
            self._build_vlm_content(sample_images, instruction)
            for sample_images, instruction in zip(image_batch, instruction_batch, strict=True)
        ]
        inputs = {
            "vlm_content": vlm_content,
            "state": torch.as_tensor(
                normalized_state_batch,
                dtype=torch.float32,
                device=self.model.device,
            ),
            "embodiment_id": torch.full(
                (batch_size,),
                self.embodiment_id,
                dtype=torch.long,
                device=self.model.device,
            ),
        }

        self.eval()
        normalized_actions = self._sample_normalized_actions(inputs)
        action_np = normalized_actions.float().cpu().numpy()
        decoded = self._decode_action_batch(processor, action_np, raw_state_batch)
        return decoded.astype(np.float32, copy=False)

    def _sample_normalized_actions(self, inputs: dict[str, Any]) -> torch.Tensor:
        self._ensure_predict_action_collator()
        model = self.model
        collated = self.collator([{"vlm_content": vlm} for vlm in inputs["vlm_content"]]).data["inputs"]
        collated["state"] = inputs["state"]
        collated["embodiment_id"] = inputs["embodiment_id"]
        backbone_inputs, action_inputs = model.prepare_input(collated)
        action_head = model.action_head

        device = model.device
        with torch.no_grad():
            backbone_output = model.backbone(backbone_inputs)
            # Qwen3Backbone.forward casts hidden_states to fp32 (training-side
            # precision policy); the official server keeps bf16 for the whole
            # model, so cast back here on the eval path. The action head is
            # already bf16 (GROOT_ALLOW_TRAINABLE_PARAM_BF16=1 in the factory),
            # so running without autocast keeps every intermediate bf16.
            if self._predict_action_use_bf16 and device.type == "cuda":
                backbone_output["backbone_features"] = backbone_output["backbone_features"].to(
                    torch.bfloat16
                )
            backbone_output = action_head.process_backbone_output(backbone_output)
            vl_embeds = backbone_output.backbone_features
            batch_size = vl_embeds.shape[0]

            embodiment_id = self._to_embodiment_tensor(
                action_inputs.embodiment_id,
                batch_size=batch_size,
                device=vl_embeds.device,
            )
            state_tensor = action_inputs.state
            if state_tensor.ndim == 2:
                state_tensor = state_tensor.unsqueeze(1)
            state_features = action_head.state_encoder(state_tensor, embodiment_id)

            # Integrate the flow over the DiT's *trained* horizon
            # (``config.action_horizon``, 40 for the released N1.7 checkpoints),
            # not over the per-embodiment decoded horizon. The DiT attends over
            # all action slots at once, so shortening the sequence changes every
            # output; official slices the first ``len(delta_indices)`` steps only
            # after sampling (processing_gr00t_n1d7.py:307/341). ``action_head``'s
            # own value follows the YAML ``model.action_horizon`` and is the
            # decoded length, which is why it must not be used here.
            sample_horizon = int(getattr(self, "model_action_horizon", 0) or action_head.action_horizon)
            actions = torch.randn(
                size=(batch_size, sample_horizon, action_head.action_dim),
                dtype=vl_embeds.dtype,
                device=vl_embeds.device,
            )
            dt = 1.0 / float(action_head.num_inference_timesteps)

            for step in range(action_head.num_inference_timesteps):
                timestep = int(
                    (step / float(action_head.num_inference_timesteps))
                    * action_head.num_timestep_buckets
                )
                timesteps = torch.full(
                    (batch_size,),
                    timestep,
                    dtype=torch.long,
                    device=vl_embeds.device,
                )
                action_features = action_head.action_encoder(actions, timesteps, embodiment_id)
                if action_head.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1],
                        dtype=torch.long,
                        device=vl_embeds.device,
                    )
                    action_features = action_features + action_head.position_embedding(pos_ids).unsqueeze(0)

                state_action_embeds = torch.cat((state_features, action_features), dim=1)
                if action_head.config.use_alternate_vl_dit:
                    model_output, _ = action_head.model(
                        hidden_states=state_action_embeds,
                        encoder_hidden_states=vl_embeds,
                        encoder_attention_mask=backbone_output.backbone_attention_mask,
                        timestep=timesteps,
                        return_all_hidden_states=True,
                        image_mask=backbone_output.image_mask,
                        backbone_attention_mask=backbone_output.backbone_attention_mask,
                    )
                else:
                    model_output, _ = action_head.model(
                        hidden_states=state_action_embeds,
                        encoder_hidden_states=vl_embeds,
                        encoder_attention_mask=backbone_output.backbone_attention_mask,
                        timestep=timesteps,
                        return_all_hidden_states=True,
                    )

                pred = action_head.action_decoder(model_output, embodiment_id)
                pred_velocity = pred[:, -actions.shape[1] :]
                actions = actions + dt * pred_velocity

        # Keep only the per-embodiment decoded horizon (len(action delta_indices)).
        decoded_horizon = int(getattr(self, "action_horizon", 0) or actions.shape[1])
        return actions[:, :decoded_horizon]

    def _normalize_state_batch(
        self,
        processor: StateActionProcessor,
        raw_state_batch: np.ndarray,
    ) -> np.ndarray:
        state_dict = self._slice_state_batch(raw_state_batch)
        normalized = processor.apply_state(state_dict, self.embodiment_tag)
        packed = np.concatenate([np.asarray(normalized[key]) for key in self.state_keys], axis=-1)
        # The action head's state_encoder is shared across embodiments and always
        # consumes max_state_dim, so training's GrootN1d7FeatureTransform._pack_state
        # zero-pads the per-embodiment state to that width. Mirror it here.
        max_state_dim = int(self.config.max_state_dim)
        if packed.shape[-1] < max_state_dim:
            padding = np.zeros(
                (*packed.shape[:-1], max_state_dim - packed.shape[-1]),
                dtype=packed.dtype,
            )
            packed = np.concatenate([packed, padding], axis=-1)
        elif packed.shape[-1] > max_state_dim:
            packed = packed[..., :max_state_dim]
        return packed


    def _decode_action_batch(
        self,
        processor: StateActionProcessor,
        action_batch: np.ndarray,
        raw_state_batch: np.ndarray,
    ) -> np.ndarray:
        decoded_samples = []
        for action_chunk, state_vec in zip(action_batch, raw_state_batch, strict=True):
            action_dict = self._split_action_chunk(processor, action_chunk)
            state_dict = self._slice_state_sample(state_vec)
            decoded_dict = processor.unapply_action(
                action_dict,
                self.embodiment_tag,
                state=state_dict,
            )
            decoded_samples.append(
                np.concatenate([np.asarray(decoded_dict[key]) for key in self.action_keys], axis=-1)
            )
        return np.stack(decoded_samples, axis=0)

    def _split_action_chunk(
        self,
        processor: StateActionProcessor,
        action_chunk: np.ndarray,
    ) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        start = 0
        norm_params = processor.norm_params[self.embodiment_tag]["action"]
        horizon = len(self.modality_config["action"].delta_indices)
        for key in self.action_keys:
            joint_dim = int(np.asarray(norm_params[key]["dim"]).item())
            out[key] = action_chunk[:horizon, start : start + joint_dim]
            start += joint_dim
        return out

    def _slice_state_batch(self, state_batch: np.ndarray) -> dict[str, np.ndarray]:
        return {
            key: state_batch[:, meta["start"] : meta["end"]]
            for key, meta in self.modality_meta["state"].items()
            if key in self.state_keys
        }

    def _slice_state_sample(self, state_vector: np.ndarray) -> dict[str, np.ndarray]:
        return {
            key: state_vector[meta["start"] : meta["end"]]
            for key, meta in self.modality_meta["state"].items()
            if key in self.state_keys
        }

    def _coerce_state_batch(self, state: Any, batch_size: int) -> np.ndarray:
        if state is None:
            if not self._predict_action_validation_zero_state:
                raise ValueError(
                    "GR00T-N1.7 predict_action requires a raw state matching the "
                    "configured embodiment_tag. Use validation_zero_state=True "
                    "only for local interface validation."
                )
            return np.zeros((batch_size, self.raw_state_dim), dtype=np.float32)

        if isinstance(state, dict):
            state_batch = self._flatten_state_dict(state)
        else:
            state_batch = _as_numpy(state)
            if state_batch.ndim == 1:
                state_batch = state_batch[None, :]
            elif state_batch.ndim != 2:
                raise ValueError(f"GR00T-N1.7 state must be [D] or [B, D], got {state_batch.shape}")

        if state_batch.shape[0] == 1 and batch_size > 1:
            state_batch = np.repeat(state_batch, batch_size, axis=0)
        if state_batch.shape[0] != batch_size:
            raise ValueError(
                f"GR00T-N1.7 state batch size {state_batch.shape[0]} does not match images batch {batch_size}"
            )

        if state_batch.shape[-1] < self.raw_state_dim:
            padding = np.zeros(
                (state_batch.shape[0], self.raw_state_dim - state_batch.shape[-1]),
                dtype=state_batch.dtype,
            )
            state_batch = np.concatenate([state_batch, padding], axis=-1)
        elif state_batch.shape[-1] > self.raw_state_dim:
            state_batch = state_batch[:, : self.raw_state_dim]
        return np.asarray(state_batch, dtype=np.float32)

    def _flatten_state_dict(self, state: dict[str, Any]) -> np.ndarray:
        values = []
        batch_size = None
        for key in self.state_keys:
            if key not in state:
                raise KeyError(f"Missing GR00T-N1.7 state group {key!r}")
            arr = _as_numpy(state[key])
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim != 2:
                raise ValueError(f"State group {key!r} must be [D] or [B, D], got {arr.shape}")
            batch_size = arr.shape[0] if batch_size is None else batch_size
            if arr.shape[0] != batch_size:
                raise ValueError("All GR00T-N1.7 state groups must share the same batch size")
            values.append(arr)
        return np.concatenate(values, axis=-1)

    def _normalize_image_batch(self, images: Any, instructions: Any) -> list[list[Image.Image]]:
        instruction_count = None if isinstance(instructions, str) else _safe_len(instructions)

        if _is_image_like(images):
            samples = [[images]]
        elif isinstance(images, dict):
            samples = [[images[key] for key in sorted(images)]]
        elif isinstance(images, (list, tuple)):
            if not images:
                raise ValueError("GR00T-N1.7 predict_action requires at least one image")
            if all(_is_image_like(item) for item in images):
                if instruction_count is not None and instruction_count == len(images) and len(images) > 1:
                    samples = [[item] for item in images]
                else:
                    samples = [list(images)]
            else:
                samples = []
                for sample in images:
                    if _is_image_like(sample):
                        samples.append([sample])
                    elif isinstance(sample, dict):
                        samples.append([sample[key] for key in sorted(sample)])
                    elif isinstance(sample, (list, tuple)):
                        if not sample:
                            raise ValueError("GR00T-N1.7 image samples must not be empty")
                        samples.append(list(sample))
                    else:
                        raise TypeError(
                            f"Unsupported GR00T-N1.7 image sample type: {type(sample).__name__}"
                        )
        else:
            raise TypeError(f"Unsupported GR00T-N1.7 images type: {type(images).__name__}")

        return [[_to_pil_image(image) for image in sample] for sample in samples]

    def _apply_eval_image_transform(
        self, image_batch: list[list[Image.Image]]
    ) -> list[list[Image.Image]]:
        """Apply the official crop/resize pipeline so callers only need to
        supply the env's native camera image (matching training-time
        ``GrootN1d7FeatureTransform._build_vlm_content``, which applies the
        same albumentations pipeline before assembling ``vlm_content``)."""
        transform = getattr(self, "_predict_action_eval_image_transform", None)
        if transform is None:
            return image_batch
        transformed = [
            [
                Image.fromarray(transform(image=np.asarray(image.convert("RGB")))["image"])
                for image in sample
            ]
            for sample in image_batch
        ]
        self._log_eval_image_transform_once(image_batch, transformed)
        return transformed

    def _log_eval_image_transform_once(
        self,
        before: list[list[Image.Image]],
        after: list[list[Image.Image]],
    ) -> None:
        """Log per-view pixel stats for the first transformed batch.

        The training path builds its images through ``apply_with_replay`` and a
        CHW tensor stack while eval feeds HWC arrays straight into the same
        albumentations pipeline. Those two routes should agree, but a silent
        divergence here is invisible to static review (the transform exists and
        is called on both sides), so dump the actual values once per process for
        value-by-value comparison against a training batch.
        """
        if getattr(self, "_predict_action_image_debug_logged", False):
            return
        # The server warmup feeds an all-zero dummy image; logging that would burn
        # the one-shot on a frame that proves nothing. Wait for a real observation.
        if all(
            not np.asarray(raw.convert("RGB")).any()
            for sample in before
            for raw in sample
        ):
            return
        self._predict_action_image_debug_logged = True
        for sample_idx, (raw_sample, out_sample) in enumerate(zip(before, after, strict=True)):
            for view_idx, (raw, out) in enumerate(zip(raw_sample, out_sample, strict=True)):
                raw_arr = np.asarray(raw.convert("RGB"))
                out_arr = np.asarray(out)
                logger.info(
                    "GR00T-N1.7 eval image transform sample=%d view=%d: "
                    "in shape=%s dtype=%s mean=%.2f std=%.2f min=%d max=%d -> "
                    "out shape=%s dtype=%s mean=%.2f std=%.2f min=%d max=%d",
                    sample_idx,
                    view_idx,
                    raw_arr.shape,
                    raw_arr.dtype,
                    float(raw_arr.mean()),
                    float(raw_arr.std()),
                    int(raw_arr.min()),
                    int(raw_arr.max()),
                    out_arr.shape,
                    out_arr.dtype,
                    float(out_arr.mean()),
                    float(out_arr.std()),
                    int(out_arr.min()),
                    int(out_arr.max()),
                )

    @staticmethod
    def _normalize_instruction_batch(instructions: Any, batch_size: int) -> list[str]:
        if isinstance(instructions, str):
            return [instructions] * batch_size
        instruction_list = [str(item) for item in list(instructions)]
        if len(instruction_list) == 1 and batch_size > 1:
            instruction_list = instruction_list * batch_size
        if len(instruction_list) != batch_size:
            raise ValueError(
                f"GR00T-N1.7 instruction batch size {len(instruction_list)} does not match images batch {batch_size}"
            )
        return instruction_list

    @staticmethod
    def _build_vlm_content(images: Iterable[Image.Image], instruction: str) -> dict[str, Any]:
        pil_images = [image.convert("RGB") for image in images]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in pil_images],
                    {"type": "text", "text": _formalize_language(instruction)},
                ],
            }
        ]
        return {"text": None, "images": pil_images, "conversation": conversation}

    def _get_predict_action_processor(self, dataset_stats: Dict[str, Any] | None) -> StateActionProcessor:
        if dataset_stats is None:
            if self._predict_action_default_processor is None:
                raise RuntimeError(
                    "GR00T-N1.7 predict_action has not been configured with checkpoint statistics. "
                    "Call configure_predict_action(...) before predict_action(..., dataset_stats=None)."
                )
            return self._predict_action_default_processor
        statistics = self._coerce_statistics(dataset_stats)
        cache_key = json.dumps(statistics, sort_keys=True)
        if cache_key not in self._predict_action_processor_cache:
            self._predict_action_processor_cache[cache_key] = self._build_state_action_processor(statistics)
        return self._predict_action_processor_cache[cache_key]

    def _coerce_statistics(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        if self.embodiment_tag in statistics:
            embodiment_stats = statistics[self.embodiment_tag]
            if not {"state", "action"}.issubset(embodiment_stats):
                raise ValueError(
                    f"Statistics for {self.embodiment_tag!r} must include 'state' and 'action'."
                )
            return {self.embodiment_tag: embodiment_stats}

        if "observation.state" in statistics and "action" in statistics:
            return convert_lerobot_stats_to_groot_n1d7_format(
                statistics,
                self.embodiment_tag,
                modality_meta=self.modality_meta,
                modality_config=self.modality_config,
            )

        raise ValueError(
            "GR00T-N1.7 statistics must be either checkpoint-style "
            "{embodiment: {state, action, relative_action}} or raw LeRobot "
            "{'observation.state', 'action', 'relative_action'} stats."
        )

    def _build_state_action_processor(self, statistics: Dict[str, Any]) -> StateActionProcessor:
        processor = StateActionProcessor(
            modality_configs={self.embodiment_tag: self.modality_config},
            statistics=statistics,
            use_percentiles=self._predict_action_use_percentiles,
            clip_outliers=self._predict_action_clip_outliers,
            apply_sincos_state_encoding=False,
            use_relative_action=True,
        )
        processor.eval()
        return processor

    @staticmethod
    def _to_embodiment_tensor(value: Any, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            return torch.full((batch_size,), int(value), dtype=torch.long, device=device)
        value = value.to(device=device, dtype=torch.long)
        if value.ndim == 0:
            return value.unsqueeze(0).expand(batch_size)
        if value.ndim > 1:
            value = value.flatten()
        if value.shape[0] == 1 and batch_size > 1:
            return value.expand(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(f"embodiment_id batch size {value.shape[0]} does not match {batch_size}")
        return value

    def _ensure_predict_action_collator(self) -> None:
        if self.collator is not None:
            return
        self.collator = Gr00tN1d7DataCollator(
            model_name=self.config.model_name,
            model_type=self.config.backbone_model_type,
            transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
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


def _restore_module_params_fp32(module: nn.Module) -> None:
    for parameter in module.parameters():
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


def _formalize_language(instruction: Any) -> str:
    """Lowercase and strip punctuation, matching the training-time transform.

    ``GrootN1d7FeatureTransform`` applies ``re.sub(r"[^\\w\\s]", "", text.lower())``
    when ``formalize_language`` is set (true for every released GR00T-N1.7
    checkpoint), so the eval path must normalize instructions identically or the
    prompt tokens drift from what the checkpoint was trained on.
    """
    return re.sub(r"[^\w\s]", "", str(instruction).lower())


def _is_image_like(value: Any) -> bool:
    return isinstance(value, (Image.Image, np.ndarray, torch.Tensor))

def _to_pil_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = _as_numpy(image)
    if arr.ndim != 3:
        raise ValueError(f"GR00T-N1.7 images must be rank-3, got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"GR00T-N1.7 images must have 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        if finite.size and finite.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return None
