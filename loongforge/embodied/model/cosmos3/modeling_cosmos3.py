# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Cosmos (NVIDIA cosmos-framework) under the OpenMDW-1.1 License.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3 architecture: MoT video generation SFT via rectified flow matching.

Supports two batch formats:
1. Cosmos3Batch (from cosmos3_dataset.py): raw videos, online VAE encode in forward
2. Legacy PackedBatch: pre-packed PackedSequence (for smoke tests)
"""

import json
import logging
import os
import pathlib as _pathlib_mod
import pickle
import sys
import types
from typing import Dict

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint import FileSystemReader
from transformers import AutoTokenizer
from accelerate import init_on_device

from loongforge.embodied.model.cosmos3.cosmos3_vfm_network import (
    Cosmos3VFMNetwork,
    Cosmos3VFMNetworkConfig,
)
from loongforge.embodied.model.registry import register_model
from loongforge.embodied.model.cosmos3.modeling_configuration_cosmos3 import Cosmos3ModelConfig
from loongforge.embodied.model.cosmos3.flow_matching import compute_flow_matching_loss
from loongforge.embodied.model.cosmos3.modeling_utils import has_noisy_tokens
from loongforge.embodied.model.cosmos3.rectified_flow import RectifiedFlow
from loongforge.embodied.model.cosmos3.sequence_packing import (
    PackedSequence, SequencePlan, pack_input_sequence, add_special_tokens,
)
from loongforge.embodied.model.cosmos3.data_and_condition import GenerationDataClean
from loongforge.embodied.model.cosmos3.unified_mot import Qwen3VLMoTConfig, Qwen3VLTextForCausalLM
from loongforge.embodied.model.cosmos3.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface
from loongforge.embodied.data.datasets.cosmos3.transforms.cosmos3_preprocessor import Cosmos3Batch

logger = logging.getLogger(__name__)


@register_model("cosmos3")
class Cosmos3(nn.Module):
    """Cosmos3: MoT (Mixture of Transformers) video generation model.

    Supports online VAE encoding: receives raw videos, encodes to latents,
    then runs noise injection + packing + MoT forward + flow matching loss.
    """

    @classmethod
    def from_pretrained(cls, cfg) -> "Cosmos3":
        """from_pretrained."""
        return cls(cfg)

    def __init__(self, config: Cosmos3ModelConfig):
        """__init__."""
        super().__init__()
        # Build language model (MoT-wrapped Qwen3-VL)
        qwen3_vl_path = config.qwen3_vl_path
        with open(os.path.join(qwen3_vl_path, "config.json")) as f:
            hf_cfg = json.load(f)
        text_cfg_dict = hf_cfg.get("text_config", hf_cfg)
        mot_config = Qwen3VLMoTConfig({"text_config": text_cfg_dict},
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
        )
        # Meta-init: allocate all params on "meta" device (no storage, ~ms).
        # Materialization happens post-FSDP wrap via self.materialize(); weights
        # are then loaded by self.load_pretrained() into the sharded DTensors.
        # buffers (include_buffers=False) stay on CPU so register_buffer values
        # like dim_spatial_range remain materialized for shape-only use during init.
        with init_on_device("meta", include_buffers=False):
            language_model = Qwen3VLTextForCausalLM(mot_config)

        # Build VFM network config
        latent_channel_size = config.latent_channel_size
        latent_patch_size = config.latent_patch_size
        self.latent_patch_size = latent_patch_size
        self.latent_channel_size = latent_channel_size
        self.position_embedding_type = config.position_embedding_type
        self.enable_fps_modulation = config.enable_fps_modulation
        self.base_fps = config.base_fps
        self.unified_3d_mrope_reset_spatial_ids = config.unified_3d_mrope_reset_spatial_ids
        self.unified_3d_mrope_temporal_modality_margin = config.unified_3d_mrope_temporal_modality_margin
        self.encode_exact_durations = config.encode_exact_durations

        vfm_config = Cosmos3VFMNetworkConfig(
            vision_gen=config.vision_gen,
            action_gen=config.action_gen,
            sound_gen=config.sound_gen,
            vlm_config=mot_config.text_config,
            latent_patch_size=latent_patch_size,
            latent_channel_size=latent_channel_size,
            latent_downsample_factor=config.latent_downsample_factor,
            position_embedding_type=self.position_embedding_type,
            max_latent_h=config.max_latent_h,
            max_latent_w=config.max_latent_w,
            max_latent_t=config.max_latent_t,
            joint_attn_implementation=config.joint_attn_implementation,
            action_dim=config.max_action_dim,
            num_embodiment_domains=config.num_embodiment_domains,
            temporal_compression_factor_vision=config.vae_temporal_compression,
            enable_fps_modulation=self.enable_fps_modulation,
            base_fps=self.base_fps,
        )
        with init_on_device("meta", include_buffers=False):
            self.net = Cosmos3VFMNetwork(language_model, vfm_config)

        # Rectified flow scheduler
        self.rectified_flow = RectifiedFlow(
            velocity_field=None,
            train_time_distribution=config.train_time_distribution,
            train_time_weight_method=config.train_time_weight_method,
            shift=config.shift,
        )
        # Store raw shift so analytic formula can pick it up at training time.
        self.shift = float(config.shift)

        self.action_loss_weight = config.action_loss_weight

        # VAE encoder (loaded lazily on first forward to avoid init-time weight loading)
        self._vae_encoder = None
        self._vae_path = config.vae_path

        # Special tokens (initialized lazily)
        self._special_tokens = None
        self._tokenizer_path = qwen3_vl_path

        ## Training config
        self.train_modules = config.train_modules
        self.keys_to_skip_loading = config.keys_to_skip_loading

    def freeze_modules(self):
        """Freeze understanding pathway parameters (text/non-gen weights)."""
        for name, param in self.net.named_parameters():
            if not any(k in name for k in self.train_modules):
                param.requires_grad = False

    def _get_vae_encoder(self):
        """Lazy-load VAE encoder on first use. Loads on the model's CUDA device
        (matches cosmos's tokenizer_vision_gen GPU placement) so encoding does
        not become a CPU-bound bottleneck. Frozen + no_grad keeps peak activation
        memory small relative to 80GB headroom.
        """
        if self._vae_encoder is None:
            self._vae_encoder = Wan2pt2VAEInterface(
                vae_path=self._vae_path,
                encode_exact_durations=self.encode_exact_durations
            )
        return self._vae_encoder

    def _get_special_tokens(self):
        """Lazy-init special tokens dict."""
        if self._special_tokens is None:
            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)
            tokenizer, new_tokens = add_special_tokens(tokenizer)
            self._special_tokens = new_tokens
            self._special_tokens["eos_token_id"] = tokenizer.eos_token_id
        return self._special_tokens

    @property
    def backbone(self) -> nn.Module:
        """Return the underlying language-model backbone."""
        return self.net.language_model

    def materialize(self, device):
        """Allocate empty CUDA tensors for meta params (post-FSDP shard) then run
        each submodule's init_weights to populate them. Mirrors cosmos's
        ``hf_model._apply(empty_like, ...) + tie_embeddings``.
        """
        self.net.to_empty(device=device)
        self.net.init_weights(buffer_device=device)
        logger.info("Cosmos3 materialized on %s", device)

    def load_pretrained(self, path: str, device=None):
        """Load pretrained weights from DCP or standard format."""
        # Resolve model subdir
        model_dir = os.path.join(path, "model")

        if 'pathlib._local' not in sys.modules:
            _local = types.ModuleType('pathlib._local')
            _local.PosixPath = _pathlib_mod.PosixPath
            sys.modules['pathlib._local'] = _local

        logger.info(f"Loading Cosmos3 DCP checkpoint from: {model_dir}")

        # Checkpoint keys have 'net.' prefix — load into full model state_dict
        # Only include keys that exist in checkpoint to avoid strict key errors
        metadata_path = os.path.join(model_dir, ".metadata")
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        ckpt_keys = set(metadata.state_dict_metadata.keys())

        full_sd = self.state_dict()
        state_dict = {k: v for k, v in full_sd.items() if k in ckpt_keys}

        # Skip the action-head weights on checkpoint load
        _before = len(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if not any(s in k for s in self.keys_to_skip_loading)}
        logger.info(f"DCP: skipping {_before - len(state_dict)} action-head keys on load (kept fresh init)")

        storage_reader = FileSystemReader(model_dir)
        dcp.load(state_dict, storage_reader=storage_reader)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(f"DCP: {len(missing)} missing keys (first 5: {missing[:5]})")
        if unexpected:
            logger.warning(f"DCP: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
        logger.info("Cosmos3 DCP checkpoint loaded successfully")

    def encode(self, images, instructions, **kwargs):
        """Not implemented; call ``forward`` directly during training."""
        raise NotImplementedError("Use forward() directly.")

    def forward(self, batch: Cosmos3Batch, iteration=None, **kwargs) -> Dict[str, torch.Tensor]:
        """Training forward: online VAE encode → noise → pack → MoT → loss.

        Accepts Cosmos3Batch (raw videos) from cosmos3_dataset.py.
        """
        if batch.actions is not None:
            return self._forward_raw_video_with_action(batch, iteration=iteration)
        else:
            raise NotImplementedError

    def _remove_padding_from_latent(
            self, x0_tokens_vision: list[torch.Tensor], frame_size: list[torch.Tensor], spatial_factor=16
        ) -> list[torch.Tensor]:
        """
        Remove reflection padding from encoded latent vision tokens.
        """
        batch_size = len(x0_tokens_vision)
        cropped_latents = []
        for i in range(batch_size):
            fs = frame_size[i]
            if fs.dim() == 2:
                fs = fs[0]
            orig_h = int(fs[2].item())
            orig_w = int(fs[3].item())
            if orig_h // spatial_factor == 0 or orig_w // spatial_factor == 0:
                logger.warning(
                    f"Zero-sized latent found: orig_h: {orig_h}, orig_w: {orig_w}, spatial_factor: {spatial_factor}"
                )
            orig_h_latent = max(orig_h // spatial_factor, 1)
            orig_w_latent = max(orig_w // spatial_factor, 1)
    
            cropped_latent = x0_tokens_vision[i][:, :, :, :orig_h_latent, :orig_w_latent].contiguous()
            cropped_latents.append(cropped_latent)
        return cropped_latents
    
    
    def _forward_raw_video_with_action(self, batch, iteration=None) -> Dict[str, torch.Tensor]:
        """Action-policy training pipeline (vision + action joint flow matching).

        Mirrors :meth:`_forward_raw_video` but additionally:
        1. injects per-frame Gaussian noise into the action tokens (skipping
           conditioning frames via the SequencePlan condition_mask),
        2. forwards both noised action and noised vision tokens through the
           MoT,
        3. aggregates a vision + action flow-matching loss with
           ``raw_action_dim`` masking on the padded action channels.
        """
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype
        vae = self._get_vae_encoder()
        special_tokens = self._get_special_tokens()
        cpu_kwargs = {"device": "cpu", "dtype": torch.float32}

        B = len(batch.videos)
        action_loss_weight = float(self.action_loss_weight)

        # 1. VAE encode each video on the model device (GPU).
        latents = [vae.encode(
                video.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0).unsqueeze(0)
            ).contiguous().float() for video in batch.videos]
        if hasattr(batch, 'image_sizes'):
            latents = self._remove_padding_from_latent(latents, batch.image_sizes)
        latents = [latent[0] for latent in latents]
        
        # 2. Sample timesteps + sigmas — analytic cosmos shift, iteration-seeded.
        t = self.rectified_flow.sample_train_time(B, iteration=iteration)
        timesteps, sigmas = self.rectified_flow.get_timesteps_sigmas_from_shift(
            t, shift=self.shift, max_timestep=self.rectified_flow.noise_scheduler.config.num_train_timesteps,
            tensor_kwargs={"device": "cuda", "dtype": torch.float32},
        )

        # 3. Vision noise injection + velocity targets.
        # Build per-frame condition masks (matches cosmos noisy_mask_vision behavior).
        # cosmos: sigmas_vision[i] = sigmas[i] * (1 - cond_mask), so conditioning frames stay clean.
        condition_masks_vision = []
        for plan, lat in zip(batch.sequence_plans, latents):
            T_lat = lat.shape[1]  # [C, T, H, W]
            cm = torch.zeros(T_lat, 1, 1, device=device, dtype=torch.float32)
            for fi in plan.condition_frame_indexes_vision:
                if 0 <= fi < T_lat:
                    cm[fi] = 1.0
            condition_masks_vision.append(cm)

        # Seeded noise generator (matches cosmos omni_mot_model.py).
        # Offset +32768 distinguishes noise seed from sigma seed in sample_train_time.
        _noise_gen = None
        if iteration is not None and torch.are_deterministic_algorithms_enabled():
            _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            _noise_gen = torch.Generator(device=device)
            _noise_gen.manual_seed(int(iteration) * 65536 + _rank + 32768)

        noised_latents: list[torch.Tensor] = []
        vt_targets_vision: list[torch.Tensor] = []

        for i, latent in enumerate(latents):
            # Compute interpolation in fp32 then cast to model dtype (matches cosmos behavior).
            latent_fp32 = latent.float()
            noise = torch.randn(latent.shape, device=latent.device, dtype=torch.float32, generator=_noise_gen)
            noisy_mask_v = 1.0 - condition_masks_vision[i]  # [T, 1, 1] fp32
            sigma_i = sigmas[i].to(device=device, dtype=torch.float32).view(-1, 1, 1)  # [1,1,1] in base mode
            sigma_i = sigma_i * noisy_mask_v  # [T, 1, 1] — broadcasts across [C, T, H, W]
            x_t = sigma_i * noise + (1.0 - sigma_i) * latent_fp32  # fp32
            noised_latents.append(x_t.to(dtype=dtype).unsqueeze(0))
            vt_targets_vision.append((noise - latent_fp32).unsqueeze(0))  # fp32, no masking — matches cosmos

        # 4. Action noise injection. Conditioning frames are kept clean by zeroing
        #    sigma along their condition_mask entries (matches cosmos's
        #    sigmas_action = sigma * (1 - condition_mask)). The padded channels
        #    beyond raw_action_dim are zeroed out to keep loss masking correct.
        actions_clean: list[torch.Tensor] = [a.to(device=device, dtype=torch.float32) for a in batch.actions]
        raw_action_dims: list[torch.Tensor] = [d.to(device) for d in batch.raw_action_dims]
        action_domain_ids: list[torch.Tensor] = [d.to(device) for d in batch.action_domain_ids]

        # condition_mask shape per cosmos: [T, 1] with 1 = clean, 0 = noisy.
        condition_masks_action = []
        for plan, act in zip(batch.sequence_plans, actions_clean):
            T = act.shape[0]
            cm = torch.zeros(T, 1, device=device, dtype=torch.float32)
            for fi in plan.condition_frame_indexes_action:
                if 0 <= fi < T:
                    cm[fi] = 1.0
            condition_masks_action.append(cm)

        noised_actions: list[torch.Tensor] = []
        vt_targets_action: list[torch.Tensor] = []
        for i, act in enumerate(actions_clean):
            T, D = act.shape
            eps = torch.randn(T, D, device=device, dtype=torch.float32, generator=_noise_gen)
            sigma_i = sigmas[i].to(device=device, dtype=torch.float32)
            sigma_i = sigma_i.view(1, 1).expand(T, 1) * (1.0 - condition_masks_action[i])  # [T, 1]
            xt = (sigma_i * eps + (1.0 - sigma_i) * act).float()                                     # [T, D]
            vt = eps - act
            raw_dim = int(raw_action_dims[i].item())
            if raw_dim < D:
                xt[:, raw_dim:] = 0.0
                # NOTE: cosmos does NOT zero vt tail dims; only xt is zeroed.
            noised_actions.append(xt.to(dtype=dtype))
            vt_targets_action.append(vt)  # [T, D] — matches cosmos reference

        # 5. Build GenerationDataClean with both vision and action.
        is_image = all(lat.shape[2] == 1 for lat in noised_latents)
        fps_vision = torch.tensor(batch.fps_values, dtype=torch.float32)
        fps_action = torch.tensor(batch.fps_values, dtype=torch.float32)

        gen_data_clean = GenerationDataClean(
            batch_size=B,
            is_image_batch=is_image,
            x0_tokens_vision=noised_latents,
            fps_vision=fps_vision,
            x0_tokens_action=noised_actions,
            fps_action=fps_action,
            action_domain_id=action_domain_ids,
            raw_action_dim=raw_action_dims,
        )

        input_timesteps = timesteps.unsqueeze(1)  # [B, 1]
        packed_seq = pack_input_sequence(
            sequence_plans=batch.sequence_plans,
            input_text_indexes=batch.text_token_ids,
            gen_data_clean=gen_data_clean,
            input_timesteps=input_timesteps.cpu(),
            special_tokens=special_tokens,
            latent_patch_size=self.latent_patch_size,
            position_embedding_type=self.position_embedding_type,
            unified_3d_mrope_reset_spatial_ids=self.unified_3d_mrope_reset_spatial_ids,
            unified_3d_mrope_temporal_modality_margin=self.unified_3d_mrope_temporal_modality_margin,
            enable_fps_modulation=self.enable_fps_modulation,
            base_fps=self.base_fps,
        )

        # 6. Forward through MoT.
        packed_seq.to_cuda()
        output_dict = self.net(packed_seq, fps_vision=fps_vision.to(device), fps_action=fps_action.to(device))

        # 7. Vision + action flow-matching losses.
        vision = packed_seq.vision
        action = packed_seq.action
        ts_dev = input_timesteps.to(device)
        loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)

        if has_noisy_tokens(vision) and vt_targets_vision:
            loss_vision, _ = compute_flow_matching_loss(
                pred=output_dict["preds_vision"],
                target=vt_targets_vision,
                condition_mask=vision.condition_mask,
                timesteps=ts_dev,
                has_valid_tokens=True,
                rectified_flow=self.rectified_flow,
                tensor_kwargs_fp32={"device": device, "dtype": torch.float32},
            )
            loss = loss + loss_vision

        if action is not None and has_noisy_tokens(action):
            loss_action, _ = compute_flow_matching_loss(
                pred=output_dict["preds_action"],
                target=vt_targets_action,
                condition_mask=action.condition_mask,
                timesteps=ts_dev,
                has_valid_tokens=True,
                rectified_flow=self.rectified_flow,
                tensor_kwargs_fp32={"device": device, "dtype": torch.float32},
                raw_action_dim=raw_action_dims,
            )
            loss = loss + action_loss_weight * loss_action
        else:
            # Keep action heads in the backward graph so FSDP/DDP sync stays consistent.
            dummy = 0.0 * sum(p.sum() for p in output_dict.get("preds_action", []))
            loss = loss + dummy

        return loss, {"vision_loss": loss_vision, "action_loss": loss_action}


    def predict_action(self, **kwargs) -> Dict[str, np.ndarray]:
        """Action prediction (inference) - not yet implemented for Cosmos3."""
        raise NotImplementedError("Cosmos3 predict_action not yet implemented.")
