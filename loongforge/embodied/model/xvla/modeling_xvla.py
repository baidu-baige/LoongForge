# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from X-VLA (https://github.com/2toinf/X-VLA).
# Copyright 2025 2toINF. All rights reserved.
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
"""X-VLA model implementation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import os
import torch

from pathlib import Path
from safetensors.torch import load_file
from transformers import PreTrainedModel
from .modeling_florence2 import Florence2ForConditionalGeneration
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .model_configuration_xvla import XVLAConfig, resolve_domain_id
from loongforge.embodied.model.registry import register_model
from loongforge.embodied.model.xvla.xvla_processor import (
    XVLATokenizerCore,
    XVLAImageProcessorCore,
)


class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • Florence2 encoder-only backbone (vision-language)
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing + loss)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        """
        Initialize XVLA model components.

        Builds the Florence2 backbone (encoder only, decoder/lm_head removed),
        the SoftPromptedTransformer action head, and the action space for the
        specified action_mode (e.g., 'ee6d', 'joint', 'auto').
        """
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        # Action space (dimensions + hooks)
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = self.action_space.dim_proprio

        # Florence2 backbone (encoder only)
        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        lm = self.vlm.language_model
        del lm.model.decoder
        del lm.lm_head

        projection_dim = self.vlm.config.projection_dim

        # Temporal/action head
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
            attn_dropout=config.attn_dropout,
            mlp_dropout=config.mlp_dropout,
        )

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via Florence2 encoder.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)         # [B*V]
        flat_images = pixel_values.flatten(0, 1)                # [B*V, C, H, W]

        all_feats = self.vlm._encode_image(flat_images)         # [B*V, N, D]
        N, D = all_feats.shape[1:]
        image_features = torch.where(
            flat_mask.view(-1, 1, 1), all_feats, all_feats.new_zeros(())
        ).view(B, V, N, D)                                       # [B, V, N, D]

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)  # [B, L, D]

        merged_embeds, _attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # first view: [B, N, D]
            inputs_embeds,         # [B, L, D]
        )
        # ``_merge_input_ids_with_image_features`` constructs the mask with
        # ``torch.ones(...)`` for both image and prefix tokens, so it is
        # provably all-ones on this path. Passing it into the Florence2
        # encoder triggers ``0 in attention_mask`` (line 1815 of
        # ``modeling_florence2.py``), which is a Python ``__contains__``
        # that forces a host sync (``aten::is_nonzero`` on a scalar,
        # ~200ms/step in NSys). Since the mask is always all-ones we hand
        # ``None`` to the encoder — mathematically equivalent (adding a
        # 4D zero bias is identity) and sync-free.

        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=None,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # remaining views flattened
        return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,  # [B, T=num_actions, D=dim_action]
        valid_mask: torch.Tensor = None,  # [B] bool; None -> all valid
    ) -> Dict[str, torch.Tensor]:
        """
        1) Encode multimodal inputs.
        2) Diffusion-style noisy mixture of actions: x_t = t*noise + (1-t)*gt.
        3) Space-specific preprocessing, prediction, and supervised loss.
        """
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)

        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        return self.action_space.compute_loss(pred_action, action, valid_mask=valid_mask)

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        enc = self.forward_vlm(input_ids, image_input, image_mask)

        B = input_ids.shape[0]
        D = self.action_space.dim_action

        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)


# ═══════════════════════════════════════════════════════════════
# XVLAPolicy — LoongForge trainer entry point
# ═══════════════════════════════════════════════════════════════


@register_model("xvla")
class XVLAPolicy(torch.nn.Module):
    """LoongForge policy wrapper around XVLA, aligned with PI05Policy interface."""

    def __init__(self, config: XVLAConfig):
        """
        Initialize the XVLAPolicy wrapper.

        Wraps the XVLA model and stores config-derived settings
        (image_size, num_image_views) for use during training and inference.
        """
        super().__init__()
        self.config = config
        self.model = XVLA(config)

        if config.enable_torch_compile:
            # Also fix the last batch: keep a stable, static shape so guards
            # do not trigger a recompile at the tail. The trainer feeds the
            # same per_device_batch_size on every step in this workload, but
            # end-of-epoch tail batches could otherwise differ.
            compiled = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
                dynamic=False,
            )
        else:
            compiled = self.model
        # Hold the (optionally compiled) forward entry WITHOUT registering it as
        # a submodule. It either *is* self.model or wraps it via
        # OptimizedModule._orig_mod, so its parameters are already covered by the
        # "model.*" keys. If assigned as a normal nn.Module attribute, every
        # parameter would be duplicated in state_dict() under "_compiled_model.*"
        # (or "_compiled_model._orig_mod.*"), bloating checkpoints and producing
        # spurious "missing keys" reports on load_pretrained. object.__setattr__
        # bypasses nn.Module submodule registration; the parameters remain
        # trainable via the registered "model" submodule.
        object.__setattr__(self, "_compiled_model", compiled)
        self._processor_path = ""
        self._num_image_views = int(getattr(config, "num_image_views", 3) or 3)
        self._tokenize_transform = None
        self._image_transform = None

    def _prepare_inputs(self, batch) -> Dict[str, torch.Tensor]:
        """Map an XVLAProcessor-style batch dict into XVLA model forward arguments.

        The XVLA collator (``XVLAPreprocessor``) emits X-VLA-native batch keys via
        :class:`XVLAProcessor`:
          - ``input_ids``    token ids [B, L]
          - ``image_input``  multi-view images [B, V, 3, H, W]
          - ``image_mask``   per-view validity mask [B, V]
          - ``proprio``      proprio state [B, state_dim]
          - ``domain_id``    domain index [B]
          - ``action``       target action chunk [B, T, D]
        """
        action = batch["action"]
        device = action.device

        image_input = batch["image_input"].to(device)  # [B, V, 3, H, W]
        B = image_input.shape[0]

        image_mask = batch.get("image_mask")
        if image_mask is None:
            image_mask = torch.ones(
                image_input.shape[:2], dtype=torch.bool, device=device
            )
        else:
            image_mask = image_mask.to(device).bool()

        input_ids = batch.get("input_ids")
        if input_ids is None:
            input_ids = batch["observation.language.tokens"]
        input_ids = input_ids.to(device)

        proprio = batch.get("proprio")
        if proprio is None:
            proprio = batch["observation.state"]
        proprio = proprio.to(device)
        # Pad/trim proprio to the action-space dimension expected by the
        # transformer (EE6D uses 20). Dataset state may be lower-dim (e.g. 8).
        target_proprio_dim = self.model.action_space.dim_action
        cur_dim = proprio.shape[-1]
        if cur_dim < target_proprio_dim:
            proprio = torch.nn.functional.pad(proprio, (0, target_proprio_dim - cur_dim))
        elif cur_dim > target_proprio_dim:
            proprio = proprio[..., :target_proprio_dim]

        domain_id = batch.get("domain_id")
        if domain_id is None:
            domain_id = torch.zeros(B, dtype=torch.long, device=device)
        else:
            domain_id = domain_id.to(device)

        action = action.to(device)

        return {
            "input_ids": input_ids,
            "image_input": image_input,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
            "action": action,
        }

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """Training forward pass; returns the action-space loss dict.

        The action space returns per-component losses (e.g. position/rotate6D/
        gripper). The trainer expects a single ``action_loss`` key, so we sum the
        components and expose it alongside the individual terms for logging.
        """
        inputs = self._prepare_inputs(batch)
        loss_dict = self._compiled_model(**inputs)
        total = sum(v for v in loss_dict.values() if torch.is_tensor(v))
        loss_dict = {**loss_dict, "action_loss": total}
        return total, loss_dict

    @torch.no_grad()
    def predict_action_chunk(
        self,
        input_ids: torch.Tensor,
        image_input: torch.Tensor,
        image_mask: torch.Tensor,
        domain_id: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> np.ndarray:
        """Predict an action chunk from a preprocessed XVLA batch.

        Low-level entry point for callers that already have model-ready tensors
        (matching :meth:`XVLA.generate_actions`).
        """
        self.eval()
        actions = self.model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
            steps=steps,
        )
        return actions.float().cpu().numpy()

    def _get_tokenize_transform(self):
        """Lazily build (and cache) the Florence2 tokenize core.

        Uses :class:`XVLATokenizerCore` from the model package rather than the
        ``BaseTransform`` wrapper in ``loongforge.embodied.data``, so
        ``predict_action`` does not pull in the training-side DataLoader /
        DistributedContext import chain.
        """
        cached = getattr(self, "_tokenize_transform", None)
        if cached is None:
            cached = XVLATokenizerCore(tokenizer_path=self._processor_path)
            self._tokenize_transform = cached
        return cached

    def _get_image_transform(self):
        """Lazily build (and cache) the Florence2 image encode core.

        Uses :class:`XVLAImageProcessorCore` from the model package rather than
        the ``BaseTransform`` wrapper in ``loongforge.embodied.data``, so
        ``predict_action`` does not pull in the training-side DataLoader /
        DistributedContext import chain.
        """
        cached = getattr(self, "_image_transform", None)
        if cached is None:
            cached = XVLAImageProcessorCore(
                tokenizer_path=self._processor_path,
                num_views=self._num_image_views,
            )
            self._image_transform = cached
        return cached

    @torch.no_grad()
    def predict_action(
        self,
        images,
        instructions,
        state=None,
        dataset_stats=None,
        domain_id=None,
    ) -> np.ndarray:
        """Eval-facing entry point matching the shared ``predict_action`` interface.

        Args:
            images: Batched images. Either a list of length ``B`` of per-sample
                    view lists (each view a CHW tensor / HWC ndarray / PIL
                    image), or a flat list of length ``B`` of single-view
                    images (auto-wrapped into ``[[img]] * B``).
            instructions: List[str] of language instructions (length ``B``).
            state: Optional proprio state. Accepts ``None``, a 1-D vector, or a
                    ``[B, D_state]`` tensor / ndarray / list. When ``None`` a
                    zero proprio of the action-space's expected dim is used.
            dataset_stats: Unused for XVLA (its action space handles
                    normalization internally). Kept for interface compatibility.

        Returns:
            ndarray of shape ``[B, num_actions, dim_action]``.
        """
        # XVLA does not consume dataset_stats because action normalization is handled by action_space.
        del dataset_stats
        self.eval()
        device = next(self.parameters()).device

        if not isinstance(images, (list, tuple)) or len(images) == 0:
            raise ValueError("predict_action: images must be a non-empty list.")
        if not isinstance(images[0], (list, tuple)):
            images = [[img] for img in images]
        B = len(images)

        if not isinstance(instructions, (list, tuple)):
            instructions = [instructions]
        if len(instructions) != B:
            raise ValueError(
                f"predict_action: got {B} image samples but {len(instructions)} instructions."
            )

        image_tf = self._get_image_transform()
        pil_batch = [[image_tf._to_pil(im) for im in views] for views in images]
        encoded = image_tf.encode_image_batch(pil_batch)
        image_input = encoded["image_input"].to(device)
        image_mask = encoded["image_mask"].to(device)

        tok = self._get_tokenize_transform().encode_language_batch(
            [str(t) for t in instructions]
        )
        input_ids = tok["input_ids"].to(device)

        dim_proprio = int(self.model.action_space.dim_proprio)
        if state is None:
            proprio = torch.zeros(B, dim_proprio, dtype=torch.float32, device=device)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            state = state.to(dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.shape[0] != B:
                raise ValueError(
                    f"predict_action: state batch {state.shape[0]} != images batch {B}."
                )
            cur = state.shape[-1]
            if cur < dim_proprio:
                state = torch.nn.functional.pad(state, (0, dim_proprio - cur))
            elif cur > dim_proprio:
                state = state[..., :dim_proprio]
            proprio = state.to(device)

        domain_value = resolve_domain_id(getattr(self.config, "robot_type", ""))
        domain_id = torch.full((B,), domain_value, dtype=torch.long, device=device) if domain_id is None else domain_id

        actions = self.model.generate_actions(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
            domain_id=domain_id,
            proprio=proprio,
        )
        return actions.float().cpu().numpy()

    @classmethod
    def from_pretrained(cls, config_or_path, processor_path=None) -> "XVLAPolicy":
        """Create XVLAPolicy from XVLAConfig, a pretrained path string, or a config dict."""
        if isinstance(config_or_path, XVLAConfig):
            cfg = config_or_path
            pretrained_path = None
        elif isinstance(config_or_path, (str, os.PathLike)):
            # Plain path string: load config from checkpoint directory, then load weights.
            cfg = XVLAConfig()
            pretrained_path = str(config_or_path)
        else:
            outer_config = config_or_path if isinstance(config_or_path, dict) else vars(config_or_path)
            config_block = outer_config["model"] if "model" in outer_config else outer_config
            config_block = config_block if isinstance(config_block, dict) else vars(config_block)

            cfg = XVLAConfig(**config_block)
            pretrained_path = outer_config["pretrained_path"] if "pretrained_path" in outer_config else None

        policy = cls(cfg)
        if processor_path:
            policy._processor_path = processor_path
        if pretrained_path:
            policy.load_pretrained(pretrained_path)
        return policy

    def load_pretrained(self, pretrained_path: str, strict: bool = False, device=None):
        """Load XVLA weights from a HuggingFace-style checkpoint directory.

        This policy wraps the core network as ``self.model`` (an ``XVLA``
        instance), so its ``state_dict`` keys are prefixed with ``model.``
        (i.e. ``model.vlm.*`` / ``model.transformer.*``). Some published X-VLA
        checkpoints (e.g. ``X-VLA-WidowX``) store keys without that wrapper
        prefix (``vlm.*`` / ``transformer.*``). We re-add the ``model.`` prefix
        when it is missing so the weights map onto this policy.
        """
        path = Path(pretrained_path)
        # Remember the checkpoint directory so lazily-built processors
        # (tokenizer / image_processor) can be loaded from it later.
        self._processor_path = str(path.parent if path.is_file() else path) \
            if self._processor_path is None else self._processor_path
        safetensors_file = path / "model.safetensors" if path.is_dir() else path
        load_kwargs = {"device": str(device)} if device is not None else {}
        state_dict = load_file(str(safetensors_file), **load_kwargs)

        # Align checkpoint keys with this policy's "model." wrapper prefix.
        own_keys = set(self.state_dict().keys())
        needs_prefix = any(
            (not k.startswith("model.")) and (f"model.{k}" in own_keys)
            for k in state_dict
        )
        if needs_prefix:
            state_dict = {
                (k if k.startswith("model.") else f"model.{k}"): v
                for k, v in state_dict.items()
            }

        # Florence2 shares one embedding table across `model.shared` and the
        # encoder's `embed_tokens` (see Florence2LanguageModel._tie_weights).
        # Published X-VLA checkpoints only store `...model.shared.weight` and
        # rely on weight tying to populate `...encoder.embed_tokens.weight`.
        # When the checkpoint config leaves `tie_word_embeddings` unset, that
        # tie is not re-applied at load time, so `encoder.embed_tokens.weight`
        # is reported as a missing key. Mirror the shared table into the
        # encoder embedding key so both are populated from the single source.
        shared_key = "model.vlm.language_model.model.shared.weight"
        encoder_emb_key = "model.vlm.language_model.model.encoder.embed_tokens.weight"
        if shared_key in state_dict and encoder_emb_key not in state_dict:
            state_dict[encoder_emb_key] = state_dict[shared_key]

        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        if missing:
            print(f"[xvla] load_pretrained missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[xvla] load_pretrained unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        return self
