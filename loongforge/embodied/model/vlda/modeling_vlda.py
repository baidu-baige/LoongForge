# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from VLDA and openpi; preserves the VLDA world/action training objectives.

"""VLDA two-stage world model and world-conditioned action policy."""

import copy
import hashlib
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from loongforge.embodied.model.pi05.modeling_pi05 import _get_gemma_config
from loongforge.embodied.model.registry import register_model

from . import vlda_preprocessing as _preprocessing
from .model_configuration_vlda import VLDAModelConfig
from .vjepa2_encoder import VJEPA2Encoder
from .vlda_gemma import PaliGemmaWithExpertModel


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class VLDAPytorch(nn.Module):
    """World and action experts with the original VLDA losses and gradient paths."""

    def __init__(self, config: VLDAModelConfig):
        super().__init__()
        self.config = config
        paligemma_config = _get_gemma_config("gemma_2b")
        action_expert_config = _get_gemma_config("gemma_300m")
        world_expert_config = _get_gemma_config("gemma_300m")
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            world_expert_config,
            use_adarms=[False, True, True],
            precision=config.dtype,
            attention_backend=config.attention_backend,
            compile_model=config.compile_model,
            compile_mode=config.compile_mode,
        )
        self.vjepa_encoder = VJEPA2Encoder(config.vjepa)
        self.teacher_vjepa_encoder = VJEPA2Encoder(config.vjepa)
        vjepa_dim = config.vjepa.embed_dim
        world_dim = world_expert_config.width
        self.delta_high_to_low = nn.Sequential(
            nn.Linear(vjepa_dim, 4 * config.delta_latent_dim),
            nn.SiLU(),
            nn.Linear(4 * config.delta_latent_dim, config.delta_latent_dim),
        )
        self.delta_low_to_high = nn.Sequential(
            nn.Linear(config.delta_latent_dim, 4 * config.delta_latent_dim),
            nn.SiLU(),
            nn.Linear(4 * config.delta_latent_dim, vjepa_dim),
        )
        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.world_in_proj = nn.Linear(config.delta_latent_dim, world_expert_config.width)
        self.world_out_proj = nn.Linear(world_expert_config.width, config.delta_latent_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.world_time_mlp_in = nn.Linear(world_dim, world_dim)
        self.world_time_mlp_out = nn.Linear(world_dim, world_dim)
        for p in self.teacher_vjepa_encoder.parameters():
            p.requires_grad = False
        if config.stage == 1:
            for p in self.paligemma_with_expert.paligemma.parameters():
                p.requires_grad = False
            for p in self.paligemma_with_expert.action_gemma_expert.parameters():
                p.requires_grad = False
            for p in self.action_in_proj.parameters():
                p.requires_grad = False
            for p in self.action_out_proj.parameters():
                p.requires_grad = False
            for p in self.time_mlp_in.parameters():
                p.requires_grad = False
            for p in self.time_mlp_out.parameters():
                p.requires_grad = False
            self.delta_to_action_noise = None
            self.bias_gamma = None
            self.delta_to_align = None
            self.action_to_align = None
            self.world_to_action_token = None
        else:
            self.delta_to_action_noise = nn.Sequential(
                nn.LayerNorm(config.delta_latent_dim),
                nn.Linear(config.delta_latent_dim, 4 * config.delta_latent_dim),
                nn.SiLU(),
                nn.Linear(4 * config.delta_latent_dim, config.action_horizon * config.max_action_dim),
                nn.Tanh(),
            )
            nn.init.zeros_(self.delta_to_action_noise[3].weight)
            nn.init.zeros_(self.delta_to_action_noise[3].bias)
            self.bias_gamma = nn.Parameter(torch.tensor(0.01))
            self.delta_to_align = nn.Sequential(
                nn.LayerNorm(config.delta_latent_dim), nn.Linear(config.delta_latent_dim, self.config.contrast_dim)
            )
            self.action_to_align = nn.Sequential(
                nn.LayerNorm(action_expert_config.width),
                nn.Linear(action_expert_config.width, self.config.contrast_dim),
            )
            self.world_to_action_token = nn.Sequential(
                nn.LayerNorm(config.delta_latent_dim), nn.Linear(config.delta_latent_dim, action_expert_config.width)
            )
        self.gradient_checkpointing_enabled = False
        torch.set_float32_matmul_precision("high")

    @torch.no_grad()
    def initialize_training_modules(self, seed: int):
        """Initialize VLDA-only modules on CPU before loading checkpoint weights."""
        modules = {
            name: getattr(self, name)
            for name in (
                "delta_high_to_low",
                "delta_low_to_high",
                "world_in_proj",
                "world_out_proj",
                "world_time_mlp_in",
                "world_time_mlp_out",
                "delta_to_action_noise",
                "delta_to_align",
                "action_to_align",
                "world_to_action_token",
            )
        }
        modules["world_gemma_expert"] = self.paligemma_with_expert.world_gemma_expert
        with torch.random.fork_rng(devices=[]):
            for name, module in modules.items():
                if module is None:
                    continue
                module_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
                torch.random.default_generator.manual_seed(module_seed)
                for _, child in sorted(module.named_modules()):
                    if isinstance(child, nn.Linear):
                        if name == "world_gemma_expert":
                            # Match openpi's FP32 normal initialization before BF16 conversion.
                            weight = torch.empty(child.weight.shape, dtype=torch.float32, device="cpu")
                            nn.init.normal_(weight, std=module.config.initializer_range)
                            child.weight.copy_(weight)
                            if child.bias is not None:
                                child.bias.zero_()
                        else:
                            child.reset_parameters()
                    elif isinstance(child, nn.LayerNorm):
                        child.reset_parameters()
        if self.delta_to_action_noise is not None:
            nn.init.zeros_(self.delta_to_action_noise[3].weight)
            nn.init.zeros_(self.delta_to_action_noise[3].bias)
            self.bias_gamma.fill_(0.01)

    def drop_training_only_modules(self):
        self.vjepa_encoder = None
        self.teacher_vjepa_encoder = None
        self.delta_high_to_low = None
        self.delta_low_to_high = None
        self.delta_to_align = None
        self.action_to_align = None
        return self

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.action_gemma_expert.model.gradient_checkpointing = True
        self.paligemma_with_expert.world_gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for VLDA model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.action_gemma_expert.model.gradient_checkpointing = False
        self.paligemma_with_expert.world_gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for VLDA model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    @torch.no_grad()
    def update_teacher_ema(self, m: float):
        """Update teacher V-JEPA2 encoder via EMA of student weights.

        Uses ``torch._foreach_mul_`` / ``torch._foreach_add_`` (same as
        original V-JEPA2) for efficient in-place update:

            teacher = m * teacher + (1 - m) * student
        """
        params_k = []
        params_q = []
        for param_q, param_k in zip(self.vjepa_encoder.parameters(), self.teacher_vjepa_encoder.parameters()):
            params_q.append(param_q)
            params_k.append(param_k)
        torch._foreach_mul_(params_k, m)
        torch._foreach_add_(params_k, params_q, alpha=1 - m)

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        (observation, vjepa_images, vjepa_future_images) = _preprocessing.preprocess_observation(
            observation,
            train=train,
            image_resolution_vjepa=(self.config.vjepa.img_size, self.config.vjepa.img_size),
        )
        if vjepa_images is not None and vjepa_future_images is not None:
            return (
                list(observation.images.values()),
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.state,
                list(vjepa_images.values()),
                list(vjepa_future_images.values()),
            )
        else:
            return (
                list(observation.images.values()),
                list(observation.image_masks.values()),
                observation.tokenized_prompt,
                observation.tokenized_prompt_mask,
                observation.state,
                None,
                None,
            )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def sample_time_world(self, bsize, device):
        time_beta = sample_beta(1.0, 1.5, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def flow_matching(self, x, noise=None, time=None, noise_bias=None):
        if noise is None:
            noise = self.sample_noise(x.shape, x.device)
        if time is None:
            time = self.sample_time(x.shape[0], x.device)
        if noise_bias is not None:
            noise = noise + noise_bias * self.bias_gamma
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * x
        u_t = noise - x
        return (x_t, u_t, time)

    def jit(self, x, noise=None, time=None):
        if noise is None:
            noise = self.sample_noise(x.shape, x.device)
        if time is None:
            time = self.sample_time_world(x.shape[0], x.device)
        time_expanded = time[:, None, None]
        z = time_expanded * x + (1 - time_expanded) * noise
        v = (x - z) / (1 - time_expanded).clamp_min(0.05)
        return (z, v, time)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, vjepa_images=None, vjepa_future_images=None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Embed images with SigLIP + V-JEPA2 and language tokens.

        SigLIP tokens go into the LLM backbone (prefix). V-JEPA2-AC tokens
        are returned separately for auxiliary use (e.g., world model).

        images: [(B C H W), (B C H W), ...]
        vjepa_images: [(B, C, T, H, W)]
        vjepa_future_images: [(B, C, T, H, W)]

        Returns:
            embs: Concatenated SigLIP image + language embeddings. (B, prefix_tokens, hidden_size)
            pad_masks: Padding masks for the prefix sequence.
            att_masks: Attention masks for the prefix sequence.
            vjepa_embs: V-JEPA2 patch tokens. (B P D)
            vjepa_future_embs: V-JEPA2 future patch tokens. (B P D)
        """
        assert (
            vjepa_images is None
            and vjepa_future_images is None
            or (vjepa_images is not None and vjepa_future_images is not None)
        ), "vjepa_images and vjepa_future_images should be None or not None at the same time"
        embs = []
        pad_masks = []
        att_masks = []
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            (bsize, num_img_embs) = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        if vjepa_images is not None:
            vjepa_embs = []
            vjepa_future_embs = []
            for vjepa_img, vjepa_future_img in zip(vjepa_images, vjepa_future_images, strict=True):

                def vjepa_embed_func(img):
                    return self.vjepa_encoder(img)

                def vjepa_teacher_embed_func(img):
                    return self.teacher_vjepa_encoder(img)

                vjepa_emb = self._apply_checkpoint(vjepa_embed_func, vjepa_img)
                vjepa_future_emb = self._apply_checkpoint(vjepa_teacher_embed_func, vjepa_future_img)
                vjepa_embs.append(vjepa_emb)
                vjepa_future_embs.append(vjepa_future_emb)
            vjepa_embs = torch.cat(vjepa_embs)
            vjepa_future_embs = torch.cat(vjepa_future_embs)
        else:
            vjepa_embs = None
            vjepa_future_embs = None
        return (embs, pad_masks, att_masks, vjepa_embs, vjepa_future_embs)

    def embed_suffix(
        self, noisy_actions=None, timestep=None, noisy_world=None, world_timestep=None, delta_to_aciton=None
    ):
        """
        Embed state, noisy_actions, timestep to prepare for Expert Gemma processing.
        Also Preprocess the noisy_world and world_timestep for World Expert.
        """
        assert (
            noisy_world is None and world_timestep is None or (noisy_world is not None and world_timestep is not None)
        ), "nosiy_world and world_timestep should be None or not None at the same time"
        assert noisy_actions is None and timestep is None or (noisy_actions is not None and timestep is not None), (
            "nosiy_world and world_timestep should be None or not None at the same time"
        )
        if noisy_actions is not None:
            embs = []
            pad_masks = []
            att_masks = []
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.action_in_proj.out_features, min_period=0.004, max_period=4.0, device=timestep.device
            )
            time_emb = time_emb.type(dtype=timestep.dtype)

            def action_proj_func(noisy_actions):
                return self.action_in_proj(noisy_actions)

            action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)
            if delta_to_aciton is not None:
                action_emb = torch.cat([self.world_to_action_token(delta_to_aciton), action_emb], dim=1)

            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb
            embs.append(action_time_emb)
            (bsize, action_time_dim) = action_time_emb.shape[:2]
            action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
            pad_masks.append(action_time_mask)
            att_masks += [1] + [0] * (self.config.action_horizon - 1 + int(delta_to_aciton is not None))
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
            att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        else:
            (embs, pad_masks, att_masks, adarms_cond) = (None, None, None, None)
        if world_timestep is not None:

            def world_proj_func(noisy_deltas):
                return self.world_in_proj(noisy_deltas)

            world_emb = self._apply_checkpoint(world_proj_func, noisy_world)
            world_time_emb = create_sinusoidal_pos_embedding(
                world_timestep, world_emb.shape[-1], min_period=0.004, max_period=4.0, device=world_timestep.device
            ).to(dtype=world_timestep.dtype)

            def world_time_mlp_func(x):
                x = self.world_time_mlp_in(x)
                x = F.silu(x)
                x = self.world_time_mlp_out(x)
                return F.silu(x)

            world_adarms_cond = self._apply_checkpoint(world_time_mlp_func, world_time_emb)
            (bsize, world_token_num) = world_emb.shape[:2]
            world_pad_mask = torch.ones(bsize, world_token_num, dtype=torch.bool, device=world_timestep.device)
            world_attn_mask = [1] + [0] * (world_token_num - 1)
            world_attn_mask = torch.tensor(world_attn_mask, dtype=world_emb.dtype, device=world_emb.device)
            world_attn_mask = world_attn_mask[None, :].expand(bsize, world_token_num)
        else:
            (world_emb, world_pad_mask, world_attn_mask, world_adarms_cond) = (None, None, None, None)
        return (embs, pad_masks, att_masks, adarms_cond, world_emb, world_pad_mask, world_attn_mask, world_adarms_cond)

    def compute_token_set_mse_loss(
        self,
        pred_tokens,
        target_tokens,
        *,
        epsilon=0.05,
        sinkhorn_iters=50,
        normalize_cost_for_plan=True,
        detach_plan=True,
    ):
        """
        - epsilon: Entropy regularization coefficient controlling transport plan sharpness.
            Smaller: Hard one-to-one matching, higher alignment accuracy but gradient oscillation.
            Larger: Smooth soft matching, stable gradients with weaker regularization.
        - sinkhorn_iters: Sinkhorn iteration count.
            More tokens or smaller epsilon require more iterations to converge.
            50 iterations suffice for convergence with hundreds of tokens or fewer.
        - normalize_cost_for_plan: Whether to normalize cost matrix before computing transport plan.
            Decouples epsilon from the feature dimension and cost scale.
        Notes:
            - This replaces Chamfer with a soft one-to-one transport plan.
            - `detach_plan=True` saves memory and gives stable gradients through the cost;
            set False if you want gradients through the Sinkhorn iterations too.
        """
        if pred_tokens.shape[0] != target_tokens.shape[0]:
            raise ValueError(f"Batch mismatch: {pred_tokens.shape} vs {target_tokens.shape}")
        if pred_tokens.shape[-1] != target_tokens.shape[-1]:
            raise ValueError(f"Dim mismatch: {pred_tokens.shape} vs {target_tokens.shape}")
        pred_tokens = pred_tokens.to(dtype=torch.float32)
        target_tokens = target_tokens.to(dtype=torch.float32)
        (B, N, D) = pred_tokens.shape
        M = target_tokens.shape[1]
        cost = torch.cdist(pred_tokens, target_tokens, p=2).pow(2) / D
        if normalize_cost_for_plan:
            scale = cost.detach().mean(dim=(1, 2), keepdim=True).clamp_min(1e-06)
            plan_cost = cost / scale
        else:
            plan_cost = cost

        def _sinkhorn_plan(c):
            log_mu = c.new_full((B, N), -math.log(N))
            log_nu = c.new_full((B, M), -math.log(M))
            log_K = -c / epsilon
            u = torch.zeros_like(log_mu)
            v = torch.zeros_like(log_nu)
            for _ in range(sinkhorn_iters):
                u = log_mu - torch.logsumexp(log_K + v[:, None, :], dim=2)
                v = log_nu - torch.logsumexp(log_K + u[:, :, None], dim=1)
            log_P = log_K + u[:, :, None] + v[:, None, :]
            return torch.exp(log_P)

        if detach_plan:
            with torch.no_grad():
                transport_plan = _sinkhorn_plan(plan_cost)
        else:
            transport_plan = _sinkhorn_plan(plan_cost)
        loss = (transport_plan * cost).sum(dim=(1, 2)).mean()
        return loss

    def compute_pooled_future_loss(self, vjepa_embs, delta_high_pred, vjepa_future_embs):
        """
        vjepa_embs: [B, P, D_vjepa]
        delta_high_pred: [B, K, D_vjepa]
        """
        (B, P, D) = vjepa_embs.shape
        K = delta_high_pred.shape[1]
        grid_size = int(P**0.5)
        pooled_grid_size = int(K**0.5)
        if grid_size * grid_size != P:
            raise ValueError(f"V-JEPA token count must form a square grid, got P={P}")
        if pooled_grid_size * pooled_grid_size != K:
            raise ValueError(f"delta_token_num must form a square grid for pooled future loss, got K={K}")
        curr_2d = vjepa_embs.transpose(1, 2).view(B, D, grid_size, grid_size)
        fut_2d = vjepa_future_embs.transpose(1, 2).view(B, D, grid_size, grid_size)
        curr_pooled = F.adaptive_avg_pool2d(curr_2d, (pooled_grid_size, pooled_grid_size)).flatten(2).transpose(1, 2)
        fut_pooled = F.adaptive_avg_pool2d(fut_2d, (pooled_grid_size, pooled_grid_size)).flatten(2).transpose(1, 2)
        target_delta_high = fut_pooled - curr_pooled
        return self.compute_token_set_mse_loss(delta_high_pred, target_delta_high)

    def compute_delta_action_align_loss(
        self, world_output, pos_actions, neg_actions, tau=0.07, delta_token_tau=0.1, detach_action_in_proj=True
    ):
        """
        world_output: [B, K, D_latent]
        pos_actions: [B, action_horizon, action_dim]
        neg_actions: [B, N, action_dim]
        """
        if isinstance(neg_actions, np.ndarray):
            neg_actions = torch.from_numpy(neg_actions).to(dtype=pos_actions.dtype, device=pos_actions.device)
        elif not isinstance(neg_actions, torch.Tensor):
            raise ValueError(f"Unknown neg_actions type f{type(neg_actions)}")
        (B, H, action_dim) = pos_actions.shape
        crop_action_sequences = self.config.crop_action_sequences
        if crop_action_sequences is not None:
            if crop_action_sequences <= 0:
                raise ValueError(f"crop_action_sequences must be positive, got {crop_action_sequences}")
            if H > crop_action_sequences:
                pos_actions = pos_actions[:, :crop_action_sequences, :]
                H = crop_action_sequences
        if neg_actions.dim() == 4:
            neg_actions = neg_actions.flatten(1, 2)
        neg_actions = neg_actions.to(dtype=pos_actions.dtype)
        N = neg_actions.shape[1]
        if detach_action_in_proj:
            with torch.no_grad():
                pos_action_hidden = self.action_in_proj(pos_actions)
                neg_action_hidden = self.action_in_proj(neg_actions)
        else:
            pos_action_hidden = self.action_in_proj(pos_actions)
            neg_action_hidden = self.action_in_proj(neg_actions)
        z_pos_action = self.action_to_align(pos_action_hidden)
        z_neg_action = self.action_to_align(neg_action_hidden)
        z_pos_action = F.normalize(z_pos_action, dim=-1)
        z_neg_action = F.normalize(z_neg_action, dim=-1)
        z_delta = self.delta_to_align(world_output)
        z_delta = F.normalize(z_delta, dim=-1)
        pos_sim_token = torch.einsum("bkc,bhc->bhk", z_delta, z_pos_action)
        neg_sim_token = torch.einsum("bkc,bnc->bnk", z_delta, z_neg_action)
        pos_sim = torch.logsumexp(pos_sim_token / delta_token_tau, dim=-1) * delta_token_tau
        neg_sim = torch.logsumexp(neg_sim_token / delta_token_tau, dim=-1) * delta_token_tau
        neg_sim_expand = neg_sim[:, None, :].expand(B, H, N)
        logits = torch.cat([pos_sim[:, :, None], neg_sim_expand], dim=-1)
        logits = logits / tau
        labels = torch.zeros(B * H, dtype=torch.long, device=logits.device)
        loss_align = F.cross_entropy(logits.reshape(B * H, 1 + N), labels)
        return loss_align

    def forward_world_only(self, prefix_embs, prefix_pad_masks, prefix_att_masks, vjepa_embs, vjepa_future_embs):
        delta_high = vjepa_future_embs - vjepa_embs
        scores = delta_high.norm(p=2, dim=-1)
        (_, topk_indices) = torch.topk(scores, k=self.config.delta_token_num, dim=1)
        batch_idx = torch.arange(delta_high.shape[0], device=delta_high.device)[:, None].expand(
            -1, self.config.delta_token_num
        )
        delta_high_topk = delta_high[batch_idx, topk_indices]
        delta_low = self.delta_high_to_low(delta_high_topk)
        (world_z_t, world_v_t, world_time) = self.jit(delta_low)
        (_, _, _, _, suffix_world_embs, suffix_world_pad_mask, suffix_world_attn_mask, world_adarms_cond) = (
            self.embed_suffix(None, None, world_z_t, world_time)
        )
        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            suffix_world_embs = suffix_world_embs.to(dtype=torch.bfloat16)
        pad_masks = torch.cat([prefix_pad_masks, suffix_world_pad_mask], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_world_attn_mask], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Gemma already checkpoints per layer; do not checkpoint the whole pass again.
        ((_, suffix_out), _) = self.paligemma_with_expert.forward_stage1(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            inputs_embeds=[prefix_embs, suffix_world_embs],
            adarms_cond=[None, world_adarms_cond],
        )
        suffix_out = suffix_out[:, -self.config.delta_token_num :].to(dtype=torch.float32)

        def world_out_proj_func(suffix_out):
            return self.world_out_proj(suffix_out)

        world_x_t = self._apply_checkpoint(world_out_proj_func, suffix_out)
        world_v_t_pred = (world_x_t - world_z_t) / (1 - world_time[:, None, None]).clamp_min(0.05)
        delta_high_pred = self.delta_low_to_high(world_x_t)
        delta_loss = self.compute_token_set_mse_loss(world_v_t_pred, world_v_t)
        future_loss = self.compute_pooled_future_loss(vjepa_embs, delta_high_pred, vjepa_future_embs)
        return (delta_loss, future_loss)

    def forward(self, observation, actions, noise=None, time=None) -> tuple[Tensor, ...]:
        """Return the scalar world losses and, in stage 2, alignment/action losses."""
        (images, img_masks, lang_tokens, lang_masks, state, vjepa_images, vjepa_future_images) = (
            self._preprocess_observation(observation, train=True)
        )
        (prefix_embs, prefix_pad_masks, prefix_att_masks, vjepa_embs, vjepa_future_embs) = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, vjepa_images, vjepa_future_images
        )
        if self.config.stage == 1:
            return self.forward_world_only(
                prefix_embs, prefix_pad_masks, prefix_att_masks, vjepa_embs, vjepa_future_embs
            )
        else:
            delta_high = vjepa_future_embs - vjepa_embs
            scores = delta_high.norm(p=2, dim=-1)
            (_, topk_indices) = torch.topk(scores, k=self.config.delta_token_num, dim=1)
            batch_idx = torch.arange(delta_high.shape[0], device=delta_high.device)[:, None].expand(
                -1, self.config.delta_token_num
            )
            delta_high_topk = delta_high[batch_idx, topk_indices]
            delta_low = self.delta_high_to_low(delta_high_topk)
            (world_z_t, world_v_t, world_time) = self.jit(delta_low)
            del delta_high, scores, topk_indices, batch_idx, delta_high_topk, delta_low
            (_, _, _, _, suffix_world_embs, suffix_world_pad_mask, suffix_world_attn_mask, world_adarms_cond) = (
                self.embed_suffix(None, None, world_z_t, world_time)
            )
            if (
                self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
                suffix_world_embs = suffix_world_embs.to(dtype=torch.bfloat16)
            pad_masks = torch.cat([prefix_pad_masks, suffix_world_pad_mask], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_world_attn_mask], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            (world_output, llm_states) = self.paligemma_with_expert.forward_stage2_pass1(
                attention_mask_world=att_2d_masks_4d,
                position_ids_world=position_ids,
                llm_embeds=prefix_embs,
                world_embeds=suffix_world_embs,
                world_adarms_cond=world_adarms_cond,
            )
            world_output = world_output[:, -self.config.delta_token_num :].to(dtype=torch.float32)

            def world_out_proj_func(world_output):
                return self.world_out_proj(world_output)

            world_x_t = self._apply_checkpoint(world_out_proj_func, world_output)
            world_v_t_pred = (world_x_t - world_z_t) / (1 - world_time[:, None, None]).clamp_min(0.05)
            del world_output
            delta_high_pred = self.delta_low_to_high(world_x_t)
            delta_loss = self.compute_token_set_mse_loss(world_v_t_pred, world_v_t)
            future_loss = self.compute_pooled_future_loss(vjepa_embs, delta_high_pred, vjepa_future_embs)
            align_loss = self.compute_delta_action_align_loss(world_x_t, actions, observation.neg_actions)
            del world_v_t_pred, delta_high_pred, prefix_embs, vjepa_embs, vjepa_future_embs
            delta_mean = world_x_t.mean(dim=1)
            delta_bias = self.delta_to_action_noise(delta_mean)
            delta_bias = delta_bias.view(delta_mean.shape[0], self.config.action_horizon, self.config.max_action_dim)
            (action_x_t, action_u_t, action_time) = self.flow_matching(actions, noise, time, delta_bias)
            delta_mean = delta_mean.unsqueeze(1)
            del world_x_t
            (suffix_action_embs, suffix_action_pad_mask, suffix_action_attn_mask, action_adarms_cond, _, _, _, _) = (
                self.embed_suffix(action_x_t, action_time, None, None, delta_mean)
            )
            if (
                self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                suffix_action_embs = suffix_action_embs.to(dtype=torch.bfloat16)
            pad_masks = torch.cat([prefix_pad_masks, suffix_action_pad_mask], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_action_attn_mask], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            action_output = self.paligemma_with_expert.forward_stage2_pass2(
                llm_states=llm_states,
                action_embeds=suffix_action_embs,
                attention_mask_action=att_2d_masks_4d,
                position_ids_action=position_ids,
                action_adarms_cond=action_adarms_cond,
            )
            action_output = action_output[:, -self.config.action_horizon :].to(dtype=torch.float32)

            def action_out_proj_func(suffix_out):
                return self.action_out_proj(suffix_out)

            action_v_t = self._apply_checkpoint(action_out_proj_func, action_output)
            action_loss = F.mse_loss(action_u_t, action_v_t, reduction="none").mean()
            return (delta_loss, future_loss, align_loss, action_loss)

    @torch.no_grad()
    def world_expert_infer(
        self, device, bsize, prefix_pad_masks, past_key_values, num_steps=None, noise=None
    ) -> Tensor:
        """Generate delta tokens"""
        num_steps = self.config.world_inference_steps if num_steps is None else num_steps
        if noise is None:
            noise = self.sample_noise([bsize, self.config.delta_token_num, self.config.delta_latent_dim], device)
        z_t = noise
        dt = torch.tensor(1.0 / num_steps, dtype=torch.float32, device=device)
        for step in range(num_steps):
            expanded_time = torch.full((bsize,), step / num_steps, dtype=torch.float32, device=device)
            x_pred = self.denoise_step(prefix_pad_masks, past_key_values, z_t, expanded_time, True)
            v_t = (x_pred - z_t) / (1 - expanded_time[:, None, None]).clamp_min(0.05)
            z_t = z_t + dt * v_t
        return z_t

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=None, world_noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        num_steps = self.config.action_inference_steps if num_steps is None else num_steps
        bsize = observation.state.shape[0]
        (images, img_masks, lang_tokens, lang_masks, _, _, _) = self._preprocess_observation(observation, train=False)
        (prefix_embs, prefix_pad_masks, prefix_att_masks, _, _) = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        (_, past_key_values) = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )
        delta_token = self.world_expert_infer(device, bsize, prefix_pad_masks, past_key_values, noise=world_noise)
        delta_mean = delta_token.mean(dim=1)
        del delta_token
        noise_bias = self.delta_to_action_noise(delta_mean)
        noise_bias = noise_bias.view(delta_mean.shape[0], self.config.action_horizon, self.config.max_action_dim)
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        noise = noise + noise_bias * self.bias_gamma
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        x_t = noise
        delta_mean = delta_mean.unsqueeze(1)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(prefix_pad_masks, past_key_values, x_t, expanded_time, False, delta_mean)
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep, is_world=False, delta_to_action=None):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        if is_world:
            (_, _, _, _, suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond) = self.embed_suffix(
                None, None, x_t, timestep
            )
        else:
            (suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond, _, _, _, _) = self.embed_suffix(
                x_t, timestep, None, None, delta_to_action
            )
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        if is_world:
            self.paligemma_with_expert.world_gemma_expert.model.config._attn_implementation = "eager"
            inputs_embeds = [None, None, suffix_embs]
            adarms_conds = [None, None, adarms_cond]
        else:
            self.paligemma_with_expert.action_gemma_expert.model.config._attn_implementation = "eager"
            inputs_embeds = [None, suffix_embs, None]
            adarms_conds = [None, adarms_cond, None]
        (outputs_embeds, _) = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=copy.deepcopy(past_key_values),
            inputs_embeds=inputs_embeds,
            use_cache=False,
            adarms_cond=adarms_conds,
        )
        if is_world:
            suffix_out = outputs_embeds[2].to(dtype=torch.float32)
            suffix_out = suffix_out[:, -self.config.delta_token_num :]
            return self.world_out_proj(suffix_out)
        else:
            suffix_out = outputs_embeds[1].to(dtype=torch.float32)
            suffix_out = suffix_out[:, -self.config.action_horizon :]
            return self.action_out_proj(suffix_out)


_STAGE2_PREFIXES = (
    "delta_to_action_noise.",
    "bias_gamma",
    "delta_to_align.",
    "action_to_align.",
    "world_to_action_token.",
)
_WORLD_PREFIXES = (
    "paligemma_with_expert.world_gemma_expert.",
    "vjepa_encoder.",
    "teacher_vjepa_encoder.",
    "delta_high_to_low.",
    "delta_low_to_high.",
    "world_in_proj.",
    "world_out_proj.",
    "world_time_mlp_in.",
    "world_time_mlp_out.",
)


def remap_vlda_state_dict(state_dict):
    """Translate original openpi/VLDA and LoongForge wrapper parameter names."""
    mapped = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model.architecture.pi05_model.", "architecture.pi05_model.", "model.", "pi05."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        key = key.replace(".gemma_expert.", ".action_gemma_expert.")
        key = key.replace("action_time_mlp_in.", "time_mlp_in.").replace("action_time_mlp_out.", "time_mlp_out.")
        for module in ("language_model", "vision_tower", "multi_modal_projector"):
            key = key.replace(f".paligemma.{module}.", f".paligemma.model.{module}.")
        mapped[key] = value
    return mapped


@register_model("vlda")
class VLDAPolicy(nn.Module):
    """LoongForge training/evaluation contract around the VLDA core model."""

    def __init__(self, config: VLDAModelConfig):
        super().__init__()
        self.config = config
        self.model = VLDAPytorch(config)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self._tokenizer_path = ""
        self._inference_transform = None
        self._inference_data_cfg = None
        self._dataset_stats = None

    @classmethod
    def from_pretrained(cls, model_cfg):
        """Construct structure; the trainer loads initialization/resume weights."""
        return cls(model_cfg)

    def forward(self, batch):
        losses = self.model(batch, batch.actions)
        if self.config.stage == 1:
            delta, future = losses
            total = delta + future
            metrics = {"delta_loss": delta, "future_loss": future}
        else:
            delta, future, align, action = losses
            total = delta + future + action + self.config.align_loss_weight * align
            metrics = {"delta_loss": delta, "future_loss": future, "align_loss": align, "action_loss": action}
        metrics["loss"] = total
        return total, {key: value.detach() for key, value in metrics.items()}

    @torch.no_grad()
    def update_teacher_ema(self, momentum):
        self.model.update_teacher_ema(momentum)

    def load_pretrained(self, path, device=None, *, resume=False):
        """Load base pi05, stage-one VLDA, or a complete VLDA checkpoint.

        Only expected new modules may be missing during initialization. Resume
        requires every parameter, including teacher and stage-two heads.
        """
        from pathlib import Path

        from safetensors.torch import load_file

        path = Path(path)
        if path.is_dir():
            path = path / ("model.safetensors" if (path / "model.safetensors").exists() else "pytorch_model.pt")
        raw_state = (
            load_file(str(path), device="cpu")
            if path.suffix == ".safetensors"
            else torch.load(path, map_location="cpu", weights_only=True)
        )
        state = remap_vlda_state_dict(raw_state)
        has_world = any(key.startswith("world_in_proj.") for key in state)
        has_stage2 = "bias_gamma" in state
        if self.config.stage == 2 and not has_world:
            raise ValueError("VLDA stage 2 must initialize from a stage 1/2 VLDA checkpoint")
        expected = self.model.state_dict()
        # save_model may omit either name of the tied PaliGemma embedding/head.
        embedding = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        head = "paligemma_with_expert.paligemma.lm_head.weight"
        for destination, source in ((embedding, head), (head, embedding)):
            if destination in expected and destination not in state and source in state:
                state[destination] = state[source]
        allowed_missing = ()
        if not resume:
            if not has_world:
                allowed_missing += _WORLD_PREFIXES
            if self.config.stage == 2 and not has_stage2:
                allowed_missing += _STAGE2_PREFIXES
        missing = sorted(key for key in expected if key not in state and not key.startswith(allowed_missing))
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected:
            raise ValueError(f"VLDA checkpoint mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}")
        self.model.load_state_dict(state, strict=resume)
        if not has_world:
            if not self.config.vjepa_checkpoint:
                raise ValueError("Initializing VLDA from pi05 requires model.vjepa_checkpoint")
            checkpoint = torch.load(self.config.vjepa_checkpoint, map_location="cpu", weights_only=False)
            self.model.vjepa_encoder.load_encoder_state(checkpoint[self.config.vjepa_online_key])
            self.model.teacher_vjepa_encoder.load_encoder_state(checkpoint[self.config.vjepa_teacher_key])
        logging.getLogger(__name__).info(
            "Loaded VLDA weights from %s (world=%s, stage2=%s)", path, has_world, has_stage2
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        if self.config.stage != 2:
            raise ValueError("Action inference requires a stage 2 VLDA policy")
        self.eval()
        return self.model.sample_actions(batch.state.device, batch, **kwargs)[..., : self.config.action_dim]

    @torch.no_grad()
    def predict_action(self, images, instructions, state=None, dataset_stats=None):
        """Return unnormalized environment actions with shape [B,H,action_dim]."""
        from loongforge.embodied.data.datasets.vlda.transforms.data_configuration_vlda import VLDADataConfig
        from loongforge.embodied.data.datasets.vlda.transforms.vlda_collator import VLDAPreprocessor
        from loongforge.embodied.data.datasets.vlda.transforms.vlda_transform import (
            VLDATransform,
            decode_actions,
            encode_state,
        )

        if self._inference_transform is None:
            stats = dataset_stats if dataset_stats is not None else self._dataset_stats
            if stats is None:
                raise ValueError("VLDA inference requires the training norm_stats.json")
            stats = stats.get("norm_stats", stats)
            stats = {key: {name: np.asarray(value) for name, value in values.items()} for key, values in stats.items()}
            data_cfg = self._inference_data_cfg or VLDADataConfig()
            self._inference_transform = VLDATransform(self.config, data_cfg, stats, self._tokenizer_path)
        transform = self._inference_transform
        batch_size = len(instructions)
        if state is None:
            if self.config.discrete_state_input or transform.data_cfg.delta_joint_actions:
                raise ValueError("This VLDA configuration requires the current robot state")
            state = np.zeros((batch_size, self.config.state_dim), dtype=np.float32)
        state = np.asarray(state).reshape(batch_size, self.config.state_dim)
        examples = []
        for views, instruction, proprio in zip(images, instructions, state, strict=True):
            if not 1 <= len(views) <= 3:
                raise ValueError("VLDA expects one to three views in base/left/right order")
            examples.append(
                transform(
                    {
                        "images": list(views) + [None] * (3 - len(views)),
                        "prompt": instruction,
                        "state": proprio,
                    }
                )
            )
        batch = VLDAPreprocessor()(examples).to(next(self.parameters()).device)
        actions = transform.action_normalizer.inverse(self.predict_action_chunk(batch).cpu().float()).numpy()
        proprio = encode_state(state, transform.data_cfg.adapt_to_pi)
        return decode_actions(actions, proprio[:, None], transform.data_cfg).astype(np.float32)
