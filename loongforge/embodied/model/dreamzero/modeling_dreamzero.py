# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DreamZero policy module.

The policy wraps ``WANPolicyHead`` with the shared training interface. The
forward path returns ``(loss, log_loss_dict)`` following the standard embodied
trainer contract, and model-owned hooks provide LoRA target defaults and
frozen-module mode management.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

try:
    from lerobot.policies.pretrained import PreTrainedPolicy
except Exception:  # pragma: no cover - lerobot may be optional in some envs
    PreTrainedPolicy = torch.nn.Module  # type: ignore[misc, assignment]

from transformers.feature_extraction_utils import BatchFeature

from loongforge.embodied.model.registry import register_model

from .model_configuration_dreamzero import DreamZeroConfig
from .modules.action_head_tf import WANPolicyHead, WANPolicyHeadConfig
from .precomputed_cache import build_precomputed_cache_config

if TYPE_CHECKING:
    from loongforge.embodied.distributed.context import DistributedContext

__all__ = ["DreamZeroPolicy", "DreamZeroConfig", "build_action_head_config"]

logger = logging.getLogger(__name__)


def build_action_head_config(model_config: DreamZeroConfig) -> WANPolicyHeadConfig:
    """Materialise WANPolicyHeadConfig from DreamZeroConfig.

    Field mapping is intentional and explicit — DreamZeroConfig is the
    trainer-facing config, while WANPolicyHeadConfig is the action-head
    internal config.
    """
    precomputed_cache = build_precomputed_cache_config(model_config)
    return WANPolicyHeadConfig(
        # Tiling / VAE
        model_dtype=model_config.model_dtype,
        tiled=model_config.vae_tiled,
        tile_size_height=model_config.vae_tile_size_height,
        tile_size_width=model_config.vae_tile_size_width,
        tile_stride_height=model_config.vae_tile_stride_height,
        tile_stride_width=model_config.vae_tile_stride_width,
        num_frame_per_block=model_config.num_frame_per_block,
        target_video_height=model_config.target_video_height,
        target_video_width=model_config.target_video_width,
        # Geometry
        hidden_size=model_config.action_head_hidden_size,
        max_seq_len=model_config.max_chunk_size,
        action_dim=model_config.max_action_dim,
        action_horizon=model_config.action_horizon,
        num_frames=model_config.num_frames,
        input_embedding_dim=model_config.input_embedding_dim,
        # Flow-matching noise
        num_dit_steps=model_config.num_dit_steps,
        noise_beta_alpha=model_config.noise_beta_alpha,
        noise_beta_beta=model_config.noise_beta_beta,
        noise_s=model_config.noise_s,
        decouple_video_action_noise=model_config.decouple_video_action_noise,
        video_noise_beta_alpha=model_config.video_noise_beta_alpha,
        video_noise_beta_beta=model_config.video_noise_beta_beta,
        num_timestep_buckets=model_config.num_timestep_buckets,
        # Default-off performance controls.
        batch_vae_encode=model_config.batch_vae_encode,
        prompt_emb_cache=model_config.prompt_emb_cache,
        prompt_emb_cache_max_entries=model_config.prompt_emb_cache_max_entries,
        precomputed_video_latents=precomputed_cache.video_latents.enabled,
        precomputed_video_latents_key=precomputed_cache.video_latents.batch_key,
        precomputed_video_latents_layout=(
            precomputed_cache.video_latents.layout or "bcthw"
        ),
        precomputed_video_latents_strict=(
            precomputed_cache.video_latents.required
        ),
        precomputed_first_frame_latents=(
            precomputed_cache.first_frame_latents.enabled
        ),
        precomputed_first_frame_latents_key=(
            precomputed_cache.first_frame_latents.batch_key
        ),
        precomputed_first_frame_latents_strict=(
            precomputed_cache.first_frame_latents.required
        ),
        precomputed_prompt_embs=precomputed_cache.prompt_embs.enabled,
        precomputed_prompt_embs_key=precomputed_cache.prompt_embs.batch_key,
        precomputed_prompt_embs_strict=precomputed_cache.prompt_embs.required,
        skip_precomputed_pixel_preprocess=(
            precomputed_cache.skip_pixel_preprocess
        ),
        precomputed_first_frame_only=precomputed_cache.first_frame_only,
        # Trainability
        tune_projector=model_config.tune_projector,
        tune_diffusion_model=model_config.tune_diffusion_model,
        # Embodiments
        max_num_embodiments=model_config.max_num_embodiments,
    )


def _config_to_dreamzero(cfg: Any) -> DreamZeroConfig:
    """Convert OmegaConf/dict/object config to DreamZeroConfig."""
    if isinstance(cfg, DreamZeroConfig):
        return cfg
    valid = {f.name for f in fields(DreamZeroConfig)}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            try:
                cfg = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                cfg = OmegaConf.to_container(cfg, resolve=False)
    except Exception:
        pass
    if isinstance(cfg, dict):
        raw = cfg
    else:
        values = vars(cfg)
        raw = {
            name: values[name]
            for name in valid
            if name in values
        }
    return DreamZeroConfig(**{k: v for k, v in raw.items() if k in valid})


@register_model("dreamzero")
class DreamZeroPolicy(PreTrainedPolicy):
    """DreamZero policy = WANPolicyHead (action head) + frozen encoders + CausalWanModel.

    The four submodules (text_encoder, image_encoder, vae, model) are passed
    in by the provider so the unit test / ckpt-conversion paths can swap in
    minimal stubs without changing the policy class.
    """

    config_class = DreamZeroConfig
    name = "dreamzero"

    def __init__(
        self,
        config: DreamZeroConfig,
        text_encoder: torch.nn.Module,
        image_encoder: torch.nn.Module,
        vae: torch.nn.Module,
        model: torch.nn.Module,
    ):
        """Assemble the DreamZero policy from its action-head components."""
        super().__init__(config) if PreTrainedPolicy is not torch.nn.Module else super().__init__()
        self.config = config
        head_cfg = build_action_head_config(config)
        self.action_head = WANPolicyHead(
            config=head_cfg,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            vae=vae,
            model=model,
        )
        # Closed-loop inference: per-episode action queue.
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._initial_param_rounding_done = False
        self._train_rng_reset_after_first_batch_done = False

    @classmethod
    def from_pretrained(cls, cfg: Any) -> "DreamZeroPolicy":
        """Build DreamZero from the new embodied model registry entrypoint."""
        from .dreamzero_provider import dreamzero_model_provider

        return dreamzero_model_provider(config=_config_to_dreamzero(cfg))

    @staticmethod
    def default_lora_targets() -> dict[str, Any]:
        """Return DreamZero's default PEFT targets and full-train modules."""
        return {
            "target_modules": (
                r"action_head\.model\..*\."
                r"(?:q|k|v|o|ffn\.[02])"
            ),
            "modules_to_save": [
                "action_head.model.state_encoder",
                "action_head.model.action_encoder",
                "action_head.model.action_decoder",
            ],
        }

    @staticmethod
    def lora_requires_pretrained_checkpoint() -> bool:
        """DreamZero providers load component checkpoints from model config."""
        return False

    # ------------------------------------------------------------------
    # Shared training-loop interface
    # ------------------------------------------------------------------
    def _prepare_action_input(self, batch: Any) -> Any:
        """Prepare batch tensors for DreamZero forward."""
        if torch.is_tensor(batch):
            target_device = self.action_head.device
            if torch.is_floating_point(batch):
                target_dtype = self.action_head.dtype
                if batch.device == target_device and batch.dtype == target_dtype:
                    return batch
                return batch.to(device=target_device, dtype=target_dtype, non_blocking=True)
            if batch.device == target_device:
                return batch
            return batch.to(device=target_device, non_blocking=True)
        if isinstance(batch, BatchFeature):
            return BatchFeature(data={k: self._prepare_action_input(v) for k, v in batch.items()})
        if isinstance(batch, dict):
            return {k: self._prepare_action_input(v) for k, v in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self._prepare_action_input(v) for v in batch)
        if isinstance(batch, list):
            return [self._prepare_action_input(v) for v in batch]
        return batch

    def set_frozen_modules_to_eval_mode(self) -> None:
        """Keep frozen encoders/VAE in eval mode after outer train() calls."""
        self.action_head.set_frozen_modules_to_eval_mode()

    def train(self, mode: bool = True):
        """Set training mode while keeping frozen encoders in eval mode."""
        super().train(mode)
        if mode:
            self.set_frozen_modules_to_eval_mode()
        return self

    def on_after_data_iterators_initialized(
        self,
        *,
        args: Any,
        completed_steps: int,
        optimizer: torch.optim.Optimizer | None,
        ctx: DistributedContext,
    ) -> None:
        """Apply one-time DreamZero FSDP initialization compatibility rounding."""
        del args
        if optimizer is None or self._initial_param_rounding_done:
            return
        if completed_steps != 0:
            self._initial_param_rounding_done = True
            return

        dtype_name = self.config.fsdp_initial_param_rounding_dtype
        target_dtype = self._resolve_initial_param_rounding_dtype(dtype_name)
        if target_dtype is None:
            self._initial_param_rounding_done = True
            return

        rounded_params = 0
        rounded_numel = 0
        fallback_params = 0
        fallback_numel = 0
        skipped_params = 0

        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group.get("params", []):
                    if param is None or not param.requires_grad:
                        continue
                    if not torch.is_floating_point(param):
                        skipped_params += 1
                        continue

                    local = self._mutable_local_tensor(param)
                    if local is not None:
                        if not torch.is_floating_point(local):
                            skipped_params += 1
                            continue
                        local.copy_(local.to(dtype=target_dtype).to(dtype=local.dtype))
                        rounded_params += 1
                        rounded_numel += int(local.numel())
                        continue

                    param.data.copy_(
                        param.data.to(dtype=target_dtype).to(dtype=param.dtype)
                    )
                    fallback_params += 1
                    fallback_numel += int(param.numel())

        self._initial_param_rounding_done = True

        if ctx.is_main:
            logger.info(
                "DreamZero initial FSDP param rounding: dtype=%s "
                "local_params=%d local_numel=%d fallback_params=%d "
                "fallback_numel=%d skipped=%d",
                str(target_dtype).replace("torch.", ""),
                rounded_params,
                rounded_numel,
                fallback_params,
                fallback_numel,
                skipped_params,
            )

    @staticmethod
    def _resolve_initial_param_rounding_dtype(dtype_name: Any) -> torch.dtype | None:
        """Resolve the configured initial parameter-rounding dtype."""
        if dtype_name is None:
            return None
        mode = str(dtype_name).strip().lower()
        if mode in {"", "0", "false", "no", "off", "none", "null"}:
            return None
        if mode in {"1", "true", "yes", "on", "bf16", "bfloat16"}:
            return torch.bfloat16
        if mode in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(
            "DreamZeroConfig.fsdp_initial_param_rounding_dtype must be one of "
            "null/bf16/bfloat16/fp16/float16"
        )

    @staticmethod
    def _mutable_local_tensor(param: torch.Tensor) -> torch.Tensor | None:
        """Return the mutable local tensor backing a regular or DTensor parameter."""
        data = param.data
        local = getattr(data, "_local_tensor", None)
        if torch.is_tensor(local):
            return local
        detached = param.detach()
        local = getattr(detached, "_local_tensor", None)
        if torch.is_tensor(local):
            return local
        if torch.is_tensor(data) and not hasattr(data, "to_local"):
            return data
        return None

    def on_after_train_batch_fetch(
        self,
        *,
        args: Any,
        completed_steps: int,
        micro_step: int,
        batch: Any = None,
    ) -> None:
        """Apply DreamZero's first-forward RNG policy after the first batch fetch."""
        del micro_step, batch
        if self._train_rng_reset_after_first_batch_done:
            return
        if not self.config.train_rng_reset_after_first_batch:
            self._train_rng_reset_after_first_batch_done = True
            return
        if args.resume or completed_steps != 0:
            self._train_rng_reset_after_first_batch_done = True
            return

        seed = self.config.train_rng_seed
        if seed is None:
            seed = int(args.seed or 0)
        seed = int(seed)

        cpu_burn = int(self.config.train_rng_cpu_burn)
        if cpu_burn < 0:
            raise ValueError("DreamZeroConfig.train_rng_cpu_burn must be >= 0")

        from loongforge.embodied.train.utils.utils import set_seed

        set_seed(seed)
        if cpu_burn:
            torch.rand(cpu_burn)
        self._train_rng_reset_after_first_batch_done = True
        logger.info(
            "DreamZero reset training RNG after first batch fetch: seed=%d cpu_burn=%d",
            seed,
            cpu_burn,
        )

    def forward(
        self, batch: dict[str, Tensor], reduction: str = "mean"
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run a training step.

        Args:
            batch: Dict containing keys consumed by ``WANPolicyHead.forward``:
                ``images``, ``text``, ``text_attention_mask``, ``state``,
                ``action``, ``embodiment_id``, ``action_mask``, ``has_real_action``.
            reduction: Always ``"mean"`` for now; per-sample reduction is
                a P1 feature.

        Returns the total ``loss`` for backward and detached component metrics
        for logging.
        """
        prepared_batch = self._prepare_action_input(batch)
        action_input = (
            prepared_batch
            if isinstance(prepared_batch, BatchFeature)
            else BatchFeature(data=prepared_batch)
        )
        # WANPolicyHead expects a backbone_output positional arg; it is not
        # used in forward (DreamZero ActionHead is the entire policy). Pass
        # an empty BatchFeature to keep the abstract interface intact.
        out = self.action_head(BatchFeature(data={}), action_input)

        loss = out["loss"]
        log_loss_dict = {
            "loss": loss.detach(),
            "dynamics_loss": out["dynamics_loss"].detach(),
            "action_loss": out["action_loss"].detach(),
        }
        if reduction == "none":  # pragma: no cover - P1
            raise NotImplementedError("DreamZeroPolicy.forward(reduction='none') is not implemented")
        return loss, log_loss_dict

    # ------------------------------------------------------------------
    # PreTrainedPolicy inference interface (closed-loop autoregressive)
    # ------------------------------------------------------------------
    def get_optim_params(self) -> list:
        """Return trainable params; deferred to optimizer builder in trainer."""
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self) -> None:
        """Reset closed-loop state at episode boundary (called on env reset).

        Clears the action queue and the action head's autoregressive caches
        (KV / cross-attention / cached image+text features, current_start_frame)
        so the next ``select_action`` re-warms from the first frame.
        """
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.action_head.reset_inference()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict one action chunk via the closed-loop video+action denoiser.

        Returns a tensor shaped ``(batch, action_horizon, action_dim)`` in the
        model's (normalised) action space — denormalisation back to raw action
        units is handled by the dataset-level ``StateActionTransform.unapply``
        (q99) at the eval boundary.
        """
        self.eval()
        prepared = self._prepare_action_input(batch)
        action_input = prepared if isinstance(prepared, BatchFeature) else BatchFeature(data=prepared)
        out = self.action_head.lazy_joint_video_action(BatchFeature(data={}), action_input)
        return out["action_pred"]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action step, refilling the queue from a fresh chunk."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # (batch, n_action_steps, action_dim) -> queue of (batch, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # Megatron pipeline APIs expect set_input_tensor even when pipeline parallel size is 1.
    # Provide a no-op shim to satisfy forward_backward_no_pipelining.
    def set_input_tensor(self, input_tensor):
        """Set input tensor for pipeline parallelism (no-op for PP=1)."""
        self._input_tensor = input_tensor

    # Megatron checkpointing.generate_state_dict calls model.state_dict_for_save_checkpoint().
    # Default to torch nn.Module.state_dict() — adequate for non-PP, non-distributed-optim cases.
    def state_dict_for_save_checkpoint(self, destination=None, prefix: str = "", keep_vars: bool = False):
        """Return a Megatron-compatible checkpoint state dictionary."""
        return self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
