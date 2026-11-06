# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from DreamZero under the Apache-2.0 License.

"""DreamZero flow-matching action head.

The module owns DreamZero's training loss path and closed-loop inference path.
Submodules are supplied by ``dreamzero_provider``.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import logging
from typing import cast

from einops import rearrange
import torch
from torch import nn
import torch.distributed as dist
from torch.distributions import Beta
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from .base_action_head import ActionHead
from .flow_match_scheduler import FlowMatchScheduler
from .flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


logger = logging.getLogger(__name__)

# Per-layer KV / cross-attention cache element (one tensor per DiT layer).
# Upstream uses ``KVCacheType: TypeAlias = torch.Tensor``; the closed-loop
# caches are ``list[KVCacheType]`` (one entry per layer).
KVCacheType = torch.Tensor


def _dreamzero_resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    raw = str(value or "bfloat16").strip().lower()
    if raw in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    if raw in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported DreamZero model_dtype={value!r}")


def _dreamzero_skip_precomputed_pixel_preprocess(config, data) -> bool:
    if not bool(config.skip_precomputed_pixel_preprocess):
        return False
    if not (
        bool(config.precomputed_video_latents)
        and bool(config.precomputed_video_latents_strict)
        and bool(config.precomputed_first_frame_latents)
        and bool(config.precomputed_first_frame_latents_strict)
    ):
        return False
    video_key = str(config.precomputed_video_latents_key or "video_latents")
    first_frame_key = str(
        config.precomputed_first_frame_latents_key or "first_frame_latents"
    )
    return (
        data.get(video_key) is not None
        and data.get(first_frame_key) is not None
    )


_DREAMZERO_COMPILE_LOGGED = set()


def _dreamzero_rank0() -> bool:
    try:
        import torch.distributed as _d

        return not (_d.is_available() and _d.is_initialized()) or _d.get_rank() == 0
    except Exception:
        return True


def _dreamzero_log_once(key: str, msg: str) -> None:
    if key in _DREAMZERO_COMPILE_LOGGED:
        return
    _DREAMZERO_COMPILE_LOGGED.add(key)
    if _dreamzero_rank0():
        logger.info("%s", msg)


def _dreamzero_compile_loss_fn(fn, label: str, *, enabled: bool = True, mode: str = "reduce-overhead"):
    if not enabled:
        return fn
    if not hasattr(torch, "compile"):
        _dreamzero_log_once(
            f"{label}:missing",
            f"[dreamzero-fuse-loss] torch.compile unavailable, use eager {label}",
        )
        return fn
    kwargs = {
        "mode": mode,
        "fullgraph": False,
    }
    try:
        compiled = torch.compile(fn, **kwargs)
    except Exception as exc:
        _dreamzero_log_once(
            f"{label}:fail",
            f"[dreamzero-fuse-loss] failed to wrap {label}: {exc}",
        )
        return fn
    _dreamzero_log_once(
        label,
        f"[dreamzero-fuse-loss] wrapped {label} mode={kwargs['mode']}",
    )
    return compiled


def _dreamzero_weighted_video_loss(video_noise_pred, training_target, timestep_weight):
    dynamics_loss_per_sample = (
        video_noise_pred.float() - training_target.float()
    ).square().mean(dim=(1, 3, 4))
    return dynamics_loss_per_sample, (dynamics_loss_per_sample * timestep_weight).mean()


def _dreamzero_weighted_action_loss(
    action_noise_pred,
    training_target_action,
    action_mask,
    has_real_action,
    timestep_action_weight,
):
    action_loss_per_dim = (
        action_noise_pred.float() - training_target_action.float()
    ).square()
    action_loss_per_dim = action_loss_per_dim * action_mask
    action_loss_per_dim = action_loss_per_dim * has_real_action[:, None, None].float()
    return (action_loss_per_dim.mean(dim=2) * timestep_action_weight).mean()


def _dreamzero_prompt_emb_cache_key(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> bytes:
    ids = input_ids.detach().to(device="cpu", copy=True).contiguous()
    mask = attention_mask.detach().to(device="cpu", copy=True).contiguous()
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(ids.dtype).encode("ascii"))
    digest.update(str(mask.dtype).encode("ascii"))
    digest.update(repr(tuple(ids.shape)).encode("ascii"))
    digest.update(repr(tuple(mask.shape)).encode("ascii"))
    digest.update(ids.numpy().tobytes())
    digest.update(mask.numpy().tobytes())
    return digest.digest()


@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    """Configuration for WANPolicyHead training + inference paths.

    Inference-only fields (``cfg_scale``, ``sigma_shift``,
    ``num_inference_steps``, ``num_dit_steps``,
    ``decouple_inference_noise``, ``video_inference_final_noise``) are
    included for the closed-loop autoregressive path.
    """

    # Diffusion / VAE tiling
    add_pos_embed: bool = field(default=True)
    model_dtype: str = field(default="float32")
    input_embedding_dim: int = field(default=1536)
    backbone_embedding_dim: int = field(default=1536)
    tiled: bool = field(default=True)
    tile_size_height: int = field(default=34)
    tile_size_width: int = field(default=34)
    tile_stride_height: int = field(default=18)
    tile_stride_width: int = field(default=16)
    num_frame_per_block: int = field(default=1)
    target_video_height: int | None = field(default=None)
    target_video_width: int | None = field(default=None)

    # Inference-only (closed-loop autoregressive flow-matching).
    num_inference_steps: int = field(default=16)
    num_dit_steps: int = field(default=1)
    inference_seed: int = field(default=1140)
    cfg_scale: float = field(default=5.0)
    denoising_strength: float = field(default=1.0)
    sigma_shift: float = field(default=5.0)
    decouple_inference_noise: bool = field(default=False)
    video_inference_final_noise: float = field(default=0.0)

    # Trainability flags
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)

    # Sequence/action geometry
    hidden_size: int = field(default=1024)
    max_seq_len: int = field(default=1024)
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    num_frames: int = field(default=None)

    # Flow-matching noise sampling
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999)
    use_high_noise_emphasis: bool = field(default=False)
    high_noise_beta_alpha: float = field(default=3.0)
    decouple_video_action_noise: bool = field(default=False)
    video_noise_beta_alpha: float = field(default=3.0)
    video_noise_beta_beta: float = field(default=1.0)
    num_timestep_buckets: int = field(default=1000)
    batch_vae_encode: bool = field(default=False)
    prompt_emb_cache: str = field(default="")
    prompt_emb_cache_max_entries: int = field(default=128)
    precomputed_video_latents: bool = field(default=False)
    precomputed_video_latents_key: str = field(default="video_latents")
    precomputed_video_latents_layout: str = field(default="bcthw")
    precomputed_video_latents_strict: bool = field(default=False)
    precomputed_first_frame_latents: bool = field(default=False)
    precomputed_first_frame_latents_key: str = field(default="first_frame_latents")
    precomputed_first_frame_latents_strict: bool = field(default=False)
    precomputed_prompt_embs: bool = field(default=False)
    precomputed_prompt_embs_key: str = field(default="prompt_embs")
    precomputed_prompt_embs_strict: bool = field(default=False)
    skip_precomputed_pixel_preprocess: bool = field(default=False)
    precomputed_first_frame_only: bool = field(default=False)

    max_num_embodiments: int = field(default=32)

    detection_coeff: float = field(default=1.0)
    freeze_decode_layer: bool = field(default=False)
    expand_batch: int | None = field(default=None)
    use_vlln: bool = field(default=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    """Flow-matching action head.

    Constructor takes pre-built submodules (no Hydra). The provider
    (``dreamzero_provider.py``) is responsible for instantiating the four
    submodules from ``DreamZeroConfig`` and passing them here.

    Args:
        config: ``WANPolicyHeadConfig`` (training-relevant subset).
        text_encoder: ``WanTextEncoder`` (frozen).
        image_encoder: CLIP-like image encoder exposing ``encode_image`` (frozen).
        vae: ``WanVideoVAE`` / ``WanVideoVAE38`` exposing ``encode`` (frozen).
        model: ``CausalWanModel`` (trainable).
    """

    config_class = WANPolicyHeadConfig
    def __init__(
        self,
        config: WANPolicyHeadConfig,
        text_encoder: nn.Module,
        image_encoder: nn.Module,
        vae: nn.Module,
        model: nn.Module,
    ):
        super().__init__()
        # Tiling / geometry
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.hidden_size = config.hidden_size
        self.num_frames = config.num_frames
        self.input_embedding_dim = config.input_embedding_dim
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon

        # Submodules — externally constructed.
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.model = model

        # Flow-matching scheduler for training.
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

        # Video pixel normalisation for uint8 → [-1, 1].
        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Beta distributions for noise sampling.
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)

        # Training scheduler timesteps.
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)

        # Device used inside forward for tensor placement. Refresh at forward
        # entry after distributed wrapping has moved parameters to the runtime device.
        self._device = self.device

        self.config = config
        self._dreamzero_runtime_dtype = _dreamzero_resolve_torch_dtype(config.model_dtype)
        self._dreamzero_prompt_emb_cache = OrderedDict()
        self._dreamzero_prompt_emb_cache_hits = 0
        self._dreamzero_prompt_emb_cache_misses = 0
        self._dreamzero_video_loss_impl = _dreamzero_compile_loss_fn(
            _dreamzero_weighted_video_loss,
            "weighted_video_loss",
            mode="reduce-overhead",
        )
        self._dreamzero_action_loss_impl = _dreamzero_compile_loss_fn(
            _dreamzero_weighted_action_loss,
            "weighted_action_loss",
            mode="reduce-overhead",
        )

        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

        # --------------------------------------------------------------
        # Inference-only closed-loop state.
        # Unused by the training forward; populated lazily on first
        # ``lazy_joint_video_action`` call and cleared by ``reset_inference``.
        # --------------------------------------------------------------
        self.num_inference_steps = config.num_inference_steps
        self.seed = config.inference_seed
        self.cfg_scale = config.cfg_scale
        self.denoising_strength = config.denoising_strength
        self.sigma_shift = config.sigma_shift

        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None

        # Inference parallelism (single-rank by default; multi-rank P2P
        # exchange wired through ``parallelize`` if ever needed).
        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None

        # TensorRT inference is not wired in this implementation.
        self.trt_engine = None

        self.dit_step_mask = self._build_dit_step_mask()

    def _build_dit_step_mask(self) -> list[bool]:
        """DIT-step skip mask for inference.

        Fixed boolean schedule selecting which diffusion steps run the DiT
        versus reuse the previous prediction. ``num_dit_steps=1`` intentionally
        falls back to an all-True mask. The first step must always run.
        """
        num_dit_steps = self.config.num_dit_steps
        masks = {
            5: [
                True, True, True, False, False, False, False, True,
                False, False, False, False, True, False, False, False,
            ],
            6: [
                True, True, False, False, False, True, False, False,
                False, False, True, False, False, False, True, True,
            ],
            7: [
                True, True, True, False, False, False, True, False,
                False, False, True, False, False, False, True, True,
            ],
            8: [
                True, True, True, False, False, False, True, False,
                False, False, True, False, False, True, True, True,
            ],
        }
        mask = masks.get(num_dit_steps, [True] * 16)
        assert mask[0], "first DIT step must be True"
        return mask

    # ------------------------------------------------------------------
    # Trainability
    # ------------------------------------------------------------------
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        """Enable/disable gradients for the projector and diffusion model."""
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        logger.info(f"Tune action head projector: {self.tune_projector}")
        logger.info(f"Tune action head diffusion model: {self.tune_diffusion_model}")

        # Encoders / VAE are always frozen.
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

        self.print_trainable_params()

    def print_trainable_params(self):
        """Log total and trainable parameter counts for the diffusion model."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Diffusion model params: total={total:,} trainable={trainable:,}")

    def set_frozen_modules_to_eval_mode(self):
        """Hugging Face calls model.train() each step; force frozen modules to eval."""
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def sample_time(self, batch_size, device, dtype):
        """Sample normalized flow-matching timesteps from the beta distribution."""
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    @staticmethod
    def _dreamzero_autocast_enabled(device_type: str) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device_type))
        except TypeError:
            if device_type == "cuda":
                return bool(torch.is_autocast_enabled())
            return False

    @staticmethod
    def _dreamzero_autocast_dtype(device_type: str) -> torch.dtype | None:
        try:
            return torch.get_autocast_dtype(device_type)
        except (AttributeError, RuntimeError):
            if device_type == "cuda":
                try:
                    return torch.get_autocast_gpu_dtype()
                except (AttributeError, RuntimeError):
                    return None
            return None

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Wrap a raw batch dict into a ``BatchFeature``."""
        return BatchFeature(data=batch)

    def preprocess_image(self, image):
        """Normalize image pixels to [-1, 1] and reorder to channel-first."""
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def _prompt_seq_lens(self, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        return seq_lens

    def _encode_prompt_raw(self, input_ids, attention_mask):
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        return prompt_emb

    def _postprocess_prompt_emb(self, prompt_emb, seq_lens):
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def _encode_prompt_uncached(self, input_ids, attention_mask):
        seq_lens = self._prompt_seq_lens(attention_mask)
        prompt_emb = self._encode_prompt_raw(input_ids, attention_mask)
        prompt_emb = self._postprocess_prompt_emb(prompt_emb, seq_lens)
        return prompt_emb

    def _prompt_emb_cache_mode(self) -> str | None:
        raw = str(self.config.prompt_emb_cache or "").strip().lower()
        if raw in {"1", "true", "yes", "on", "gpu", "cuda", "device"}:
            return "gpu"
        if raw == "cpu":
            return "cpu"
        return None

    def _prompt_emb_cache_max_entries(self) -> int:
        try:
            return max(1, int(self.config.prompt_emb_cache_max_entries))
        except (TypeError, ValueError):
            return 128

    def _get_cached_prompt_embs(self, input_ids, attention_mask, cache_mode: str):
        keys = [
            _dreamzero_prompt_emb_cache_key(input_ids[i], attention_mask[i])
            for i in range(input_ids.shape[0])
        ]
        cached = []
        for key in keys:
            value = self._dreamzero_prompt_emb_cache.get(key)
            if value is None:
                self._dreamzero_prompt_emb_cache_misses += len(keys) - len(cached)
                return keys, None
            self._dreamzero_prompt_emb_cache.move_to_end(key)
            cached.append(value)
        self._dreamzero_prompt_emb_cache_hits += len(cached)
        device = input_ids.device
        if not torch.is_tensor(input_ids) or input_ids.device.type == "cpu":
            device = torch.device(self._device)
        return keys, torch.stack(
            [x.to(device=device, dtype=torch.bfloat16, non_blocking=True) for x in cached],
            dim=0,
        )

    def _update_prompt_emb_cache(self, keys, prompt_emb_raw, cache_mode: str) -> None:
        if not keys:
            return
        max_entries = self._prompt_emb_cache_max_entries()
        for key, emb in zip(keys, prompt_emb_raw.detach()):
            if cache_mode == "cpu":
                value = emb.to(device="cpu", dtype=torch.bfloat16, copy=True)
            else:
                value = emb.to(dtype=torch.bfloat16, copy=True)
            self._dreamzero_prompt_emb_cache[key] = value
            self._dreamzero_prompt_emb_cache.move_to_end(key)
        while len(self._dreamzero_prompt_emb_cache) > max_entries:
            self._dreamzero_prompt_emb_cache.popitem(last=False)

    def _all_ranks_have_prompt_cache_hit(self, local_hit: bool, input_ids: torch.Tensor) -> bool:
        if not (dist.is_available() and dist.is_initialized()):
            return local_hit

        device = input_ids.device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device(self._device)
        hit = torch.tensor(1 if local_hit else 0, device=device, dtype=torch.int32)
        dist.all_reduce(hit, op=dist.ReduceOp.MIN)
        return bool(hit.item())

    def encode_prompt(self, input_ids, attention_mask):
        """Encode prompt tokens into text embeddings, using cache when enabled."""
        cache_mode = self._prompt_emb_cache_mode()
        if cache_mode is None:
            return self._encode_prompt_uncached(input_ids, attention_mask)

        _dreamzero_log_once(
            "prompt_emb_cache",
            f"[dreamzero-text] prompt_emb_cache={cache_mode}, "
            "reuse raw frozen text embeddings by token ids and attention mask",
        )
        keys, cached = self._get_cached_prompt_embs(input_ids, attention_mask, cache_mode)
        local_hit = cached is not None
        global_hit = self._all_ranks_have_prompt_cache_hit(local_hit, input_ids)
        if cached is not None and global_hit:
            seq_lens = self._prompt_seq_lens(attention_mask)
            return self._postprocess_prompt_emb(cached, seq_lens)

        seq_lens = self._prompt_seq_lens(attention_mask)
        prompt_emb_raw = self._encode_prompt_raw(input_ids, attention_mask)
        self._update_prompt_emb_cache(keys, prompt_emb_raw, cache_mode)
        return self._postprocess_prompt_emb(prompt_emb_raw, seq_lens)

    def _scheduler_lookup_by_id(
        self,
        name: str,
        table: torch.Tensor,
        timestep_id: torch.Tensor,
        ref: torch.Tensor,
        dtype=None,
    ):
        values = table[timestep_id] if timestep_id.device.type == "cpu" else table.to(timestep_id.device)[timestep_id]
        return values.to(device=ref.device, dtype=dtype or ref.dtype, non_blocking=True)

    def _scheduler_timesteps_by_id(self, timestep_id: torch.Tensor, ref: torch.Tensor):
        return self._scheduler_lookup_by_id(
            "timesteps",
            self.scheduler.timesteps,
            timestep_id,
            ref,
            dtype=self.scheduler.timesteps.dtype,
        )

    def _scheduler_add_noise_by_id(self, original_samples, noise, timestep_id):
        sigma = self._scheduler_lookup_by_id(
            "sigmas",
            self.scheduler.sigmas,
            timestep_id,
            original_samples,
            dtype=original_samples.dtype,
        )
        while sigma.dim() < original_samples.dim():
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def _scheduler_training_weight_by_id(self, timestep_id, ref_tensor):
        return self._scheduler_lookup_by_id(
            "linear_timesteps_weights",
            self.scheduler.linear_timesteps_weights,
            timestep_id,
            ref_tensor,
            dtype=torch.float32,
        )

    def _ensure_vae_on_device(self, ref_tensor):
        """Move the VAE to the active input device/dtype when needed."""
        ref_device = ref_tensor.device
        ref_dtype = torch.bfloat16
        first_param = next(self.vae.parameters(), None)
        needs_move = first_param is None or first_param.device != ref_device or first_param.dtype != ref_dtype
        if needs_move:
            self.vae.to(device=ref_device, dtype=ref_dtype)
        self.vae.eval()
        self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        """Encode video frames into VAE latents, moving the VAE onto device first."""
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def _precomputed_video_latents(self, data, videos):
        """Return precomputed video latents from ``data`` if configured, else None."""
        if not bool(self.config.precomputed_video_latents):
            return None
        latent_key = str(self.config.precomputed_video_latents_key or "video_latents")
        latents = data.get(latent_key)
        if latents is None:
            msg = (
                f"[dreamzero-vae] precomputed_video_latents=true, "
                f"but batch has no {latent_key!r}; fallback to online VAE encode"
            )
            if bool(self.config.precomputed_video_latents_strict):
                raise KeyError(msg)
            _dreamzero_log_once("precomputed_video_latents_missing", msg)
            return None
        if not torch.is_tensor(latents):
            raise TypeError(f"batch[{latent_key!r}] must be a torch.Tensor, got {type(latents)!r}")

        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(
                f"batch[{latent_key!r}] must have shape [B,C,T,H,W] (or [C,T,H,W] before collate), "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[0] != videos.shape[0]:
            raise ValueError(
                f"batch[{latent_key!r}] batch size {latents.shape[0]} "
                f"does not match videos batch size {videos.shape[0]}"
            )

        layout = str(self.config.precomputed_video_latents_layout or "bcthw").lower()
        if layout == "btchw":
            latents = latents.transpose(1, 2)
        elif layout != "bcthw":
            raise ValueError(
                "precomputed_video_latents_layout must be 'bcthw' or 'btchw', "
                f"got {layout!r}"
            )

        latents = latents.to(device=self._device, dtype=self._runtime_dtype(), non_blocking=True)
        _dreamzero_log_once(
            "precomputed_video_latents",
            f"[dreamzero-vae] precomputed_video_latents=true, "
            f"use batch[{latent_key!r}] as video latents; layout={layout}",
        )
        return latents

    def _precomputed_first_frame_latents(self, data, image):
        if not bool(self.config.precomputed_first_frame_latents):
            return None
        latent_key = str(
            self.config.precomputed_first_frame_latents_key or "first_frame_latents"
        )
        latents = data.get(latent_key)
        if latents is None:
            msg = (
                f"[dreamzero-vae] precomputed_first_frame_latents=true, "
                f"but batch has no {latent_key!r}; fallback to online first-frame VAE encode"
            )
            if bool(self.config.precomputed_first_frame_latents_strict):
                raise KeyError(msg)
            _dreamzero_log_once("precomputed_first_frame_latents_missing", msg)
            return None
        if not torch.is_tensor(latents):
            raise TypeError(f"batch[{latent_key!r}] must be a torch.Tensor, got {type(latents)!r}")
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        if latents.ndim != 5:
            raise ValueError(
                f"batch[{latent_key!r}] must have shape [B,C,T,H,W] (or [C,T,H,W] before collate), "
                f"got {tuple(latents.shape)}"
            )
        if latents.shape[0] != image.shape[0]:
            raise ValueError(
                f"batch[{latent_key!r}] batch size {latents.shape[0]} does not match image batch size {image.shape[0]}"
            )
        latents = latents.to(device=self._device, dtype=self._runtime_dtype(), non_blocking=True)
        _dreamzero_log_once(
            "precomputed_first_frame_latents",
            f"[dreamzero-vae] precomputed_first_frame_latents=true, "
            f"use batch[{latent_key!r}] as first-frame VAE latents",
        )
        return latents

    def _precomputed_prompt_embs(self, data, input_ids, attention_mask):
        if not bool(self.config.precomputed_prompt_embs):
            return None
        prompt_key = str(self.config.precomputed_prompt_embs_key or "prompt_embs")
        prompt_embs = data.get(prompt_key)
        if prompt_embs is None:
            msg = (
                f"[dreamzero-text] precomputed_prompt_embs=true, "
                f"but batch has no {prompt_key!r}; fallback to online text encode"
            )
            if bool(self.config.precomputed_prompt_embs_strict):
                raise KeyError(msg)
            _dreamzero_log_once("precomputed_prompt_embs_missing", msg)
            return None
        if not torch.is_tensor(prompt_embs):
            raise TypeError(f"batch[{prompt_key!r}] must be a torch.Tensor, got {type(prompt_embs)!r}")
        if prompt_embs.ndim == 2:
            prompt_embs = prompt_embs.unsqueeze(0)
        if prompt_embs.ndim != 3:
            raise ValueError(
                f"batch[{prompt_key!r}] must have shape [B,L,C] (or [L,C] before collate), "
                f"got {tuple(prompt_embs.shape)}"
            )
        if prompt_embs.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"batch[{prompt_key!r}] batch size {prompt_embs.shape[0]} "
                f"does not match text batch size {input_ids.shape[0]}"
            )
        if prompt_embs.shape[1] != input_ids.shape[1]:
            raise ValueError(
                f"batch[{prompt_key!r}] text length {prompt_embs.shape[1]} "
                f"does not match input_ids length {input_ids.shape[1]}"
            )
        prompt_embs = prompt_embs.to(device=self._device, dtype=torch.bfloat16, non_blocking=True)
        _dreamzero_log_once(
            "precomputed_prompt_embs",
            f"[dreamzero-text] precomputed_prompt_embs=true, "
            f"use batch[{prompt_key!r}] as raw prompt embeddings and apply batch postprocess",
        )
        seq_lens = self._prompt_seq_lens(attention_mask)
        return self._postprocess_prompt_emb(prompt_embs, seq_lens)

    def encode_clip_image(self, image, data=None):
        """Encode image frames with the CLIP image encoder."""
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            return self.image_encoder.encode_image(image)

    def _needs_first_frame_latent_for_train(self) -> bool:
        return bool(getattr(self.model, "concat_first_frame_latent", False)) or (
            getattr(self.model, "model_type", None) == "i2v"
        )

    def encode_image(self, image, num_frames, height, width, data=None):
        """Encode image conditioning into CLIP context and first-frame VAE latents."""
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            precomputed_first_frame_latents = self._precomputed_first_frame_latents(data or {}, image)
            if precomputed_first_frame_latents is None:
                image_input = image.transpose(1, 2)
                image_zeros = torch.zeros(
                    batch_size, 3, num_frames - 1, height, width,
                    dtype=torch.bfloat16, device=self._device,
                )
                self._ensure_vae_on_device(image_input)
                with torch.no_grad():
                    y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            else:
                y = precomputed_first_frame_latents
            num_t = y.shape[2]
            h_latent, w_latent = y.shape[3], y.shape[4]
            msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
            msk[:, :, 0:1, :, :] = 1
            new_image = y[:, :, 0:1]
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image

    def prepare_extra_input(self, latents=None):
        """Return additional model inputs derived from latents (none by default)."""
        return {}

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self._device = self.device
        self.set_frozen_modules_to_eval_mode()

        data = action_input
        embodiment_id = action_input.embodiment_id
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask
        runtime_dtype = self._runtime_dtype()
        state_features = action_input.state.to(dtype=runtime_dtype)
        actions = action_input.action.to(dtype=runtime_dtype)
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]
        skip_precomputed_pixel_preprocess = _dreamzero_skip_precomputed_pixel_preprocess(
            self.config, data
        )
        if skip_precomputed_pixel_preprocess:
            _dreamzero_log_once(
                "skip_precomputed_pixel_preprocess",
                "[dreamzero-data] skip_precomputed_pixel_preprocess=true, "
                "skip pixel normalize/resize because strict video/first-frame caches are present",
            )
        video_latent_key = str(
            self.config.precomputed_video_latents_key or "video_latents"
        )
        precomputed_first_frame_only = (
            bool(self.config.precomputed_first_frame_only)
            and bool(self.config.precomputed_video_latents)
            and not self._needs_first_frame_latent_for_train()
            and data.get(video_latent_key) is not None
        )
        if precomputed_first_frame_only:
            _dreamzero_log_once(
                "precomputed_first_frame_only",
                "[dreamzero-data] precomputed_first_frame_only=true, "
                "normalize only the first frame because video latents are precomputed",
            )
        if precomputed_first_frame_only:
            videos = rearrange(videos[:, :1], "b t h w c -> b c t h w")
        else:
            videos = rearrange(videos, "b t h w c -> b c t h w")
        if not skip_precomputed_pixel_preprocess and videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=runtime_dtype)
        precomputed_prompt_embs = self._precomputed_prompt_embs(
            data,
            data["text"],
            data["text_attention_mask"],
        )
        if precomputed_prompt_embs is None:
            prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])
        else:
            prompt_embs = precomputed_prompt_embs
        # Optional resize so latent spatial dims align with DiT.
        target_h = self.config.target_video_height
        target_w = self.config.target_video_width
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if not skip_precomputed_pixel_preprocess and target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)
            videos = videos.to(dtype=self._runtime_dtype())

        precomputed_latents = self._precomputed_video_latents(data, videos)
        if precomputed_latents is None:
            latents = self.encode_video(
                videos, self.tiled,
                (self.tile_size_height, self.tile_size_width),
                (self.tile_stride_height, self.tile_stride_width),
            )
        else:
            latents = precomputed_latents
        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)
        if self._needs_first_frame_latent_for_train():
            clip_feas, ys, _ = self.encode_image(image, num_frames, height, width, data=data)
        else:
            clip_feas = self.encode_clip_image(image, data=data)
            ys = None
        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device) if ys is not None else None
        prompt_embs = prompt_embs.to(self._device)

        # Loss prep
        noise = torch.randn_like(latents)
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)

        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
        elif self.config.use_high_noise_emphasis:
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
        else:
            timestep_id = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (noise.shape[0], noise.shape[1]),
            )

        timestep_id_block = timestep_id[:, 1:].reshape(
            timestep_id.shape[0], -1, self.num_frame_per_block,
        )
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1] - 1) == (
                self.model.num_action_per_block // self.num_frame_per_block
            ), (
                f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, "
                f"video.shape, {videos.shape}, latents.shape, {latents.shape}"
            )
            assert (noise.shape[1] - 1) / state_features.shape[1] == (
                self.num_frame_per_block // self.model.num_state_per_block
            ), (
                f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, "
                f"video.shape, {videos.shape}, latents.shape, {latents.shape}"
            )

            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                timestep_action_id = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (actions.shape[0], actions.shape[1]),
                )
            else:
                timestep_action_id = timestep_id_block.repeat(
                    1, 1, actions.shape[1] // (noise.shape[1] - 1),
                )
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
        else:
            noise_action = None
            timestep_action_id = None
        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        seq_len = noise.shape[1] * (noise.shape[3] // 2) * (noise.shape[4] // 2)

        _dreamzero_log_once(
            "scheduler_timestep_id_lookup",
            "[dreamzero-scheduler] use sampled timestep_id directly "
            "for add_noise/training_weight",
        )
        timestep = self._scheduler_timesteps_by_id(timestep_id, latents)
        noisy_latents = self._scheduler_add_noise_by_id(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep_id.flatten(0, 1),
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = (noise - latents).transpose(1, 2)

        if actions.numel() > 0:
            timestep_action = self._scheduler_timesteps_by_id(timestep_action_id, actions)
            noisy_actions = self._scheduler_add_noise_by_id(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action_id.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = noise_action - actions
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep,
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action,
                    clean_x=latents.transpose(1, 2),
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action,
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                )

            # DiT patch_embedding stride (1,2,2): output spatial may be smaller than latent
            # when H or W is odd → crop target to match.
            if training_target.shape != video_noise_pred.shape:
                training_target = training_target[
                    ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
                ]
            timestep_weight = self._scheduler_training_weight_by_id(
                timestep_id.flatten(0, 1),
                video_noise_pred,
            ).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            dynamics_loss_per_sample, weighted_dynamics_loss = self._dreamzero_video_loss_impl(
                video_noise_pred,
                training_target,
                timestep_weight,
            )

            if actions.numel() > 0:
                timestep_action_weight = self._scheduler_training_weight_by_id(
                    timestep_action_id.flatten(0, 1),
                    action_noise_pred,
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = self._dreamzero_action_loss_impl(
                    action_noise_pred,
                    training_target_action,
                    action_mask,
                    has_real_action,
                    timestep_action_weight,
                )
                loss = weighted_dynamics_loss + weighted_action_loss
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss

        return BatchFeature(data={
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        })

    # ------------------------------------------------------------------
    # Inference — closed-loop autoregressive flow-matching
    # ------------------------------------------------------------------
    def reset_inference(self) -> None:
        """Clear closed-loop autoregressive state for a new episode.

        Mirrors the DreamZero serve-time ``reset`` (current_start_frame -> 0)
        and additionally drops the cached image/text features and KV caches so
        the next ``lazy_joint_video_action`` re-warms from the first frame.
        """
        self.kv_cache1 = None
        self.kv_cache_neg = None
        self.crossattn_cache = None
        self.crossattn_cache_neg = None
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        """Generate a random noise tensor, optionally seeded, on the target device."""
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def _get_caches(self, kv_caches_input: list[KVCacheType]) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            return [kv_caches_input[0]] if self.ip_rank == 0 else [kv_caches_input[1]]
        assert len(kv_caches_input) <= 2
        kv_caches = [kv_caches_input[0]]
        if self.cfg_scale != 1.0:
            kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                return [(data["text"], data["text_attention_mask"])]
            return [(data["text_negative"], data["text_attention_mask_negative"])]
        text_inputs = [(data["text"], data["text_attention_mask"])]
        if self.cfg_scale != 1.0:
            text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs

    def _create_kv_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device, frame_seqlen: int,
    ) -> tuple[list[KVCacheType], list[KVCacheType]]:
        """Per-GPU KV cache. Uses model num_heads/head_dim (5B=24, 14B=40)."""
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        kv_cache1: list[KVCacheType] = []
        kv_cache_neg: list[KVCacheType] = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device))
            kv_cache_neg.append(torch.zeros([2, batch_size, 0, num_heads, head_dim], dtype=dtype, device=device))
        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[list[KVCacheType], list[KVCacheType]]:
        """Per-GPU cross-attention cache (text seqlen pinned to 512)."""
        num_heads = self.model.num_heads
        head_dim = self.model.dim // num_heads
        crossattn_cache: list[KVCacheType] = []
        crossattn_cache_neg: list[KVCacheType] = []
        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device)
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, num_heads, head_dim], dtype=dtype, device=device)
            )
        return crossattn_cache, crossattn_cache_neg

    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action,
        timestep_action,
        state,
        embodiment_id,
        context,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                noisy_input,
                timestep,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                context=prompt_emb,
                seq_len=seq_len,
                y=y,
                clip_feature=clip_feature,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_frame=kv_cache_metadata["start_frame"],
            )
            if kv_cache_metadata["update_kv_cache"]:
                for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                    kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device)
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self, predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])
        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(
                op=dist.isend, tensor=pred,
                group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group,
            )
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(
                op=dist.irecv, tensor=other_pred,
                group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group,
            )
            for other_pred in other_predictions
        ]
        reqs = dist.batch_isend_irecv(send_ops + recv_ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list, output_predictions)

    def should_run_model(self, index, current_timestep, prev_predictions):
        """Return whether the DiT should run for this diffusion step index."""
        return self.dit_step_mask[index]

    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        """Extrapolate the next prediction from cached first-order slopes."""
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2
        v_prime = (f1 - f2) / h_past
        damping_factor = 0.25
        return f1 + (v_prime * h_curr) * damping_factor

    @torch.no_grad()
    def lazy_joint_video_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        """Closed-loop autoregressive video+action denoising (one frame block).

        Advances ``current_start_frame`` by ``num_frame_per_block`` each call
        and reuses the per-episode KV / cross-attention caches. TensorRT
        scaffolding is intentionally omitted.
        """
        self._device = self.device
        self.set_frozen_modules_to_eval_mode()
        data = action_input

        videos = data["images"]
        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        # Wan 5B: resize to target resolution so latent matches DiT.
        target_h = self.config.target_video_height
        target_w = self.config.target_video_width
        if target_h is None or target_w is None:
            if getattr(self.model, "frame_seqlen", None) in (50, 55):
                target_h, target_w = 176, 320
            else:
                target_h, target_w = None, None
        if target_h is not None and target_w is not None:
            _, _, _, h, w = videos.shape
            if (h, w) != (target_h, target_w):
                b, c, t, _, _ = videos.shape
                videos = torch.nn.functional.interpolate(
                    videos.reshape(b * t, c, h, w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, c, t, target_h, target_w)

        # Session reset triggers: new/changed language, single frame, or
        # exceeding the local attention window.
        if self.language is None:
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            self.current_start_frame = 0
            self.language = data["text"]
        elif videos.shape[2] == 1:
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            self.current_start_frame = 0

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] in (4, 9):
            # real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(image, self.num_frames, height, width, data=data)
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)
        assert self.clip_feas is not None and self.ys is not None, "clip_feas and ys must be set"

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
        elif self.current_start_frame != 0:
            # real-world execution path
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                pass  # no further action
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
                first_frame = videos[:, :, 0:1]
                videos = torch.cat([first_frame, videos], dim=2)
            else:
                first_frame = videos[:, :, 0:1]
                videos = torch.cat([first_frame, videos], dim=2)
            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        noise_obs = self.generate_noise(
            (image.shape[0], image.shape[1], self.num_frame_per_block, image.shape[3], image.shape[4]),
            seed=self.seed, device=self._device, dtype=torch.bfloat16,
        )
        noise_action = self.generate_noise(
            (image.shape[0], self.action_horizon, self.model.action_dim),
            seed=self.seed, device=self._device, dtype=torch.bfloat16,
        )
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        # DiT patch_embedding uses stride (1,2,2): tokens per frame = (H//2)*(W//2)
        frame_seqlen = (height // 2) * (width // 2)
        seq_len = num_frames * frame_seqlen

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size, dtype=noise_obs.dtype, device=noise_obs.device, frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size, dtype=noise_obs.dtype, device=noise_obs.device,
            )

        assert self.kv_cache1 is not None and self.kv_cache_neg is not None
        assert self.crossattn_cache is not None and self.crossattn_cache_neg is not None
        kv_caches = self._get_caches([self.kv_cache1, self.kv_cache_neg])
        crossattn_caches = self._get_caches([self.crossattn_cache, self.crossattn_cache_neg])

        # First-frame KV warm-up.
        if self.current_start_frame == 0:
            timestep = torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None, timestep_action=None, state=None, embodiment_id=None,
                context=prompt_embs, seq_len=frame_seqlen, y=self.ys[:, :, 0:1],
                clip_feature=self.clip_feas, kv_caches=kv_caches, crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(start_frame=0, update_kv_cache=True),
            )
            self.current_start_frame += 1

        # Warm up prior frame block into KV cache (autoregressive context).
        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            if self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
            else:
                y = self.ys[:, :, -self.num_frame_per_block:]
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64,
            ) * 0
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None, timestep_action=None, state=None, embodiment_id=None,
                context=prompt_embs, seq_len=seq_len, y=y,
                clip_feature=self.clip_feas, kv_caches=kv_caches, crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Inference schedulers (UniPC multistep): one for video, one for action.
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps, shift=1, use_dynamic_shifting=False,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps, shift=1, use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        # Decoupled inference: video sigmas end at video_final_noise instead of 0.
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)

        prev_predictions = []
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]

            timestep = torch.ones(
                [batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon], device=noise_obs.device, dtype=torch.int64,
            ) * action_timestep

            if self.should_run_model(index, current_timestep, prev_predictions):
                if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(start_frame=self.current_start_frame, update_kv_cache=False),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, flow_pred_uncond_action = predictions[1]
                flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                if len(prev_predictions) > 2:
                    prev_predictions.pop(0)
            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep, sample=noisy_input, step_index=index, return_dict=False,
            )[0]
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep, sample=noisy_input_action, step_index=index, return_dict=False,
            )[0]

        latents_action = noisy_input_action
        output = noisy_input
        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        return BatchFeature(data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)})

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def device(self):
        """Return the device of the model's parameters."""
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        """Return the dtype of the model's parameters."""
        return next(iter(self.parameters())).dtype

    def _runtime_dtype(self):
        """Return the configured runtime compute dtype."""
        return self._dreamzero_runtime_dtype
