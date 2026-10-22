import math

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipEncoder as TransformersSiglipEncoder
from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer as TransformersSiglipEncoderLayer
from transformers.models.siglip.modeling_siglip import SiglipMultiheadAttentionPoolingHead, SiglipVisionEmbeddings
from transformers.utils import can_return_tuple

# ── Fine-grained NVTX / torch.profiler range markers (opt-in, gated) ──
try:
    from loongforge.embodied.profiling_nvtx import nvtx_range, nvtx_push, nvtx_pop
except Exception:  # pragma: no cover - profiling is optional
    import contextlib as _contextlib

    def nvtx_push(_msg):  # type: ignore
        return None

    def nvtx_pop():  # type: ignore
        return None

    @_contextlib.contextmanager
    def nvtx_range(_msg):  # type: ignore
        yield


def get_transformers_siglip_vision_config(vision_in_channels: int = 3):
    return CONFIG_MAPPING['siglip_vision_model'](
        hidden_size=1152,
        intermediate_size=4304,
        num_channels=vision_in_channels,
        num_attention_heads=16,
        num_hidden_layers=27,
        num_image_tokens=256,
        patch_size=14,
        projection_dim=2304,
        projector_hidden_act='gelu_fast',
        torch_dtype='float32',
        vision_use_head=False,
    )


class SiglipEncoderLayer(TransformersSiglipEncoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool = False,
        temporal_attention: bool = False,
        **kwargs,  # newer transformers passes `output_hidden_states` etc.
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor]:
        # Absorb extra kwargs (e.g. output_hidden_states) that newer transformers
        # SiglipEncoder.forward forwards to its layers — we don't use them here,
        # but their presence in the signature avoids TypeError.
        if not temporal_attention:
            return super().forward(hidden_states, attention_mask, output_attentions=output_attentions)

        if hidden_states.ndim != 4:
            raise ValueError(f'temporal hidden_states must have shape (B,T,N,D), got {tuple(hidden_states.shape)}')

        batch_size, num_frames, num_patches, embed_dim = hidden_states.shape
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size * num_patches, num_frames, embed_dim)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        hidden_states = hidden_states.reshape(batch_size, num_patches, num_frames, embed_dim).permute(0, 2, 1, 3)
        hidden_states = residual + hidden_states

        spatial_states = hidden_states.reshape(batch_size * num_frames, num_patches, embed_dim)
        layer_outputs = super().forward(
            spatial_states,
            attention_mask=None,
            output_attentions=output_attentions,
        )
        hidden_states = layer_outputs[0].reshape(batch_size, num_frames, num_patches, embed_dim)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (layer_outputs[1],)
        return outputs


class SiglipEncoder(TransformersSiglipEncoder):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False


class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_ada_rms_norm: bool = False):
        super().__init__()
        self.eps = eps
        self.use_ada_rms_norm = use_ada_rms_norm
        if use_ada_rms_norm:
            self.dense = nn.Linear(dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, cond: torch.Tensor | None = None):
        with nvtx_range('rmsnorm'):
            normed_inputs = self._norm(x.float())

            if self.use_ada_rms_norm:
                modulation = self.dense(cond)
                scale, shift, gate = torch.chunk(modulation.unsqueeze(1), 3, dim=-1)
                normed_inputs = normed_inputs.float() * (1.0 + scale.float()) + shift.float()
                return normed_inputs.type_as(x), gate.type_as(x)

            # Gemma2 uses (1.0 + weight) instead of just weight
            # See https://github.com/huggingface/transformers/pull/29402
            output = normed_inputs * (1.0 + self.weight.float())
            return output.type_as(x)

    def extra_repr(self):
        if self.use_ada_rms_norm:
            return f'{tuple(self.dense.weight.shape)}, eps={self.eps}, use_ada_rms_norm=True'
        else:
            return f'{tuple(self.weight.shape)}, eps={self.eps}'


class SiglipVisionTransformer(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig,
        temporal_attention_interval: int = 4,
        detach_history_frames: bool = True,
    ):
        super().__init__()
        self.config = config
        self.config._attn_implementation = 'sdpa'
        if temporal_attention_interval <= 0:
            raise ValueError(f'temporal_attention_interval must be positive, got {temporal_attention_interval}')
        self.temporal_attention_interval = temporal_attention_interval
        self.detach_history_frames = detach_history_frames
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.use_head = True if not hasattr(config, 'vision_use_head') else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        image_attention_mask: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        """Forward pass of the SigLIP vision encoder.

        Args:
            pixel_values: Image tensor expected by SigLIP, shaped (B, C, H, W)
                or video tensor shaped (B, T, C, H, W).
            image_attention_mask: Optional valid-frame mask for video input,
                shaped (B, T). True means the frame is real, False means padded.
            output_attentions: Whether to return attention maps.
            output_hidden_states: Whether to return hidden states.
            interpolate_pos_encoding: Enable pos-encoding interpolation for different sizes.

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and optionally pooled output.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if pixel_values.ndim == 5:
            return self._forward_video(
                pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )
        if pixel_values.ndim != 4:
            raise ValueError(f'pixel_values must have shape (B,C,H,W) or (B,T,C,H,W), got {tuple(pixel_values.shape)}')

        with nvtx_range('siglip_embed'):
            hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with nvtx_range('siglip_encoder'):
                encoder_outputs: BaseModelOutput = self.encoder(
                    inputs_embeds=hidden_states,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
            last_hidden_state = encoder_outputs.last_hidden_state
            last_hidden_state = self.post_layernorm(last_hidden_state)

            pooler_output = self.head(last_hidden_state) if self.use_head else None

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

    def _forward_video(
        self,
        pixel_values: torch.Tensor,
        image_attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        if pixel_values.ndim != 5:
            raise ValueError(f'video pixel_values must have shape (B,T,C,H,W), got {tuple(pixel_values.shape)}')
        batch_size, num_frames = pixel_values.shape[:2]
        if num_frames < 1:
            raise ValueError('video pixel_values must contain at least one frame')

        if image_attention_mask is not None:
            if tuple(image_attention_mask.shape) != (batch_size, num_frames):
                raise ValueError(f'image_attention_mask must have shape {(batch_size, num_frames)}, got {tuple(image_attention_mask.shape)}')
            image_attention_mask = image_attention_mask.to(device=pixel_values.device, dtype=torch.bool)

        # With a single frame, preserve the exact old SigLIP code path.
        if num_frames == 1:
            return self.forward(
                pixel_values[:, 0],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        # Memory-efficient training path: history frames are processed under
        # no_grad and only feed the current-frame computation as detached K/V
        # context at temporal-attention layers. The shared SigLIP encoder still
        # receives gradient updates from the current frame's path.
        if self.training and self.detach_history_frames and not output_attentions and not output_hidden_states:
            return self._forward_video_history_detached(
                pixel_values,
                image_attention_mask=image_attention_mask,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        flat_pixel_values = pixel_values.reshape(batch_size * num_frames, *pixel_values.shape[2:])
        hidden_states = self.embeddings(flat_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        num_patches, embed_dim = hidden_states.shape[1:]
        hidden_states = hidden_states.reshape(batch_size, num_frames, num_patches, embed_dim)
        hidden_states = hidden_states + self._temporal_position_embedding(num_frames, embed_dim, hidden_states)
        hidden_states = hidden_states.to(dtype=torch.bfloat16)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for layer_idx, encoder_layer in enumerate(self.encoder.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states[:, -1],)

                if layer_idx % self.temporal_attention_interval == 0:
                    hidden_states, spatial_attn_weights = self._apply_temporal_spatial_block(
                        encoder_layer,
                        hidden_states,
                        image_attention_mask,
                        output_attentions,
                    )
                else:
                    hidden_states, spatial_attn_weights = self._apply_spatial_block(encoder_layer, hidden_states, output_attentions)
                if output_attentions:
                    all_attentions = all_attentions + (spatial_attn_weights,)

            current_hidden_state = hidden_states[:, -1]
            if output_hidden_states:
                encoder_states = encoder_states + (current_hidden_state,)

            last_hidden_state = self.post_layernorm(current_hidden_state)
            pooler_output = self.head(last_hidden_state) if self.use_head else None

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=encoder_states,
                attentions=all_attentions,
            )

    def _forward_video_history_detached(
        self,
        pixel_values: torch.Tensor,
        image_attention_mask: torch.Tensor | None,
        interpolate_pos_encoding: bool,
    ) -> BaseModelOutputWithPooling:
        """Two-pass video forward that keeps gradients only on the current
        frame.

        Pass 1 runs the (T-1) history frames through the full SigLIP stack under
        ``torch.no_grad`` and caches their hidden states at the input of every
        temporal-attention layer. Pass 2 runs the current frame with grad and,
        at every temporal-attention layer, concatenates the cached (detached)
        history states so the temporal block still mixes the full T frames.
        Backward only retraces 1 frame's worth of activations through SigLIP,
        which removes the T-fold growth in the activation-checkpointing
        boundaries.
        """
        batch_size, num_frames = pixel_values.shape[:2]

        # ------- Pass 1: history under no_grad -------
        history_pixels = pixel_values[:, :-1]
        history_mask = image_attention_mask[:, :-1] if image_attention_mask is not None else None
        cached_history_states: list[torch.Tensor] = []

        with torch.no_grad():
            flat_history = history_pixels.reshape(batch_size * (num_frames - 1), *history_pixels.shape[2:])
            history_states = self.embeddings(flat_history, interpolate_pos_encoding=interpolate_pos_encoding)
            num_patches, embed_dim = history_states.shape[1:]
            history_states = history_states.reshape(batch_size, num_frames - 1, num_patches, embed_dim)
            # Use the original T-frame pos table, sliced for history positions
            # so each history frame keeps its absolute offset relative to "now".
            full_pos_emb = self._temporal_position_embedding(num_frames, embed_dim, history_states)
            history_states = history_states + full_pos_emb[:, :-1]
            history_states = history_states.to(dtype=torch.bfloat16)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for layer_idx, encoder_layer in enumerate(self.encoder.layers):
                    if layer_idx % self.temporal_attention_interval == 0:
                        cached_history_states.append(history_states)
                        history_states, _ = self._apply_temporal_spatial_block(
                            encoder_layer,
                            history_states,
                            history_mask,
                            output_attentions=False,
                            apply_missing_camera_fallback=False,
                        )
                    else:
                        history_states, _ = self._apply_spatial_block(encoder_layer, history_states, output_attentions=False)
            del history_states

        # ------- Pass 2: current frame with grad -------
        current_pixels = pixel_values[:, -1]
        hidden_states = self.embeddings(current_pixels, interpolate_pos_encoding=interpolate_pos_encoding)
        num_patches, embed_dim = hidden_states.shape[1:]
        hidden_states = hidden_states.reshape(batch_size, 1, num_patches, embed_dim)
        # Current frame's temporal pos embedding is zero by construction
        # (pos_emb is normalized to subtract the last frame), so no add needed.
        hidden_states = hidden_states.to(dtype=torch.bfloat16)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            cache_iter = iter(cached_history_states)
            for layer_idx, encoder_layer in enumerate(self.encoder.layers):
                if layer_idx % self.temporal_attention_interval == 0:
                    history_h = next(cache_iter)
                    full_h = torch.cat([history_h, hidden_states], dim=1)
                    full_h, _ = self._apply_temporal_spatial_block(encoder_layer, full_h, image_attention_mask, output_attentions=False)
                    hidden_states = full_h[:, -1:]
                else:
                    hidden_states, _ = self._apply_spatial_block(encoder_layer, hidden_states, output_attentions=False)

            current_hidden_state = hidden_states[:, 0]
            last_hidden_state = self.post_layernorm(current_hidden_state)
            pooler_output = self.head(last_hidden_state) if self.use_head else None

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=None,
                attentions=None,
            )

    @staticmethod
    def _temporal_position_embedding(num_frames: int, embed_dim: int, reference: torch.Tensor) -> torch.Tensor:
        if embed_dim % 2 != 0:
            raise ValueError(f'temporal position embedding requires an even hidden size, got {embed_dim}')
        dtype = torch.float32
        timesteps = torch.arange(-(num_frames - 1), 1, device=reference.device, dtype=dtype)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=reference.device, dtype=dtype) * (-math.log(10000.0) / embed_dim))
        pos_emb = torch.zeros(num_frames, embed_dim, device=reference.device, dtype=dtype)
        pos_emb[:, 0::2] = torch.sin(timesteps[:, None] * div_term[None, :])
        pos_emb[:, 1::2] = torch.cos(timesteps[:, None] * div_term[None, :])
        pos_emb = pos_emb - pos_emb[-1:]
        return pos_emb.to(dtype=reference.dtype)[None, :, None, :]

    def _apply_temporal_spatial_block(
        self,
        encoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        image_attention_mask: torch.Tensor | None,
        output_attentions: bool,
        apply_missing_camera_fallback: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, num_frames, num_patches, embed_dim = hidden_states.shape
        temporal_attention_mask = self._build_temporal_attention_mask(
            batch_size=batch_size,
            num_frames=num_frames,
            num_patches=num_patches,
            frame_mask=image_attention_mask,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            apply_missing_camera_fallback=apply_missing_camera_fallback,
        )
        layer_outputs = encoder_layer(
            hidden_states,
            temporal_attention_mask,
            output_attentions=output_attentions,
            temporal_attention=True,
        )
        spatial_attn_weights = layer_outputs[1] if output_attentions else None
        return layer_outputs[0], spatial_attn_weights

    @staticmethod
    def _build_temporal_attention_mask(
        batch_size: int,
        num_frames: int,
        num_patches: int,
        frame_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
        apply_missing_camera_fallback: bool = True,
    ) -> torch.Tensor:
        # `apply_missing_camera_fallback` reads `frame_mask[:, -1]` as the current
        # frame's validity. Callers that pass a frame_mask whose last column is
        # not the current frame (e.g. a history-only slice) must set this to
        # False, otherwise the fallback wrongly treats a missing last-history
        # frame as "current frame missing" and unmasks the whole history.
        allowed = torch.ones(num_frames, num_frames, dtype=torch.bool, device=device).tril()
        allowed = allowed[None, None, :, :]
        if frame_mask is not None:
            frame_mask = frame_mask.to(device=device, dtype=torch.bool).clone()
            if apply_missing_camera_fallback:
                missing_camera = ~frame_mask[:, -1]
                if missing_camera.any():
                    frame_mask[missing_camera] = True
            key_mask = frame_mask[:, None, None, :]
            allowed = allowed & key_mask
            allowed = allowed | torch.eye(num_frames, dtype=torch.bool, device=device)[None, None, :, :]
        allowed = allowed.expand(batch_size, num_patches, num_frames, num_frames).reshape(batch_size * num_patches, 1, num_frames, num_frames)
        mask = torch.zeros_like(allowed, dtype=dtype)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        return mask

    @staticmethod
    def _apply_spatial_block(
        encoder_layer: nn.Module,
        hidden_states: torch.Tensor,
        output_attentions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, num_frames, num_patches, embed_dim = hidden_states.shape
        spatial_states = hidden_states.reshape(batch_size * num_frames, num_patches, embed_dim)
        layer_outputs = encoder_layer(
            spatial_states,
            attention_mask=None,
            output_attentions=output_attentions,
        )
        spatial_states = layer_outputs[0].reshape(batch_size, num_frames, num_patches, embed_dim)
        attn_weights = layer_outputs[1] if output_attentions else None
        return spatial_states, attn_weights


# Copied from transformers.models.paligemma.modeling_paligemma.PaliGemmaMultiModalProjector
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, vision_hidden_size: int = 1152, projection_dim: int = 2304):
        super().__init__()
        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project vision features to the transformer hidden size."""
        hidden_states = self.linear(image_features)
        return hidden_states


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to x [B, L, H, D]."""
    with nvtx_range('rotary'):
        dtype = x.dtype
        x = x.to(torch.float32)

        d_half = x.shape[-1] // 2
        x1, x2 = x.split(d_half, dim=-1)

        # Rotate: out1 = x1 * cos - x2 * sin, out2 = x2 * cos + x1 * sin
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin

        return res.to(dtype)


class RoPEEmbedding(nn.Module):
    """Precomputed RoPE embeddings for improved performance.

    This implementation precomputes sin/cos values for a maximum sequence length, avoiding redundant trigonometric calculations during forward passes.
    """

    def __init__(self, dim: int, max_wavelength: int = 10_000, max_seq_len: int = 8192, attention_scaling: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_wavelength = max_wavelength
        self.max_seq_len = max_seq_len
        self.attention_scaling = attention_scaling

        # Precompute frequency exponents and inverse frequencies
        d_half = dim // 2
        freq_exponents = (2.0 / dim) * torch.arange(d_half, dtype=torch.float32)
        inv_freq = 1.0 / (max_wavelength**freq_exponents)

        # Precompute sin and cos for all positions up to max_seq_len
        # Shape: [max_seq_len, d_half]
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_seq_len, d_half]

        # Precompute sin and cos values
        # We expand to [max_seq_len, 1, d_half] for broadcasting in forward
        cos_cached = torch.cos(freqs).unsqueeze(1)  # [max_seq_len, 1, d_half]
        sin_cached = torch.sin(freqs).unsqueeze(1)  # [max_seq_len, 1, d_half]

        # Register as buffers so they automatically move to the correct device with the model
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self, positions: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns RoPE embeddings (cos, sin) for the given positions.

        Args:
            positions: Position indices of shape [B, L]

        Returns:
            Tuple of (cos, sin) tensors, each of shape [B, L, 1, D/2]
        """
        cos = self.cos_cached[positions]  # [B, L, 1, d_half]
        sin = self.sin_cached[positions]  # [B, L, 1, d_half]
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos, sin


class Gemma2AttentionWithExpert(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        # PaliGemma2 params
        paligemma_hidden_size: int = 2304,
        paligemma_num_attention_heads: int = 8,
        paligemma_num_key_value_heads: int = 4,
        paligemma_head_dim: int = 256,
        paligemma_attention_bias: bool = False,
        paligemma_attn_logit_softcapping: float | None = None,
        paligemma_query_pre_attn_scalar: float = 256.0,
        # Expert params
        expert_hidden_size: int = 1024,
        expert_num_attention_heads: int = 8,
        expert_num_key_value_heads: int = 4,
        expert_head_dim: int = 256,
        expert_attention_bias: bool = False,
        expert_query_pre_attn_scalar: float = 256.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_hidden_size, paligemma_num_attention_heads * paligemma_head_dim, bias=paligemma_attention_bias),
                nn.Linear(expert_hidden_size, expert_num_attention_heads * expert_head_dim, bias=expert_attention_bias),
            ]
        )
        self.k_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_hidden_size, paligemma_num_key_value_heads * paligemma_head_dim, bias=paligemma_attention_bias),
                nn.Linear(expert_hidden_size, expert_num_key_value_heads * expert_head_dim, bias=expert_attention_bias),
            ]
        )
        self.v_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_hidden_size, paligemma_num_key_value_heads * paligemma_head_dim, bias=paligemma_attention_bias),
                nn.Linear(expert_hidden_size, expert_num_key_value_heads * expert_head_dim, bias=expert_attention_bias),
            ]
        )
        self.o_proj = nn.ModuleList(
            [
                nn.Linear(paligemma_num_attention_heads * paligemma_head_dim, paligemma_hidden_size, bias=paligemma_attention_bias),
                nn.Linear(expert_num_attention_heads * expert_head_dim, expert_hidden_size, bias=expert_attention_bias),
            ]
        )

        self.paligemma_num_attention_heads = paligemma_num_attention_heads
        self.paligemma_num_key_value_heads = paligemma_num_key_value_heads
        self.paligemma_head_dim = paligemma_head_dim
        self.paligemma_attn_logit_softcapping = paligemma_attn_logit_softcapping
        self.paligemma_scaling = paligemma_query_pre_attn_scalar**-0.5

        self.expert_num_attention_heads = expert_num_attention_heads
        self.expert_num_key_value_heads = expert_num_key_value_heads
        self.expert_head_dim = expert_head_dim
        self.expert_scaling = expert_query_pre_attn_scalar**-0.5

        assert self.paligemma_scaling == self.expert_scaling, f'paligemma_scaling: {self.paligemma_scaling}, expert_scaling: {self.expert_scaling}'
        assert paligemma_head_dim == expert_head_dim, f'paligemma_head_dim: {paligemma_head_dim}, expert_head_dim: {expert_head_dim}'
        assert (
            paligemma_num_attention_heads == expert_num_attention_heads
        ), f'paligemma_num_attention_heads: {paligemma_num_attention_heads}, expert_num_attention_heads: {expert_num_attention_heads}'
        assert (
            paligemma_num_key_value_heads == expert_num_key_value_heads
        ), f'paligemma_num_key_value_heads: {paligemma_num_key_value_heads}, expert_num_key_value_heads: {expert_num_key_value_heads}'

    def forward(
        self,
        inputs_embeds: list[torch.Tensor | None],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: dict | None = None,
        fill_kv_cache: bool = False,
        auto_regression_inference_mode: bool = False,
    ) -> tuple[list[torch.Tensor | None], torch.Tensor | None, torch.Tensor | None]:
        """Multi-source attention over PaliGemma and Expert streams.

        Args:
            inputs_embeds: [paligemma_embeds, expert_embeds]. Each is (B, L, D) or None.
            position_embeddings: (cos, sin) tuple for RoPE.
            attention_mask: (B, L, L) attention mask.
            use_cache: Whether to use KV cache.
            past_key_values: Optional cache dict per layer.
            fill_kv_cache: If True, fill cache; otherwise, append to it.
            auto_regression_inference_mode: If True, operates in auto-regressive mode.

        Returns:
            A tuple containing:
            - A list of optional output embeddings for each stream.
            - The present key states for caching.
            - The present value states for caching.
        """
        query_states_list = []
        key_states_list = []
        value_states_list = []

        nvtx_push('attn_qkv')
        if inputs_embeds[0] is not None:
            # PaliGemma2
            hidden_states = inputs_embeds[0]
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, self.paligemma_num_attention_heads, self.paligemma_head_dim)
            query_states_list.append(self.q_proj[0](hidden_states).view(hidden_shape))

            hidden_shape_kv = (*input_shape, self.paligemma_num_key_value_heads, self.paligemma_head_dim)
            key_states_list.append(self.k_proj[0](hidden_states).view(hidden_shape_kv))
            value_states_list.append(self.v_proj[0](hidden_states).view(hidden_shape_kv))

        if inputs_embeds[1] is not None:
            # Expert
            hidden_states = inputs_embeds[1]
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, self.expert_num_attention_heads, self.expert_head_dim)
            query_states_list.append(self.q_proj[1](hidden_states).view(hidden_shape))

            hidden_shape_kv = (*input_shape, self.expert_num_key_value_heads, self.expert_head_dim)
            key_states_list.append(self.k_proj[1](hidden_states).view(hidden_shape_kv))
            value_states_list.append(self.v_proj[1](hidden_states).view(hidden_shape_kv))

        query_states = torch.cat(query_states_list, dim=1)
        key_states = torch.cat(key_states_list, dim=1)
        value_states = torch.cat(value_states_list, dim=1)
        nvtx_pop()  # attn_qkv

        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb(query_states, cos, sin)
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        # Prepare present key/value states for cache management outside the layer.
        present_key_states = None
        present_value_states = None

        if use_cache:
            if fill_kv_cache:
                # Prefill stage: return this step's KV to the caller; do not mutate external dict here.
                present_key_states = key_states
                present_value_states = value_states
            else:
                # Decode/suffix stage: read existing KV and concatenate for attention computation.
                key_states = torch.cat([past_key_values[self.layer_idx]['key_states'], key_states], dim=1)
                value_states = torch.cat([past_key_values[self.layer_idx]['value_states'], value_states], dim=1)
                if auto_regression_inference_mode:
                    # Return concatenated KV so the caller can persist it.
                    present_key_states = key_states
                    present_value_states = value_states

        # For simplicity, we assume both branches have same head configs.
        # This can be made more generic if needed.
        num_att_heads = self.paligemma_num_attention_heads
        num_key_value_heads = self.paligemma_num_key_value_heads
        head_dim = self.paligemma_head_dim
        batch_size = query_states.shape[0]

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Keep GQA math equivalent while avoiding an Inductor SDPA backward graph
        # that can produce NaN gradients for the proprio-memory prefix shape.
        if num_key_value_heads != num_att_heads:
            key_states = torch.repeat_interleave(key_states, num_att_heads // num_key_value_heads, dim=1)
            value_states = torch.repeat_interleave(value_states, num_att_heads // num_key_value_heads, dim=1)

        nvtx_push('attn_sdpa')
        att_output = nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=self.paligemma_scaling,
            attn_mask=attention_mask[:, None, :, :] if attention_mask is not None else None,
            is_causal=False,
        )
        nvtx_pop()  # attn_sdpa

        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_att_heads * head_dim)

        nvtx_push('attn_o_proj')
        outputs_embeds = []
        start = 0
        if inputs_embeds[0] is not None:
            hidden_states = inputs_embeds[0]
            end = start + hidden_states.shape[1]
            if att_output.dtype != self.o_proj[0].weight.dtype:
                att_output_i = att_output[:, start:end].to(self.o_proj[0].weight.dtype)
            else:
                att_output_i = att_output[:, start:end]
            out_emb = self.o_proj[0](att_output_i)
            outputs_embeds.append(out_emb)
            start = end
        else:
            outputs_embeds.append(None)

        if inputs_embeds[1] is not None:
            hidden_states = inputs_embeds[1]
            end = start + hidden_states.shape[1]
            if att_output.dtype != self.o_proj[1].weight.dtype:
                att_output_i = att_output[:, start:end].to(self.o_proj[1].weight.dtype)
            else:
                att_output_i = att_output[:, start:end]
            out_emb = self.o_proj[1](att_output_i)
            outputs_embeds.append(out_emb)
        else:
            outputs_embeds.append(None)

        nvtx_pop()  # attn_o_proj
        return outputs_embeds, present_key_states, present_value_states


class Gemma2MLP(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 4096, hidden_act: str = 'gelu_pytorch_tanh'):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated MLP block used in both streams."""
        with nvtx_range('mlp_proj'):
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma2DecoderLayerWithExpert(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        # PaliGemma2 params
        paligemma_hidden_size: int = 2304,
        paligemma_num_attention_heads: int = 8,
        paligemma_num_key_value_heads: int = 4,
        paligemma_head_dim: int = 256,
        paligemma_attention_bias: bool = False,
        paligemma_attn_logit_softcapping: float | None = None,
        paligemma_query_pre_attn_scalar: float = 256.0,
        paligemma_intermediate_size: int = 9216,
        paligemma_hidden_act: str = 'gelu_pytorch_tanh',
        paligemma_rms_norm_eps: float = 1e-6,
        # Expert params
        expert_hidden_size: int = 1024,
        expert_num_attention_heads: int = 8,
        expert_num_key_value_heads: int = 4,
        expert_head_dim: int = 256,
        expert_attention_bias: bool = False,
        expert_query_pre_attn_scalar: float = 256.0,
        expert_intermediate_size: int = 2048,
        expert_hidden_act: str = 'gelu_pytorch_tanh',
        expert_rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = Gemma2AttentionWithExpert(
            layer_idx,
            paligemma_hidden_size,
            paligemma_num_attention_heads,
            paligemma_num_key_value_heads,
            paligemma_head_dim,
            paligemma_attention_bias,
            paligemma_attn_logit_softcapping,
            paligemma_query_pre_attn_scalar,
            expert_hidden_size,
            expert_num_attention_heads,
            expert_num_key_value_heads,
            expert_head_dim,
            expert_attention_bias,
            expert_query_pre_attn_scalar,
        )

        self.mlps = nn.ModuleList(
            [
                Gemma2MLP(paligemma_hidden_size, paligemma_intermediate_size, paligemma_hidden_act),
                Gemma2MLP(expert_hidden_size, expert_intermediate_size, expert_hidden_act),
            ]
        )

        self.input_layernorms = nn.ModuleList(
            [
                Gemma2RMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                Gemma2RMSNorm(expert_hidden_size, eps=expert_rms_norm_eps, use_ada_rms_norm=True),
            ]
        )
        self.post_attention_layernorms = nn.ModuleList(
            [
                Gemma2RMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                Gemma2RMSNorm(expert_hidden_size, eps=expert_rms_norm_eps),
            ]
        )
        self.pre_feedforward_layernorms = nn.ModuleList(
            [
                Gemma2RMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                Gemma2RMSNorm(expert_hidden_size, eps=expert_rms_norm_eps, use_ada_rms_norm=True),
            ]
        )
        self.post_feedforward_layernorms = nn.ModuleList(
            [
                Gemma2RMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                Gemma2RMSNorm(expert_hidden_size, eps=expert_rms_norm_eps),
            ]
        )

    def gated_residual(self, x, y, gate):
        if x is None or y is None:
            return None
        if gate is None:
            return x + y
        return x + y * gate

    def forward(
        self,
        inputs_embeds: list[torch.Tensor | None],
        adarms_cond: list[torch.Tensor | None],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: dict | None = None,
        fill_kv_cache: bool = False,
        auto_regression_inference_mode: bool = False,
    ) -> tuple[list[torch.Tensor | None], torch.Tensor | None, torch.Tensor | None]:
        """Decoder layer with dual-stream attention and optional AdaRMS
        modulation.

        Args:
            inputs_embeds: [paligemma, expert] embeds.
            adarms_cond: Optional conditioning vectors for AdaRMS.
            position_embeddings: (cos, sin) tuple for RoPE.
            attention_mask: (B, L, L) attention mask.
            use_cache: Whether to use KV cache.
            past_key_values: Optional cache dict.
            fill_kv_cache: Whether to fill or reuse KV cache.
            auto_regression_inference_mode: Whether to use auto-regression inference mode.

        Returns:
            A tuple containing:
            - A list of optional updated hidden states per stream.
            - The present key states for caching.
            - The present value states for caching.
        """
        residuals_pre_attn = list(inputs_embeds)
        normed_embeds = []
        attn_gates = []

        nvtx_push('attn')
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                if adarms_cond[i] is not None:
                    normed_h, attn_gate = self.input_layernorms[i](hidden_states, adarms_cond[i])
                    normed_embeds.append(normed_h)
                    attn_gates.append(attn_gate)
                else:
                    normed_h = self.input_layernorms[i](hidden_states)
                    normed_embeds.append(normed_h)
                    attn_gates.append(None)
            else:
                normed_embeds.append(None)
                attn_gates.append(None)

        attn_outputs, present_key_states, present_value_states = self.self_attn(
            normed_embeds, position_embeddings, attention_mask, use_cache, past_key_values, fill_kv_cache, auto_regression_inference_mode
        )

        hs_after_attn_list = []
        for i, (residual, attn_output, attn_gate) in enumerate(zip(residuals_pre_attn, attn_outputs, attn_gates)):
            if residual is not None:
                normed_attn_out = self.post_attention_layernorms[i](attn_output)
                hs_after_attn = self.gated_residual(residual, normed_attn_out, attn_gate)
                hs_after_attn_list.append(hs_after_attn)
            else:
                hs_after_attn_list.append(None)
        nvtx_pop()  # attn

        nvtx_push('mlp')
        outputs = []
        for i, hidden_states in enumerate(hs_after_attn_list):
            if hidden_states is not None:
                residual_pre_mlp = hidden_states

                if adarms_cond[i] is not None:
                    normed_h, mlp_gate = self.pre_feedforward_layernorms[i](hidden_states, adarms_cond[i])
                else:
                    normed_h = self.pre_feedforward_layernorms[i](hidden_states)
                    mlp_gate = None

                mlp_out = self.mlps[i](normed_h)
                normed_mlp_out = self.post_feedforward_layernorms[i](mlp_out)

                hs_after_mlp = self.gated_residual(residual_pre_mlp, normed_mlp_out, mlp_gate)
                outputs.append(hs_after_mlp)
            else:
                outputs.append(None)
        nvtx_pop()  # mlp

        return outputs, present_key_states, present_value_states


class PaliGemma2WithExpertModel(nn.Module):
    """PyTorch implementation of the PaliGemma2-3B model integrated with an
    expert mixture-of-experts architecture."""

    def __init__(
        self,
        vision_in_channels: int = 3,
        enable_next_token_prediction: bool = True,
        temporal_attention_interval: int = 4,
        detach_history_frames: bool = True,
        # Paligemma2 params
        paligemma_vocab_size: int = 257216,
        paligemma_pad_token_id: int = 0,
        paligemma_num_hidden_layers: int = 26,
        paligemma_hidden_size: int = 2304,
        paligemma_num_attention_heads: int = 8,
        paligemma_num_key_value_heads: int = 4,
        paligemma_head_dim: int = 256,
        paligemma_attention_bias: bool = False,
        paligemma_attn_logit_softcapping: float | None = None,
        paligemma_query_pre_attn_scalar: float = 256.0,
        paligemma_intermediate_size: int = 9216,
        paligemma_hidden_act: str = 'gelu_pytorch_tanh',
        paligemma_rms_norm_eps: float = 1e-6,
        paligemma_final_logit_softcapping: float | None = None,
        # Expert params
        expert_hidden_size: int = 1024,
        expert_num_attention_heads: int = 8,
        expert_num_key_value_heads: int = 4,
        expert_head_dim: int = 256,
        expert_attention_bias: bool = False,
        expert_query_pre_attn_scalar: float = 256.0,
        expert_intermediate_size: int = 2048,
        expert_hidden_act: str = 'gelu_pytorch_tanh',
        expert_rms_norm_eps: float = 1e-6,
        # RoPE params
        rope_max_wavelength: int = 10_000,
        rope_max_seq_len: int = 8192,
    ):
        super().__init__()
        # Note: PaliGemma2-3B doesn't use attn_logit_softcapping. So the default value in the codebase is None.
        self.paligemma_final_logit_softcapping = paligemma_final_logit_softcapping

        siglip_vision_config = get_transformers_siglip_vision_config(vision_in_channels)

        # Vision and projection
        self.vision_tower = SiglipVisionTransformer(
            siglip_vision_config,
            temporal_attention_interval=temporal_attention_interval,
            detach_history_frames=detach_history_frames,
        )
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=siglip_vision_config.hidden_size, projection_dim=paligemma_hidden_size
        )
        self.paligemma_hidden_size = paligemma_hidden_size

        # RoPE
        self.rope_embedding = RoPEEmbedding(dim=paligemma_head_dim, max_wavelength=rope_max_wavelength, max_seq_len=rope_max_seq_len)

        # Language embed
        self.embed_tokens = nn.Embedding(paligemma_vocab_size, paligemma_hidden_size, paligemma_pad_token_id)

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Gemma2DecoderLayerWithExpert(
                    layer_idx=i,
                    paligemma_hidden_size=paligemma_hidden_size,
                    paligemma_num_attention_heads=paligemma_num_attention_heads,
                    paligemma_num_key_value_heads=paligemma_num_key_value_heads,
                    paligemma_head_dim=paligemma_head_dim,
                    paligemma_attention_bias=paligemma_attention_bias,
                    paligemma_attn_logit_softcapping=paligemma_attn_logit_softcapping,
                    paligemma_query_pre_attn_scalar=paligemma_query_pre_attn_scalar,
                    paligemma_intermediate_size=paligemma_intermediate_size,
                    paligemma_hidden_act=paligemma_hidden_act,
                    paligemma_rms_norm_eps=paligemma_rms_norm_eps,
                    expert_hidden_size=expert_hidden_size,
                    expert_num_attention_heads=expert_num_attention_heads,
                    expert_num_key_value_heads=expert_num_key_value_heads,
                    expert_head_dim=expert_head_dim,
                    expert_attention_bias=expert_attention_bias,
                    expert_query_pre_attn_scalar=expert_query_pre_attn_scalar,
                    expert_intermediate_size=expert_intermediate_size,
                    expert_hidden_act=expert_hidden_act,
                    expert_rms_norm_eps=expert_rms_norm_eps,
                )
                for i in range(paligemma_num_hidden_layers)
            ]
        )

        # Final norms
        self.norms = nn.ModuleList(
            [
                Gemma2RMSNorm(paligemma_hidden_size, eps=paligemma_rms_norm_eps),
                Gemma2RMSNorm(expert_hidden_size, eps=expert_rms_norm_eps, use_ada_rms_norm=True),
            ]
        )

        if enable_next_token_prediction:
            self.lm_head = nn.Linear(paligemma_hidden_size, paligemma_vocab_size, bias=False)
            self.tie_weights()

    def language_out_proj(self, pre_logit: torch.Tensor):
        logits = self.lm_head(pre_logit)
        if self.paligemma_final_logit_softcapping is not None:
            logits = logits / self.paligemma_final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.paligemma_final_logit_softcapping
        return logits

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings
        # Re-tie after replacing embeddings
        self.tie_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_linear: nn.Linear):
        self.lm_head = new_linear
        # Re-tie in case caller expects tied weights
        self.tie_weights()

    def tie_weights(self):
        """Tie output head weight to input embedding weight (HF-style).

        This mirrors the common HF behavior: if shapes align, share the same storage.
        """
        if not hasattr(self, 'lm_head') or not hasattr(self, 'embed_tokens'):
            return
        output_module = self.lm_head
        input_emb = self.embed_tokens

        # Validate shapes before tying
        if isinstance(output_module, nn.Linear) and isinstance(input_emb, nn.Embedding) and output_module.weight.shape == input_emb.weight.shape:
            output_module.weight = input_emb.weight
        else:
            # Fall back: clone weights when direct tie is impossible
            # (e.g., if dimensions changed temporarily). This keeps training workable.
            with torch.no_grad():
                if isinstance(output_module, nn.Linear) and isinstance(input_emb, nn.Embedding):
                    if output_module.weight.shape[::-1] == input_emb.weight.shape[::-1]:
                        output_module.weight.copy_(input_emb.weight)
                # Otherwise do nothing; caller should ensure compatible shapes.

    def embed_image(self, image: torch.Tensor, image_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode images with SigLIP and project to hidden size."""
        image_outputs = self.vision_tower(image, image_attention_mask=image_attention_mask)
        selected_image_feature = image_outputs.last_hidden_state
        # Ensure dtype matches projector weights for mixed precision training
        selected_image_feature = selected_image_feature.to(self.multi_modal_projector.linear.weight.dtype)
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token ids into continuous vectors."""
        return self.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        adarms_cond: list[torch.FloatTensor] | None = None,
        auto_regression_inference_mode: bool = False,
    ) -> tuple[list[torch.Tensor | None], dict]:
        """Run the stacked dual-stream decoder with optional caching and
        AdaRMS.

        Args:
            attention_mask: (B, L, L) attention mask for both streams.
            position_ids: (B, L) RoPE positions.
            past_key_values: Optional KV cache dict to reuse.
            inputs_embeds: [paligemma_embeds, expert_embeds].
            use_cache: Whether to use KV cache.
            fill_kv_cache: If True, populate cache from inputs.
            adarms_cond: Optional per-stream modulation vectors for AdaRMS.
            auto_regression_inference_mode: Whether to use auto-regression inference mode.

        Returns:
            (outputs_embeds, past_key_values): outputs per stream and the KV cache.
        """
        if inputs_embeds is None:
            inputs_embeds = [None, None]
        if adarms_cond is None:
            adarms_cond = [None, None]
        inputs_embeds = [input_embed.to(dtype=torch.bfloat16) if input_embed is not None else None for input_embed in inputs_embeds]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if use_cache and past_key_values is None:
                past_key_values = {}

            read_kv_cache = past_key_values
            if use_cache and (fill_kv_cache or auto_regression_inference_mode):
                updated_kv_cache = {}
            else:
                updated_kv_cache = past_key_values

            with nvtx_range('rope'):
                position_embeddings = self.rope_embedding(position_ids)

            hidden_states_list = inputs_embeds
            for layer_idx, layer in enumerate(self.layers):
                nvtx_push(f'gemma_layer_{layer_idx}')
                layer_outputs, present_key_states, present_value_states = layer(
                    hidden_states_list,
                    adarms_cond=adarms_cond,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=read_kv_cache,
                    fill_kv_cache=fill_kv_cache,
                    auto_regression_inference_mode=auto_regression_inference_mode,
                )
                hidden_states_list = layer_outputs
                if use_cache:
                    if fill_kv_cache and present_key_states is not None:
                        updated_kv_cache[layer_idx] = {
                            'key_states': present_key_states,
                            'value_states': present_value_states,
                        }
                    elif auto_regression_inference_mode and present_key_states is not None:
                        updated_kv_cache[layer_idx] = {
                            'key_states': present_key_states,
                            'value_states': present_value_states,
                        }
                nvtx_pop()  # gemma_layer

            outputs_embeds = []
            with nvtx_range('final_norm'):
                for i, hidden_states in enumerate(hidden_states_list):
                    if hidden_states is not None:
                        if adarms_cond[i] is not None:
                            out_emb, _ = self.norms[i](hidden_states, adarms_cond[i])
                        else:
                            out_emb = self.norms[i](hidden_states)
                        outputs_embeds.append(out_emb)
                    else:
                        outputs_embeds.append(None)

            return outputs_embeds, updated_kv_cache
