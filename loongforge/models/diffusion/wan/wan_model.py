# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""wan model"""

import torch
import torch.nn as nn
from torch import Tensor

import math
from typing import Tuple, Dict, Literal, Optional, Any
from einops import rearrange
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.spec_utils import ModuleSpec
from .wan_transformer_block import WanTransformerBlock

from loongforge.utils import get_args
from .communications import (
    split_forward_gather_backward,
    gather_forward_split_backward,
)
from megatron.core.parallel_state import (
    get_context_parallel_group,
)
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from torch.amp import autocast


# ---------------------------------------------------------------------------
# Per-sample boundary CP split helpers
# ---------------------------------------------------------------------------

def _build_thd_reorder_indices(seq_len_padded, cp_size):
    """Build indices that restore per-sample global THD order after CP gather.

    Per-sample boundary CP split gathers tokens in chunk-major order:
    ``[sample0_chunk0, sample1_chunk0, ..., sample0_chunk1, ...]``.
    Packed attention and loss expect sample-major order:
    ``[sample0_full, sample1_full, ...]``.
    """
    sample_lengths = [seq_len_padded[index].item() for index in range(seq_len_padded.shape[0])]
    total_length = sum(sample_lengths)
    chunk_lengths = [sample_length // cp_size for sample_length in sample_lengths]

    sample_offsets = [0]
    for sample_length in sample_lengths:
        sample_offsets.append(sample_offsets[-1] + sample_length)

    gathered_to_global = torch.empty(total_length, dtype=torch.long)
    gathered_offset = 0
    for cp_chunk_index in range(cp_size):
        for sample_index, chunk_length in enumerate(chunk_lengths):
            global_offset = sample_offsets[sample_index] + cp_chunk_index * chunk_length
            for token_offset in range(chunk_length):
                gathered_to_global[gathered_offset + token_offset] = global_offset + token_offset
            gathered_offset += chunk_length

    global_to_gathered = torch.empty(total_length, dtype=torch.long)
    for global_index in range(total_length):
        global_to_gathered[gathered_to_global[global_index].item()] = global_index

    return gathered_to_global, global_to_gathered


class _THDSplitForCP(torch.autograd.Function):
    """Autograd function for per-sample boundary CP split with tight packing.

    Forward: select this CP rank's per-sample chunks.
    Backward: scatter gradients back to original positions.
    """

    @staticmethod
    def forward(ctx, x, cu_seqlens, seq_len_padded, cp_size, cp_rank):
        """Select this CP rank THD chunks for forward propagation."""
        num_sequences = cu_seqlens.shape[0] - 1
        indices = []
        total_actual = cu_seqlens[-1].item()
        for sequence_index in range(num_sequences):
            actual_start = cu_seqlens[sequence_index].item()
            actual_len = cu_seqlens[sequence_index + 1].item() - actual_start
            padded_len = seq_len_padded[sequence_index].item()
            chunk_len = padded_len // cp_size
            local_offset = cp_rank * chunk_len
            actual_take = min(chunk_len, max(0, actual_len - local_offset))
            for token_offset in range(actual_take):
                indices.append(actual_start + local_offset + token_offset)
            for pad_offset in range(chunk_len - actual_take):
                indices.append(total_actual + pad_offset)
        indices_t = torch.tensor(indices, dtype=torch.long, device=x.device)
        ctx.save_for_backward(indices_t)
        ctx.cp_size = cp_size
        ctx.total_len = x.shape[0]
        return x.index_select(0, indices_t)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter local THD chunk gradients back to original sequence positions."""
        indices_t, = ctx.saved_tensors
        grad_input = torch.zeros(
            ctx.total_len, *grad_output.shape[1:],
            dtype=grad_output.dtype, device=grad_output.device,
        )
        grad_input.index_add_(0, indices_t, grad_output)
        return grad_input, None, None, None, None


def thd_split_for_cp(x, cu_seqlens, seq_len_padded, cp_size, cp_rank):
    """Per-sample boundary CP split with correct autograd."""
    return _THDSplitForCP.apply(x, cu_seqlens, seq_len_padded, cp_size, cp_rank)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(
                dim // 2
            ),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(
        self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float
    ):
        super().__init__()
        self.hidden_size = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        shift, scale = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


from .wan_config import WanConfig
class WanModel(VisionModule):
    """Wan Transformer language model"""

    def __init__(
        self,
        config: WanConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        require_vae_embedding: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ):
        super().__init__(config=config)
        self.require_clip_embedding = config.require_clip_embedding
        self.require_vae_embedding = require_vae_embedding
        self.args = get_args()
        self.pre_process = pre_process
        self.post_process = post_process
        self.freq_dim = config.freq_dim
        self.has_image_input = config.has_image_input
        self.patch_size = config.latent_patch_size
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        self.hidden_size = config.hidden_size
        in_dim = config.in_dim
        out_dim = config.out_dim
        text_dim = config.text_dim
        patch_size = self.patch_size
        has_image_input = self.has_image_input
        eps = config.norm_epsilon
        self.has_image_pos_emb = config.has_image_pos_emb

        self.patch_embedding = nn.Conv3d(
            in_dim, config.hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, self.hidden_size), nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.head = Head(self.hidden_size, out_dim, patch_size, eps)
        head_dim = self.hidden_size // self.config.num_attention_heads
        f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)
        self.register_buffer('freqs_f', f_freqs, persistent=False)
        self.register_buffer('freqs_h', h_freqs, persistent=False)
        self.register_buffer('freqs_w', w_freqs, persistent=False)
        _f = (config.num_latent_frames - 1) // config.vae_temporal_compress + 1
        _h = config.max_latent_height // config.vae_spatial_compress // config.latent_patch_size[1]
        _w = config.max_latent_width // config.vae_spatial_compress // config.latent_patch_size[2]
        self._grid_f, self._grid_h, self._grid_w = _f, _h, _w
        _freqs = torch.cat([
            f_freqs[:_f].view(_f, 1, 1, -1).expand(_f, _h, _w, -1),
            h_freqs[:_h].view(1, _h, 1, -1).expand(_f, _h, _w, -1),
            w_freqs[:_w].view(1, 1, _w, -1).expand(_f, _h, _w, -1),
        ], dim=-1).reshape(_f * _h * _w, 1, -1)
        self.register_buffer('freqs_3d', _freqs, persistent=False)
        self.register_buffer('freqs_3d_cos', _freqs.real.squeeze(1).contiguous(), persistent=False)
        self.register_buffer('freqs_3d_sin', _freqs.imag.squeeze(1).contiguous(), persistent=False)

        if has_image_input:
            self.img_emb = MLP(
                1280, self.hidden_size, has_pos_emb=self.has_image_pos_emb
            )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True)
        )

        self.decoder = WanTransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=False,
        )

    def patchify(self, x: torch.Tensor):
        with torch.backends.cudnn.flags(enabled=False):
            x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def pad_image(self, clip):
        seq = clip.shape[1]
        pad_num = (
            self.config.context_parallel_size - seq % self.config.context_parallel_size
        )
        pad_num = pad_num % self.config.context_parallel_size
        if pad_num != 0:
            pad = torch.zeros(clip.shape[0], pad_num, clip.shape[2], device=clip.device, dtype=clip.dtype)
            clip = torch.cat([clip, pad], dim=1)
        return pad_num, clip

    def _build_packed_freqs(self, grid_sizes, seq_len_q_padded):
        """Build per-sample 3D RoPE frequencies for packing mode."""
        all_freqs = []
        f_freqs = self.freqs_f
        h_freqs = self.freqs_h
        w_freqs = self.freqs_w
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            freq_i = torch.cat(
                [
                    f_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)
            padded_len = seq_len_q_padded[i].item()
            if freq_i.shape[0] < padded_len:
                pad_shape = (padded_len - freq_i.shape[0], 1, freq_i.shape[2])
                freq_i = torch.cat(
                    [freq_i, torch.zeros(pad_shape, dtype=freq_i.dtype, device=freq_i.device)],
                    dim=0,
                )
            all_freqs.append(freq_i)
        return torch.cat(all_freqs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[Any] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """Wan forward. Dispatches to _forward_packed or _forward_single."""
        use_packing = grid_sizes is not None

        if use_packing:
            return self._forward_packed(
                x, timestep, context, clip_feature, y,
                packed_seq_params, grid_sizes, **kwargs,
            )
        else:
            return self._forward_single(
                x, timestep, context, clip_feature, y, **kwargs,
            )


    def _forward_single(
        self, x, timestep, context, clip_feature, y, **kwargs,
    ):
        """Non-packing forward path (matches new baseline exactly)."""
        f, h, w = self._grid_f, self._grid_h, self._grid_w
        freqs = self.freqs_3d
        rotary_pos_cos = self.freqs_3d_cos
        rotary_pos_sin = self.freqs_3d_sin

        if self.has_image_input:
            if clip_feature is None:
                raise ValueError("Wan2.1 I2V forward requires clip_feature when has_image_input=True.")
            if y is None and self.require_vae_embedding:
                raise ValueError("Wan2.1 I2V forward requires y when require_vae_embedding=True.")
            _, clip_feature = self.pad_image(clip_feature)

        t_for_head = None
        timestep_mod = None
        if self.pre_process:
            target_dtype = next(self.decoder.parameters()).dtype
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep)).to(dtype=target_dtype)
            t_s = t.unsqueeze(0)
            t_for_head = t

            timestep_mod = self.time_projection(t).unflatten(1, (6, self.hidden_size)).to(dtype=target_dtype)
            context = self.text_embedding(context).to(dtype=target_dtype)
            if y is not None and self.require_vae_embedding:
                x = torch.cat([x, y], dim=1)
            clip_embedding = None
            if clip_feature is not None and self.require_clip_embedding:
                with autocast("cuda", dtype=torch.bfloat16):
                    clip_embedding = self.img_emb(clip_feature)
                    clip_embedding = rearrange(
                        clip_embedding, "B S C -> S B C"
                    ).contiguous()

            x, (f, h, w) = self.patchify(x)
            freqs = torch.cat([
                self.freqs_f[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
            rotary_pos_cos = freqs.real.squeeze(1).contiguous()
            rotary_pos_sin = freqs.imag.squeeze(1).contiguous()

            x = rearrange(x, f"B S C ->S B C").contiguous()
            timestep_mod = rearrange(timestep_mod, f"B S C ->S B C").contiguous()
            context = rearrange(context, f"B S C ->S B C").contiguous()

            if self.has_image_input and clip_feature is not None and self.require_clip_embedding:
                context = torch.cat([clip_embedding, context], dim=0)

            cp = self.config.context_parallel_size
            if cp > 1:
                x = split_forward_gather_backward(
                    x, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                if not self.has_image_input:
                    # Wan2.2: split context across CP group
                    context = split_forward_gather_backward(
                        context, get_context_parallel_group(), dim=0, grad_scale="down"
                    )
                freqs = split_forward_gather_backward(
                    freqs, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                rotary_pos_cos = split_forward_gather_backward(
                    rotary_pos_cos, get_context_parallel_group(), dim=0, grad_scale="down"
                )
                rotary_pos_sin = split_forward_gather_backward(
                    rotary_pos_sin, get_context_parallel_group(), dim=0, grad_scale="down"
                )

            for layer in self.decoder.layers:
                layer.t_s = t_s
                layer._packing_cross_packed_seq_params = None
                layer._packing_num_samples = None
                layer._packing_cu_seqlens_q_padded = None
        else:
            x = None
            context = None
        extra_block_kwargs = {}
        x = self.decoder(
            hidden_states=x,
            attention_mask=None,
            context=context,
            context_mask=None,
            inference_params=None,
            packed_seq_params=None,
            rotary_pos_emb=freqs,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            timestep_mod=timestep_mod,
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return x
        assert t_for_head is not None, (
            "WanModel post-process requires t_for_head, "
            "but pre_process=False. Pipeline parallel (pp>1) is not yet supported."
        )

        t = t_for_head.to(torch.bfloat16)

        x = rearrange(x, f"S B C ->B S C").contiguous()
        x = self.head(x, t)

        if self.config.context_parallel_size > 1:
            x = gather_forward_split_backward(
                x, get_context_parallel_group(), dim=1, grad_scale="up"
            )

        x = self.unpatchify(x, (f, h, w))
        return x


    def _project_packed_latents(self, x, grid_sizes, seq_len_q_padded, cu_seqlens_q_padded):
        """Apply Conv3d patch projection to each packed sample independently."""
        patch_frames, patch_height, patch_width = self.patch_size
        patch_chunks = []
        for sample_index, grid_size in enumerate(grid_sizes.tolist()):
            frame_count, height_count, width_count = grid_size
            sample_seq_len = frame_count * height_count * width_count
            sample_start = cu_seqlens_q_padded[sample_index].item()
            sample_tokens = x[sample_start:sample_start + sample_seq_len].squeeze(1)
            input_channels = sample_tokens.shape[-1] // (patch_frames * patch_height * patch_width)
            sample_5d = sample_tokens.reshape(
                frame_count, height_count, width_count,
                input_channels, patch_frames, patch_height, patch_width,
            )
            sample_5d = sample_5d.permute(3, 0, 4, 1, 5, 2, 6).contiguous()
            sample_5d = sample_5d.reshape(
                1,
                input_channels,
                frame_count * patch_frames,
                height_count * patch_height,
                width_count * patch_width,
            )
            projected_5d = self.patch_embedding(sample_5d)
            projected = projected_5d.squeeze(0).permute(1, 2, 3, 0).reshape(
                -1, 1, self.hidden_size
            )
            pad_len = seq_len_q_padded[sample_index].item() - projected.shape[0]
            if pad_len > 0:
                projected = torch.nn.functional.pad(projected, (0, 0, 0, 0, 0, pad_len))
            patch_chunks.append(projected)
        return torch.cat(patch_chunks, dim=0)

    def _embed_packed_context(self, context, cross_attn_params, num_samples):
        """Apply text embedding without crossing packed sample boundaries."""
        if num_samples == 1:
            return self.text_embedding(context)

        cu_seqlens_kv_padded = cross_attn_params.cu_seqlens_kv_padded
        embedded_chunks = []
        for sample_index in range(num_samples):
            sample_start = cu_seqlens_kv_padded[sample_index].item()
            sample_end = cu_seqlens_kv_padded[sample_index + 1].item()
            embedded_chunks.append(self.text_embedding(context[sample_start:sample_end]))
        return torch.cat(embedded_chunks, dim=0)

    @staticmethod
    def _build_local_padded_cu_seqlens(seq_len_padded, cp_size, device):
        """Build local per-sample boundaries after CP split."""
        boundaries = [0]
        for sample_index in range(seq_len_padded.shape[0]):
            local_len = seq_len_padded[sample_index].item() // cp_size
            boundaries.append(boundaries[-1] + local_len)
        return torch.tensor(boundaries, dtype=torch.int32, device=device)

    def _split_packed_inputs_for_cp(
        self,
        x,
        context,
        freqs,
        freqs_cos,
        freqs_sin,
        self_attn_params,
        cross_attn_params,
    ):
        """Split packed video, text, and RoPE tensors on per-sample boundaries."""
        cp_size = self.config.context_parallel_size
        if cp_size <= 1:
            return x, context, freqs, freqs_cos, freqs_sin, None

        cp_rank = parallel_state.get_context_parallel_rank()
        seq_len_q_padded = self_attn_params._seq_len_q_padded
        cu_seqlens_q_padded = self_attn_params.cu_seqlens_q_padded
        seq_len_kv_padded = cross_attn_params._seq_len_kv_padded
        cu_seqlens_kv_padded = cross_attn_params.cu_seqlens_kv_padded

        x = thd_split_for_cp(x, cu_seqlens_q_padded, seq_len_q_padded, cp_size, cp_rank)
        context = thd_split_for_cp(
            context, cu_seqlens_kv_padded, seq_len_kv_padded, cp_size, cp_rank
        )
        freqs = thd_split_for_cp(freqs, cu_seqlens_q_padded, seq_len_q_padded, cp_size, cp_rank)
        freqs_cos = thd_split_for_cp(
            freqs_cos.unsqueeze(1), cu_seqlens_q_padded, seq_len_q_padded, cp_size, cp_rank
        ).squeeze(1)
        freqs_sin = thd_split_for_cp(
            freqs_sin.unsqueeze(1), cu_seqlens_q_padded, seq_len_q_padded, cp_size, cp_rank
        ).squeeze(1)
        local_cu_seqlens_q_padded = self._build_local_padded_cu_seqlens(
            seq_len_q_padded, cp_size, self_attn_params.cu_seqlens_q.device
        )
        return x, context, freqs, freqs_cos, freqs_sin, local_cu_seqlens_q_padded

    def _set_packing_state_on_layers(
        self,
        self_attn_params,
        cross_attn_params,
        num_samples,
        local_cu_seqlens_q_padded,
    ):
        """Expose packed metadata needed by WAN layers."""
        cp_size = self.config.context_parallel_size
        for layer in self.decoder.layers:
            layer._packing_cross_packed_seq_params = cross_attn_params
            layer._packing_num_samples = num_samples
            if cp_size > 1 and local_cu_seqlens_q_padded is not None:
                layer._packing_cu_seqlens_q_padded = local_cu_seqlens_q_padded
            else:
                layer._packing_cu_seqlens_q_padded = self_attn_params.cu_seqlens_q_padded
            layer.t_s = None

    def _restore_packed_output_order(self, x, timestep_state, seq_len_q_padded, num_samples):
        """Gather CP shards and restore packed sample-major token order."""
        cp_size = self.config.context_parallel_size
        if cp_size > 1:
            num_trailing_tokens = 7 * num_samples
            local_video_len = x.shape[0] - num_trailing_tokens
            timestep_state = x[-num_samples:, :, :]
            x_video_local = x[:local_video_len, :, :]
            x_gathered = gather_forward_split_backward(
                x_video_local, get_context_parallel_group(), dim=0, grad_scale="up"
            )
            _, global_to_gathered = _build_thd_reorder_indices(seq_len_q_padded, cp_size)
            return x_gathered.index_select(0, global_to_gathered.to(x_gathered.device)), timestep_state

        num_trailing_tokens = 7 * num_samples
        timestep_state = x[-num_samples:, :, :]
        return x[:-num_trailing_tokens, :, :], timestep_state

    def _apply_packed_head(self, x, timestep_state, self_attn_params, seq_len_q_padded, num_samples):
        """Apply output head independently for each packed sample."""
        cp_size = self.config.context_parallel_size
        if cp_size > 1:
            packed_boundaries = [0]
            for sample_index in range(num_samples):
                packed_boundaries.append(
                    packed_boundaries[-1] + seq_len_q_padded[sample_index].item()
                )
        else:
            packed_boundaries = self_attn_params.cu_seqlens_q.tolist()

        head_outputs = []
        for sample_index in range(num_samples):
            sample_start = packed_boundaries[sample_index]
            sample_end = packed_boundaries[sample_index + 1]
            sample_x = x[:, sample_start:sample_end, :]
            sample_timestep_state = timestep_state[sample_index:sample_index + 1, 0, :].to(torch.bfloat16)
            head_outputs.append(self.head(sample_x, sample_timestep_state))
        return torch.cat(head_outputs, dim=1)

    def _forward_packed(
        self, x, timestep, context, clip_feature, y,
        packed_seq_params, grid_sizes, **kwargs,
    ):
        """Forward pass for packed variable-length sequences."""
        self_attn_params = packed_seq_params["self_attention"]
        cross_attn_params = packed_seq_params["cross_attention"]
        num_samples = grid_sizes.shape[0]
        seq_len_q_padded = self_attn_params._seq_len_q_padded
        freqs = self._build_packed_freqs(grid_sizes, seq_len_q_padded)
        freqs_cos = freqs.real.squeeze(1).contiguous()
        freqs_sin = freqs.imag.squeeze(1).contiguous()

        if self.pre_process:
            timestep_state = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            timestep_mod = self.time_projection(timestep_state).unflatten(1, (6, self.hidden_size))
            timestep_mod = rearrange(timestep_mod, "B S C -> S B C").contiguous()

            if y is not None and self.require_vae_embedding:
                raise NotImplementedError("Packed WAN I2V training with VAE image embeddings is not supported yet.")
            if clip_feature is not None and self.require_clip_embedding:
                raise NotImplementedError("Packed WAN I2V training with CLIP image embeddings is not supported yet.")
            x = x.to(dtype=self.patch_embedding.weight.dtype)
            x = self._project_packed_latents(
                x, grid_sizes, seq_len_q_padded, self_attn_params.cu_seqlens_q_padded
            )
            context = self._embed_packed_context(context, cross_attn_params, num_samples)
            x, context, freqs, freqs_cos, freqs_sin, local_cu_seqlens_q_padded = (
                self._split_packed_inputs_for_cp(
                    x, context, freqs, freqs_cos, freqs_sin, self_attn_params, cross_attn_params
                )
            )
            trailing_timestep_mod = timestep_mod.permute(1, 0, 2).reshape(-1, 1, self.hidden_size)
            trailing_timestep_state = timestep_state.unsqueeze(1)
            x = torch.cat([x, trailing_timestep_mod, trailing_timestep_state], dim=0)
        else:
            x = None
            context = None
            timestep_mod = None
            timestep_state = None
            local_cu_seqlens_q_padded = None

        self._set_packing_state_on_layers(
            self_attn_params, cross_attn_params, num_samples, local_cu_seqlens_q_padded
        )

        x = self.decoder(
            hidden_states=x,
            attention_mask=None,
            context=context,
            context_mask=None,
            inference_params=None,
            packed_seq_params=self_attn_params,
            rotary_pos_emb=freqs,
            rotary_pos_cos=freqs_cos,
            rotary_pos_sin=freqs_sin,
            timestep_mod=timestep_mod,
        )
        if not self.post_process:
            return x

        x, timestep_state = self._restore_packed_output_order(
            x, timestep_state, seq_len_q_padded, num_samples
        )
        x = x.to(torch.bfloat16)
        x = rearrange(x, "S B C -> B S C").contiguous()
        x = self._apply_packed_head(
            x, timestep_state, self_attn_params, seq_len_q_padded, num_samples
        )
        return rearrange(x, "B S C -> S B C").contiguous()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1"
        self.decoder.set_input_tensor(input_tensor[0])
