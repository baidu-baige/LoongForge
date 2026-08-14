# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# The KV-cache design (``update_cache`` levels, sliding ``attn_window``, per-chunk token
# accounting) follows upstream's ``wan_va/modules/model.py``, where cache support is built
# into the single transformer class shared by training and inference. Parts were
# cross-checked against the LingBot-VA port in LeRobot (Apache-2.0, Copyright 2024 The
# HuggingFace Inc. team), which is a third-party port, not the authoritative reference.

"""KV-cache streaming variants of the LingBot-VA Wan transformer.

These subclasses add autoregressive inference capability (single-stream forward
+ sliding-window KV cache) on top of the training-only ``wan_model`` classes,
*without* touching the training code path. The module/parameter names are kept
identical to ``WanTransformer3DModel`` so a training checkpoint's ``state_dict``
loads into ``WanStreamingTransformer3DModel`` directly.

Reference: the upstream LingBot-VA streaming server
(``wan_va/wan_va_server.py``) and its LeRobot port
(``lerobot/policies/lingbot_va/utils.py``).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .flow_match import get_mesh_id
from .wan_model import WanAttention, WanTransformerBlock, WanTransformer3DModel


def get_mesh_id_streaming(
    frames: int,
    height: int,
    width: int,
    token_type: int,
    frame_shift: int = 0,
    action: bool = False,
):
    """Grid ids for autoregressive inference: the training grid shifted on the frame axis.

    Each chunk must be placed at its *absolute* frame position within the episode, so rope
    stays consistent with the frames already held in the KV cache. Training always starts a
    sample at frame 0 and therefore has no need for the shift.

    Shifting the frame axis after the fact is exactly equivalent to building the grid from
    ``arange(frame_shift, frames + frame_shift)``: the frame ids are that range, and the
    action stream's fractional per-sub-step offset is added to the same axis, so a constant
    commutes with it. Reusing the training helper keeps a single definition of the grid.
    """
    grid = get_mesh_id(frames, height, width, token_type, action=action)
    if frame_shift:
        grid[0] = grid[0] + frame_shift
    return grid


def data_seq_to_patch(
    patch_size, data_seq, latent_num_frames, latent_height, latent_width, batch_size=1
):
    """Reshape a flattened patch sequence back into a ``[B, C, F, H, W]`` latent grid.

    Inverse of the latent stream's input patchification. Inference needs it to turn the
    transformer's token-sequence output back into a latent grid the flow-matching
    scheduler can step on.
    """
    p_t, p_h, p_w = patch_size
    data_patch = data_seq.reshape(
        batch_size,
        latent_num_frames // p_t,
        latent_height // p_h,
        latent_width // p_w,
        p_t,
        p_h,
        p_w,
        -1,
    )
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class WanStreamingAttention(WanAttention):
    """``WanAttention`` with a sliding-window KV cache for self-attention.

    The cache lives only on self-attention modules (``cross_attention_dim_head
    is None``). Cross-attention reuses the base-class behaviour unchanged.
    ``update_cache`` semantics (mirrors upstream):

    * ``0`` — read-only: the new key/value are temporarily stored, used for the
      attention, then removed again (``restore_cache``).
    * ``1`` — write as a *predicted* slot (denoising intermediate): kept in the
      cache but flagged ``is_pred`` so ``clear_pred_cache`` can drop it later.
    * ``2`` — write as a *real* slot (observed keyframe / executed action): kept
      permanently for the closed-loop world-model feedback.
    """

    def __init__(self, *args, **kwargs):
        """Build the base attention, then attach an empty cache registry for self-attention."""
        super().__init__(*args, **kwargs)
        # Only self-attention keeps a KV cache (cross-attention has no cache).
        self.attn_caches = {} if not self.is_cross else None

    # ── cache management ─────────────────────────────────────────────
    def clear_pred_cache(self, cache_name: str) -> None:
        """Invalidate the slots written as denoising intermediates, keeping the real ones."""
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        if cache is None:
            return
        cache["mask"][cache["is_pred"]] = False

    def clear_cache(self, cache_name: str) -> None:
        """Drop the whole cache entry, releasing its key/value tensors."""
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(
        self,
        cache_name: str,
        total_token_len: int,
        num_head: int,
        head_dim: int,
        device,
        dtype,
        batch_size: int,
    ) -> None:
        """Preallocate a fixed-size key/value pool plus its slot bookkeeping tensors.

        The pool is bounded by ``total_token_len``, which is what makes the sliding
        window work: slots are recycled rather than appended, so memory stays flat
        over an arbitrarily long episode.
        """
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            "k": torch.empty([batch_size, total_token_len, num_head, head_dim], device=device, dtype=dtype),
            "v": torch.empty([batch_size, total_token_len, num_head, head_dim], device=device, dtype=dtype),
            "id": torch.full((total_token_len,), -1, device=device),
            "mask": torch.zeros((total_token_len,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((total_token_len,), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name: str, key_size: int):
        """Reserve ``key_size`` free slots, evicting the oldest entries when the pool is full.

        Eviction order is by write id, which is what implements the sliding attention
        window: the oldest chunks fall out of the cache as new ones arrive.
        """
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            # Evict the oldest used slots to make room (ring-buffer behaviour).
            used = mask.nonzero(as_tuple=False).squeeze(-1)
            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]
            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            raise RuntimeError(f"KV cache exhausted: need {key_size} free slots, have {free.numel()}.")
        return free[:key_size]

    def _next_cache_id(self, cache_name: str):
        """Return the write id to stamp on the next batch of slots (monotonic per cache)."""
        ids = self.attn_caches[cache_name]["id"]
        mask = self.attn_caches[cache_name]["mask"]
        if mask.any():
            return ids[mask].max() + 1
        return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name: str, key, value, is_pred: bool):
        """Write one key/value block into the pool and return the slots it occupies.

        ``is_pred`` marks the block as a denoising intermediate so ``clear_pred_cache``
        can drop it once the chunk's final value is known.
        """
        cache = self.attn_caches[cache_name]
        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)
        new_id = self._next_cache_id(cache_name)
        cache["k"][:, slots] = key
        cache["v"][:, slots] = value
        cache["mask"][slots] = True
        cache["id"][slots] = new_id
        cache["is_pred"][slots] = is_pred
        return slots

    def restore_cache(self, cache_name: str, slots) -> None:
        """Release slots written by a read-only call, leaving the cache as it was."""
        self.attn_caches[cache_name]["mask"][slots] = False

    # ── forward ──────────────────────────────────────────────────────
    def forward(self, q, k, v, rotary_emb=None, update_cache=0, cache_name="pos"):
        """Attend over the incoming tokens plus every valid slot in the KV cache.

        See the class docstring for the three ``update_cache`` levels. Cross-attention
        modules have no cache and degrade to plain attention over ``k`` / ``v``.
        """
        kv_cache = (
            self.attn_caches[cache_name]
            if (self.attn_caches is not None) and (cache_name in self.attn_caches)
            else None
        )

        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
        value = self.to_v(v).unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            query = self._apply_rotary(query, rotary_emb)
            key = self._apply_rotary(key, rotary_emb)

        slots = None
        if kv_cache is not None and kv_cache["k"] is not None:
            slots = self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]["k"]
            value_pool = self.attn_caches[cache_name]["v"]
            mask = self.attn_caches[cache_name]["mask"]
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)

        if update_cache == 0 and slots is not None:
            self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanStreamingTransformerBlock(WanTransformerBlock):
    """``WanTransformerBlock`` whose self-attention keeps a KV cache."""

    def __init__(self, dim, ffn_dim, num_heads, cross_attn_norm=False, eps=1e-6, attn_mode="torch"):
        """Build the base block, then swap in the cache-capable self-attention module."""
        super().__init__(dim, ffn_dim, num_heads, cross_attn_norm, eps, attn_mode)
        # Replace the base self-attention with the streaming variant; parameter
        # names are unchanged so a training checkpoint still loads.
        head_dim = dim // num_heads
        self.attn1 = WanStreamingAttention(
            dim, num_heads, head_dim, eps, cross_attention_dim_head=None, attn_mode=attn_mode
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
    ):
        """Run the block, forwarding the cache controls to self-attention.

        Identical in arithmetic to the base block; the only additions are the
        ``update_cache`` / ``cache_name`` passthrough.
        """
        # ``temb`` is the [B, L, 6, C] modulation tensor produced by
        # ``WanTransformer3DModel._time_embed``.
        modulation = self.scale_shift_table[None] + temb.float()
        shift, scale, gate, ff_shift, ff_scale, ff_gate = rearrange(
            modulation, "b l n c -> b n l c"
        ).chunk(6, dim=1)

        # 1. Self-attention (with KV cache)
        normed = self.norm1(hidden_states.float())
        normed = (normed * (1 + scale.squeeze(1)) + shift.squeeze(1)).to(hidden_states.dtype)
        self_update = self.attn1(
            normed,
            normed,
            normed,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + self_update * gate.squeeze(1)).to(hidden_states.dtype)

        # 2. Cross-attention (no cache)
        if isinstance(self.norm2, nn.Identity):
            cross_input = hidden_states
        else:
            cross_input = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        cross_update = self.attn2(cross_input, encoder_hidden_states, encoder_hidden_states)
        hidden_states = hidden_states + cross_update

        # 3. Feed-forward
        normed = self.norm3(hidden_states.float())
        normed = (normed * (1 + ff_scale.squeeze(1)) + ff_shift.squeeze(1)).to(hidden_states.dtype)
        ffn_update = self.ffn(normed)
        hidden_states = (hidden_states.float() + ffn_update.float() * ff_gate.squeeze(1)).to(
            hidden_states.dtype
        )
        return hidden_states


class WanStreamingTransformer3DModel(WanTransformer3DModel):
    """Dual-stream Wan transformer with autoregressive KV-cache inference.

    Training still goes through the base class ``forward`` (``train_mode=True``
    delegates to the inherited dual-stream path). Inference uses the single-stream
    ``forward`` below, which processes one stream (video latent or action) at a
    time and updates/reads the KV cache.
    """

    def __init__(self, *args, **kwargs):
        """Build the base model, then rebuild its blocks with streaming attention."""
        super().__init__(*args, **kwargs)
        # Rebuild the transformer blocks with streaming attention. Parameter
        # names are unchanged (``blocks.N.attn1.*``) so training checkpoints load.
        inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.blocks = nn.ModuleList(
            [
                WanStreamingTransformerBlock(
                    inner_dim,
                    self.config.ffn_dim,
                    self.config.num_attention_heads,
                    self.config.cross_attn_norm,
                    self.config.eps,
                    attn_mode=self.config.attn_mode,
                )
                for _ in range(self.config.num_layers)
            ]
        )

    # ── KV-cache management ──────────────────────────────────────────
    def clear_cache(self, cache_name: str) -> None:
        """Release the named cache on every block, e.g. at an episode boundary."""
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name: str) -> None:
        """Drop the denoising intermediates on every block, keeping the real slots."""
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name: str,
        attn_window: int,
        latent_token_per_chunk: int,
        action_token_per_chunk: int,
        device,
        dtype,
        batch_size: int,
    ) -> None:
        """Allocate the per-block KV pools sized for one sliding ``attn_window``.

        The window holds interleaved video-latent and action chunks, hence the split
        into ``attn_window // 2`` chunks of each kind. This bound is what keeps peak
        memory flat regardless of episode length.
        """
        total_token_len = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_token_len,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                device,
                dtype,
                batch_size,
            )

    # ── single-stream inference forward ──────────────────────────────
    def forward(self, input_dict, update_cache=0, cache_name="pos", action_mode=False, train_mode=False):
        """Run one stream through the transformer, or delegate to the training path.

        ``train_mode=True`` forwards to the base class's dual-stream training path
        untouched. Otherwise a single stream is processed: ``action_mode`` selects the
        action embedder / projection, and the video-latent path patchifies its input.
        ``update_cache`` and ``cache_name`` are handed to self-attention.
        """
        if train_mode:
            # Training dual-stream path (inherited from the base class).
            return super().forward(input_dict)

        if action_mode:
            # Action input embedding: [B, C, F, apf, 1] -> [B, F*apf, C]
            latent_hidden_states = rearrange(input_dict["noisy_latents"], "b c f h w -> b (f h w) c")
            latent_hidden_states = self.action_embedder(latent_hidden_states)
        else:
            # Video-latent input embedding with patchification.
            latent_hidden_states = rearrange(
                input_dict["noisy_latents"],
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            latent_hidden_states = self.patch_embedding_mlp(latent_hidden_states)

        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])

        latent_grid_id = input_dict["grid_id"]
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # [1, L, 1, C]
        patch_scale_h, patch_scale_w = (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict["timesteps"],
            (input_dict["noisy_latents"].shape[-2] // patch_scale_h)
            * (input_dict["noisy_latents"].shape[-1] // patch_scale_w),
            dim=1,
        )
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # [B, L, 6, C]

        for block in self.blocks:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) * (1.0 + scale) + shift).type_as(
            latent_hidden_states
        )

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size)
            )

        return latent_hidden_states
