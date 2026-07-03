# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4 RoPE helpers."""

import torch
from torch import Tensor


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: Tensor, freqs: Tensor, offset: int = 0
) -> Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg],
                freqs[
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg
                ],
            ]
        )
    return freqs[offset : offset + x.size(0)]


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    remove_interleaving: bool = False,
) -> Tensor:
    if freqs.dim() == t.dim() + 1 and freqs.size(-2) == 1:
        freqs = freqs.squeeze(-2)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)
    if inverse:
        sin_ = -sin_

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)

    if multi_latent_attention and remove_interleaving:
        x1, x2 = torch.chunk(t, 2, dim=-1)
        t = torch.stack((x1, x2), dim=-1).flatten(start_dim=-2)

    return torch.cat((t, t_pass), dim=-1)


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    inverse: bool = False,
    remove_interleaving: bool = False,
    cp_group: torch.distributed.ProcessGroup = None,
    **kwargs,
) -> Tensor:
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()

    if freqs.dim() >= 1 and freqs.size(0) == cu_seqlens[-1]:
        sequence_splits = torch.split(t, seqlens)
        freq_slices = []
        for i, x in enumerate(sequence_splits):
            seq_start_offset = cu_seqlens[i].item()
            freq_slices.append(
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, seq_start_offset)
            )
        freqs_packed = torch.cat(freq_slices, dim=0)
    elif "offsets" in kwargs:
        offsets = kwargs["offsets"]
        sequence_splits = torch.split(t, seqlens)
        freqs_packed = torch.cat(
            [
                _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs, seq_start_offset)
                for x, seq_start_offset in zip(sequence_splits, offsets)
            ],
            dim=0,
        )
    else:
        sequence_splits = torch.split(t, seqlens)
        freqs_packed = torch.cat(
            [_get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs) for x in sequence_splits],
            dim=0,
        )

    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1),
        freqs_packed,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        mscale=mscale,
        inverse=inverse,
        remove_interleaving=remove_interleaving,
    ).squeeze(1)


def apply_dsv4_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config,
    cu_seqlens: Tensor = None,
    mscale: float = 1.0,
    cp_group: torch.distributed.ProcessGroup = None,
    inverse: bool = False,
    remove_interleaving: bool = True,
    **kwargs,
) -> Tensor:
    """Apply DSv4's MLA RoPE layout in the no-fusion path."""
    if cu_seqlens is None:
        return _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=config.rotary_interleaved,
            multi_latent_attention=config.multi_latent_attention,
            mscale=mscale,
            inverse=inverse,
            remove_interleaving=remove_interleaving,
        )

    return _apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        multi_latent_attention=config.multi_latent_attention,
        mscale=mscale,
        inverse=inverse,
        remove_interleaving=remove_interleaving,
        cp_group=cp_group,
        **kwargs,
    )
