# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""WAN-family DiT primitives and one block (Lane B, L2).

This module is a real algorithm implementation, not a forwarder to the
existing ``wan_video_dit.py``. Submodule names deliberately match the
existing ones (``self_attn.{q,k,v,o,norm_q,norm_k}`` / ``norm1`` / ``norm2``
/ ``norm3`` / ``ffn`` / ``modulation``), so an existing checkpoint only needs
a prefix move to load strictly.

The one structural difference from the existing code is on the
cross-attention KV side: KV is grouped by slot name into
``cross_attn.kv.<slot>.{k,v,norm_k}``, replacing the hardcoded ``k_img`` of
i2v and its convention that "the first 257 tokens are the image". A slot name
is a condition name on ``ConditionBus`` (§4.4), which is where "the Nth
condition input" stops being conflated with "the Nth output head".

A layer is split into ``attention_io`` and ``post_attention``, with the one
attention call in between performed across streams by ``StreamEngine``. That
split is what lets two streams enter the same joint attention (the
non-reorderable step in interfaces §7).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from ..state import QKV

__all__ = [
    "FP32LayerNorm", "RMSNorm", "SelfAttentionIO", "SlottedCrossAttention",
    "WanDiTBlock", "assemble_freqs_3d", "modulate", "rope_apply",
    "rope_freqs_1d", "rope_freqs_3d", "sinusoidal_embedding",
]


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply ``x * (1 + scale) + shift``.

    This is the only place in the tree that writes this formula. The order of
    ``shift`` and ``scale`` appears both ways in the existing code
    (``scale, shift = ...chunk(2)`` and ``shift, scale = ...``); a swap keeps
    the shapes valid and part of the numeric comparison still passes, so
    named parameters make the order readable at the call site.

    Args:
        x: Input activations.
        shift: Additive term, broadcastable to ``x``.
        scale: Multiplicative term applied as ``1 + scale``.

    Returns:
        The modulated tensor, same shape as ``x``.
    """
    return x * (1 + scale) + shift


def sinusoidal_embedding(width: int, position: Tensor) -> Tensor:
    """Build the sinusoidal timestep embedding.

    Args:
        width: Embedding width, i.e. ``StreamSpec.timestep_embed_width``
            (``freq_dim=256`` in the existing code). It is the input width of
            the ``time_embedding`` Linear and is unrelated to
            residual_width. Must be even.
        position: ``[B]`` timestep values.

    Returns:
        ``[B, width]``, the concatenation of cos and sin halves. The float64
        intermediate is kept from the existing code: with t in [0, 1000] the
        float32 phase error is visible.

    Raises:
        ValueError: If ``width`` is odd, or if ``position`` is not 1-D.
    """
    if width % 2 != 0:
        raise ValueError(f"sinusoidal_embedding: width must be even, got {width}")
    if position.ndim != 1:
        raise ValueError(
            f"sinusoidal_embedding: position must be 1-D [B], "
            f"got {tuple(position.shape)}"
        )
    half = width // 2
    dtype = position.dtype if position.is_floating_point() else torch.float32
    scales = torch.pow(
        10000,
        -torch.arange(half, dtype=torch.float64,
                      device=position.device).div(half))
    sinusoid = torch.outer(position.to(torch.float64), scales)
    return torch.cat([sinusoid.cos(), sinusoid.sin()], dim=1).to(dtype)


def rope_freqs_1d(dim: int, end: int, theta: float = 10000.0) -> Tensor:
    """Build a 1-D RoPE frequency table.

    Args:
        dim: Width of the rotated axis. Must be even.
        end: Number of positions to tabulate.
        theta: Base of the geometric frequency progression.

    Returns:
        ``[end, dim/2]`` complex tensor. The action stream uses it directly.

    Raises:
        ValueError: If ``dim`` is odd.
    """
    if dim % 2 != 0:
        raise ValueError(f"rope_freqs_1d: dim must be even, got {dim}")
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).double()[:dim // 2] / dim))
    return torch.polar(torch.ones_like(freqs),
                       torch.outer(torch.arange(end).double(), freqs))


def rope_freqs_3d(dim: int, end: int = 1024,
                  theta: float = 10000.0) -> tuple[Tensor, Tensor, Tensor]:
    """Build the three RoPE frequency tables (t, h, w) for 3-D positions.

    The split is kept from the existing code:
    ``dim - 4*(dim//6)`` / ``2*(dim//6)`` / ``2*(dim//6)``. With
    head_dim=128 that is 44 / 42 / 42 (22 + 21 + 21 = 64 on the complex
    side). An even three-way split is a different, non-equivalent weight
    semantics, so do not "fix" it in passing.

    Args:
        dim: Total width, equal to the sum of the three table widths.
        end: Number of positions to tabulate per axis.
        theta: Base of the geometric frequency progression.

    Returns:
        The ``(t, h, w)`` tables, each complex.
    """
    sixth = dim // 6
    return (rope_freqs_1d(dim - 4 * sixth, end, theta),
            rope_freqs_1d(2 * sixth, end, theta),
            rope_freqs_1d(2 * sixth, end, theta))


def assemble_freqs_3d(parts: Sequence[Tensor],
                      grid: tuple[int, int, int]) -> Tensor:
    """Expand the three tables over a patch grid into per-token frequencies.

    Args:
        parts: The ``(t, h, w)`` frequency tables.
        grid: Patch grid ``(t, h, w)``.

    Returns:
        ``[t*h*w, 1, dim/2]``. The flattening order is (t, h, w) in C order
        and must match the patchify flatten order and the frame/row/col
        order of ``GeometrySpec.build_labels``. Getting one axis out of step
        across the three does not change the shape, it just encodes a
        spatial position as a temporal one.

    Raises:
        ValueError: If any table has fewer rows than the grid needs. The
            table ``end`` must cover the largest side of
            ``GeometrySpec.patch_grid``.
    """
    t, h, w = grid
    for axis, (table, need) in enumerate(zip(parts, grid, strict=True)):
        if table.shape[0] < need:
            raise ValueError(
                f"assemble_freqs_3d: frequency table for axis {axis} has only "
                f"{table.shape[0]} rows, the grid needs {need}. The table "
                "`end` must cover the largest side of GeometrySpec.patch_grid"
            )
    return torch.cat([
        parts[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
        parts[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
        parts[2][:w].view(1, 1, w, -1).expand(t, h, w, -1),
    ], dim=-1).reshape(t * h * w, 1, -1)


def rope_apply(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply RoPE to a 4-D activation tensor.

    The existing function of the same name takes ``[B, L, H*D]`` and
    rearranges internally; this one takes 4-D directly because ``QKV``
    already stores heads and head_dim on separate axes (required by the V3
    check). The numerics are identical.

    Args:
        x: ``[B, L, H, D]`` activations.
        freqs: ``[L, 1, D/2]`` complex frequency table.

    Returns:
        The rotated tensor, same shape and dtype as ``x``.

    Raises:
        ValueError: If ``x`` is not 4-D, or if the table covers fewer
            positions than the sequence length.
    """
    if x.ndim != 4:
        raise ValueError(
            f"rope_apply: x must be 4-D [B, L, H, D], got {tuple(x.shape)}")
    length = x.shape[1]
    if freqs.shape[0] < length:
        raise ValueError(
            f"rope_apply: frequency table covers only {freqs.shape[0]} "
            f"positions, but the sequence length is {length}"
        )
    pairs = torch.view_as_complex(
        x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(pairs * freqs[:length].to(x.device))
    return rotated.flatten(3).to(x.dtype)


class RMSNorm(nn.Module):
    """RMSNorm with a weight and no bias, with statistics taken in float32.

    It acts on the whole ``attn_width = num_heads * head_dim`` vector, not
    per head. That is the shape the existing ``norm_q`` / ``norm_k`` have;
    making it per head would leave the checkpoint's ``[attn_width]`` tensor
    with nowhere to go.

    Attributes:
        eps: Variance epsilon.
        weight: ``[width]`` learnable scale.
    """

    def __init__(self, width: int, *, eps: float = 1e-6) -> None:
        """Create the norm.

        Args:
            width: Vector width, normally ``num_heads * head_dim``.
            eps: Variance epsilon.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the last axis by its RMS and scale.

        Args:
            x: Input whose last axis has length ``width``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm computed in float32; the caller chooses affine or not.

    In bf16 the LayerNorm variance swallows small activations, so the
    existing code uniformly upcasts. Keeping the same behaviour matters:
    otherwise the numeric gap against the existing weights accumulates to
    several percent.
    """

    def __init__(self, width: int, *, eps: float = 1e-6,
                 elementwise_affine: bool = False) -> None:
        """Create the norm.

        Args:
            width: Normalized width.
            eps: Variance epsilon.
            elementwise_affine: Whether to allocate weight and bias.
        """
        super().__init__(width, eps=eps,
                         elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize in float32 and cast back to the input dtype.

        Args:
            x: Input whose last axis has length ``width``.

        Returns:
            Tensor of the same shape and dtype as ``x``.
        """
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias,
                            self.eps).type_as(x)


class SelfAttentionIO(nn.Module):
    """The two halves of self-attention: QKV (``qkv``) and output projection.

    The attention call in between is deliberately absent; the engine does it
    once across streams. The three q/k/v Linears and the o Linear live in
    one class so their submodule names match the existing
    ``self_attn.{q,k,v,o}`` exactly.

    Attributes:
        num_heads: Number of heads.
        head_dim: Per-head width.
    """

    def __init__(self, *, residual_width: int, num_heads: int, head_dim: int,
                 eps: float = 1e-6, qk_norm: bool = True) -> None:
        """Build the projections and optional q/k norms.

        Args:
            residual_width: Block input and output width.
            num_heads: Number of heads.
            head_dim: Per-head width; attn_width is num_heads * head_dim.
            eps: Epsilon for the q/k RMSNorms.
            qk_norm: When False, ``norm_q`` / ``norm_k`` are Identity.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_width = num_heads * head_dim
        self.q = nn.Linear(residual_width, attn_width)
        self.k = nn.Linear(residual_width, attn_width)
        self.v = nn.Linear(residual_width, attn_width)
        self.o = nn.Linear(attn_width, residual_width)
        self.norm_q = RMSNorm(attn_width, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(attn_width, eps=eps) if qk_norm else nn.Identity()

    def _heads(self, flat: Tensor) -> Tensor:
        """Split an ``attn_width`` axis into ``(num_heads, head_dim)``."""
        return flat.unflatten(-1, (self.num_heads, self.head_dim))

    def qkv(self, x: Tensor, *, freqs: Tensor | None) -> QKV:
        """Project the residual into Q/K/V.

        The norms act on the unsplit vector, as in the existing code; head
        splitting and RoPE come after.

        Args:
            x: ``[B, L, residual_width]`` residual.
            freqs: RoPE frequency table, or None to add no positional
                encoding for this stream.

        Returns:
            A ``QKV`` triple, each ``[B, L, H, D]``.
        """
        q = self._heads(self.norm_q(self.q(x)))
        k = self._heads(self.norm_k(self.k(x)))
        v = self._heads(self.v(x))
        if freqs is not None:
            q = rope_apply(q, freqs)
            k = rope_apply(k, freqs)
        return QKV(q=q, k=k, v=v)

    def project(self, attn_out: Tensor) -> Tensor:
        """Project the attention output back to the residual width.

        Args:
            attn_out: ``[B, L, H, D]`` attention output.

        Returns:
            ``[B, L, residual_width]``.
        """
        return self.o(attn_out.flatten(2))


class _CrossKV(nn.Module):
    """KV side of one cross-attention slot.

    Submodule names match the existing ``k`` / ``v`` / ``norm_k``, so the
    i2v ``k_img`` simply becomes "the slot named clip_image".
    """

    def __init__(self, *, residual_width: int, attn_width: int,
                 eps: float, qk_norm: bool) -> None:
        """Build the k/v projections and the optional key norm.

        Args:
            residual_width: Width of the condition context tensor.
            attn_width: num_heads * head_dim.
            eps: Epsilon for ``norm_k``.
            qk_norm: When False, ``norm_k`` is Identity.
        """
        super().__init__()
        self.k = nn.Linear(residual_width, attn_width)
        self.v = nn.Linear(residual_width, attn_width)
        self.norm_k = RMSNorm(attn_width, eps=eps) if qk_norm else nn.Identity()


class SlottedCrossAttention(nn.Module):
    """Cross-attention whose KV is grouped by slot name; one slot = one
    condition.

    The existing i2v code packs two conditions into one context tensor and
    splits it by "the first 257 tokens are the image"
    (``wan_video_dit.py:188-204``). That convention computes the wrong thing
    silently once the image token count changes, and the number 257 is
    written separately in three files. Here lookup is by name: a slot name is
    a condition name on ``ConditionBus``, and the length comes from the data.

    Each slot's attention output is summed first and passed through a single
    ``o`` afterwards, as in the existing code. Applying ``o`` per slot and
    then summing would add one bias per slot, which is a different weight
    semantics.

    Attributes:
        slots: Slot names, in the order they are summed.
        num_heads: Number of heads.
        head_dim: Per-head width.
    """

    def __init__(self, *, residual_width: int, num_heads: int, head_dim: int,
                 slots: Sequence[str], eps: float = 1e-6,
                 qk_norm: bool = True) -> None:
        """Build the query side and one KV group per slot.

        Args:
            residual_width: Block input and output width.
            num_heads: Number of heads.
            head_dim: Per-head width.
            slots: Slot names; must be non-empty.
            eps: Epsilon for the RMSNorms.
            qk_norm: When False, the q/k norms are Identity.

        Raises:
            ValueError: If ``slots`` is empty. A stream with no
                cross-attention should leave ``StreamSpec.cross_kv_slots``
                empty and have the caller skip building this module.
        """
        super().__init__()
        if not slots:
            raise ValueError(
                "SlottedCrossAttention needs at least one slot; a stream "
                "without cross-attention should leave "
                "StreamSpec.cross_kv_slots empty so the caller does not "
                "build this module"
            )
        self.slots = tuple(slots)
        self.num_heads = num_heads
        self.head_dim = head_dim
        attn_width = num_heads * head_dim
        self.q = nn.Linear(residual_width, attn_width)
        self.o = nn.Linear(attn_width, residual_width)
        self.norm_q = RMSNorm(attn_width, eps=eps) if qk_norm else nn.Identity()
        self.kv = nn.ModuleDict({
            slot: _CrossKV(residual_width=residual_width,
                           attn_width=attn_width, eps=eps, qk_norm=qk_norm)
            for slot in self.slots
        })

    def forward(self, x: Tensor, context: Mapping[str, Tensor]) -> Tensor:
        """Attend from the residual to every condition slot and sum.

        Args:
            x: ``[B, L, W]`` residual providing the queries.
            context: Slot name to ``[B, Lc, W]`` condition tensor. The key
                set must equal the slot set exactly: a missing slot means one
                weight group was not called this step (the loss still goes
                down and its grads are None), an extra key means a condition
                has nowhere to land.

        Returns:
            ``[B, L, W]`` to be added to the residual by the caller.

        Raises:
            KeyError: If a slot has no context, or if context has a slot this
                layer does not declare.
        """
        missing = [slot for slot in self.slots if slot not in context]
        if missing:
            raise KeyError(
                f"SlottedCrossAttention: slots {missing} have no context; "
                f"this layer declares {list(self.slots)}, "
                f"received {sorted(context)}"
            )
        stray = sorted(set(context) - set(self.slots))
        if stray:
            raise KeyError(
                f"SlottedCrossAttention: context has extra slots {stray}; "
                f"this layer declares {list(self.slots)}"
            )
        shape = (self.num_heads, self.head_dim)
        q = self.norm_q(self.q(x)).unflatten(-1, shape).transpose(1, 2)
        total: Tensor | None = None
        for slot in self.slots:
            kv = self.kv[slot]
            ctx = context[slot]
            k = kv.norm_k(kv.k(ctx)).unflatten(-1, shape).transpose(1, 2)
            v = kv.v(ctx).unflatten(-1, shape).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v)
            total = out if total is None else total + out
        assert total is not None            # slots non-empty per __init__
        return self.o(total.transpose(1, 2).flatten(2))


class WanDiTBlock(nn.Module):
    """One WAN-family layer, split into ``attention_io`` and ``post_attention``.

    The one joint attention call **must** be inserted between the two halves.
    A single ``forward`` cannot express that, which is why the existing code
    interleaves both streams inside one MoT forward (``mot.py``'s
    ``_build_expert_attention_io`` / ``_mixed_attention`` /
    ``_apply_expert_post_block`` are the hand-written version of this split).

    Empty ``cross_kv_slots`` means ``cross_attn`` and ``norm3`` are not built
    at all, rather than built and never called: building them would put the
    weights in state_dict, consume memory, and make a strict load look
    correct.
    """

    def __init__(self, *, residual_width: int, num_heads: int, head_dim: int,
                 ffn_width: int, num_modulation_slots: int,
                 cross_kv_slots: Sequence[str] = (),
                 eps: float = 1e-6, qk_norm: bool = True,
                 cross_attn_norm: bool = True) -> None:
        super().__init__()
        self.self_attn = SelfAttentionIO(
            residual_width=residual_width, num_heads=num_heads,
            head_dim=head_dim, eps=eps, qk_norm=qk_norm)
        self.norm1 = FP32LayerNorm(residual_width, eps=eps)
        self.norm2 = FP32LayerNorm(residual_width, eps=eps)
        if cross_kv_slots:
            self.cross_attn = SlottedCrossAttention(
                residual_width=residual_width, num_heads=num_heads,
                head_dim=head_dim, slots=cross_kv_slots, eps=eps,
                qk_norm=qk_norm)
            self.norm3 = FP32LayerNorm(
                residual_width, eps=eps, elementwise_affine=cross_attn_norm)
        else:
            self.cross_attn = None
            self.norm3 = None
        self.ffn = nn.Sequential(
            nn.Linear(residual_width, ffn_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_width, residual_width),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, num_modulation_slots, residual_width)
            / residual_width ** 0.5)

    def combine_modulation(self, packed: Tensor) -> Tensor:
        """Add this layer's learnable modulation to the timestep modulation.

        The addition happens once in **every** layer because ``modulation``
        is a per-layer parameter, so it cannot be hoisted into ``prepare``
        and cached once.

        Args:
            packed: ``[B, slots, W]`` timestep modulation for this step.

        Returns:
            ``[B, slots, W]`` sum of the per-layer parameter and ``packed``.

        Raises:
            ValueError: If the slot count of ``packed`` differs from this
                layer's ``modulation`` slot count.
        """
        if packed.shape[-2] != self.modulation.shape[1]:
            raise ValueError(
                f"WanDiTBlock.combine_modulation: timestep modulation has "
                f"{packed.shape[-2]} slots, this layer's learnable "
                f"modulation has {self.modulation.shape[1]}"
            )
        return self.modulation.to(dtype=packed.dtype,
                                  device=packed.device) + packed

    def attention_io(self, x: Tensor, *, shift_msa: Tensor, scale_msa: Tensor,
                     freqs: Tensor | None) -> QKV:
        """Run the first half: norm1, modulation, q/k/v with qk-norm and RoPE.

        No attention is performed here.

        Args:
            x: ``[B, L, W]`` residual.
            shift_msa: Additive modulation for the attention branch.
            scale_msa: Multiplicative modulation for the attention branch.
            freqs: RoPE table, or None for no positional encoding.

        Returns:
            A ``QKV`` triple, each ``[B, L, H, D]``.
        """
        return self.self_attn.qkv(
            modulate(self.norm1(x), shift_msa, scale_msa), freqs=freqs)

    def post_attention(self, x: Tensor, attn_out: Tensor, *,
                       gate_msa: Tensor, shift_mlp: Tensor,
                       scale_mlp: Tensor, gate_mlp: Tensor,
                       context: Mapping[str, Tensor]) -> Tensor:
        """Run the second half of the layer, after the joint attention call.

        The order is: o projection, gated residual, cross-attention, FFN,
        gated residual.

        Args:
            x: ``[B, L, W]`` residual **before** modulation (the
                ``residual_x`` of the existing ``_apply_expert_post_block``).
                Passing the post-modulation copy modulates the residual
                twice; the shapes stay valid and the loss still goes down,
                which makes it one of the hardest defects to spot.
            attn_out: ``[B, L, H, D]`` joint attention output for this stream.
            gate_msa: Gate on the attention branch.
            shift_mlp: Additive modulation for the FFN branch.
            scale_mlp: Multiplicative modulation for the FFN branch.
            gate_mlp: Gate on the FFN branch.
            context: Slot name to condition tensor for cross-attention.

        Returns:
            ``[B, L, W]`` updated residual.

        Raises:
            ValueError: If this block has no ``cross_attn`` but ``context``
                is non-empty.
        """
        x = x + gate_msa * self.self_attn.project(attn_out)
        if self.cross_attn is None:
            if context:
                raise ValueError(
                    f"WanDiTBlock has no cross_attn (cross_kv_slots is "
                    f"empty) but received context slots {sorted(context)}"
                )
        else:
            assert self.norm3 is not None
            x = x + self.cross_attn(self.norm3(x), context)
        gated = self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x + gate_mlp * gated
