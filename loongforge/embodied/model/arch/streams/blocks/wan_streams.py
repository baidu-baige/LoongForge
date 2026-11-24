# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``StreamModule`` implementations for the two WAN-family streams (Lane B, L2).

``WanDiTStream`` owns all parameters of one stream and implements the
four-stage protocol. ``WanVideoStream`` and ``ActionStream`` override only two
things, the **entry** and the **positional encoding**: the first uses a
``Conv3d`` patchify plus 3-D RoPE, the second a single ``Linear`` plus 1-D
RoPE. The in-layer algorithm of both streams is identical
(``WanDiTBlock``), which is why they can be packed into the same joint
attention, and that is the whole of MoT's "two parameter sets, one attention".

Every shape fact is read from the spec; this module contains no magic
numbers:
  - residual width / heads / head width / layer count / FFN width /
    modulation slots / cross-attention slots come from ``StreamSpec``;
  - entry channel count comes from
    ``geometry.latent_channels + world.concat_channels(name)``;
  - condition projector input width comes from ``world.condition(slot).width``;
  - token count comes from ``geometry.padded_seq_len`` /
    ``geometry.action_horizon``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from ...geometry import GeometrySpec
from ..conditions import ConditionBus, InjectionKind
from ..spec import StreamSpec
from ..state import QKV, StreamState
from .wan_dit import (
    WanDiTBlock,
    assemble_freqs_3d,
    rope_freqs_1d,
    rope_freqs_3d,
    sinusoidal_embedding,
)

if TYPE_CHECKING:          # typing only; not imported at runtime so that
                           # spec and streams do not form an import cycle
    from ...spec.world import WorldModelSpec

__all__ = ["ActionStream", "WanDiTStream", "WanVideoStream",
           "build_context_projector"]


class _MLPProj(nn.Module):
    """``LayerNorm -> Linear -> GELU -> Linear -> LayerNorm``, held in ``proj``.

    The submodule names stay ``proj.{0,1,3,4}``: the existing CLIP projector
    (``MLPProj`` in ``wan_video_dit.py:301-311``) has exactly this shape, and
    renaming would stop checkpoints from matching.
    """

    def __init__(self, in_width: int, out_width: int) -> None:
        """Build the five-stage projector.

        Args:
            in_width: Input width.
            out_width: Output width.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_width),
            nn.Linear(in_width, in_width),
            nn.GELU(),
            nn.Linear(in_width, out_width),
            nn.LayerNorm(out_width),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


def build_context_projector(form: str, *, in_width: int,
                            out_width: int) -> nn.Module:
    """Build a condition projector: condition width to stream residual width.

    Both shapes come from the existing code and are **not** interchangeable:
      - ``"mlp2"`` = ``Linear -> GELU(tanh) -> Linear``, the existing
        ``text_embedding`` (``wan_video_dit.py:373-377``);
      - ``"mlp_proj"`` = the five-stage form with two LayerNorms, the
        existing ``img_emb`` (``:398``, the CLIP path).

    The shape name is a **pure implementation detail** carried by
    ``Spec.params`` (key ``context_proj``) and appears in no spec field: it
    changes no externally visible shape, only how these few layers are
    arranged.

    Args:
        form: ``"mlp2"`` or ``"mlp_proj"``.
        in_width: Condition width before projection.
        out_width: Stream residual width.

    Returns:
        The projector module.

    Raises:
        ValueError: If ``form`` is not one of the two known names.
    """
    if form == "mlp2":
        return nn.Sequential(
            nn.Linear(in_width, out_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(out_width, out_width),
        )
    if form == "mlp_proj":
        return _MLPProj(in_width, out_width)
    raise ValueError(
        f"build_context_projector: unknown projector shape {form!r}; "
        "valid names are 'mlp2' (existing text_embedding) and "
        "'mlp_proj' (existing img_emb)"
    )


def _condition_width(world: WorldModelSpec, slot: str) -> int:
    """Return a condition's declared width, raising if it has none.

    ``ConditionSpec.width is None`` means "this condition enters no weight
    shape", yet the projector of a cross-attention slot needs exactly that
    width. When the two contradict each other it must fail at construction
    time; otherwise the width has to be inferred from the first batch during
    forward, which is what the existing engine does.

    Args:
        world: The world model spec holding the condition table.
        slot: Condition name.

    Returns:
        The condition width in features.

    Raises:
        ValueError: If the condition declares no ``width``.
    """
    width = world.condition(slot).width
    if width is None:
        raise ValueError(
            f"condition {slot!r} declares no width but it is a cross-attn "
            "slot of some stream; the cross-attn condition projector needs "
            "that width to build its Linear"
        )
    return width


class WanDiTStream(nn.Module):
    """All parameters of one WAN-family stream plus the four-stage protocol.

    The two subclasses override only the entry and the positional encoding.

    Parameter list (submodule names match the existing code, so a checkpoint
    only needs a prefix move):
      ``context_embedding.<slot>``  one condition projector per cross-attn slot
      ``time_embedding``            sinusoidal embedding to residual width
                                    (only when ``needs_timestep``)
      ``time_projection``           residual width to slots * residual width
                                    (same condition)
      ``blocks.<i>``                one ``WanDiTBlock`` per layer
    The entry (``patch_embedding`` / ``action_encoder``) is built in the
    subclass; it is the one parameter group where the two streams differ.

    ``world`` is required: condition widths (``ConditionSpec.width``) and the
    channel-concat channel count live only in ``WorldModelSpec``, and letting
    a stream walk the condition table itself would create a second
    implementation of "slot name to width".

    Attributes:
        spec: The stream spec this module was built from.
        geometry: Shared geometry, the single source of static shapes.
        world: The world model spec holding conditions.
    """

    def __init__(self, *, spec: StreamSpec, geometry: GeometrySpec,
                 world: WorldModelSpec, context_proj: Mapping[str, str] | None = None,
                 eps: float = 1e-6, qk_norm: bool = True,
                 cross_attn_norm: bool = True) -> None:
        """Build the per-stream parameters from the spec.

        Args:
            spec: Stream spec: widths, layer count, modulation and
                cross-attention slots.
            geometry: Shared geometry.
            world: World model spec holding the condition table.
            context_proj: Optional slot name to projector shape name; slots
                not listed default to ``"mlp2"``.
            eps: Epsilon for the norms inside each block.
            qk_norm: Whether to use RMSNorm on q and k.
            cross_attn_norm: Whether ``norm3`` is affine.

        Raises:
            ValueError: If ``context_proj`` names a slot that is not in
                ``spec.cross_kv_slots``.
        """
        super().__init__()
        self.spec = spec
        self.geometry = geometry
        self.world = world
        width = spec.residual_width
        forms = dict(context_proj or {})
        unknown = sorted(set(forms) - set(spec.cross_kv_slots))
        if unknown:
            raise ValueError(
                f"stream {spec.name!r}: params['context_proj'] gives "
                f"projector shapes for slots {unknown}, which are absent "
                f"from this stream's cross_kv_slots="
                f"{list(spec.cross_kv_slots)}"
            )
        self.context_embedding = nn.ModuleDict({
            slot: build_context_projector(
                forms.get(slot, "mlp2"),
                in_width=_condition_width(world, slot),
                out_width=width)
            for slot in spec.cross_kv_slots
        })
        if spec.needs_timestep:
            embed_width = spec.timestep_embed_width
            assert embed_width is not None      # guaranteed by __post_init__
            self.time_embedding = nn.Sequential(
                nn.Linear(embed_width, width), nn.SiLU(),
                nn.Linear(width, width))
            self.time_projection = nn.Sequential(
                nn.SiLU(),
                nn.Linear(width, width * spec.modulation_slots.width))
        else:
            self.time_embedding = None
            self.time_projection = None
        self.blocks = nn.ModuleList([
            WanDiTBlock(
                residual_width=width, num_heads=spec.num_heads,
                head_dim=spec.head_dim, ffn_width=spec.mlp_width,
                num_modulation_slots=spec.modulation_slots.width,
                cross_kv_slots=spec.cross_kv_slots, eps=eps,
                qk_norm=qk_norm, cross_attn_norm=cross_attn_norm)
            for _ in range(spec.num_layers)
        ])
        # Per-token RoPE frequency table, ``[L, 1, head_dim/2]`` complex;
        # None means no rotation is applied. Subclasses fill it in their own
        # __init__. It is deliberately **not** a buffer: a complex tensor
        # cannot survive ``Module.to(dtype=bf16)``, and the existing
        # checkpoints have no such key, so registering a buffer would add an
        # unexpected key to a strict load.
        self._freqs: Tensor | None = None

    # ---- the two things a subclass must provide ----

    def _embed(self, x: Tensor, bus: ConditionBus | None) -> Tensor:
        """Turn the raw input tensor into ``[B, L, residual_width]``.

        The entry parameters live in the subclass.

        Args:
            x: Raw stream input.
            bus: Condition bus, or None when the stream needs no conditions.

        Returns:
            ``[B, L, residual_width]`` tokens.

        Raises:
            NotImplementedError: Always; subclasses override this.
        """
        raise NotImplementedError

    # ---- the four-stage protocol ----

    def prepare(self, x: Tensor, *, timestep: Tensor | None = None,
                conditions: ConditionBus | None = None) -> StreamState:
        """Tokenize, embed the timestep, and project the conditions in one go.

        Cross-attention conditions are projected **once** here and stored in
        ``extras["cross_kv"]``. The existing code re-projects them in every
        layer even though there is only one projector, which is pure waste.

        Args:
            x: Raw stream input for ``_embed``.
            timestep: ``[B]`` timesteps; required iff ``needs_timestep``. A
                stream declaring ``token_timestep`` also accepts the
                per-frame form its ``_expand_timestep`` documents.
            conditions: Condition bus; required if the stream declares
                cross-attention or channel-concat conditions.

        Returns:
            A ``StreamState`` with hidden tokens, optional modulation, and
            ``extras`` holding ``cross_kv`` and, when applicable,
            ``t_embed``. Without ``token_timestep`` the modulation is
            ``[B, slots, W]`` and ``t_embed`` is ``[B, W]``; with it they
            gain a token axis and become ``[B, L, slots, W]`` and
            ``[B, L, W]``.

        Raises:
            ValueError: If the entry output width differs from
                ``residual_width``, if ``timestep`` is passed to a stream
                with ``needs_timestep=False``, or if it is missing for a
                stream with ``needs_timestep=True``.
        """
        tokens = self._embed(x, conditions)
        if tokens.shape[-1] != self.spec.residual_width:
            raise ValueError(
                f"stream {self.spec.name!r}: the entry produced width "
                f"{tokens.shape[-1]} but "
                f"residual_width={self.spec.residual_width}"
            )
        context, context_key_valid = self._embed_context(conditions)
        extras: dict[str, Any] = {"cross_kv": context}
        if context_key_valid:
            extras["cross_kv_mask"] = context_key_valid
        modulation: Tensor | None = None
        if self.time_embedding is None:
            if timestep is not None:
                raise ValueError(
                    f"stream {self.spec.name!r} has needs_timestep=False "
                    "but received a timestep; it has no time_embedding to "
                    "consume it"
                )
        else:
            if timestep is None:
                raise ValueError(
                    f"stream {self.spec.name!r} has needs_timestep=True but "
                    "prepare received no timestep (V5 should have caught "
                    "this at construction time)"
                )
            assert self.spec.timestep_embed_width is not None
            assert self.time_projection is not None
            values = self._expand_timestep(timestep)
            # As master's pre_dit: the embedding follows t's dtype, then the
            # weights' (t stays float32 under RngPolicy(sigma_dtype="float32")).
            time_input = sinusoidal_embedding(
                self.spec.timestep_embed_width, values.reshape(-1))
            t_embed = self.time_embedding(
                time_input.to(dtype=self.time_embedding[0].weight.dtype))
            if self.spec.token_timestep is not None:
                t_embed = t_embed.unflatten(0, tuple(values.shape))
            extras["t_embed"] = t_embed
            modulation = self.time_projection(t_embed).unflatten(
                -1, (self.spec.modulation_slots.width, self.spec.residual_width))
        return StreamState(stream=self.spec.name, hidden=tokens,
                           modulation=modulation, extras=extras)

    def pre_attn(self, state: StreamState, *, local_layer_id: int) -> QKV:
        block = self.blocks[local_layer_id]
        slots = self._modulation(block, state)
        return block.attention_io(state.hidden,
                                  shift_msa=slots["shift_msa"],
                                  scale_msa=slots["scale_msa"],
                                  freqs=self._freqs)

    def post_attn(self, state: StreamState, attn_out: Tensor, *,
                  local_layer_id: int) -> StreamState:
        block = self.blocks[local_layer_id]
        slots = self._modulation(block, state)
        hidden = block.post_attention(
            state.hidden, attn_out,
            gate_msa=slots["gate_msa"], shift_mlp=slots["shift_mlp"],
            scale_mlp=slots["scale_mlp"], gate_mlp=slots["gate_mlp"],
            context=state.extras.get("cross_kv", {}),
            context_key_valid=state.extras.get("cross_kv_mask"))
        return state.replace_hidden(hidden)

    def finalize(self, state: StreamState) -> StreamState:
        """Return the state unchanged.

        The final norm of the WAN family lives in the output head (L4), not
        in the stream.

        Args:
            state: The state after the last layer.

        Returns:
            The same state object.
        """
        return state

    # ---- internals ----

    def _expand_timestep(self, timestep: Tensor) -> Tensor:
        """Return the timestep values the modulation is built from.

        The base behaviour is the per-sample one: ``[B]`` passes through
        untouched. Per-token modulation needs a token axis to broadcast
        along, which only a concrete stream geometry can supply, so the
        stream implementation that has one overrides this.

        Args:
            timestep: The timestep as ``prepare`` received it.

        Returns:
            The values to embed, of any shape; ``prepare`` restores that
            shape after the flattened sinusoidal embedding.

        Raises:
            ValueError: If the spec declares ``token_timestep`` but this
                implementation only supports the per-sample form.
        """
        if self.spec.token_timestep is None:
            return timestep
        raise ValueError(
            f"stream {self.spec.name!r} declares token_timestep but its "
            f"implementation {type(self).__name__} supports only per-sample "
            "timestep modulation; per-token modulation needs an axis to "
            "broadcast the timestep along"
        )

    def _embed_context(
            self, bus: ConditionBus | None
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Fetch each cross-attn condition by slot name and project it once.

        An empty slot set returns two empty dicts without touching the bus.

        ``APPEND_TO_SLOT`` conditions are concatenated onto the tail of the
        host slot along the token axis (``torch.cat(..., dim=1)``) **before**
        projection, so they share the host's single ``k/v/norm_k`` group.
        Concatenating after projection would need one projection per piece,
        while the existing code has only one projector (the proprio tensor in
        ``vpp2.py:520-539`` is concatenated ahead of ``text_embedding``).

        Concatenation order comes from ``WorldModelSpec.appended_to``
        (declaration order), not from dictionary order or bus insertion
        order.

        The key padding masks are concatenated in that same order, so they
        cannot drift out of step with the context. A slot whose pieces all
        carry ``key_valid=None`` is **left out** of the mask dict rather than
        given a mask of all True: the maskless attention call is the one the
        existing code makes, and that is what keeps the no-padding case
        bit-identical.

        Args:
            bus: Condition bus, or None.

        Returns:
            A pair: slot name to ``[B, Lc, residual_width]`` projected
            condition, and slot name to ``[B, Lc]`` bool key padding mask
            holding only the slots that have one.

        Raises:
            ValueError: If the stream declares cross-attention slots but no
                bus was passed to ``prepare``.
        """
        if not self.spec.cross_kv_slots:
            return {}, {}
        if bus is None:
            raise ValueError(
                f"stream {self.spec.name!r} declares cross_kv_slots "
                f"{list(self.spec.cross_kv_slots)} but prepare received no "
                "condition bus"
            )
        out: dict[str, Tensor] = {}
        masks: dict[str, Tensor] = {}
        for slot in self.spec.cross_kv_slots:
            pieces = [bus.require(slot)]
            pieces.extend(bus.require(extra) for extra
                          in self.world.appended_to(slot, self.spec.name))
            values = [piece.value for piece in pieces]
            raw = values[0] if len(values) == 1 else torch.cat(values, dim=1)
            out[slot] = self.context_embedding[slot](raw)
            if any(piece.key_valid is not None for piece in pieces):
                masks[slot] = torch.cat(
                    [piece.key_valid if piece.key_valid is not None
                     else piece.value.new_ones(piece.value.shape[:2],
                                               dtype=torch.bool)
                     for piece in pieces], dim=1)
        return out, masks

    def _modulation(self, block: WanDiTBlock,
                    state: StreamState) -> dict[str, Tensor]:
        """Return this layer's modulation slots, each broadcastable over L.

        Each slot is ``[B, 1, W]``, or ``[1, 1, W]`` when the stream has no
        timestep modulation, or ``[B, L, W]`` when the spec declares
        ``token_timestep``.

        ``unpack`` uses ``unbind``, which **removes** the slot axis, so the
        broadcast axis has to be added back in the per-sample case. Skipping
        that step multiplies the residual by a ``[B, W]`` tensor as if it
        were ``[B, L]``, and the shapes happen to be valid. In the per-token
        case the axis is already there and adding another one would be the
        mirror-image defect.

        Which case applies is read from the spec, never from
        ``packed.ndim``. The pattern being replaced is
        ``has_seq = len(t_mod.shape) == 4`` (``fastwam/wan/dit.py:439``),
        which silently takes the wrong branch as soon as a shape changes.

        The slot axis itself is pinned to ``-2`` and so is the same in both
        cases: the slots sit immediately left of the width in every layout.

        ``state.modulation is None`` (a condition-only stream) falls back to
        zeros, so the formula degrades to "per-layer learnable modulation
        only" without an extra branch.

        Args:
            block: The layer whose learnable modulation is used.
            state: Current stream state.

        Returns:
            Slot name to broadcastable modulation tensor.
        """
        slots = self.spec.modulation_slots
        packed = state.modulation
        if packed is None:
            packed = block.modulation.new_zeros((1, slots.width, 1))
        unpacked = slots.unpack(block.combine_modulation(packed),
                                slot_axis=-2)
        if self.spec.token_timestep is not None:
            return unpacked
        return {name: tensor.unsqueeze(1)
                for name, tensor in unpacked.items()}


def _pad_tokens(tokens: Tensor, total: int) -> Tensor:
    """Zero-pad ``[B, L, W]`` up to ``total`` tokens.

    The alignment length comes from ``GeometrySpec.padded_seq_len``, the
    single source of static shapes (INV-2). Padded positions carry
    ``is_padding=True`` in the labels and visibility masks them out, so the padding
    value does not affect the result; zero is used so their grads are
    explicitly 0 instead of NaN.

    Args:
        tokens: ``[B, L, W]`` tokens.
        total: Target token count.

    Returns:
        ``[B, total, W]``.

    Raises:
        ValueError: If ``L`` already exceeds ``total``.
    """
    length = tokens.shape[1]
    if length == total:
        return tokens
    if length > total:
        raise ValueError(
            f"the entry produced {length} tokens, more than "
            f"padded_seq_len={total}"
        )
    pad = tokens.new_zeros((tokens.shape[0], total - length, tokens.shape[2]))
    return torch.cat([tokens, pad], dim=1)


def _pad_freqs(freqs: Tensor, total: int) -> Tensor:
    """Pad the per-token RoPE table to ``total`` rows with ones.

    Ones mean the identity rotation. Padding with zeros would multiply q/k at
    the padded positions straight to 0; those positions are masked anyway,
    but the identity rotation makes "take the first L rows" and "take the
    whole table then slice" produce the same values, removing a discrepancy
    that only appears when align_to > 1.

    Args:
        freqs: ``[L, 1, D/2]`` complex table.
        total: Target row count.

    Returns:
        ``[total, 1, D/2]``, or ``freqs`` unchanged if it is long enough.
    """
    rows = freqs.shape[0]
    if rows >= total:
        return freqs
    ones = torch.ones(total - rows, *freqs.shape[1:], dtype=freqs.dtype)
    return torch.cat([freqs, ones], dim=0)


class WanVideoStream(WanDiTStream):
    """Video stream: ``Conv3d`` patchify entry plus 3-D RoPE.

    The entry channel count is **derived**: ``geometry.latent_channels``
    plus the summed width of all ``CHANNEL_CONCAT`` conditions
    (``world.concat_channels``). The existing ``in_dim=36`` is
    16 + 4 (mask) + 16 (condition-frame latent).

    Concatenation **order** follows the declaration order of the condition
    table. Changing the order changes the weight layout of the entry
    convolution, so it needs one single source rather than being left to
    dictionary order inside forward.
    """

    def __init__(self, *, spec: StreamSpec, geometry: GeometrySpec,
                 world: WorldModelSpec, rope_end: int = 1024,
                 **kwargs: Any) -> None:
        """Build the patchify entry and the 3-D RoPE table.

        Args:
            spec: Stream spec.
            geometry: Shared geometry; supplies latent channels, patch size,
                patch grid, and padded sequence length.
            world: World model spec; supplies the channel-concat conditions.
            rope_end: Number of positions tabulated per RoPE axis.
            **kwargs: Forwarded to ``WanDiTStream.__init__``.
        """
        super().__init__(spec=spec, geometry=geometry, world=world, **kwargs)
        self._concat_slots = tuple(
            cond.name for cond in world.conditions
            if cond.kind is InjectionKind.CHANNEL_CONCAT
            and spec.name in cond.targets
        )
        in_channels = geometry.latent_channels + world.concat_channels(spec.name)
        self.patch_embedding = nn.Conv3d(
            in_channels, spec.residual_width,
            kernel_size=geometry.patch_size, stride=geometry.patch_size)
        self._freqs = _pad_freqs(
            assemble_freqs_3d(rope_freqs_3d(spec.head_dim, rope_end),
                              geometry.patch_grid),
            geometry.padded_seq_len)

    def _embed(self, x: Tensor, bus: ConditionBus | None) -> Tensor:
        """Concatenate channel conditions, patchify, flatten, and pad.

        Args:
            x: ``[B, C, F, H, W]`` latent, with C = latent_channels.
            bus: Condition bus; required if this stream has channel-concat
                conditions.

        Returns:
            ``[B, padded_seq_len, residual_width]`` tokens.

        Raises:
            ValueError: If ``x`` is not 5-D, if its channel count differs
                from ``geometry.latent_channels``, or if channel-concat
                conditions are declared but no bus was passed.
        """
        if x.ndim != 5:
            raise ValueError(
                f"stream {self.spec.name!r}: the entry takes 5-D "
                f"[B, C, F, H, W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.geometry.latent_channels:
            raise ValueError(
                f"stream {self.spec.name!r}: the entry takes "
                f"{self.geometry.latent_channels} latent channels "
                f"(geometry.latent_channels), got {x.shape[1]}; "
                "channel-concat conditions are fetched from the bus by this "
                "method, do not concatenate them beforehand"
            )
        if self._concat_slots:
            if bus is None:
                raise ValueError(
                    f"stream {self.spec.name!r} has channel-concat "
                    f"conditions {list(self._concat_slots)} but prepare "
                    "received no condition bus"
                )
            x = torch.cat([x] + [bus.require(slot).value
                                 for slot in self._concat_slots], dim=1)
        tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        return _pad_tokens(tokens, self.geometry.padded_seq_len)

    def _expand_timestep(self, timestep: Tensor) -> Tensor:
        """Broadcast the timestep over the frame axis, pinning a clean prefix.

        Only runs when the spec declares ``token_timestep``; without it the
        base per-sample path is kept, so VPP2 is untouched.

        Two input forms are accepted, matching the two existing call sites:
          - ``[B]``, one sampled timestep per sample, which every noised
            frame shares (fastwam ``wan/dit.py:762-779``, InternW0-Delta
            ``wan_video_dit.py:820-823``);
          - ``[B, F - clean_prefix]``, one timestep per noised frame, which
            an L3 ``BroadcastSpec(axes=("frame",))`` produces
            (InternW0-Delta ``wan_video_dit.py:824-825``).
        The clean prefix is **never** part of the input: its value is not
        sampled but fixed at 0, so accepting it would let a caller pass a
        value that is then discarded.

        Padding rows added by ``align_to`` get timestep 0. Those positions
        are masked out of attention and cut off before the head, so the
        value is unobservable; 0 is used because it is the one value already
        meaningful in this tensor.

        Args:
            timestep: ``[B]`` or ``[B, F - clean_prefix]`` timesteps.

        Returns:
            ``[B, padded_seq_len]`` per-token timesteps.

        Raises:
            ValueError: If ``timestep`` is neither of the two accepted
                shapes, or if ``clean_prefix`` is not smaller than the latent
                frame count.
        """
        scope = self.spec.token_timestep
        if scope is None:
            return timestep
        frames = self.geometry.patch_grid[0]
        noised = frames - scope.clean_prefix
        if noised <= 0:
            raise ValueError(
                f"stream {self.spec.name!r}: token_timestep.clean_prefix="
                f"{scope.clean_prefix} leaves no noised frame out of the "
                f"{frames} latent frames of geometry.patch_grid"
            )
        batch = timestep.shape[0]
        if timestep.ndim == 1:
            per_frame = timestep[:, None].expand(batch, noised)
        elif timestep.ndim == 2 and timestep.shape[1] == noised:
            per_frame = timestep
        else:
            raise ValueError(
                f"stream {self.spec.name!r} declares token_timestep, so its "
                f"timestep must be [B] or [B, {noised}] (the noised frames, "
                f"i.e. patch_grid[0]={frames} minus clean_prefix="
                f"{scope.clean_prefix}), got {tuple(timestep.shape)}"
            )
        if scope.clean_prefix:
            per_frame = torch.cat(
                [per_frame.new_zeros((batch, scope.clean_prefix)), per_frame],
                dim=1)
        tokens = per_frame[:, :, None].expand(
            batch, frames, self.geometry.tokens_per_frame).reshape(batch, -1)
        missing = self.geometry.padded_seq_len - tokens.shape[1]
        if missing:
            tokens = torch.cat([tokens, tokens.new_zeros((batch, missing))],
                               dim=1)
        return tokens


class ActionStream(WanDiTStream):
    """Action stream: one ``Linear`` entry plus 1-D RoPE.

    The in-layer algorithm is identical to the video stream.

    The entry width is ``geometry.action_dim`` (the packed action width, 7
    for VPP2 libero) and the token count is ``geometry.action_horizon``,
    which is the same source the layout uses for this stream's segment
    length (``layout._resolve_token_counts``), so the two cannot drift
    apart.
    """

    def __init__(self, *, spec: StreamSpec, geometry: GeometrySpec,
                 world: WorldModelSpec, rope_end: int = 1024,
                 **kwargs: Any) -> None:
        """Build the Linear entry and the 1-D RoPE table.

        Args:
            spec: Stream spec.
            geometry: Shared geometry; supplies action_dim and
                action_horizon.
            world: World model spec.
            rope_end: Number of positions tabulated for RoPE.
            **kwargs: Forwarded to ``WanDiTStream.__init__``.

        Raises:
            ValueError: If ``geometry.action_dim`` is not positive, leaving
                the entry Linear without an input width.
        """
        super().__init__(spec=spec, geometry=geometry, world=world, **kwargs)
        if geometry.action_dim <= 0:
            raise ValueError(
                f"stream {spec.name!r} is an action stream but "
                f"geometry.action_dim={geometry.action_dim}; the entry "
                "Linear has no input width to use"
            )
        self.action_encoder = nn.Linear(geometry.action_dim,
                                        spec.residual_width)
        self._freqs = rope_freqs_1d(spec.head_dim, rope_end).unsqueeze(1)

    def _embed(self, x: Tensor, bus: ConditionBus | None) -> Tensor:
        """Project packed actions to the residual width.

        Args:
            x: ``[B, action_horizon, action_dim]`` actions.
            bus: Unused; the action stream has no channel-concat conditions.

        Returns:
            ``[B, action_horizon, residual_width]`` tokens.

        Raises:
            ValueError: If ``x`` is not 3-D, or if its step count or width
                differ from ``geometry.action_horizon`` /
                ``geometry.action_dim``.
        """
        del bus                     # the action stream has no concat conditions
        if x.ndim != 3:
            raise ValueError(
                f"stream {self.spec.name!r}: the entry takes 3-D "
                f"[B, T, action_dim], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.geometry.action_horizon:
            raise ValueError(
                f"stream {self.spec.name!r}: the entry takes "
                f"{self.geometry.action_horizon} steps "
                f"(geometry.action_horizon), got {x.shape[1]}"
            )
        if x.shape[2] != self.geometry.action_dim:
            raise ValueError(
                f"stream {self.spec.name!r}: the entry takes width "
                f"{self.geometry.action_dim} (geometry.action_dim), "
                f"got {x.shape[2]}"
            )
        return self.action_encoder(x)
