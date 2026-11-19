# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 stream specs and modulation slots (Lane B).

Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.1 / §4.2:
a stream is described by one frozen dataclass whose derived quantities are
properties. Which fields may be overridden from YAML follows rule R5 in
``mot_spec_config.zh-CN.md`` §3.5: only shape and optimizer knobs that change
with a different pretrained checkpoint are YAML-visible; topology and
algorithm fields are written in ``spec/providers/<family>.py`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from ..spec.core import Spec

__all__ = [
    "LayerMapping", "StreamSpec", "ModulationSlots",
    "DEFAULT_WAN_SLOTS", "HEAD_SLOTS",
]


class LayerMapping(Enum):
    """How the layers of one stream map onto joint layer indices.

    This enum is what relaxes the older "all three num_layers must be equal"
    check. After the relaxation, streams that take part in joint attention
    only have to agree on attn_width and head_dim (validators V2 / V3);
    unequal layer counts are reconciled here.
    """

    ONE_TO_ONE = "one_to_one"   # stream layers == joint layers, every layer
    STRIDED = "strided"         # participate every `stride` layers
                                # (stream layers < joint layers)
    AT_LAYERS = "at_layers"     # only on layers listed in
                                # participating_layers


@dataclass(frozen=True)
class ModulationSlots:
    """Modulation slots addressed by name instead of by the literals 6 and 2.

    Why names matter more than counts: the existing code contains both
    ``scale, shift = ...chunk(2)`` and ``shift, scale = ...chunk(2)`` in the
    same file. The literal 2 cannot tell the two orders apart, a named slot
    table can. A swapped order keeps the shapes valid and part of the numeric
    comparison still passes, which makes it one of the hardest defects to
    find.
    """

    names: tuple[str, ...]      # slot names; the order IS the slot order
                                # inside the packed tensor

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("ModulationSlots.names must not be empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError(
                f"ModulationSlots.names has duplicates: {self.names}"
            )

    @property
    def width(self) -> int:
        """Number of slots.

        Returns:
            The slot count. Every parameter shape that multiplies by the
            number of slots reads this value instead of a literal.
        """
        return len(self.names)

    def index(self, name: str) -> int:
        """Map a slot name to its index in the packed tensor.

        Args:
            name: Slot name.

        Returns:
            Zero-based slot index.

        Raises:
            KeyError: If the name is not in this slot table. Construction
                time is therefore enough to catch a wrong slot name.
        """
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"modulation slot {name!r} does not exist; this slot table "
                f"is {list(self.names)}"
            ) from None

    def has(self, name: str) -> bool:
        """Report whether a slot exists.

        Args:
            name: Slot name.

        Returns:
            True if present. Used by branches of the form "only run this
            modulation step when the slot exists".
        """
        return name in self.names

    def unpack(self, packed: Tensor, *, slot_axis: int) -> dict[str, Tensor]:
        """Split a packed modulation tensor into {slot name: tensor}.

        Args:
            packed: Tensor whose ``slot_axis`` has length ``self.width``.
            slot_axis: Axis holding the slots. It must be passed explicitly;
                inferring it from ``ndim`` is forbidden. The pattern being
                replaced is ``has_seq = len(t_mod.shape) == 4`` followed by
                an axis choice, which silently takes the wrong branch and
                raises nothing once a shape changes.

        Returns:
            One tensor per slot name, with ``slot_axis`` removed.

        Raises:
            ValueError: If ``packed.shape[slot_axis]`` is not ``self.width``.
        """
        if packed.shape[slot_axis] != self.width:
            raise ValueError(
                f"ModulationSlots.unpack: packed.shape[{slot_axis}]="
                f"{packed.shape[slot_axis]} != slot count {self.width}"
            )
        parts = packed.unbind(dim=slot_axis)
        return dict(zip(self.names, parts, strict=True))

    def gather(self, packed: Tensor, group: tuple[str, ...], *,
               slot_axis: int) -> Tensor:
        """Select a group of slots by name and keep the packed layout.

        Why this exists: call sites such as ``.chunk(2, dim=2)`` do not mean
        "two slots", they split six slots into the attention triple and the
        FFN triple. Unpacking and re-concatenating would change both the
        meaning and the cost, so ``gather`` keeps the migration diff at those
        sites to one line.

        Args:
            packed: Tensor whose ``slot_axis`` has length ``self.width``.
            group: Slot names to select, in the desired output order.
            slot_axis: Axis holding the slots.

        Returns:
            Tensor with ``slot_axis`` reduced to ``len(group)``.
        """
        indices = torch.tensor([self.index(name) for name in group],
                               dtype=torch.long, device=packed.device)
        return packed.index_select(slot_axis, indices)


# Two preset slot sets covering all 24 + 16 literal sites in the current code
DEFAULT_WAN_SLOTS = ModulationSlots((
    "shift_msa", "scale_msa", "gate_msa",     # triple before self-attention
    "shift_mlp", "scale_mlp", "gate_mlp",     # triple before the FFN
))
HEAD_SLOTS = ModulationSlots(("shift_out", "scale_out"))   # output/action head


@dataclass(frozen=True)
class StreamSpec:
    """Every shape fact about one stream. Frozen, and the object of C1-C6.

    Separating ``residual_width`` from ``attn_width`` is the point of this
    class: five models have the two unequal (fastwam / motus / pi05 / VPP2 /
    EWAM; the VPP2 action stream with residual 512 and attn 5120 is the
    direct evidence), while the current code often packs both numbers into a
    single expression (``.unflatten(2, (6, 3072))`` hardcodes the slot count
    and the residual width at once).

    Field configurability has two tiers, and the boundary is rule R5 of
    spec_config §4.2. Fields marked **[YAML]** may be overridden from YAML
    (they change when the pretrained weights change, or they get swept
    repeatedly while tuning). The rest are written only in
    ``spec/providers/<family>.py``: they are topology and algorithm, so a
    different value means a different model.

    Attributes:
        name: Stream name. Used in three places at once: the ``stream``
            label value in TokenLayout, the capture hook key, and the
            optimizer parameter group.
        impl: Spec for this stream's block implementation.
            ``WorldModel.__init__`` calls
            ``build(impl, spec=self, geometry=...)`` and the product must
            satisfy the StreamModule protocol (§4.3). Required, no default,
            and never YAML-visible.
        residual_width: [YAML] Residual stream width, i.e. the block's input
            and output width.
        num_heads: [YAML] Number of attention heads.
        head_dim: [YAML] Per-head width. Must be equal across joint streams
            (validator V3).
        num_layers: [YAML] Layer count of this stream alone, independent of
            the joint layer count.
        ffn_width: [YAML] FFN intermediate width. None means
            4 x residual_width. Only absolute values are accepted: the
            lingbot_va value 14336 is 3.5 x 4096, which no ratio field could
            express cleanly.
        layer_mapping: How local layers map onto joint layer indices.
        participating_layers: Joint layer indices, AT_LAYERS only.
        stride: Constant layer stride, STRIDED only.
        modulation_slots: Slot table for this stream. The slot set is
            algorithm: six slots for the WAN family, two for an action head.
            Changing the set changes the forward formula.
        cross_kv_slots: Cross-attention slot names inside this stream's
            block. Empty means no cross-attention. Slot names line up with
            ConditionBus condition names (§4.4); this is where "the Nth
            condition input" stops being conflated with "the Nth output
            head".
        needs_timestep: Whether this stream consumes a timestep. False means
            a condition-only stream such as VL.
        timestep_embed_width: Input width of the sinusoidal timestep
            embedding (``freq_dim=256`` in the current code). It is the input
            width of the ``time_embedding`` Linear and is unrelated to
            residual_width, so it cannot be derived from it. It is strictly
            equivalent to ``needs_timestep`` (see ``__post_init__``): the
            EWAM vl stream has needs_timestep=False, so this is None, which
            already states "this stream has no time_embedding weights"
            without a separate convention in the provider.
        writable: False means a read-only stream: it has incoming edges but
            no outgoing ones, and its residual is never updated.
        source_key: Batch key holding this stream's input tensor. Empty means
            use ``name``. This is the stream-side dual of
            ``ConditionSpec.source_key``: just as a condition declares which
            batch key feeds it, so does a stream. In the current code the
            same thing appears as hardcoded ``batch["latents"]`` lookups
            inside ``training_loss``, so a new dataset forces a model code
            change. Topology and data contract, not YAML-visible.
        param_prefix: state_dict prefix. Empty means use ``name``. INV-3:
            when migrating an existing checkpoint this must be set to the
            prefix that checkpoint already uses.
        lr_scale: [YAML] Learning-rate scale for this stream's parameters.
    """

    # ---- identity ----
    name: str
    # ---- implementation ----
    impl: Spec
    # ---- shapes ----
    residual_width: int
    num_heads: int
    head_dim: int
    num_layers: int
    ffn_width: int | None = None
    # ---- layer mapping ----
    layer_mapping: LayerMapping = LayerMapping.ONE_TO_ONE
    participating_layers: tuple[int, ...] | None = None
    stride: int | None = None
    # All three are topology: the EWAM VL stream taking part in exactly 28
    # joint layers is part of the model definition, and another layer set
    # would be another model, so none of them is YAML-visible.
    # ---- modulation ----
    modulation_slots: ModulationSlots = DEFAULT_WAN_SLOTS
    cross_kv_slots: tuple[str, ...] = ()
    # ---- behaviour ----
    needs_timestep: bool = True
    timestep_embed_width: int | None = None
    writable: bool = True
    source_key: str = ""
    # ---- parameter governance ----
    param_prefix: str = ""
    lr_scale: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("residual_width", "num_heads", "head_dim", "num_layers"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(
                    f"StreamSpec({self.name!r}).{field_name} must be positive,"
                    f" got {value}"
                )
        if self.ffn_width is not None and self.ffn_width <= 0:
            raise ValueError(
                f"StreamSpec({self.name!r}).ffn_width must be positive or "
                f"None, got {self.ffn_width}"
            )
        if self.layer_mapping is LayerMapping.AT_LAYERS:
            if not self.participating_layers:
                raise ValueError(
                    f"StreamSpec({self.name!r}) has layer_mapping=AT_LAYERS "
                    "and must give a non-empty participating_layers"
                )
            if len(set(self.participating_layers)) \
                    != len(self.participating_layers):
                raise ValueError(
                    f"StreamSpec({self.name!r}).participating_layers has "
                    f"duplicates: {self.participating_layers}"
                )
            if len(self.participating_layers) != self.num_layers:
                raise ValueError(
                    f"StreamSpec({self.name!r}): participating_layers has "
                    f"{len(self.participating_layers)} entries but "
                    f"num_layers={self.num_layers}; the two must be equal"
                )
        elif self.participating_layers is not None:
            raise ValueError(
                f"StreamSpec({self.name!r}).participating_layers is only "
                f"meaningful with layer_mapping=AT_LAYERS, currently "
                f"{self.layer_mapping}"
            )
        if self.layer_mapping is LayerMapping.STRIDED:
            if self.stride is None or self.stride <= 0:
                raise ValueError(
                    f"StreamSpec({self.name!r}) has layer_mapping=STRIDED "
                    f"and must give a positive stride, got {self.stride}"
                )
        elif self.stride is not None:
            raise ValueError(
                f"StreamSpec({self.name!r}).stride is only meaningful with "
                f"layer_mapping=STRIDED, currently {self.layer_mapping}"
            )
        if self.lr_scale <= 0:
            raise ValueError(
                f"StreamSpec({self.name!r}).lr_scale must be positive, got "
                f"{self.lr_scale}"
            )
        if len(set(self.cross_kv_slots)) != len(self.cross_kv_slots):
            raise ValueError(
                f"StreamSpec({self.name!r}).cross_kv_slots has duplicates: "
                f"{self.cross_kv_slots}"
            )
        # The two timestep declarations must agree. Both one-sided states are
        # hard to diagnose: needs_timestep=True with a missing width cannot
        # build time_embedding at all, while needs_timestep=False with a
        # width builds a weight set that is never called yet still enters
        # state_dict, occupies memory, and makes a strict load look correct.
        if self.needs_timestep and self.timestep_embed_width is None:
            raise ValueError(
                f"StreamSpec({self.name!r}): needs_timestep=True requires "
                "timestep_embed_width, the input width of time_embedding; it "
                "cannot be derived from residual_width"
            )
        if not self.needs_timestep and self.timestep_embed_width is not None:
            raise ValueError(
                f"StreamSpec({self.name!r}): needs_timestep=False but "
                f"timestep_embed_width={self.timestep_embed_width} was given; "
                "that builds a weight set that is never called yet still "
                "enters state_dict"
            )
        if self.timestep_embed_width is not None \
                and self.timestep_embed_width <= 0:
            raise ValueError(
                f"StreamSpec({self.name!r}).timestep_embed_width must be "
                f"positive, got {self.timestep_embed_width}"
            )

    # ---- Derived quantities are always properties, never fields
    # ---- (master plan §1 rule 3) ----

    @property
    def attn_width(self) -> int:
        """Attention width, equal to num_heads x head_dim.

        Returns:
            The width seen by QKV projections. This is NOT the same number as
            residual_width; five models have the two unequal. The current
            code treating them as one number is the direct reason this class
            exists.
        """
        return self.num_heads * self.head_dim

    @property
    def mlp_width(self) -> int:
        """FFN intermediate width.

        Returns:
            ``ffn_width`` when set, otherwise 4 x residual_width. This is the
            only place the 4x default is written.
        """
        if self.ffn_width is not None:
            return self.ffn_width
        return 4 * self.residual_width

    @property
    def state_dict_prefix(self) -> str:
        """Prefix for this stream's entries in state_dict.

        Returns:
            ``param_prefix``, or the stream name when it is empty (INV-3).
        """
        return self.param_prefix or self.name

    @property
    def batch_key(self) -> str:
        """Batch key holding this stream's input tensor.

        Returns:
            ``source_key``, or the stream name when it is empty. Same idiom
            as ``state_dict_prefix``: leaving the declaration empty means
            "same as the stream name", so a provider writes a line only when
            the two differ.
        """
        return self.source_key or self.name

    def participates_in_joint(self, layer_id: int) -> bool:
        """Report whether this stream joins attention at a joint layer.

        All three LayerMapping cases are decided in this one method. The
        engine must not branch on ``layer_mapping`` itself, otherwise the
        STRIDED boundary condition would have two implementations.

        Args:
            layer_id: Joint layer index.

        Returns:
            True if this stream contributes QKV at that joint layer.

        Raises:
            ValueError: If ``layer_id`` is negative.
        """
        if layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {layer_id}")
        match self.layer_mapping:
            case LayerMapping.ONE_TO_ONE:
                return layer_id < self.num_layers
            case LayerMapping.STRIDED:
                assert self.stride is not None    # guaranteed by __post_init__
                if layer_id % self.stride != 0:
                    return False
                return layer_id // self.stride < self.num_layers
            case LayerMapping.AT_LAYERS:
                assert self.participating_layers is not None
                return layer_id in self.participating_layers

    def resolve_layer_index_map(self,
                               group_num_layers: int) -> dict[int, int]:
        """Build the joint-layer-index to local-layer-index map.

        Resolved once at construction time; the forward pass only looks the
        table up. Why construction time: the current code recomputes "do I
        take part in this layer" on every layer, and that decision depends
        only on the spec, not on data. Computing it in forward would bake a
        constant into the graph and would also expose a Python branch to CUDA
        graph capture (INV-2).

        Args:
            group_num_layers: Number of joint layers in the attention group.

        Returns:
            A dict mapping joint layer index to this stream's local layer
            index, containing only the joint layers this stream takes part
            in. Layers it skips are absent from the key set.

        Raises:
            ValueError: If ``group_num_layers`` is not positive, or if the
                number of resolved participating layers differs from
                ``num_layers``. A mismatch means some layer gets no
                parameters, or some parameters are never called.
        """
        if group_num_layers <= 0:
            raise ValueError(
                f"group_num_layers must be positive, got {group_num_layers}"
            )
        index_map: dict[int, int] = {}
        local = 0
        for layer_id in range(group_num_layers):
            if self.participates_in_joint(layer_id):
                index_map[layer_id] = local
                local += 1
        if local != self.num_layers:
            raise ValueError(
                f"StreamSpec({self.name!r}): layer_mapping="
                f"{self.layer_mapping.value} resolves to {local} "
                f"participating layers under a joint layer count of "
                f"{group_num_layers}, but num_layers={self.num_layers}. The "
                "two must be equal; a difference means some layer gets no "
                "parameters, or some parameters are never called"
            )
        return index_map
