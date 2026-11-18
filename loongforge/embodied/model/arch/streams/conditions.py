# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""L2 condition injection: ``InjectionKind`` / ``Condition`` / ``ConditionBus``.

Lane B. Signatures follow ``docs/design/mot_layer_interfaces.zh-CN.md`` §4.4;
``ConditionSpec`` configurability follows ``mot_spec_config.zh-CN.md`` §3.5,
where only ``enabled`` and ``dropout_prob`` are YAML-visible.

This module closes one of the gaps shared by all three surveyed frameworks:
"the Nth condition input" and "the Nth output head" were treated as the same
index. Conditions travel on this bus, heads live in ``heads/``, and the two
never meet.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

from torch import Tensor

from ..spec.core import Spec

__all__ = ["InjectionKind", "Condition", "ConditionBus", "ConditionSpec"]


class InjectionKind(Enum):
    """How a condition enters the model.

    The five ways are physically different operations, not parameters of one
    operation.
    """

    CROSS_ATTENTION = "cross_attention"   # used as cross-attention KV
    ADD_TO_MODULATION = "add_to_modulation"   # added to the modulation
                                              # vector, timestep style
    CONCAT_TOKENS = "concat_tokens"       # extra tokens in the joint sequence
    CHANNEL_CONCAT = "channel_concat"     # concatenated onto the residual
                                          # channel axis, once at the entry
    APPEND_TO_SLOT = "append_to_slot"     # appended to an existing
                                          # cross-attention slot
    # The fifth differs from the first in parameters, not position:
    # APPEND_TO_SLOT creates no new k/v/norm_k. It concatenates along the
    # token axis BEFORE the host slot's condition projector, so it shares the
    # host's KV weights. VPP2 proprio is this kind (``vpp2.py:520-539``: the
    # output of one ``Linear(proprio_dim, text_dim)`` is cat'ed onto the tail
    # of the text context). Forcing it into CROSS_ATTENTION would add
    # 3 x num_layers tensors that do not exist in the checkpoint.


@dataclass(frozen=True)
class Condition:
    """One materialized condition.

    Attributes:
        name: Key under which it sits on the bus.
        kind: Injection mechanism.
        value: The condition tensor.
        targets: Set of stream names it is injected into. One condition may
            feed several streams or just one. An empty set is rejected
            because a condition with no target is dead configuration, which
            in the current code can only be found by reading the code.
    """

    name: str
    kind: InjectionKind
    value: Tensor
    targets: frozenset[str]

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(
                f"Condition({self.name!r}).targets must not be empty: a "
                "condition with no injection target is dead configuration"
            )

    def goes_to(self, stream: str) -> bool:
        """Report whether this condition is injected into ``stream``."""
        return stream in self.targets


@dataclass
class ConditionBus:
    """The condition bus, populated by step 2 of ``WorldModel.forward``.

    Stream modules then pick conditions off it by name.

    Why a bus instead of several ``forward`` parameters: the number of
    conditions differs per model (two for VPP2, three for EWAM, zero for a
    monolithic model) and a parameter list is part of the signature. With a
    bus, adding a condition changes no signature.

    Attributes:
        conditions: Mapping from condition name to ``Condition``.
    """

    conditions: dict[str, Condition] = field(default_factory=dict)

    def add(self, condition: Condition) -> None:
        """Register one condition.

        Args:
            condition: The condition to register.

        Raises:
            ValueError: If the name is already registered. Overwriting by
                name would be silent data loss, so one name has one source.
        """
        if condition.name in self.conditions:
            raise ValueError(
                f"ConditionBus already has condition {condition.name!r}; "
                "same-name conditions must not overwrite (one name, one "
                "source)"
            )
        self.conditions[condition.name] = condition

    def for_stream(self, stream: str) -> dict[str, Condition]:
        """Collect every condition injected into one stream.

        Args:
            stream: Stream name.

        Returns:
            Name to condition mapping, empty when none apply. An empty dict
            is the normal case for a monolithic model with no conditions: the
            stream module's "iterate over my conditions" loop is then an
            empty loop and needs no branch.
        """
        return {name: cond for name, cond in self.conditions.items()
                if cond.goes_to(stream)}

    def of_kind(self, stream: str, kind: InjectionKind) -> dict[str, Condition]:
        """Filter one stream's conditions by injection mechanism.

        Args:
            stream: Stream name.
            kind: Injection mechanism to keep.

        Returns:
            Name to condition mapping. Each of the stream module's injection
            steps reads only its own kind.
        """
        return {name: cond for name, cond in self.for_stream(stream).items()
                if cond.kind is kind}

    def declared_names(self) -> frozenset[str]:
        """Return the set of registered condition names.

        Returns:
            Names, used by validator V7 to reconcile against each stream's
            ``cross_kv_slots``.
        """
        return frozenset(self.conditions)

    def require(self, name: str) -> Condition:
        """Fetch a condition by name.

        Args:
            name: Condition name.

        Returns:
            The registered condition.

        Raises:
            KeyError: If absent; the message lists every registered name.
        """
        try:
            return self.conditions[name]
        except KeyError:
            raise KeyError(
                f"ConditionBus has no condition {name!r}; "
                f"registered: {sorted(self.conditions)}"
            ) from None

    def __len__(self) -> int:
        """Return the number of registered conditions."""
        return len(self.conditions)

    def __iter__(self) -> Iterator[Condition]:
        """Iterate over the registered conditions in insertion order."""
        return iter(self.conditions.values())


@dataclass(frozen=True)
class ConditionSpec:
    """Declaration of one condition, resolved at construction time.

    During ``forward`` it produces a ``Condition``. Per spec_config §3.5,
    ``enabled`` and ``dropout_prob`` are [YAML]-overridable because they get
    swept while tuning; the remaining fields are topology.

    Attributes:
        name: Condition name, and the bus key.
        kind: Injection mechanism.
        targets: Target stream names. Topology, not YAML-visible.
        width: Last-axis width of this condition's value. It is the only
            input the consumer side has for building weights:
            CROSS_ATTENTION means the input width of the target stream's
            ``text_embedding`` Linear (4096 for VPP2); CHANNEL_CONCAT means
            the channel count appended to the channel axis (4 for the VPP2
            mask and 16 for the conditioning-frame latent, which added to
            ``geometry.latent_channels`` gives the patchify ``in_dim=36``);
            APPEND_TO_SLOT must equal the host slot's width because both go
            through the same projector. None means this condition enters no
            weight shape, as for ADD_TO_MODULATION which is a plain add.
            Topology and shape, not YAML-visible.
        encoder: Spec of the module producing the value. None means the
            batch supplies the value directly.
        source_key: Batch key holding the raw input. It always names the raw
            source regardless of ``encoder``: when ``encoder`` is not None,
            ``value = encoder(batch[source_key])``.
        source_width: [YAML] Last-axis width of the raw input in the batch,
            i.e. the input width of ``encoder``. Meaningful only when an
            encoder is present, and it is a different number from ``width``:
            VPP2 proprio is 8 (libero) or 16 (robodojo) and only becomes the
            text slot's 4096 after encoding. It changes with the dataset,
            which is why this one field is configurable.
        append_to: APPEND_TO_SLOT only: which slot to append to. The slot
            name is the host's ``ConditionSpec.name``.
        enabled: [YAML] Set False to switch this condition off.
        dropout_prob: [YAML] Condition dropout probability used for CFG.
    """

    name: str
    kind: InjectionKind
    targets: tuple[str, ...]
    width: int | None = None
    encoder: Spec | None = None
    source_key: str | None = None
    source_width: int | None = None
    append_to: str | None = None
    enabled: bool = True
    dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(
                f"ConditionSpec({self.name!r}).targets must not be empty"
            )
        for field_name in ("width", "source_width"):
            value = getattr(self, field_name)
            if value is not None and value <= 0:
                raise ValueError(
                    f"ConditionSpec({self.name!r}).{field_name} must be "
                    f"positive or None, got {value}"
                )
        if self.kind is InjectionKind.CHANNEL_CONCAT and self.width is None:
            raise ValueError(
                f"ConditionSpec({self.name!r}): kind=CHANNEL_CONCAT requires "
                "width, the addend to the patchify in_channels; without it "
                "the stream's entry convolution cannot be built"
            )
        if (self.append_to is not None) \
                is not (self.kind is InjectionKind.APPEND_TO_SLOT):
            raise ValueError(
                f"ConditionSpec({self.name!r}): append_to and "
                "kind=APPEND_TO_SLOT must be set together or not at all, got "
                f"append_to={self.append_to!r}, kind={self.kind.value!r}"
            )
        if self.append_to == self.name:
            raise ValueError(
                f"ConditionSpec({self.name!r}).append_to points at itself"
            )
        if self.kind is InjectionKind.APPEND_TO_SLOT and self.width is None:
            raise ValueError(
                f"ConditionSpec({self.name!r}): kind=APPEND_TO_SLOT requires "
                "width; it must equal the host slot's width because both go "
                "through the same condition projector, and V7 reconciles on "
                "it"
            )
        if self.source_width is not None and self.encoder is None:
            raise ValueError(
                f"ConditionSpec({self.name!r}).source_width="
                f"{self.source_width} is only meaningful with an encoder; "
                "without one the raw input IS the value, so width is the "
                "only number"
            )
        if len(set(self.targets)) != len(self.targets):
            raise ValueError(
                f"ConditionSpec({self.name!r}).targets has duplicates: "
                f"{self.targets}"
            )
        if self.encoder is None and self.source_key is None:
            raise ValueError(
                f"ConditionSpec({self.name!r}): encoder and source_key must "
                "not both be None, otherwise this condition's value has "
                "nowhere to come from"
            )
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError(
                f"ConditionSpec({self.name!r}).dropout_prob must be in "
                f"[0, 1), got {self.dropout_prob}"
            )

    def materialize(self, value: Tensor) -> Condition:
        """Wrap an already computed tensor into a ``Condition``.

        This is the only path from declaration to instance.

        Args:
            value: The condition tensor for this forward pass.

        Returns:
            A ``Condition`` carrying this spec's name, kind and targets.
        """
        return Condition(name=self.name, kind=self.kind, value=value,
                         targets=frozenset(self.targets))
