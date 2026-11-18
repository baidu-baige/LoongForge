# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Resolution of the [YAML] overrides a provider accepts.

Three helpers, one per strictness rule of
``docs/design/mot_spec_config.zh-CN.md`` §3.8: stream scalars
(``resolve_scalars``), the ``geometry:`` section (``resolve_geometry``), and
the refusal to hand a provider the config object at all (``reject_config``).

None of the three knows any topology, so none of them is family-specific: the
set of legal scalar keys is a parameter, owned by the topology family that
declares those scalars.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any

from ..geometry import GeometrySpec


def resolve_scalars(model_type: str, defaults: dict[str, int],
                    overrides: dict[str, Any],
                    legal_keys: frozenset[str]) -> dict[str, int]:
    """Apply stream scalar overrides, raising on any key not consumed.

    Strict rather than silently ignoring, which is the third of the deliberate
    strictness rules in spec_config §3.8: staying silent turns one mistyped key
    into a value that was "configured but had no effect", and that can only be
    found by reading the provider source.

    ``defaults`` holds only the values that **differ from the family
    defaults**; an empty table is legal (fastwam's is empty).

    The set of legal keys is ``legal_keys``, not ``set(defaults)``.
    Otherwise the scalars a provider does not override could not be overridden
    from YAML either, and "can this be overridden from YAML" is a family-level
    fact that must not shift with a provider's default table.

    The keys of ``defaults`` are validated too: otherwise the symptom of a
    missing or misspelled key there is a value that can never take effect.

    Args:
        model_type: Provider key, used in the error message.
        defaults: Provider defaults, a subset of ``legal_keys``.
        overrides: Keyword arguments collected from ``**stream_scalars``.
        legal_keys: Every scalar name the family accepts, which is
            where the strictness comes from.

    Returns:
        ``defaults`` updated by ``overrides``.

    Raises:
        ValueError: Either mapping carries a key outside
            ``legal_keys``.
    """
    for label, keys in (("default table has illegal keys", set(defaults)),
                        ("got keys it does not consume", set(overrides))):
        stray = sorted(keys - legal_keys)
        if stray:
            raise ValueError(
                f"provider({model_type!r}) {label} {stray}; "
                f"legal stream scalars are {sorted(legal_keys)} "
                "(spec_config §3.8)"
            )
    resolved = dict(defaults)
    resolved.update(overrides)
    return resolved


_GEOMETRY_KEYS: frozenset[str] = frozenset(
    field.name for field in fields(GeometrySpec))
"""Field names of ``GeometrySpec``, i.e. the keys a YAML ``geometry:`` section
may carry. Derived from the dataclass instead of being listed by hand: every
name here is a constructor argument, so reflection cannot pull in anything a
caller is not allowed to set (contrast ``STREAM_SCALAR_KEYS``, where the
signature also holds non-overridable arguments such as ``video_impl``)."""


def resolve_geometry(model_type: str, defaults: dict[str, Any],
                      overrides: Mapping[str, Any] | None) -> GeometrySpec:
    """Build the geometry, applying the YAML ``geometry:`` section on top.

    The merge happens *before* the provider derives anything from the geometry.
    That ordering is mandatory, not stylistic: visibility rules are built from
    ``geometry.tokens_per_frame`` and the flow heads read
    ``geometry.latent_channels`` / ``geometry.action_dim``. Replacing the
    geometry after the fact (``dataclasses.replace(spec, geometry=...)``) would
    leave those three derived values pointing at the pre-override numbers, and
    nothing downstream can detect the mismatch.

    Args:
        model_type: Provider key, used only in the error message.
        defaults: The provider's hardcoded geometry, as constructor kwargs.
        overrides: The YAML ``geometry:`` section, or ``None``.

    Returns:
        The merged ``GeometrySpec``. Its ``__post_init__`` runs every
        divisibility check, so an override that breaks the pixel → VAE → patch
        chain fails here rather than at the first forward pass.

    Raises:
        ValueError: An override key is not a ``GeometrySpec`` field. Unknown
            keys are rejected rather than ignored, because a silently dropped
            key reads exactly like a key that had no effect.
    """
    if not overrides:
        return GeometrySpec(**defaults)
    stray = sorted(set(overrides) - _GEOMETRY_KEYS)
    if stray:
        raise ValueError(
            f"provider({model_type!r}) got geometry keys that do not exist: "
            f"{stray}; legal geometry fields are {sorted(_GEOMETRY_KEYS)}"
        )
    return GeometrySpec(**{**defaults, **dict(overrides)})


def reject_config(model_type: str, config: Any) -> None:
    """Reject a positional config object.

    ``build_world_model`` translates the YAML sections into *keyword arguments*
    and never hands the config object itself to a provider: a provider that can
    read the config would be free to pull topology out of YAML, which is
    exactly what R5 forbids. The parameter itself is kept so that the mistake
    raises here instead of binding to ``**stream_scalars`` and surfacing as an
    unrelated "stray scalar key" error.
    """
    if config is not None:
        raise NotImplementedError(
            f"provider({model_type!r}) got a config object as a positional "
            "argument. Applying config belongs to build_world_model: it "
            "translates the five YAML sections into keyword arguments "
            "(geometry= / <stream>_<scalar>= / <stream>_shift= / subclass "
            "fields), and a provider never sees the config object itself (R5)"
        )
