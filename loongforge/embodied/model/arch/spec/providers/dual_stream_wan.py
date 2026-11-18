# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``dual_stream_wan`` family: a video and an action stream of
equal depth sharing one joint attention.

Family-level assembly follows ``docs/design/mot_spec_model_matrix.zh-CN.md``
§4.2. Three models sit in the family, one package each: ``vpp2/`` /
``fastwam/`` / ``internw0_delta/``.

``_dual_stream_wan_streams`` is **not registered**: it is only the shared
first half of the three, and neither YAML nor the launch arguments can name it
(spec_config §3.8). The sibling provider packages call it directly.
"""

from __future__ import annotations

from ...streams.spec import StreamSpec
from ..core import Spec


def _dual_stream_wan_streams(
    video_impl: Spec,
    action_impl: Spec,
    video_residual_width: int = 3072,
    video_num_heads: int = 24,
    video_head_dim: int = 128,
    video_num_layers: int = 30,
    video_ffn_width: int = 14336,
    action_residual_width: int = 1024,
    action_num_heads: int = 24,
    action_head_dim: int = 128,
    action_num_layers: int = 30,
    action_ffn_width: int = 4096,
    cross_kv_slots: tuple[str, ...] = ("text",),
    video_cross_kv_slots: tuple[str, ...] | None = None,
    timestep_embed_width: int = 256,
) -> tuple[StreamSpec, StreamSpec]:
    """Build the two streams shared by the three providers in the family.

    The topology is hardcoded here; only numbers stay as parameters.

    The defaults are the measured values of the baseline fastwam
    (``model/fastwam/modeling_configuration_fastwam.py:75-96``). Note that the
    action stream has residual width 1024 while its attention is
    24 x 128 = 3072: the two are simply not equal.

    Both ``impl`` arguments are required: the three providers use different
    block classes, so the family cannot supply a default.

    The two streams must have equal ``num_layers``, which is the precondition
    for per-layer joint attention. §4.2 claims V3 guarantees it; that is wrong
    (V3 checks ``head_dim`` equality), and **V2 only blocks half of it**.
    Measured against ``resolve_layer_index_map``
    (``streams/spec.py:263-290``), with the group's joint depth taken from
    ``video.num_layers``:

      - action stream **deeper** than the joint depth => V2 raises;
      - action stream **shallower** than the joint depth => V2 **passes**.
        Under ONE_TO_ONE, ``participates_in_joint`` tests
        ``layer_id < self.num_layers`` (``:253``), so a shallow stream is read
        as "participates only in the first N joint layers". That is the legal
        shape STRIDED / AT_LAYERS express, and V2 has no reason to reject it.

    None of the three family members has that shape, so the precondition can
    only be stated by the **family itself**. The explicit equality check below
    is that statement; do not delete it and fall back on V2.

    Args:
        video_impl: ``Spec`` for the video stream block.
        action_impl: ``Spec`` for the action stream block.
        video_residual_width: Video residual width.
        video_num_heads: Video attention head count.
        video_head_dim: Video per-head dimension.
        video_num_layers: Video depth, which is also the joint depth.
        video_ffn_width: Video feed-forward width.
        action_residual_width: Action residual width.
        action_num_heads: Action attention head count.
        action_head_dim: Action per-head dimension.
        action_num_layers: Action depth; must equal ``video_num_layers``.
        action_ffn_width: Action feed-forward width.
        cross_kv_slots: Cross-attention slot names for both streams, one text
            slot per layer by default. All three family members use WAN DiT
            blocks with one per-layer text cross-attention (``context`` in
            VPP2's ``wan_video_dit.py`` and ``text_embedding`` at
            ``action_dit.py:405``). It lives at family level, not in a
            provider, because "how many cross-attention slots exist" is
            family topology.
        video_cross_kv_slots: Video-only override; ``None`` means the same as
            ``cross_kv_slots``. It exists because an i2v video stream can have
            one extra slot: ``I2VCrossAttention.forward``
            (``wan_video_dit.py:187-204``) runs **two** cross attentions,
            ``k_img/v_img`` over the first 257 CLIP tokens and ``k/v`` over
            the text tokens after them, then sums the two results through one
            ``o`` projection. That is two slots, not one slot holding two
            concatenated inputs; declaring it as two conditions removes the
            need for the ``ctx[:, :257]`` split point.
        timestep_embed_width: 256 for both streams, the ``freq_dim`` at
            ``wan_video_dit.py:326`` and ``action_dit.py:259-260``, which both
            read from the same config entry. It is a family constant rather
            than a [YAML] knob: changing it changes the input width of the
            first ``time_embedding`` Linear, which means a different set of
            pretrained weights.

    Returns:
        ``(video_spec, action_spec)``, in that order.

    Raises:
        ValueError: ``video_num_layers != action_num_layers``.
    """
    if video_num_layers != action_num_layers:
        raise ValueError(
            "the dual_stream_wan family requires both streams to have equal "
            "depth, which per-layer joint attention depends on; got "
            f"video_num_layers={video_num_layers}, "
            f"action_num_layers={action_num_layers}. "
            "V2 only rejects the case where action has more layers than the "
            "joint attention; fewer layers it would accept as a shallow "
            "stream that joins only the first few layers, so this check must "
            "live in the family"
        )
    return (
        StreamSpec(
            name="video",
            impl=video_impl,
            residual_width=video_residual_width,
            num_heads=video_num_heads,
            head_dim=video_head_dim,
            num_layers=video_num_layers,
            ffn_width=video_ffn_width,
            cross_kv_slots=(cross_kv_slots if video_cross_kv_slots is None
                            else video_cross_kv_slots),
            timestep_embed_width=timestep_embed_width,
            writable=True,
        ),
        StreamSpec(
            name="action",
            impl=action_impl,
            residual_width=action_residual_width,
            num_heads=action_num_heads,
            head_dim=action_head_dim,
            num_layers=action_num_layers,
            ffn_width=action_ffn_width,
            cross_kv_slots=cross_kv_slots,
            timestep_embed_width=timestep_embed_width,
            writable=True,
        ),
    )


STREAM_SCALAR_KEYS: frozenset[str] = frozenset({
    "video_residual_width", "video_num_heads", "video_head_dim",
    "video_num_layers", "video_ffn_width",
    "action_residual_width", "action_num_heads", "action_head_dim",
    "action_num_layers", "action_ffn_width",
})
"""The ten numeric parameter names of ``_dual_stream_wan_streams``.

Kept in sync with that signature **by hand**: deriving it with
``inspect.signature`` would also pull in ``video_impl`` / ``cross_kv_slots``,
and those two are not [YAML]-overridable.
"""
