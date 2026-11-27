# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``internw0_delta`` provider: the ``dual_stream_wan``
family with five loss terms.

Its two streams match the dual-stream WAN family defaults field by field.
What is specific to this provider is the five-head output table, the
distillation losses, and the ``flex`` attention backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ....diffusion.schedule import RectifiedFlow
from ....diffusion.spec import DiffusionSpec
from ....diffusion.stepping import EulerSampler, UniformSteps
from ....diffusion.timesteps import TimestepSampler, UniformShift
from ....geometry import GeometrySpec
from ....heads.flow import build_flow_head
from ....losses import CosineDistance, LossSpec, WeightRamp
from ....heads.hidden_ref import HiddenStateRef
from ....heads.spec import HeadSpec
from ....streams.blocks import ActionStream, WanVideoStream
from ....streams.conditions import ConditionSpec, InjectionKind
from ....streams.encoders import linear_token_encoder
from ....streams.engine import AttentionGroup
from ....streams.spec import TokenTimestep
from ....visibility.combine import Visibility
from ....visibility.phases import (CONDITION, REAL_VIDEO, Phase, phase_labels)
from ....visibility.rules import frame_causal, label_edge, same_stream
from ...core import Spec, register_provider
from ...overrides import (StreamShift, reject_config, resolve_diffusion,
                          resolve_geometry, resolve_scalars)
from ...world import WorldModelSpec
from ..dual_stream_wan import STREAM_SCALAR_KEYS, _dual_stream_wan_streams


# ---- the track_align head impl, used by this provider only ----


def track_query_head(*, spec: HeadSpec, geometry: GeometrySpec,
                     world: WorldModelSpec, **params: Any) -> Any:
    """Placeholder for InternW0-Δ's ``track_align`` head.

    This is **not** a flow head, so it does not reuse ``build_flow_head``. The
    existing implementation is a student decoder with learned queries
    (``InternW0-Delta/src/wam/model/modules/distillation/track_bridge.py:14-54``):
    a ``video_to_track`` Linear, ``track_queries`` of shape
    ``[1, 16, hidden]``, a two-layer 16-head ``TrackQueryDecoder``, and a
    four-stage ``student_projector``, finally mean-pooled along the query axis
    into ``[B, 1430]``.

    That shape matches none of the four flow forms (it emits a **2D** tensor
    with the token axis pooled away), so routing it through
    ``build_flow_head`` could only produce a head with mismatched shapes.
    Hence the explicit error instead of picking the closest form.

    Args:
        spec: Head declaration; only ``name`` and ``output_width`` are used,
            in the error message.
        geometry: Unused.
        world: Unused.
        **params: Unused.

    Raises:
        NotImplementedError: Always. The specialized track head is not
            implemented in this provider yet.
    """
    del geometry, world, params
    raise NotImplementedError(
        f"head {spec.name!r} (track_align, output_width={spec.output_width}) "
        "is a student decoder with learned queries, not a flow head; "
        "see track_bridge.py:14-54, implementation belongs to Lane C"
    )


_INTERNW0_DELTA_STREAM_SCALARS: dict[str, int] = {}
"""InternW0-Δ's two streams match the family defaults **field by field**.

The legacy InternW0-Delta WAM configuration specifies that video
uses hidden width 3072, FFN width 14336, 24 heads, head dimension
128, and 30 layers. Action uses hidden width 1024, FFN width 4096,
24 heads, head dimension 128, and 30 layers.

**Important detail**: the legacy factory does not rewrite the action
stream's attention width to ``num_heads x attn_head_dim``. It only **validates**
that the two streams have equal head
count and head dimension, raising when they differ. So the action stream's
residual width of 1024 next to an attention width of 24 x 128 = 3072 is kept
**as is** here, which is exactly why ``residual_width`` and the attention
width have to be two separate fields.
"""


def _internw0_delta_visibility() -> dict[str, Visibility]:
    """Build InternW0-Δ's L1 visibility.

    The visibility below follows the legacy InternW0-Delta frame-attention
    implementation.

    Current layout (single camera, 96 tokens per frame): video 0-479 =
    anchor 0-95 + recent 96-191 + current-clean 192-287 +
    noisy-future 288-479; **delta 480-575**; action 576-607.

    The seven writes in the code: ``:27-31`` per-frame causality over the
    three condition frames; ``:35`` ``mask[288:480, :480] = True`` (the noisy
    future is **fully bidirectional** over all real video and sees neither
    delta nor action); ``:38-46`` delta reads ``[96, 288)`` (recent plus
    current-clean, **excluding anchor**); ``:47-50`` delta reads delta;
    ``:52`` action reads ``[0, 288)`` (condition frames only, so it **never**
    sees the noisy future); ``:53-54`` action reads delta; ``:55`` action
    reads action.

    Four of those seven are expressed here; the other three are the delta
    rules, and they are **not** blocked by the algebra. §2.2.1 of the L0/L1
    design doc measures all seven bit-exact against this builder once delta
    tokens get a segment, with no further rule needed than the ``label_edge``
    already used below. What is missing is addressability, not expressiveness:

      1. **Delta tokens have nowhere to live.** Those 96 tokens are appended
         inside the video stream (appended once at
         ``wan_video_dit.py:588-590``, trimmed off at ``:959-967``), so they
         appear neither in ``GeometrySpec``'s token accounting nor as their own
         segment. They are also not a ``StreamSpec``: they run on the video
         expert's parameters, so the fix is to let one stream own several
         segments, not to invent a stream. That is ruling 2 of §2.2.2 and is
         still open; ``Phase.DELTA`` is reserved for the day it closes.

    Both halves of the mask are written as phase edges rather than as token
    counts:

      - the condition rows are causal among themselves while the noisy-future
        rows are bidirectional over all real video. That is two different
        intra-stream rules inside one stream, which needs a query-side
        selector; ``frame_causal()`` alone measured 48/784 cells too strict;
      - "the action rows read the condition frames" was previously
        ``prefix_only(condition_frames x tokens_per_frame)``, equal to
        ``[0, 288)`` only because the condition frames happen to sit at the
        front of the joint sequence. The phase edge says what is meant, and
        takes no token count, so the expression no longer depends on geometry.

    Returns:
        ``{"video": ..., "action": ...}`` visibility rules.
    """
    return {
        "video": ((label_edge("phase", CONDITION, CONDITION) & frame_causal())
                  | label_edge("phase", REAL_VIDEO, (Phase.FUTURE,))),
        "action": (same_stream()
                   | label_edge("phase", CONDITION, (Phase.ACTION,))),
    }


@dataclass(frozen=True)
class _InternW0DeltaDiffusion:
    """YAML-overridable diffusion parameters for InternW0-Delta (``diffusion.<stream>.shift``)."""

    video: StreamShift = StreamShift(shift=6.0)
    action: StreamShift = StreamShift(shift=1.0)


@register_provider("internw0_delta")
def _internw0_delta(config: Any = None, *,
                    geometry: Mapping[str, Any] | None = None,
                    condition_frames: int = 3,
                    inference_steps: int = 10,
                    action_width: int = 80,
                    proprio_dim: int = 80,
                    diffusion: Mapping[str, Any] | None = None,
                    **stream_scalars: Any) -> WorldModelSpec:
    """Build the InternW0-Δ spec; ``--model-name internw0_delta``.

    The family member with the most loss terms (**five**).

    Its two streams are assembled with exactly the family default values (see
    ``_INTERNW0_DELTA_STREAM_SCALARS``); all differences are in L3/L4: five
    supervision terms plus two alignment heads that read hidden states from
    **intermediate layers**. That is precisely why ``losses`` has to be a list
    and ``HiddenStateRef`` has to carry a ``layer``.

    First-hand facts live in ``projects/InternW0-Delta/``, cited by line
    number at each use.

    Args:
        config: Positional config object, rejected (see ``reject_config``).
        geometry: The YAML ``geometry:`` section, merged over the defaults.
        condition_frames: Leading clean condition frames, 3 today; the action
            stream reads ``condition_frames x tokens_per_frame`` video tokens.
        inference_steps: Euler steps at inference.
        action_width: Packed action dimension, 80 = 20 DoF x 4 chunks.
        proprio_dim: Raw proprioception width, 80 here.
        diffusion: The YAML ``diffusion:`` section, resolved over this
            provider's defaults before the samplers are built
            (``resolve_diffusion``); ``None`` when YAML has none.
        **stream_scalars: Per-stream width and depth overrides.

    Returns:
        The validated ``WorldModelSpec`` for InternW0-Δ.

    Raises:
        NotImplementedError: ``config`` is not ``None``.
        ValueError: An unknown scalar or geometry key, or an invalid model
            specification.
    """
    reject_config("internw0_delta", config)

    scalars = resolve_scalars("internw0_delta",
                              _INTERNW0_DELTA_STREAM_SCALARS, stream_scalars,
                              STREAM_SCALAR_KEYS)
    video, action = _dual_stream_wan_streams(
        video_impl=Spec(WanVideoStream),
        action_impl=Spec(ActionStream),
        video_token_timestep=TokenTimestep(clean_prefix=condition_frames),
        **scalars,
    )
    group = AttentionGroup(streams=(video, action),
                           num_layers=video.num_layers)

    # Geometry. First-hand numbers: ``configs/data/pretrain.yaml:10``
    # ``video_size (384, 256)`` = (H, W), ``:33`` ``num_output_cameras: 1``,
    # ``configs/pretrain/dataset.yaml:320`` ``action_size 32`` and
    # ``:322`` ``action_video_freq_ratio 4``. The VAE is the same as fastwam's,
    # Wan2.2 ``WanVideoVAE38`` (``wan_video_vae.py:1358-1387``, 16/4/z_dim 48),
    # with ``patch_size [1,2,2]`` (``configs/model/wam.yaml:16-19``).
    #
    # **``num_frames=17`` is not any number in the config; noting why**:
    # the config says ``num_frames 33`` (``dataset.yaml:319``) and decodes
    # 9 RGB frames (``range(0, 33, 4)`` at
    # ``robot_video_dataset.py:130``), so the noised window is
    # 1 + (9-1)/4 = **3 latent frames**. But the video prefix entering MoT is
    # ``cat([anchor_latents, recent_latents, noisy_window], dim=2)``
    # (``losses.py:598-600``) = anchor 1 + recent 1 + window 3 = **5 latent
    # frames** = 5 x 96 = **480 tokens** (``wan_video_dit.py:814``).
    # ``GeometrySpec`` has **no field for "concatenate N extra condition latent
    # frames"**, and ``padded_seq_len`` is the only source of the static shape
    # (INV-2), so it has to come out as 480. 17 is the equivalent pre-
    # compression frame count that yields 5 latent frames (1 + 16/4 = 5).
    # Cost: ``num_frames`` in this provider no longer equals the frame count on
    # the data side. The real fix is to give ``GeometrySpec`` a condition-frame
    # count field so this contrived 17 disappears. That belongs to Lane A.
    # Derived: latent (5, 24, 16) -> patch (5, 12, 8), so 96 tokens per frame
    # and 480 tokens total; ``num_chunks = 32 / 4 = 8``.
    geometry = resolve_geometry("internw0_delta", dict(
        frame_height=384,
        frame_width=256,
        num_frames=17,
        num_views=1,
        latent_channels=48,
        action_horizon=32,
        action_video_freq_ratio=4,
        # Action stream entry width = ``action_flow`` head output width
        # (same note as in the vpp2 provider).
        # For InternW0-Δ it is 80 = 20 DoF x 4 chunks.
        action_dim=action_width,
    ), geometry)

    # Each stream samples its own t using ``UniformShift``, with different
    # shift values: video 6.0 and action 1.0. These values are shared by
    # training and inference in the legacy WAM configuration.
    #
    # **``seperated_timestep: true`` in the legacy WAM configuration is not
    # per-frame
    # sampling.** This follows the legacy loss construction:
    #   ``cat([zeros(B, num_condition_frames - 1),
    #          timestep_video[:, None].expand(B, future_latent_frames)], dim=1)``
    # so a **single ``[B]`` draw** is broadcast over the future frames while
    # the condition frames are pinned to 0. Broadcast therefore stays at the
    # default per-sample, **not** ``sample_shape=("batch","frame")``, which
    # would mean F independent draws and does not match the code. The declared
    # ``noise_levels`` describe the clean and noisy cases, while assigning a
    # level to each token remains part of the training data path.
    # A separate ``[B, T]`` per-action-token path exists in the legacy RTC
    # branch, but it is disabled in the default topology.
    #
    # YAML may set the two shifts as ``diffusion.<stream>.shift``; any other
    # key raises (§1.15).
    diffusion_values = resolve_diffusion(
        "internw0_delta", _InternW0DeltaDiffusion(), diffusion)
    diffusion = DiffusionSpec(
        timesteps={
            "video": TimestepSampler(UniformShift(shift=diffusion_values.video.shift)),
            "action": TimestepSampler(UniformShift(shift=diffusion_values.action.shift)),
        },
        schedule=RectifiedFlow(),
        sampling_gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        solver=EulerSampler(),
        # Declare the clean and noisy levels used by the visibility rules.
        # The training path still has to attach the corresponding labels to
        # individual tokens.
        noise_levels=(0.0, 1.0),
    )

    # Five terms with weights 0.5, 1.0, 0.5, 0.1, and a 0.1 track weight after
    # warmup. Kept on WorldModelSpec.losses rather than DiffusionSpec: the
    # self-distillation CosineDistance and the track WeightRamp are not
    # diffusion objectives.
    losses = (
        LossSpec(name="video_flow", head="video_flow", weight=0.5),
        LossSpec(name="action_flow", head="action_flow", weight=1.0),
        LossSpec(name="future_delta", head="future_delta", weight=0.5),
        LossSpec(name="semantic_future", head="semantic_future",
                 objective=CosineDistance(), weight=0.1),
        LossSpec(name="track_align", head="track_align",
                 ramp=WeightRamp(start_step=0, end_step=2000,
                                 start_value=0.0, end_value=0.1)),
    )

    # Five output heads are declared below. The first two predict flow,
    # and the remaining three support future-delta, semantic, and track losses.
    # Supervision weights are read from ``LossSpec`` only; non-default
    # ``HeadSpec.weight`` values are rejected by ``WorldModel.compute_losses``.
    heads = (
        # 48 = ``out_dim`` (the legacy WAM model configuration), whose source is
        # ``geometry.latent_channels``: entry channels and exit channels are
        # the same number, so this reads the geometry instead of repeating a
        # literal. The actual Linear width inside the head is
        # 48 x prod(patch) = 192, which does not go
        # into output_width.
        HeadSpec(name="video_flow",
                 impl=Spec(build_flow_head, {"form": "patch_flow"}),
                 source=HiddenStateRef(stream="video"),
                 output_width=geometry.latent_channels,
                 needs_timestep_embedding=True),
        # The action head is a bare ``Linear(1024, 80)`` using the packed
        # action width. It has no modulation and therefore
        # ``needs_timestep_embedding=False``.
        HeadSpec(name="action_flow",
                 impl=Spec(build_flow_head, {"form": "linear"}),
                 source=HiddenStateRef(stream="action"),
                 output_width=geometry.action_dim),
        # future_delta: ``Sequential(LayerNorm(3072), Linear(3072, 384))``
        # (the legacy future-delta projection) reshaped to
        # ``(2, 192)``, where ``patch_dim = 48 x 1 x 2 x 2 = 192``
        # (the legacy action-model projection) and ``future_delta_num_frames =
        # (33-1)//4//4 = 2``, so output_width is **192** (the
        # per-frame width); the frame count 2 is a separate axis and does not
        # go into this field. No modulation, so needs_timestep_embedding=False.
        #
        # This head corresponds to the 96 delta tokens at ``[480:576]`` of
        # the video stream. ``HiddenStateRef`` has no ``segment`` field, so the
        # current declaration can identify only the video stream and uses its
        # final output. The delta span remains an explicit limitation of the
        # current geometry representation.
        HeadSpec(name="future_delta",
                 impl=Spec(build_flow_head, {"form": "norm_linear"}),
                 source=HiddenStateRef(stream="video"),
                 output_width=192),
        # semantic_future: **no projection layer**, ``1 - cos`` computed
        # directly on the raw 3072-dim hidden states.
        # student = the delta segment ``[480:576]`` of **layer 8**,
        # teacher = ``[384:480]`` of **layer 20**, detached
        # (the legacy semantic loss construction and its configured capture layers).
        #
        # The current declaration cannot express two parts of this loss: (1)
        # ``HeadSpec.source`` is a **single** ref, so it cannot hold the
        # teacher side; (2) there is no ``detach`` field, yet the teacher must
        # be detached, otherwise this self-distillation term pushes gradients
        # back into layer 20 and becomes a different optimization problem.
        # Only the student side is declared here; the teacher side and detach
        # are not represented by ``HeadSpec`` yet. The declaration below
        # therefore describes the student output shape, rather than the complete
        # teacher-student loss computation.
        HeadSpec(name="semantic_future",
                 impl=Spec(build_flow_head, {"form": "identity"}),
                 source=HiddenStateRef(stream="video", layer=8),
                 output_width=video.residual_width),
        # track_align: reads **layer 14** (``source_layer: 14`` at
        # the distillation configuration) ->
        # ``video_to_track = Linear(3072, 1024)`` (the legacy track bridge) ->
        # the tail of ``student_projector``, ``Linear(2048, 1430)``,
        # with teacher width 1430, so output_width is the final 1430.
        #
        # It is **not a flow head** (a decoder with learned queries, pooled
        # along the query axis into a 2D tensor), so impl goes through
        # ``track_query_head`` rather than ``build_flow_head``: none of the
        # four forms can express it, and forcing the closest one would only
        # produce a head whose shapes do not match.
        HeadSpec(name="track_align", impl=Spec(track_query_head),
                 source=HiddenStateRef(stream="video", layer=14),
                 output_width=1430),
    )

    return WorldModelSpec(
        model_type="internw0_delta",
        geometry=geometry,
        attention_groups=(group,),
        diffusion=diffusion,
        heads=heads,
        losses=losses,
        # One text cross-attention slot per layer, consumed by both streams.
        # Unlike vpp2, this provider does not use CLIP conditioning.
        # Both streams use text context width 4096, which is the input width
        # of their ``text_embedding`` projections.
        # Proprioception is projected to width 4096 and appended to the text
        # context for both streams. The spec validation checks that the appended
        # condition has the same width as its host slot.
        # ``source_width=80`` is the configured proprioception width.
        conditions=(
            ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                          targets=("video", "action"), width=4096,
                          source_key="context"),
            ConditionSpec(name="proprio", kind=InjectionKind.APPEND_TO_SLOT,
                          targets=("video", "action"), width=4096,
                          append_to="text", source_width=proprio_dim,
                          encoder=Spec(linear_token_encoder),
                          source_key="proprio"),
        ),
        visibility=_internw0_delta_visibility(),
        # ``anchor_frames`` is deliberately left at 0. The model forces
        # exactly one anchor frame (``world_action_model.py:302-305``), but
        # every rule above reads the whole ``CONDITION`` set, so the
        # anchor/recent split is invisible today. It gets declared together
        # with the delta segment, whose R3 (``[recent, current] -> delta``) is
        # the only rule that needs to tell them apart.
        token_labels=phase_labels(condition_frames=condition_frames),
        # The only member of the family that uses flex:
        # ``mot_attention_backend: flex`` at ``wam.yaml:9``.
        # Block size **128**: ``wam.yaml:10`` says 64, but
        # ``configs/task/pretrain.yaml:17`` overrides it to 128, and the mask
        # side has yet another hardcoded ``self.flex_mask_block_size = 128``
        # (``:113``). Three places for one number, two distinct values; 128 is
        # taken because it is the value that takes effect. Noted: this is
        # exactly why V9 requires flex to be given an explicit positive
        # block_size.
        # Also noted: ``mixture_of_transformers.py:593-603`` has a **silent**
        # fallback (``force_sdpa = not bool(key_mask_all.all().item())``),
        # which is both a D2H sync and a way to make "which backend am I using"
        # unknowable at runtime.
        attention_backend="flex",
        attention_block_size=128,
    )
