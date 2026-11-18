# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``internw0_delta`` provider: the ``dual_stream_wan``
family with five loss terms.

Follows ``docs/design/mot_spec_model_matrix.zh-CN.md`` §4.6. Its two streams
match the family defaults field by field; what is specific to it is the head
set (five, including the distillation heads) and the ``flex`` backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....diffusion.loss import CosineDistance, LossSpec, WeightRamp
from ....diffusion.schedule import RectifiedFlow
from ....diffusion.spec import DiffusionSpec
from ....diffusion.stepping import EulerSampler, UniformSteps
from ....diffusion.timesteps import TimestepSampler, UniformShift
from ....geometry import GeometrySpec
from ....heads.flow import build_flow_head
from ....heads.hidden_ref import HiddenStateRef
from ....heads.spec import HeadSpec
from ....streams.blocks import ActionStream, WanVideoStream
from ....streams.conditions import ConditionSpec, InjectionKind
from ....streams.encoders import linear_token_encoder
from ....streams.engine import AttentionGroup
from ....visibility.combine import Visibility
from ....visibility.rules import (frame_causal, prefix_only, same_stream,
                                  stream_edge)
from ...core import Spec, register_provider
from ...overrides import reject_config, resolve_geometry, resolve_scalars
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
        NotImplementedError: Always. Implementation belongs to Lane C.
    """
    del geometry, world, params
    raise NotImplementedError(
        f"head {spec.name!r} (track_align, output_width={spec.output_width}) "
        "is a student decoder with learned queries, not a flow head; "
        "see track_bridge.py:14-54, implementation belongs to Lane C"
    )


_INTERNW0_DELTA_STREAM_SCALARS: dict[str, int] = {}
"""InternW0-Δ's two streams match the family defaults **field by field**.

First-hand source ``projects/InternW0-Delta/configs/model/wam.yaml``: video
``hidden_dim 3072`` (:21) / ``ffn_dim 14336`` (:22) / ``num_heads 24`` (:26) /
``attn_head_dim 128`` (:27) / ``num_layers 30`` (:28); action
``hidden_dim 1024`` (:44) / ``ffn_dim 4096`` (:45) / ``num_heads 24`` (:46) /
``attn_head_dim 128`` (:47) / ``num_layers 30`` (:48).

**Recorded**: the design doc says ``factory.py`` rewrites the action stream's
attention width to ``num_heads x attn_head_dim``. It does not;
``factory.py:101-110`` only **validates** that the two streams have equal head
count and head dimension, raising when they differ. So the action stream's
residual width of 1024 next to an attention width of 24 x 128 = 3072 is kept
**as is** here, which is exactly why ``residual_width`` and the attention
width have to be two separate fields.
"""


def _internw0_delta_visibility(tokens_per_frame: int,
                               condition_frames: int) -> dict[str, Visibility]:
    """Build InternW0-Δ's L1 visibility.

    First-hand facts: ``frame_attention.py:6-56``.

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

    **Four things cannot be expressed; each is recorded, resolved toward
    "stricter rather than more permissive", never silently:**

      1. **Delta tokens are not a stream.** Those 96 tokens are appended
         inside the video stream (appended once at
         ``wan_video_dit.py:588-590``, trimmed off at ``:959-967``), so they
         appear neither in ``GeometrySpec``'s token accounting nor as their
         own ``StreamSpec``. The three delta rules and the action->delta edge
         therefore have nowhere to land. §4.6.7 says delta should become a
         derived property of ``GeometrySpec``; this waits for that ruling.
      2. **``frame_causal()`` is too strict for the noisy future.** Rows
         288-479 are fully bidirectional in the code; per-frame causality
         would limit them to the past. Stricter is preferred: in an
         equivalence test, "saw tokens it should not have" still converges
         numerically and cannot be detected, while "saw too few" shows up
         immediately.
      3. **The action side can only be written ``prefix_only(3 x 96)``** for
         "reads the three condition frames only". It happens to equal
         ``[0, 288)`` at ``:52`` because the condition frames sit at the front
         of the sequence; the two facts diverge as soon as the layout
         changes. ``history(policy)`` is where it really belongs.
      4. The names ``sees`` / ``reads`` / ``delta_reads_local`` used in the
         design doc **do not exist** in ``visibility/rules.py``, and among the
         nine atomic rules ``frame_causal`` takes ``window`` rather than
         ``condition_frames``.

    Args:
        tokens_per_frame: Video tokens per latent frame (96 today).
        condition_frames: Number of leading clean condition frames the action
            stream may read.

    Returns:
        ``{"video": ..., "action": ...}`` visibility rules.
    """
    return {
        "video": same_stream() & frame_causal(),
        "action": same_stream() | (
            stream_edge("video", "action")
            & prefix_only(condition_frames * tokens_per_frame)
        ),
    }


@register_provider("internw0_delta")
def _internw0_delta(config: Any = None, *,
                    geometry: Mapping[str, Any] | None = None,
                    condition_frames: int = 3,
                    video_shift: float = 6.0,
                    action_shift: float = 1.0,
                    inference_steps: int = 10,
                    action_width: int = 80,
                    proprio_dim: int = 80,
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
        video_shift: ``shift`` of the video ``UniformShift`` (6.0).
        action_shift: ``shift`` of the action ``UniformShift`` (1.0).
        inference_steps: Euler steps at inference.
        action_width: Packed action dimension, 80 = 20 DoF x 4 chunks.
        proprio_dim: Raw proprioception width, 80 here.
        **stream_scalars: Per-stream width and depth overrides.

    Returns:
        The validated ``WorldModelSpec`` for InternW0-Δ.

    Raises:
        NotImplementedError: ``config`` is not ``None``.
        ValueError: An unknown scalar or geometry key, or a V1-V9 failure.
    """
    reject_config("internw0_delta", config)

    scalars = resolve_scalars("internw0_delta",
                              _INTERNW0_DELTA_STREAM_SCALARS, stream_scalars,
                              STREAM_SCALAR_KEYS)
    video, action = _dual_stream_wan_streams(
        video_impl=Spec(WanVideoStream),
        action_impl=Spec(ActionStream),
        **scalars,
    )
    group = AttentionGroup(members=(video, action),
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

    # L3. Each stream samples its own t, both ``UniformShift``, with different
    # shift values: video 6.0 (``wam.yaml:53-56``, same value for train and
    # infer) and action 1.0 (``:57-60``, likewise same for both).
    #
    # **``seperated_timestep: true`` (``wam.yaml:30``) is not per-frame
    # sampling.** Checked literally against ``losses.py:624-634``:
    #   ``cat([zeros(B, num_condition_frames - 1),
    #          timestep_video[:, None].expand(B, future_latent_frames)], dim=1)``
    # so a **single ``[B]`` draw** is broadcast over the future frames while
    # the condition frames are pinned to 0. Broadcast therefore stays at the
    # default per-sample, **not** ``sample_shape=("batch","frame")``, which
    # would mean F independent draws and does not match the code. The per-frame
    # structure is expressed by the two ``noise_levels`` tiers plus the L1
    # prefix rule for condition frames.
    # (The ``[B, T]`` per-action-token path at ``action_dit.py:348-392`` does
    # exist, but it hangs off RTC and ``wam.yaml:40-42`` has
    # ``enabled: false``, so it is inactive in the default topology.)
    diffusion = DiffusionSpec(
        timesteps={
            "video": TimestepSampler(UniformShift(shift=video_shift)),
            "action": TimestepSampler(UniformShift(shift=action_shift)),
        },
        # Five terms. The weights map one-to-one onto
        # ``0.5*video + 1.0*action + 0.5*future_delta
        #   + 0.1*semantic_future + track_weight*track_raw``
        # at ``losses.py:992-998``
        # (``wam.yaml:62`` / ``:63`` / ``:71`` / ``:76``).
        # This is why ``losses`` has to be a **list** rather than two fields
        # named video and action.
        losses=(
            LossSpec(name="video_flow", head="video_flow", weight=0.5),
            LossSpec(name="action_flow", head="action_flow", weight=1.0),
            LossSpec(name="future_delta", head="future_delta", weight=0.5),
            # ``1 - cos`` rather than MSE (``losses.py:58-72``): this
            # self-distillation term aligns direction, not magnitude, so
            # swapping in MSE changes the training problem.
            LossSpec(name="semantic_future", head="semantic_future",
                     objective=CosineDistance(), weight=0.1),
            # The 0.1 on track is an **upper bound**, not a constant weight:
            # ``weight = loss_weight * min(1.0, step / warmup_steps)`` at
            # ``track_bridge.py:157-163``, with ``warmup_steps: 2000``
            # (``configs/task/4D_distillation.yaml:28-30``).
            # This is exactly what I4′ ``WeightRamp`` is for: the model does
            # not count steps itself, the trainer passes step in from
            # ``batch["global_step"]``.
            # Note that ``LossSpec.weight_at`` **ignores** ``weight`` when a
            # ramp is present, so the final value goes in ``end_value`` rather
            # than in ``weight``.
            LossSpec(name="track_align", head="track_align",
                     ramp=WeightRamp(start_step=0, end_step=2000,
                                     start_value=0.0, end_value=0.1)),
        ),
        schedule=RectifiedFlow(),
        gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        sampler=EulerSampler(),
        # Two tiers = clean condition frames + noisy future frames (the zeros
        # segment at ``losses.py:624-634``).
        noise_levels=(0.0, 1.0),
    )

    # L4. Five heads; the last three are everything that distinguishes this
    # model from fastwam.
    #
    # **Where the weights live**: ``HeadSpec`` also has a ``weight`` (the one
    # field in §3.7 that YAML may override). It is **never used** here; all
    # five weights are written on ``LossSpec``. Being able to write the same
    # number in two places is a redundancy in the current signatures, noted
    # here: Lane C should rule that only one place survives, otherwise ``0.5``
    # can appear on both objects with only one of them taking effect.
    heads = (
        # 48 = ``out_dim`` (``wam.yaml:25``), whose source is
        # ``geometry.latent_channels``: entry channels and exit channels are
        # the same number, so this reads the geometry instead of repeating a
        # literal. The actual Linear width inside the head is
        # 48 x prod(patch) = 192 (``wan_video_dit.py:406``), which does not go
        # into output_width.
        HeadSpec(name="video_flow",
                 impl=Spec(build_flow_head, {"form": "patch_flow"}),
                 source=HiddenStateRef(stream="video"),
                 output_width=geometry.latent_channels,
                 needs_timestep=True),
        # ``Linear(1024, 80)`` at ``action_dit.py:128`` (``action_dim 80``,
        # ``configs/data/pretrain.yaml:34``). A bare Linear
        # (``return self.head(tokens)`` at ``:443``), so no modulation and
        # ``needs_timestep=False``.
        HeadSpec(name="action_flow",
                 impl=Spec(build_flow_head, {"form": "linear"}),
                 source=HiddenStateRef(stream="action"),
                 output_width=geometry.action_dim),
        # future_delta: ``Sequential(LayerNorm(3072), Linear(3072, 384))``
        # (``wan_video_dit.py:557-563`` / ``:657-667``) reshaped to
        # ``(2, 192)``, where ``patch_dim = 48 x 1 x 2 x 2 = 192``
        # (``world_action_model.py:248``) and ``future_delta_num_frames =
        # (33-1)//4//4 = 2`` (``:243-247``), so output_width is **192** (the
        # per-frame width); the frame count 2 is a separate axis and does not
        # go into this field. No modulation, so needs_timestep=False.
        #
        # **Noted**: this head reads the 96 delta tokens at ``[480:576]`` of
        # the video stream, but ``HiddenStateRef`` has **no ``segment``
        # field**, so "which segment to read" cannot be expressed; today it can
        # only be written as "consumes the video stream's final output". Same
        # root cause as the geometry note above (delta tokens have no position
        # of their own); both get resolved once Lane A adds condition-frame and
        # delta fields to ``GeometrySpec``.
        HeadSpec(name="future_delta",
                 impl=Spec(build_flow_head, {"form": "norm_linear"}),
                 source=HiddenStateRef(stream="video"),
                 output_width=192),
        # semantic_future: **no projection layer**, ``1 - cos`` computed
        # directly on the raw 3072-dim hidden states (``losses.py:58-72``).
        # student = the delta segment ``[480:576]`` of **layer 8**,
        # teacher = ``[384:480]`` of **layer 20**, detached
        # (``losses.py:959-976``; layer numbers from ``wam.yaml:74-75``).
        #
        # **Noted, two things that cannot be expressed**: (1)
        # ``HeadSpec.source`` is a **single** ref, so it cannot hold the
        # teacher side; (2) there is no ``detach`` field, yet the teacher must
        # be detached, otherwise this self-distillation term pushes gradients
        # back into layer 20 and becomes a different optimization problem.
        # Only the student side is declared here; the teacher side and detach
        # are left for Lane C to rule on (either ``HeadSpec`` gains a
        # ``reference: HiddenStateRef | None``, or alignment losses like this
        # stop going through ``HeadSpec`` and get their own category).
        HeadSpec(name="semantic_future",
                 impl=Spec(build_flow_head, {"form": "identity"}),
                 source=HiddenStateRef(stream="video", layer=8),
                 output_width=video.residual_width),
        # track_align: reads **layer 14** (``source_layer: 14`` at
        # ``configs/task/4D_distillation.yaml:24``) ->
        # ``video_to_track = Linear(3072, 1024)`` (``track_bridge.py:26``) ->
        # the tail of ``student_projector``, ``Linear(2048, 1430)``
        # (``:38-43``), with ``teacher_dim 1430``
        # (``4D_distillation.yaml:27``), so output_width is the final 1430.
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
        groups=(group,),
        diffusion=diffusion,
        heads=heads,
        # One text cross-attention slot per layer
        # (``wan_video_dit.py:330-331``), consumed by both streams.
        # **Difference from vpp2**: there is **no CLIP** here
        # (``require_clip_embedding: false`` at ``wam.yaml:31``, asserted off
        # at ``wan_video_dit.py:472``).
        # ``width=4096`` = ``text_dim`` (``wam.yaml:24`` for video, ``:49`` for
        # action, same value on both streams), the input width of the
        # ``text_embedding`` of both streams.
        # proprio has the same shape as in vpp2 and is appended to the tail of
        # this slot (``input_builder.py:550-555`` and
        # ``Linear(80, 4096)`` + ``cat(dim=1)`` at
        # ``proprio_encoder.py:46-47``). Its output width matching the text
        # slot is not a coincidence but a consequence of being concatenated
        # into the same context, and V7 now checks it.
        # ``source_width=80`` is the only one of the three models that is not
        # 8, which is why it has to be configurable.
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
        visibility=_internw0_delta_visibility(geometry.tokens_per_frame,
                                              condition_frames),
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
