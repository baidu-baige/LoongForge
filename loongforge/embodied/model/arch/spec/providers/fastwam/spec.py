# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``fastwam`` provider: the baseline of the
``dual_stream_wan`` family.

The family defaults in ``dual_stream_wan.py`` are fastwam's own measured
values, so this provider overrides no stream scalar at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....diffusion.loss import LossSpec
from ....diffusion.schedule import RectifiedFlow
from ....diffusion.spec import DiffusionSpec
from ....diffusion.stepping import EulerSampler, UniformSteps
from ....diffusion.timesteps import TimestepSampler, UniformShift
from ....heads.flow import build_flow_head
from ....heads.hidden_ref import HiddenStateRef
from ....heads.spec import HeadSpec
from ....streams.blocks import ActionStream, WanVideoStream
from ....streams.conditions import ConditionSpec, InjectionKind
from ....streams.encoders import linear_token_encoder
from ....streams.engine import AttentionGroup
from ....streams.spec import TokenTimestep
from ....visibility.combine import Visibility
from ....visibility.presets import get_preset
from ....visibility.rules import prefix_only, same_stream, stream_edge
from ...core import Spec, register_provider
from ...overrides import reject_config, resolve_geometry, resolve_scalars
from ...world import WorldModelSpec
from ..dual_stream_wan import STREAM_SCALAR_KEYS, _dual_stream_wan_streams


_FASTWAM_STREAM_SCALARS: dict[str, int] = {}
"""fastwam overrides **no stream scalar at all**.

The ten defaults of ``_dual_stream_wan_streams`` are its measured values
(``model/fastwam/modeling_configuration_fastwam.py:75-96``: video
3072/24x128/30/14336, action 1024/24x128/30/4096). The empty table is
deliberate: copying the same numbers again would give one fact a second
source, and the family defaults were chosen to be "baseline fastwam" in the
first place. YAML can still override key by key (see ``resolve_scalars``).
"""


def _fastwam_visibility(tokens_per_frame: int) -> dict[str, Visibility]:
    """Build fastwam's L1 visibility.

    First-hand facts: ``model/fastwam/mot/fastwam.py:433-456``.

    Three writes, same shape as VPP2: video->video from
    ``build_video_to_video_mask``, action->action all True (``:452``), and
    action->**the first latent frame** (``:455``, where
    ``first_frame_tokens = min(video_tokens_per_frame, video_seq_len)``).
    Again there is **no video -> action edge**.

    **Two mismatches with the fastwam row of design doc §2. The code wins and
    the mismatch is recorded here:**

      1. §2 writes the action side as ``same_stream() & chunk_causal()``;
         code ``:452`` is all True with no chunk causality at all. Adding that
         rule would hide the action stream's own future chunks.
      2. §2's video side maps to ``first_frame_causal``, but **the preset of
         that name and the code of that name are not the same thing**: the
         code (``model/fastwam/wan/dit.py:595-600``) starts all True and then
         removes the edges from the first frame's rows to later frames, so
         every frame except the first is fully bidirectional and can even see
         future frames. The preset ``frame_causal() | prefix_only(n)`` is
         per-frame causal instead.
         None of the nine atomic rules can express the code's shape ("only
         the first frame is restricted"). The **stricter** preset is used
         here: what the equivalence tests need is "never grant extra
         visibility". Being too permissive would let the model see tokens it
         should not while the numbers still converge, and that kind of
         deviation cannot be detected.

    The ``same_stream() &`` term cannot be dropped: the preset only constrains
    the frame axis, and action tokens carry frame labels too (the layout tags
    them 0..F-1 according to ``action_video_freq_ratio``). Without that term,
    ``frame_causal()`` would open action columns with smaller frame ids to
    video query rows, which is the opposite of "there is no video -> action
    edge" above.

    Args:
        tokens_per_frame: Video tokens per latent frame, which is also the
            prefix length the action stream may read.

    Returns:
        ``{"video": ..., "action": ...}`` visibility rules.
    """
    first_frame_causal = get_preset("first_frame_causal")
    return {
        "video": same_stream() & first_frame_causal(
            prefix_tokens=tokens_per_frame),
        "action": same_stream() | (
            stream_edge("video", "action") & prefix_only(tokens_per_frame)
        ),
    }


@register_provider("fastwam")
def _fastwam(config: Any = None, *,
             geometry: Mapping[str, Any] | None = None,
             video_shift: float = 5.0,
             action_shift: float = 5.0,
             inference_steps: int = 20,
             action_width: int = 7,
             proprio_dim: int | None = 8,
             **stream_scalars: Any) -> WorldModelSpec:
    """Build the fastwam spec; ``--model-name fastwam``.

    The baseline member of the family, already present in the landing repo.

    Three structural differences from vpp2, all numeric or topological rather
    than contract changes:

    1. **The video timestep is not a constant.** fastwam really samples t for
       both streams (``video_scheduler`` and ``action_scheduler`` carry the
       **same numbers** at ``modeling_configuration_fastwam.py:98-103``:
       ``train_shift: 5.0`` / ``infer_shift: 5.0``), so both get
       ``UniformShift(5.0)`` rather than vpp2's ``Constant(1.0)``.
       **Design doc §2's fastwam row says video 3.0 / action 1.0, which
       contradicts the code**; the code wins and the mismatch is recorded.
    2. **``seperated_timestep: True``** (``:82``). Its actual shape in code is
       ``token_timesteps = ones(B, F, tokens_per_frame) * t.view(B,1,1)``
       followed by ``token_timesteps[:, 0, :] = 0``
       (``wan/dit.py:762-779``; the non-separated branch at ``:780`` raises
       NotImplementedError), that is **one ``[B]`` draw broadcast to all
       frames, with the first latent frame pinned clean**.
       So it is **not** ``BroadcastSpec(axes=("frame",))``, which would mean F
       independent draws -- the sampling code at
       ``action/schedulers.py:51-58`` indeed produces only ``[B]``. What it is
       instead is ``TokenTimestep(clean_prefix=1)`` on the video stream: the
       single draw is broadcast over the frame axis inside L2 and the video
       modulation becomes ``[B, S, 6, D]``. ``noise_levels`` and the
       ``prefix_only(tokens_per_frame)`` rule in L1 stay, but they do not
       stand in for this: they decide what attends to what, whereas this
       decides the shift / scale / gate the clean frame's own residual gets.
       The action stream stays per-sample.
    3. **Single condition frame, two cameras.** ``num_video_frames: 9`` is the
       frame count **after** subsampling (``observation_delta_indices`` in
       ``data_configuration_fastwam.py:64-73`` shows the raw count is
       ``(T-1) x ratio + 1``), and those 9 frames are what the VAE sees.

    The default ``inference_steps`` of 20 comes from the code signature
    (``mot/fastwam.py:982``); no config in the repo overrides it, and it is
    neither the same source nor the same value as vpp2's 10.

    Args:
        config: Positional config object, rejected (see ``reject_config``).
        geometry: The YAML ``geometry:`` section, merged over the defaults.
        video_shift: ``shift`` of the video stream's ``UniformShift``.
        action_shift: ``shift`` of the action stream's ``UniformShift``.
        inference_steps: Euler steps at inference.
        action_width: Packed action dimension; entry width of the action
            stream and output width of the ``action_flow`` head.
        proprio_dim: Raw proprioception width, or ``None`` to drop the
            proprioception condition entirely.
        **stream_scalars: Per-stream width and depth overrides.

    Returns:
        The validated ``WorldModelSpec`` for fastwam.

    Raises:
        NotImplementedError: ``config`` is not ``None``.
        ValueError: An unknown scalar or geometry key, or a V1-V9 failure.
    """
    reject_config("fastwam", config)

    scalars = resolve_scalars("fastwam", _FASTWAM_STREAM_SCALARS,
                              stream_scalars, STREAM_SCALAR_KEYS)
    video, action = _dual_stream_wan_streams(
        video_impl=Spec(WanVideoStream),
        action_impl=Spec(ActionStream),
        video_token_timestep=TokenTimestep(clean_prefix=1),
        **scalars,
    )
    group = AttentionGroup(streams=(video, action),
                           num_layers=video.num_layers)

    # Geometry from ``configs/models/embodied/fastwam.yaml``:
    # ``num_video_frames: 9`` (:27), ``image_size: 224`` (:28),
    # ``action_horizon: 32`` (:13), ``action_video_freq_ratio: 4`` (:35).
    # The two cameras are concatenated along **width**
    # (``cat(dim=-1)`` at ``fastwam_transform.py:57-65``, libero_2cam).
    # The VAE keeps the defaults 16/4: fastwam uses **Wan2.2's
    # ``WanVideoVAE38``** (``model/fastwam/wan/vae.py:1469-1499``:
    # ``z_dim=48``, ``upsampling_factor=16``,
    # ``temporal_downsample_factor=4``), a **different generation** from
    # vpp2's Wan2.1 (stride 8, 16 channels).
    # ``latent_channels=48`` is that ``z_dim``, and also the ``out_dim`` at
    # ``modeling_configuration_fastwam.py:78``: two sides of one number, whose
    # single source is here.
    # Derived: latent (3, 14, 28) -> patch (3, 7, 14), so 98 tokens per frame
    # and 294 tokens in total; ``num_chunks = 32 / 4 = 8``.
    geometry = resolve_geometry("fastwam", dict(
        frame_height=224,
        frame_width=224,
        num_frames=9,
        num_views=2,
        latent_channels=48,
        action_horizon=32,
        action_video_freq_ratio=4,
        # Action stream entry width = ``action_flow`` head output width
        # (same note as in the vpp2 provider).
        action_dim=action_width,
    ), geometry)

    diffusion = DiffusionSpec(
        timesteps={
            "video": TimestepSampler(UniformShift(shift=video_shift)),
            "action": TimestepSampler(UniformShift(shift=action_shift)),
        },
        # ``lambda_video`` and ``lambda_action`` are both 1.0 (``:104-106``)
        losses=(
            LossSpec(name="video_flow", head="video_flow"),
            LossSpec(name="action_flow", head="action_flow"),
        ),
        # Both master schedulers use the 1000-step training timestep
        # (``video/action_num_train_timesteps``, ``mot/fastwam.py:53-56``), and
        # ``pre_dit`` receives ``sigma * 1000`` (``action/schedulers.py:51-58``).
        schedule=RectifiedFlow(num_train_timesteps=1000),
        # Inference runs the action stream only: ``infer_action``
        # (``mot/fastwam.py:982``) calls ``mot.prefill_video_cache``
        # (``:1081``) once, then each step calls only
        # ``forward_action_with_video_cache`` (``:774`` / ``:1106``).
        sampling_gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        solver=EulerSampler(),
        # Two levels = first latent frame clean, the rest noisy
        # (``wan/dit.py:645``).
        noise_levels=(0.0, 1.0),
    )

    heads = (
        # 48 is the Wan2.2 VAE latent channel count, sourced from
        # ``geometry.latent_channels``: entry channels and exit channels are
        # one number, so the geometry is read here instead of repeating the
        # literal.
        # The Linear inside the head is actually 48 x prod(patch) = 192 wide;
        # the unflatten happens inside the head and does not enter
        # ``output_width``.
        HeadSpec(name="video_flow",
                 impl=Spec(build_flow_head, {"form": "patch_flow"}),
                 source=HiddenStateRef(stream="video"),
                 output_width=geometry.latent_channels,
                 needs_timestep_embedding=True),
        # ``Linear(hidden_dim, action_dim)`` at
        # ``model/fastwam/action/dit.py:132`` gives 7 (``action_dim: 7``,
        # ``fastwam.yaml:12``). A bare Linear with no modulation, so
        # ``needs_timestep_embedding=False``, unlike the video head's ``Head``
        # (``wan/dit.py:487-508``) which carries a ``(1, 2, dim)`` modulation.
        HeadSpec(name="action_flow",
                 impl=Spec(build_flow_head, {"form": "linear"}),
                 source=HiddenStateRef(stream="action"),
                 output_width=geometry.action_dim),
    )

    return WorldModelSpec(
        model_type="fastwam",
        geometry=geometry,
        attention_groups=(group,),
        diffusion=diffusion,
        heads=heads,
        # Same shape as vpp2: one text cross-attention slot per layer, fed to
        # both streams.
        # ``width=4096`` is the ``text_dim`` at
        # ``modeling_configuration_fastwam.py:90`` (video) and ``:105``
        # (action). Both streams agree, so one condition suffices; it is the
        # input width of each stream's ``text_embedding`` Linear.
        # Proprioception is again one token appended to the tail of this slot
        # (``Linear(proprio_dim, text_dim)`` at ``mot/fastwam.py:78`` ->
        # ``cat([context, proprio_token], dim=1)`` at ``:268``).
        # One difference from vpp2: here ``proprio_dim`` **may be None**
        # (``modeling_configuration_fastwam.py:60``; None means the whole
        # ``proprio_encoder`` is not built, and ``:861-862`` explicitly
        # refuses proprioception input). In that case the condition is not
        # declared at all rather than declared with ``enabled=False``, because
        # the latter would still build a weight in ``condition_encoders`` that
        # the checkpoint does not contain.
        conditions=(
            ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                          targets=("video", "action"), width=4096,
                          source_key="context"),
        ) + ((
            ConditionSpec(name="proprio", kind=InjectionKind.APPEND_TO_SLOT,
                          targets=("video", "action"), width=4096,
                          append_to="text", source_width=proprio_dim,
                          encoder=Spec(linear_token_encoder),
                          source_key="proprio"),
        ) if proprio_dim is not None else ()),
        visibility=_fastwam_visibility(geometry.tokens_per_frame),
        # ``mot/fastwam.py:433-456`` produces a dense ``[S, S]`` bool mask.
        attention_backend="dense",
    )
