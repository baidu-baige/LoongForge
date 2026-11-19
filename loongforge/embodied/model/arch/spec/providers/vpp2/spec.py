# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""``vpp2`` provider: the ``dual_stream_wan`` family with an i2v
video stream.

Follows ``docs/design/mot_spec_model_matrix.zh-CN.md`` §4.3 / §4.4. The
code-level facts come from ``vpp2.py`` / ``mot.py`` under
``projects/video-prediction-policy-2/src/vpp2/models/wan21_14b/``, cited by
line number at each use.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ....diffusion.loss import LossSpec
from ....diffusion.schedule import RectifiedFlow
from ....diffusion.spec import DiffusionSpec
from ....diffusion.stepping import EulerSampler, UniformSteps
from ....diffusion.timesteps import Constant, TimestepSampler, UniformShift
from ....heads.flow import build_flow_head
from ....heads.hidden_ref import HiddenStateRef
from ....heads.spec import HeadSpec
from ....streams.blocks import ActionStream, WanVideoStream
from ....streams.conditions import ConditionSpec, InjectionKind
from ....streams.encoders import linear_token_encoder
from ....streams.engine import AttentionGroup
from ....visibility.combine import Visibility
from ....visibility.presets import get_preset
from ....visibility.rules import same_stream, stream_edge
from ...core import Spec, register_provider
from ...overrides import reject_config, resolve_geometry, resolve_scalars
from ...world import WorldModelSpec
from ..dual_stream_wan import STREAM_SCALAR_KEYS, _dual_stream_wan_streams


_VPP2_STREAM_SCALARS: dict[str, int] = {
    # First-hand source: ``video_dit_config`` (:169-190) and
    # ``action_dit_config`` (:191-200) in ``configs/libero/train_action.yaml``.
    # Matches the VPP2 row of model_matrix §3: video 5120/5120,
    # action 512/5120 (residual width / attention width).
    "video_residual_width": 5120,
    "video_num_heads": 40,
    "video_head_dim": 128,
    "video_num_layers": 40,
    "video_ffn_width": 13824,
    "action_residual_width": 512,
    "action_num_heads": 40,
    "action_head_dim": 128,
    "action_num_layers": 40,
    "action_ffn_width": 2048,
}
"""VPP2's stream scalars.

These are [YAML]-overridable, with the defaults written in the provider (the
15-line YAML of §4.1 overrides this table field by field). The family defaults
are fastwam's, and VPP2 differs from them on every entry.
"""


def _vpp2_visibility(video_prefix_frames: int) -> dict[str, Visibility]:
    """Build VPP2's L1 visibility.

    Keys are **stream names**; each value is the rule on that stream's query
    side.

    First-hand facts (``vpp2.py:879-906``, three writes into one ``[S, S]``
    bool matrix):

      - ``:889`` video -> video;
      - ``:894`` action -> action, all True;
      - ``:905`` action -> the first N video frames
        (``visible_video_tokens = tokens_per_frame x visible_frames``).

    There is **no video -> action edge**, so the ``video`` side is
    ``same_stream()``. That is also the structural reason inference can run
    the action head alone.

    Three differences from the §4.3 example, all to be ruled on before the
    equivalence tests. This function follows the example and records the
    differences here rather than silently picking a side:

      1. the example has ``chunk_causal()``; the current code at ``:894`` is
         all True. Adding that rule would hide the action stream's own future
         chunks;
      2. the example's comment says "the most recent N frames", while
         ``:905`` takes the **first** N frames (a prefix). A sliding window
         corresponds to ``frame_causal(N)``, a prefix to ``prefix_only(n)``;
      3. today's clean condition is two methods returning ``[B]`` bools
         (``:908-921`` / ``:923-936``), which INV-1 forbids from entering
         ``structural``. ``noise_gate`` lifts it into a rule over per-token
         labels, and that is the subject of the 12/04 gate.

    Args:
        video_prefix_frames: Number of leading video frames the action
            stream may read.

    Returns:
        ``{"video": ..., "action": ...}`` visibility rules.
    """
    noise_gated_history = get_preset("noise_gated_history")
    return {
        "video": same_stream(),
        "action": same_stream() | (
            stream_edge("video", "action")
            & noise_gated_history(window=video_prefix_frames)
        ),
    }


@register_provider("vpp2")
def _vpp2(config: Any = None, *,
          geometry: Mapping[str, Any] | None = None,
          video_prefix_frames: int = 3,
          action_shift: float = 5.0,
          inference_steps: int = 10,
          action_width: int = 7,
          proprio_dim: int = 8,
          **stream_scalars: Any) -> WorldModelSpec:
    """Build the VPP2 spec; this is what ``--model-name vpp2`` resolves to.

    **Differences from the code examples in §4.3 / §4.4.** The examples are
    illustrative; the authority is ``mot_layer_interfaces.zh-CN.md`` plus the
    dataclasses already landed. Nine differences, each recorded rather than
    silently resolved:

    1. ``GeometrySpec(image_size=, temporal_ratio=)`` becomes
       ``frame_height=`` / ``frame_width=`` / ``action_video_freq_ratio=``,
       per the field table of interfaces §3.1. Height and width are separate
       because non-square inputs exist.
    2. ``WorldModelSpec(streams=, routes=)`` becomes
       ``attention_groups=(AttentionGroup(streams=..., num_layers=..., routes=...),)``.
       "Which streams share one joint attention" has to be visible in the
       type (EWAM's VL stream does not share one).
    3. Two per-stream ``DiffusionSpec`` objects become one, with ``timesteps``
       indexed by stream name and ``losses`` a list. The number of loss terms
       differs per model (2 / 3 / 5), and only a list holds that.
    4. ``DiffusionSpec(timestep=, loss=, sample_shape=)`` becomes
       ``timesteps=`` / ``losses=``. How many t values are drawn is decided by
       ``TimestepSampler.broadcast.axes``, so there is no separate
       ``sample_shape``.
    5. ``LossSpec(term=, reduce=, ignore_pad=)`` becomes
       ``LossSpec(name=, head=)``. ``reduce`` and ``ignore_pad`` belong to
       ``broadcast`` and the ``is_padding`` labels (interface I2) instead of being
       two more knobs on the loss.
    6. ``HeadSpec(inputs=(HiddenStateRef(layer=-1),))`` becomes
       ``source=HiddenStateRef(stream=...)``. Taking the final output is
       ``layer=None``; ``-1`` would be confused with a joint layer id.
    7. ``HeadSpec(inference=False)`` becomes
       ``sampling_gate=UniformSteps(skip={"video"})``. "Drop the video stream at
       inference" is a stepping policy, not a property of a head, otherwise
       the same fact has two sources.
    8. ``Spec(ShiftedUniform, ...)`` / ``Spec(Constant, ...)`` become
       ``TimestepSampler(UniformShift(...))`` /
       ``TimestepSampler(Constant(...))``. Distributions are plain
       dataclasses; ``Spec`` is used only for implementations that become an
       ``nn.Module``.
    9. ``Spec(EulerSampler, {"num_steps": N})`` becomes ``EulerSampler()``
       plus ``UniformSteps(num_steps=N)``. The step count belongs to the
       ``SteppingGate``, decoupled from the sampler.

    Args:
        config: Positional config object, rejected. It exists because
            ``build_world_model_spec`` historically called ``provider(config)``;
            applying YAML overrides is step 3 of ``build_world_model_spec`` and
            belongs to the config layer, so a non-``None`` value raises
            instead of being ignored.
        geometry: The YAML ``geometry:`` section, merged over the provider's
            defaults before anything is derived from the geometry.
        video_prefix_frames: Leading video frames the action stream reads.
        action_shift: ``shift`` of the action stream's ``UniformShift``.
        inference_steps: Euler steps at inference.
        action_width: Packed action dimension, both the action stream's entry
            width and the ``action_flow`` head's output width.
        proprio_dim: Raw proprioception width before the token encoder.
        **stream_scalars: Per-stream width and depth overrides, validated
            against ``STREAM_SCALAR_KEYS``.

    Returns:
        The validated ``WorldModelSpec`` for VPP2.

    Raises:
        NotImplementedError: ``config`` is not ``None``.
        ValueError: A stream scalar or geometry key is unknown, or one of
            V1-V9 fails.
    """
    reject_config("vpp2", config)

    # 1. Stream scalars. An unconsumed key raises; spec_config §3.8 rule 3.
    scalars = resolve_scalars("vpp2", _VPP2_STREAM_SCALARS, stream_scalars,
                              STREAM_SCALAR_KEYS)

    video, action = _dual_stream_wan_streams(
        # ``context_proj`` is a **pure implementation detail** (how the
        # projector layers are arranged), so it travels in ``Spec.params``
        # and gets no spec field:
        #   clip_image -> ``mlp_proj``, today ``img_emb = MLPProj(1280, dim)``
        #     (``wan_video_dit.py:398``, five stages with two LayerNorms);
        #   text -> ``mlp2``, today ``text_embedding`` (``:373-377``).
        # The two cannot be swapped: swapping them makes the same checkpoint
        # fail to match by key.
        video_impl=Spec(WanVideoStream, {
            "context_proj": {"clip_image": "mlp_proj", "text": "mlp2"},
        }),
        # ``cross_attn_norm=False`` applies to the **action stream only**:
        # today ``action_dit.py:114-120`` builds its ``DiTBlock`` without
        # passing ``cross_attn_norm``, so it takes the default ``False`` at
        # ``wan_video_dit.py:222``, which makes ``norm3`` an
        # ``elementwise_affine=False`` norm with **no** weight/bias keys.
        # On the video side all four configs write ``cross_attn_norm: true``.
        # The symptom of omitting this line is 2 x num_layers unexpected keys
        # under strict loading, all of them on the action stream, which is
        # easily misread as "the action stream weights are misaligned".
        # In fastwam both streams use an affine ``nn.LayerNorm``
        # (``wan/dit.py:426``, ``action/dit.py:121-129``), so that provider
        # keeps the default True.
        action_impl=Spec(ActionStream, {"cross_attn_norm": False}),
        # The i2v video stream has two cross-attention slots
        # (``I2VCrossAttention`` runs twice); the action stream has only the
        # text one (``action_dit.py`` uses the plain ``CrossAttention``).
        video_cross_kv_slots=("clip_image", "text"),
        **scalars,
    )

    # 2. One attention group, two streams in it, no KV routes.
    #    Joint depth == each stream's own depth (ONE_TO_ONE, resolved by V2).
    #    ``routes=()`` is deliberate: under VPP2's default config, cross-stream
    #    information flows through the **joint attention** itself
    #    (``mot.py:916-935`` concatenates both streams' q/k/v, runs attention
    #    once, then splits by seq_lens).
    #    The ``video_kv_projections`` inside ActionDiT
    #    (``action_dit.py:124-146``: per-layer K/V projections in the action
    #    stream that read frozen video features) is what ``KVRoute`` would
    #    model, but it hangs off ``use_video_tokens``, which defaults to False
    #    (``action_dit.py:67``) and is enabled by neither training config, so
    #    the default topology has no such edge.
    group = AttentionGroup(streams=(video, action),
                           num_layers=video.num_layers)

    # 3. L3. The two streams use different timestep distributions, which is
    #    why ``timesteps`` is indexed by stream name.
    diffusion = DiffusionSpec(
        timesteps={
            # video: ``train_sampling_strategy: pure_noise``
            # (``train_action.yaml:204``; the code at ``vpp2.py:1608`` pins it
            # to num_train_timesteps), so t is always at the pure-noise end.
            # Expressed as ``Constant`` rather than an ``if not add_noise``
            # inside the stream: the latter would make "this stream's noise
            # level" invisible in the spec, and ``noise_gate`` could not read
            # it.
            # The ``beta_alpha: 7.0 / beta_beta: 1.0`` configured in the same
            # section **takes no part in sampling** under pure_noise, so no
            # ``BetaDist`` is declared here; declaring one would add a knob
            # that appears to be in effect but is not.
            "video": TimestepSampler(Constant(value=1.0)),
            # action: ``train_shift: 5.0`` (``train_action.yaml:211``), and
            # ``infer_shift`` in the same section is also 5.0, so training and
            # inference agree and one distribution slot suffices.
            # The video stream has train 5.0 / infer 3.0, which **differ**, but
            # it is dropped entirely at inference by the gate below
            # (``skip={"video"}``), so that 3.0 has nowhere to land. Expressing
            # "two distributions for one stream, train vs infer" would need
            # another field on DiffusionSpec; add it when some model actually
            # runs the video stream at inference.
            "action": TimestepSampler(UniformShift(shift=action_shift)),
        },
        # ``lambda_video`` and ``lambda_action`` are both 1.0
        # (``train_action.yaml:217-218``)
        losses=(
            LossSpec(name="video_flow", head="video_flow"),
            LossSpec(name="action_flow", head="action_flow"),
        ),
        schedule=RectifiedFlow(),
        # ``skip={"video"}`` is the ``HeadSpec(inference=False)`` of the §4.4
        # example: "drop the video stream at inference" is a stepping policy,
        # not a property of a head.
        # The step count comes from ``evaluation.num_inference_steps: 10``
        # (``train_action.yaml:224``, same value as ``policy/VPP2/deploy.yml:21``).
        sampling_gate=UniformSteps(num_steps=inference_steps,
                          skip=frozenset({"video"})),
        solver=EulerSampler(),
        # Two levels: clean (condition frames) and noisy. A non-empty tuple is
        # what makes the ``noise_gate`` rule usable, which the second rule in
        # ``_vpp2_visibility`` below depends on.
        noise_levels=(0.0, 1.0),
    )

    # 4. L0. The geometry is defined before the head table because the
    #    ``video_flow`` head takes its ``output_width`` from
    #    ``geometry.latent_channels``: the entry channel count and the exit
    #    channel count are the same number.
    #
    # Values from the libero config (``train_action.yaml:77-83``):
    # ``num_frames: 65`` / ``action_horizon: 32`` /
    # ``action_video_freq_ratio: 4`` / ``video_size: [224, 448]``.
    # The 448 is the width **after** concatenation, so it is split here into
    # 224 x ``num_views=2`` (``concat_multi_camera: horizontal``,
    # ``num_output_cameras: 2``).
    #
    # ``vae_spatial_stride=8`` must be given explicitly: VPP2 uses the
    # **Wan2.1 VAE** (``wan_video_vae.py:648-650``: ``upsampling_factor = 8``,
    # ``temporal_downsample_factor = 4``, ``z_dim = 16``), while the
    # ``GeometrySpec`` default of 16 belongs to the Wan2.2 generation
    # (fastwam's ``WanVideoVAE38``). ``latent_channels=16`` is the other side
    # of the same fact: 16 channels means Wan2.1 means 8x spatial compression.
    # It equals the default and is still written out, because it comes as a set
    # with ``vae_spatial_stride`` and reading one without the other suggests
    # they can be changed independently.
    # Derived: latent (17, 28, 56) -> patch (17, 14, 28), so 392 tokens per
    # frame and 6664 tokens in total; ``num_chunks = 32 / 4 = 8``.
    # These fields are marked [YAML] (spec_config §3.3); the override is
    # applied in step 3 of ``build_world_model_spec``.
    geometry = resolve_geometry("vpp2", dict(
        frame_height=224,
        frame_width=224,
        num_frames=65,
        num_views=2,
        vae_spatial_stride=8,
        latent_channels=16,
        action_horizon=32,
        action_video_freq_ratio=4,
        # The action stream's entry Linear input width and the ``action_flow``
        # head's output width are the **same number**
        # (``action_encoder`` at ``action_dit.py:148`` and ``head`` at
        # ``:150``), so it belongs to the geometry and both ends read one
        # source. For libero it is 7 x 1 = 7.
        action_dim=action_width,
    ), geometry)

    # 5. L4. Both heads consume their stream's **final output**
    #    (layer=None), but **only the video head consumes the timestep**:
    #    ``Head.forward(x, t)`` (``wan_video_dit.py:287-297``) carries a
    #    ``(1, 2, dim)`` modulation parameter, exactly the two ``HEAD_SLOTS``.
    #    The action side is a bare ``Linear`` (``action_dit.py:150`` builds an
    #    ``nn.Linear``, and ``post_dit`` at ``:423`` calls it once). The
    #    modulated ``ActionHead`` at ``:20-33`` of the same file is **never
    #    instantiated**; it is dead code and must not be used as a reference.
    heads = (
        # ``output_width`` reads ``geometry.latent_channels`` (= 16, the
        # ``out_dim`` at ``wan_video_dit.py:328``) instead of writing another
        # literal 16: the stream's entry convolution and the head's exit
        # Linear must use the same channel count, and two copies would let one
        # side change with a new VAE while shapes still line up (the patch
        # unflatten absorbs the difference into ``x prod(patch_size)``).
        # That ``x prod(patch_size)`` factor lives inside the head (``:286``)
        # and does not enter ``output_width``, otherwise one number would have
        # two meanings.
        HeadSpec(name="video_flow",
                 impl=Spec(build_flow_head, {"form": "patch_flow"}),
                 source=HiddenStateRef(stream="video"),
                 output_width=geometry.latent_channels,
                 needs_timestep_embedding=True),
        # The action head's width is the **packed** action dimension =
        # raw_action_dim x action_chunk_size (the inverse of ``vpp2.py:244``;
        # the output width of the Linear at ``action_dit.py:150``).
        # For libero it is 7 x 1 = 7 (``train_action.yaml:193`` / ``:166``).
        # It reads ``geometry.action_dim`` rather than ``action_width``:
        # the geometry can be overridden from YAML, and reading one side here
        # and the other there would make entry width and exit width disagree
        # after an override.
        HeadSpec(name="action_flow",
                 impl=Spec(build_flow_head, {"form": "linear"}),
                 source=HiddenStateRef(stream="action"),
                 output_width=geometry.action_dim),
    )

    # 6. Conditions. Four of them, each checked against the code line by line.
    #    ``width`` is the only basis on which the consumer builds weights, so
    #    every condition must give it (a CHANNEL_CONCAT without one is
    #    rejected by ``ConditionSpec`` itself).
    #
    #    The i2v "clean condition frames" are deliberately not a fifth
    #    condition: replacing the first few frames of the noisy sequence along
    #    the time axis (``vpp2.py:815-817``, ``dim=2``) is the same fact as
    #    ``video_prefix_frames`` and the noise_gate rule below, so declaring
    #    it as a condition would create a second source.
    #    The ``cond_latent`` below is a different thing: concatenation along
    #    the channel axis.
    conditions = (
        # The CLIP image feature is the video stream's **own second**
        # cross-attention slot, not a span concatenated into the text slot:
        # ``I2VCrossAttention.forward`` (``:187-204``) computes a separate
        # attention with ``k_img/v_img`` and adds it to the text one.
        # Width 1280 is the input width of ``MLPProj`` (``:302`` / ``:398``).
        # Once it is a separate condition, the ``ctx[:, :257]`` split point no
        # longer needs to exist.
        ConditionSpec(name="clip_image",
                      kind=InjectionKind.CROSS_ATTENTION,
                      targets=("video",), width=1280,
                      source_key="clip_fea"),
        # Text feeds both streams (``text_embedding`` at ``:522`` on the video
        # side, same name at ``action_dit.py:405`` on the action side).
        # Width 4096 = ``video_dit_config.text_dim`` (``vpp2.py:386``).
        ConditionSpec(name="text", kind=InjectionKind.CROSS_ATTENTION,
                      targets=("video", "action"), width=4096,
                      source_key="context"),
        # The 20 channel-concatenated channels. ``pre_dit:482-484``
        # concatenates unconditionally when x has 16 channels, and ``:485``
        # then asserts the result equals ``in_dim=36``:
        #     36 = 16 (this stream's latent) + 4 (mask) + 16 (condition-frame
        #     latent)
        # so the 20 here is 4 + 16. The mask is **generated** by
        # ``_build_condition:444-445`` from the condition frames' temporal
        # length rather than being an independent input, so it is not a second
        # condition but part of the channel count this one contributes.
        # ``source_key`` may be absent (``:419-420``: None means take x's
        # first latent frame), but that default belongs to Lane B
        # materialization, so the batch key is declared explicitly here.
        ConditionSpec(name="cond_latent",
                      kind=InjectionKind.CHANNEL_CONCAT,
                      targets=("video",), width=20,
                      source_key="condition_latents"),
        # Proprioception is not a fifth KV slot but one token **appended to
        # the tail of the text slot**: ``_append_proprio_to_context``
        # (``vpp2.py:520-539``) concatenates the output of
        # ``Linear(proprio_dim, text_dim)`` onto the context, called at
        # ``:843`` (training) and ``:1597`` (inference), after which each
        # stream runs its own ``text_embedding``. So it shares the text slot's
        # per-layer k/v/norm_k.
        # Declaring it as CROSS_ATTENTION would create 3 x num_layers tensors
        # that the checkpoint does not have.
        # ``width=4096`` must equal the host slot's width (V7 checks this);
        # ``source_width`` is the dataset-dependent number: 8 for libero, 16
        # for robodojo (``train_action.yaml:169`` / ``train_100k.yaml:206``).
        ConditionSpec(name="proprio", kind=InjectionKind.APPEND_TO_SLOT,
                      targets=("video", "action"), width=4096,
                      append_to="text", source_width=proprio_dim,
                      encoder=Spec(linear_token_encoder),
                      source_key="proprio"),
    )

    return WorldModelSpec(
        model_type="vpp2",
        geometry=geometry,
        attention_groups=(group,),
        diffusion=diffusion,
        heads=heads,
        conditions=conditions,
        visibility=_vpp2_visibility(video_prefix_frames),
        # Today a dense ``[S, S]`` bool mask is fed straight to the attention
        # kernel (``flash_attention(..., ctx_mask=)`` at ``mot.py:144-157``),
        # so the default backend is recorded as ``dense``. Switching to
        # ``flex`` also requires ``attention_block_size`` (V9), and that size
        # must line up with ``geometry.align_to``. Ownership of ``align_to``
        # is an open item in master plan §10-1, so the backend switch waits
        # for that ruling.
        attention_backend="dense",
    )
