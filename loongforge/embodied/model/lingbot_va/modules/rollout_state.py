# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LingBot-VA under the Apache-2.0 License.
# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
#
# The rollout state machine mirrors upstream's ``wan_va/wan_va_server.py`` (``_reset`` /
# ``_infer`` / ``_compute_kv_cache``), where the same fields are server instance
# attributes because one server serves one episode. Here they are grouped per episode so
# the stateless eval front end can key them by ``episode_id``.

"""Per-episode autoregressive rollout state for LingBot-VA inference.

LingBot-VA is a *streaming* world model: within one episode the transformer KV cache,
the causal VAE ``feat_cache`` and the executed-action history all carry over from step
to step. The shared eval front end (``GenericPredictActionPolicy``) is stateless and
keyed only by ``episode_id``, so this module owns that per-episode state on the model
side.

Two facts about the eval contract shape this design:

* With the default action-chunk cache the model would be called only once per chunk and
  would never see the intermediate observations the closed-loop feedback needs. Eval
  must therefore run with ``disable_action_cache=True`` so ``predict_action`` is invoked
  once per env step; the action queue lives here instead of in the eval layer.
* The transformer KV cache belongs to the (single) transformer instance and cannot be
  partitioned per episode, so only one episode may be active at a time. A new episode is
  recognised by ``episode_step == 0`` and tears the previous state down. An observation
  for a *different* episode arriving mid-rollout means the caller is interleaving
  episodes over one model instance, which cannot be served and therefore raises rather
  than silently resetting the episode that was switched away from.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import torch

from .wan_codec import WanVAEStreamingWrapper

logger = logging.getLogger(__name__)


@dataclass
class LingBotVARolloutState:
    """Mutable autoregressive state for exactly one episode."""

    episode_id: str
    keyframe_stride: int

    # Actions of the current chunk that have not been handed to the env yet.
    action_queue: deque = field(default_factory=deque)
    # Raw per-frame camera dicts observed since the last chunk, awaiting VAE encoding.
    obs_buffer: list = field(default_factory=list)
    # Previous chunk's actions in model-normalized space, fed back into the KV cache.
    executed_actions: torch.Tensor | None = None
    # Latent of the very first observation; conditions the first chunk and is prepended
    # to the first feedback so latent and action frame counts stay aligned.
    init_latent: torch.Tensor | None = None

    # Absolute latent-frame offset, used to build rope grid ids across chunks.
    frame_st_id: int = 0
    # Index of the action being executed within the current chunk.
    exec_step: int = 0
    # Sub-step index (within a predicted frame) of the last executed action.
    prev_j: int = 0
    first_chunk: bool = True
    started: bool = False

    # Text conditioning is constant within an episode, so it is encoded once.
    prompt: str | None = None
    prompt_embeds: torch.Tensor | None = None
    negative_prompt_embeds: torch.Tensor | None = None

    # Streaming VAE wrappers own the causal ``feat_cache``; they are per-episode even
    # though the underlying VAE weights are shared.
    streaming_vae: WanVAEStreamingWrapper | None = None
    streaming_vae_half: WanVAEStreamingWrapper | None = None

    def should_buffer_keyframe(self) -> bool:
        """Whether the observation arriving now is on a keyframe sub-step boundary.

        One keyframe every ``action_per_frame / 4`` executed sub-steps yields exactly
        ``frame_chunk_size * 4`` frames per chunk, which the VAE's x4 temporal downsample
        collapses back into ``frame_chunk_size`` latent frames.
        """
        return (self.prev_j + 1) % self.keyframe_stride == 0

    def advance_exec_step(self, action_per_frame: int) -> None:
        """Record the sub-step index of the action just handed out."""
        self.prev_j = self.exec_step % action_per_frame
        self.exec_step += 1

    def begin_chunk(self) -> None:
        """Clear the per-chunk observation buffer and restart the execution counter."""
        self.obs_buffer = []
        self.exec_step = 0


class RolloutStateStore:
    """Owns the single active :class:`LingBotVARolloutState` and its teardown.

    Teardown is a *callback* rather than direct transformer access so this module stays
    free of model internals: the caller passes whatever needs to happen when an episode
    ends (clearing the transformer KV cache, dropping the streaming VAE state).
    """

    def __init__(self, on_release=None) -> None:
        """Create an empty store; ``on_release`` runs when an episode's state is torn down."""
        self._state: LingBotVARolloutState | None = None
        self._on_release = on_release

    @property
    def active_episode_id(self) -> str | None:
        """Id of the episode currently holding the state, or ``None`` when idle."""
        return self._state.episode_id if self._state is not None else None

    def get(self, episode_id: str) -> LingBotVARolloutState | None:
        """Return the state for ``episode_id``, or ``None`` if it is not the active one."""
        if self._state is not None and self._state.episode_id == episode_id:
            return self._state
        return None

    def start(self, episode_id: str, keyframe_stride: int) -> LingBotVARolloutState:
        """Release any previous episode and begin a fresh one."""
        self.release()
        self._state = LingBotVARolloutState(
            episode_id=episode_id, keyframe_stride=keyframe_stride
        )
        return self._state

    def get_or_start(
        self, episode_id: str, keyframe_stride: int, episode_step: int
    ) -> LingBotVARolloutState:
        """Fetch the active state, starting a new episode when required.

        A new episode begins when ``episode_step == 0``; that also covers a re-run of
        the same id (the eval runner reuses ids across repeats), where the stale state
        must go.

        Any other id mismatch means two episodes are being interleaved over one model
        instance. Since the transformer KV cache cannot be partitioned per episode,
        silently starting over would reset the other episode mid-rollout and still keep
        returning actions, i.e. corrupt the eval without failing it. That case raises.
        """
        state = self.get(episode_id)
        if state is not None and episode_step != 0:
            return state
        if episode_step != 0:
            active = self.active_episode_id
            if active is not None:
                raise RuntimeError(
                    f"episode {episode_id!r} arrived at step {episode_step} while episode "
                    f"{active!r} is still active. Only one episode may be active at a time "
                    "(the transformer KV cache belongs to the model instance), so the "
                    "interleaved episodes cannot both be served."
                )
            logger.warning(
                "starting episode %r at step %d with no prior state; treating this "
                "observation as the episode's first one",
                episode_id,
                episode_step,
            )
        return self.start(episode_id, keyframe_stride)

    def release(self) -> None:
        """Tear down the active episode's state, if any."""
        if self._state is None:
            return
        if self._on_release is not None:
            self._on_release(self._state)
        self._state = None
