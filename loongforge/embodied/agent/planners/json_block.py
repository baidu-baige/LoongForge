# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""PEAR-style loop: one fenced JSON action per turn, no tool schema."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from loongforge.embodied.agent.core.embodiment import Embodiment
from loongforge.embodied.agent.core.recorder import Recorder
from loongforge.embodied.agent.core.types import EpisodeResult, EpisodeSpec
from loongforge.embodied.agent.planners.llm import LLMClient, LLMError, for_transcript
from loongforge.embodied.agent.toolsets.pose_delta import InvalidAction, PoseDeltaSession, parse_action


class JsonBlockPlanner:
    """Run an episode with the incremental pose protocol.

    Like PEAR, the episode ends on an answer without a valid action; the full
    conversation is kept unless the client's image window is set.
    """

    mode = "pose_delta"

    def __init__(self, llm: LLMClient, max_turns: int = 100) -> None:
        self.llm = llm
        self.max_turns = max_turns

    def run(self, embodiment: Embodiment, spec: EpisodeSpec, recorder: Recorder) -> EpisodeResult:
        """One episode."""
        started = time.time()
        obs = embodiment.reset(spec)
        session = PoseDeltaSession(embodiment)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": session.system_prompt(obs)}]
        recorder.transcript(messages[0])
        end_reason, error, turns = "max_turns", None, 0

        while turns < self.max_turns:
            frames = {cam: recorder.frame(f"{cam}", rgb) for cam, rgb in session.frames(obs).items()}
            messages.append({"role": "user", "content": session.turn(obs, frames)})
            recorder.transcript(for_transcript(messages[-1]))
            try:
                reply = self.llm.chat(messages)
            except LLMError as exc:
                end_reason, error = "llm_error", str(exc)
                break
            turns += 1
            messages.append({"role": "assistant", "content": reply["content"]})
            recorder.transcript({"role": "assistant", "content": reply["content"], "reasoning": reply["reasoning"]})
            try:
                action = parse_action(reply["content"])
            except InvalidAction as exc:
                end_reason, error = "parse", str(exc)
                recorder.trace({"turn": turns, "error": error})
                break
            before = session.state(obs)
            result = session.act(obs, action)
            obs = embodiment.observe()
            recorder.trace(
                {
                    "turn": turns,
                    "state": before,
                    "action": action,
                    "outcome": result.outcome,
                    "executed": result.executed,
                    "steps": obs.steps,
                }
            )
            if obs.success:
                end_reason = "success"
                break
            if obs.remaining_steps <= 0:
                end_reason = "budget"
                break

        return EpisodeResult(
            spec=spec,
            success=obs.success,
            end_reason=end_reason,
            steps=obs.steps,
            llm_turns=turns,
            prompt_tokens=self.llm.usage.prompt_tokens,
            completion_tokens=self.llm.usage.completion_tokens,
            seconds=time.time() - started,
            error=error,
            extra={"mode": self.mode, "model": self.llm.config.model},
        )
