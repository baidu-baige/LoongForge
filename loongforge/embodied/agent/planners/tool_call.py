# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tool-calling loop: the model calls tools; motions are followed by fresh camera frames."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from loongforge.embodied.agent.core.embodiment import Embodiment
from loongforge.embodied.agent.core.recorder import Recorder
from loongforge.embodied.agent.core.toolkit import Toolkit
from loongforge.embodied.agent.core.types import EpisodeResult, EpisodeSpec, Obs
from loongforge.embodied.agent.planners.llm import LLMClient, LLMError, for_transcript, image_part, text_part
from loongforge.embodied.agent.toolsets.pose import PoseToolset, pose_summary

NO_TOOL_REMINDER = "Call a tool to continue, or call finish to end the episode."


class ToolCallPlanner:
    """Run an episode with the absolute pose toolset.

    Every motion consumes a turn and is followed by a user message with the
    new frames; the client's image window bounds how many frame sets stay in
    the request. The episode ends on ``finish``, task success, the step
    budget or ``max_turns`` LLM calls.
    """

    mode = "pose"

    def __init__(self, llm: LLMClient, max_turns: int = 100) -> None:
        self.llm = llm
        self.max_turns = max_turns

    def run(self, embodiment: Embodiment, spec: EpisodeSpec, recorder: Recorder) -> EpisodeResult:
        """One episode."""
        started = time.time()
        embodiment.reset(spec)
        toolset = PoseToolset(embodiment, embodiment.observe(depth=True))
        toolkit = Toolkit(toolset.tools(), recorder)
        advancing = {item.name for item in toolkit.list_tools() if item.advances}
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": toolset.system_prompt()},
            self._frames_message(toolset.obs, recorder, "Initial observation."),
        ]
        for message in messages:
            recorder.transcript(for_transcript(message))
        end_reason, error, turns = "max_turns", None, 0

        while turns < self.max_turns:
            try:
                reply = self.llm.chat(messages, toolkit.openai_tools())
            except LLMError as exc:
                end_reason, error = "llm_error", str(exc)
                break
            turns += 1
            assistant: Dict[str, Any] = {"role": "assistant", "content": reply["content"] or None}
            if reply["tool_calls"]:
                assistant["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in reply["tool_calls"]
                ]
            messages.append(assistant)
            recorder.transcript({**assistant, "reasoning": reply["reasoning"]})

            if not reply["tool_calls"]:
                messages.append({"role": "user", "content": NO_TOOL_REMINDER})
                recorder.transcript(messages[-1])
                continue
            moved = False
            for call in reply["tool_calls"]:
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError as exc:
                    result_data = {"error": f"arguments are not JSON: {exc}"}
                else:
                    result_data = toolkit.execute(call["name"], arguments if isinstance(arguments, dict) else {}).data
                    moved = moved or (call["name"] in advancing and "error" not in result_data)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result_data)})
                recorder.transcript(messages[-1])
                if toolset.finished is not None or toolset.obs.success or toolset.obs.remaining_steps <= 0:
                    break
            if toolset.obs.success:
                end_reason = "success"
                break
            if toolset.finished is not None:
                end_reason = "finish"
                break
            if toolset.obs.remaining_steps <= 0:
                end_reason = "budget"
                break
            if moved:
                messages.append(self._frames_message(toolset.obs, recorder, "Frames after the motion."))
                recorder.transcript(for_transcript(messages[-1]))

        obs = toolset.obs
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
            extra={
                "mode": self.mode,
                "model": self.llm.config.model,
                "tool_calls": toolkit.calls,
                "summary": toolset.finished,
            },
        )

    @staticmethod
    def _frames_message(obs: Obs, recorder: Recorder, header: str) -> Dict[str, Any]:
        parts = [text_part(f"{header}\n{json.dumps(pose_summary(obs))}")]
        for camera, frame in obs.cameras.items():
            height, width = frame.rgb.shape[:2]
            parts.append(text_part(f"[{camera}] {width}x{height}"))
            parts.append(image_part(frame.rgb, recorder.frame(camera, frame.rgb)))
        return {"role": "user", "content": parts}
