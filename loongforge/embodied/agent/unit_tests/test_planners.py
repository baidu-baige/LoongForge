# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Both direct-drive modes end to end on a point robot with a scripted LLM."""

from __future__ import annotations

import json

import numpy as np
import pytest

from loongforge.embodied.agent.core.recorder import Recorder
from loongforge.embodied.agent.core.types import EpisodeSpec
from loongforge.embodied.agent.planners.json_block import JsonBlockPlanner
from loongforge.embodied.agent.planners.llm import OMITTED_IMAGE, image_part, text_part, to_openai
from loongforge.embodied.agent.planners.tool_call import ToolCallPlanner
from loongforge.embodied.agent.toolsets.pose_delta import InvalidAction, PoseDeltaSession, parse_action
from loongforge.embodied.agent.unit_tests.fakes import FakeEmbodiment, ScriptedLLM

SPEC = EpisodeSpec("libero_spatial", 0, 0)


def block(dx=0.0, dy=0.0, dz=0.0, width=0.0) -> str:
    action = {"arm": {"x": dx, "y": dy, "z": dz, "roll": 0, "pitch": 0, "yaw": 0}, "gripper": {"width": width}}
    return "thinking...\n```json\n" + json.dumps(action) + "\n```"


def test_parse_action_takes_the_last_block_and_checks_the_schema() -> None:
    assert parse_action("```json\n{}\n```\n" + block(dx=1))["arm"]["x"] == 1
    for bad in [
        "no block",
        '```json\n{"arm": {}}\n```',
        block().replace('"yaw": 0', '"yaw": "0"'),
        "```json\n{oops}\n```",
    ]:
        with pytest.raises(InvalidAction):
            parse_action(bad)


def test_pose_delta_uses_base_frame_centimetres_and_pear_image_convention() -> None:
    robot = FakeEmbodiment()
    obs = robot.reset(SPEC)
    session = PoseDeltaSession(robot)
    assert session.state(obs)["arm"]["x"] == 50.0  # base at x = -0.5 m
    assert "move to the right target" in session.system_prompt(obs)  # left/right swapped with the mirrored frames
    assert session.frames(obs)["agentview"][0, -1, 0] == 255  # white left half shown on the right
    session.act(obs, parse_action(block(dx=10, dz=-30, width=-1)))
    move = robot.moves[-1]
    assert np.allclose(move["pos"], [0.1, 0.0, 0.2]) and move["opening"] == -1.0


def test_json_block_planner_reaches_success(tmp_path) -> None:
    llm = ScriptedLLM([{"content": block(dx=5)}, {"content": block(dx=5, dz=-30)}])
    result = JsonBlockPlanner(llm).run(FakeEmbodiment(), SPEC, Recorder(tmp_path))
    assert (result.success, result.end_reason, result.llm_turns, result.steps) == (True, "success", 2, 20)
    turn = llm.sent[0][1]["content"]
    assert json.loads(turn[0]["text"].split("\n\n")[0])["arm"]["z"] == 50.0
    assert [p["text"] for p in turn if p["type"] == "text"][1:] == ["[agentview]", "[wrist]"]
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2


def test_json_block_planner_stops_on_invalid_answer(tmp_path) -> None:
    result = JsonBlockPlanner(ScriptedLLM([{"content": "I am not sure."}])).run(
        FakeEmbodiment(), SPEC, Recorder(tmp_path)
    )
    assert (result.success, result.end_reason) == (False, "parse")


def test_json_block_planner_stops_at_step_budget(tmp_path) -> None:
    llm = ScriptedLLM([{"content": block(dy=1)}] * 5)
    result = JsonBlockPlanner(llm).run(FakeEmbodiment(step_limit=30), SPEC, Recorder(tmp_path))
    assert (result.end_reason, result.steps, result.llm_turns) == ("budget", 30, 3)


def call(name, **arguments):
    return {"id": f"call_{name}", "name": name, "arguments": json.dumps(arguments)}


def test_tool_call_planner_moves_and_succeeds(tmp_path) -> None:
    llm = ScriptedLLM(
        [
            {"tool_calls": [call("pixel_to_world", camera="agentview", pixels=[[32, 42]])]},
            {"content": "go", "tool_calls": [call("move_to", xyz=[0.1, 0.0, 0.2], gripper="close")]},
        ]
    )
    robot = FakeEmbodiment()
    result = ToolCallPlanner(llm).run(robot, SPEC, Recorder(tmp_path))
    assert (result.success, result.end_reason, result.llm_turns) == (True, "success", 2)
    tool_reply = json.loads(llm.sent[1][-1]["content"])
    assert np.allclose(tool_reply["points"][0]["xyz"], [0.1, 0.0, 0.0])
    assert robot.moves[-1]["opening"] == -1.0
    assert any(p["type"] == "image" for p in llm.sent[0][1]["content"])


def test_tool_call_planner_finish_errors_and_frames(tmp_path) -> None:
    llm = ScriptedLLM(
        [
            {"tool_calls": [call("move_to", xyz=[0.3, 0.0])]},  # bad arguments: no motion, no frames
            {"tool_calls": [call("move_to", xyz=[0.3, 0.0, 0.3])]},
            {"content": "thinking aloud"},  # no tool call: reminded
            {"tool_calls": [call("finish", summary="gave up")]},
        ]
    )
    result = ToolCallPlanner(llm).run(FakeEmbodiment(), SPEC, Recorder(tmp_path))
    assert (result.success, result.end_reason, result.extra["summary"]) == (False, "finish", "gave up")
    after_bad, after_move, after_text = llm.sent[1][-1], llm.sent[2][-1], llm.sent[3][-1]
    assert after_bad["role"] == "tool" and "bad arguments" in after_bad["content"]
    assert after_move["role"] == "user" and any(p["type"] == "image" for p in after_move["content"])
    assert after_text["role"] == "user" and "Call a tool" in after_text["content"]


def test_image_window_keeps_only_recent_images() -> None:
    rgb = np.zeros((2, 2, 3), np.uint8)
    messages = [{"role": "user", "content": [text_part(str(i)), image_part(rgb)]} for i in range(3)]
    sent = to_openai(messages, max_image_messages=1)
    kinds = [[p["type"] for p in m["content"]] for m in sent]
    assert kinds == [["text", "text"], ["text", "text"], ["text", "image_url"]]
    assert sent[0]["content"][1]["text"] == OMITTED_IMAGE
    assert sum(p["type"] == "image_url" for m in to_openai(messages) for p in m["content"]) == 3
