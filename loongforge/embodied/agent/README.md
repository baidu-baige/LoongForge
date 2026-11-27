# Direct-drive embodied agent

An LLM drives the robot through closed-loop motion primitives, with no VLA in
the loop. The agent core does not depend on the eval harness; embodiments reuse
each benchmark's scene protocol (LIBERO: official init states, `num_steps_wait`
settling, `SUITE_MAX_STEPS`), so results are comparable with VLA evals.

## Modes

| `--mode` | Model output per turn | Observation | Planner |
|-|-|-|-|
| `pose_delta` | one fenced JSON block changing the base-frame pose (cm, rad) and gripper (PEAR protocol) | 2 RGB frames + pose numbers; no depth, no tools, no `finish` | `planners/json_block.py` |
| `pose` | tool calls: `move_to` (world xyz, rpy), `set_gripper`, `pixel_to_world`, `finish` | RGB frames after each motion; depth back-projection on request | `planners/tool_call.py` |

Both modes run the same controller (`embodiments/libero/controller.py`, derived
from PEAR's `MotionController`): PD on the tool pose through robosuite's OSC
until the arm rests at the target or stops short. Only the interface differs.

## Layout

```
core/         types, @tool + Toolkit, Embodiment protocol, geometry, perception, Recorder
embodiments/  libero/ (env.py, controller.py, environment.md)
toolsets/     pose_delta.py (PEAR protocol), pose.py (absolute pose tools)
planners/     llm.py (chat-completions client), json_block.py, tool_call.py
run.py        CLI over tasks x init states
unit_tests/   fake robot + scripted LLM tests; LIBERO tests skip without LIBERO
```

## Run

The LIBERO simulator must be importable (LIBERO + robosuite 1.4 or 1.5); the
agent itself needs only numpy, pydantic, pillow and the `openai` SDK.

```bash
# PEAR-style, e.g. Qwen3.8-27B-PEAR served by vLLM
python -m loongforge.embodied.agent.run --mode pose_delta --suite libero_spatial \
  --tasks 0-9 --episodes 0-9 --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8010/v1 \
  --out runs/pose_delta

# absolute pose tools on any OpenAI-compatible endpoint
OPENAI_BASE_URL=... OPENAI_API_KEY=... python -m loongforge.embodied.agent.run --mode pose \
  --suite libero_spatial --tasks 0 --episodes 0-4 --model <model> --max-image-messages 3 --out runs/pose
```

Each episode writes `<out>/<suite>/task<k>/init<i>/` with `result.json`,
`trace.jsonl` (tool calls or actions), `transcript.jsonl` (LLM messages, images
as frame names) and `frames/`. `<out>/results.jsonl` has one line per episode;
finished episodes are skipped on rerun.

```bash
python -m pytest loongforge/embodied/agent/unit_tests
```

## Not yet included

Code direct drive (`run_program` in a sandbox), offline programs and the RSI
loop, the Sys2 + Sys1 mode (`vla_act` via the eval `PolicyClient`), parallel
episode launch, and embodiments beyond LIBERO.
