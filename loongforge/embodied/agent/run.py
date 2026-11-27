# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run direct-drive agent episodes.

Example (PEAR-style incremental pose on a local vLLM)::

    python -m loongforge.embodied.agent.run --mode pose_delta \\
        --suite libero_spatial --tasks 0 --episodes 0-4 \\
        --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8010/v1 --out runs/pear

Each episode writes ``<out>/<suite>/task<k>/init<i>/`` (result.json, trace.jsonl,
transcript.jsonl, frames/); ``<out>/results.jsonl`` collects one line per
episode, and episodes that already have result.json are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from loongforge.embodied.agent.core.recorder import Recorder
from loongforge.embodied.agent.core.types import EpisodeResult, EpisodeSpec
from loongforge.embodied.agent.planners.json_block import JsonBlockPlanner
from loongforge.embodied.agent.planners.llm import LLMClient, LLMConfig
from loongforge.embodied.agent.planners.tool_call import ToolCallPlanner

PLANNERS = {"pose_delta": JsonBlockPlanner, "pose": ToolCallPlanner}
logger = logging.getLogger("loongforge.embodied.agent")


def parse_range(text: str) -> List[int]:
    """'0-2,5' -> [0, 1, 2, 5]."""
    values: List[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            values.extend(range(int(low), int(high) + 1))
        elif part:
            values.append(int(part))
    return values


def build_embodiment(args: argparse.Namespace):
    """The embodiment named by ``--embodiment``."""
    if args.embodiment == "libero":
        from loongforge.embodied.agent.embodiments.libero.env import LiberoEmbodiment

        return LiberoEmbodiment(resolution=args.resolution, seed=args.env_seed)
    raise ValueError(f"unknown embodiment {args.embodiment!r}")


def build_parser() -> argparse.ArgumentParser:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=sorted(PLANNERS), required=True)
    parser.add_argument("--embodiment", default="libero")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--tasks", default="0", help="task ids, e.g. 0-9")
    parser.add_argument("--episodes", default="0", help="init-state indices, e.g. 0-9")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; default $OPENAI_BASE_URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    parser.add_argument("--max-turns", type=int, default=100, help="LLM calls per episode")
    parser.add_argument(
        "--max-image-messages", type=int, default=0, help="image-bearing messages kept in requests; 0 keeps all"
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None, help="sent as extra_body.reasoning_effort")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--env-seed", type=int, default=7)
    return parser


def main(argv=None) -> None:
    """Run every (task, episode) pair serially."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extra_body = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    embodiment = build_embodiment(args)
    try:
        for task in parse_range(args.tasks):
            for init_idx in parse_range(args.episodes):
                spec = EpisodeSpec(args.suite, task, init_idx)
                episode_dir = out / spec.tag
                if (episode_dir / "result.json").exists():
                    logger.info("skip %s (done)", spec.tag)
                    continue
                recorder = Recorder(episode_dir)
                llm = LLMClient(
                    LLMConfig(
                        model=args.model,
                        base_url=args.base_url,
                        api_key=os.environ.get(args.api_key_env),
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_image_messages=args.max_image_messages,
                        extra_body=extra_body,
                    )
                )
                planner = PLANNERS[args.mode](llm, max_turns=args.max_turns)
                try:
                    result = planner.run(embodiment, spec, recorder)
                except Exception as exc:  # keep the batch going; the episode records the failure
                    logger.exception("episode %s failed", spec.tag)
                    result = EpisodeResult(
                        spec, False, "error", 0, llm.usage.calls, error=f"{type(exc).__name__}: {exc}"
                    )
                record = result.to_dict()
                recorder.result(record)
                with open(out / "results.jsonl", "a") as handle:
                    handle.write(json.dumps(record) + "\n")
                logger.info(
                    "%s success=%s end=%s steps=%d turns=%d",
                    spec.tag,
                    result.success,
                    result.end_reason,
                    result.steps,
                    result.llm_turns,
                )
    finally:
        embodiment.close()


if __name__ == "__main__":
    main()
