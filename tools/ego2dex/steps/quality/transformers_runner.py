#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Direct Transformers L3 runner for a local Qwen3.5 checkpoint.

This avoids an HTTP server and is useful when the runtime has CUDA available
but the container/network namespace does not expose localhost services.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


VLM_MODEL_ENV = "EGO2ROBOT_VLM_MODEL"


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
    text = str(content).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    candidate = match.group(1) if match else text
    if not match:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    # Small checkpoints occasionally emit a trailing comma in JSON objects.
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict) or "is_consistent" not in parsed:
        raise ValueError("response JSON lacks is_consistent")
    return {
        "is_consistent": bool(parsed["is_consistent"]),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


def review_one(
    model: Any, processor: Any, request: dict[str, Any], device: Any, max_tokens: int, max_frames: int | None
) -> dict[str, Any]:
    import torch

    video = Path(request["video"])
    if not video.is_file():
        return {"is_consistent": False, "confidence": 0.0, "reasoning": f"video not found: {video}"}
    task = str(request.get("task_description", "manipulation"))
    prompt = str(request.get("prompt") or f"Task Description: {task}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = {
        "text": [text],
        "videos": [str(video)],
        "return_tensors": "pt",
        "do_sample_frames": True,
        "fps": 4,
    }
    if max_frames is not None:
        processor_kwargs["max_frames"] = max_frames
    inputs = processor(**processor_kwargs).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    generated = output[:, inputs["input_ids"].shape[1] :]
    content = processor.batch_decode(generated, skip_special_tokens=True)[0]
    try:
        return _extract_json(content)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "is_consistent": False,
            "confidence": 0.0,
            "reasoning": f"invalid VLM JSON: {exc}; raw={content[:1000]}",
        }


def run(
    model_dir: str | None,
    requests_path: str,
    output_path: str,
    max_tokens: int = 256,
    max_frames: int | None = None,
    max_episodes: int | None = None,
    device: str | None = None,
) -> None:
    model_dir = (model_dir or os.environ.get(VLM_MODEL_ENV) or "").strip()
    if not model_dir:
        raise ValueError(f"model directory is required; pass --model_dir or set {VLM_MODEL_ENV}")
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model_path = Path(model_dir)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    print(f"loading {model_path} on {torch_device} with {dtype}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map=None, local_files_only=True
    )
    model.to(torch_device).eval()

    results: dict[str, dict[str, Any]] = {}
    with Path(requests_path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_episodes is not None and index >= max_episodes:
                break
            request = json.loads(line)
            episode = str(request["episode"])
            print(f"[{index + 1}] reviewing {episode}", flush=True)
            try:
                result = review_one(model, processor, request, torch_device, max_tokens, max_frames)
            except Exception as exc:
                result = {
                    "is_consistent": False,
                    "confidence": 0.0,
                    "reasoning": f"VLM request failed: {type(exc).__name__}: {exc}",
                }
            result["model_name"] = str(model_path)
            results[episode] = result
            print(json.dumps(result, ensure_ascii=False), flush=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(results)} VLM results to {out}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 L3 review directly with Transformers")
    env_model = os.environ.get(VLM_MODEL_ENV) or None
    parser.add_argument(
        "--model_dir",
        default=env_model,
        required=env_model is None,
        help=f"local Qwen3.5 checkpoint (or set {VLM_MODEL_ENV})",
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None, help="default: cuda:0 when CUDA is available")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_frames", type=int, default=None, help="debug cap; omit to use the model video sampler")
    parser.add_argument("--max_episodes", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args.model_dir, args.requests, args.output, args.max_tokens, args.max_frames, args.max_episodes, args.device)
