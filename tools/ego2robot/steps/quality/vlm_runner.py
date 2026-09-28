#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent SGLang client for Ego2Robot L3 quality review.

SGLang exposes an OpenAI-compatible endpoint. Because custom video FPS
sampling is not consistently supported by SGLang releases, this client decodes
the video locally, uniformly caps the requested FPS samples across the full
timeline, and submits the chronological frames as a multi-image prompt.
Successful decisions are checkpointed atomically so an interrupted batch can
resume without reviewing completed episodes again.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import requests


DEFAULT_MAX_FRAMES = 32
VLM_MODEL_ENV = "EGO2ROBOT_VLM_MODEL"


def _resolve_model(model: str | None) -> str:
    """Resolve the server model id from CLI input or the environment."""
    resolved = (model or os.environ.get(VLM_MODEL_ENV) or "").strip()
    if not resolved:
        raise ValueError(
            f"VLM model is required; pass --model or set {VLM_MODEL_ENV}")
    return resolved


def _extract_json(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    candidate = fenced.group(1) if fenced else text
    if not fenced:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start:end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    # Another small-model variant appends a quote after the final comma:
    # `"reasoning": "...",\n"}`. Remove that quote before parsing.
    candidate = re.sub(r",\s*[\"']\s*}$", "}", candidate)
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        # Small Qwen models occasionally emit Python-style escaped
        # apostrophes (\\') inside an otherwise JSON response. JSON does not
        # define that escape, but replacing it is safe because an apostrophe
        # has no special meaning inside a JSON double-quoted string. Keep the
        # original exception for genuinely malformed/truncated responses.
        repaired = candidate.replace("\\'", "'")
        if repaired == candidate:
            raise first_error
        result = json.loads(repaired)
    if not isinstance(result, dict) or "is_consistent" not in result:
        raise ValueError("response JSON lacks is_consistent")
    confidence = float(result.get("confidence", 0.0))
    return {
        "is_consistent": bool(result["is_consistent"]),
        "confidence": min(max(confidence, 0.0), 1.0),
        "reasoning": str(result.get("reasoning", "")),
    }


def _uniform_indices(count: int, limit: int) -> list[int]:
    """Return at most ``limit`` indices spread across ``range(count)``."""
    if count <= 0:
        return []
    if limit <= 0:
        raise ValueError(f"max_frames must be positive, got {limit}")
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return [round(i * (count - 1) / (limit - 1)) for i in range(limit)]


def _sample_video(path: Path, fps: float, max_frames: int | None,
                  max_edge: int, jpeg_quality: int) -> list[str]:
    """Decode bounded, chronological JPEG data URLs across the full video."""
    import av
    from PIL import Image

    if fps <= 0:
        raise ValueError(f"sample fps must be positive, got {fps}")
    frame_limit = DEFAULT_MAX_FRAMES if max_frames is None else int(max_frames)
    if frame_limit <= 0:
        raise ValueError(f"max_frames must be positive, got {frame_limit}")

    def sampled_frames():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate else 30.0
            next_time = 0.0
            for index, frame in enumerate(container.decode(stream)):
                timestamp = (float(frame.time) if frame.time is not None
                             else index / source_fps)
                if timestamp + 1e-6 < next_time:
                    continue
                yield frame
                next_time = max(next_time + 1.0 / fps, timestamp + 1.0 / fps)

    # The first pass stores only a count. The second pass converts and encodes
    # the uniformly selected frames, keeping image memory bounded by the cap.
    candidate_count = sum(1 for _ in sampled_frames())
    selected = set(_uniform_indices(candidate_count, frame_limit))
    frames: list[str] = []
    for sample_index, frame in enumerate(sampled_frames()):
        if sample_index not in selected:
            continue
        image = frame.to_image().convert("RGB")
        if max(image.size) > max_edge:
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        frames.append(f"data:image/jpeg;base64,{encoded}")
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return frames


def _review_one(request: dict[str, Any], server_url: str, model: str,
                timeout: float, max_tokens: int, max_frames: int | None,
                max_edge: int, jpeg_quality: int) -> dict[str, Any]:
    video = Path(request["video"])
    if not video.is_file():
        raise FileNotFoundError(video)
    fps = float(request.get("sample_fps", 4.0))
    images = _sample_video(video, fps, max_frames, max_edge, jpeg_quality)
    prompt = str(request.get("prompt") or
                 f"Task Description: {request.get('task_description', 'manipulation')}")
    content = [
        {"type": "image_url", "image_url": {"url": image}}
        for image in images
    ]
    content.append({
        "type": "text",
        "text": (f"The preceding {len(images)} frames are sampled chronologically "
                 f"across the full video, targeting up to {fps:g} FPS.\n{prompt}"),
    })
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "top_k": 20,
    }
    response = requests.post(
        server_url.rstrip("/") + "/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer EMPTY"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    result = _extract_json(body["choices"][0]["message"]["content"])
    result.update({"model_name": model, "sample_fps": fps,
                   "sampled_frames": len(images), "backend": "sglang"})
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    temporary.replace(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _health_check(server_url: str, timeout: float) -> None:
    response = requests.get(
        server_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer EMPTY"},
        timeout=min(timeout, 30.0),
    )
    response.raise_for_status()


def run(requests_path: str, output_path: str,
        server_url: str = "http://127.0.0.1:8000/v1",
        model: str | None = None,
        timeout: float = 600.0, max_tokens: int = 512,
        max_episodes: int | None = None, concurrency: int = 4,
        retries: int = 2, max_frames: int | None = DEFAULT_MAX_FRAMES,
        max_edge: int = 448, jpeg_quality: int = 85,
        overwrite: bool = False) -> None:
    model = _resolve_model(model)
    _health_check(server_url, timeout)
    requests_to_run = []
    with Path(requests_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                requests_to_run.append(json.loads(line))
    if max_episodes is not None:
        requests_to_run = requests_to_run[:max_episodes]

    output = Path(output_path)
    error_path = output.with_name(output.stem + ".errors.json")
    results = {} if overwrite else _load_json_object(output)
    errors = {} if overwrite else _load_json_object(error_path)
    pending = [r for r in requests_to_run if str(r["episode"]) not in results]
    print(f"SGLang ready: {len(requests_to_run)} requested, "
          f"{len(results)} cached, {len(pending)} pending", flush=True)
    lock = threading.Lock()

    def work(request: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        episode = str(request["episode"])
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return episode, _review_one(
                    request, server_url, model, timeout, max_tokens,
                    max_frames, max_edge, jpeg_quality)
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 10))
        assert last_error is not None
        raise RuntimeError(
            f"{episode} failed after {retries + 1} attempt(s): "
            f"{type(last_error).__name__}: {last_error}") from last_error

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        future_to_episode = {
            executor.submit(work, request): str(request["episode"])
            for request in pending
        }
        completed = 0
        for future in as_completed(future_to_episode):
            episode = future_to_episode[future]
            completed += 1
            try:
                _, result = future.result()
                with lock:
                    results[episode] = result
                    errors.pop(episode, None)
                    _atomic_json(output, results)
                    _atomic_json(error_path, errors)
                print(f"[{completed}/{len(pending)}] {episode}: "
                      f"consistent={result['is_consistent']} "
                      f"confidence={result['confidence']:.2f}", flush=True)
            except Exception as exc:
                error = {"error": f"{type(exc).__name__}: {exc}"}
                with lock:
                    errors[episode] = error
                    _atomic_json(error_path, errors)
                print(f"[{completed}/{len(pending)}] {episode}: ERROR {error['error']}",
                      flush=True)
    print(f"wrote {len(results)} decisions to {output}; "
          f"{len(errors)} unresolved errors in {error_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run concurrent Qwen3.5 L3 review through SGLang")
    parser.add_argument("--requests", required=True,
                        help="quality_curation/vlm_requests.jsonl")
    parser.add_argument("--output", required=True,
                        help="destination vlm_results.json")
    parser.add_argument("--server_url", default="http://127.0.0.1:8000/v1")
    env_model = os.environ.get(VLM_MODEL_ENV) or None
    parser.add_argument(
        "--model", default=env_model, required=env_model is None,
        help=f"model id exposed by SGLang (or set {VLM_MODEL_ENV})")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help=("maximum uniformly sampled frames per video "
                              f"(default: {DEFAULT_MAX_FRAMES})"))
    parser.add_argument("--max_edge", type=int, default=448)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--overwrite", action="store_true",
                        help="discard cached decisions and review all requests")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args.requests, args.output, args.server_url, args.model, args.timeout,
        args.max_tokens, args.max_episodes, args.concurrency, args.retries,
        args.max_frames, args.max_edge, args.jpeg_quality, args.overwrite)
