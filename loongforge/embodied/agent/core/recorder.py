# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode artifacts: tool trace, LLM transcript, key frames and result."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)


class Recorder:
    """Write one episode's files under ``out_dir``.

    Layout: ``result.json``, ``trace.jsonl`` (one line per tool call or env
    action), ``transcript.jsonl`` (LLM messages, images replaced by frame
    file names) and ``frames/<index>_<label>.png``.
    """

    def __init__(self, out_dir: os.PathLike | str) -> None:
        self.dir = Path(out_dir)
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        self._frame_index = 0

    def trace(self, record: Dict[str, Any]) -> None:
        """Append one tool call or action record."""
        self._append("trace.jsonl", record)

    def transcript(self, record: Dict[str, Any]) -> None:
        """Append one LLM message or response record."""
        self._append("transcript.jsonl", record)

    def frame(self, label: str, rgb: np.ndarray) -> str:
        """Save an RGB frame and return its path relative to the episode directory."""
        from PIL import Image

        name = f"frames/{self._frame_index:04d}_{label}.png"
        self._frame_index += 1
        Image.fromarray(np.ascontiguousarray(rgb)).save(self.dir / name)
        return name

    def result(self, data: Dict[str, Any]) -> None:
        """Write result.json atomically."""
        tmp = self.dir / "result.json.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=_jsonable))
        tmp.replace(self.dir / "result.json")

    def _append(self, name: str, record: Dict[str, Any]) -> None:
        with open(self.dir / name, "a") as handle:
            handle.write(json.dumps(record, default=_jsonable, ensure_ascii=False) + "\n")
