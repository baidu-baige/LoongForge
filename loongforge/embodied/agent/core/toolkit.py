# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Register tools and execute model-issued calls."""

from __future__ import annotations

import json
import logging
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional

from pydantic import ValidationError

from loongforge.embodied.agent.core.recorder import Recorder
from loongforge.embodied.agent.core.tool import Tool
from loongforge.embodied.agent.core.types import ToolResult

logger = logging.getLogger(__name__)


class Toolkit:
    """The tools of one episode.

    ``execute`` never raises for a bad call: unknown names, invalid arguments
    and handler exceptions come back as ``ToolResult.data["error"]`` so the
    model can read and correct them. Every call is appended to the recorder's
    trace.
    """

    def __init__(self, tools: Iterable[Tool], recorder: Optional[Recorder] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for item in tools:
            if item.name in self._tools:
                raise ValueError(f"duplicate tool name: {item.name}")
            self._tools[item.name] = item
        self._recorder = recorder
        self.calls = 0

    def list_tools(self) -> List[Tool]:
        """The registered tools, in registration order."""
        return list(self._tools.values())

    def openai_tools(self) -> List[Dict[str, Any]]:
        """All tool schemas in OpenAI chat-completions form."""
        return [item.openai_schema() for item in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Validate and run one call."""
        self.calls += 1
        started = time.perf_counter()
        item = self._tools.get(name)
        if item is None:
            result = ToolResult({"error": f"unknown tool: {name}", "tools": sorted(self._tools)})
        else:
            result = self._run(item, arguments)
        if self._recorder is not None:
            self._recorder.trace(
                {
                    "tool": name,
                    "arguments": arguments,
                    "result": result.data,
                    "images": sorted(result.images),
                    "seconds": round(time.perf_counter() - started, 3),
                }
            )
        return result

    @staticmethod
    def _run(item: Tool, arguments: Dict[str, Any]) -> ToolResult:
        try:
            parsed = item.args_model.model_validate(arguments)
        except ValidationError as exc:
            errors = exc.errors(include_url=False, include_context=False, include_input=False)
            return ToolResult(
                {"error": f"bad arguments for {item.name}", "details": json.loads(json.dumps(errors, default=str))}
            )
        try:
            return item.handler(**{key: getattr(parsed, key) for key in type(parsed).model_fields})
        except Exception as exc:  # the model reads the failure and decides what to do next
            logger.exception("tool %s failed", item.name)
            return ToolResult({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3)})
