# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Declare tools from typed functions.

A tool's JSON schema comes from the handler's type annotations and the ``Args:``
section of its Google-style docstring; its description is the docstring's first
paragraph. Arguments are validated strictly before the handler runs.
"""

from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

from loongforge.embodied.agent.core.types import ToolResult

_ARG_LINE = re.compile(r"^ {4}(\w+)(?:\s*\([^)]*\))?:\s*(.*)$")


@dataclass(frozen=True)
class Tool:
    """A callable the planner can expose to the model."""

    name: str
    description: str
    handler: Callable[..., ToolResult]
    args_model: Type[BaseModel]
    advances: bool  # whether it moves the robot (and so consumes episode steps)

    def openai_schema(self) -> Dict[str, Any]:
        """Function schema for the OpenAI chat-completions ``tools`` field."""
        parameters = self.args_model.model_json_schema()
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }


def _docstring_parts(doc: str) -> tuple[str, Dict[str, str]]:
    """Split a docstring into its summary paragraph and per-argument descriptions."""
    doc = inspect.cleandoc(doc or "")
    summary = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    args: Dict[str, str] = {}
    lines = doc.splitlines()
    if "Args:" not in lines:
        return summary, args
    current: Optional[str] = None
    for line in lines[lines.index("Args:") + 1 :]:
        if line and not line.startswith(" "):
            break  # the next section (Returns:, Raises:)
        match = _ARG_LINE.match(line)
        if match:
            current = match.group(1)
            args[current] = match.group(2).strip()
        elif current and line.strip():
            args[current] += " " + line.strip()
    return summary, args


def tool(func: Optional[Callable] = None, *, name: Optional[str] = None, advances: bool = False):
    """Mark a method or function as a tool; see the module docstring."""

    def wrap(fn: Callable) -> Callable:
        fn.__agent_tool__ = {"name": name or fn.__name__, "advances": advances}
        return fn

    return wrap(func) if func is not None else wrap


def build_tool(fn: Callable) -> Tool:
    """Create the :class:`Tool` for a function or bound method marked with :func:`tool`."""
    meta = getattr(fn, "__agent_tool__")
    summary, arg_docs = _docstring_parts(fn.__doc__ or "")
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: Dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.name == "self":
            continue
        annotation = hints.get(param.name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, Field(default, description=arg_docs.get(param.name)))
    args_model = create_model(f"{meta['name']}_args", __config__=ConfigDict(extra="forbid", strict=True), **fields)
    return Tool(meta["name"], summary, fn, args_model, meta["advances"])


def collect_tools(owner: Any) -> list[Tool]:
    """Build every tool declared on ``owner``'s class, bound to ``owner``, in definition order."""
    tools = []
    for attr in type(owner).__dict__:
        member = getattr(owner, attr, None)
        if callable(member) and hasattr(member, "__agent_tool__"):
            tools.append(build_tool(member))
    return tools
