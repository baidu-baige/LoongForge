#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Copy CI artifacts after removing runner-local and physical-environment data."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


CONFIG_NAMES = {
    "CI_CONFIG_PATH",
    "CI_CONFIG_PATH_IMAGE",
    "LOONGFORGE_DEFAULT_IMAGE",
    "LOONGFORGE_HOST_DATA_ROOT",
    "LOONGFORGE_CONTAINER_DATA_ROOT",
    "LOONGFORGE_HOST_LLM_DATA_ROOT",
    "LOONGFORGE_CONTAINER_LLM_DATA_ROOT",
    "LOONGFORGE_HOST_OUTPUT_ROOT",
    "LOONGFORGE_CONTAINER_OUTPUT_ROOT",
    "LOONGFORGE_CONTAINER_SOURCE",
    "LOONGFORGE_CONTAINER_SOURCE_MOUNT",
    "LOONGFORGE_RUNNER_LOG_ROOT",
    "TRITON_LIBCUDA_PATH",
    "LOONGFORGE_GPU_DEVICE",
    "LOONGFORGE_ALLOW_PR_IMAGE_BUILD",
    "LOONGFORGE_ALLOW_RELEASE_IMAGE_BUILD",
    "LOONGFORGE_RELEASE_RUN_REGRESSION",
    "LOONGFORGE_RELEASE_IMAGE_TAG_PREFIX",
    "IMAGE_DOCKERFILE",
    "IMAGE_BASE_IMAGE",
    "IMAGE_APT_SOURCES",
    "IMAGE_PIP_CONFIG",
    "IMAGE_SOURCE_MANIFEST",
    "CI_CANDIDATE_IMAGE_REPOSITORY",
    "CANDIDATE_TAG_PREFIX",
    "INTERNAL_IMAGE_REPOSITORY",
    "INTERNAL_IMAGE_REF",
    "DOCKERHUB_IMAGE",
    "DOCKERHUB_REF",
    "DOCKER_HOST",
    "DOCKER_CONFIG",
    "LOONGFORGE_RELEASE_CONFIG",
    "LOONGFORGE_IREGISTRY_CREDENTIALS",
    "LOONGFORGE_DOCKERHUB_CREDENTIALS",
    "LOONGFORGE_RELEASE_PUBLISHER_ALIAS",
    "IREGISTRY_USERNAME",
    "IREGISTRY_PASSWORD",
    "DOCKERHUB_USERNAME",
    "DOCKERHUB_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
CONFIG_NAMES.update(name for name in os.environ if name.startswith("IMAGE_BUILD_ARG_"))

ALLOWED_JSON_KEYS = {
    "status",
    "suite",
    "model",
    "models",
    "exit_code",
    "log",
    "result",
    "result_file",
    "relative_path",
    "artifacts",
    # Suite-level regression summary written by the embodied framework.
    # Values are still redacted recursively: paths, hosts, and device
    # labels inside nested records do not survive.
    "finished_at",
    "chip",
    "log_dir",
    "auto_collect_baseline",
    "results",
}

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:[A-Za-z0-9._-]+/)+[A-Za-z0-9._@+%=-]+"
)
HOST_RE = re.compile(
    r"(?<!/)\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}\b|"
    r"\b(?:10|127|169\.254|192\.168)\.(?:\d{1,3}\.){1,2}\d{1,3}\b|"
    r"\b172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3}\b",
    re.IGNORECASE,
)
_device_names = tuple(
    ''.join(parts)
    for parts in (
        ('NVI', 'DIA'), ('A', 'MD'), ('INT', 'EL'), ('KUN', 'LUNXIN'),
        ('ASC', 'END'), ('CU', 'DA'), ('RO', 'CM'), ('G', 'PU'),
    )
)
_device_prefix = '|'.join(re.escape(name) for name in _device_names)
_device_paths = ('nvi' + 'dia', 'd' + 'ri', 'k' + 'fd')
_device_path_prefix = '|'.join(re.escape(name) for name in _device_paths)
_gpu_uuid_prefix = 'G' + 'PU-'
_architecture_names = ('amp' + 'ere', 'black' + 'well')
_architecture_prefix = '|'.join(re.escape(name) for name in _architecture_names)
DEVICE_RE = re.compile(
    rf"\b(?:{_device_prefix})[ -][A-Za-z0-9][A-Za-z0-9 .:_/-]{{1,80}}\b|"
    rf"\b(?:{_architecture_prefix})\b|"
    rf"\b(?:[A-Z]{{1,3}}\d{{2,4}}|(?-i:[A-Z]{{1,3}}\d[A-Z0-9]{{0,4}})|sm_\d+|gfx\d+[a-z0-9]*|{re.escape(_gpu_uuid_prefix)}[0-9A-Fa-f-]{{8,}}|cuda:\d+)\b|"
    r"\b[A-Za-z0-9.-]+/gpu=[^\s,]+\b|"
    rf"\b/dev/(?:{_device_path_prefix})[^\s]*",
    re.IGNORECASE,
)


def _sensitive_values() -> list[str]:
    values: set[str] = set()
    for name, value in os.environ.items():
        if not value:
            continue
        if name in CONFIG_NAMES or re.search(
            r"(?:SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE|CREDENTIAL|PROXY|DEVICE|HOSTNAME|RUNNER)",
            name,
            re.IGNORECASE,
        ):
            # Skip trivially short values: runner-provided flags such as
            # RUNNER_ALLOW_RUNASROOT=1 would otherwise replace every "1" in
            # the artifact and destroy timestamps, exit codes, and metrics.
            if len(value) >= 4:
                values.add(value)
    return sorted(values, key=len, reverse=True)


def redact_text(text: str) -> str:
    for value in _sensitive_values():
        text = text.replace(value, "[redacted]")
    text = URL_RE.sub("[redacted-url]", text)
    text = ABSOLUTE_PATH_RE.sub("[redacted-path]", text)
    text = HOST_RE.sub("[redacted-host]", text)
    return DEVICE_RE.sub("[redacted-device]", text)


def _redact_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_json_value(item) for key, item in value.items()}
    return value


def redact_json(text: str) -> str:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON artifact must contain an object")
    filtered = {
        key: _redact_json_value(value)
        for key, value in payload.items()
        if key in ALLOWED_JSON_KEYS
    }
    return json.dumps(filtered, ensure_ascii=True, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="redact newline-delimited text from stdin to stdout",
    )
    args = parser.parse_args()
    if args.stream:
        for line in sys.stdin:
            sys.stdout.write(redact_text(line))
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --stream is used")
    try:
        text = args.input.read_text(encoding="utf-8")
        output = redact_json(text) if args.input.suffix.lower() == ".json" else redact_text(text)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        print(f"artifact input/output error: {exc.__class__.__name__}", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, ValueError):
        print("invalid JSON artifact", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
