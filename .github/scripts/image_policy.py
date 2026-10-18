#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Reject credentials, internal configuration, and prohibited tools in images."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys


CREDENTIAL_PATTERN = re.compile(
    r"(?i)(?:\bAKIA[0-9A-Z]{16}\b|"
    r"\b(?:access[_-]?key(?:[_-]?id)?|secret[_-]?(?:access[_-]?)?key|"
    r"aws[_-](?:access|secret)[_-]?key(?:[_-]?id)?)"
    r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_.-]{16,}|"
    r"\b(?:authorization|x-bce-security-token)\s*[:=]\s*['\"]?"
    r"(?:AWS4|bce-auth-v1|bearer|basic)\b)"
)
# AK/SK are accepted only as explicit uppercase field names.  Short variable
# names such as `sk` are common in CUDA/C++ source and are not credentials.
SHORT_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:AK|SK)\s*[:=]\s*['\"]?[A-Za-z0-9/+_.-]{16,}"
)
INTERNAL_PATTERN = re.compile(
    r"(?i)(?:[a-z0-9-]+\.)*baidu-int\.com\b|"
    r"(?:[a-z0-9-]+\.)*baidubce\.com\b|"
    r"\biregistry\.[a-z0-9.-]+|/ssd[0-9]+/|/workspace/(?:runner-|runner/|secrets/)"
)

FILESYSTEM_PROBE = r"""
set -eu
if command -v bcecmd >/dev/null 2>&1; then
  exit 40
fi
if [ -n "$(find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
  -type f -name bcecmd -print -quit 2>/dev/null)" ]; then
  exit 40
fi
if [ -n "$(find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o \
  \( -type f -o -type l \) \
  \( -path '*/.bce/*' -o -path '*/.aws/credentials' -o -path '*/.config/bce/*' \
     -o -name source-manifest.env -o -name source_manifest.env \
     -o -path '*/.docker/config.json' -o -name .netrc \) -print -quit 2>/dev/null)" ]; then
  exit 41
fi
internal_pattern='baidu-int\.com|baidubce\.com|iregistry\.|/ssd[0-9]+/|/workspace/(runner-|runner/|secrets/)'
# Public base images ship SDK examples and documentation containing fake AKIA
# values and parameter names.  Credential content is therefore scanned in
# operator/configuration locations, while internal endpoints are scanned over
# all image paths.  Tracked project content is covered by sensitive_scan.py.
scan_files() {
  pattern="$1"
  shift
  find "$@" -xdev \
    \( -path /workspace/LoongForge -o -path /proc -o -path /sys -o -path /dev \
       -o -path /etc/resolv.conf -o -path /etc/hosts -o -path /etc/hostname \) -prune -o \
    -type f -size -4M -readable -exec grep -I -E -l "$pattern" {} + 2>/dev/null | head -1
}
# Source distributions contain code and documentation with names such as
# `access_key = self.access_key`; filesystem scanning therefore requires a
# quoted literal (or a known signed token) before flagging a credential.
named_credential_pattern='(access[_-]?key([_-]?id)?|secret[_-]?(access[_-]?)?key|aws[_-](access|secret)[_-]?key([_-]?id)?)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9/+.-]{16,}([[:space:]]|$)|AKIA[0-9A-Z]{16}|(authorization|x-bce-security-token)[[:space:]]*[:=][[:space:]]*(AWS4|bce-auth-v1|bearer|basic)'
short_credential_pattern='(^|[^[:alnum:]_])(AK|SK)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9/+.-]{16,}([[:space:]]|$)'
if [ -n "$(scan_files "$named_credential_pattern" /root /home /etc /workspace)" ] || \
   [ -n "$(scan_files "$short_credential_pattern" /root /home /etc /workspace)" ] || \
   [ -n "$(scan_files "$internal_pattern" /root /home /etc /opt /usr/local /workspace)" ]; then
  exit 42
fi
"""


def docker(docker_bin: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = [docker_bin, *args]
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(command, 124, "", "")


def image_metadata(inspect_json: str, history_json: str) -> str:
    """Return image metadata without daemon-local graph-driver paths."""
    parsed = json.loads(inspect_json)
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise json.JSONDecodeError("multiple image records", inspect_json, 0)
        inspect = parsed[0]
    else:
        inspect = parsed
    if not isinstance(inspect, dict):
        raise json.JSONDecodeError("image record is not an object", inspect_json, 0)
    inspect.pop("GraphDriver", None)
    return json.dumps(inspect, sort_keys=True) + "\n" + history_json


def fail(message: str) -> int:
    print(f"image-policy: {message}", file=sys.stderr)
    return 1


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1]:
        print("usage: image_policy.py IMAGE", file=sys.stderr)
        return 2
    image = sys.argv[1]
    docker_bin = os.environ.get("DOCKER_BIN", "docker")

    inspect = docker(docker_bin, "image", "inspect", image)
    if inspect.returncode != 0:
        return fail("inspect failed")

    history = docker(
        docker_bin,
        "history",
        "--no-trunc",
        "--format",
        "{{json .CreatedBy}}",
        image,
    )
    if history.returncode != 0:
        return fail("history inspection failed")
    try:
        metadata = image_metadata(inspect.stdout, history.stdout)
    except json.JSONDecodeError:
        return fail("inspect failed")
    if CREDENTIAL_PATTERN.search(metadata) or SHORT_CREDENTIAL_PATTERN.search(metadata):
        return fail("sensitive metadata")
    if INTERNAL_PATTERN.search(metadata):
        return fail("internal metadata")

    probe = docker(
        docker_bin,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        FILESYSTEM_PROBE,
    )
    if probe.returncode == 40:
        return fail("prohibited executable")
    if probe.returncode == 41:
        return fail("credential material")
    if probe.returncode == 42:
        return fail("sensitive filesystem content")
    if probe.returncode != 0:
        return fail("filesystem inspection failed")

    print("image-policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
