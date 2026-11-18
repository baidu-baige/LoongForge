#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Verify a Docker Hub tag through the authenticated Registry API."""

from __future__ import annotations

import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def registry_request(request, phase):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read(4 * 1024 * 1024 + 1)
                if len(body) > 4 * 1024 * 1024:
                    raise RuntimeError("publisher: verify response is too large")
                return response.headers, body
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            reason = f"HTTP {status}"
            retry = status in (408, 429) or 500 <= status < 600 or (
                status == 404 and phase == "manifest"
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
        ):
            reason, retry = "network error", True
        # Do not log response bodies, URLs or authorization headers.
        print(
            f"publisher: verify {phase} attempt {attempt + 1}/4: {reason}",
            file=sys.stderr,
        )
        if not retry or attempt == 3:
            raise RuntimeError(f"publisher: verify {phase} failed: {reason}")
        time.sleep(2 ** (attempt + 1))


def decode_response(body, phase):
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        raise RuntimeError(f"publisher: verify {phase} returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"publisher: verify {phase} returned invalid JSON")
    return payload


def verify(config_dir, image, source_id, push_digest):
    repository, tag = image.rsplit(":", 1)
    host, separator, remainder = repository.partition("/")
    hub_hosts = ("docker.io", "index.docker.io", "registry-1.docker.io")
    if separator and ("." in host or ":" in host or host == "localhost"):
        if host not in hub_hosts:
            raise RuntimeError("publisher: verify requires a Docker Hub repository")
        repository = remainder
    if "/" not in repository:
        repository = "library/" + repository

    config = json.loads((Path(config_dir) / "config.json").read_text())
    auths = config.get("auths", {})
    auth = next(
        (
            auths[key].get("auth")
            for key in ("https://index.docker.io/v1/", *hub_hosts)
            if auths.get(key, {}).get("auth")
        ),
        "",
    )
    if not auth:
        raise RuntimeError("publisher: verify requires the promotion's Docker Hub login")
    query = urllib.parse.urlencode(
        {"service": "registry.docker.io", "scope": f"repository:{repository}:pull"}
    )
    _, body = registry_request(
        urllib.request.Request(
            "https://auth.docker.io/token?" + query,
            headers={"Authorization": "Basic " + auth},
        ),
        "authentication",
    )
    payload = decode_response(body, "authentication")
    token = payload.get("token") or payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("publisher: verify authentication returned no token")

    path = (
        urllib.parse.quote(repository, safe="/")
        + "/manifests/"
        + urllib.parse.quote(tag, safe="")
    )
    headers, body = registry_request(
        urllib.request.Request(
            "https://registry-1.docker.io/v2/" + path,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": (
                    "application/vnd.docker.distribution.manifest.v2+json, "
                    "application/vnd.oci.image.manifest.v1+json"
                ),
            },
        ),
        "manifest",
    )
    # Hash the manifest bytes as well as checking the registry's digest header.
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    declared_digest = headers.get("Docker-Content-Digest")
    if digest != push_digest or (declared_digest and declared_digest != push_digest):
        raise RuntimeError("publisher: remote tag does not match the pushed digest")
    manifest = decode_response(body, "manifest")
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("digest") != source_id:
        raise RuntimeError("publisher: remote image content does not match the source")
    return digest


if __name__ == "__main__":
    try:
        print(verify(*sys.argv[1:]))
    except RuntimeError as error:
        sys.exit(str(error))
    except (OSError, ValueError, TypeError, KeyError):
        sys.exit("publisher: verify received invalid configuration or response")
