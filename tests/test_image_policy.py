# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys


def run_policy(tmp_path, *, inspect="[{}]", history="", probe_status=0):
    docker = tmp_path / "docker"
    log = tmp_path / "docker.log"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_DOCKER_LOG\"\n"
        'if [[ "$1 $2" == "image inspect" ]]; then printf "%s" "$FAKE_INSPECT"; exit 0; fi\n'
        'if [[ "$1" == history ]]; then printf "%s" "$FAKE_HISTORY"; exit 0; fi\n'
        'if [[ "$1" == run ]]; then exit "$FAKE_PROBE_STATUS"; fi\n'
        "exit 2\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "DOCKER_BIN": str(docker),
        "FAKE_DOCKER_LOG": str(log),
        "FAKE_INSPECT": inspect,
        "FAKE_HISTORY": history,
        "FAKE_PROBE_STATUS": str(probe_status),
    })
    result = subprocess.run(
        [sys.executable, ".github/scripts/image_policy.py", "candidate:test"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    docker_log = log.read_text(encoding="utf-8") if log.exists() else ""
    return result, docker_log


def test_image_policy_accepts_clean_image_with_hardened_probe(tmp_path):
    result, docker_log = run_policy(tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == "image-policy: ok"
    assert "--network none" in docker_log
    assert "--read-only" in docker_log
    assert "--cap-drop ALL" in docker_log
    assert "no-new-privileges" in docker_log
    assert "-path /etc/resolv.conf" in docker_log
    assert "-path /etc/hosts" in docker_log
    assert "-path /etc/hostname" in docker_log


def test_image_policy_accepts_public_references_and_short_source_variables(tmp_path):
    metadata = "https://cloud.baidu.com/docs\nsk = make_tensor(storage.smem_k.data())"
    result, _ = run_policy(tmp_path, history=metadata)

    assert result.returncode == 0
    assert result.stdout.strip() == "image-policy: ok"


def test_image_policy_rejects_bcecmd_without_printing_a_path(tmp_path):
    result, _ = run_policy(tmp_path, probe_status=40)

    assert result.returncode == 1
    assert result.stderr.strip() == "image-policy: prohibited executable"
    assert "/" not in result.stderr


def test_image_policy_rejects_credential_files(tmp_path):
    result, _ = run_policy(tmp_path, probe_status=41)

    assert result.returncode == 1
    assert result.stderr.strip() == "image-policy: credential material"


def test_image_policy_rejects_sensitive_metadata_without_echoing_it(tmp_path):
    marker = "access_" + "key_id=" + "fixture-value-123"
    result, _ = run_policy(tmp_path, inspect=json.dumps([{"Config": {"Env": [marker]}}]))

    assert result.returncode == 1
    assert result.stderr.strip() == "image-policy: sensitive metadata"
    assert marker not in result.stdout + result.stderr


def test_image_policy_rejects_explicit_uppercase_ak_metadata(tmp_path):
    marker = "AK=" + "fixturevalue1234567890"
    result, _ = run_policy(tmp_path, history=marker)

    assert result.returncode == 1
    assert result.stderr.strip() == "image-policy: sensitive metadata"
    assert marker not in result.stdout + result.stderr


def test_image_policy_rejects_internal_endpoint_metadata(tmp_path):
    endpoint = "https://proxy." + "baidu-int" + ".com/service"
    result, _ = run_policy(tmp_path, history=endpoint)

    assert result.returncode == 1
    assert result.stderr.strip() == "image-policy: internal metadata"
    assert endpoint not in result.stdout + result.stderr


def test_image_policy_rejects_sensitive_filesystem_text(tmp_path):
    result, _ = run_policy(tmp_path, probe_status=42)

    assert result.returncode == 1
    assert result.stderr.strip() == "image-policy: sensitive filesystem content"


def test_image_policy_accepts_public_sdk_source_with_credential_parameter_names(tmp_path):
    result, _ = run_policy(tmp_path, history="")
    assert result.returncode == 0
