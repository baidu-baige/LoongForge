# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "release_request.py"
SPEC = importlib.util.spec_from_file_location("release_request", MODULE_PATH)
release_request = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules["release_request"] = release_request
SPEC.loader.exec_module(release_request)


def request(**overrides):
    values = {
        "build_image": "true",
        "run_regression": "true",
        "publish_internal": "true",
        "publish_dockerhub": "true",
        "execution_slot": "primary",
        "llm_model": "deepseek_v2_lite",
        "vla_model": "pi05_ddp",
        "baseline_chip": "p",
    }
    values.update(overrides)
    return release_request.parse_release_request(**values)


def test_full_release_is_valid():
    result = request()
    assert result.build_image is True
    assert result.run_regression is True
    assert result.publish_internal is True
    assert result.publish_dockerhub is True
    assert result.baseline_chip == "p"


@pytest.mark.parametrize("chip", ["", "../a", "p/a", "p\nother=value"])
def test_regression_rejects_missing_or_invalid_baseline_chip(chip):
    with pytest.raises(release_request.ReleaseRequestError, match="baseline_chip"):
        request(baseline_chip=chip)


def test_dry_run_reason_can_validate_all_stages_disabled():
    result = request(
        build_image="false",
        run_regression="false",
        publish_internal="false",
        publish_dockerhub="false",
        skip_regression_reason="__dry_run__",
    )
    assert result.skip_regression_reason == "__dry_run__"


def test_build_only_is_valid():
    result = request(
        run_regression="false",
        publish_internal="false",
        publish_dockerhub="false",
        baseline_chip="",
    )
    assert result.build_image is True
    assert result.run_regression is False


def test_rehearsal_is_valid():
    result = request(publish_internal="false", publish_dockerhub="false")
    assert result.run_regression is True
    assert result.publish_internal is False


def test_secondary_execution_slot_is_valid():
    result = request(execution_slot="secondary")
    assert result.execution_slot == "secondary"


def test_invalid_execution_slot_is_rejected():
    with pytest.raises(release_request.ReleaseRequestError):
        request(execution_slot="remote-host")


def test_regression_can_use_existing_image():
    result = request(
        build_image="false",
        publish_internal="false",
        publish_dockerhub="false",
        existing_image_ref="registry.example.test/team/project:v1",
    )
    assert result.build_image is False
    assert result.run_regression is True


def test_regression_can_use_local_existing_image():
    result = request(
        build_image="false",
        publish_internal="false",
        publish_dockerhub="false",
        existing_image_ref="loongforge-release:0.2.2-remote-rehearsal-fixed",
    )
    assert result.run_regression is True


def test_publishing_without_regression_requires_reason():
    with pytest.raises(release_request.ReleaseRequestError):
        request(run_regression="false")


def test_regression_requires_build_or_existing_image():
    with pytest.raises(release_request.ReleaseRequestError):
        request(
            build_image="false",
            publish_internal="false",
            publish_dockerhub="false",
        )


def test_at_least_one_stage_must_be_enabled():
    with pytest.raises(release_request.ReleaseRequestError):
        request(
            build_image="false",
            run_regression="false",
            publish_internal="false",
            publish_dockerhub="false",
        )


def test_invalid_debug_suffix_is_rejected():
    with pytest.raises(release_request.ReleaseRequestError):
        request(debug_tag_suffix="bad suffix")


def test_invalid_image_reference_is_rejected():
    with pytest.raises(release_request.ReleaseRequestError):
        request(existing_image_ref="not-an-image")


def test_cli_accepts_execution_slot():
    result = subprocess.run(
        [
            "python3",
            str(MODULE_PATH),
            "--build-image",
            "false",
            "--run-regression",
            "true",
            "--publish-internal",
            "false",
            "--publish-dockerhub",
            "false",
            "--execution-slot",
            "secondary",
            "--llm-model",
            "deepseek_v2_lite",
            "--vla-model",
            "pi05_ddp",
            "--baseline-chip",
            "p",
            "--existing-image-ref",
            "registry.example.test/team/project:v1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "execution_slot=secondary" in result.stdout
    assert "baseline_chip=p" in result.stdout


def test_invalid_model_is_rejected():
    with pytest.raises(release_request.ReleaseRequestError):
        request(llm_model="../private-model")

# --- Release model validation ---

SCRIPT = (
    Path(__file__).parents[1]
    / ".github"
    / "scripts"
    / "validate_release_models.py"
)
SOURCE = Path(__file__).parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("validate_release_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_current_release_models_are_valid():
    module = load_script()
    assert module.validate_llm_model(SOURCE, "deepseek_v2_lite") is None
    assert module.validate_vla_model(SOURCE, "pi05_ddp") is None


def test_unknown_models_are_rejected():
    module = load_script()
    assert module.validate_llm_model(SOURCE, "unknown_model") is not None
    assert module.validate_vla_model(SOURCE, "unknown_model") is not None


def test_cli_validates_models():
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--llm-model",
            "deepseek_v2_lite",
            "--vla-model",
            "pi05_ddp",
            "--source",
            str(SOURCE),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
