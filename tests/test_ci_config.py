# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from tools.ci_config import baseline_models, load_env_file


def test_load_env_file_does_not_execute_values(tmp_path: Path):
    config = tmp_path / "ci.env"
    config.write_text("# local only\nMODEL=llama3_8b\nVALUE='a b'\n")

    assert load_env_file(config) == {"MODEL": "llama3_8b", "VALUE": "a b"}


def test_baseline_models_defaults_to_models_present_for_environment(tmp_path: Path):
    baseline = tmp_path / "baseline" / "default" / "A"
    baseline.mkdir(parents=True)
    (baseline / "qwen3_14b.json").write_text("{}")
    (baseline / "llama3_8b.json").write_text("{}")

    import os

    old = os.environ.get("LOONGFORGE_BASELINE_A")
    os.environ["LOONGFORGE_BASELINE_A"] = "A"
    try:
        assert baseline_models("a", tmp_path / "baseline") == ["llama3_8b", "qwen3_14b"]
        assert baseline_models("a", tmp_path / "baseline", ["llama3_8b"]) == ["llama3_8b"]
    finally:
        if old is None:
            os.environ.pop("LOONGFORGE_BASELINE_A", None)
        else:
            os.environ["LOONGFORGE_BASELINE_A"] = old


def test_requested_model_without_baseline_is_not_selected(tmp_path: Path):
    baseline = tmp_path / "baseline" / "default" / "A"
    baseline.mkdir(parents=True)
    (baseline / "qwen3_14b.json").write_text("{}")

    import os

    os.environ["LOONGFORGE_BASELINE_A"] = "A"
    assert baseline_models("a", tmp_path / "baseline", ["missing"]) == []
