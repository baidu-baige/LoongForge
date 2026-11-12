# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the Kimi K3 language-only training examples."""

import argparse
import os
import subprocess
from pathlib import Path
from runpy import run_path


ROOT = Path(__file__).resolve().parents[3]
CONSTANTS = run_path(str(ROOT / "loongforge/models/catalog.py"))


def test_kimi_k3_llm_example_uses_the_llm_training_path():
    script = (ROOT / "examples/kimi_k3/pretrain/pretrain_kimi_k3_llm.sh").read_text()

    assert CONSTANTS["LanguageModelFamilies"].KIMI_K3_LLM == "kimi_k3_llm"
    assert "MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-" in script
    assert "configs/models/kimi_k3/kimi_k3_backbone.yaml" in script
    assert "model_type=kimi_k3_llm" in script
    assert "--training-phase pretrain" in script
    assert "KimiTaskEncoder" not in script
    assert "configs/models/kimi_k3/kimi_k3.yaml" not in script


def test_kimi_k3_llm_sft_example_uses_the_llm_training_path():
    script = (ROOT / "examples/kimi_k3/finetuning/sft_kimi_k3_llm.sh").read_text()

    assert "MODEL_CONFIG_FILE=${MODEL_CONFIG_FILE:-" in script
    assert "configs/models/kimi_k3/kimi_k3_backbone.yaml" in script
    assert "model_type=kimi_k3_llm" in script
    assert "--training-phase sft" in script
    assert "--chat-template kimi-k3-hf" in script
    assert "--sft-dataset openai" in script
    assert "KimiTaskEncoder" not in script
    assert "configs/models/kimi_k3/kimi_k3.yaml" not in script


def test_kimi_k3_llm_sft_logging_keeps_model_override():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--log-timers-to-tensorboard", action="store_true")
    parser.add_argument("--log-memory-to-tensorboard", action="store_true")
    parser.add_argument("--log-validation-ppl-to-tensorboard", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-exp-name")

    for wandb_env in (
        {},
        {"WANDB_API_KEY": ""},
        {"WANDB_API_KEY": "test", "WANDB_PROJECT": "kimi-k3", "WANDB_NAME": "sft"},
        {"WANDB_API_KEY": "test", "WANDB_PROJECT": "kimi-k3", "WANDB_NAME": "sft run [1]"},
    ):
        result = _run_sft_script(wandb_env)
        assert result.returncode == 0, result.stderr
        argv = result.stdout.split("\0")[:-1]
        args, overrides = parser.parse_known_args(argv[argv.index("--log-interval"):])

        assert overrides == ["model_type=kimi_k3_llm"]
        assert args.tensorboard_dir == "/tmp/kimi k3 logs"
        assert args.log_interval == 1
        assert args.log_timers_to_tensorboard
        assert args.log_memory_to_tensorboard
        assert args.log_validation_ppl_to_tensorboard
        assert args.wandb_project == wandb_env.get("WANDB_PROJECT")
        assert args.wandb_exp_name == wandb_env.get("WANDB_NAME")


def test_kimi_k3_llm_sft_rejects_missing_wandb_values():
    for variable in ("WANDB_PROJECT", "WANDB_NAME"):
        for value in (None, ""):
            wandb_env = {"WANDB_API_KEY": "test", "WANDB_PROJECT": "kimi-k3", "WANDB_NAME": "sft"}
            if value is None:
                del wandb_env[variable]
            else:
                wandb_env[variable] = value
            result = _run_sft_script(wandb_env)
            assert result.returncode != 0
            assert variable in result.stderr
            assert not result.stdout


def _run_sft_script(wandb_env):
    env = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
    env.update(wandb_env)
    env["TENSORBOARD_PATH"] = "/tmp/kimi k3 logs"
    return subprocess.run(
        [
            "bash", "-c",
            'torchrun() { printf "%s\\0" "$@"; }; source "$1"',
            "capture", str(ROOT / "examples/kimi_k3/finetuning/sft_kimi_k3_llm.sh"),
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )


if __name__ == "__main__":
    test_kimi_k3_llm_example_uses_the_llm_training_path()
    test_kimi_k3_llm_sft_example_uses_the_llm_training_path()
    test_kimi_k3_llm_sft_logging_keeps_model_override()
    test_kimi_k3_llm_sft_rejects_missing_wandb_values()
