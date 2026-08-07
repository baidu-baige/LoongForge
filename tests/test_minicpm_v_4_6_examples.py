# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = (
    REPO_ROOT / "examples/minicpm_v_4_6/pretrain/pretrain_minicpm_v_4_6.sh",
    REPO_ROOT / "examples/minicpm_v_4_6/finetuning/sft_minicpm_v_4_6.sh",
)


def test_minicpm_training_scripts_use_standard_argument_groups():
    expected_groups = (
        "MODEL_CONFIG_ARGS",
        "DATA_ARGS",
        "TRAINING_ARGS",
        "MODEL_PARALLEL_ARGS",
        "LOGGING_ARGS",
    )

    for script in TRAINING_SCRIPTS:
        content = script.read_text(encoding="utf-8")
        definitions = [content.index(f"{group}=(") for group in expected_groups]
        expansions = [content.rindex(f'"${{{group}[@]}}"') for group in expected_groups]

        assert definitions == sorted(definitions)
        assert expansions == sorted(expansions)
        assert "--optimizer-backend" not in content


def test_minicpm_sft_script_runs_repository_entrypoint_with_alignment_overrides(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    capture = tmp_path / "torchrun-argv.txt"
    torchrun = fake_bin / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${CAPTURE_PATH}\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    entrypoint = tmp_path / "alignment_entrypoint.py"
    entrypoint.write_text("raise SystemExit(0)\n", encoding="utf-8")
    script = REPO_ROOT / "examples/minicpm_v_4_6/finetuning/sft_minicpm_v_4_6.sh"
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURE_PATH": str(capture),
        "DATA_PATH": str(tmp_path / "data.jsonl"),
        "PRETRAINED_CHECKPOINT": str(tmp_path / "checkpoint"),
        "SAVE_PATH": str(tmp_path / "output"),
        "TRAIN_ENTRYPOINT": str(entrypoint),
        "GPUS_PER_NODE": "1",
        "TRAIN_ITERS": "10",
        "LEARNING_RATE": "1e-5",
        "LR_WARMUP_ITERS": "1",
        "USE_DISTRIBUTED_OPTIMIZER": "0",
        "DETERMINISTIC_MODE": "1",
        "DISABLE_BF16_REDUCED_PRECISION_MATMUL": "1",
        "SEED": "1234",
    }

    subprocess.run(["bash", str(script)], env=environment, check=True)

    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert str(entrypoint) in arguments
    assert arguments[arguments.index("--lr") + 1] == "1e-5"
    assert arguments[arguments.index("--lr-warmup-iters") + 1] == "1"
    assert arguments[arguments.index("--train-iters") + 1] == "10"
    assert "--use-distributed-optimizer" not in arguments
    assert "--deterministic-mode" in arguments
    assert "--disable-bf16-reduced-precision-matmul" in arguments
    assert arguments[arguments.index("--seed") + 1] == "1234"
