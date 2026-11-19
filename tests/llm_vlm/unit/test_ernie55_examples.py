# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check launcher arguments without starting distributed workers."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "examples/ernie5.5"


@pytest.mark.parametrize(
    "script,entrypoint,options",
    [
        (
            "pretrain/pretrain_ernie5_5_flash_base_bf16.sh",
            "loongforge/train.py",
            {"--training-phase": "pretrain", "--save": "/tmp/train output"},
        ),
        (
            "finetuning/sft_ernie5_5_flash_base_bf16.sh",
            "loongforge/train.py",
            {"--training-phase": "sft", "--chat-template": "ernie5.5-hf", "--sft-dataset": "openai"},
        ),
    ],
)
def test_launcher_keeps_paths_and_node_topology(tmp_path, script, entrypoint, options):
    capture = tmp_path / "arguments.json"
    launcher = tmp_path / "torchrun"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    launcher.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CAPTURE=str(capture),
        LOONGFORGE_PATH=str(ROOT),
        MEGATRON_PATH="/tmp/Megatron test",
        TOKENIZER_PATH="/tmp/tokenizer input",
        CHECKPOINT_PATH="/tmp/model input",
        CHECKPOINT_PATH_SAVE="/tmp/train output",
        HF_MODEL_PATH="/tmp/HF input",
        DATA_PATH="/tmp/data input",
        SAVE_ROOT="/tmp/mcore output",
        LOAD="/tmp/mcore input",
        SAVE="/tmp/hf output",
        WORLD_SIZE="2",
        RANK="1",
        GPUS_PER_NODE="8",
        EP_SIZE="16",
    )
    subprocess.run(["bash", str(EXAMPLES / script)], env=env, check=True)
    argv = json.loads(capture.read_text())
    assert str(ROOT / entrypoint) in argv
    for key, expected in {
        "--nnodes": "2",
        "--node_rank": "1",
        "--nproc_per_node": "8",
        "--expert-model-parallel-size": "16",
        "--expert-tensor-parallel-size": "1",
        "--config-file": str(ROOT / "configs/models/ernie5_5/ernie5_5_flash_base.yaml"),
        **options,
    }.items():
        assert argv[argv.index(key) + 1] == expected
    assert "--context-parallel-size" not in argv
    assert "--calculate-per-token-loss" not in argv


@pytest.mark.parametrize(
    "direction,load_platform,save_platform,load_path,save_path",
    [
        ("hf_to_mcore", "huggingface", "mcore", "/tmp/HF input", "/tmp/mcore output"),
        ("mcore_to_hf", "mcore", "huggingface", "/tmp/mcore input", "/tmp/HF output"),
    ],
)
def test_conversion_uses_model_and_mapping_yaml(
    tmp_path, direction, load_platform, save_platform, load_path, save_path,
):
    capture = tmp_path / "arguments.json"
    launcher = tmp_path / "python"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    launcher.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        CAPTURE=str(capture),
        LOONGFORGE_PATH=str(ROOT),
        MEGATRON_PATH="/tmp/Megatron test",
        HF_MODEL_PATH="/tmp/HF input",
        SAVE_ROOT="/tmp/mcore output",
        LOAD="/tmp/mcore input",
        SAVE="/tmp/HF output",
        PP_SIZE="2",
        EP_SIZE="2",
    )
    script = EXAMPLES / f"checkpoint_convert/convert_ernie5_5_flash_base_{direction}_bf16.sh"
    subprocess.run(["bash", str(script)], env=env, check=True)
    argv = json.loads(capture.read_text())
    assert argv[0] == str(ROOT / "tools/mcore_checkpoint_convert/module_convertor/model.py")
    assert argv[argv.index("--config_file") + 1] == str(
        ROOT / "configs/models/ernie5_5/ernie5_5_flash_base.yaml"
    )
    assert argv[argv.index("--convert_file") + 1] == str(
        ROOT / "configs/models/ernie5_5/ckpt_convert/ernie5_5_convert.yaml"
    )
    for key, value in {
        "load_platform": load_platform,
        "save_platform": save_platform,
        "load_ckpt_path": load_path,
        "save_ckpt_path": save_path,
        "tensor_model_parallel_size": "1",
        "pipeline_model_parallel_size": "2",
        "expert_parallel_size": "2",
        "expert_tensor_parallel_size": "1",
    }.items():
        assert f"--{key}={value}" in argv
    assert not any(arg.startswith(("--hf_official_config_file", "--hf-official-config-file")) for arg in argv)
