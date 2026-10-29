# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for embodied DeepSpeed configuration and resume discovery."""

import json
from types import SimpleNamespace

import pytest

from loongforge.embodied.distributed.deepspeed import (
    get_latest_deepspeed_checkpoint,
    load_deepspeed_config,
)


def _args(**overrides):
    values = {
        "per_device_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "dtype": "bfloat16",
        "clip_grad": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_deepspeed_config_resolves_vpp2_reference_geometry(tmp_path):
    path = tmp_path / "zero2.json"
    path.write_text(
        json.dumps(
            {
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "gradient_accumulation_steps": "auto",
                "zero_optimization": {"stage": 2},
            }
        )
    )
    config = load_deepspeed_config(str(path), _args(), world_size=8)
    assert config["train_micro_batch_size_per_gpu"] == 32
    assert config["gradient_accumulation_steps"] == 1
    assert config["train_batch_size"] == 256
    assert config["bf16"]["enabled"] is True
    assert config["fp16"]["enabled"] is False
    assert "gradient_clipping" not in config
    assert config["zero_optimization"]["stage"] == 2


def test_deepspeed_config_rejects_mismatched_explicit_batch(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "train_batch_size": 8,
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "zero_optimization": {"stage": 2},
            }
        )
    )
    with pytest.raises(ValueError, match="train_micro_batch_size_per_gpu"):
        load_deepspeed_config(str(path), _args(), world_size=8)


def test_latest_deepspeed_checkpoint_fails_closed(tmp_path):
    complete = tmp_path / "steps_20"
    tag = complete / "deepspeed/global_step20"
    tag.mkdir(parents=True)
    (tag / "mp_rank_00_model_states.pt").touch()
    (tag / "zero_pp_rank_0_mp_rank_00_optim_states.pt").touch()
    for kind in ("rng", "dataloader"):
        (complete / kind).mkdir()
        for rank in range(2):
            (complete / kind / f"rank_{rank}.pt").touch()
    (complete / "resume_meta.json").write_text(json.dumps({
        "completed_steps": 20, "epoch": 3, "ckpt_format": "deepspeed", "world_size": 2,
    }))
    assert get_latest_deepspeed_checkpoint(str(tmp_path)) == (str(complete), 20, 3)
    (tmp_path / "steps_30").mkdir()
    with pytest.raises(FileNotFoundError):
        get_latest_deepspeed_checkpoint(str(tmp_path))


def test_deepspeed_clip_contract_is_not_silently_ignored(tmp_path):
    path = tmp_path / "zero2.json"
    path.write_text(json.dumps({"zero_optimization": {"stage": 2}}))
    with pytest.raises(ValueError, match="clip-grad"):
        load_deepspeed_config(str(path), _args(clip_grad=1.0), world_size=8)
    path.write_text(json.dumps({"gradient_clipping": "auto"}))
    assert load_deepspeed_config(str(path), _args(clip_grad=1.0), 8)["gradient_clipping"] == 1.0
