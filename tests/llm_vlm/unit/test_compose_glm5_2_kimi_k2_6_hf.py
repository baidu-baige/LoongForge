# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


SCRIPT = (
    # tests/llm_vlm/unit -> tests/llm_vlm -> tests -> repo root
    Path(__file__).resolve().parents[3]
    / "examples/glm5.2_vit/checkpoint_convert/compose_glm5_2_kimi_k2_6_hf.py"
)
SPEC = importlib.util.spec_from_file_location("compose_weights", SCRIPT)
compose_weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compose_weights)


def _checkpoint(path, tensors, shard_data, config="{}", tensor_shards=None):
    path.mkdir()
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": tensors}), encoding="utf-8"
    )
    for shard, data in shard_data.items():
        (path / shard).write_bytes(data)
    for shard, tensors_for_shard in (tensor_shards or {}).items():
        save_file(tensors_for_shard, str(path / shard))
    (path / "config.json").write_text(config, encoding="utf-8")


def _load_composed(path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    state_dict = {}
    for shard in set(index["weight_map"].values()):
        state_dict.update(load_file(path / shard))
    return state_dict


def test_compose_materialize_is_independent_and_symlink_mode_is_explicit(tmp_path):
    glm = tmp_path / "glm"
    kimi = tmp_path / "kimi"
    _checkpoint(
        glm,
        {"model.embed_tokens.weight": "model-00001.safetensors"},
        {"model-00001.safetensors": b"glm"},
        '{"model":"glm"}',
        tensor_shards={
            "model-00001.safetensors": {
                "model.embed_tokens.weight": torch.tensor([10.0])
            }
        },
    )
    _checkpoint(
        kimi,
        {"vision_tower.a": "model-00001.safetensors", "text.a": "other.safetensors"},
        {"model-00001.safetensors": b"kimi", "other.safetensors": b"ignored"},
        tensor_shards={
            "model-00001.safetensors": {
                "vision_tower.a": torch.tensor([1.0]),
                "model.embed_tokens.weight": torch.tensor([2.0]),
            }
        },
    )

    linked = tmp_path / "linked"
    compose_weights.compose(glm, kimi, linked)
    assert (linked / "glm-model-00001.safetensors").is_symlink()
    assert (linked / "config.json").is_symlink()
    assert not (linked / "kimi-model-00001.safetensors").is_symlink()
    assert set(load_file(linked / "kimi-model-00001.safetensors")) == {
        "vision_tower.a"
    }
    loaded = _load_composed(linked)
    assert torch.equal(loaded["model.embed_tokens.weight"], torch.tensor([10.0]))
    assert torch.equal(loaded["vision_tower.a"], torch.tensor([1.0]))

    materialized = tmp_path / "materialized"
    compose_weights.compose(glm, kimi, materialized, materialize=True)
    assert not (materialized / "glm-model-00001.safetensors").is_symlink()
    assert not (materialized / "config.json").is_symlink()
    assert set(load_file(materialized / "kimi-model-00001.safetensors")) == {
        "vision_tower.a"
    }
    composed = json.loads((materialized / "model.safetensors.index.json").read_text())
    assert composed["metadata"]["total_size"] == sum(
        path.stat().st_size
        for path in {
            materialized / "glm-model-00001.safetensors",
            materialized / "kimi-model-00001.safetensors",
        }
    )
    assert json.loads((materialized / "composition.json").read_text())[
        "tensor_files_rewritten"
    ] == 1

    materialized_glm = (materialized / "glm-model-00001.safetensors").read_bytes()
    (glm / "model-00001.safetensors").write_bytes(b"changed")
    assert (materialized / "glm-model-00001.safetensors").read_bytes() == materialized_glm


def test_compose_rejects_tensor_name_conflicts(tmp_path):
    glm = tmp_path / "glm"
    kimi = tmp_path / "kimi"
    _checkpoint(glm, {"vision_tower.a": "glm.safetensors"}, {"glm.safetensors": b"glm"})
    _checkpoint(kimi, {"vision_tower.a": "kimi.safetensors"}, {"kimi.safetensors": b"kimi"})

    with pytest.raises(ValueError, match="Tensor name conflict"):
        compose_weights.compose(glm, kimi, tmp_path / "output")
