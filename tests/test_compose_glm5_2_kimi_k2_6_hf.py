import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/glm5.2_vit/checkpoint_convert/compose_glm5_2_kimi_k2_6_hf.py"
)
SPEC = importlib.util.spec_from_file_location("compose_weights", SCRIPT)
compose_weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compose_weights)


def _checkpoint(path, tensors, shard_data, config="{}"):
    path.mkdir()
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": tensors}), encoding="utf-8"
    )
    for shard, data in shard_data.items():
        (path / shard).write_bytes(data)
    (path / "config.json").write_text(config, encoding="utf-8")


def test_compose_materialize_is_independent_and_symlink_mode_is_explicit(tmp_path):
    glm = tmp_path / "glm"
    kimi = tmp_path / "kimi"
    _checkpoint(glm, {"language.a": "model-00001.safetensors"}, {"model-00001.safetensors": b"glm"}, '{"model":"glm"}')
    _checkpoint(
        kimi,
        {"vision_tower.a": "model-00001.safetensors", "text.a": "other.safetensors"},
        {"model-00001.safetensors": b"kimi", "other.safetensors": b"ignored"},
    )

    linked = tmp_path / "linked"
    compose_weights.compose(glm, kimi, linked)
    assert (linked / "glm-model-00001.safetensors").is_symlink()
    assert (linked / "config.json").is_symlink()

    materialized = tmp_path / "materialized"
    compose_weights.compose(glm, kimi, materialized, materialize=True)
    assert not (materialized / "glm-model-00001.safetensors").is_symlink()
    assert not (materialized / "config.json").is_symlink()
    assert (materialized / "kimi-model-00001.safetensors").read_bytes() == b"kimi"
    assert json.loads((materialized / "model.safetensors.index.json").read_text())["metadata"]["total_size"] == 7

    (glm / "model-00001.safetensors").write_bytes(b"changed")
    assert (materialized / "glm-model-00001.safetensors").read_bytes() == b"glm"


def test_compose_rejects_tensor_name_conflicts(tmp_path):
    glm = tmp_path / "glm"
    kimi = tmp_path / "kimi"
    _checkpoint(glm, {"vision_tower.a": "glm.safetensors"}, {"glm.safetensors": b"glm"})
    _checkpoint(kimi, {"vision_tower.a": "kimi.safetensors"}, {"kimi.safetensors": b"kimi"})

    with pytest.raises(ValueError, match="Tensor name conflict"):
        compose_weights.compose(glm, kimi, tmp_path / "output")
