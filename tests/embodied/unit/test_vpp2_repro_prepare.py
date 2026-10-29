# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare-run checks with synthetic file fixtures, not model baselines."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location(
    "prepare_run", Path(__file__).resolve().parents[3] / "examples/embodied/vpp2/prepare_run.py"
)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def run_setup(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    for name in [
        "libero_video_step010000.pt",
        "libero/action_init.pt",
        "libero/dataset_stats.json",
        "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "alignment/init_seed42.pt",
    ]:
        path = assets / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic fixture")
    for suite in ["spatial", "object", "goal", "10"]:
        meta = assets / f"libero_subset/libero_{suite}_no_noops_lerobot/meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{}")
        (meta / "tasks.jsonl").write_text(json.dumps(dict(task="test fixture")))
    cache = assets / "text_cache_real"
    cache.mkdir()
    prompt = "A video recorded from a robot's point of view executing the following instruction: test fixture"
    key = hashlib.sha256(prompt.encode()).hexdigest()
    (cache / f"{key}.t5_len128.wan21i2v14b480p.pt").write_text("synthetic fixture")
    out = tmp_path / "run"
    monkeypatch.setenv("VPP2_ASSETS_ROOT", str(assets))
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    for name in [
        "VPP2_VIDEO_CHECKPOINT",
        "VPP2_ALIGNMENT_INIT",
        "VPP2_DATASET_ROOT",
        "TRAIN_ITERS",
        "GPUS_PER_NODE",
        "SEED",
        "SCHEDULE_STEPS",
        "PER_DEVICE_BATCH_SIZE",
        "NUM_WORKERS",
        "VPP2_ACTION_TRAIN_MICROBATCH_SIZE",
    ]:
        monkeypatch.delenv(name, raising=False)
    return assets, out


def test_prepare_base_config_preserves_horizon_without_gpu(run_setup, monkeypatch):
    assets, out = run_setup
    repo = Path(__file__).resolve().parents[4] / "video-prediction-policy-2"
    if not repo.is_dir():
        pytest.skip("sibling BASE checkout is required")
    monkeypatch.setattr(sys, "argv", ["prepare_run", "--repo", str(repo), "--side", "base"])
    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda _: "0.18.9")
    prepare.main()
    cfg = OmegaConf.load(out / "config.yaml")
    assert cfg.max_steps == 100 and cfg.comparison_schedule_steps == 30000 and cfg.warmup_steps == 500
    assert cfg.model.action_dit_pretrained_path == str(assets / "libero/action_init.pt")
    assert cfg.save_final_checkpoint is False
    assert cfg.data.train.context_len == 128 and len(cfg.data.train.dataset_dirs) == 4
    assert (out / "asset_sha256.json").is_file()
    with pytest.raises(FileExistsError):
        prepare.main()


def test_prepare_dependency_guard_precedes_output_creation(run_setup, monkeypatch):
    _, out = run_setup
    monkeypatch.setattr(sys, "argv", ["prepare_run", "--repo", str(Path.cwd()), "--side", "loongforge"])
    monkeypatch.setattr(prepare.importlib.metadata, "version", lambda _: "0.17.6")
    with pytest.raises(RuntimeError, match="0.18.9"):
        prepare.main()
    assert not out.exists()


def test_native_prepare_rejects_old_chunk_override(run_setup, monkeypatch):
    _, out = run_setup
    monkeypatch.setenv("VPP2_ACTION_TRAIN_MICROBATCH_SIZE", "1")
    monkeypatch.setattr(sys, "argv", ["prepare_run", "--repo", str(Path.cwd()), "--side", "loongforge", "--dry-run"])
    with pytest.raises(ValueError, match="MICROBATCH|chunk|batch"):
        prepare.main()
    assert not out.exists()
