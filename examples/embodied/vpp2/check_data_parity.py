# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU real-data fingerprints: 8 two-sample batches plus episode-tail probes."""

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate

base_root = Path(os.environ["VPP2_BASE_ROOT"]).resolve()
trans_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(base_root / "src"), str(trans_root)]
from loongforge.embodied.data.datasets.vpp2.transforms.data_configuration_vpp2 import VPP2DataConfig
from loongforge.embodied.data.datasets.vpp2.transforms.vpp2_transform import (
    VPP2KeyMappingTransform,
    VPP2LinearNormalizeTransform,
)
from loongforge.embodied.data.datasets.vpp2.vpp2_dataset import build_vpp2_lerobot_dataset
from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    cfg = OmegaConf.load(base_root / "configs/libero/train_action.yaml")
    assets = Path(os.environ["VPP2_ASSETS_ROOT"])
    data_root = Path(os.environ.get("VPP2_DATASET_ROOT", str(assets / "libero_subset")))
    roots = [str(data_root / f"libero_{suite}_no_noops_lerobot") for suite in ["spatial", "object", "goal", "10"]]
    cfg.data.train.dataset_dirs = roots
    cfg.data.train.pretrained_norm_stats = str(assets / "libero/dataset_stats.json")
    cfg.data.train.text_embedding_cache_dir = str(assets / "text_cache_real")
    base = instantiate(cfg.data.train)
    model_cfg = VPP2ModelConfig()
    data_cfg = VPP2DataConfig(
        norm_stats_path=cfg.data.train.pretrained_norm_stats,
        text_embedding_cache_dir=cfg.data.train.text_embedding_cache_dir,
    )
    trans = build_vpp2_lerobot_dataset(
        model_cfg, data_cfg, SimpleNamespace(dataset_path=",".join(roots), video_backend="pyav")
    )
    assert len(base) == len(trans) and len(base) > 0
    norm = VPP2LinearNormalizeTransform(data_cfg.norm_stats_path, data_cfg.delta_action_dim_mask)
    mapper = VPP2KeyMappingTransform(text_embedding_cache_dir=data_cfg.text_embedding_cache_dir)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(base), generator=generator)[:16].tolist() + [0, len(base) - 1]
    all_equal = True
    for start in range(0, len(indices), 2):
        selected = indices[start : start + 2]
        # Strict access prevents BASE's exception->random-sample fallback.
        b = default_collate([base.get_strict_item(index)[1] for index in selected])
        t = default_collate([mapper.apply(norm.apply(trans[index])) for index in selected])
        record = {}
        for key, value in b.items():
            other = t[key]
            if isinstance(value, torch.Tensor):
                same = torch.equal(value, other) and value.dtype == other.dtype and value.shape == other.shape
                record[key] = dict(
                    equal=same,
                    shape=list(value.shape),
                    base_dtype=str(value.dtype),
                    trans_dtype=str(other.dtype),
                    max_delta=float((value.float() - other.float()).abs().max()),
                    base_sha=digest(value),
                    trans_sha=digest(other),
                )
            else:
                record[key] = dict(equal=value == other)
        all_equal = all_equal and all(item["equal"] for item in record.values())
        print(json.dumps(dict(batch=start // 2, indices=selected, keys=record)), flush=True)
    raise SystemExit(0 if all_equal else 1)


if __name__ == "__main__":
    main()
