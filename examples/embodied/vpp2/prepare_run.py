#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a fresh, audited run directory; --dry-run never imports CUDA/DS."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--side", choices=["base", "loongforge"], required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    e = os.environ
    assets = Path(e["VPP2_ASSETS_ROOT"]).resolve()
    data = Path(e.get("VPP2_DATASET_ROOT", str(assets / "libero_subset"))).resolve()
    video = Path(
        e.get("VPP2_VIDEO_CHECKPOINT", str(assets / "libero_video_step010000.pt"))
    ).resolve()
    init = Path(e.get("VPP2_ALIGNMENT_INIT", str(assets / "alignment/init_seed42.pt"))).resolve()
    out = Path(e["OUTPUT_DIR"]).resolve()
    steps, horizon = int(e.get("TRAIN_ITERS", 100)), int(e.get("SCHEDULE_STEPS", 30000))
    world, batch = int(e.get("GPUS_PER_NODE", 8)), int(e.get("PER_DEVICE_BATCH_SIZE", 32))
    workers = int(e.get("NUM_WORKERS", 0))
    # Native training uses one full logical batch, exactly as upstream.
    chunk = batch
    if e.get("VPP2_ACTION_TRAIN_MICROBATCH_SIZE"):
        raise ValueError("Native parity no longer supports internal batch chunks")
    seed = int(e.get("SEED", 42))
    if min(steps, horizon, world, batch, chunk) <= 0 or steps > horizon or workers < 0:
        raise ValueError("Invalid run geometry or stop > schedule horizon")
    if workers != 0:
        raise ValueError(
            "Audited comparison uses NUM_WORKERS=0; worker-seed parity is not established"
        )
    if world * batch != 256 and e.get("ALLOW_BATCH_CHANGE") != "1":
        raise ValueError(
            "Expected global batch256. ALLOW_BATCH_CHANGE=1 labels a diagnostic, not paper parity"
        )
    roots = [data / f"libero_{s}_no_noops_lerobot" for s in ["spatial", "object", "goal", "10"]]
    stats, cache = assets / "libero/dataset_stats.json", assets / "text_cache_real"
    files = [
        video,
        assets / "libero/action_init.pt",
        stats,
        assets / "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        assets / "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for root in roots:
        if not (root / "meta/info.json").is_file():
            raise FileNotFoundError(root)
        for line in (root / "meta/tasks.jsonl").read_text().splitlines():
            task = json.loads(line)["task"]
            prompt = (
                "A video recorded from a robot's point of view executing the following instruction: "
                + task
            )
            key = hashlib.sha256(prompt.encode()).hexdigest()
            path = cache / f"{key}.t5_len128.wan21i2v14b480p.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
    cfg = None
    if args.side == "base":
        cfg = OmegaConf.load(args.repo / "configs/libero/train_action.yaml")
        cfg.output_dir = str(out)
        cfg.batch_size, cfg.num_workers, cfg.max_steps = batch, workers, steps
        cfg.num_epochs = 1000000
        cfg.warmup_steps = 500
        cfg.comparison_schedule_steps = horizon
        cfg.gradient_accumulation_steps, cfg.seed = 1, seed
        cfg.save_every = int(e.get("SAVE_INTERVAL", 0))
        cfg.save_final_checkpoint = False
        cfg.eval_every, cfg.log_every = 0, 1
        cfg.model.model_id = str(assets / "Wan2.1-I2V-14B-480P")
        cfg.model.dit_checkpoint_path = str(video)
        cfg.model.action_dit_pretrained_path = str(assets / "libero/action_init.pt")
        cfg.data.train.dataset_dirs = [str(r) for r in roots]
        cfg.data.train.pretrained_norm_stats = str(stats)
        cfg.data.train.text_embedding_cache_dir = str(cache)
        cfg.reference_global_batch_size = world * batch
        cfg.wandb.enabled = False
    report = {
        "side": args.side,
        "output_dir": str(out),
        "steps": steps,
        "schedule_steps": horizon,
        "world_size": world,
        "per_device_batch": batch,
        "ga": 1,
        "global_batch": world * batch,
        "seed": seed,
        "num_workers": workers,
        "internal_chunk": chunk,
        "init": str(init),
        "dtype": "bfloat16",
        "autocast": False,
        "deterministic": e.get("VPP2_DETERMINISTIC", "0") == "1",
        "engine": "deepspeed_zero2",
        "dataset_roots": [str(r) for r in roots],
        "stats_sha256": sha256(stats),
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        print("DRY RUN: assets/config checked; no training and no dependency/GPU sufficiency claim")
        return
    ds = importlib.metadata.version("deepspeed")
    if ds != "0.18.9":
        raise RuntimeError(
            f"Reproduction pins deepspeed==0.18.9; installed {ds}. Use an isolated environment."
        )
    if not init.is_file():
        raise FileNotFoundError(
            f"{init}; run prepare_alignment_init.py once and use the SAME file on both sides"
        )
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_manifest.json").write_text(json.dumps(report, indent=2))
    (out / "pip_freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    # NVIDIA images record torch/torchvision as direct wheel URLs, not == pins.
    # Preserve both provenance (freeze) and actual distribution versions.
    dependencies = [
        "torch",
        "numpy",
        "deepspeed",
        "accelerate",
        "diffusers",
        "transformers",
        "torchvision",
        "av",
    ]
    (out / "dependency_versions.json").write_text(
        json.dumps({name: importlib.metadata.version(name) for name in dependencies}, indent=2)
    )
    for cmd, name in [
        (["rev-parse", "HEAD"], "git_head.txt"),
        (["status", "--short"], "git_status.txt"),
        (["diff", "--binary"], "git_diff.patch"),
    ]:
        (out / name).write_text(
            subprocess.check_output(["git", "-C", str(args.repo), *cmd], text=True)
        )
    names = subprocess.check_output(
        ["git", "-C", str(args.repo), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    sources = {}
    for name in names:
        path = args.repo / name
        if (
            path.is_file()
            and path.suffix in {".py", ".sh", ".yaml", ".toml", ".json"}
            and not any(x in path.parts for x in ["logs", "outputs"])
        ):
            sources[name] = sha256(path)
    (out / "source_sha256.json").write_text(json.dumps(sources, indent=2, sort_keys=True))
    # Full checkpoint hashes avoid treating a path/size as a weight identity.
    (out / "asset_sha256.json").write_text(
        json.dumps(
            {str(f): sha256(f) for f in [*files, init, *sorted(cache.glob("*.pt"))]}, indent=2
        )
    )
    (out / "dataset_meta_sha256.json").write_text(
        json.dumps(
            {
                str(f): sha256(f)
                for r in roots
                for f in sorted((r / "meta").rglob("*"))
                if f.is_file()
            },
            indent=2,
        )
    )
    ds_config = json.loads(
        (
            args.repo
            / (
                "configs/libero/deepspeed_zero2.json"
                if args.side == "base"
                else "examples/embodied/vpp2/zero2_bf16.json"
            )
        ).read_text()
    )
    (out / "zero2.json").write_text(json.dumps(ds_config, indent=2))
    if cfg is not None:
        OmegaConf.save(cfg, out / "config.yaml", resolve=True)
        accel = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "DEEPSPEED",
            "deepspeed_config": {
                "deepspeed_config_file": str(out / "zero2.json"),
                "zero3_init_flag": False,
            },
            "mixed_precision": None,
            "num_machines": 1,
            "num_processes": world,
            "machine_rank": 0,
            "rdzv_backend": "static",
            "same_network": True,
            "use_cpu": False,
        }
        OmegaConf.save(OmegaConf.create(accel), out / "accelerate.yaml")


if __name__ == "__main__":
    main()
