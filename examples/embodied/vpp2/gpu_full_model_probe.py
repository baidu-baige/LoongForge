#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Full-14B GPU diagnostic: B1, model-parallel frozen video, CPU FP32 AdamW.
NOT canonical ZeRO2. All layers/tokens, real VAE/CLIP/data/UMT5; no autocast/clip.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def digest(items):
    h, rows = hashlib.sha256(), {}
    for name, value in items:
        if value is None:
            rows[name] = None
            h.update((name + ":NONE").encode())
            continue
        value = value.detach().cpu().contiguous()
        th = hashlib.sha256(
            f"{name}:{tuple(value.shape)}:{value.dtype}".encode()
            + value.reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        rows[name] = {"sha256": th, "shape": list(value.shape), "dtype": str(value.dtype)}
        h.update(th.encode())
    return {"sha256": h.hexdigest(), "tensors": rows}


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--side", choices=["base", "loongforge"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.repeats) <= 0:
        raise ValueError("Positive steps/repeats required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "executed_probe.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(8)
    if os.environ.get("VPP2_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assets, base, init = [
        Path(os.environ[k]).resolve()
        for k in ["VPP2_ASSETS_ROOT", "VPP2_BASE_ROOT", "VPP2_ALIGNMENT_INIT"]
    ]
    data = Path(os.environ.get("VPP2_DATASET_ROOT", str(assets / "libero_subset"))).resolve()
    roots = [
        str(data / f"libero_{s}_no_noops_lerobot") for s in ["spatial", "object", "goal", "10"]
    ]
    cfg = OmegaConf.load(base / "configs/libero/train_action.yaml")
    cfg.model.model_id = str(assets / "Wan2.1-I2V-14B-480P")
    cfg.model.dit_checkpoint_path = os.environ.get(
        "VPP2_VIDEO_CHECKPOINT", str(assets / "libero_video_step010000.pt")
    )
    cfg.model.action_dit_pretrained_path = str(assets / "libero/action_init.pt")
    cfg.data.train.dataset_dirs = roots
    cfg.data.train.pretrained_norm_stats = str(assets / "libero/dataset_stats.json")
    cfg.data.train.text_embedding_cache_dir = str(assets / "text_cache_real")
    OmegaConf.save(cfg, out / "base_config.yaml", resolve=True)
    repo = base if args.side == "base" else Path(os.environ["VPP2_LOONG_ROOT"]).resolve()
    write(
        out / "manifest.json",
        {
            "side": args.side,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "steps": args.steps,
            "repeats": args.repeats,
            "seed": args.seed,
            "global_batch": 1,
            "reference_global_batch": 256,
            "engine": "single_process_model_parallel_CPU_FP32_AdamW",
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "autocast": False,
            "grad_clip": None,
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "schedule_steps": 30000,
            "full_plan_acceptance": False,
            "dataset_roots": roots,
            "init": str(init),
            "dependencies": {
                n: importlib.metadata.version(n)
                for n in [
                    "torch",
                    "deepspeed",
                    "accelerate",
                    "diffusers",
                    "transformers",
                    "torchvision",
                    "numpy",
                    "av",
                ]
            },
            "git_head": subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    for cmd, name in [
        (["diff", "--binary"], "git_diff.patch"),
        (["status", "--short"], "git_status.txt"),
    ]:
        (out / name).write_bytes(subprocess.check_output(["git", "-C", str(repo), *cmd]))
    (out / "nvidia_before.txt").write_bytes(subprocess.check_output(["nvidia-smi"]))
    print("Hashing actual input assets", flush=True)
    files = [
        Path(cfg.model.dit_checkpoint_path),
        Path(cfg.model.action_dit_pretrained_path),
        init,
        Path(cfg.data.train.pretrained_norm_stats),
        assets / "Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
        assets / "Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    ]
    files += sorted((assets / "text_cache_real").glob("*.pt"))
    files += [
        f
        for root in map(Path, roots)
        for sub in ["meta", "data", "videos"]
        for f in sorted((root / sub).rglob("*"))
        if f.is_file()
    ]
    write(out / "asset_sha256.json", {str(f): sha(f) for f in files})
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    write(
        out / "source_sha256.json",
        {
            n: sha(repo / n)
            for n in names
            if (repo / n).is_file()
            and Path(n).suffix in {".py", ".sh", ".json", ".yaml", ".toml"}
            and not any(x in Path(n).parts for x in ["logs", "outputs"])
        },
    )
    seed(args.seed)
    print("Preparing real dataset samples independently", flush=True)
    if args.side == "base":
        dataset = instantiate(cfg.data.train)
        get = lambda i: dataset.get_strict_item(i)[1]
    else:
        from loongforge.embodied.data.datasets.vpp2.transforms.data_configuration_vpp2 import (
            VPP2DataConfig,
        )
        from loongforge.embodied.data.datasets.vpp2.transforms.vpp2_transform import (
            VPP2KeyMappingTransform,
            VPP2LinearNormalizeTransform,
        )
        from loongforge.embodied.data.datasets.vpp2.vpp2_dataset import build_vpp2_lerobot_dataset
        from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig

        dc = VPP2DataConfig(
            norm_stats_path=cfg.data.train.pretrained_norm_stats,
            text_embedding_cache_dir=cfg.data.train.text_embedding_cache_dir,
        )
        dataset = build_vpp2_lerobot_dataset(
            VPP2ModelConfig(),
            dc,
            SimpleNamespace(dataset_path=",".join(roots), video_backend="pyav"),
        )
        norm = VPP2LinearNormalizeTransform(dc.norm_stats_path, dc.delta_action_dim_mask)
        mapper = VPP2KeyMappingTransform(text_embedding_cache_dir=dc.text_embedding_cache_dir)
        get = lambda i: mapper.apply(norm.apply(dataset[i]))
    g = torch.Generator().manual_seed(args.seed)
    indices = []
    while len(indices) < args.steps:
        indices.extend(torch.randperm(len(dataset), generator=g).tolist())
    indices = indices[: args.steps]
    batches = [default_collate([get(i)]) for i in indices]
    write(
        out / "input_fingerprints.json",
        [
            {
                "index": i,
                "tensors": digest((k, v) for k, v in sorted(b.items()) if torch.is_tensor(v)),
                "non_tensors": {k: v for k, v in b.items() if not torch.is_tensor(v)},
            }
            for i, b in zip(indices, batches)
        ],
    )
    print("Loading full released model on CPU", flush=True)
    if args.side == "base":
        from alignment import load_alignment_init

        from vpp2.libero.runtime import create_action_model

        kwargs = OmegaConf.to_container(cfg.model, resolve=True)
        kwargs.pop("_target_")
        model = create_action_model(**kwargs, device="cpu", model_dtype=torch.bfloat16)
        load_alignment_init(model, init)
    else:
        from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig
        from loongforge.embodied.model.vpp2.modeling_vpp2 import VPP2Model

        mc = VPP2ModelConfig(
            model_id=cfg.model.model_id,
            video_checkpoint_path=cfg.model.dit_checkpoint_path,
            action_dit_pretrained_path=cfg.model.action_dit_pretrained_path,
            alignment_init_path=str(init),
            init_on_meta=True,
            disable_train_autocast=True,
        )
        model = VPP2Model.from_pretrained(mc)
        model.materialize("cpu", "bfloat16")
        write(
            out / "strict_load.json", model.load_pretrained(mc.video_checkpoint_path, device="cpu")
        )
    model.eval().requires_grad_(False)
    model.set_train_mode("action_only")
    model.action_expert.train().requires_grad_(True)
    model.proprio_encoder.train().requires_grad_(True)
    if os.environ.get("VPP2_GPU_WAIT_FILE"):
        print("CPU load complete; waiting before GPU placement", flush=True)
        while not Path(os.environ["VPP2_GPU_WAIT_FILE"]).is_file():
            time.sleep(5)
    from vpp2.libero.model_parallel import place_action_model

    write(
        out / "placement.json",
        place_action_model(
            model,
            action_device="cuda:0",
            video_devices=("cuda:1", "cuda:2", "cuda:3", "cuda:4"),
            encoder_device="cuda:5",
        ),
    )
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    print("Hashing ALL actual model parameters, not sampled elements", flush=True)
    write(out / "initial_all_parameters.json", digest(model.named_parameters()))
    initial = [p.detach().float().cpu().clone() for p in parameters]
    # Reuse the complete initial hash rather than transfer frozen14B twice.
    initial_report = json.loads((out / "initial_all_parameters.json").read_text())
    frozen_names = {n for n, p in model.named_parameters() if not p.requires_grad}
    frozen_rows = {n: v for n, v in initial_report["tensors"].items() if n in frozen_names}
    frozen_before = {
        "sha256": hashlib.sha256(
            "".join(v["sha256"] for v in frozen_rows.values()).encode()
        ).hexdigest(),
        "tensors": frozen_rows,
    }
    write(out / "frozen_before.json", frozen_before)
    write(out / "module_modes.json", {n: m.training for n, m in model.named_modules()})
    for rep in range(args.repeats):
        run = out / f"run{rep + 1}"
        run.mkdir()
        with torch.no_grad():
            for param, value in zip(parameters, initial):
                param.copy_(value)
        masters = [torch.nn.Parameter(value.clone()) for value in initial]
        optimizer = torch.optim.AdamW(
            masters, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-8, foreach=False
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        main = CosineAnnealingLR(optimizer, T_max=29500, eta_min=1e-6)
        warm = LinearLR(optimizer, start_factor=1 / 500, end_factor=1.0, total_iters=500)
        scheduler = SequentialLR(optimizer, [warm, main], milestones=[500])
        seed(args.seed)
        start = time.monotonic()
        for step, batch in enumerate(batches, 1):
            tick = time.monotonic()
            lr_used = optimizer.param_groups[0]["lr"]
            loss, metrics = model.training_loss(batch, training_stage="action")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss {rep=} {step=}")
            loss.backward()
            if any(p.grad is not None for p in model.video_expert.parameters()):
                raise RuntimeError("Frozen video got gradients")
            grad_items, sq = [], 0.0
            for (name, param), master in zip(named, masters):
                master.grad = (
                    None
                    if param.grad is None
                    else param.grad.detach().to(device="cpu", dtype=torch.float32)
                )
                grad_items.append((name, param.grad))
                if master.grad is not None:
                    if not torch.isfinite(master.grad).all():
                        raise FloatingPointError(f"nonfinite gradient {name}")
                    sq += master.grad.double().square().sum().item()
            grad_hash = None
            if step in {1, args.steps}:
                report = digest(grad_items)
                write(run / f"grad_step{step:06d}.json", report)
                grad_hash = report["sha256"]
                if args.snapshot and rep == 0 and step == 1:
                    torch.save(
                        {n: p.grad.detach().cpu() for n, p in named if p.grad is not None},
                        run / "first_grad.pt",
                    )
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for param, master in zip(parameters, masters):
                    param.copy_(master)
                    param.grad = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            row = {
                "step": step,
                "loss": loss.item(),
                "action_loss": float(metrics["loss_action"]),
                "video_loss": float(metrics["loss_video"]),
                "grad_norm": sq**0.5,
                "grad_hash": grad_hash,
                "lr_used": lr_used,
                "lr": optimizer.param_groups[0]["lr"],
                "sample_index": indices[step - 1],
                "elapsed": time.monotonic() - start,
                "step_seconds": time.monotonic() - tick,
                "action_timestep": metrics.get("action_timestep"),
                "action_weight": metrics.get("action_weight"),
            }
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(dict(side=args.side, repeat=rep + 1, **row)), flush=True)
        write(run / "final_parameters.json", digest(named))
        delta = [
            float((p.detach().float().cpu() - value).abs().max())
            for p, value in zip(parameters, initial)
        ]
        write(
            run / "parameter_delta.json",
            {
                "by_name": dict(zip([n for n, p in named], delta)),
                "changed": sum(d > 0 for d in delta),
                "total": len(delta),
            },
        )
        if args.snapshot and rep == 0:
            torch.save({n: p.detach().cpu() for n, p in named}, run / "final_parameters.pt")
        del optimizer, scheduler, masters, grad_items
    frozen_after = digest((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    write(out / "frozen_after.json", frozen_after)
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen full-parameter fingerprint changed")
    write(
        out / "verification.json",
        {
            "frozen_unchanged": True,
            "full_frozen_hash": True,
            "trainable_tensors": len(named),
            "trainable_numel": sum(p.numel() for p in parameters),
            "peak_allocated": {
                str(i): torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
            },
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "canonical_zero2_pass": False,
            "full_plan_acceptance": False,
        },
    )
    print("COMPLETED REAL FULL-MODEL GPU DIAGNOSTIC", flush=True)


if __name__ == "__main__":
    main()
