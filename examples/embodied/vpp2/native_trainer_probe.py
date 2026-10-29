#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Real trainer lifecycle + real LIBERO pipeline with a TINY VPP2 model.

Explicit backend regression, NOT full14B or canonical model precision evidence.
Both sides independently load/transform real data. Only model architecture,
mock VAE/CLIP and context dimension are reduced. Training loops are inherited.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from loongforge.embodied.model.vpp2.alignment import parameter_fingerprint, seed_training
from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig
from loongforge.embodied.data.datasets.vpp2.transforms.data_configuration_vpp2 import VPP2DataConfig
from loongforge.embodied.train.training_args import TrainingArgs
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def digest_tensor(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def fingerprint(sample):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": digest_tensor(v)}
            for k, v in sample.items() if torch.is_tensor(v) and k != "sample_index"}


class TinyVAE(torch.nn.Module):
    temporal_downsample_factor = 4

    def encode(self, videos, **kwargs):
        return [torch.nn.functional.adaptive_avg_pool3d(
            v.mean(0, keepdim=True), (1, 4, 4)).repeat(16, 1, 1, 1) for v in videos]


class TinyCLIP(torch.nn.Module):
    def forward(self, videos):
        return torch.zeros(len(videos), 257, 1280, device=videos[0].device, dtype=videos[0].dtype)


def make_model(side, cfg, out):
    prefix = "vpp2.models.wan21_14b." if side == "base" else "loongforge.embodied.model.vpp2.modules."
    video_cls = importlib.import_module(prefix + "wan_video_dit").WanVideoDiT
    action_cls = importlib.import_module(prefix + "action_dit").ActionDiT
    mot_cls = importlib.import_module(prefix + "mot").MoT
    model_cls = (
        importlib.import_module("vpp2.libero.action_model").LiberoActionModel if side == "base"
        else importlib.import_module("loongforge.embodied.model.vpp2.modeling_vpp2").VPP2Model
    )

    class ObservedModel(model_cls):
        def training_loss(self, sample, **kwargs):
            # Identical dimensional reduction, diagnostic-only; raw data hashes
            # are retained before this adaptation.
            with (out / f"inputs_rank{os.environ.get('RANK', '0')}.jsonl").open("a") as f:
                f.write(json.dumps(fingerprint(sample)) + "\n")
            sample = dict(sample, context=sample["context"][..., :16])
            return super().training_loss(sample, **kwargs)

    torch.manual_seed(20261027)
    video = video_cls(dim=24, hidden_dim=24, ffn_dim=48, freq_dim=8, text_dim=16,
                      num_heads=2, attn_head_dim=12, num_layers=2, use_gradient_checkpointing=False)
    action = action_cls(hidden_dim=16, action_dim=7, ffn_dim=32, text_dim=16, freq_dim=8,
                        eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=2)
    mot = mot_cls({"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    model = ObservedModel(video_expert=video, action_expert=action, mot=mot, vae=TinyVAE(),
                          clip_encoder=TinyCLIP(), text_dim=16, proprio_dim=8, device="cpu",
                          torch_dtype=torch.bfloat16, video_train_sampling_strategy="pure_noise",
                          action_visible_video_frames=3, action_only_first_frame_fast_path=True)
    model.config = cfg
    model.to(device="cuda", dtype=torch.bfloat16)
    model.device = torch.device("cuda", torch.cuda.current_device())
    model.eval().requires_grad_(False)
    model.action_expert.train().requires_grad_(True)
    model.proprio_encoder.train().requires_grad_(True)
    if side == "loongforge":
        model.freeze_modules()
    return model


def base_config(assets, out, batch, steps):
    cfg = OmegaConf.load(Path(os.environ["VPP2_BASE_ROOT"]) / "configs/libero/train_action.yaml")
    cfg.output_dir, cfg.batch_size, cfg.num_workers, cfg.max_steps = str(out), batch, 0, steps
    cfg.num_epochs, cfg.log_every, cfg.save_every, cfg.eval_every = 100000, 1, 0, 0
    cfg.save_final_checkpoint = False
    cfg.wandb.enabled = False
    cfg.data.train.dataset_dirs = [str(assets / f"libero_subset/libero_{s}_no_noops_lerobot")
                                   for s in ("spatial", "object", "goal", "10")]
    cfg.data.train.pretrained_norm_stats = str(assets / "libero/dataset_stats.json")
    cfg.data.train.text_embedding_cache_dir = str(assets / "text_cache_real")
    return cfg


def save_final(model, out):
    # FSDP2 full_tensor is collective: call this observer on EVERY rank.
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            value = param.detach()
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            state[name] = value.cpu()
    if int(os.environ.get("RANK", 0)) == 0:
        torch.save(state, out / "final.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("base", "loongforge"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--strategy", choices=("deepspeed", "ddp", "fsdp"), default="deepspeed")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    a = parser.parse_args()
    out = a.output.resolve()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if rank == 0:
        out.mkdir(parents=True, exist_ok=False)
        if a.resume_from:
            target = out / "checkpoints" / a.resume_from.name
            shutil.copytree(a.resume_from, target)
        (out / ".prepared").touch()
    # Avoid initialize/destroy/reinitialize of a process group with the same
    # rendezvous store: the inherited trainer alone owns distributed setup.
    deadline = time.monotonic() + 60
    while not (out / ".prepared").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Rank zero failed to prepare the new run directory")
        time.sleep(0.05)
    import torch.distributed as dist
    assets = Path(os.environ["VPP2_ASSETS_ROOT"])
    # FSDP is a lifecycle smoke test, not a BASE numerical comparison. The
    # opt-in comparison fingerprint expects replicated params, not DTensors.
    model_cfg = VPP2ModelConfig(alignment_seed=None if a.strategy == "fsdp" else 42,
                                disable_train_autocast=True)
    cfg = base_config(assets, out, a.batch, a.steps)
    if rank == 0:
        write(out / "scope.json", {
            "side": a.side, "strategy": a.strategy, "world": int(os.environ["WORLD_SIZE"]),
            "batch": a.batch, "steps": a.steps, "model": "tiny_2layer_MoT", "real_dataset": True,
            "mock_vae_clip": True, "context_dimension": 16, "canonical_zero2_pass": False,
            "inherited_training_loop": True, "internal_chunks": False,
        })
    if a.side == "base":
        from vpp2.libero.trainer import LiberoTrainer

        class NativeBaseTrainer(LiberoTrainer):
            def _build_scheduler(self, scheduler_type, total_train_steps, warmup_steps=0):
                return super()._build_scheduler(scheduler_type, 30000, warmup_steps)

            def _build_loader(self, dataset, worker_init_fn=None):
                loader = super()._build_loader(dataset, worker_init_fn)
                loader.generator = torch.Generator().manual_seed(42)
                return loader

            def _wandb_log(self, payload):
                if self.accelerator.is_main_process and "train/loss" in payload:
                    row = dict(iteration=self.global_step, action_loss=payload["train/loss_action"],
                               video_loss=payload["train/loss_video"], learning_rate=payload["train/lr"],
                               grad_norm=float(self.model.get_global_grad_norm()))
                    with (out / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps({"train": row}) + "\n")
                return super()._wandb_log(payload)

        model = make_model(a.side, model_cfg, out)
        dataset = instantiate(cfg.data.train)
        trainer = NativeBaseTrainer(model, dataset, cfg=cfg)
        if rank == 0:
            write(out / "initial_trainable.json", parameter_fingerprint(model))
            write(out / "effective_deepspeed.json", trainer.model._config._param_dict)
        seed_training(42, rank)
        trainer.train()  # inherited official loop, no manual backward/step here
        if rank == 0:
            save_final(model, out)
        trainer.accelerator.wait_for_everyone()
        engine = trainer.model
        engine.destroy()
        engine.optimizer = None
        trainer.accelerator.end_training()
        if dist.is_initialized():
            dist.destroy_process_group()
    else:
        data_cfg = VPP2DataConfig(norm_stats_path=cfg.data.train.pretrained_norm_stats,
                                  text_embedding_cache_dir=cfg.data.train.text_embedding_cache_dir)
        args = replace(TrainingArgs(), model_name="vpp2", dataset_format="lerobot_datasets",
                       dataset_strategy="vpp2", dataset_path=",".join(cfg.data.train.dataset_dirs),
                       video_backend="pyav", num_workers=0, per_device_batch_size=a.batch,
                       train_iters=a.steps, distributed_strategy=a.strategy,
                       deepspeed_config=os.environ.get("DEEPSPEED_CONFIG"), init_on_meta=False,
                       dtype="bfloat16", optimizer="AdamW", lr_base=1e-4, weight_decay=.01,
                       adam_beta1=.9, adam_beta2=.95, lr_warmup_iters=500, lr_decay_iters=30000,
                       lr_decay_style="linear_warmup_cosine_annealing", min_lr=1e-6,
                       clip_grad=0.0, check_for_nan_in_loss_and_grad=False, loss_spike_threshold=float("inf"),
                       seed=42, set_seed_by_rank=True, deterministic_mode=True, output_dir=str(out),
                       save_interval=a.save_interval, log_interval=1, wandb_mode="disabled",
                       batch_drop_last=False, resume=bool(a.resume_from))

        class NativeTransTrainer(FinetuneTrainer):
            def _build_model(self):
                return make_model(a.side, self.model_cfg, out)

            def _on_step_end(self, metrics):
                if self.completed_steps == self.train_iters:
                    save_final(getattr(self.model, "module", self.model), out)

        trainer = NativeTransTrainer(args, model_cfg, data_cfg)
        trainer.train()  # inherited full setup/GA/backward/step/log/save/teardown


if __name__ == "__main__":
    main()
