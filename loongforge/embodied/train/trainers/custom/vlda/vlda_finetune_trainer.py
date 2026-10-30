# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""VLDA adds post-optimizer teacher EMA and checkpoint data assets."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist

from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer
from loongforge.embodied.train.utils.utils import set_seed


class VLDAFinetuneTrainer(FinetuneTrainer):
    """Keep the common training loop; update the teacher once per optimizer step."""

    def _build_model(self):
        args = self.training_args
        if args.distributed_strategy != "ddp" or args.init_on_meta:
            raise ValueError("VLDA currently supports DDP and DDP + ZeRO1; FSDP teacher EMA is not implemented")
        if args.use_lora or args.save_format not in ("safetensors", "pt"):
            raise ValueError("VLDA uses full-model safetensors/pt checkpoints")
        if args.dataset_strategy != "vlda":
            raise ValueError("VLDA requires --dataset-strategy vlda for future frames and negative actions")
        self._resume_vlda_manifest = None
        model = super()._build_model()
        model.model.initialize_training_modules(args.seed)
        return model

    def _configure_backend_precision(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _on_after_data_iterators_initialized(self):
        # Model construction and iterator startup must not shift training RNG.
        if not self.training_args.resume:
            set_seed(self.training_args.seed, self.training_args.set_seed_by_rank)

    def _load_pretrained(self, path):
        self.model.load_pretrained(path, resume=self.training_args.resume)
        self.logger.log_pretrained_loaded(path)

    def _resume_contract(self):
        model = asdict(self.model_cfg)
        model.pop("vjepa_checkpoint")  # Complete checkpoints already contain both encoders.
        data = asdict(self.data_cfg)
        data.pop("norm_stats_path")  # Resume uses the saved snapshot.
        args = self.training_args
        names = (
            "seed",
            "set_seed_by_rank",
            "train_iters",
            "per_device_batch_size",
            "gradient_accumulation_steps",
            "num_workers",
            "dataloader_seed_workers",
            "batch_drop_last",
            "sampler_shuffle",
            "distributed_sampler_mode",
            "lr_base",
            "min_lr",
            "lr_warmup_iters",
            "lr_decay_iters",
            "lr_decay_start_iters",
            "lr_decay_style",
            "custom_lr_lambda",
            "optimizer",
            "weight_decay",
            "weight_decay_grouping",
            "adam_beta1",
            "adam_beta2",
            "adam_eps",
            "clip_grad",
            "zero_optimizer",
            "zero_master_param_dtype",
            "dtype",
        )
        return {
            "randomness_version": 1,
            "model": model,
            "data": data,
            "training": {name: getattr(args, name) for name in names},
            "world_size": self.ctx.world_size,
        }

    def _handle_resume(self, path, step, epoch):
        with (Path(path) / "vlda_config.json").open() as stream:
            manifest = json.load(stream)
        stats_path = Path(path) / "assets/norm_stats.json"
        if hashlib.sha256(stats_path.read_bytes()).hexdigest() != manifest["norm_stats_sha256"]:
            raise ValueError("VLDA checkpoint normalization statistics changed")
        # Older checkpoints always decayed immediately after warmup.
        manifest["resume_contract"]["training"].setdefault("lr_decay_start_iters", 0)
        if manifest["resume_contract"] != self._resume_contract():
            raise ValueError("VLDA resume configuration changed; use --pretrained-checkpoint for a new run")
        self._resume_vlda_manifest = manifest
        self.data_cfg = replace(self.data_cfg, norm_stats_path=str(stats_path))
        self._load_pretrained(path)
        self.completed_steps = step
        self.current_epoch = epoch
        self.logger.log_resume(step)

    def _build_dataloaders(self):
        loaders = super()._build_dataloaders()
        dataset = loaders["vla"].dataset
        if self._resume_vlda_manifest is not None:
            if dataset.fingerprint != self._resume_vlda_manifest["dataset_fingerprint"]:
                raise ValueError("VLDA dataset metadata changed since the saved checkpoint")
        return loaders

    def _on_step_end(self, metrics):
        # BaseTrainer increments completed_steps before this hook. The original
        # VLDA schedule uses the just-finished step's zero-based index.
        fraction = (self.completed_steps - 1) / self.train_iters
        momentum = self.model_cfg.ema_start + fraction * (self.model_cfg.ema_end - self.model_cfg.ema_start)
        unwrap_model(self.model).update_teacher_ema(momentum)
        metrics["teacher_ema"] = momentum

    def _save_checkpoint(self):
        super()._save_checkpoint()
        error = [None]
        if self.ctx.is_main:
            try:
                path = Path(self.checkpoint_dir) / f"steps_{self.completed_steps}"
                dataset = self.dataloaders["vla"].dataset
                assets = path / "assets"
                assets.mkdir(exist_ok=True)
                stats = {
                    key: {name: value.tolist() for name, value in values.items()}
                    for key, values in dataset.meta.stats.items()
                }
                with (assets / "norm_stats.json").open("w") as stream:
                    json.dump({"norm_stats": stats}, stream, indent=2)
                manifest = {
                    "model": asdict(self.model_cfg),
                    "data": asdict(self.data_cfg),
                    "resume_contract": self._resume_contract(),
                    "dataset_fingerprint": dataset.fingerprint,
                    "norm_stats_sha256": hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest(),
                }
                with (path / "vlda_config.json").open("w") as stream:
                    json.dump(manifest, stream, indent=2)
            except Exception as exc:
                error[0] = str(exc)
        if self.ctx.is_distributed:
            dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise RuntimeError(f"Could not save VLDA checkpoint assets: {error[0]}")
