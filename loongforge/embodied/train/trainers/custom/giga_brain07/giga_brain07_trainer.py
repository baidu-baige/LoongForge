# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""GigaBrain07Trainer — 方案B: giga-brain-0.7 on the LoongForge BaseTrainer loop.

Runs the LoongForge BaseTrainer/FinetuneTrainer step skeleton + xvla-style
logger, while **model / data / transform / loss / scheduler stay the giga
originals** so loss aligns with the giga-brain-0 baseline. Only the compute
hooks are overridden; everything else (grad-accum loop, logging, clip, nan
guard, checkpoint) is inherited.

Alignment points vs giga_brain_0/configs/giga_brain_0_07_a2d_finetune.py:
  (1) data  : giga LeRobotDataset + GigaBrain07Transform (NOT LeRobotV3Dataset)
  (2) sched : giga WarmupCosineScheduler (warmup=10 decay=300 end=0.1)
  (3) step  : grad_accum=1 -> per-step == per-opt-step; opt.step then sched.step
  (4) optim : AdamW betas(0.9,0.95) eps1e-8 wd1e-10; mixed_precision=no (fp32,
              no autocast); seed=6666; activation ckpt on. Single-GPU => FSDP2
              sharding is a no-op, kept as plain fp32 module (same math).
"""

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader

# giga model/data code is imported via fully-qualified LoongForge paths now
# (loongforge.embodied.model.giga_brain07.* / data.datasets.giga_brain07.*),
# so the vendored packages are no longer injected onto sys.path as bare
# top-level names. Only the LoongForge repo root must be importable, which
# the launcher already guarantees. No sys.path / PYTHONPATH mutation here.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ...supervised.finetune_trainer import FinetuneTrainer  # noqa: E402

# ── Fine-grained NVTX / torch.profiler range markers (opt-in, gated) ──
try:
    from loongforge.embodied.profiling_nvtx import nvtx_range, set_enabled
except Exception:  # pragma: no cover - profiling is optional
    import contextlib as _contextlib

    def set_enabled(_flag):  # type: ignore
        return None

    @_contextlib.contextmanager
    def nvtx_range(_msg):  # type: ignore
        yield


class GigaBrain07Trainer(FinetuneTrainer):
    """BaseTrainer-paradigm trainer that reuses the giga-brain-0.7 originals."""

    # ───────────────────────── model + loss (giga get_models) ─────────────────
    def _build_model(self) -> nn.Module:
        # Route-C: build via the shared model registry (build_model ->
        # @register_model("giga_brain07") -> build_giga_brain07_policy), the same
        # path XVLAPolicy is reached. TF32, pretrained load, loss-module attach
        # and num_embodiments now live in the registered factory / on the policy.
        from loongforge.embodied.model.registry import build_model

        model = build_model(self.model_cfg)
        model.to(self.ctx.device)
        model.train()
        # Honor model.enable_nvtx_profiling from the slim config at build time.
        set_enabled(bool(getattr(self.model_cfg, "enable_nvtx_profiling", False)))
        return model

    def _wrap_model_for_training(self):
        m = self.model_cfg
        if self.ctx.world_size > 1:
            super()._wrap_model_for_training()
            self._maybe_apply_kv_slice()  # [compile] no-op unless model.decoder_kv_slice
            self._maybe_apply_compile()  # [compile] no-op unless model.compile_model
            return
        if self.training_args.activation_checkpoint_module_patterns:
            # xvla-style: reuse the shared apply_activation_checkpointing (same
            # function wrap_model() uses in the multi-GPU path), selecting modules
            # by key-glob patterns from --activation-checkpoint-module-patterns
            # (giga passes paligemma_with_expert.layers.0..N). Loss-transparent.
            from loongforge.embodied.distributed.activation_checkpointing import (
                apply_activation_checkpointing,
            )

            apply_activation_checkpointing(
                self.model,
                self.training_args.activation_checkpoint_module_patterns,
                self.training_args.activation_checkpoint_skip_modules,
            )
            if self.ctx.is_main:
                n_ckpt = sum(
                    1 for mod in self.model.modules()
                    if mod.__class__.__name__ == "CheckpointWrapper"
                )
                import logging as _lg
                _lg.getLogger(__name__).info(
                    "GigaBrain07: activation checkpointing wrapped %d modules", n_ckpt
                )
        self.model = self.model.to(self.ctx.device)
        self._maybe_apply_kv_slice()  # [compile] no-op unless model.decoder_kv_slice
        self._maybe_apply_compile()  # [compile] no-op unless model.compile_model

    # ───────────────────────── [compile] opt-in per-layer KV slice ─────────────────
    def _maybe_apply_kv_slice(self) -> None:
        """Hand each decoder layer only its own past_key_values entry.

        Registered as a forward-pre-hook with prepend=True AFTER FSDP2 wrapping,
        so it runs before FSDP2's ``_pre_forward`` input cast. The layer only
        ever reads ``past_key_values[self.layer_idx]``, so the sliced dict is
        semantically identical; it just stops FSDP from casting (and AC from
        retaining) fp32 copies of all 26 layers' KV at every layer.
        """
        if not bool(getattr(self.model_cfg, "decoder_kv_slice", False)):
            return
        import logging as _lg
        import re as _re
        from loongforge.embodied.model.giga_brain07.paligemma2_with_expert import (
            Gemma2DecoderLayerWithExpert,
        )

        pat = _re.compile(r"(^|\.)paligemma_with_expert\.layers\.(\d+)$")
        n = 0
        for name, mod in self.model.named_modules():
            if not pat.search(name):
                continue
            inner = mod
            while not isinstance(inner, Gemma2DecoderLayerWithExpert):
                inner = getattr(inner, "_checkpoint_wrapped_module", None)
                if inner is None:
                    break
            if inner is None:
                continue
            idx = int(inner.self_attn.layer_idx)

            def _hook(module, args, kwargs, idx=idx):
                pkv = kwargs.get("past_key_values")
                if isinstance(pkv, dict) and idx in pkv and len(pkv) > 1:
                    kwargs = dict(kwargs)
                    kwargs["past_key_values"] = {idx: pkv[idx]}
                    return args, kwargs
                return None

            mod.register_forward_pre_hook(_hook, prepend=True, with_kwargs=True)
            n += 1
        if self.ctx.is_main:
            _lg.getLogger(__name__).info("GigaBrain07 decoder_kv_slice ENABLED on %d layers", n)
        if n == 0:
            raise RuntimeError("model.decoder_kv_slice=true but no decoder layer matched")

    # ───────────────────────── [compile] opt-in regional torch.compile ──────────────
    def _maybe_apply_compile(self) -> None:
        """Regionally torch.compile inner compute modules (default OFF).

        Runs after AC + FSDP2 wrapping. Only the *inner* module's instance
        ``forward`` is replaced by a compiled callable, so the FSDP2 pre/post
        forward hooks and the non-reentrant AC CheckpointWrapper stay eager
        around each compiled region (the AC recompute re-enters the same
        compiled graph). All layers of one class share one code object, so
        Dynamo compiles one graph per region class (params are graph inputs
        with inline_inbuilt_nn_modules) instead of one per layer.
        """
        m = self.model_cfg
        if not bool(getattr(m, "compile_model", False)):
            return
        import logging as _lg
        log = _lg.getLogger(__name__)
        from loongforge.embodied.model.giga_brain07.paligemma2_with_expert import (
            Gemma2AttentionWithExpert,
            Gemma2DecoderLayerWithExpert,
            Gemma2MLP,
            SiglipEncoderLayer,
        )

        scope = {s.strip() for s in str(m.compile_scope).split(",") if s.strip()}
        known = {"decoder", "vision", "decoder_mlp", "decoder_attn"}
        if not scope or scope - known:
            raise ValueError(f"model.compile_scope={m.compile_scope!r}; allowed {sorted(known)}")
        mode = str(m.compile_mode) if m.compile_mode not in (None, "", "default") else None
        kw = dict(
            backend="inductor",
            mode=mode,
            dynamic=getattr(m, "compile_dynamic", None),
            fullgraph=bool(getattr(m, "compile_fullgraph", False)),
        )

        def _in_decoder(name: str) -> bool:
            return "paligemma_with_expert.layers." in name

        def _decoder_kv_shim(layer, compiled):
            # The 2-stage train forward (prefix fill_kv_cache -> suffix reading
            # past_key_values[self.layer_idx]) makes Dynamo guard on the int
            # ``self_attn.layer_idx`` -> one recompile per layer, and after
            # recompile_limit(8) the remaining layers silently run EAGER. For
            # the cache-read call we hand the layer a 1-entry view {0: kv} and
            # present layer_idx=0 for the duration of the call, so all 26
            # layers share ONE graph. Same tensors -> numerics unchanged; the
            # AC recompute re-enters this shim with the same args.
            attn = layer.self_attn if hasattr(layer, "self_attn") else layer
            real_idx = int(attn.layer_idx)
            import inspect
            sig = inspect.signature(layer.forward)

            def fwd(*args, **kwargs):
                ba = sig.bind(*args, **kwargs)
                ba.apply_defaults()
                a = ba.arguments
                pkv = a.get("past_key_values")
                if (a.get("use_cache") and not a.get("fill_kv_cache")
                        and isinstance(pkv, dict) and real_idx in pkv):
                    a["past_key_values"] = {0: pkv[real_idx]}
                    attn.layer_idx = 0
                    try:
                        return compiled(*ba.args, **ba.kwargs)
                    finally:
                        attn.layer_idx = real_idx
                return compiled(*ba.args, **ba.kwargs)

            return fwd

        counts: Dict[str, int] = {}
        for name, mod in self.model.named_modules():
            tag = None
            if "decoder" in scope and isinstance(mod, Gemma2DecoderLayerWithExpert):
                tag = "decoder"
            elif "vision" in scope and isinstance(mod, SiglipEncoderLayer):
                tag = "vision"
            elif ("decoder_mlp" in scope and "decoder" not in scope
                  and isinstance(mod, Gemma2MLP) and _in_decoder(name)):
                tag = "decoder_mlp"
            elif ("decoder_attn" in scope and "decoder" not in scope
                  and isinstance(mod, Gemma2AttentionWithExpert) and _in_decoder(name)):
                tag = "decoder_attn"
            if tag is None or getattr(mod, "_gb0_compiled", False):
                continue
            compiled = torch.compile(mod.forward, **kw)
            if tag in ("decoder", "decoder_attn"):
                compiled = _decoder_kv_shim(mod, compiled)
            mod.forward = compiled
            mod._gb0_compiled = True
            counts[tag] = counts.get(tag, 0) + 1
        if self.ctx.is_main:
            log.info(
                "GigaBrain07 torch.compile ENABLED: scope=%s compiled=%s kwargs=%s",
                sorted(scope), counts, kw,
            )
        if not counts:
            raise RuntimeError("model.compile_model=true but no module matched compile_scope")
        if self.ctx.is_main:
            import atexit

            def _report_dynamo():
                try:
                    from torch._dynamo.utils import counters
                    gb = dict(counters.get("graph_break", {}))
                    st = dict(counters.get("stats", {}))
                    fr = dict(counters.get("frames", {}))
                    print("[gb0-compile-health] frames=%s stats=%s graph_breaks=%d %s"
                          % (fr, st, sum(gb.values()), list(gb.items())[:10]), flush=True)
                except Exception as e:  # pragma: no cover
                    print("[gb0-compile-health] unavailable: %r" % (e,), flush=True)

            atexit.register(_report_dynamo)

    # ───────────────────────── [compile] opt-in per-step peak-memory log ─────────────
    def _on_step_end(self, metrics):
        super()._on_step_end(metrics)
        if os.environ.get("GB0_MEM_LOG", "0") == "1" and torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 2**30
            resv = torch.cuda.max_memory_reserved() / 2**30
            if self.ctx.is_main:
                print("[gb0-mem] peak_allocated_GB=%.2f peak_reserved_GB=%.2f" % (alloc, resv),
                      flush=True)
            torch.cuda.reset_peak_memory_stats()

    # ───────────────────────── optimizer (giga AdamW, single group) ───────────
    def _build_optimizer(self) -> torch.optim.Optimizer:
        m = self.model_cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Match native giga (giga-brain-0): giga_train.optimizers.build_optimizer
        # builds AdamW from dict(type="AdamW", betas, lr, eps, weight_decay) with
        # no `foreach` key, so torch uses its default (fused multi-tensor path on
        # CUDA). Omit `foreach` here too so the optimizer matches native exactly.
        opt = torch.optim.AdamW(
            params,
            lr=m.lr,
            betas=m.adam_betas(),
            eps=m.adam_eps,
            weight_decay=m.weight_decay,
        )
        # Tag the optimizer step for nsys / torch.profiler (gated no-op when the
        # NVTX switch is off; purely an annotation, no numerics change). Bind as
        # a real method (types.MethodType) so torch's LRScheduler step-tracking
        # patch, which reads ``optimizer.step.__func__``, keeps working.
        import types as _types
        _orig_step = opt.step

        def _nvtx_step(_self, *args, **kwargs):
            with nvtx_range('optimizer_step'):
                return _orig_step(*args, **kwargs)

        opt.step = _types.MethodType(_nvtx_step, opt)
        return opt

    # ───────────────────────── backward (gated NVTX region) ──────────────────
    def _backward_loss(self, loss, log_loss_dict, log_dict, grad_accum) -> None:
        with nvtx_range('backward'):
            super()._backward_loss(loss, log_loss_dict, log_dict, grad_accum)

    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        d = self.data_cfg
        # Data infra (reader + collator + sampler + transform) is self-contained
        # under the giga_brain07 data package; no top-level vendored
        # ``giga_datasets`` / ``giga_train`` imports are used here.
        from loongforge.embodied.data.datasets.transforms.collator import (
            build_preprocessor,
        )
        from loongforge.embodied.data.datasets.giga_brain07.sampler import (
            GigaBrainSampler,
        )
        from loongforge.embodied.data.datasets.transforms.pipeline import (
            build_transforms_from_args,
        )
        # 方案乙 (route B): dataset *reading* now goes through the shared
        # LeRobot v3 reader (loongforge.embodied.data.datasets.lerobot_dataset
        # .LeRobotV3Dataset) via the giga_brain07 strategy, instead of the
        # vendored giga_datasets.LeRobotDataset. Divergent giga params
        # (action chunk -> action_horizon, video backend, per-sample meta
        # attach) are passed as parameters. Per-sample output parity with the
        # giga reader was verified on AgiBot task_327 (see
        # giga_analysis/probe_parity.py). Genuinely giga-specific hardening
        # (bounded-memory decode, HF-cache locking, thin metadata, repack) is
        # intentionally dropped here and can be re-added later as separately
        # gated parameters on the shared reader.
        from loongforge.embodied.data.datasets.giga_brain07 import (
            build_giga_brain07_dataset,
        )

        dataset = build_giga_brain07_dataset(self.model_cfg, self.data_cfg, self.training_args)
        # xvla-style: build the per-sample transform through the shared pipeline
        # (build_transforms_from_args -> @register_transform_builder("giga_brain07")),
        # then attach it exactly as dataloader._apply_transform does (poke
        # ``_transform``). The transform_cfg assembly now lives in the registered
        # builder in giga_brain07/transforms/giga_brain07_transform.py.
        dataset_stats = getattr(getattr(dataset, "meta", None), "stats", {})
        transform = build_transforms_from_args(
            self.model_cfg, self.data_cfg, self.training_args, dataset, dataset_stats,
        )
        if transform is not None:
            dataset._transform = transform

        bs = int(d.batch_size_per_gpu)
        grad_accum = int(self.training_args.gradient_accumulation_steps)
        global_bs = bs * int(self.ctx.world_size) * grad_accum
        sampler = GigaBrainSampler(
            dataset,
            batch_size=global_bs,
            shuffle=bool(d.shuffle),
            seed=int(d.sampler_seed),
        )
        batch_sampler = BatchSampler(sampler, batch_size=bs, drop_last=False)

        num_workers = int(d.num_workers)
        kwargs = dict(
            batch_sampler=batch_sampler,
            collate_fn=build_preprocessor(
                "giga_brain07", self.model_cfg, self.data_cfg, self.training_args,
            ),
            num_workers=num_workers,
            pin_memory=bool(d.pin_memory),
        )
        if num_workers > 0:
            kwargs.update(
                persistent_workers=bool(d.persistent_workers),
                prefetch_factor=int(d.prefetch_factor),
            )
        return {"vla": DataLoader(dataset, **kwargs)}

    # ───────────────────────── batch -> device (dict/list aware) ──────────────
    def _move_batch_to_device(self, batch):
        device = self.ctx.device

        def move(v):
            if torch.is_tensor(v):
                return v.to(device, non_blocking=True)
            if isinstance(v, dict):
                return {k: move(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(move(x) for x in v)
            return v

        return move(batch)

    # ───────────────────────── forward + loss (giga forward_step/parse) ───────
    def _train_forward(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Route-C xvla-style contract: the policy's forward(batch) returns
        # (loss, log_dict). The flow-matching loss assembly moved verbatim into
        # GigaBrain07Policy._train_step; no autocast wrapper (giga runs fp32,
        # mixed_precision="no"), matching the pre-refactor numerics.
        return self.model(batch)
