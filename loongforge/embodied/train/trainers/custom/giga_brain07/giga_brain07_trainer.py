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
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader

# ── Make the vendored giga packages importable (mirror train_giga_brain07.py) ──
_EMB = Path(__file__).resolve().parents[4]            # .../loongforge/embodied
_MODEL_GB = _EMB / "model" / "giga_brain07"
_DATA_GB = _EMB / "data" / "datasets" / "giga_brain07"
for _p in (str(_MODEL_GB), str(_DATA_GB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
_pp = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(
    dict.fromkeys([str(_MODEL_GB), str(_DATA_GB), *_pp])
)
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
        m = self.model_cfg
        # giga enables TF32 (accelerate Trainer allow_tf32=True). LoongForge only
        # *disables* TF32 on --disable-tf32, so match giga explicitly here else
        # matmul precision (and hence loss) drifts from the baseline.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import (
            GigaBrain07Policy,
        )
        from giga_brain_0.giga_brain_0_loss import GigaBrain07Loss

        model = GigaBrain07Policy.from_pretrained(m.pretrained)
        loaded_vlm = getattr(model, "vlm_type", m.vlm_type)
        if loaded_vlm != "paligemma2":
            raise ValueError(f"GigaBrain07 only supports PaliGemma2, got {loaded_vlm!r}")
        model.to(self.ctx.device)
        model.train()

        # Honor model.enable_nvtx_profiling from the slim config at build time;
        # env LOONGFORGE_NVTX=1 still force-on wins inside set_enabled(). No-op
        # (zero overhead / zero numerics change) when the switch is off.
        set_enabled(bool(getattr(m, "enable_nvtx_profiling", False)))

        self._giga_loss = GigaBrain07Loss(
            llm_loss_weight=float(m.llm_loss_weight),
            flow_action_dim_loss_weight_cfg=None,
        )
        cfg = getattr(model, "config", {})
        try:
            self._num_embodiments = int(cfg.get("num_embodiments", m.num_embodiments))
        except AttributeError:
            self._num_embodiments = int(getattr(cfg, "num_embodiments", m.num_embodiments))
        return model

    # ───────────────────────── wrap (single-GPU: AC only, no dist) ────────────
    def _wrap_model_for_training(self):
        m = self.model_cfg
        if self.ctx.world_size > 1:
            super()._wrap_model_for_training()
            self._maybe_apply_kv_slice()  # [compile] no-op unless model.decoder_kv_slice
            self._maybe_apply_compile()  # [compile] no-op unless model.compile_model
            return
        if m.activation_checkpointing and m.activation_class_names:
            from giga_train.strategies import module_auto_wrap_policy
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            policy = module_auto_wrap_policy(list(m.activation_class_names))
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                auto_wrap_policy=policy,
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
        from giga_models.models.vla.giga_brain_0.paligemma2_with_expert import (
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
        from giga_models.models.vla.giga_brain_0.paligemma2_with_expert import (
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
        # foreach=False lowers the transient peak when AdamW first materializes
        # exp_avg / exp_avg_sq (loss-neutral: per-parameter elementwise math is
        # identical to the fused/foreach path) so the fp32 3.5B states fit 80GB.
        opt = torch.optim.AdamW(
            params,
            lr=m.lr,
            betas=m.adam_betas(),
            eps=m.adam_eps,
            weight_decay=m.weight_decay,
            foreach=False,
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

    # ───────────────────────── scheduler (giga WarmupCosine via LambdaLR) ──────
    def _build_scheduler(self):
        m = self.model_cfg
        from giga_train.schedulers import build_scheduler as giga_build_scheduler

        n = int(self.train_iters)
        return giga_build_scheduler(
            dict(
                type="WarmupCosineScheduler",
                warmup_steps=int(m.warmup_steps),
                decay_steps=int(m.decay_steps),
                end_value=float(m.scheduler_end_value),
            ),
            optimizer=self.optimizer,
            epoch_size=n,
            max_epochs=1,
            max_steps=n,
        )

    # ───────────────────────── dataloader (giga LeRobot + transform) ──────────
    def _build_dataloaders(self) -> Dict[str, DataLoader]:
        d = self.data_cfg
        from giga_datasets import DefaultCollator, load_dataset
        from giga_train import build_sampler, build_transform

        data_or_config = [
            dict(
                _class_name="LeRobotDataset",
                data_path=d.data_path,
                delta_info={"action": int(d.action_chunk)},
                meta_name=d.meta_name,
                video_backend=d.video_backend,
                torchcodec_decoder_cache_size=int(d.torchcodec_decoder_cache_size),
                decode_video_keys=list(d.camera_keys),
            )
        ]
        transform_cfg = dict(
            type="GigaBrain07Transform",
            is_train=bool(d.is_train),
            state_input_mode=d.state_input_mode,
            observation_memory_size=int(d.observation_memory_size),
            agent_pos_config={"state": int(d.agent_pos_state)},
            use_quaternion_to_6d=bool(d.use_quaternion_to_6d),
            robot_type_embodiment_id_overrides={d.robot_type: d.embodiment_override},
            delta_action_cfg=dict(
                selector="embodiment_id",
                use_delta_joint_actions=bool(d.use_delta_joint_actions),
                mask_unsupervised_action_dims_for_noise=bool(
                    d.mask_unsupervised_action_dims_for_noise
                ),
                mask={d.embodiment_id: list(d.delta_mask)},
            ),
            norm_cfg=dict(
                norm_stats_path={d.embodiment_id: d.norm_stats_path},
                use_quantiles=bool(d.use_quantiles),
            ),
            image_cfg=dict(
                resize_imgs_with_padding=list(d.resize_hw),
                enable_image_aug=bool(d.enable_image_aug),
                present_img_keys=list(d.camera_keys),
                enable_depth_img=bool(d.enable_depth_img),
                vlm_type=d.vlm_type,
            ),
            prompt_cfg=dict(
                tokenizer_model_path=d.tokenizer_model_path,
                fast_tokenizer_path=d.fast_tokenizer_path,
                enable_control_mode_token=bool(d.enable_control_mode_token),
                enable_end_effector_token=bool(d.enable_end_effector_token),
                max_length=int(d.max_length),
                state_input_mode=d.state_input_mode,
                discrete_state_input=bool(d.discrete_state_input),
                discrete_state_input_for_pose_embodiments=bool(
                    d.discrete_state_input_for_pose_embodiments
                ),
                encode_action_input=bool(d.encode_action_input),
                encode_sub_task_input=bool(d.encode_sub_task_input),
                encoded_action_horizon=int(d.encoded_action_horizon),
                sample_ratios=dict(d.sample_ratios),
                vlm_type=d.vlm_type,
            ),
        )

        dataset = load_dataset(data_or_config)
        transform = build_transform(transform_cfg)
        dataset.set_transform(transform)

        bs = int(d.batch_size_per_gpu)
        grad_accum = int(self.training_args.gradient_accumulation_steps)
        global_bs = bs * int(self.ctx.world_size) * grad_accum
        sampler = build_sampler(
            dict(type="DefaultSampler", shuffle=bool(d.shuffle), seed=int(d.sampler_seed)),
            dataset=dataset,
            batch_size=global_bs,
        )
        batch_sampler = BatchSampler(sampler, batch_size=bs, drop_last=False)

        num_workers = int(d.num_workers)
        kwargs = dict(
            batch_sampler=batch_sampler,
            collate_fn=DefaultCollator(),
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
        # [compile] opt-in one-shot saved-activation census (GB0_MEM_DIAG=1);
        # default path is a plain delegation (no behavior change).
        if os.environ.get("GB0_MEM_DIAG", "0") != "1":
            return self._train_forward_impl(batch)
        self._gb0_diag_calls = getattr(self, "_gb0_diag_calls", 0) + 1
        if self._gb0_diag_calls != 2 or not self.ctx.is_main:
            return self._train_forward_impl(batch)
        import collections
        stack, hooks, seen = [], [], {}

        def _post(m, a, o, n):
            if stack and stack[-1] == n:
                stack.pop()
            return None  # never replace the module output

        for name, mod in self.model.named_modules():
            hooks.append(mod.register_forward_pre_hook(lambda m, a, n=name: stack.append(n)))
            hooks.append(mod.register_forward_hook(lambda m, a, o, n=name: _post(m, a, o, n)))

        def pack(t):
            try:
                if t.is_cuda:
                    st = t.untyped_storage()
                    k = st.data_ptr()
                    if k not in seen:
                        seen[k] = (st.nbytes(), stack[-1] if stack else "<top>",
                                   tuple(t.shape), str(t.dtype))
            except Exception:
                pass
            return t

        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        # per-module allocated-memory delta (post - pre), synchronized
        curve = []
        pre_mem = {}

        def _mpre(m, a, n):
            torch.cuda.synchronize()
            pre_mem.setdefault(n, []).append(torch.cuda.memory_allocated())

        def _mpost(m, a, o, n):
            torch.cuda.synchronize()
            now = torch.cuda.memory_allocated()
            p = pre_mem[n].pop() if pre_mem.get(n) else now
            curve.append((n, (now - p) / 2**30, (now - m0) / 2**30))
            return None

        for name, mod in self.model.named_modules():
            short = name.replace("._checkpoint_wrapped_module", "")
            depth = short.count(".")
            if (short.startswith("paligemma_with_expert.layers.") and depth == 2) or \
               (short.startswith("paligemma_with_expert.vision_tower.encoder.layers.") and depth == 4) or \
               (depth <= 1 and short):
                if short != name:
                    continue  # hook the CheckpointWrapper (outer) only
                hooks.append(mod.register_forward_pre_hook(lambda m, a, n=short: _mpre(m, a, n)))
                hooks.append(mod.register_forward_hook(lambda m, a, o, n=short: _mpost(m, a, o, n)))
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            if os.environ.get("GB0_MEM_SNAP"):
                torch.cuda.memory._record_memory_history(max_entries=400000)
            out = self._train_forward_impl(batch)
            if os.environ.get("GB0_MEM_SNAP"):
                torch.cuda.memory._dump_snapshot(os.environ["GB0_MEM_SNAP"])
                torch.cuda.memory._record_memory_history(enabled=None)
        torch.cuda.synchronize()
        m1 = torch.cuda.memory_allocated()
        for h in hooks:
            h.remove()
        for n, d, c in curve:
            print("[gb0-memcurve] %-60s delta=%.3f GB cum=%.2f GB" % (n, d, c), flush=True)
        agg = collections.Counter()
        for nb, name, _, _ in seen.values():
            agg[".".join(name.split(".")[:5])] += nb
        tot = sum(v[0] for v in seen.values())
        print("[gb0-memdiag] fwd_alloc_growth_GB=%.2f saved_outside_AC_GB=%.2f n=%d"
              % ((m1 - m0) / 2**30, tot / 2**30, len(seen)), flush=True)
        for k, v in agg.most_common(25):
            print("[gb0-memdiag]   %-70s %.3f GB" % (k, v / 2**30), flush=True)
        for nb, name, shp, dt in sorted(seen.values(), key=lambda x: -x[0])[:15]:
            print("[gb0-memdiag]   big %.3f GB %s %s %s" % (nb / 2**30, name, shp, dt), flush=True)
        return out

    def _train_forward_impl(self, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from giga_brain_0.giga_brain_0_trainer import (
            _apply_skip_loss_mask,
            _get_flow_action_horizon,
        )
        from giga_models.utils.action_horizon import downsample_flow_action_tensors

        b = batch
        images = b["images"]
        img_masks = b["image_masks"]
        lang_tokens = b["lang_tokens"]
        lang_masks = b["lang_masks"]
        lang_att_masks = b["lang_att_masks"]
        lang_loss_masks = b["lang_loss_masks"]
        fast_action_indicator = b["fast_action_indicator"]
        subtask_indicator = b["subtask_indicator"]
        actions = b["action"]
        action_loss_mask = b["action_loss_mask"]
        action_dim_loss_mask = b.get("action_dim_loss_mask")
        action_fps = b.get("action_fps")
        skip_loss_mask = b.get("skip_loss_mask")

        traj = b.get("traj")
        traj_loss_mask = b.get("traj_loss_mask") if traj is not None else None

        lang_loss_masks, action_loss_mask, action_dim_loss_mask, traj_loss_mask = (
            _apply_skip_loss_mask(
                skip_loss_mask, lang_loss_masks, action_loss_mask,
                action_dim_loss_mask, traj_loss_mask,
            )
        )

        emb_ids = b["embodiment_id"]
        flow_action_horizon = _get_flow_action_horizon(self.model)
        if action_dim_loss_mask is None:
            actions, action_loss_mask = downsample_flow_action_tensors(
                actions, action_loss_mask, flow_action_horizon, action_fps=action_fps,
            )
        else:
            actions, action_loss_mask, action_dim_loss_mask = downsample_flow_action_tensors(
                actions, action_loss_mask, flow_action_horizon, action_fps=action_fps,
                action_dim_loss_mask=action_dim_loss_mask,
            )
        if actions.ndim != 3 or actions.shape[-1] != 32:
            raise ValueError(f"Flow actions must be [B,T,32], got {tuple(actions.shape)}")

        m = self.model_cfg
        batch_mul = int(m.flow_matching_batch_mul) if bool(m.enable_multi_flow_samples) else 1
        noisy_model_input, timesteps = self._giga_loss.add_noise(
            actions, action_dim_loss_mask, batch_mul=batch_mul,
        )
        action_emb_ids = emb_ids
        if batch_mul > 1:
            action_emb_ids = emb_ids.repeat_interleave(batch_mul, dim=0)
            action_loss_mask = action_loss_mask.repeat_interleave(batch_mul, dim=0)
            if action_dim_loss_mask is not None:
                action_dim_loss_mask = action_dim_loss_mask.repeat_interleave(batch_mul, dim=0)

        _rf_fwd = nvtx_range('forward_model')
        _rf_fwd.__enter__()
        model_pred = self.model(
            images, img_masks, lang_tokens, lang_masks,
            noisy_model_input, timesteps, emb_ids,
            lang_att_masks=lang_att_masks,
            fast_action_indicator=fast_action_indicator,
            subtask_indicator=subtask_indicator,
            lang_loss_masks=lang_loss_masks,
            state_memory=b.get("observation.state_memory", None),
            state_memory_masks=b.get("observation.state_memory_mask", None),
            proprioception=b.get("observation.proprioception", None),
            agent_pos_mask=b.get("observation.agent_pos_mask", None),
            proprioception_present=b.get("observation.proprioception_present", None),
            state=b.get("observation.state"),
            action_emb_ids=action_emb_ids,
            flow_matching_batch_mul=batch_mul,
        )
        model_pred = {
            k: v if k in {"lang_logits", "lang_logits_mask", "lang_token_loss"}
            or not isinstance(v, torch.Tensor) else v.float()
            for k, v in model_pred.items()
        }
        _rf_fwd.__exit__(None, None, None)  # forward_model
        with nvtx_range('loss_compute'):
            losses = self._giga_loss(
                model_pred, lang_tokens, lang_loss_masks, action_loss_mask,
                traj, traj_loss_mask,
                action_dim_loss_mask=action_dim_loss_mask,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                embodiment_id=action_emb_ids,
                num_embodiments=getattr(self, "_num_embodiments", 0),
            )

        # giga parse_losses: drop metric/* (diagnostics), mean rest, sum = total.
        for k in [k for k in list(losses.keys()) if k.startswith("metric/")]:
            losses.pop(k)
        losses = {k: v.mean() for k, v in losses.items()}
        total = sum(losses.values())
        log = {k: v.detach() for k, v in losses.items()}
        log["total_loss"] = total.detach()
        return total, log
