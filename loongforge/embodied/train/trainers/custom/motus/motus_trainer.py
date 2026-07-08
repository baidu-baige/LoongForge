# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motus finetune trainer.

Extends :class:`FinetuneTrainer` with the two Motus-specific runtime pieces that
the source ``Motus/train/train.py`` (``UniDiffuserTrainer``) owns:

  1. **VAE side-stream prefetch** — the next batch's H2D copies + (no-grad) VAE encode
     run on a dedicated CUDA stream so they overlap the current batch's
     forward/backward on the main stream.
  2. **DeepSpeed / torch.compile wrapping** — under ``--use-deepcompile`` the model
     is wrapped by a DeepSpeed engine (direct ``deepspeed.initialize`` +
     ``engine.compile()``, no accelerate) that owns optimizer/backward/step and
     schedules the ZeRO-1 gradient reduce into the compiled inductor graph.

Only three hooks differ from the generic FinetuneTrainer:

  - ``_wrap_model_for_training``: model wrap (DDP by default, or the DeepSpeed
    engine path) + create the VAE side stream.
  - ``_run_forward_backward_block``: the prefetch/consume gradient-accumulation loop
    that consumes pre-encoded latents (replaces the base zero_grad + _forward_backward).
  - ``_clean_nan_gradients``: no-op under the DeepSpeed-engine paths (the engine owns
    gradient handling).

All data / optimizer / scheduler / checkpoint paths are inherited unchanged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any, Dict

import torch

from loongforge.embodied.train.utils.utils import resolve_dtype
from loongforge.embodied.distributed.utils import unwrap_model
from loongforge.embodied.train.trainers.supervised.finetune_trainer import FinetuneTrainer

logger = logging.getLogger(__name__)


def _patch_deepcompile_none_grads():
    """Make DeepSpeed DeepCompile's ZeRO-1 backward reduce pass tolerate params
    that have NO gradient in the compiled graph (grad_node is None).

    DeepCompile's add_z1_reduce_bw (deepspeed/compile/passes/zero1_compile.py)
    iterates every registered graph-input param and inserts a reduce_grad op on its
    backward-graph gradient node, assuming that node always exists. But Motus has a
    few trainable params whose output is structurally discarded and therefore have
    no gradient: e.g. the LAST understanding-expert block's post-attention output
    projection / norm2 / FFN (und_tokens after the final layer is never read by an
    output head). Stock DeepSpeed ZeRO tolerates None grads (treats them as zero and
    skips reduce); DeepCompile does not and crashes with
    'AttributeError: NoneType has no attribute name' in _make_node_meta.

    This gated patch skips the reduce insertion for params whose grad is None (there
    is nothing to reduce), matching stock ZeRO semantics. Applied only under
    DeepCompile, so the default path is untouched.
    """
    from deepspeed.compile.passes import zero1_compile
    from deepspeed.compile.fx import add_postprocess, move_primals_to_head, _make_node_meta
    import torch as _torch

    def add_z1_reduce_bw_safe(gm, graph_id, param_manager):
        graph = gm.graph
        pm = param_manager[graph_id]
        _, param_name_to_grad = pm.get_bwd_mapping(graph)
        skipped = []
        for param_name in pm.param_names:
            grad_node = param_name_to_grad[param_name]
            if grad_node is None:
                # Param has no gradient in this graph (dead output). Nothing to
                # reduce; stock ZeRO would leave grad=None. Skip it.
                skipped.append(param_name)
                continue
            assert param_name in pm.ds_ids, f"param_name={param_name} not in ds_ids"
            ds_id = pm.ds_ids[param_name]
            new_node = add_postprocess(graph,
                                       grad_node,
                                       _torch.ops.dc.reduce_grad.default,
                                       extra_args=[graph_id, ds_id],
                                       name=f"reduce_param_{param_name}",
                                       meta=_make_node_meta(grad_node, param_name, True))
            new_node.meta["val"] = None
        if skipped:
            logger.info(f"DeepCompile: skipped in-graph reduce for {len(skipped)} "
                        f"param(s) with no gradient (dead output): {skipped}")
        gm.graph = move_primals_to_head(graph)
        return gm

    zero1_compile.add_z1_reduce_bw = add_z1_reduce_bw_safe
    logger.info("DeepCompile: patched add_z1_reduce_bw to tolerate None-grad params.")


class MotusTrainer(FinetuneTrainer):
    """Motus finetune trainer with DDP/DeepSpeed wrapping + VAE side-stream prefetch.

    By default the model is DDP-wrapped and trained with the client optimizer.

    When ``--use-deepcompile`` is set the trainer takes an alternative path: the
    model is wrapped by a DeepSpeed engine (direct ``deepspeed.initialize`` +
    ``engine.compile()``, no accelerate) that owns the optimizer/backward/step and
    schedules the ZeRO-1 gradient reduce into the compiled inductor graph. In that
    case the DDP wrap is skipped and forward routes to the model's
    ``_deepcompile_core`` while the frozen preprocessing and fp32 loss run eager.
    """

    def __init__(self, training_args, model_cfg, data_cfg):
        self._deepcompile_enabled = bool(getattr(training_args, "use_deepcompile", False))
        # PARITY (default inert): EAGER_DEEPSPEED=1 wraps the model in a DeepSpeed
        # ZeRO-1 engine via direct deepspeed.initialize but SKIPS engine.compile()
        # and the static dense-flash attn path, so forward/backward run EAGER (same
        # varlen path as the DDP/base eager run) while ZeRO-1 owns the reduce/step.
        # Used to isolate whether the step-2 loss divergence is DeepCompile-specific
        # (graph partition / in-graph reduce order) or already present in eager+ZeRO.
        self._eager_deepspeed = os.environ.get("EAGER_DEEPSPEED", "0") == "1"
        # PARITY (default inert): PLAIN_COMPILE=1 wraps the model in an EAGER
        # DeepSpeed ZeRO-1 engine (deepspeed.initialize, NO engine.compile) exactly
        # like EAGER_DEEPSPEED, but then applies a PLAIN torch.compile (inductor,
        # mode="default", no DeepCompile fx passes) to _deepcompile_core over the
        # SAME static dense-flash + fp64-rope path. This mirrors base motus
        # _build_torch_compile (models/motus.py:_build_torch_compile). It isolates
        # whether the step-2 video divergence comes from DeepCompile's in-graph
        # ZeRO-1 reduce (diverges) vs plain inductor forward codegen of the static
        # video path (would also diverge if inductor itself is the cause).
        self._plain_compile = os.environ.get("PLAIN_COMPILE", "0") == "1"
        # True whenever a DeepSpeed engine (compiled or eager) owns wrapping /
        # zero_grad / backward / step (as opposed to the DDP + client-optimizer
        # path).
        self._ds_engine_enabled = (
            self._deepcompile_enabled
            or self._eager_deepspeed
            or self._plain_compile
        )

        super().__init__(training_args, model_cfg, data_cfg)
        # VAE prefetch pipeline state (created in _wrap_model_for_training).
        self._vae_stream: torch.cuda.Stream | None = None
        # Carries the pre-encoded latents from the previous prefetch across
        # micro-steps AND across optimizer steps (pipelined, like the source).
        self._prefetched: Dict[str, Any] | None = None

    def _inner_motus(self):
        """Return the raw ``Motus`` (the module that owns ``_deepcompile_core`` /
        ``_deepcompile_preprocess`` / ``forward`` routing), unwrapping whatever
        currently wraps it.

        ``build_model`` returns a :class:`MotusPolicy` framework wrapper whose
        ``self.model`` is the real ``Motus``; under DeepCompile the DeepSpeed engine
        then wraps the ``Motus`` directly (``engine.module`` is the ``Motus``). This
        peels both layers if present:
          - ``.module``  unwraps a DeepSpeed engine / DDP wrapper,
          - ``.model``   unwraps the ``MotusPolicy`` (``Motus`` has no ``.model``,
                         so this is a no-op once we reach it).
        """
        m = self.model
        m = getattr(m, "module", m)
        m = getattr(m, "model", m)
        return m


    # ═══════════════════════════════════════════════
    # Override 1: wrapping + side-stream setup
    # ═══════════════════════════════════════════════

    def _wrap_model_for_training(self) -> None:
        """DDP-wrap the model, then create the VAE prefetch side stream.

        By default Motus uses standard DDP wrapping (the base implementation
        performs the DDP/FSDP wrap).

        Under DeepCompile the DDP/FSDP wrap is SKIPPED — the DeepSpeed engine
        (built in ``_build_optimizer``) replaces it. Here we only prepare the raw
        model: enable the static dense-flash attn path (so the whole-module
        compile does not hit the varlen flash graph break) and set the instance
        flag that routes ``Motus.forward`` to ``_deepcompile_core``.
        """
        self._compute_dtype = resolve_dtype(self.training_args.dtype)

        if self._eager_deepspeed:
            # Eager DeepSpeed: the engine (built in _build_optimizer) owns wrapping.
            # Do NOT enable the static dense-flash attn path — forward stays eager
            # varlen, matching the base eager+deepspeed run (model.training_step).
            if torch.cuda.is_available():
                self._vae_stream = torch.cuda.Stream()
            logger.info(
                "MotusTrainer[EagerDeepSpeed]: skipped DDP wrap (DeepSpeed engine "
                "owns wrapping); eager varlen forward (no static attn path); "
                "VAE side-stream=%s",
                self._vae_stream is not None,
            )
            return

        if self._plain_compile:
            # torch.compile + eager DeepSpeed, IMPLEMENTED EXACTLY LIKE BASE MOTUS
            # (bak_dpc/Motus _build_torch_compile):
            #   - enable the SAME static dense-flash + fp64-rope path (graph_fhw +
            #     use_sdpa) so the compiled region traces cleanly (no varlen break);
            #   - wrap core_forward_static with a PLAIN torch.compile (inductor,
            #     mode="default", fullgraph=False, dynamic=False).
            # Crucially we DO NOT set raw._deepcompile_enabled: the forward must run
            # through raw.training_step (which calls the now-compiled
            # core_forward_static directly), NOT through engine.__call__ ->
            # _deepcompile_core. This mirrors base, where the loop calls
            # model.training_step(...) on the raw module and only backward goes
            # through the engine. NO DeepCompile fx passes; the eager ZeRO-1 engine
            # (built in _build_optimizer) owns the normal post-backward gradient reduce.
            raw = self._inner_motus()
            raw.enable_deepcompile_static_path()
            raw.core_forward_static = torch.compile(
                raw.core_forward_static,
                backend="inductor",
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
            if torch.cuda.is_available():
                self._vae_stream = torch.cuda.Stream()
            logger.info(
                "MotusTrainer[PlainCompile]: skipped DDP wrap (eager DeepSpeed engine "
                "owns wrapping); enabled static attn path; torch.compile(core_forward_static, "
                "mode=default); forward via raw.training_step (base-parity, NOT "
                "engine.__call__/_deepcompile_core); VAE side-stream=%s",
                self._vae_stream is not None,
            )
            return

        if self._deepcompile_enabled:
            raw = self._inner_motus()
            raw.enable_deepcompile_static_path()
            raw._deepcompile_enabled = True
            if torch.cuda.is_available():
                self._vae_stream = torch.cuda.Stream()
            logger.info(
                "MotusTrainer[DeepCompile]: skipped DDP wrap (DeepSpeed engine "
                "owns wrapping); enabled static attn path; VAE side-stream=%s",
                self._vae_stream is not None,
            )
            return

        super()._wrap_model_for_training()

        if torch.cuda.is_available():
            self._vae_stream = torch.cuda.Stream()
        logger.info(
            "MotusTrainer: DDP-wrapped model; VAE prefetch side-stream=%s",
            self._vae_stream is not None,
        )


    # ═══════════════════════════════════════════════
    # DeepCompile: engine build + optimizer-step override
    # ═══════════════════════════════════════════════

    def _load_deepcompile_config(self) -> dict:
        """Load the DeepSpeed JSON config and inject batch / accumulation sizes.

        ``train_micro_batch_size_per_gpu`` / ``gradient_accumulation_steps`` /
        ``train_batch_size`` come from the training args (the bundled config keeps
        them ``"auto"``), so the ds config stays a single source of truth for the
        ZeRO-1 + ``compile.deepcompile`` settings while the batch geometry follows
        the launch flags.
        """
        path = self.training_args.deepspeed_config
        if not path:
            raise ValueError(
                "--use-deepcompile requires --deepspeed-config to point at the "
                "DeepSpeed JSON (e.g. examples/embodied/motus/zero1_deepcompile.json)."
            )
        with open(path) as f:
            cfg = json.load(f)
        micro = int(self.training_args.per_device_batch_size)
        gas = int(self.training_args.gradient_accumulation_steps)
        cfg["train_micro_batch_size_per_gpu"] = micro
        cfg["gradient_accumulation_steps"] = gas
        cfg["train_batch_size"] = micro * gas * self.ctx.world_size
        return cfg

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer; under DeepCompile also build/compile the engine.

        Default path: delegate to the base (AdamW with per-module LR groups).

        DeepCompile path: build the same AdamW on the (still unwrapped) model, then
        wrap the model with a DeepSpeed engine via direct ``deepspeed.initialize``
        (no accelerate) and run ``engine.compile()`` so DeepCompile's ZeRO-1 fx
        passes schedule the gradient reduce into the compiled inductor graph. The
        engine replaces ``self.model`` (the base loop calls ``self.model(...)`` /
        the deepcompile step calls ``self.model.backward``/``.step``). The returned
        optimizer is the DeepSpeed-managed optimizer, which ``_build_scheduler``
        then attaches the LR schedule to.
        """
        if not self._ds_engine_enabled:
            return super()._build_optimizer()

        import deepspeed
        from loongforge.embodied.optimizer import build_optimizer

        if getattr(self.training_args, "zero_optimizer", False):
            raise ValueError(
                "--use-deepcompile / EAGER_DEEPSPEED / PLAIN_COMPILE is incompatible "
                "with --zero-optimizer: the DeepSpeed engine provides ZeRO-1 itself, "
                "so build a plain AdamW (drop --zero-optimizer from the launch args)."
            )

        adamw = build_optimizer(self.model, self.training_args)
        # Must be applied before engine.compile / the first compiled backward.
        _patch_deepcompile_none_grads()

        ds_config = self._load_deepcompile_config()
        _eager_engine = self._eager_deepspeed or self._plain_compile
        if _eager_engine:
            # Strip the compile block so deepspeed.initialize builds a plain eager
            # ZeRO-1 engine (no DeepCompile fx passes). Everything else — ZeRO-1
            # stage/overlap/bucketing, bf16, batch geometry — stays identical.
            # PLAIN_COMPILE still gets a torch.compile'd core (applied in
            # _wrap_model_for_training), just not DeepCompile's in-graph reduce.
            ds_config.pop("compile", None)

        inner_motus = self._inner_motus()
        engine, ds_optimizer, _, _ = deepspeed.initialize(
            model=inner_motus,
            optimizer=adamw,
            config=ds_config,
            dist_init_required=False,
        )
        self.model = engine
        if _eager_engine:
            # NO engine.compile(): forward/backward run eager; ZeRO-1 does the
            # normal (non-graph) gradient reduce + fp32-master Adam step.
            logger.info(
                "MotusTrainer[%s]: DeepSpeed ZeRO-1 engine initialized (eager, NO "
                "engine.compile).",
                "PlainCompile" if self._plain_compile else "EagerDeepSpeed",
            )
            return adamw
        engine.compile()
        logger.info(
            "MotusTrainer[DeepCompile]: DeepSpeed engine initialized + compiled "
            "(ZeRO-1 + deepcompile)."
        )
        # Return the CLIENT AdamW (not the DeepSpeedZeroOptimizer wrapper) so the
        # framework LR scheduler can read ``optimizer.defaults["lr"]`` (the ZeRO
        # wrapper does not expose ``.defaults``). ZeRO-1 wraps this same AdamW and
        # steps it in place via its live ``param_groups``, so scheduler LR updates
        # to ``adamw.param_groups[*]["lr"]`` still drive the actual optimizer step.
        # The engine (self.model) owns backward/step; we never step this directly.
        return adamw

    def _train_step(self):
        """Optimizer step. Under DeepCompile the engine owns zero_grad / clip /
        step (scheduled in-graph), so skip the base clip + optimizer.step + NaN
        cleanup and only run the forward-backward block + LR scheduler step.

        The default (non-deepcompile) path defers entirely to the base skeleton.
        """
        if not self._ds_engine_enabled:
            return super()._train_step()

        self._step_loss_is_nan = False
        self._step_loss_spiked = False

        log_dict = self._run_forward_backward_block()

        if self._step_loss_is_nan:
            self.nan_iterations += 1
        if self._step_loss_spiked:
            self.skipped_iterations += 1

        # DeepSpeed engine.step() (called per micro in the fb block) already ran
        # the optimizer update + grad zero; only advance the LR schedule here.
        self.lr_scheduler.step()

        # grad_norm is not separately computed on the deepcompile path (the reduce
        # is fused into the backward graph). Report 0.0 to keep the log schema.
        return log_dict, 0.0

    # ═══════════════════════════════════════════════
    # Override 2: prefetch/consume forward-backward block
    # ═══════════════════════════════════════════════


    def _run_forward_backward_block(self) -> dict:
        """Gradient-accumulation loop over pre-encoded (prefetched) VAE latents.

        Mirrors ``UniDiffuserTrainer.train``'s inner loop: prime the pipeline on
        the first call, then for each micro-step consume the already-encoded
        current batch while launching the next batch's VAE encode on the side
        stream. ``zero_grad`` fires only on the first micro-step; the base
        ``_train_step`` performs clip + optimizer/scheduler step afterwards.

        VAE tail-prefetch (always on): the next batch's raw CPU fetch happens at
        the head of the micro-step (overlapping the previous step's GPU tail, no
        side-stream work), but the GPU H2D + VAE encode launch is deferred until
        AFTER ``_train_step_from_latents`` is dispatched, so VAE(N+1) overlaps
        THIS step's backward rather than its eager preprocess (where it would
        otherwise saturate the SM and serialize the blocking DTOH host-reads).
        Only the encode moves to the tail; the dataloader fetch stays at the head
        to avoid a synchronous dataloader stall in the step body.
        """
        grad_accum = self.training_args.gradient_accumulation_steps

        with self._stage_timers("forward-backward"):
            # Prime the pipeline once (first optimizer step, first micro-step).
            if self._prefetched is None:
                self._prefetched = self._prefetch_batch(self._next_raw_batch())

            accum: Dict[str, torch.Tensor] = {}
            for micro in range(grad_accum):
                cur = self._prefetched

                raw_next = self._next_raw_batch()
                if micro == 0 and not self._ds_engine_enabled:
                    with self._stage_timers("optimizer-zero-grad"):
                        self.optimizer.zero_grad(set_to_none=True)
                is_last = micro == grad_accum - 1
                step_metrics = self._train_step_from_latents(cur, is_last)
                self._prefetched = self._prefetch_batch(raw_next)

                for k, v in step_metrics.items():
                    accum[k] = accum[k] + v if k in accum else v

        # Average across micro-steps, emit floats for logging/aggregation.
        log_dict: Dict[str, float] = {}
        for k, v in accum.items():
            val = v / grad_accum
            log_dict[k] = val.detach().item() if torch.is_tensor(val) else float(val)
        return log_dict

    # ═══════════════════════════════════════════════
    # Override 3: NaN-grad cleanup (no-op under the DeepSpeed-engine paths)
    # ═══════════════════════════════════════════════

    def _clean_nan_gradients(self) -> None:
        """Skip host-side NaN/Inf grad cleanup under the DeepSpeed-engine paths.

        When a DeepSpeed engine owns backward/step (DeepCompile / eager DeepSpeed /
        plain-compile), the engine handles gradients internally and the source
        trainer performs no such cleanup either. Otherwise delegate to the base
        implementation.
        """
        if self._ds_engine_enabled:
            return
        super()._clean_nan_gradients()

    # ═══════════════════════════════════════════════
    # Helpers — VAE prefetch pipeline (ported from source train.py)
    # ═══════════════════════════════════════════════

    def _next_raw_batch(self):
        """Fetch the next CPU batch (epoch cycling handled by the base loop)."""
        return self._fetch_batch_cpu("vla")

    def _prefetch_batch(self, batch) -> Dict[str, Any]:
        """Move a raw MotusPreparedBatch to GPU and VAE-encode its latents.

        Under the DeepSpeed-engine paths the copies + encode are issued on the
        dedicated side stream, fully decoupled from the main stream so they overlap
        the previous batch's compute; the only cross-stream dependency is the
        recorded completion event, which the main stream waits on in
        :meth:`_train_step_from_latents`. Otherwise everything runs on the main
        stream.
        """
        model = unwrap_model(self.model)
        device = self.ctx.device
        dtype = self._compute_dtype
        # The frozen VAE runs independently of the compiled core, and the main
        # stream waits on the completion event before consuming, so the side stream
        # is safe.
        use_async = self._vae_stream is not None and self._ds_engine_enabled

        def _copies_and_encode():
            language_embeddings = batch.language_embedding
            if language_embeddings is not None:
                language_embeddings = language_embeddings.to(device, dtype=dtype, non_blocking=True)
            state = batch.initial_state
            if state is not None:
                state = state.to(device, dtype=dtype, non_blocking=True)
            actions = batch.action_sequence.to(device, dtype=dtype, non_blocking=True)
            vlm_inputs = batch.vlm_inputs
            if vlm_inputs is not None:
                vlm_inputs = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in vlm_inputs.items()
                }
            if batch.clean_full_latent is not None:
                # Offline latent cache hit (--latent-cache-dir): skip the frozen
                # VAE encode (and the first_frame/video_frames H2D it needed).
                # Keep fp32 (NO dtype cast) so the value is bit-identical to the
                # online encode output; condition_frame_latent is the delta-0
                # slice, exactly as encode_video_latents produces it.
                clean_full_latent = batch.clean_full_latent.to(device, non_blocking=True)
                condition_frame_latent = clean_full_latent[:, :, 0:1, :, :]
            else:
                first_frame = batch.first_frame.to(device, dtype=dtype, non_blocking=True)
                video_frames = batch.video_frames.to(device, dtype=dtype, non_blocking=True)
                clean_full_latent, condition_frame_latent = model.encode_video_latents(
                    first_frame, video_frames
                )
                if os.environ.get("LATENT_DUMP", "0") == "1":
                    self._latent_dump_batch(first_frame, video_frames, clean_full_latent)
            return clean_full_latent, condition_frame_latent, state, actions, language_embeddings, vlm_inputs

        if use_async:
            # No wait_stream(main): the side stream must NOT depend on the main
            # stream's in-flight graph. The caching-host allocator's event
            # tracking keeps pinned buffers safe even after `batch` is dropped.
            with torch.cuda.stream(self._vae_stream):
                (clean_full_latent, condition_frame_latent, state, actions,
                 language_embeddings, vlm_inputs) = _copies_and_encode()
            event = torch.cuda.Event()
            event.record(self._vae_stream)
        else:
            (clean_full_latent, condition_frame_latent, state, actions,
             language_embeddings, vlm_inputs) = _copies_and_encode()
            event = None

        return {
            "clean_full_latent": clean_full_latent,
            "condition_frame_latent": condition_frame_latent,
            "event": event,
            "state": state,
            "actions": actions,
            "language_embeddings": language_embeddings,
            "vlm_inputs": vlm_inputs,
        }

    def _latent_dump_batch(self, first_frame, video_frames, clean_full_latent) -> None:
        """Gated (LATENT_DUMP=1) diagnostic dump of the exact VAE encode inputs +
        output for a batch, consumed by verify_offline_latents.py to bit-compare an
        offline recompute against this ONLINE latent. Inert unless LATENT_DUMP=1.

        Dumps the bf16 encode inputs (first_frame / video_frames as fed to
        ``encode_video_latents``) and the fp32 ``clean_full_latent`` output, so the
        offline script can replay the identical normalization + ``vae.encode`` path
        at the same batch size and diff bit-for-bit. ``.to("cpu")`` is a blocking
        copy, so it is safe to call inside the side-stream context.
        """
        max_steps = int(os.environ.get("LATENT_DUMP_STEPS", "3"))
        count = getattr(self, "_latent_dump_count", 0)
        if count >= max_steps:
            return
        dump_dir = os.environ.get("LATENT_DUMP_DIR", "/tmp/latent_dump")
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"rank{self.ctx.rank}_step{count}.pt")
        torch.save(
            {
                "first_frame": first_frame.detach().to("cpu"),
                "video_frames": video_frames.detach().to("cpu"),
                "clean_full_latent": clean_full_latent.detach().to("cpu"),
                "compute_dtype": str(self._compute_dtype),
                "batch_size": int(first_frame.shape[0]),
            },
            path,
        )
        self._latent_dump_count = count + 1
        logger.info(
            "[LATENT_DUMP] wrote %s (batch=%d, latent=%s)",
            path, first_frame.shape[0], tuple(clean_full_latent.shape),
        )

    def _init_grad_probe_hooks(self, raw) -> None:
        """Register per-parameter autograd hooks that accumulate the LOCAL (pre-reduce)
        gradient norm per submodule. Only under GRAD_PROBE=1; registered once.

        Reading ``p.grad`` after ``engine.backward()`` is UNRELIABLE under this run's
        DeepSpeed ZeRO-1 config (stage 1, overlap_comm + contiguous_gradients): the
        engine redirects ``p.grad`` into a flat partition buffer for params in the
        current rank's optimizer partition and leaves the rest stale (see
        deepspeed/runtime/zero/stage_1_and_2.py: copy_grads_in_partition ->
        ``grad_reduc.data = new_grad_tensor.data``). A ``Tensor.register_hook`` instead
        fires during autograd the instant the local grad is produced by the (compiled)
        backward, BEFORE the engine mutates it -- giving the pure per-rank local grad,
        which isolates the compiled-backward difference and is directly comparable to
        base (rank0 sees the same data shard on both sides).
        """
        if os.environ.get("GRAD_PROBE") != "1":
            return
        if getattr(self, "_grad_probe_hooked", False):
            return
        self._grad_probe_hooked = True
        self._grad_probe_active = False
        self._grad_probe_acc = {}
        self._grad_probe_cnt = {}

        def _make_hook(key):
            def _hook(grad):
                if not self._grad_probe_active or grad is None:
                    return
                self._grad_probe_acc[key] = (
                    self._grad_probe_acc.get(key, 0.0)
                    + float(grad.detach().float().pow(2).sum())
                )
                self._grad_probe_cnt[key] = self._grad_probe_cnt.get(key, 0) + 1
            return _hook

        for key, modname in (("video", "video_model"), ("action", "action_expert"), ("und", "und_expert")):
            mod = getattr(raw, modname, None)
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    p.register_hook(_make_hook(key))
        logger.info(
            "MotusTrainer[GRAD_PROBE]: registered local-grad hooks on "
            "video_model / action_expert / und_expert (pre-reduce, rank0 comparable to base)"
        )

    def _grad_probe_begin(self, raw) -> None:
        """Arm the grad hooks for the first few steps and reset the accumulators."""
        if os.environ.get("GRAD_PROBE") != "1":
            return
        self._init_grad_probe_hooks(raw)
        _gs = getattr(raw, "_parity_fm_step", None)
        self._grad_probe_active = (_gs is None or _gs <= 5)
        self._grad_probe_acc = {}
        self._grad_probe_cnt = {}

    def _grad_probe_end(self, raw, phase: str) -> None:
        """After backward: print per-submodule local grad norm on rank0."""
        if os.environ.get("GRAD_PROBE") != "1" or not getattr(self, "_grad_probe_active", False):
            return
        self._grad_probe_active = False
        import math
        import torch.distributed as _distgp
        _rk = _distgp.get_rank() if _distgp.is_available() and _distgp.is_initialized() else 0
        if _rk != 0:
            return
        _gs = getattr(raw, "_parity_fm_step", None)

        def _n(key):
            return (math.sqrt(self._grad_probe_acc.get(key, 0.0)), self._grad_probe_cnt.get(key, 0))

        vN, vn = _n("video")
        aN, an = _n("action")
        uN, un = _n("und")
        print(
            f"[GRAD_PROBE] phase={phase} fm_step={_gs} "
            f"video_gnorm={vN:.6e}(n={vn}) "
            f"action_gnorm={aN:.6e}(n={an}) "
            f"und_gnorm={uN:.6e}(n={un})",
            flush=True,
        )

    def _train_step_from_latents(self, item: Dict[str, Any], is_last: bool) -> Dict[str, torch.Tensor]:
        """One micro-step consuming pre-encoded latents; returns detached losses."""
        self.model.train()

        # Make the side-stream work visible to the main compute stream and tell
        # the allocator these tensors are consumed on the main stream so they are
        # not freed/reused before the main stream is done with them.
        if item["event"] is not None:
            main = torch.cuda.current_stream()
            main.wait_event(item["event"])
            for t in (
                item["clean_full_latent"], item["condition_frame_latent"],
                item["state"], item["actions"], item["language_embeddings"],
            ):
                if torch.is_tensor(t):
                    t.record_stream(main)
            if item["vlm_inputs"] is not None:
                for v in item["vlm_inputs"].values():
                    if torch.is_tensor(v):
                        v.record_stream(main)

        grad_accum = self.training_args.gradient_accumulation_steps

        if self._eager_deepspeed:
            # Eager DeepSpeed ZeRO-1: run the SAME eager varlen forward as the base
            # eager+deepspeed run (model.training_step), then let the DeepSpeed engine
            # own backward (loss scaling by grad_accum via the ds config) + step +
            # ZeRO reduce. No engine.compile(), no static attn path, no manual scale.
            raw = unwrap_model(self.model)
            with self._stage_timers("forward-compute"):
                loss_dict = raw.training_step(
                    clean_full_latent=item["clean_full_latent"],
                    condition_frame_latent=item["condition_frame_latent"],
                    state=item["state"],
                    actions=item["actions"],
                    language_embeddings=item["language_embeddings"],
                    vlm_inputs=item["vlm_inputs"],
                    return_dict=True,
                )
            with self._stage_timers("backward-compute"):
                self._grad_probe_begin(raw)
                self.model.backward(loss_dict["total_loss"])
                self._grad_probe_end(raw, "eager")
                self.model.step()
            return {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            }

        if self._plain_compile:
            # forward through raw.training_step on the unwrapped module (which calls
            # the torch.compile'd core_forward_static directly, with preprocess + fp32
            # loss eager around it), NOT through engine.__call__/_deepcompile_core.
            # This mirrors base train.py, where the loop calls
            # model.training_step(...) on the raw model and only backward/step go
            # through the (eager ZeRO-1) engine. Keeping the forward off the engine's
            # __call__ is the key base-parity fix: routing the compiled core through
            # engine.__call__ diverged (video loss stuck at ~untrained level) whereas
            # the raw.training_step path converges like base.
            raw = unwrap_model(self.model)
            with self._stage_timers("forward-compute"):
                loss_dict = raw.training_step(
                    clean_full_latent=item["clean_full_latent"],
                    condition_frame_latent=item["condition_frame_latent"],
                    state=item["state"],
                    actions=item["actions"],
                    language_embeddings=item["language_embeddings"],
                    vlm_inputs=item["vlm_inputs"],
                    return_dict=True,
                )
            with self._stage_timers("backward-compute"):
                self._grad_probe_begin(raw)
                self.model.backward(loss_dict["total_loss"])
                self._grad_probe_end(raw, "plain_compile")
                self.model.step()
            return {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            }

        if self._deepcompile_enabled:
            # DeepCompile split: the frozen preprocessing (VAE latents in, frozen
            # VLM, T5 padding, FM sampling) and the fp32 loss run EAGER here; only
            # the all-trainable core (_deepcompile_core) is compiled by DeepCompile
            # (engine.compile). This avoids the "All param inputs should have
            # param_id" assertion that whole-training_step compilation hits on the
            # frozen backbone params. DeepSpeed owns the loss scaling (engine.backward
            # divides by the gradient_accumulation_steps in the ds config) and the
            # accumulation boundary / all-reduce, so we do NOT no_sync or manually
            # scale, and we call engine.step() every micro-step.
            raw = unwrap_model(self.model)
            with self._stage_timers("forward-compute"):
                core_inputs, loss_ctx = raw._deepcompile_preprocess(
                    clean_full_latent=item["clean_full_latent"],
                    condition_frame_latent=item["condition_frame_latent"],
                    state=item["state"],
                    actions=item["actions"],
                    language_embeddings=item["language_embeddings"],
                    vlm_inputs=item["vlm_inputs"],
                )
                # engine.__call__ -> compiled module.forward -> _deepcompile_core
                video_pred, action_pred = self.model(**core_inputs)
                loss_dict = raw._deepcompile_loss(video_pred, action_pred, loss_ctx)
            with self._stage_timers("backward-compute"):
                self.model.backward(loss_dict["total_loss"])
                self.model.step()
            return {
                k: (v.detach() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            }

        # Gate DDP gradient sync so the all-reduce fires exactly once per
        # optimizer step (only on the final micro-step).
        need_no_sync = (
            self.ctx.is_distributed
            and not is_last
            and hasattr(self.model, "no_sync")
        )
        sync_ctx = self.model.no_sync() if need_no_sync else contextlib.nullcontext()

        model = unwrap_model(self.model)
        with sync_ctx:
            with self._stage_timers("forward-compute"):
                loss_dict = model.training_step(
                    clean_full_latent=item["clean_full_latent"],
                    condition_frame_latent=item["condition_frame_latent"],
                    state=item["state"],
                    actions=item["actions"],
                    language_embeddings=item["language_embeddings"],
                    vlm_inputs=item["vlm_inputs"],
                    return_dict=True,
                )
                scaled_loss = loss_dict["total_loss"] / grad_accum
            with self._stage_timers("backward-compute"):
                scaled_loss.backward()

        return {
            k: (v.detach() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
        }
