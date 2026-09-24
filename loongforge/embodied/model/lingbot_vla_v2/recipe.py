# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LingBot VLA v2 training recipe independent of parallelism selection."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import torch

from loongforge.embodied.model.lingbot_vla_v2.batch_enricher import (
    LingbotVlaV2BatchEnricher,
)

logger = logging.getLogger(__name__)


# Block lists eligible for regional ``torch.compile``, as submodule keys under
# ``policy.model.qwenvl_with_expert``. Compiling one block at a time rather than
# the whole tower is deliberate:
#   * the backbone's ``flex_attention`` call lives in
#     ``QwenvlWithExpertV2Model.forward``, outside the layers, so a per-layer
#     region leaves the separately compiled ``flex_compile`` path untouched;
#   * ``build_block_mask`` is ``@torch.compiler.disable``, and it also sits
#     outside the layers, so no region has to graph-break around it;
#   * each block is its own graph, so a shape change recompiles one block
#     instead of the model.
_REGIONAL_COMPILE_TARGETS = {
    "vision": "qwenvl.visual.blocks",
    "vlm_decoder": "qwenvl.model.language_model.layers",
    "action_expert": "qwen_expert.model.layers",
}


class LingbotVlaV2Recipe:
    """Own LingBot batch, teacher, optimizer, and model-forward semantics."""

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.batch_enricher = LingbotVlaV2BatchEnricher(model_cfg)

    def setup_parallel_state(self, trainer) -> None:
        """No-op: the model's parallel-state singleton is initialised at build time.

        The upstream ``lingbotvla`` package keeps its own parallel-state global
        (read by its fused-MoE). It is initialised as plain replicated DDP by the
        ``_apply_lingbot_parallel_state_shim`` runtime seam inside
        ``build_foundation_model`` (see ``upstream_adapter``), so there is nothing
        to set up here. The harness's own distributed state is managed separately
        by the trainer.
        """
        return

    def setup(self, trainer) -> None:
        """Wire regional compile and the batch enricher.

        Flex-attention compilation on the model is installed by the
        ``_apply_lingbot_flex_compile_shim`` runtime seam (gated by the same
        ``model.flex_compile`` flag) at build time, so it is not wired here.
        """
        self._apply_regional_compile(trainer)
        self.batch_enricher.setup(trainer.ctx, trainer.model_cfg, trainer.training_args)

    def _apply_regional_compile(self, trainer) -> None:
        """Compile the block lists named by ``model.regional_compile``, per block.

        ``nn.Module.compile`` is used rather than ``torch.compile(module)`` because
        it installs the compiled call on the existing module instead of returning a
        wrapper. By the time this runs, the replicated-sharded parameter manager has already
        captured this module tree, so replacing submodules would leave it holding
        the uncompiled ones.

        The blocks derive from ``GradientCheckpointingLayer``, whose ``__call__``
        forwards to ``nn.Module.__call__``, which is what dispatches to the
        compiled call. The compiled region therefore lands *inside* the checkpoint
        rather than around it, which is the nesting that lets the recompute pass
        reuse the cached graph instead of retracing.
        """
        targets = list(self.model_cfg.regional_compile)
        if not targets:
            return
        unknown = sorted(set(targets) - set(_REGIONAL_COMPILE_TARGETS))
        if unknown:
            raise ValueError(
                f"unknown model.regional_compile targets: {', '.join(unknown)}; "
                f"valid targets are {sorted(_REGIONAL_COMPILE_TARGETS)}"
            )
        net = trainer.model.policy.model.qwenvl_with_expert
        for target in targets:
            blocks = net.get_submodule(_REGIONAL_COMPILE_TARGETS[target])
            for block in blocks:
                # dynamic=False matches flex_compile: shapes are padded to a
                # multiple of FLEX_SPARSE_BLOCK_SIZE, so specializing is correct
                # and avoids the dynamic-shape guards on every block.
                block.compile(dynamic=False)
            logger.info(
                "Regional compile: %s blocks under %s",
                len(blocks),
                _REGIONAL_COMPILE_TARGETS[target],
            )

    def submit_teacher(self, batch):
        """Start the frozen teachers for ``batch`` and return a handle, or ``None``.

        Split out of ``prepare_batch`` so the trainer can start them a step early,
        inside the previous step's optimizer window. Measured: that window absorbs
        79% of the teacher's 145.7 ms, against 34% when the teacher runs into the
        forward, because it is 375.9 of its 389.4 ms NCCL and so has the tensor-core
        and latency slack the forward does not.

        ``None`` means the targets are already in ``batch.data``: either the runner
        is unavailable, in which case this blocked and computed them, or there is
        nothing to compute.
        """
        return self.batch_enricher.submit(batch)

    def prepare_batch(self, trainer, batch, teacher=None) -> None:
        """Downcast float inputs, drop unused keys, and attach teacher targets."""
        data = batch.data
        data.pop("rep_id", None)
        if trainer.model.policy.training:
            for name in ("state", "actions"):
                value = data.get(name)
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    data[name] = value.to(torch.bfloat16)

        # Teacher targets are only read by the auxiliary heads, which run after the
        # backbone, so they can be computed concurrently with the student forward.
        # The handle is resolved inside FlowMatchingV2.forward right before the depth
        # head. ``teacher`` is set when the trainer already started them a step ago.
        if teacher is None:
            teacher = self.submit_teacher(batch)
        if teacher is not None:
            trainer.model.policy.set_pending_teacher(teacher)

    def forward(self, trainer, batch, teacher=None):
        """Prepare the batch, seed the flow-matching draw, and run the model."""
        self._maybe_dump_tape(trainer, batch)
        self.prepare_batch(trainer, batch, teacher)
        self._set_fm_step_seed(trainer)
        return trainer.model(batch)

    @staticmethod
    def _maybe_dump_tape(trainer, batch) -> None:
        """Optionally snapshot the *raw* per-step batch to disk for replay parity.

        Gated by ``LINGBOT_DUMP_TAPE`` (a directory). Runs BEFORE ``prepare_batch``
        so the snapshot still holds the teacher inputs (``pil_images`` /
        ``future_pil_images``) that ``batch_enricher`` pops during enrichment, and
        before the bf16 downcast — i.e. the portable, harness-agnostic inputs. One
        file per (step, rank): ``step{N}_rank{R}.pt``. No-op when the env var is unset,
        so the normal training path is byte-for-byte unchanged.
        """
        import os

        dump_dir = os.environ.get("LINGBOT_DUMP_TAPE")
        if not dump_dir:
            return
        import torch

        step = int(trainer.completed_steps) + 1
        rank = int(getattr(trainer.ctx, "local_rank", 0))
        os.makedirs(dump_dir, exist_ok=True)

        def _to_cpu(value):
            if torch.is_tensor(value):
                return value.detach().to("cpu")
            if isinstance(value, (list, tuple)):
                return type(value)(_to_cpu(v) for v in value)
            if isinstance(value, dict):
                return {k: _to_cpu(v) for k, v in value.items()}
            return value

        snapshot = {k: _to_cpu(v) for k, v in batch.data.items()}
        torch.save(snapshot, os.path.join(dump_dir, f"step{step}_rank{rank}.pt"))

    @staticmethod
    def _set_fm_step_seed(trainer) -> None:
        """Seed the flow-matching noise/time draw per step and per rank.

        Without this the draw comes off the global CUDA RNG, which makes it a
        function of how much RNG the setup consumed (teachers, parameter manager,
        torch.compile) — so the same batch and the same weights give a different
        fm loss in a different trainer, and every rank draws the *same* noise and
        the same timesteps because all ranks share one seed (base_trainer seeds
        every rank identically on purpose). The seed formula is the benchmark
        lingbot-vla-v2 one (tasks/vla/train_lingbotvla.py), including the rank
        term that breaks that symmetry:

            seed + step * 1_000_003 + local_rank * 97

        ``completed_steps`` is incremented after the step, so +1 matches the
        upstream ``global_step``, which is incremented before the forward.
        """
        from loongforge.embodied.model.lingbot_vla_v2.fm_seed import (
            set_fm_step_seed,
        )

        step = int(trainer.completed_steps) + 1
        seed = int(trainer.training_args.seed)
        local_rank = int(getattr(trainer.ctx, "local_rank", 0))
        set_fm_step_seed(seed + step * 1_000_003 + local_rank * 97)

    def build_optimizer(self, trainer):
        """Build the vendored Muon optimizer, or ``None`` for other optimizers."""
        if trainer.training_args.optimizer.lower() != "muon":
            return None
        from loongforge.embodied.train.trainers.custom.replicated_sharded_training.muon import (
            build_muon_optimizer,
        )
        from loongforge.embodied.model.lingbot_vla_v2.moe_load_balance import (
            build_moe_load_balance_hook,
        )

        cfg = trainer.model_cfg
        args = SimpleNamespace(
            use_moe=cfg.use_moe,
            use_moe_expert_lr=cfg.use_moe_expert_lr,
            token_moe_layers=list(cfg.token_moe_layers),
            token_num_experts=cfg.token_num_experts,
            token_top_k=cfg.token_top_k,
            muon_momentum=cfg.muon_momentum,
            muon_nesterov=cfg.muon_nesterov,
            muon_ns_steps=cfg.muon_ns_steps,
            muon_adjust_lr_fn=cfg.muon_adjust_lr_fn,
            muon_exclude_name_patterns=list(cfg.muon_exclude_name_patterns or []),
        )
        optimizer = build_muon_optimizer(
            trainer._optimizer_parameter_model,
            args,
            lr=trainer.training_args.lr_base,
            weight_decay=trainer.training_args.weight_decay,
            parameter_policy=trainer._optimizer_runtime.parameter_policy,
        )
        if cfg.use_moe:
            optimizer.register_step_pre_hook(
                build_moe_load_balance_hook(
                    trainer.model,
                    coeff=cfg.bias_update_speed,
                    bias_centering=cfg.bias_centering,
                    update_interval=cfg.bias_update_interval,
                )
            )
        return optimizer

    def wire_parameter_sync(self, trainer) -> None:
        """Hand Muon-managed master parameters to the replicated-sharded publish callback."""
        from loongforge.embodied.train.trainers.custom.replicated_sharded_training.muon import (
            DistributedMuon,
            split_muon_adamw_params,
        )

        muon = next(
            (
                opt
                for opt in trainer.optimizer.optimizers
                if isinstance(opt, DistributedMuon)
            ),
            None,
        )
        if muon is None:
            return
        runtime = trainer._optimizer_runtime
        # The parameter policy already routes muon_exclude_name_patterns to AdamW
        # (they are folded into its adamw markers), so the split is driven by the
        # policy alone -- passing the patterns again here would be a no-op.
        _, _, muon_names, _ = split_muon_adamw_params(
            runtime.compute_parameter_view(),
            parameter_policy=runtime.parameter_policy,
        )
        runtime.set_in_step_updated_names(muon_names)
        muon.param_update_callback = runtime.on_master_updated

    def close(self) -> None:
        """Shut down the batch enricher's teacher runner."""
        self.batch_enricher.close()


__all__ = ["LingbotVlaV2Recipe"]
