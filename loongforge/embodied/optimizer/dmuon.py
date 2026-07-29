# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""DMuon optimizer integration."""

from __future__ import annotations

import inspect
import logging

import torch
import torch.nn as nn

from loongforge.embodied.distributed.fsdp_utils import (
    find_fsdp_root_module,
    get_fsdp_root_sharded_params,
)
from loongforge.embodied.distributed.utils import is_dmuon_model, unwrap_model
from loongforge.embodied.optimizer.clip_gradients import clip_gradients

logger = logging.getLogger(__name__)

# ── foreach AdamW ────────────────────────────────────────────────────────────
# ``dmuon.Muon`` updates the FSDP2 symmetric shards one tensor at a time. The
# subclass below batches that update into multi-tensor kernels. It lives here
# rather than as a patch to ``dmuon/optim/muon.py`` in site-packages, because an
# out-of-tree patch is invisible to git and is silently lost when the image is
# rebuilt or dmuon is reinstalled — the speedup disappears without any error.
#
# Only ``_step_adamw_params`` is overridden. ``_step_dedicated_adamw_params``
# keeps its scalar loop, so the DMuon-dedicated parameters are untouched.

# Base-class internals the override relies on. ``dmuon.Muon`` is third-party, so
# the coupling is checked once instead of failing mid-step.
_REQUIRED_MUON_API = (
    "_step_adamw_params",
    "_profile_add",
    "_first_step_log",
)

_FOREACH_MUON_CLASS = None


def _adamw_foreach_update_bucket(entries: list[tuple], group: dict) -> None:
    """Run the scalar AdamW operation sequence as multi-tensor kernels."""
    params = [entry[1] for entry in entries]
    grads = [entry[2] for entry in entries]
    exp_avgs = [entry[3] for entry in entries]
    exp_avg_sqs = [entry[4] for entry in entries]

    lr = group["lr"]
    beta1, beta2 = group["betas"]
    wd = group["weight_decay"]
    eps = group["eps"]
    step = entries[0][5]

    if wd > 0:
        torch._foreach_mul_(params, 1.0 - lr * wd)

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(
        exp_avg_sqs,
        grads,
        grads,
        value=1.0 - beta2,
    )

    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, bc2**0.5)
    torch._foreach_add_(denoms, eps)
    torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-(lr / bc1))

    for p in (entry[0] for entry in entries):
        p.grad = None


def _step_adamw_params_foreach(optimizer, params, group: dict) -> None:
    """Apply AdamW to FSDP2 local shards with bounded foreach buckets.

    Takes the optimizer explicitly instead of being a method so the subclass
    body stays small; ``optimizer`` is the ``dmuon.Muon`` instance.
    """
    grouped_entries: dict[tuple, list[tuple]] = {}
    for p in params:
        if p.grad is None:
            continue
        grad = p.grad._local_tensor if hasattr(p.grad, "_local_tensor") else p.grad
        param = p._local_tensor if hasattr(p, "_local_tensor") else p.data

        state = optimizer.state[p]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] += 1

        # The step count is part of the key because the bias correction is
        # applied per bucket, so a bucket must be step-uniform.
        key = (param.device, param.dtype, grad.dtype, state["step"])
        grouped_entries.setdefault(key, []).append(
            (
                p,
                param,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                state["step"],
            )
        )

    bucket_count = 0
    tensor_count = 0
    max_temp_bytes = 0
    max_bucket_bytes = optimizer._adamw_foreach_max_temp_bytes
    for entries in grouped_entries.values():
        bucket: list[tuple] = []
        bucket_temp_bytes = 0
        for entry in entries:
            exp_avg_sq = entry[4]
            entry_temp_bytes = exp_avg_sq.numel() * exp_avg_sq.element_size()
            if bucket and bucket_temp_bytes + entry_temp_bytes > max_bucket_bytes:
                _adamw_foreach_update_bucket(bucket, group)
                bucket_count += 1
                tensor_count += len(bucket)
                max_temp_bytes = max(max_temp_bytes, bucket_temp_bytes)
                bucket = []
                bucket_temp_bytes = 0
            bucket.append(entry)
            bucket_temp_bytes += entry_temp_bytes

        if bucket:
            _adamw_foreach_update_bucket(bucket, group)
            bucket_count += 1
            tensor_count += len(bucket)
            max_temp_bytes = max(max_temp_bytes, bucket_temp_bytes)

    optimizer._profile_add("adamw_foreach_buckets", bucket_count)
    optimizer._profile_add("adamw_foreach_tensors", tensor_count)
    optimizer._profile_add("adamw_foreach_max_temp_bytes", max_temp_bytes)
    if not optimizer._adamw_foreach_logged:
        optimizer._first_step_log(
            "AdamW foreach enabled: "
            f"tensors={tensor_count} buckets={bucket_count} "
            f"max_temp_bytes={max_temp_bytes}"
        )
        optimizer._adamw_foreach_logged = True


def _build_foreach_muon_class(dmuon):
    """Create (once) the ``dmuon.Muon`` subclass that batches the AdamW update.

    Built lazily because ``dmuon`` is an optional dependency that this module
    deliberately does not import at module scope.
    """
    global _FOREACH_MUON_CLASS
    if _FOREACH_MUON_CLASS is not None:
        return _FOREACH_MUON_CLASS

    class ForeachAdamWMuon(dmuon.Muon):
        """``dmuon.Muon`` with the symmetric AdamW update batched into foreach calls."""

        def __init__(self, *args, foreach_bucket_mib: float = 64.0, **kwargs) -> None:
            """Initialize the instance."""
            super().__init__(*args, **kwargs)
            # Assigned after super().__init__ on purpose: while the legacy
            # site-packages patch is still installed, its constructor sets these
            # same attributes from the old DMUON_ADAMW_FOREACH* environment
            # variables. Writing afterwards makes the CLI arguments win.
            self._adamw_foreach_max_temp_bytes = max(
                1, int(foreach_bucket_mib * 1024 * 1024)
            )
            self._adamw_foreach_logged = False

        def _step_adamw_params(self, params, group: dict) -> None:
            """Apply one AdamW subgroup's hyperparameters to managed params."""
            _step_adamw_params_foreach(self, params, group)

    _FOREACH_MUON_CLASS = ForeachAdamWMuon
    return _FOREACH_MUON_CLASS


def resolve_muon_class(dmuon, training_args):
    """Return the Muon class to instantiate for this run.

    Falls back to the unmodified ``dmuon.Muon`` unless
    ``--dmuon-adamw-foreach`` is set, so a dmuon upgrade that breaks the
    private-API assumptions above can only affect runs that opted in.
    """
    if not training_args.dmuon_adamw_foreach:
        return dmuon.Muon

    missing = [name for name in _REQUIRED_MUON_API if not hasattr(dmuon.Muon, name)]
    if missing:
        raise RuntimeError(
            "--dmuon-adamw-foreach requires dmuon.Muon internals that are "
            f"missing in the installed dmuon: {', '.join(missing)}. Drop the "
            "flag to fall back to the scalar AdamW path."
        )
    return _build_foreach_muon_class(dmuon)


def _build_ns_backend(dmuon, training_args):
    """Build ns backend."""
    coefficients = training_args.dmuon_ns_coefficients
    if coefficients == "default":
        return training_args.dmuon_ns_backend
    if coefficients != "wallx_muon":
        raise ValueError(
            "Unsupported DMuon ns coefficients: "
            f"{coefficients!r}. Supported: default, wallx_muon."
        )
    if training_args.dmuon_ns_backend != "direct":
        raise ValueError(
            "dmuon_ns_coefficients='wallx_muon' requires dmuon_ns_backend='direct'."
        )
    # (a, b, c) = (3.4445, -4.7750, 2.0315) are the tuned quintic Newton-Schulz
    # coefficients from Keller Jordan's Muon (https://github.com/KellerJordan/Muon)
    wallx_coefficients = [
        [3.4445, -4.7750, 2.0315]
        for _ in range(training_args.dmuon_ns_steps)
    ]
    return dmuon.NewtonSchulz(
        backend="direct",
        coefficients=wallx_coefficients,
    )


class DMuonOptimizerAdapter(torch.optim.Optimizer):
    """Optimizer-compatible shim around ``dmuon.Muon``.

    DMuon uses dedicated parameters that are not clipped by the normal FSDP2
    parameter traversal.  The trainer already calls ``optimizer.clip_grad_norm``
    when present, so this adapter keeps DMuon-specific gradient handling out of
    the trainer loop while preserving PyTorch scheduler expectations.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        scheduler_peak_lr: float,
        root_optimizer_prefetch: bool = False,
    ) -> None:
        """Initialize the instance."""
        self._optimizer = optimizer
        self._model = model
        self._scheduler_peak_lr = scheduler_peak_lr
        self.last_dmuon_clip_stats = None
        self._refresh_public_optimizer_state()
        self._hook_for_profile = None
        self._root_optimizer_prefetch = root_optimizer_prefetch
        # Resolved lazily on the first step: the FSDP wrapping may still be
        # pending when the optimizer is constructed.
        self._root_prefetch_plan = None

    def _refresh_public_optimizer_state(self) -> None:
        """Refresh public optimizer state."""
        self.param_groups = self._optimizer.param_groups
        self.state = self._optimizer.state
        self.defaults = dict(self._optimizer.defaults)
        self.defaults["lr"] = self._scheduler_peak_lr

    def __getattr__(self, name):
        """Getattr."""
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            optimizer = object.__getattribute__(self, "_optimizer")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        return getattr(optimizer, name)

    def zero_grad(self, set_to_none: bool = True):
        """Zero grad."""
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Step."""
        if closure is None:
            plan = self._resolve_root_prefetch_plan()
            if plan is not None:
                return self._step_root_then_rest(*plan)
        return self._optimizer.step(closure=closure)

    def _resolve_root_prefetch_plan(self):
        """Return the cached ``(root_params, launch_unshard)`` plan, if any."""
        if not self._root_optimizer_prefetch:
            return None
        if self._root_prefetch_plan is None:
            # FSDP wrapping may still be pending at construction time, so the
            # root group is discovered once on the first step and cached.
            self._root_prefetch_plan = self._build_root_prefetch_plan() or False
        return self._root_prefetch_plan or None

    def _build_root_prefetch_plan(self):
        """Locate the FSDP root param group and build its async-unshard callback."""
        if not torch.cuda.is_available():
            return None
        root = find_fsdp_root_module(unwrap_model(self._model))
        if root is None:
            logger.warning(
                "Root optimizer/unshard prefetch requested, but no FSDP2 root "
                "module was found; falling back to a plain optimizer step."
            )
            return None
        root_params = get_fsdp_root_sharded_params(root)
        if not root_params:
            logger.warning("No sharded parameters found in the FSDP root group.")
            return None

        def launch_unshard():
            root.unshard(async_op=True)

        return (root_params, launch_unshard)

    @torch.no_grad()
    def _step_root_then_rest(self, root_params, after_root):
        """Update root FSDP AdamW params, launch a callback, then finish step.

        The callback is intended to enqueue the root FSDP async unshard.  DMuon
        keeps the root parameters in its symmetric AdamW groups, so updating
        that subset first lets FSDP's communication streams run while the
        remaining Muon/AdamW work executes on the compute stream.
        """
        optimizer = self._optimizer
        required = (
            "_adamw_group_params",
            "_step_adamw_params",
            "_ensure_grads_ready",
            "_step_muon",
            "_step_dedicated_adamw",
            "_step_adamw",
            "_profile_begin_step",
            "_first_step_progress_begin",
            "_first_step_progress_end",
        )
        if any(not hasattr(optimizer, name) for name in required):
            raise RuntimeError(
                "Installed dmuon.Muon lacks the private subset-step API needed "
                "for root optimizer/unshard overlap."
            )
        root_ids = {id(param) for param in root_params}
        if not root_ids:
            return optimizer.step()

        optimizer._profile_begin_step()
        first_step_started_at = optimizer._first_step_progress_begin()
        try:
            if not optimizer._replicate_async:
                # Sync mode prepares every dedicated gradient before updates.
                # Async mode deliberately keeps its group-local prepare/update
                # pipeline intact below; root symmetric FSDP grads are already
                # ready when the trainer enters optimizer.step().
                prepare_token = optimizer._profile_event_start("prepare_muon_grads")
                try:
                    optimizer._ensure_grads_ready()
                finally:
                    optimizer._profile_event_end(prepare_token)

            root_count = 0
            root_token = optimizer._profile_event_start("root_adamw")
            try:
                for group_idx, params in optimizer._adamw_group_params.items():
                    root_group = [param for param in params if id(param) in root_ids]
                    if root_group:
                        root_count += len(root_group)
                        optimizer._step_adamw_params(
                            root_group,
                            optimizer.param_groups[group_idx],
                        )
            finally:
                optimizer._profile_event_end(root_token)
            if root_count == 0:
                raise RuntimeError(
                    "No root parameters were found in DMuon's symmetric AdamW "
                    "groups; refusing to run a partial optimizer step."
                )

            after_root()

            if optimizer._replicate_async:
                pipeline_token = optimizer._profile_event_start("group_pipeline")
                try:
                    optimizer._step_muon_and_dispatch_groups_async()
                finally:
                    optimizer._profile_event_end(pipeline_token)
            else:
                muon_token = optimizer._profile_event_start("muon")
                try:
                    optimizer._step_muon()
                finally:
                    optimizer._profile_event_end(muon_token)
                dedicated_token = optimizer._profile_event_start("dedicated_adamw")
                try:
                    optimizer._step_dedicated_adamw()
                finally:
                    optimizer._profile_event_end(dedicated_token)

            # Root grads were cleared above. The normal AdamW pass therefore
            # updates only the remaining FSDP symmetric parameters.
            adamw_token = optimizer._profile_event_start("adamw")
            try:
                optimizer._step_adamw()
            finally:
                optimizer._profile_event_end(adamw_token)

            if not optimizer._replicate_async:
                publish_token = optimizer._profile_event_start("post_step_publish")
                try:
                    from dmuon import broadcast_all_updates

                    broadcast_all_updates(optimizer.model)
                finally:
                    optimizer._profile_event_end(publish_token)

            fence_token = optimizer._profile_event_start("isolated_process_group_fence")
            try:
                from dmuon.utils import fence_isolated_process_groups

                fence_isolated_process_groups(optimizer.model)
            finally:
                optimizer._profile_event_end(fence_token)
            optimizer._grads_ready = False
        except Exception:
            optimizer._first_step_progress_end(first_step_started_at, failed=True)
            raise
        optimizer._first_step_progress_end(first_step_started_at)
        return None

    def state_dict(self):
        """State dict."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict."""
        result = self._optimizer.load_state_dict(state_dict)
        self._refresh_public_optimizer_state()
        return result

    def add_param_group(self, param_group):
        """Add param group."""
        result = self._optimizer.add_param_group(param_group)
        self._refresh_public_optimizer_state()
        return result

    def clip_grad_norm(self, max_norm: float) -> float:
        """Clip grad norm."""
        import dmuon

        fsdp_norm = float(clip_gradients(self._model, max_norm))
        clip_stats = dmuon.clip_grad_norm_(self._optimizer, max_norm)
        self.last_dmuon_clip_stats = clip_stats.as_dict()
        dmuon_norm = float(clip_stats.total_norm)
        return float((fsdp_norm**2 + dmuon_norm**2) ** 0.5)


def build_dmuon_optimizer(
    model: nn.Module,
    training_args,
    *,
    param_groups: list[dict] | None = None,
):
    """Build a dmuon.Muon optimizer from TrainingArgs."""
    import dmuon

    if not is_dmuon_model(model):
        raise RuntimeError(
            "DMuon optimizer requires dmuon.dedicate_params() before optimizer creation."
        )
    if param_groups is not None and "param_groups" not in inspect.signature(dmuon.Muon).parameters:
        raise RuntimeError(
            "Installed dmuon.Muon does not support param_groups=. Update dmuon "
            "or remove --lr-group."
        )

    logger.info(
        "DMuon optimizer: muon_lr=%s momentum=%s ns_steps=%s; "
        "adamw_lr=%s betas=(%s, %s) wd=%s eps=%s; "
        "ns_backend=%s ns_coefficients=%s nesterov=%s",
        training_args.dmuon_muon_lr,
        training_args.dmuon_momentum,
        training_args.dmuon_ns_steps,
        training_args.dmuon_adamw_lr,
        training_args.adam_beta1,
        training_args.adam_beta2,
        training_args.weight_decay,
        training_args.adam_eps,
        training_args.dmuon_ns_backend,
        training_args.dmuon_ns_coefficients,
        training_args.dmuon_nesterov,
    )

    kwargs = {}
    if param_groups is not None:
        kwargs["param_groups"] = param_groups
        logger.info(
            "DMuon semantic LR groups: %s",
            [
                {
                    "name": group.get("group_name", group.get("name", f"group_{idx}")),
                    "lr": group.get("lr"),
                    "num_params": len(group.get("params", [])),
                }
                for idx, group in enumerate(param_groups)
            ],
        )

    muon_cls = resolve_muon_class(dmuon, training_args)
    if muon_cls is not dmuon.Muon:
        kwargs["foreach_bucket_mib"] = training_args.dmuon_adamw_foreach_bucket_mib
        logger.info(
            "DMuon optimizer class: %s (foreach AdamW, bucket=%s MiB)",
            muon_cls.__name__,
            training_args.dmuon_adamw_foreach_bucket_mib,
        )
    optimizer = muon_cls(
        model,
        lr=training_args.dmuon_muon_lr,
        momentum=training_args.dmuon_momentum,
        weight_decay=training_args.dmuon_muon_weight_decay,
        ns_steps=training_args.dmuon_ns_steps,
        adamw_lr=training_args.dmuon_adamw_lr,
        adamw_betas=(
            training_args.adam_beta1,
            training_args.adam_beta2,
        ),
        adamw_weight_decay=training_args.weight_decay,
        adamw_eps=training_args.adam_eps,
        ns_backend=_build_ns_backend(dmuon, training_args),
        nesterov=training_args.dmuon_nesterov,
        forward_prefetch_depth=training_args.dmuon_forward_prefetch_depth,
        **kwargs,
    )
    return DMuonOptimizerAdapter(
        optimizer,
        model,
        scheduler_peak_lr=training_args.lr_base,
        root_optimizer_prefetch=getattr(
            training_args, "fsdp_root_optimizer_prefetch", False
        ),
    )
