# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer construction."""

import logging
from collections.abc import MutableMapping

import torch
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from loongforge.embodied.distributed.utils import is_rank_zero
from loongforge.embodied.optimizer.lr_scheduler import build_param_groups

try:
    from transformer_engine.pytorch.optimizers import FusedAdam as _TEFusedAdam
except ImportError:
    _TEFusedAdam = None

try:
    from apex.optimizers import FusedAdam as _ApexFusedAdam
except ImportError:
    _ApexFusedAdam = None

logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = {
    "AdamW": torch.optim.AdamW,
    "TorchFusedAdamW": torch.optim.AdamW,
    "TEFusedAdamW": _TEFusedAdam,
    "ApexFusedAdamW": _ApexFusedAdam,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}


class _MultiDtypeZeroOptimizer(torch.optim.Optimizer):
    """Wraps multiple ZeroRedundancyOptimizers (one per dtype) as a single Optimizer."""

    _is_multi_dtype_zero_optimizer = True

    def __init__(self, optimizers):
        """Compose multiple ZeRO optimizers without a flat params list.

        torch.optim.Optimizer.__init__ requires a flat params iterable,
        which we cannot provide here (params are split by dtype across child
        optimizers). Initialise the required attributes manually instead.
        """
        self._optimizers = optimizers
        # Attributes expected by torch.optim.Optimizer and LR schedulers.
        self._refresh_public_optimizer_state()
        self.state = {}
        self._hook_for_profile = None  # expected by some PyTorch internals

    def _refresh_public_optimizer_state(self):
        """Refresh Optimizer-like public attributes from child optimizers."""
        self.param_groups = []
        for opt in self._optimizers:
            self.param_groups.extend(opt.param_groups)
        # ``defaults`` is read by some schedulers, e.g. Transformers'
        # cosine_with_min_lr computes ``min_lr / optimizer.defaults["lr"]``.
        # The scheduler still updates every entry in the merged ``param_groups``
        # above, so inheriting defaults from the first child optimizer does not
        # limit LR decay to the first child optimizer.
        #
        # TODO: Support explicit per-param-group LR scheduler settings so
        # min-lr floors can be configured per group instead of deriving one
        # global ratio from ``defaults["lr"]``.
        self.defaults = dict(self._optimizers[0].defaults) if self._optimizers else {}

    def zero_grad(self, set_to_none=True):
        """Clear gradients of all child optimizers."""
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Copy model gradients to FP32 masters, step, then sync updated weights back."""
        if closure is not None:
            raise NotImplementedError(
                "_MultiDtypeZeroOptimizer does not support closure-based optimizers."
            )
        for opt in self._optimizers:
            opt.step()
        return None

    def state_dict(self):
        """Return a list of state_dicts, one per child optimizer (order matches __init__)."""
        return [opt.state_dict() for opt in self._optimizers]

    def load_state_dict(self, state_dict):
        """Restore each child optimizer from the corresponding state_dict, then refresh param_groups."""
        if len(state_dict) != len(self._optimizers):
            raise ValueError(
                f"Expected {len(self._optimizers)} optimizer state dicts, got {len(state_dict)}."
            )
        for opt, opt_state_dict in zip(self._optimizers, state_dict):
            opt.load_state_dict(opt_state_dict)
        self._refresh_public_optimizer_state()

    def consolidate_state_dict(self, to=0):
        """Consolidate each child ZeRO optimizer state dict."""
        for opt in self._optimizers:
            opt.consolidate_state_dict(to=to)


class _FP32MasterStateProxy(MutableMapping):
    """Expose fp32 master optimizer state using the corresponding model params."""

    def __init__(self, master_state, model_to_master):
        self._master_state = master_state
        self._model_to_master = model_to_master
        self._master_to_model = {master: model for model, master in model_to_master.items()}

    def _translate_key(self, key):
        return self._model_to_master.get(key, key)

    def __getitem__(self, key):
        return self._master_state[self._translate_key(key)]

    def __setitem__(self, key, value):
        self._master_state[self._translate_key(key)] = value

    def __delitem__(self, key):
        del self._master_state[self._translate_key(key)]

    def __iter__(self):
        for key in self._master_state:
            yield self._master_to_model.get(key, key)

    def __len__(self):
        return len(self._master_state)


class _FP32MasterOptimizerAdapter(torch.optim.Optimizer):
    """Local optimizer adapter that updates a ZeRO shard through fp32 masters."""

    _base_optimizer_cls = None

    def __init__(self, params, **kwargs):
        if self._base_optimizer_cls is None:
            raise TypeError("_FP32MasterOptimizerAdapter requires a base optimizer class.")

        self._owned_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        self._model_to_master: dict[nn.Parameter, nn.Parameter] = {}
        master_groups = [self._make_master_group(group) for group in _normalize_param_groups(params)]

        self._optimizer = self._base_optimizer_cls(master_groups, **kwargs)
        self._hook_for_profile = None
        self._refresh_public_optimizer_state()

    def _make_master_group(self, group: dict) -> dict:
        master_group = {key: value for key, value in group.items() if key != "params"}
        master_params = []
        for param in _as_param_list(group["params"]):
            master = nn.Parameter(param.detach().float().clone(), requires_grad=param.requires_grad)
            master_params.append(master)
            self._owned_pairs.append((param, master))
            self._model_to_master[param] = master
        master_group["params"] = master_params
        return master_group

    def _refresh_public_optimizer_state(self):
        """Refresh Optimizer-like public attributes from the wrapped optimizer."""
        self.param_groups = self._optimizer.param_groups
        self.defaults = dict(self._optimizer.defaults)
        self.state = _FP32MasterStateProxy(self._optimizer.state, self._model_to_master)

    def zero_grad(self, set_to_none=True):
        """Clear gradients of the owned model/master parameter pairs."""
        for param, master in self._owned_pairs:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    param.grad.detach_()
                    param.grad.zero_()
            if master.grad is not None:
                if set_to_none:
                    master.grad = None
                else:
                    master.grad.detach_()
                    master.grad.zero_()
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Apply one optimizer step after copying model gradients to FP32 master weights."""
        if closure is not None:
            raise NotImplementedError(
                "_FP32MasterOptimizerAdapter does not support closure-based optimizers."
            )

        for param, master in self._owned_pairs:
            if param.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty_like(master)
            master.grad.copy_(param.grad.detach().to(dtype=master.dtype))

        result = self._optimizer.step()
        with torch.no_grad():
            for param, master in self._owned_pairs:
                param.copy_(master.to(dtype=param.dtype))
        return result

    def add_param_group(self, param_group):
        """Add a new parameter group, mirrored into the fp32 master optimizer."""
        master_group = self._make_master_group(param_group)
        self._optimizer.add_param_group(master_group)
        self._refresh_public_optimizer_state()

    def state_dict(self):
        """Return the underlying fp32 master optimizer's state dict."""
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load a state dict into the underlying fp32 master optimizer."""
        self._optimizer.load_state_dict(state_dict)
        self._refresh_public_optimizer_state()

    def local_checkpoint_state_dict(self):
        """Return rank-local optimizer state, including fp32 master weights."""
        return {
            "optimizer": self._optimizer.state_dict(),
            "master_params": [
                master.detach() for _, master in self._owned_pairs
            ],
        }

    def load_local_checkpoint_state_dict(self, state_dict):
        """Restore rank-local optimizer state and fp32 master weights."""
        master_params = state_dict["master_params"]
        if len(master_params) != len(self._owned_pairs):
            raise ValueError(
                f"Expected {len(self._owned_pairs)} fp32 master parameters, "
                f"got {len(master_params)}."
            )

        self._optimizer.load_state_dict(state_dict["optimizer"])
        with torch.no_grad():
            for saved_master, (_, master) in zip(master_params, self._owned_pairs):
                if saved_master.shape != master.shape:
                    raise ValueError(
                        "FP32 master parameter shape mismatch: "
                        f"expected {tuple(master.shape)}, got {tuple(saved_master.shape)}."
                    )
                master.copy_(saved_master)
        self._refresh_public_optimizer_state()


def _make_fp32_master_optimizer_class(optimizer_cls):
    """Build a local optimizer class for ZeroRedundancyOptimizer."""

    class _FP32MasterOptimizer(_FP32MasterOptimizerAdapter):
        _base_optimizer_cls = optimizer_cls

    name = getattr(optimizer_cls, "__name__", optimizer_cls.__class__.__name__)
    _FP32MasterOptimizer.__name__ = f"FP32Master{name}"
    _FP32MasterOptimizer.__qualname__ = _FP32MasterOptimizer.__name__
    return _FP32MasterOptimizer


def build_optimizer(model: nn.Module, training_args) -> torch.optim.Optimizer:
    """Build the training optimizer.

    - Build per-module LR parameter groups via ``build_param_groups``.
    - Select AdamW/Adam/SGD from ``training_args.optimizer``.
    - Use ZeRO-1 to shard optimizer states when ``--zero-optimizer`` is enabled with DDP.
    - Split mixed-dtype parameters into per-dtype ZeRO optimizers to satisfy ZeroRedundancyOptimizer dtype constraints.
    """

    groups = build_param_groups(model, training_args)
    optimizer_name = training_args.optimizer
    if optimizer_name not in OPTIMIZER_REGISTRY:
        supported = ", ".join(OPTIMIZER_REGISTRY)
        raise ValueError(f"Unknown optimizer '{training_args.optimizer}'. Supported optimizers: {supported}.")
    optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
    if optimizer_cls is None:
        raise ImportError(
            f"Optimizer '{optimizer_name}' is not available: "
            "the corresponding backend (TransformerEngine or Apex) is not installed."
        )

    kwargs = {"lr": training_args.lr_base, "weight_decay": training_args.weight_decay}
    if optimizer_cls in (torch.optim.AdamW, torch.optim.Adam):
        kwargs.update(betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_eps)
        if optimizer_name == "TorchFusedAdamW":
            kwargs["fused"] = True
    elif optimizer_name in ("TEFusedAdamW", "ApexFusedAdamW"):
        kwargs.update(betas=(training_args.adam_beta1, training_args.adam_beta2), eps=training_args.adam_eps)
        kwargs["adam_w_mode"] = True
        logger.info("Using %s optimizer", optimizer_name)

    use_zero = training_args.zero_optimizer

    dtype_stats = {}
    for group in groups:
        for p in group.get("params", []):
            if p.requires_grad:
                stats = dtype_stats.setdefault(p.dtype, {"tensors": 0, "elements": 0})
                stats["tensors"] += 1
                stats["elements"] += p.numel()
    param_dtypes = set(dtype_stats)
    if is_rank_zero():
        summary = ", ".join(
            f"{dtype}: {stats['tensors']} tensors/{stats['elements']} elems"
            for dtype, stats in sorted(dtype_stats.items(), key=lambda item: str(item[0]))
        )
        logger.info("Optimizer trainable parameter dtypes: %s", summary or "none")

    # ZeRO Stage-1: shard optimizer states across DDP ranks
    if use_zero:
        strategy = training_args.distributed_strategy
        parameters_as_bucket_view = training_args.zero_parameters_as_bucket_view
        zero_master_dtype = training_args.zero_master_param_dtype
        if strategy == "ddp":
            if zero_master_dtype == "fp32":
                logger.info("Using ZeroRedundancyOptimizer (ZeRO Stage-1) with fp32 master params")
                fp32_master_optimizer_cls = _make_fp32_master_optimizer_class(optimizer_cls)
                if len(param_dtypes) > 1:
                    logger.info(
                        f"Mixed dtype params {param_dtypes}: using per-dtype "
                        "ZeroRedundancyOptimizer with fp32 master params"
                    )
                    dtype_groups = _split_param_groups_by_dtype(groups)
                    opts = [
                        ZeroRedundancyOptimizer(
                            dtype_group,
                            optimizer_class=fp32_master_optimizer_cls,
                            parameters_as_bucket_view=parameters_as_bucket_view,
                            **kwargs,
                        )
                        for dtype_group in dtype_groups.values()
                    ]
                    return _MultiDtypeZeroOptimizer(opts)
                return ZeroRedundancyOptimizer(
                    groups,
                    optimizer_class=fp32_master_optimizer_cls,
                    parameters_as_bucket_view=parameters_as_bucket_view,
                    **kwargs,
                )
            # Check for mixed dtype parameters - ZeroRedundancyOptimizer requires uniform dtype
            if len(param_dtypes) > 1:
                # Mixed dtype: split param groups by dtype, one ZeRO optimizer per dtype
                # (mirrors the pattern in mixed_precision_train.py)
                logger.info(
                    f"Mixed dtype params {param_dtypes}: using per-dtype ZeroRedundancyOptimizer"
                )
                dtype_groups = _split_param_groups_by_dtype(groups)
                opts = [
                    ZeroRedundancyOptimizer(
                        dtype_group,
                        optimizer_class=optimizer_cls,
                        parameters_as_bucket_view=parameters_as_bucket_view,
                        **kwargs,
                    )
                    for dtype_group in dtype_groups.values()
                ]
                return _MultiDtypeZeroOptimizer(opts)
            else:
                logger.info("Using ZeroRedundancyOptimizer (ZeRO Stage-1) with DDP")
                return ZeroRedundancyOptimizer(
                    groups,
                    optimizer_class=optimizer_cls,
                    parameters_as_bucket_view=parameters_as_bucket_view,
                    **kwargs,
                )
        else:
            logger.warning(
                f"--zero-optimizer ignored: only effective with --distributed-strategy ddp, "
                f"current strategy is '{strategy}' (already shards optimizer states)."
            )
    elif training_args.zero_master_param_dtype != "none":
        logger.warning("--zero-master-param-dtype ignored: --zero-optimizer is not set.")

    optimizer = optimizer_cls(groups, **kwargs)
    return optimizer


def _as_param_list(params) -> list[nn.Parameter]:
    if isinstance(params, torch.Tensor):
        return [params]
    if isinstance(params, set):
        raise TypeError(
            "optimizer parameters need to be organized in ordered collections, "
            "but got a set."
        )
    return list(params)


def _normalize_param_groups(params) -> list[dict]:
    param_groups = list(params)
    if len(param_groups) == 0:
        raise ValueError("optimizer got an empty parameter list")
    if not isinstance(param_groups[0], dict):
        return [{"params": param_groups}]
    return [
        {**group, "params": _as_param_list(group["params"])}
        for group in param_groups
    ]


def _split_param_groups_by_dtype(groups: list[dict]) -> dict[torch.dtype, list[dict]]:
    """Split optimizer param groups by dtype while preserving per-group options.

    Only trainable parameters are split; non-trainable ones are skipped
    (they will not be updated anyway and must not be passed to ZeRO).
    """
    dtype_groups: dict[torch.dtype, list[dict]] = {}
    for group in groups:
        base_group = {k: v for k, v in group.items() if k != "params"}
        params_by_dtype: dict[torch.dtype, list[nn.Parameter]] = {}
        for p in group.get("params", []):
            if not p.requires_grad:
                continue
            params_by_dtype.setdefault(p.dtype, []).append(p)
        for dtype, params in params_by_dtype.items():
            dtype_groups.setdefault(dtype, []).append({**base_group, "params": params})
    return dtype_groups
