# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FP8 dynamic selection policy — benchmark-driven init-time FP8 decisions.

This module contains ``FP8DynamicPolicy`` and the
``selective_fp8_init_decision`` callback. Omni registers this callback into
Megatron via ``register_selective_fp8_init_decision`` during parser setup.
At model construction time, Megatron's selective-FP8 init guard calls the
registered callback to decide whether each TE module should use FP8 or BF16.
"""

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from TE module class → benchmark module_kind.
# Authoritative, model-agnostic: derived from the class itself.
# ---------------------------------------------------------------------------
_TE_CLASS_TO_MODULE_KIND = {}

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
        TELinear,
    )

    _TE_CLASS_TO_MODULE_KIND = {
        TELayerNormColumnParallelLinear: "layernorm_column",
        TEColumnParallelLinear: "column",
        TERowParallelLinear: "row",
        TEColumnParallelGroupedLinear: "column_grouped",
        TERowParallelGroupedLinear: "row_grouped",
        TELinear: "duplicated",  # MLA down-projection (parallel_mode='duplicated')
    }
except ImportError:
    pass

# Fallback: ub_name → module_kind (for edge cases where TE class is unknown).
_UB_NAME_TO_MODULE_KIND = {
    "qkv": "layernorm_column",
    "fc1": "layernorm_column",
    "proj": "row",
    "fc2": "row",
    # MLA down-projection (TELinear with parallel_mode='duplicated')
    "q_down_proj": "duplicated",
    "kv_down_proj": "duplicated",
}

_MOE_MODULE_KINDS = {"column_grouped", "row_grouped"}


# ---------------------------------------------------------------------------
# FP8DynamicPolicy
# ---------------------------------------------------------------------------
class FP8DynamicPolicy:
    """Benchmark-driven FP8 selection policy.

    Loads a JSON policy file exported by the TE parallel layer benchmark
    and provides a lookup interface to decide whether a given module should
    use FP8 based on its *module_kind*, effective token count, and parallel
    configuration.

    Policy JSON format::

        {
          "version": 1,
          "speedup_threshold": 1.0,
          "rules": {
            "layernorm_column": {
              "qkv": [{"tp": 1, "min_tokens": 12288, "measured_speedup": 1.03}],
              "fc1": [{"tp": 1, "min_tokens": 4096,  "measured_speedup": 1.18}]
            },
            "row": {
              "proj": [{"tp": 1, "min_tokens": 99999999}],
              "fc2":  [{"tp": 1, "min_tokens": 8192, "measured_speedup": 1.15}]
            },
            "column_grouped": [
              {"etp": 1, "num_gemms": 64, "min_tokens": 424, ...}
            ],
            "row_grouped": [
              {"etp": 1, "num_gemms": 64, "min_tokens": 424, ...}
            ]
          }
        }

    Dense module kinds (layernorm_column / column / row / duplicated) use a
    nested ``{ub_name: [rules]}`` dict so that same-kind modules with
    different shapes (e.g. qkv vs fc1) can have distinct thresholds. MoE
    grouped kinds keep the flat list form.
    """

    def __init__(self, policy_path: str):
        import json as _json

        with open(policy_path, "r") as f:
            data = _json.load(f)
        self._rules = data.get("rules", {})
        # Build lookup indices for fast access.
        self._dense_index = {}  # (module_kind, ub_name, tp) -> min_tokens
        self._moe_index = {}  # (module_kind, etp, num_gemms) -> min_tokens
        for module_kind, entry in self._rules.items():
            if module_kind in _MOE_MODULE_KINDS:
                for i, rule in enumerate(entry):
                    try:
                        key = (module_kind, rule["etp"], rule["num_gemms"])
                        self._moe_index[key] = rule["min_tokens"]
                    except KeyError as e:
                        raise ValueError(
                            f"Malformed FP8 policy rule at rules[{module_kind!r}][{i}]: "
                            f"missing required key {e}. Rule content: {rule}"
                        ) from None
            else:
                for ub_name, rule_list in entry.items():
                    for i, rule in enumerate(rule_list):
                        try:
                            key = (module_kind, ub_name, rule["tp"])
                            self._dense_index[key] = rule["min_tokens"]
                        except KeyError as e:
                            raise ValueError(
                                f"Malformed FP8 policy rule at "
                                f"rules[{module_kind!r}][{ub_name!r}][{i}]: "
                                f"missing required key {e}. Rule content: {rule}"
                            ) from None

    def should_use_fp8(
        self,
        module_kind: str,
        num_tokens: int,
        tp: int = 1,
        etp: int = 1,
        num_gemms: int = 1,
        ub_name: Optional[str] = None,
    ) -> bool:
        """Return True if FP8 is expected to be faster than BF16.

        For dense module kinds ``ub_name`` is required; a missing entry is
        treated conservatively (BF16).
        """
        if module_kind in _MOE_MODULE_KINDS:
            min_tokens = self._moe_index.get((module_kind, etp, num_gemms))
        else:
            min_tokens = self._dense_index.get((module_kind, ub_name, tp))
        if min_tokens is None:
            return False  # No benchmark data → conservative BF16
        return num_tokens >= min_tokens

# ---------------------------------------------------------------------------
# Global cache, lazily loaded per policy_path.
# ---------------------------------------------------------------------------
_FP8_DYNAMIC_POLICY_CACHE: Dict[str, FP8DynamicPolicy] = {}


def _resolve_policy_path(policy_path: str) -> str:
    """Resolve relative paths against the LoongForge project root."""
    if os.path.isabs(policy_path):
        return policy_path
    # Project root = parent of the loongforge package directory
    # __file__ is loongforge/train/fp8_dynamic_policy.py → up 3 levels to project root
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(_project_root, policy_path)


def _get_fp8_dynamic_policy(policy_path: Optional[str]) -> Optional[FP8DynamicPolicy]:
    """Return the cached FP8DynamicPolicy, loading it on first call."""
    if policy_path is None:
        return None
    policy_path = _resolve_policy_path(policy_path)
    policy = _FP8_DYNAMIC_POLICY_CACHE.get(policy_path)
    if policy is None:
        policy = FP8DynamicPolicy(policy_path)
        _FP8_DYNAMIC_POLICY_CACHE[policy_path] = policy
    return policy


# ---------------------------------------------------------------------------
# Auto-compute dense_num_tokens from global args if not explicitly set.
# ---------------------------------------------------------------------------
_AUTO_DENSE_NUM_TOKENS_WARNED = False


def _auto_dense_num_tokens() -> int:
    """Derive dense token count from ``args.seq_length * args.micro_batch_size``.

    .. warning::

        This uses the **global** ``args.seq_length`` which is typically the LLM
        sequence length.  In multimodal models (e.g. Qwen3-VL) where the ViT
        encoder processes a different effective token count, the auto-computed
        value will be incorrect for the ViT component.  Set
        ``fp8_dynamic_num_tokens`` explicitly in each component's config to
        avoid this.
    """
    global _AUTO_DENSE_NUM_TOKENS_WARNED
    try:
        from megatron.training import get_args

        args = get_args()
        seq_length = getattr(args, "seq_length", 0) or 0
        mbs = getattr(args, "micro_batch_size", 1) or 1
        num_tokens = seq_length * mbs
        if num_tokens > 0 and not _AUTO_DENSE_NUM_TOKENS_WARNED:
            _AUTO_DENSE_NUM_TOKENS_WARNED = True
            logger.warning(
                "fp8_dynamic_num_tokens not set; auto-computed %d from "
                "global args (seq_length=%d * micro_batch_size=%d). "
                "For multimodal models with heterogeneous sequence lengths, "
                "set fp8_dynamic_num_tokens explicitly per component config.",
                num_tokens, seq_length, mbs,
            )
        return num_tokens
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Decision function — registered into Megatron via
# ``register_selective_fp8_init_decision`` at startup (see parser.py).
# ---------------------------------------------------------------------------
def selective_fp8_init_decision(config, *, te_cls, ub_name, init_kwargs) -> bool:
    """Decide whether *this* TE module should be initialized with FP8.

    Registered into Megatron's ``fp8_utils.register_selective_fp8_init_decision``
    during ``parse_args_from_config``, so Megatron never imports this module.

    Returns:
        True  → keep FP8 for this module.
        False → disable FP8 (module runs in BF16).
    """
    from megatron.core.fp8_utils import _keep_fp8_for_ub_name

    policy_path = getattr(config, "fp8_dynamic_policy_path", None)
    policy = _get_fp8_dynamic_policy(policy_path)
    if policy is None:
        # No dynamic policy configured; fall back to static whitelist.
        return _keep_fp8_for_ub_name(ub_name)

    is_expert = init_kwargs.get("is_expert", False)
    num_gemms = init_kwargs.get("num_gemms", 1)

    # Resolve module_kind from TE class, fall back to ub_name.
    module_kind = _TE_CLASS_TO_MODULE_KIND.get(te_cls)
    if module_kind is None:
        module_kind = _UB_NAME_TO_MODULE_KIND.get(ub_name)

    # Expert modules: promote to grouped variants.
    if is_expert and module_kind in ("layernorm_column", "column", "row"):
        module_kind = (
            "column_grouped" if module_kind in ("layernorm_column", "column") else "row_grouped"
        )

    if module_kind is None:
        return _keep_fp8_for_ub_name(ub_name)

    # Determine effective token count.
    dense_num_tokens = getattr(config, "fp8_dynamic_num_tokens", 0)
    if dense_num_tokens <= 0:
        dense_num_tokens = _auto_dense_num_tokens()
    if dense_num_tokens <= 0:
        return _keep_fp8_for_ub_name(ub_name)

    tp = config.tensor_model_parallel_size
    etp = config.expert_tensor_parallel_size

    if is_expert:
        moe_topk = getattr(config, "moe_router_topk", 1) or 1
        num_tokens = dense_num_tokens * moe_topk
        return policy.should_use_fp8(
            module_kind, num_tokens, tp=tp, etp=etp, num_gemms=num_gemms
        )
    else:
        return policy.should_use_fp8(
            module_kind, dense_num_tokens, tp=tp, ub_name=ub_name
        )
