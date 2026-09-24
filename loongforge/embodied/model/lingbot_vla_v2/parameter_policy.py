# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Parameter precision policy for LingBot VLA v2 distributed strategies.

This is the LingBot instance of the shared, config-driven replicated-sharded policy. It only
supplies the two per-axis marker lists and their class labels; the matching and
the ``compute_dtype`` / ``is_comm_precision_critical`` / ``classify`` contract
live in ``MarkerParameterPolicy``. A new model reuses the same base with its own
markers -- no branching is duplicated here.

    class             compute  grad-reduce            param-publish
    ---------------------------------------------------------------------------
    vlm_backbone      bf16     bf16 / fp32 (by mode)  bf16 (always)
    action_expert     fp32     bf16 / fp32 (by mode)  fp32 / bf16 / fp8 (by knob)
    expert_gate_norm  fp32     fp32 (critical)        fp32 (critical)
    vlm_norm_bias     bf16     fp32 (critical)        fp32 (critical)

The concrete downcast dtype (bf16 vs fp8) is resolved in the replicated-sharded registry from
the global param_sync/grad_reduce knob; this policy only declares which class a
parameter belongs to.
"""

from __future__ import annotations

from loongforge.embodied.model.precision_policy import MarkerParameterPolicy

# Action-expert tensors train in fp32; everything else in bf16.
_ACTION_FP32_MARKERS = (
    "qwenvl_with_expert.qwen_expert.model.layers.",
    "qwenvl_with_expert.qwen_expert.model.norm.",
)
# Expert selection is the one place where a rounding error changes the
# computation instead of perturbing it: the MoE gate already runs its matmul in
# true fp32 (qwen2_action_expert.py) because bf16 logits flip top-k. A router
# weight that arrives bf16- or fp8-rounded reintroduces exactly that failure, so
# router/gate weights keep full-precision collectives even when everything else is
# downcast. 1-D tensors (norms, biases) are handled by the base's ndim rule: they
# are precision sensitive and too small for the bytes to matter.
_COMM_FP32_MARKERS = (
    ".gate.weight",
    "shared_expert_gate",
)
_DEFAULT_ADAMW_MARKERS = (
    "embed_tokens",
    "embedding",
    "lm_head",
    "output_layer",
)
# Class labels keyed by (fp32_compute, comm_critical), matching the table above.
_LABELS = {
    (True, False): "action_expert",
    (True, True): "expert_gate_norm",
    (False, True): "vlm_norm_bias",
    (False, False): "vlm_backbone",
}


class LingbotVlaV2ParameterPolicy(MarkerParameterPolicy):
    """Preserve the validated LingBot replicated-sharded ownership and precision contract.

    This is a plain ``MarkerParameterPolicy`` (a model-side policy describing what
    each parameter *is*). The registry wraps it in the capability-resolving adapter
    once, combining it with the run's collective precision plan -- the model never
    binds runtime precision into the policy itself.
    """

    def __init__(self, model_cfg=None):
        extra_markers = (
            () if model_cfg is None else tuple(getattr(model_cfg, "muon_exclude_name_patterns", ()) or ())
        )
        super().__init__(
            compute_fp32_markers=_ACTION_FP32_MARKERS,
            comm_critical_markers=_COMM_FP32_MARKERS,
            labels=_LABELS,
            force_comm_critical_below_ndim=2,
            adamw_markers=_DEFAULT_ADAMW_MARKERS + extra_markers,
        )


__all__ = ["LingbotVlaV2ParameterPolicy"]
