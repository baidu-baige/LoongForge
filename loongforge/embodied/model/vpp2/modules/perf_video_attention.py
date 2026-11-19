# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Default-off all-True video mask elision, retaining memory-efficient SDPA."""
import atexit
import json
import os
from pathlib import Path

import torch

_subjects = []  # Scalar diagnostics only; never retain model/tensor ownership.


def prepare(mot, tokens, all_true, layer_count, max_layers, feature_callback):
    expert = mot.mixtures["video"]
    if (not all_true or layer_count != 40 or max_layers is not None
            or feature_callback is not None
            or torch.is_grad_enabled() or torch.is_autocast_enabled() or expert.training
            or tokens.requires_grad or not tokens.is_cuda or tokens.dtype != torch.bfloat16
            or tuple(tokens.shape) != (2, 1960, 5120) or not tokens.is_contiguous()
            or mot.perf_video_execution != "eager"
            or mot.perf_action_execution != "native_graph" or not mot.perf_video_rope_fused
            or mot.use_video_tokens or mot.use_fixed_video_layers is not None):
        raise ValueError("Mask elision requires complete frozen BF16 eager-video/native-action with a structural all-True mask")
    if mot.perf_video_skip_uninitialized_fill:
        # The combined path may run only inside the guarded serial producer.
        # In particular, reject direct eager-prefill calls that bypass finally
        # restoration of the process-global allocation policy.
        if (not mot.__dict__.get("_perf_video_fill_active", False)
                or torch.utils.deterministic.fill_uninitialized_memory):
            raise ValueError("Mask/allocation bundle requires the active scoped video producer")
    record = mot.__dict__.get("_perf_video_attention_mask_record")
    if record is None:
        if any(p.requires_grad or p.device != tokens.device for p in expert.parameters()):
            raise ValueError("Mask elision requires frozen video parameters on the producer device")
        if any(not hasattr(block.norm1, "_perf_native_bf16_ln")
               or not hasattr(block.self_attn.norm_q, "_perf_rms_pointwise") for block in expert.blocks):
            raise ValueError("Mask elision requires the native-LN/RMS parent")
        record = dict(prefill_calls=0, attention_calls=0, layers_per_prefill=40,
                      requested_backend="SDPA_NO_MASK_DISPATCH", fallback="none-fail-closed",
                      mask_proof="source-constructor-bidirectional-or-three-frame", frozen=True)
        mot.__dict__["_perf_video_attention_mask_record"] = record
        _subjects.append(record)
    record["prefill_calls"] += 1
    return record


def _health():
    root = os.environ.get("OUTPUT_DIR")
    if root and _subjects:
        path = Path(root) / ("video_attention_health_rank" + os.environ.get("RANK", "0") + ".json")
        path.write_text(json.dumps(dict(rank=int(os.environ.get("RANK", "0")), subjects=_subjects,
                                       scope="frozen video self-attention mask only"), indent=2) + "\n")


atexit.register(_health)
