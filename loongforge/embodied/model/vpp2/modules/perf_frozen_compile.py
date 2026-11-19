# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Incremental compilation of frozen CLIP/VAE, outside the ActionDiT graph."""
import atexit
import functools
import json
import logging
import os
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_records = []


def _native_graph(original, name):
    from torch.utils._pytree import tree_flatten, tree_unflatten

    state = {}
    record = dict(scope=name, mode="native_graph", backend="native-cuda-graph",
                  calls=0, capture_count=0, fallback="none-fail-closed", original_ops=True)
    _records.append(record)

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Frozen native graphs require uncaptured no_grad execution")
        leaves, spec = tree_flatten((args, kwargs))
        signature = tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))
                          if isinstance(t, torch.Tensor) else (type(t), t) for t in leaves)
        if not state:
            static = [t.detach().to(device=torch.cuda.current_device()).clone()
                      if isinstance(t, torch.Tensor) else t for t in leaves]
            static_args, static_kwargs = tree_unflatten(static, spec)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()
            record["capture_start_unix"] = time.time()
            # Warm the exact original operators on the capture stream. No
            # optimizer update, RNG draws or values are reused across samples.
            with torch.cuda.stream(stream):
                for _ in range(3):
                    original(*static_args, **static_kwargs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = original(*static_args, **static_kwargs)
            if not isinstance(output, torch.Tensor):
                raise ValueError("Frozen native graphs require one tensor output")
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            record["capture_end_unix"] = time.time()
            record["capture_seconds"] = record["capture_end_unix"] - record["capture_start_unix"]
            record["capture_count"] = 1
            state.update(graph=graph, output=output, static=static, spec=spec, signature=signature)
            logger.info("VPP2 frozen %s native graph captured seconds=%.4f", name, record["capture_seconds"])
        if signature != state["signature"] or spec != state["spec"]:
            raise ValueError("Frozen graph input signature changed; disable the fixed-shape optimization")
        for target, source in zip(state["static"], leaves):
            if isinstance(source, torch.Tensor):
                target.copy_(source)
        state["graph"].replay()
        record["calls"] += 1
        return state["output"].clone()

    return forward


def enable(model, cfg):
    if cfg.perf_compile_vae_encode != "off":
        if cfg.perf_compile_vae_encode != "native_graph":
            raise ValueError("Frozen VAE compilation only supports native_graph")
        if model.vae.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.vae.parameters()):
            raise ValueError("VAE compilation requires the original frozen FP32 encoder")
        model.vae.model.encode = _native_graph(model.vae.model.encode, "vae.encode")
    if cfg.perf_compile_clip != "off":
        if cfg.perf_compile_clip != "native_graph":
            raise ValueError("Frozen CLIP compilation only supports native_graph")
        if model.clip_encoder is None or model.clip_encoder.training or any(
            p.requires_grad or p.dtype != torch.float16 for p in model.clip_encoder.parameters()
        ):
            raise ValueError("CLIP compilation requires the original frozen FP16 encoder")
        # Preserve the original FP16 resize/normalization by capturing only
        # the visual encoder forward.
        model.clip_encoder.visual.forward = _native_graph(
            model.clip_encoder.visual.forward, "clip.visual.forward",
        )


def _write_health():
    root = os.environ.get("OUTPUT_DIR")
    if root and _records:
        path = Path(root) / ("frozen_compile_health_rank" + os.environ.get("RANK", "0") + ".json")
        path.write_text(json.dumps(dict(rank=int(os.environ.get("RANK", "0")), regions=_records),
                                   indent=2, default=str) + "\n")


atexit.register(_write_health)
