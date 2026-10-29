# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, CPU-portable initialization/RNG helpers for VPP2 A/B runs.

Only the four projection tensors excluded by action_init.pt and the proprio
linear are stored here. The 1010 backbone tensors still load strictly from the
released initializer. This is a labelled comparison initialization, NOT the
unseeded random initialization of a historical official run.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

FORMAT = "vpp2_alignment_init_v1"


def seed_training(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_alignment_init(path, *, seed=42, action_dim=7, hidden_dim=512, proprio_dim=8, text_dim=4096):
    """Generate only small fp32-initialized linears, then cast to model bf16."""
    # CPU RNG is isolated from model/data setup, and CUDA is never initialized.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        layers = {
            "action_encoder": torch.nn.Linear(action_dim, hidden_dim),
            "head": torch.nn.Linear(hidden_dim, action_dim),
            "proprio_encoder": torch.nn.Linear(proprio_dim, text_dim),
        }
        payload = {
            "format": FORMAT,
            "seed": int(seed),
            "modules": {name: layer.to(torch.bfloat16).state_dict() for name, layer in layers.items()},
        }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_alignment_init(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("Not a VPP2 comparison initialization")
    modules = {
        "action_encoder": model.action_expert.action_encoder,
        "head": model.action_expert.head,
        "proprio_encoder": model.proprio_encoder,
    }
    if set(payload["modules"]) != set(modules):
        raise ValueError("Comparison initializer module coverage differs")
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name], strict=True)


def parameter_fingerprint(model):
    """Hash names/shapes/dtypes/bytes, not torch.save container metadata."""
    digest = hashlib.sha256()
    count = numel = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        tensor = param.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        count += 1
        numel += tensor.numel()
    return {"sha256": digest.hexdigest(), "tensors": count, "numel": numel}
