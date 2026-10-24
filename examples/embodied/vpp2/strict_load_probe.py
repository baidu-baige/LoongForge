#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Strict P2-2 VPP2 checkpoint and auxiliary-precision probe.

This intentionally exercises the same ``build meta -> materialize -> strict
load`` path used by the embodied trainer, but emits a compact JSON report for
the evidence ledger.  It performs no optimization step.
"""

from __future__ import annotations

import argparse
import json

from loongforge.embodied.model.vpp2.modeling_configuration_vpp2 import VPP2ModelConfig
from loongforge.embodied.model.vpp2.modeling_vpp2 import VPP2Model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = VPP2ModelConfig(init_on_meta=True)
    model = VPP2Model.from_pretrained(cfg)
    report = model.load_pretrained(args.video_checkpoint or cfg.video_checkpoint_path, device=args.device)
    report["meta_tensors_after_load"] = sum(parameter.is_meta for parameter in model.parameters())
    report["modules"] = {}
    for name, module in (
        ("video_expert", model.video_expert),
        ("action_expert", model.action_expert),
        ("proprio_encoder", model.proprio_encoder),
        ("vae", model.vae),
        ("clip_encoder", model.clip_encoder),
    ):
        if module is None:
            report["modules"][name] = None
            continue
        parameters = list(module.parameters())
        report["modules"][name] = {
            "dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
            "training": module.training,
            "trainable_parameters": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        }
    model.train()
    report["after_train_call"] = {
        name: module.training
        for name, module in (
            ("video_expert", model.video_expert),
            ("vae", model.vae),
            ("clip_encoder", model.clip_encoder),
            ("action_expert", model.action_expert),
        )
        if module is not None
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
