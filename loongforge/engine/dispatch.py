# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a resolved invocation to its engine entry point."""

from importlib import import_module
import sys

from loongforge.contracts import TrainSpec


_ENTRYPOINTS = {
    "mcore": "loongforge.engine.mcore.entrypoint",
    "torch": "loongforge.engine.torch.entrypoint",
}


def run_train(spec: TrainSpec):
    entrypoint = _ENTRYPOINTS[spec.engine]
    previous = sys.argv
    sys.argv = ["loongforge train", *spec.args]
    try:
        return import_module(entrypoint).main()
    finally:
        sys.argv = previous
