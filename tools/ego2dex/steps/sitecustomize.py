# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims loaded before the legacy Dyn-HaMR imports."""

from __future__ import annotations

import sys
import types
import inspect

import numpy as np

# chumpy 0.70 predates Python 3.11 and NumPy 2.x. This module is imported by
# Python before Dyn-HaMR loads MANO pickles, so install the aliases early.
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
for name, value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, name):
        setattr(np, name, value)

try:
    from human_body_prior.tools import model_loader
except Exception:
    model_loader = None

if model_loader is not None and not hasattr(model_loader, "load_model"):

    def load_model(*args, **kwargs):
        raise RuntimeError(
            "The legacy human_body_prior load_model API is unavailable; "
            "install a compatible release for run_prior=True."
        )

    model_loader.load_model = load_model

try:
    import human_body_prior.models  # noqa: F401
except ModuleNotFoundError:
    models = types.ModuleType("human_body_prior.models")
    models.__path__ = []
    sys.modules["human_body_prior.models"] = models

if "human_body_prior.models.vposer_model" not in sys.modules:
    vposer_module = types.ModuleType("human_body_prior.models.vposer_model")

    class VPoser:
        pass

    vposer_module.VPoser = VPoser
    sys.modules["human_body_prior.models.vposer_model"] = vposer_module

# Matplotlib 3.9 renamed pyplot.boxplot(labels=...) to tick_labels=...
# Dyn-HaMR's optimizer still uses the former spelling when saving loss plots.
try:
    import matplotlib.pyplot as _plt

    _boxplot = _plt.boxplot

    def _boxplot_legacy_compatible(*args, **kwargs):
        if "labels" in kwargs and "tick_labels" not in kwargs:
            kwargs["tick_labels"] = kwargs.pop("labels")
        return _boxplot(*args, **kwargs)

    _plt.boxplot = _boxplot_legacy_compatible
except Exception:
    pass
