# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""giga_datasets compatibility package for giga_brain07: lazily re-exports the relocated dataset reader, collator and samplers under their original names so existing imports keep resolving to the same objects."""

__version__ = '1.1.0'


def _preload_torchcodec_decoder_before_decord() -> None:
    try:
        # Decord can load FFmpeg symbols before torchcodec and break AV1 stream
        # detection in torchcodec. Preloading keeps torchcodec usable when this
        # package later imports decord-backed helpers.
        from torchcodec.decoders import VideoDecoder as _VideoDecoder  # noqa: F401
    except Exception:
        return


_preload_torchcodec_decoder_before_decord()

# ``utils`` has no back-dependency on this package's reader/collator, so it is
# safe to import eagerly (and is needed for the lerobot availability gate).
from .utils import is_lerobot_available

# The dataset-reading framework + collator were relocated into the giga_brain07
# ``transforms`` package (single merged reader module
# ``transforms.giga_brain_lerobot_dataset`` + ``transforms.giga_brain_collator``)
# to mirror the xvla-style layout. They are re-exported here under their original
# names so every existing ``from giga_datasets import ...`` keeps resolving to
# the exact same objects (logic unchanged); only their on-disk location moved.
#
# Resolution is lazy (PEP 562 ``__getattr__``): the relocated reader imports
# ``giga_datasets.utils`` back, so importing those modules eagerly here would
# create an import cycle. Deferring keeps the original eager-import *effects*
# (e.g. ``register_dataset`` populating the registry before ``load_dataset`` is
# used) without the cycle.
_PKG = "loongforge.embodied.data.datasets.giga_brain07"

_READER_NAMES = frozenset(
    {
        "BaseDataset",
        "BaseProcessor",
        "ConcatDataset",
        "Dataset",
        "WeightedConcatDataset",
        "load_config",
        "load_dataset",
        "register_dataset",
        "LeRobotDataset",
    }
)
_SAMPLER_NAMES = frozenset(
    {
        "AspectRatioSampler",
        "BucketBatchSampler",
        "BucketSampler",
        "GigaBrainSampler",
        "SpecialDatasetSampler",
    }
)

__all__ = [
    "GigaBrainCollator",
    "is_lerobot_available",
    *sorted(_READER_NAMES),
    *sorted(_SAMPLER_NAMES),
]


def _ensure_readers() -> None:
    """Import the relocated reader module so its ``@register_dataset``
    decorators populate the registry (matches the original eager behaviour)."""
    import importlib

    importlib.import_module(_PKG + ".transforms.giga_brain_lerobot_dataset")


def __getattr__(name):
    import importlib

    if name == "GigaBrainCollator":
        return getattr(importlib.import_module(_PKG + ".transforms.giga_brain_collator"), "GigaBrainCollator")
    if name in _READER_NAMES:
        _ensure_readers()
        if name == "LeRobotDataset" and not is_lerobot_available():
            raise AttributeError(name)
        module = importlib.import_module(_PKG + ".transforms.giga_brain_lerobot_dataset")
        return getattr(module, name)
    if name in _SAMPLER_NAMES:
        return getattr(importlib.import_module(__name__ + ".samplers"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
