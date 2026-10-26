# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""FastWAM sampler registry hook.

FastWAM has no bespoke index ordering, so the *default* path is reused verbatim
(``default_sampler_builder``): with the default ``--distributed-sampler-mode=cyclic``
that is ``torchdata.StatefulDistributedSampler``, whose permutation is
``randperm(seed + epoch)`` and therefore reshuffles every epoch.

This module exists to make that epoch behaviour *controllable*, because standalone
FastWAM (the reference implementation we reproduce) never advances its sampler
epoch: ``ResumableEpochSampler.__iter__`` seeds with
``seed + self.epoch + self.epoch_offset``, and the only caller of
``set_epoch_offset`` is its resume path -- so during normal training the seed stays
``seed`` and **every epoch reuses the same permutation**. ``--sampler-epoch-mode=fixed``
reproduces that.

Why a registered builder instead of patching the default one: the failure mode this
replaces was a switch written into the *wrong* sampler class, which silently did
nothing. FastWAM receives ``StatefulDistributedSampler`` (not ``_BlockShardSampler``)
because ``default_sampler_builder`` branches on ``distributed_sampler_mode``, whose
default is ``"cyclic"``. Registering here bypasses that branch entirely, so the knob
provably applies to the class actually in use.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from typing import Any, Iterator

from torch.utils.data import Sampler

from loongforge.embodied.data.datasets.sampler_builder import (
    SamplerBuilderContext,
    default_sampler_builder,
    register_sampler_builder,
)

# Announced (pid, class, mode) triples, so the construction line is printed once.
_ANNOUNCED: set = set()

# Epoch fingerprints already printed, so a pinned epoch prints once per process
# (that repetition is itself the evidence the order is fixed).
_FINGERPRINTS: set = set()

_FINGERPRINT_LEN = 256


def _fingerprint(sampler: Sampler) -> str:
    """Short digest of the permutation this sampler would yield now."""
    head = list(itertools.islice(iter(sampler), _FINGERPRINT_LEN))
    return hashlib.sha1(",".join(map(str, head)).encode()).hexdigest()[:12]


def _order_audit(sampler: Sampler) -> None:
    """Print a fingerprint of the current permutation, once per (pid, epoch).

    With the epoch pinned (mode=fixed) every iteration yields the same digest and
    this prints exactly once -- i.e. the training log itself proves the sampling
    order did not change. With mode=advance it prints one differing line per epoch.
    """
    epoch = getattr(sampler, "epoch", None)
    key = (os.getpid(), type(sampler).__name__, str(epoch))
    if key in _FINGERPRINTS:
        return
    _FINGERPRINTS.add(key)
    print(
        f"[fastwam-sampler] order epoch={epoch} sampler={type(sampler).__name__} "
        f"seed={getattr(sampler, 'seed', None)} rank={getattr(sampler, 'rank', None)}"
        f"/{getattr(sampler, 'num_replicas', None)} order_sha1={_fingerprint(sampler)}",
        flush=True,
    )


class _FixedEpochSampler(Sampler[int]):
    """Pin ``inner``'s epoch to 0 so every epoch reuses one permutation.

    Wraps rather than subclasses because the concrete class is chosen by
    ``default_sampler_builder`` (``StatefulDistributedSampler`` for the default
    ``cyclic`` mode, ``_BlockShardSampler`` for ``block``). The stateful API is
    delegated explicitly: ``StatefulDataLoader`` persists the sampler via
    ``state_dict`` / ``load_state_dict``, and dropping that would silently break
    in-epoch resume.
    """

    def __init__(self, inner: Sampler[int]) -> None:
        self._inner = inner
        if hasattr(inner, "set_epoch"):
            inner.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:  # noqa: ARG002 - epoch intentionally ignored
        """Ignore the trainer's epoch advance and keep the inner sampler at 0."""
        if hasattr(self._inner, "set_epoch"):
            self._inner.set_epoch(0)

    def __iter__(self) -> Iterator[int]:
        _order_audit(self._inner)
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)

    def state_dict(self) -> Any:
        """Delegate stateful resume to the wrapped sampler."""
        return self._inner.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        """Delegate stateful resume to the wrapped sampler."""
        self._inner.load_state_dict(state_dict)

    def __getattr__(self, item: str) -> Any:
        # Anything else StatefulDataLoader / the trainer may look up.
        return getattr(self.__dict__["_inner"], item)


@register_sampler_builder("fastwam")
def build_fastwam_sampler(context: SamplerBuilderContext) -> Sampler | None:
    """Build the FastWAM training sampler.

    ``--sampler-epoch-mode=advance`` (default) returns exactly what the default
    builder returns -- same class, same arguments, so existing runs stay
    bit-comparable. ``=fixed`` returns that same sampler wrapped so its epoch can
    never advance past 0.
    """
    sampler = default_sampler_builder(context)
    if sampler is None:  # iterable dataset or single-process: DataLoader shuffles
        return None

    mode = getattr(context.training_args, "sampler_epoch_mode", "advance")
    key = (os.getpid(), type(sampler).__name__, mode)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print(
            f"[fastwam-sampler] mode={mode} sampler={type(sampler).__name__} "
            f"seed={context.seed} shuffle={context.shuffle} "
            f"world_size={context.ctx.world_size} rank={context.ctx.rank} "
            f"sampler_epoch_mode={mode!r}",
            flush=True,
        )

    if mode != "fixed":
        return sampler

    if not hasattr(sampler, "set_epoch"):
        raise ValueError(
            f"--sampler-epoch-mode=fixed requires a sampler with set_epoch, got "
            f"{type(sampler).__name__}. Set --distributed-sampler-mode=cyclic "
            f"(the default) or use a sampler that supports epoch control."
        )
    return _FixedEpochSampler(sampler)
