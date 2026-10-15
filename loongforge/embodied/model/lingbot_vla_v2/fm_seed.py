# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Seeded flow-matching draw for reproducible parity across harnesses.

LoongForge makes the flow-matching ``noise`` / ``time`` a pure function of
``(seed, step, rank)`` so the draw is immune to how much *global* CUDA RNG the
surrounding setup consumed. The trainer sets a process-global step seed via
:func:`seed_fm_step` (which calls :func:`set_fm_step_seed`); the model's FM forward
reads it back via :func:`get_fm_step_seed` and draws ``noise`` then ``time``
from a fresh generator seeded with it.

On the upstream model these are wired in through the FM-seed runtime shim in
``upstream_adapter``; upstream itself has no equivalent and draws off the global
RNG. Keep the draw order (noise first, then the two Beta uniforms) identical to
preserve bit-for-bit parity.
"""

import torch

_FM_STEP_SEED = None


def set_fm_step_seed(seed):
    """Set the process-global flow-matching step seed (``None`` clears it)."""
    global _FM_STEP_SEED
    _FM_STEP_SEED = None if seed is None else int(seed)


def get_fm_step_seed():
    """Return the current flow-matching step seed, or ``None`` when unset."""
    return _FM_STEP_SEED


def seed_fm_step(seed, completed_steps, local_rank):
    """Seed the flow-matching noise/time draw per step and per rank.

    Without this the draw comes off the global CUDA RNG, which makes it a
    function of how much RNG the setup consumed (teachers, custom_dp optimizer,
    torch.compile) -- so the same batch and the same weights give a different
    fm loss in a different trainer, and every rank draws the *same* noise and
    the same timesteps because all ranks share one seed (base_trainer seeds
    every rank identically on purpose). The seed formula is the benchmark
    lingbot-vla-v2 one (tasks/vla/train_lingbotvla.py), including the rank
    term that breaks that symmetry:

        seed + step * 1_000_003 + local_rank * 97

    ``completed_steps`` is incremented after the step, so +1 matches the
    upstream ``global_step``, which is incremented before the forward.
    """
    step = int(completed_steps) + 1
    set_fm_step_seed(int(seed) + step * 1_000_003 + int(local_rank) * 97)


def sample_beta(alpha, beta, bsize, device, generator=None):
    """Draw ``bsize`` Beta(alpha, beta) samples.

    The optional ``generator`` makes flow-matching time sampling reproducible
    across trainers that differ in CUDA-RNG consumption before the model
    forward.
    """
    gamma1 = torch.rand((bsize,), device=device, generator=generator).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device, generator=generator).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)
