# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""The diagnostic probes still run, on every provider.

``tools/embodied/arch_probe.py`` reaches into the architecture the way a
debugging session does: provider registry, shrunken geometry, entry stage,
``state_dict`` keys. That makes it the first thing to break when any of those
move, and a broken diagnostic is worse than none, because it gets reached for
exactly when something else is already wrong.

These cases are deliberately thin. They assert that each probe runs and prints
the one line a reader would look for; what the numbers *mean* is the parity
harnesses' job.
"""

from __future__ import annotations

import pytest

from tools.embodied import arch_probe

MODEL_TYPES = ("vpp2", "fastwam", "internw0_delta")

# InternW0-Delta's ``track_align`` raises ``NotImplementedError`` at build
# time on purpose (its decoder belongs to a later lane), so any probe that
# assembles the model has to be told to leave it out.
UNBUILDABLE_HEADS = {"internw0_delta": ["track_align"]}


def run(capsys, probe: str, model_type: str) -> str:
    """Invoke the CLI the way a shell would and return what it printed.

    Args:
        capsys: Pytest capture fixture.
        probe: Subcommand name.
        model_type: Provider name.

    Returns:
        Captured stdout.
    """
    argv = [probe, model_type]
    for head in UNBUILDABLE_HEADS.get(model_type, ()):
        argv += ["--drop-head", head]
    assert arch_probe.main(argv) == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_keys_probe_prints_the_geometry_and_the_state_dict(capsys,
                                                               model_type):
    """Every stream, condition and head is named, and keys carry shapes."""
    spec = arch_probe.drop_heads(arch_probe.tiny_spec(model_type),
                                 UNBUILDABLE_HEADS.get(model_type, []))
    out = run(capsys, "keys", model_type)
    assert "latent_grid=" in out
    for condition in spec.conditions:
        assert f"condition {condition.name}:" in out
    for head in spec.heads:
        assert f"head {head.name}:" in out
    for stream in spec.streams:
        assert f"stream {stream.name}:" in out
        assert f"blocks.{stream.name}." in out
    assert out.count("\n") > 100


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_entry_probe_prints_one_line_per_stream_and_slot(capsys,
                                                             model_type):
    """Hidden, modulation and RoPE for each stream, plus the context slots."""
    spec = arch_probe.drop_heads(arch_probe.tiny_spec(model_type),
                                 UNBUILDABLE_HEADS.get(model_type, []))
    out = run(capsys, "entry", model_type)
    for stream in spec.streams:
        assert f"{stream.name}: hidden=" in out
    assert out.count("rope=") == len(spec.streams)
    assert "slot text:" in out
    assert "layout=[('video', 0," in out


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_timestep_probe_follows_the_streams_own_declaration(capsys,
                                                                model_type):
    """Two outcomes, and which one appears is the spec's decision.

    A provider whose video stream declares no ``token_timestep`` has nothing
    to expand, and the probe has to say so rather than fail; vpp2 is that
    case today. For the other two the probe's whole point is the agreement
    with both legacy formulas, so a ``False`` there is a finding even though
    the probe exits 0.
    """
    scope = arch_probe.tiny_spec(model_type).stream("video").token_timestep
    out = run(capsys, "timestep", model_type)
    if scope is None:
        assert "declares no token_timestep" in out
        return
    assert f"clean_prefix={scope.clean_prefix} " in out
    assert "matches_fastwam=True" in out
    assert "matches_internw0_delta=True" in out
    assert "per-frame input agrees with per-sample input=True" in out
