# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints write aliased modules and tied tensors once (issue #314)."""

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from safetensors.torch import load_file

import loongforge.embodied.distributed.checkpoint as ck
from loongforge.embodied.distributed import DistributedContext
from loongforge.embodied.distributed.checkpoint import (
    MODEL_STATE_SCHEMA,
    build_alias_map,
    canonical_state_dict,
)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = nn.Linear(4, 4)
        self.experts = nn.ModuleDict({"a": self.expert})
        self.embed = nn.Embedding(6, 4)
        self.head = nn.Linear(4, 6, bias=False)
        self.head.weight = self.embed.weight
        base = torch.arange(8.0)
        self.register_buffer("first", base[:4])
        self.register_buffer("second", base[4:])


class _Model(nn.Module):
    """Mirrors FastWAM/PI05: ``core.mot`` re-registers experts, wrappers alias both."""

    def __init__(self):
        super().__init__()
        self.core = _Inner()
        self.dit = self.core.experts
        self.alias = self.core


CANONICAL = {
    "core.expert.weight", "core.expert.bias", "core.embed.weight",
    "core.first", "core.second",
}


def _args(save_format):
    return SimpleNamespace(
        use_lora=False, save_format=save_format, async_save=False,
        save_training_state=True, save_processor_config=False,
    )


def _save(model, tmp_path, save_format):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ck.save_checkpoint(model, optimizer, None, 1, str(tmp_path), DistributedContext(), _args(save_format))
    return os.path.join(tmp_path, "steps_1")


def test_alias_map_resolves_chains_and_keeps_distinct_views():
    alias_map = build_alias_map(_Model())

    assert set(_Model().state_dict()) - set(alias_map) == CANONICAL
    assert alias_map["dit.a.weight"] == "core.expert.weight"
    assert alias_map["alias.experts.a.bias"] == "core.expert.bias"
    assert alias_map["core.head.weight"] == "core.embed.weight"
    assert alias_map["alias.head.weight"] == "core.embed.weight"
    assert "core.second" not in alias_map


def test_canonical_view_round_trips_strict_and_is_temporary():
    source, target = _Model(), _Model()
    with canonical_state_dict(source):
        state = source.state_dict()
    assert set(state) == CANONICAL

    with canonical_state_dict(target):
        target.load_state_dict(state, strict=True)
    torch.testing.assert_close(target.dit["a"].weight, source.core.expert.weight)
    assert target.alias.head.weight is target.core.embed.weight
    assert len(source.state_dict()) > len(CANONICAL)


def test_dcp_checkpoint_is_canonical_and_resumes(tmp_path):
    source = _Model()
    path = _save(source, tmp_path, "dcp")

    keys = dcp.FileSystemReader(os.path.join(path, "dcp")).read_metadata().state_dict_metadata
    assert {k[len("model."):] for k in keys if k.startswith("model.")} == CANONICAL
    with open(os.path.join(path, "resume_meta.json")) as f:
        assert json.load(f)["model_state_schema"] == MODEL_STATE_SCHEMA

    target = _Model()
    ck.resume_training_state(
        target, torch.optim.SGD(target.parameters(), lr=0.1), None, path, DistributedContext(),
    )
    torch.testing.assert_close(target.state_dict(), source.state_dict())


def test_legacy_duplicate_dcp_checkpoint_still_resumes(tmp_path):
    source = _Model()
    dcp_dir = tmp_path / "dcp"
    dcp.save({"model": source.state_dict()}, storage_writer=dcp.FileSystemWriter(str(dcp_dir)))

    target = _Model()
    ck.resume_training_state(target, None, None, str(tmp_path), DistributedContext())
    torch.testing.assert_close(target.state_dict(), source.state_dict())


def test_safetensors_checkpoint_is_canonical_and_loads(tmp_path):
    source = _Model()
    path = _save(source, tmp_path, "safetensors")

    assert set(load_file(os.path.join(path, "model.safetensors"))) == CANONICAL
    target = ck.load_pretrained(_Model(), path, DistributedContext())
    torch.testing.assert_close(target.state_dict(), source.state_dict())


def test_pt_checkpoint_is_canonical_and_loads(tmp_path):
    source = _Model()
    path = _save(source, tmp_path, "pt")

    saved = torch.load(os.path.join(path, "pytorch_model.pt"), map_location="cpu", weights_only=True)
    assert set(saved) == CANONICAL
    target = ck.load_pretrained(_Model(), path, DistributedContext())
    torch.testing.assert_close(target.state_dict(), source.state_dict())
    # Tied parameter stays the same object after loading the canonical key only.
    assert target.core.head.weight is target.core.embed.weight

