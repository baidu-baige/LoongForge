# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for releasing allocator-cached memory after model loading."""

from __future__ import annotations

import torch

from loongforge.embodied.eval.servers import loongforge_server


def test_release_cached_memory_calls_empty_cache_on_cuda(monkeypatch) -> None:
    calls = {"empty_cache": 0, "reserved": 0}

    def fake_reserved() -> int:
        calls["reserved"] += 1
        return 16 * 2**30 if calls["reserved"] == 1 else 8 * 2**30

    def fake_empty_cache() -> None:
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", fake_reserved)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    loongforge_server._release_loader_cached_memory()

    assert calls["empty_cache"] == 1
    assert calls["reserved"] == 2


def test_release_cached_memory_is_a_noop_without_cuda(monkeypatch) -> None:
    calls = {"empty_cache": 0}

    def fake_empty_cache() -> None:
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    loongforge_server._release_loader_cached_memory()

    assert calls["empty_cache"] == 0
