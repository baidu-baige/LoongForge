# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import argparse
from unittest.mock import Mock

import pytest

from loongforge.models import catalog
from tests.test_engine_migration import _load_functions


def test_vlm_cli_keeps_master_options():
    namespace = _load_functions(
        "loongforge/training/vlm/args.py", {"add_vlm_args"},
        argparse=argparse, constants=catalog, get_support_model_archs=Mock(return_value=[]),
    )
    parser = namespace["add_vlm_args"](argparse.ArgumentParser(allow_abbrev=False))
    defaults = parser.parse_args([])
    assert defaults.use_thumbnail is False
    assert defaults.normalize_type == "imagenet"
    assert defaults.patch_size == 14
    args = parser.parse_args(["--use_thumbnail", "--normalize_type", "clip", "--patch-size", "8"])
    assert args.use_thumbnail is True
    assert args.normalize_type == "clip"
    assert args.patch_size == 8


@pytest.mark.parametrize(("flag", "field"), [
    ("--lora-alpha", "lora_alpha"), ("--lora-dim", "lora_dim"),
])
def test_lora_cli_options_remain_available(flag, field):
    namespace = _load_functions(
        "loongforge/training/megatron_args.py", {"_add_freeze_and_init_args"},
        argparse=argparse,
    )
    parser = namespace["_add_freeze_and_init_args"](argparse.ArgumentParser(allow_abbrev=False))
    assert getattr(parser.parse_args([]), field) is None
    assert getattr(parser.parse_args([flag, "8"]), field) == 8
