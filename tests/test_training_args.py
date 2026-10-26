# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge CLI definitions: argument shapes and options removed in favour of the model config."""

import argparse

import pytest

from loongforge.training.megatron_args import add_training_args


def _parse(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser = add_training_args(parser)
    return parser.parse_args(argv)


def test_latent_patch_size_takes_three_ints():
    assert tuple(_parse([]).latent_patch_size) == (1, 1, 1)
    assert _parse(["--latent-patch-size", "1", "2", "2"]).latent_patch_size == [1, 2, 2]
    with pytest.raises(SystemExit):
        _parse(["--latent-patch-size", "122"])


@pytest.mark.parametrize("flag", ["--use-thumbnail", "--use_thumbnail"])
def test_use_thumbnail_spellings(flag):
    assert _parse([flag]).use_thumbnail is True


@pytest.mark.parametrize("flag", ["--normalize-type", "--normalize_type"])
def test_normalize_type_spellings(flag):
    assert _parse([flag, "clip"]).normalize_type == "clip"


@pytest.mark.parametrize("flag", ["--patch-size", "--lora-alpha", "--lora-dim"])
def test_options_owned_by_the_model_config_are_gone(flag):
    with pytest.raises(SystemExit):
        _parse([flag, "8"])
