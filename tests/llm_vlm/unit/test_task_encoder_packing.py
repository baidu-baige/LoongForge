# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :meth:`BaseTaskEncoder.pack_selected_samples`.

The packing method concatenates the token/label/attention tensors of a group
of :class:`BaseTaskSample` into one :class:`BaseTaskSamplePacked`, flattens the
media lists and records the sub-sample boundaries in ``cu_lengths``. The
encoder is created with ``__new__`` so the heavy ``__init__`` (args, tokenizer,
packer) never runs -- the method only reads ``self.args.seq_length``.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from loongforge.data.vlm.base_task_encoder import (
    BaseTaskEncoder,
    BaseTaskSample,
    BaseTaskSamplePacked,
    _format_packed_sample_overflow_error,
)
from loongforge.data.vlm.flavors import ENERGON_LT_7

# Energon < 7.0 declares ``__subflavor__`` as a dataclass field on the base
# ``Sample`` class, so the LoongForge sample constructors accept (and require)
# it as a kwarg; newer versions dropped the field again. The factory forwards
# it only when the installed version actually declares it.
_ACCEPTS_SUBFLAVOR = "__subflavor__" in inspect.signature(
    BaseTaskSamplePacked.__init__
).parameters


def make_encoder(seq_length: int = 64) -> BaseTaskEncoder:
    """Build an encoder carrying only the args the packing method needs."""
    encoder = BaseTaskEncoder.__new__(BaseTaskEncoder)
    encoder.args = SimpleNamespace(seq_length=seq_length)
    return encoder


def make_sample(
    key: str,
    tokens,
    labels=None,
    attn_mask=None,
    *,
    imgs=None,
    num_tiles=None,
    pixel_values_videos=None,
) -> BaseTaskSample:
    """Build a BaseTaskSample holding real 1-D tensors."""
    kwargs = dict(
        __key__=key,
        __restore_key__=(key,),
        __subflavors__={"sample_type": "unit_test"},
        tokens=torch.tensor(tokens, dtype=torch.long),
        total_len=len(tokens),
        labels=None if labels is None else torch.tensor(labels, dtype=torch.long),
        attn_mask=None if attn_mask is None else torch.tensor(attn_mask, dtype=torch.bool),
        imgs=imgs,
        num_tiles=[] if num_tiles is None else num_tiles,
        pixel_values_videos=pixel_values_videos,
    )
    if _ACCEPTS_SUBFLAVOR:
        kwargs["__subflavor__"] = None
    return BaseTaskSample(**kwargs)


def test_pack_single_sample_roundtrips_fields():
    encoder = make_encoder(seq_length=16)
    img = torch.zeros(3, 4, 4)
    video = torch.zeros(2, 3)
    sample = make_sample(
        "sample-1",
        tokens=[1, 2, 3, 4],
        labels=[-100, -100, 5, 6],
        attn_mask=[True, True, True, True],
        imgs=[img],
        num_tiles=[2],
        pixel_values_videos=[video],
    )

    packed = encoder.pack_selected_samples([sample])

    assert isinstance(packed, BaseTaskSamplePacked)
    assert packed.__key__ == "sample-1"
    assert packed.__restore_key__ == ()
    assert packed.max_length == 4
    assert packed.__subflavors__ is sample.__subflavors__
    assert torch.equal(packed.tokens, sample.tokens)
    assert torch.equal(packed.labels, sample.labels)
    assert torch.equal(packed.attn_mask, sample.attn_mask)
    assert len(packed.imgs) == 1 and packed.imgs[0] is img
    assert len(packed.pixel_values_videos) == 1 and packed.pixel_values_videos[0] is video
    assert packed.num_tiles == [2]
    assert torch.equal(packed.cu_lengths, torch.tensor([0, 4], dtype=torch.int32))


def test_pack_multiple_samples_concatenate_in_order():
    encoder = make_encoder(seq_length=64)
    img_a = torch.full((1, 2, 2), 1.0)
    img_b = torch.full((1, 2, 2), 2.0)
    img_c = torch.full((1, 2, 2), 3.0)
    video_a = torch.zeros(1, 2)
    samples = [
        make_sample(
            "a", [1, 2], [9, 9], [True, True],
            imgs=[img_a], num_tiles=[1], pixel_values_videos=[video_a],
        ),
        make_sample(
            "b", [3, 4, 5], [8, 8, 8], [True, True, True],
            imgs=[img_b, img_c], num_tiles=[2, 1],
        ),
        make_sample("c", [6], [7], [True], imgs=None, num_tiles=[]),
    ]

    packed = encoder.pack_selected_samples(samples)

    assert packed.__key__ == "a,b,c"
    assert torch.equal(packed.tokens, torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long))
    assert torch.equal(packed.labels, torch.tensor([9, 9, 8, 8, 8, 7], dtype=torch.long))
    assert torch.equal(packed.attn_mask, torch.tensor([True] * 6, dtype=torch.bool))
    assert torch.equal(packed.cu_lengths, torch.tensor([0, 2, 5, 6], dtype=torch.int32))
    assert packed.max_length == 3
    assert packed.num_tiles == [1, 2, 1]
    assert all(p is q for p, q in zip(packed.imgs, [img_a, img_b, img_c]))
    assert len(packed.pixel_values_videos) == 1
    assert packed.pixel_values_videos[0] is video_a


def test_pack_exact_fit_does_not_overflow():
    encoder = make_encoder(seq_length=5)
    samples = [
        make_sample("a", [1, 2], [1, 2], [True, True]),
        make_sample("b", [3, 4, 5], [3, 4, 5], [True] * 3),
    ]

    packed = encoder.pack_selected_samples(samples)

    assert torch.equal(packed.cu_lengths, torch.tensor([0, 2, 5], dtype=torch.int32))


def test_pack_overflow_raises_value_error_with_context():
    encoder = make_encoder(seq_length=10)
    samples = [
        make_sample("ok-1", [1, 2], [1, 2], [True, True]),
        make_sample("ok-2", [3, 4, 5], [3, 4, 5], [True] * 3),
        make_sample("overflow", [6] * 6, [6] * 6, [True] * 6),
    ]

    with pytest.raises(
        ValueError,
        match=(
            r"current_length=5, next_sample_key=overflow, next_sample_len=6, "
            r"would_be_length=11, num_samples=3"
        ),
    ):
        encoder.pack_selected_samples(samples)


def test_pack_text_only_samples_have_empty_media_lists():
    encoder = make_encoder(seq_length=8)
    sample = make_sample("text-only", [1, 2, 3], [1, 2, 3], [True] * 3)

    packed = encoder.pack_selected_samples([sample])

    assert packed.imgs == []
    assert packed.pixel_values_videos == []


@pytest.mark.parametrize("needs_subflavor", [False, True])
def test_pack_energon_subflavor_branch(needs_subflavor):
    """Both ``ENERGON_LT_7`` branches must produce identical packs."""
    if needs_subflavor != _ACCEPTS_SUBFLAVOR:
        pytest.skip(
            "installed energon "
            + ("requires" if _ACCEPTS_SUBFLAVOR else "rejects")
            + " the __subflavor__ kwarg"
        )
    encoder = make_encoder(seq_length=8)
    sample = make_sample("s1", [1, 2], [1, 2], [True, True])

    with patch(
        "loongforge.data.vlm.cookers.ENERGON_LT_7",
        needs_subflavor,
    ):
        packed = encoder.pack_selected_samples([sample])

    assert torch.equal(packed.tokens, sample.tokens)
    assert torch.equal(packed.cu_lengths, torch.tensor([0, 2], dtype=torch.int32))
    if needs_subflavor:
        assert packed.__subflavor__ is None


def test_overflow_error_formatter_reports_sample_context():
    img = torch.zeros(1, 2, 2)
    samples = [
        make_sample("s1", [1] * 2, [1] * 2, [True] * 2, imgs=[img], num_tiles=[1]),
        make_sample("s2", [2] * 3, [2] * 3, [True] * 3),
    ]

    msg = _format_packed_sample_overflow_error(
        samples, packing_seq_len=10, current_length=4, next_sample=samples[1]
    )

    assert "exceeds the maximum sequence length of 10" in msg
    assert "current_length=4" in msg
    assert "next_sample_key=s2" in msg
    assert "next_sample_len=3" in msg
    assert "would_be_length=7" in msg
    assert "num_samples=2" in msg
    assert "omitted_samples" not in msg
    assert "'key': 's1'" in msg
    assert "'total_len': 2" in msg
    assert "'cumulative_len': 2" in msg
    assert "'tokens_shape': (2,)" in msg
    assert "'num_images': 1" in msg
    assert "'num_videos': 0" in msg


def test_overflow_error_formatter_truncates_very_long_lists():
    samples = [make_sample(f"s{i}", [1] * 3, [1] * 3, [True] * 3) for i in range(10)]

    msg = _format_packed_sample_overflow_error(
        samples, packing_seq_len=100, current_length=0, next_sample=samples[0]
    )

    assert "omitted_samples=2" in msg
    assert msg.count("'key':") == 8