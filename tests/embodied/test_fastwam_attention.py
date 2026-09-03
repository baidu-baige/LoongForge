# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Tests for FastWAM structured attention masks and SDPA fallback."""

import torch
import torch.nn as nn

from loongforge.embodied.model.fastwam.attention import (
    AttentionSegment,
    build_structured_attention_mask,
    run_attention,
    video_attention_segments,
)
from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM
from loongforge.embodied.model.fastwam.mot.idm import FastWAMIDM
from loongforge.embodied.model.fastwam.mot.joint import FastWAMJoint


class _VideoExpertStub(nn.Module):
    def __init__(self, mode="first_frame_causal"):
        super().__init__()
        self.video_attention_mask_mode = mode


def _model_stub(model_type):
    model = model_type.__new__(model_type)
    nn.Module.__init__(model)
    model.video_expert = _VideoExpertStub()
    return model


def test_first_frame_mot_structured_mask_matches_dense_reference():
    video_len, action_len, tokens_per_frame = 12, 3, 4
    total_len = video_len + action_len
    segments = video_attention_segments("first_frame_causal", 0, video_len, tokens_per_frame)
    segments.append(
        AttentionSegment(video_len, total_len, ((0, tokens_per_frame), (video_len, total_len)))
    )
    structured = build_structured_attention_mask(total_len, total_len, segments, torch.device("cpu"))

    expected = torch.zeros((total_len, total_len), dtype=torch.bool)
    expected[:video_len, :video_len] = True
    expected[:tokens_per_frame, tokens_per_frame:video_len] = False
    expected[video_len:, :tokens_per_frame] = True
    expected[video_len:, video_len:] = True
    assert torch.equal(structured.dense, expected)


def test_structured_sdpa_matches_dense_sdpa_forward_and_backward():
    torch.manual_seed(7)
    segments = (
        AttentionSegment(0, 2, ((0, 2),)),
        AttentionSegment(2, 5, ((0, 5),)),
    )
    structured = build_structured_attention_mask(5, 5, segments, torch.device("cpu"))
    inputs = [torch.randn(2, 5, 12, requires_grad=True) for _ in range(3)]

    structured_out = run_attention(*inputs, num_heads=3, attention_mask=structured, backend="auto")
    structured_grads = torch.autograd.grad(structured_out.square().sum(), inputs)

    dense_inputs = [value.detach().clone().requires_grad_(True) for value in inputs]
    dense_out = run_attention(*dense_inputs, num_heads=3, attention_mask=structured.dense, backend="sdpa")
    dense_grads = torch.autograd.grad(dense_out.square().sum(), dense_inputs)

    torch.testing.assert_close(structured_out, dense_out)
    for structured_grad, dense_grad in zip(structured_grads, dense_grads):
        torch.testing.assert_close(structured_grad, dense_grad)


def test_fastwam_variant_masks_preserve_original_visibility():
    video_len, action_len, tokens_per_frame = 12, 3, 4
    total_len = video_len + action_len

    base_mask = _model_stub(FastWAM)._build_mot_attention_mask(
        video_len, action_len, tokens_per_frame, torch.device("cpu")
    ).dense
    expected_base = torch.zeros((total_len, total_len), dtype=torch.bool)
    expected_base[:video_len, :video_len] = True
    expected_base[:tokens_per_frame, tokens_per_frame:video_len] = False
    expected_base[video_len:, :tokens_per_frame] = True
    expected_base[video_len:, video_len:] = True
    assert torch.equal(base_mask, expected_base)

    joint_mask = _model_stub(FastWAMJoint)._build_mot_attention_mask(
        video_len, action_len, tokens_per_frame, torch.device("cpu")
    ).dense
    expected_joint = expected_base.clone()
    expected_joint[video_len:, :video_len] = True
    assert torch.equal(joint_mask, expected_joint)

    idm_mask = _model_stub(FastWAMIDM)._build_teacher_forcing_attention_mask(
        video_len,
        video_len,
        action_len,
        tokens_per_frame,
        tokens_per_frame,
        torch.device("cpu"),
    ).dense
    cond_start, action_start = video_len, video_len * 2
    expected_idm = torch.zeros_like(idm_mask)
    expected_idm[:video_len, :video_len] = expected_base[:video_len, :video_len]
    expected_idm[cond_start:action_start, cond_start:action_start] = expected_base[
        :video_len, :video_len
    ]
    expected_idm[action_start:, cond_start:] = True
    assert torch.equal(idm_mask, expected_idm)
