# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from loongforge.models.foundation.ernie5_5.ernie_ops import (
    Ernie55Multimax,
    Ernie55ResidualSplit,
    Ernie55SiTU,
    vha_postmix,
)
from loongforge.models.foundation.ernie5_5.ernie_attention import (
    _packed_position_ids,
    _packed_window_indices,
    ernie55_indexer_loss,
    ernie55_sparse_attention,
)
from loongforge.models.foundation.ernie5_5.ernie_model import _prepare_state_dict


@pytest.mark.parametrize("extra_state", [None, torch.empty(0, dtype=torch.uint8)])
def test_load_restores_only_empty_extra_state_and_shared_parameters(extra_state):
    class Norm(torch.nn.Linear):
        def get_extra_state(self):
            return extra_state

        def set_extra_state(self, state):
            assert state is None or state.numel() == 0

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.source = torch.nn.Parameter(torch.ones(2))
            self.alias = self.source
            self.norm = Norm(2, 2, bias=False)
            self.mtp_aliases = {"alias": "source"}
            self.register_load_state_dict_pre_hook(_prepare_state_dict)

    model = Module()
    state = {"source": torch.tensor([2.0, 3.0]), "norm.weight": torch.eye(2)}
    model.load_state_dict(state, strict=True)
    assert model.alias is model.source
    assert torch.equal(model.source, state["source"])
    with pytest.raises(RuntimeError, match="norm.weight"):
        model.load_state_dict({"source": state["source"]}, strict=True)
    with pytest.raises(RuntimeError, match="Shared MTP values differ"):
        model.load_state_dict({**state, "alias": torch.zeros(2)}, strict=True)
    with pytest.raises(ValueError, match="assign=False"):
        model.load_state_dict(state, assign=True)


def test_packed_windows_respect_document_boundaries():
    boundaries = torch.tensor([0, 5, 12, 16])
    queries = torch.arange(4, 12)
    positions = _packed_position_ids(boundaries, queries)
    assert positions.tolist() == [4, 0, 1, 2, 3, 4, 5, 6]
    windows = _packed_window_indices(queries, boundaries, 4)
    for query, position, window in zip(queries.tolist(), positions.tolist(), windows):
        expected = list(range(max(query - position, query - 3), query + 1))
        assert window[window >= 0].tolist() == expected


@pytest.mark.parametrize("per_token", [False, True])
def test_indexer_kl_query_slices_sum_to_full_loss(per_token):
    torch.manual_seed(9)
    query = torch.randn(8, 1, 2, 3)
    key = torch.randn(8, 1, 1, 3)
    scores = torch.randn(1, 8, 8, requires_grad=True)
    remote = torch.arange(8)[None, :] < torch.arange(8)[:, None] - 1
    topk = scores.masked_fill(~remote, float("-inf")).topk(3, dim=-1).indices
    full = ernie55_indexer_loss(scores, topk, query, key, remote, 0.5, 0.01, per_token)
    split = sum(
        ernie55_indexer_loss(
            scores[:, start : start + 4],
            topk[:, start : start + 4],
            query[start : start + 4],
            key,
            remote[start : start + 4],
            0.5,
            0.01 / (1 if per_token else 2),
            per_token,
        )
        for start in (0, 4)
    )
    torch.testing.assert_close(split, full)
    torch.testing.assert_close(
        torch.autograd.grad(split, scores, retain_graph=True)[0],
        torch.autograd.grad(full, scores)[0],
    )


def test_indexer_kl_masks_empty_rows_and_trains_only_scores():
    torch.manual_seed(11)
    scores = torch.randn(2, 4, 4, requires_grad=True)
    query = torch.randn(4, 2, 2, 3, requires_grad=True)
    key = torch.randn(4, 2, 1, 3, requires_grad=True)
    remote = torch.arange(4)[None, :] < torch.arange(4)[:, None]
    topk = torch.tensor([[[0, 1], [0, 2], [0, 1], [0, 2]]] * 2)
    actual = ernie55_indexer_loss(scores, topk, query, key, remote, 0.5, 0.01, False)
    expected = scores.sum() * 0
    for batch in range(2):
        for row in range(1, 4):
            selected = topk[batch, row]
            selected = selected[remote[row, selected]]
            teacher = (query[row, batch] @ key[selected, batch, 0].T * 0.5).softmax(-1)
            target = teacher.mean(0).detach()
            expected = (
                expected + (target * (target.log() - scores[batch, row, selected].log_softmax(-1))).sum() * 0.01 / 8
            )
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-5)
    (actual_grad,) = torch.autograd.grad(actual, scores, retain_graph=True)
    (expected_grad,) = torch.autograd.grad(expected, scores)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-5)
    assert torch.count_nonzero(actual_grad[:, 0]) == 0
    assert torch.autograd.grad(actual, (query, key), allow_unused=True) == (None, None)


def test_residual_split_rounds_bf16_gradients_once():
    value = torch.ones(4, dtype=torch.bfloat16, requires_grad=True)
    mapping, aggregate, residual = Ernie55ResidualSplit.apply(value)
    (mapping * 256 + aggregate - residual * 256).float().sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))


def test_sparse_attention_matches_dense_selected_keys_and_gradients():
    torch.manual_seed(7)
    query = torch.randn(3, 2, 2, 4, requires_grad=True)
    key = torch.randn(3, 2, 4, requires_grad=True)
    sink = torch.randn(2, requires_grad=True)
    indices = torch.tensor([[[0, -1], [1, 0], [2, 0]], [[-1, 0], [0, 1], [1, 2]]])
    actual = ernie55_sparse_attention(query, key, sink, indices, 0.5).view_as(query)
    batches = []
    for batch in range(2):
        rows = []
        for row in range(3):
            selected = key[indices[batch, row][indices[batch, row] >= 0], batch]
            logits = query[row, batch] @ selected.T * 0.5
            probabilities = torch.cat((logits, sink[:, None]), dim=-1).softmax(-1)[:, :-1]
            rows.append(probabilities @ selected)
        batches.append(torch.stack(rows))
    expected = torch.stack(batches, dim=1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual_grads = torch.autograd.grad(actual.square().sum(), (query, key, sink), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, sink))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)


def test_situ_is_bounded_and_uses_unclamped_sigmoid():
    activation = Ernie55SiTU(SimpleNamespace(activation_situ_beta=4.0, activation_situ_linear_beta=25.0))
    value = torch.tensor([[100.0, -100.0, 100.0, 100.0]], requires_grad=True)
    result = activation(value)
    torch.testing.assert_close(result[0, 0], torch.tensor(100.0), atol=0.1, rtol=0)
    assert abs(result[0, 1].item()) < 1e-30
    result.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_multimax_identity_initialization_still_trains_scales():
    logits = torch.tensor([[-2.0, -0.5, 0.5, 2.0]], requires_grad=True)
    ranges = torch.zeros(4, requires_grad=True)
    scales = torch.zeros(4, requires_grad=True)
    result = Ernie55Multimax.apply(logits, ranges, scales)
    torch.testing.assert_close(result, logits)
    result.sum().backward()
    torch.testing.assert_close(logits.grad, torch.ones_like(logits))
    torch.testing.assert_close(scales.grad, torch.tensor([2.5, 2.5, 4.25, 4.25]))
    torch.testing.assert_close(ranges.grad, torch.zeros_like(ranges))


def test_vha_routes_input_heads_to_output_heads():
    value = torch.tensor([[[2.0], [5.0]]])
    u = torch.tensor([[1.0], [0.0]])
    v = torch.tensor([[0.0], [3.0]])
    torch.testing.assert_close(vha_postmix(value, u, v), torch.tensor([[[2.0], [11.0]]]))
