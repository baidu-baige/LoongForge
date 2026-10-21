# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Full-hetero-DP state is one object owned by the training method."""

from loongforge.engines.megatron.distributed import hetero_dp
from loongforge.engines.megatron.distributed.hetero_dp import HeteroStepContext


def _context():
    resets = []
    ctx = HeteroStepContext(
        get_batch=lambda it: None,
        create_mock_batch=lambda: None,
        reset_step_state=lambda: resets.append(1),
        offload_enabled=False,
    )
    return ctx, resets


def test_end_step_clears_step_state_and_keeps_run_state():
    ctx, resets = _context()
    iterator = iter([1, 2])
    ctx.encoder_data_iterator = iterator
    offload = ctx.offload
    for items in (ctx.embedding_list, ctx.grad_list, ctx.visual_pos_masks_list,
                  ctx.deepstack_visual_embeds_list, ctx.deepstack_grad_list):
        items.append(object())

    ctx.end_step()

    assert resets == [1]
    assert ctx.embedding_list == []
    assert ctx.grad_list == []
    assert ctx.visual_pos_masks_list == []
    assert ctx.deepstack_visual_embeds_list == []
    assert ctx.deepstack_grad_list == []
    assert ctx.offload is offload
    assert ctx.encoder_data_iterator is iterator


def test_contexts_are_independent():
    first, _ = _context()
    second, _ = _context()
    first.embedding_list.append(object())
    assert second.embedding_list == []


def test_no_module_level_step_state():
    for name in ("embedding_list", "grad_list", "visual_pos_masks_list", "deepstack_visual_embeds_list",
                 "deepstack_grad_list", "_cpu_offload_manager", "_encoder_data_iterator", "_batch_fns",
                 "set_batch_fns", "clear_full_hetero_info"):
        assert not hasattr(hetero_dp, name), name
