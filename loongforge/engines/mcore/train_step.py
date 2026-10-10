# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os

import torch

from megatron.core import mpu
from megatron.core.utils import StragglerDetector
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine, RerunDataIterator, RerunState
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.training import get_timers
from megatron.training.utils import (
    unwrap_model,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)

from loongforge import constants
from loongforge.engines.mcore import get_args
from loongforge.engines.mcore.parallel.batch_broadcast import (
    gather_variable_shape_embeddings,
    scatter_variable_shape_embeddings,
)
from loongforge.engines.mcore.parallel.dp_balance.train_hooks import train_step_decorator

stimer = StragglerDetector()


_PRINTED_MODEL_INPUT_EXAMPLE = False

_SAMPLE_DUMP_ANSI_COLOR = {
    "T": "\033[92m",  # bright green — trainable (labels != IGNORE_INDEX)
    "C": "\033[93m",  # yellow       — context (attended but not trained)
    "P": "\033[90m",  # grey         — padded / masked
}
_SAMPLE_DUMP_ANSI_RESET = "\033[0m"
# PLACEHOLDER_TRAIN_STEP_DUMP


def dump_model_input_example_once(
    tokens, labels, attn_mask, cu_lengths=None, packed_seq_params=None
):
    """Dump the exact tensors flowing into ``model(...)``.

    Runs once per process on (rank0, tp0, cp0). Decodes ``tokens`` with each
    token segment colored by its role: green = trainable target (the model
    is asked to predict this token), grey = padded (``attn_mask`` truthy),
    yellow = context (attended but not trained). Consecutive same-role
    tokens are decoded as one segment so multi-byte BPE pieces render
    correctly. These are the same tensor objects the caller passes to
    ``model(...)`` — any disalignment would mean someone mutated them in
    between.

    Labels here are already left-shifted by the collator for next-token
    prediction (``labels[i]`` is the target the model produces from
    ``tokens[:i+1]``, i.e. equal to ``tokens[i+1]``). So a token at
    position ``i`` is trainable iff ``labels[i-1] != IGNORE_INDEX``;
    position 0 is never a trainable target.

    Pass ``cu_lengths`` / ``packed_seq_params`` only when the trainer packs
    samples; for non-packed paths leave them as ``None``.
    """
    global _PRINTED_MODEL_INPUT_EXAMPLE
    if _PRINTED_MODEL_INPUT_EXAMPLE:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    else:
        if int(os.environ.get("RANK", "0")) != 0:
            return

    try:
        if mpu.get_tensor_model_parallel_rank() != 0:
            return
        if mpu.get_context_parallel_rank() != 0:
            return
    except Exception:
        pass

    _PRINTED_MODEL_INPUT_EXAMPLE = True

    from loongforge.engines.mcore.global_vars import get_tokenizer as get_loongforge_tokenizer
    from loongforge.constants import IGNORE_INDEX
# PLACEHOLDER_TRAIN_STEP_DUMP2

    tokenizer = get_loongforge_tokenizer()

    def to_cpu_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if x is None:
            return None
        return list(x)

    def decode(ids):
        if not ids:
            return ""
        try:
            return tokenizer.detokenize(ids, skip_special_tokens=False)
        except TypeError:
            return tokenizer.detokenize(ids)

    tok = tokens[0] if tokens.dim() == 2 else tokens
    lab = labels[0] if labels.dim() == 2 else labels

    cu = cu_lengths
    if torch.is_tensor(cu):
        cu = cu[0] if cu.dim() == 2 else cu
    cu_list = to_cpu_list(cu) or []

    tok_list = to_cpu_list(tok)
    lab_list = to_cpu_list(lab)

    am = attn_mask
    am_list = None
    if torch.is_tensor(am):
        if am.dim() == 4:
            # Megatron sometimes hands a (1,1,seq,seq) causal-style mask; we
            # only care about per-position padding here, so collapse it.
            am_view = am[0, 0].diagonal()
        elif am.dim() == 2:
            am_view = am[0]
        elif am.dim() == 1:
            am_view = am
        else:
            am_view = None
        if am_view is not None:
            am_list = am_view.detach().cpu().to(torch.bool).tolist()

    seq_len = len(tok_list)
    # `labels` is the next-token-prediction target the collator produced,
    # i.e. labels[i] is the target after consuming tokens[0..i]. So token
    # position `i` is a trainable target iff labels[i-1] != IGNORE_INDEX.
    # Position 0 has no predecessor and is never a target.
    trainable_total = sum(
        1 for v in lab_list[:-1] if v != IGNORE_INDEX
    ) if lab_list else 0
# PLACEHOLDER_TRAIN_STEP_DUMP3

    def role_at(i):
        if i > 0 and lab_list[i - 1] != IGNORE_INDEX:
            return "T"
        if am_list is not None and i < len(am_list) and am_list[i]:
            return "P"
        return "C"

    header = [
        "===== model-input example (forward_step entry, same tensor as model(...)) =====",
        f"tokens.shape={tuple(tokens.shape)} labels.shape={tuple(labels.shape)} "
        f"attn_mask.shape={tuple(attn_mask.shape) if torch.is_tensor(attn_mask) else None}",
        f"local_seq_len={seq_len} trainable_tokens={trainable_total}",
        f"cu_lengths={cu_list}",
        f"packed_seq_params={packed_seq_params}",
        f"tp_rank={mpu.get_tensor_model_parallel_rank()} "
        f"cp_rank={mpu.get_context_parallel_rank()}/{mpu.get_context_parallel_world_size()} "
        f"pp_rank={mpu.get_pipeline_model_parallel_rank()}",
        f"[mask_legend] {_SAMPLE_DUMP_ANSI_COLOR['T']}T=trainable target{_SAMPLE_DUMP_ANSI_RESET} "
        f"{_SAMPLE_DUMP_ANSI_COLOR['C']}C=context attended{_SAMPLE_DUMP_ANSI_RESET} "
        f"{_SAMPLE_DUMP_ANSI_COLOR['P']}P=padded/masked{_SAMPLE_DUMP_ANSI_RESET}",
    ]

    trainable_ids = [v for v in lab_list if v != IGNORE_INDEX]

    # Decode contiguous same-role runs as single segments so multi-byte BPE
    # pieces render correctly. Splitting per-token would break UTF-8 mid-codepoint.
    colored_segments = []
    if seq_len:
        run_start = 0
        run_role = role_at(0)
        for i in range(1, seq_len):
            r = role_at(i)
            if r != run_role:
                colored_segments.append(
                    f"{_SAMPLE_DUMP_ANSI_COLOR[run_role]}"
                    f"{decode(tok_list[run_start:i])}"
                    f"{_SAMPLE_DUMP_ANSI_RESET}"
                )
                run_start = i
                run_role = r
        colored_segments.append(
            f"{_SAMPLE_DUMP_ANSI_COLOR[run_role]}"
            f"{decode(tok_list[run_start:seq_len])}"
            f"{_SAMPLE_DUMP_ANSI_RESET}"
        )
    colored_decoded = "".join(colored_segments)

    body = [
        "[input_ids]",
        str(tok_list),
        "[decoded_input | colored by role]",
        colored_decoded,
        "[decoded_trainable_labels]",
        decode(trainable_ids),
    ]

    print("\n".join(header + body + [
        "===== end model-input example =====",
    ]), flush=True)
@train_step_decorator
def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    forward_backward_func,
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    if args.enable_full_hetero_dp:
        import itertools
        import copy
        from loongforge.engines.mcore.initialize import (
            get_num_micro_batches_per_decoder_dp,
            get_num_real_micro_batches_per_decoder_dp,
        )
        from loongforge.training.methods.pretrain_vlm import (
            get_batch, get_embedding_list,
            get_visual_pos_masks_list, get_deepstack_visual_embeds_list,
            get_deepstack_grad_list, _create_mock_batch,
            get_encoder_data_iterator,
        )

        num_microbatch, encoder_rounds = get_num_micro_batches_per_decoder_dp()
        num_real_microbatch = get_num_real_micro_batches_per_decoder_dp()
        unwrapped_model = unwrap_model(model[0])

        encoder_iter = get_encoder_data_iterator()
        if encoder_iter is not None:
            if isinstance(data_iterator, list):
                data_iterator = [RerunDataIterator(data_iterator[0])] + data_iterator[1:]
            else:
                data_iterator = RerunDataIterator(data_iterator)
        else:
            if isinstance(data_iterator, list):
                first_iter, backup_iter = itertools.tee(data_iterator[0])
                data_iterator = [RerunDataIterator(first_iter)] + data_iterator[1:]
            else:
                data_iterator, backup_iter = itertools.tee(data_iterator)
                data_iterator = RerunDataIterator(data_iterator)

        pp_layer = mpu.get_pipeline_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        model_size = num_microbatch // encoder_rounds

        all_raw_batches = None
        if encoder_iter is None:
            all_raw_batches = [next(backup_iter) for _ in range(num_real_microbatch)]

        batch_list = []
        last_real_batch = None
        has_any_real = (pp_layer * tp_size < num_real_microbatch)
        embedding_list = get_embedding_list()
        visual_pos_masks_list = get_visual_pos_masks_list()
        deepstack_visual_embeds_list = get_deepstack_visual_embeds_list()
        for round in range(encoder_rounds):
            batch_list.clear()
            front = pp_layer * tp_size + round * model_size
            for tp_idx in range(tp_size):
                global_mb_idx = front + tp_idx
                if global_mb_idx >= num_real_microbatch:
                    # This microbatch is beyond real data — use mock batch
                    if not batch_list and last_real_batch is None:
                        # No real batch seen yet — need a reference for mock shape
                        if encoder_iter is not None and has_any_real:
                            last_real_batch = get_batch(encoder_iter)
                        elif all_raw_batches is not None:
                            last_real_batch = copy.deepcopy(get_batch(iter([all_raw_batches[num_real_microbatch - 1]])))
                        else:
                            iter_arg = (
                                data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
                            )
                            last_real_batch = get_batch(iter_arg)
                    mock_ref = batch_list[-1] if batch_list else last_real_batch
                    batch_list.append(_create_mock_batch(mock_ref))
                else:
                    if encoder_iter is not None:
                        batch = get_batch(encoder_iter)
                    else:
                        batch = copy.deepcopy(get_batch(iter([all_raw_batches[global_mb_idx]])))
                    last_real_batch = batch
                    batch_list.append(batch)

            input_embeds_list = []
            for i in range(tp_size):
                input_embeds = unwrapped_model.encoder_model.text_forward(
                    batch_list[i]["tokens"],
                    batch_list[i]["position_ids"]
                )
                input_embeds_list.append(input_embeds)

            batch_id = mpu.get_tensor_model_parallel_rank()
            (
                local_images,
                local_image_grid_thw,
                local_pixel_values_videos,
                local_video_grid_thw,
                local_input_ids,
                local_attn_mask,
                local_labels,
                local_cu_lengths,
                local_max_lengths,
                local_position_ids,
                local_loss_mask,
                local_packed_seq_params,
            ) = batch_list[batch_id].values()

            (
                combined_embeddings,
                decode_input,
                visual_pos_masks,
                deepstack_visual_embeds,
            ) = unwrapped_model.encoder_model(
                input_ids=local_input_ids,
                position_ids=local_position_ids,
                image_inputs=dict(
                    images=local_images,
                    image_grid_thw=local_image_grid_thw,
                ) if local_images is not None else None,
                video_inputs=dict(
                    pixel_values_videos=local_pixel_values_videos,
                    video_grid_thw=local_video_grid_thw,
                ) if local_pixel_values_videos is not None else None,
                inference_params=None,
                inputs_embeds=input_embeds_list[batch_id],
                enable_encoder_hetero_dp=True,
            )

            unwrapped_model.vit_contexts.setdefault(round, {
                "local_embedding": combined_embeddings,
                "grads": None,
                "local_visual_pos_masks": visual_pos_masks,
                "local_deepstack_visual_embeds": deepstack_visual_embeds,
                "local_deepstack_visual_embeds_grads": None,
            })

            embedding_list.append(
                gather_variable_shape_embeddings(
                    combined_embeddings, 
                    group=mpu.get_model_parallel_group()
                )
            )

            if visual_pos_masks is not None:
                visual_pos_masks_list.append(
                    gather_variable_shape_embeddings(
                        visual_pos_masks, 
                        group=mpu.get_model_parallel_group()
                    )
                )
            else:
                visual_pos_masks_list.append(None)

            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds_list.append([
                    gather_variable_shape_embeddings(embed, group=mpu.get_model_parallel_group())
                    for embed in deepstack_visual_embeds
                ])
                get_deepstack_grad_list().append(
                    [[None] * model_size for _ in range(len(deepstack_visual_embeds))]
                )
            else:
                deepstack_visual_embeds_list.append(None)
                get_deepstack_grad_list().append(None)
        batch_list.clear()

        # Offload gathered embeddings to CPU (only rank 0 holds them)
        if args.full_hetero_dp_cpu_offload:
            from loongforge.training.methods.pretrain_vlm import get_cpu_offload_manager
            from loongforge.engines.mcore.full_hetero_cpu_offload import offload_list_items
            _offload_mgr = get_cpu_offload_manager()
            _local_rank_for_offload = torch.distributed.get_rank(mpu.get_model_parallel_group())
            if _local_rank_for_offload == 0:
                for _round_idx in range(encoder_rounds):
                    if embedding_list[_round_idx] is not None:
                        offload_list_items(
                            _offload_mgr, embedding_list[_round_idx],
                            f"emb_r{_round_idx}"
                        )
                    if visual_pos_masks_list[_round_idx] is not None:
                        offload_list_items(
                            _offload_mgr, visual_pos_masks_list[_round_idx],
                            f"vpm_r{_round_idx}"
                        )
                    if deepstack_visual_embeds_list[_round_idx] is not None:
                        for _layer_idx, _layer_embeds in enumerate(
                            deepstack_visual_embeds_list[_round_idx]
                        ):
                            offload_list_items(
                                _offload_mgr, _layer_embeds,
                                f"ds_r{_round_idx}_l{_layer_idx}"
                            )
            _offload_mgr.wait_all_offloads()

        _encoder_bucket_groups = set()
        if args.overlap_grad_reduce and isinstance(model[0], DDP):
            _ddp_model = model[0]
            for param in unwrapped_model.encoder_model.parameters():
                if param in _ddp_model.param_to_bucket_group:
                    _encoder_bucket_groups.add(_ddp_model.param_to_bucket_group[param])
            for bg in _encoder_bucket_groups:
                bg.is_last_microbatch = False

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        adjust_tensor_shapes_fn = None
        # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
        # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
        # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            for optim_instance in optimizer.chained_optimizers:
                if isinstance(optim_instance, DistributedOptimizer):
                    optim_instance._copy_main_params_to_param_buffer()

        # Forward pass.
        tmp_num_microbatches = get_num_microbatches()
        tmp_seq_length = args.seq_length
        if args.enable_chunkpipe:
            tmp_seq_length = args.chunksize
            if args.training_phase != "sft":
                # Pretrain: DataLoader produces full sequences, ChunkDataIterator
                # splits them, so num_microbatches must be inflated.
                num_chunks = args.seq_length // args.chunksize
                tmp_num_microbatches *= num_chunks
            # SFT: DataLoader already produces chunk-level micro-batches,
            # num_microbatches is already correct.

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=tmp_num_microbatches,
            seq_length=tmp_seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )

    if args.enable_full_hetero_dp:
        from loongforge.training.methods.pretrain_vlm import (
            get_grad_list, get_deepstack_grad_list, clear_full_hetero_info
        )
        from loongforge.engines.mcore.initialize import (
            get_num_micro_batches_per_decoder_dp,
            get_num_real_micro_batches_per_decoder_dp,
            get_model_size,
        )

        num_microbatch, encoder_rounds = get_num_micro_batches_per_decoder_dp()
        num_real_microbatch = get_num_real_micro_batches_per_decoder_dp()
        model_size = get_model_size()
        grad_list = get_grad_list()

        # Reload offloaded grads from CPU before reshaping
        if args.full_hetero_dp_cpu_offload:
            from loongforge.training.methods.pretrain_vlm import get_cpu_offload_manager
            _reload_mgr = get_cpu_offload_manager()
            _reload_mgr.wait_all_offloads()
            for _gi in range(len(grad_list)):
                if grad_list[_gi] is None:
                    _reloaded = _reload_mgr.reload(f"grad_{_gi}")
                    if _reloaded is not None:
                        _reload_mgr.reload_sync(f"grad_{_gi}")
                        grad_list[_gi] = _reloaded

        # Reshape grad_list into per-round lists and pad with zero grads for
        # mock positions so that scatter_variable_shape_embeddings receives
        # exactly model_size entries (one per rank in the model-parallel group).
        reshaped_grad_list = []
        real_idx = 0
        for r in range(encoder_rounds):
            round_grads = []
            for pos in range(model_size):
                global_mb_idx = r * model_size + pos
                if global_mb_idx < num_real_microbatch and real_idx < len(grad_list):
                    round_grads.append(grad_list[real_idx])
                    real_idx += 1
                else:
                    # Zero grad placeholder for mock positions
                    ref = grad_list[0] if grad_list else None
                    if ref is not None:
                        round_grads.append(torch.zeros_like(ref))
                    else:
                        round_grads.append(None)
            reshaped_grad_list.append(round_grads)

        local_model = unwrap_model(model[0])
        _rsm = get_rerun_state_machine()
        _prev_rsm_state = _rsm.state
        if _rsm.state == RerunState.NOT_RUNNING_YET:
            _rsm.state = RerunState.INITIAL_RUN

        try:
            if args.overlap_grad_reduce and _encoder_bucket_groups:
                for bg in _encoder_bucket_groups:
                    bg.is_last_microbatch = False

            for round in range(encoder_rounds):
                if args.overlap_grad_reduce and _encoder_bucket_groups:
                    for bg in _encoder_bucket_groups:
                        bg.params_with_grad = set()

                src_rank = 0
                group = mpu.get_model_parallel_group()
                local_rank = torch.distributed.get_rank(group=group)
                ctx = local_model.vit_contexts[round]
                ctx["grads"] = scatter_variable_shape_embeddings(
                    reshaped_grad_list[round] if local_rank == src_rank else None,
                    local_embedding_ref=ctx["local_embedding"],
                    group=mpu.get_model_parallel_group()
                )

                deepstack_grads_for_round = get_deepstack_grad_list()[round]
                if deepstack_grads_for_round is not None:
                    ctx["local_deepstack_visual_embeds_grads"] = []
                    for i in range(len(ctx["local_deepstack_visual_embeds"])):
                        # Pad None entries (mock positions) with zero tensors for scatter
                        if local_rank == src_rank:
                            padded_ds_grads = []
                            for g in deepstack_grads_for_round[i]:
                                if g is None:
                                    padded_ds_grads.append(torch.zeros_like(ctx["local_deepstack_visual_embeds"][i]))
                                else:
                                    padded_ds_grads.append(g)
                        else:
                            padded_ds_grads = None
                        ctx["local_deepstack_visual_embeds_grads"].append(
                            scatter_variable_shape_embeddings(
                                padded_ds_grads,
                                local_embedding_ref=ctx["local_deepstack_visual_embeds"][i],
                                group=mpu.get_model_parallel_group()
                            )
                        )

                backward_tensors = [ctx["local_embedding"]]
                backward_grads = [ctx["grads"]]
                if ctx["local_deepstack_visual_embeds_grads"] is not None:
                    backward_tensors += ctx["local_deepstack_visual_embeds"]
                    backward_grads += ctx["local_deepstack_visual_embeds_grads"]

                torch.autograd.backward(
                    tensors=backward_tensors,
                    grad_tensors=backward_grads,
                    retain_graph=False,
                )
                del local_model.vit_contexts[round]

            if args.overlap_grad_reduce and _encoder_bucket_groups:
                for bg in _encoder_bucket_groups:
                    bg.is_last_microbatch = True
                    bg.start_grad_sync()
                for bg in _encoder_bucket_groups:
                    bg.finish_grad_sync()
        finally:
            _rsm.state = _prev_rsm_state

        for _round_key in list(local_model.vit_contexts.keys()):
            del local_model.vit_contexts[_round_key]
        
        clear_full_hetero_info()

    should_checkpoint, should_exit, exit_code = (
        rerun_state_machine.should_checkpoint_and_exit()
    )
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
            num_zeros_in_grad
        )

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            # Special handling for total_inputs which may be int type
            if key == "total_inputs":
                total = sum(x[key] for x in losses_reduced)
                loss_reduced[key] = total
                continue

            val = [x[key].view(-1) for x in losses_reduced]
            if val[0].numel() == 2:
                if args.enable_chunkpipe:
                    # Skip microbatches with zero tokens (e.g. chunkpipe chunks that
                    # fall entirely in the prompt region where loss_mask=0) to avoid
                    # 0/0=NaN in per-token loss computation.
                    val = [v for v in val if v[1].item() > 0]
                    if len(val) == 0:
                        loss_reduced[key] = torch.tensor(0.0)
                        continue
                if (
                    args.training_phase == constants.TrainingPhase.SFT
                    and not args.legacy_reporting_loss_reduction
                ):
                    if args.calculate_per_token_loss:
                        # SFT ChunkPipe: log as ΣS/Σn to align with token-equal-weight gradient
                        val = torch.vstack(val).sum(dim=0)
                        torch.distributed.all_reduce(
                            val,
                            group=mpu.get_data_parallel_group(with_context_parallel=True),
                        )
                        loss_reduced[key] = val[0] / val[1]
                    else:
                        # SFT non-ChunkPipe: normalize per sample (mean of S_i/n_i),
                        # in mcore the normalization happens on micro batch instead of global
                        val = torch.vstack(val)
                        val = val[:, 0] / val[:, 1]
                        val = val.mean()
                        torch.distributed.all_reduce(
                            val,
                            group=mpu.get_data_parallel_group(with_context_parallel=True),
                        )
                        val /= torch.distributed.get_world_size(
                            group=mpu.get_data_parallel_group(with_context_parallel=True)
                        )
                        loss_reduced[key] = val
                else:
                    # there is one dict per microbatch. in new reporting, we average
                    # over the total number of tokens across the global batch.
                    val = torch.vstack(val).sum(dim=0)
                    torch.distributed.all_reduce(
                        val,
                        group=mpu.get_data_parallel_group(with_context_parallel=True),
                    )
                    loss_reduced[key] = val[0] / val[1]
            elif val[0].numel() == 1:
                # For the SFT chunkpipe per-sample loss path, each micro-batch
                # reports D * S_{g,k}/(N_g * G_total), where:
                #   - G_total is the cross-rank source-group count for the step
                #     (identical on all DP ranks, supplied by the sampler)
                #   - D = data_parallel_world_size, pre-compensating the 1/D
                #     normalization that the DP all-reduce below introduces
                # Summing local chunks gives D * (1/G_total) * sum_{g in rank}
                # sum_k S_{g,k}/N_g; the subsequent DP+CP all-reduce-sum then
                # /world_size cancels the D factor and aggregates across ranks,
                # recovering the step-level per-sample loss
                # (1/G_total) * sum_g sum_k S_{g,k}/N_g — the canonical metric
                # to log regardless of how groups are distributed among ranks.
                # Other paths keep legacy mean-over-micro-batches behavior.
                if (
                    args.enable_chunkpipe
                    and getattr(args, 'sft_chunkpipe_mode', False)
                    and not args.calculate_per_token_loss
                ):
                    val = torch.cat(val).sum()
                else:
                    val = torch.cat(val).mean()
                # since we remove the dpcp allreduce in loss func
                torch.distributed.all_reduce(
                    val, group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                val /= torch.distributed.get_world_size(
                    group=mpu.get_data_parallel_group(with_context_parallel=True)
                )
                loss_reduced[key] = val
            else:
                raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
    return (
        {},
        skipped_iter,
        should_checkpoint,
        should_exit,
        exit_code,
        grad_norm,
        num_zeros_in_grad,
    )
