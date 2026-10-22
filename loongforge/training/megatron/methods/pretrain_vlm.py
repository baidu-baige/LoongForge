# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Pretrain traniner for omni models"""

import torch
from functools import partial
import copy

from megatron.training import get_timers

from typing import List
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.core import parallel_state


from transformers import DataCollatorForSeq2Seq

from loongforge import constants
from loongforge.engines.megatron import get_args, get_chat_template, get_model_config, get_tokenizer

from loongforge.engines.megatron.train_step import dump_model_input_example_once
from loongforge.training.megatron.runner import pretrain
from loongforge.training.megatron.data_padding import (
    llm_pad_to_multiple_of,
    vlm_packed_padding,
)
from loongforge.data.llm.dataloader import (
    build_sft_data_collator,
    build_sft_cyclic_iterators,
    load_pretokenized_dataset,
)
from loongforge.data.vlm.dataloader import (
    get_train_dataset,
    get_val_dataset,
    get_train_loader,
)
from loongforge.data.vlm import build_task_encoder
from loongforge.data.vlm.collator import VLMPretrainCollator
from loongforge.engines.megatron.checkpointing import load_dataloader_state
from loongforge.training.megatron.methods.utils.data_configs import sft_loader_config, vlm_data_config


from loongforge.models.multimodal.multimodal_model_provider import (
    omni_model_provider,
)
from loongforge.engines.megatron.distributed import hetero_dp
from loongforge.engines.megatron.distributed.context_parallel import get_batch_on_this_cp_rank
from loongforge.engines.megatron.distributed.loss_reduction import default_loss_func
from loongforge.engines.megatron.initialize import (
    change_parallel_state,
    get_encoder_dp_size,
    is_mock_microbatch,
)

stimer = StragglerDetector()


def _batch_has_non_dummy_value(data, key):
    """True iff `data[key]` carries non-dummy content."""
    v = data.get(key) if data is not None else None
    if torch.is_tensor(v):
        return v.numel() > 1
    if isinstance(v, (list, tuple)):
        return bool(v)
    return v is not None


def get_batch_on_this_tp_rank(data_iterator):
    """Get the current micro-batch on this rank."""
    if data_iterator is not None and mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        # Check if iterator is exhausted (data is None)
        if data is None:
            raise StopIteration("Data iterator exhausted. Check if eval_iters exceeds validation dataset size.")
    else:
        data = None

    tokens = tensor_parallel.broadcast_data(["tokens"], data, torch.int64)["tokens"]
    labels = tensor_parallel.broadcast_data(["labels"], data, torch.int64)["labels"]
    cu_lengths = tensor_parallel.broadcast_data(["cu_lengths"], data, torch.int32)["cu_lengths"]
    max_lengths = tensor_parallel.broadcast_data(["max_lengths"], data, torch.int32)["max_lengths"]
    position_ids = tensor_parallel.broadcast_data(["position_ids"], data, torch.int64)["position_ids"]
    loss_mask = tensor_parallel.broadcast_data(["loss_mask"], data, torch.int64)["loss_mask"]
    attn_mask = tensor_parallel.broadcast_data(["attn_mask"], data, torch.bool)["attn_mask"]

    # Rank 0 probes the batch; broadcast a 2-bit flag so every TP rank agrees
    # before the optional image/video broadcasts.
    tp_src = mpu.get_tensor_model_parallel_src_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    if mpu.get_tensor_model_parallel_rank() == 0:
        modality_flag = torch.tensor(
            [int(_batch_has_non_dummy_value(data, "imgs")),
             int(_batch_has_non_dummy_value(data, "pixel_values_videos"))],
            dtype=torch.int32, device="cuda",
        )
    else:
        modality_flag = torch.zeros(2, dtype=torch.int32, device="cuda")
    torch.distributed.broadcast(modality_flag, src=tp_src, group=tp_group)
    has_image = bool(modality_flag[0].item())
    has_video = bool(modality_flag[1].item())

    images = None
    image_grid_thw = None
    pixel_values_videos = None
    video_grid_thw = None
    if has_image:
        images = tensor_parallel.broadcast_data(["imgs"], data, torch.float32)["imgs"]
        image_grid_thw = tensor_parallel.broadcast_data(
            ["image_grid_thw"], data, torch.int32
        )["image_grid_thw"]
    if has_video:
        pixel_values_videos = tensor_parallel.broadcast_data(
            ["pixel_values_videos"], data, torch.float32
        )["pixel_values_videos"]
        video_grid_thw = tensor_parallel.broadcast_data(
            ["video_grid_thw"], data, torch.int32
        )["video_grid_thw"]
    cu_lengths_cpu = cu_lengths.clone()
    tokens = tokens.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    cu_lengths = cu_lengths.cuda(non_blocking=True)
    max_lengths = max_lengths.cuda(non_blocking=True)
    position_ids = position_ids.cuda(non_blocking=True)
    loss_mask = loss_mask.cuda(non_blocking=True)
    attn_mask = attn_mask.cuda(non_blocking=True)

    packed_seq_params = None
    if cu_lengths.shape != torch.Size([1, 1]):
        assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_lengths[0],
            cu_seqlens_kv=cu_lengths[0],
            cu_seqlens_cpu=cu_lengths_cpu[0],
            max_seqlen_q=max_lengths[0].item(),
            max_seqlen_kv=max_lengths[0].item(),
        )

    batch = {
        "images": images.cuda(non_blocking=True) if images is not None else None,
        "image_grid_thw": (
            image_grid_thw.cuda(non_blocking=True)
            if image_grid_thw is not None
            else None
        ),
        "pixel_values_videos": (
            pixel_values_videos.cuda(non_blocking=True)
            if pixel_values_videos is not None
            else None
        ),
        "video_grid_thw": (
            video_grid_thw.cuda(non_blocking=True)
            if video_grid_thw is not None
            else None
        ),
        "tokens": tokens,
        "attn_mask": attn_mask,
        "labels": labels,
        "cu_lengths": cu_lengths,
        "max_lengths": max_lengths,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "packed_seq_params": packed_seq_params,
    }

    return batch


def get_batch(data_iterator):
    """Generate a batch"""

    batch = get_batch_on_this_tp_rank(data_iterator)

    batch = get_batch_on_this_cp_rank(batch)

    return batch


def _create_mock_batch(reference_batch):
    """Create a mock batch based on a reference batch with loss_mask zeroed out.

    The mock batch has the same tensor shapes as the reference but will not
    contribute to gradients because loss_mask is all zeros.
    """
    mock_batch = copy.deepcopy(reference_batch)
    mock_batch["loss_mask"] = torch.zeros_like(mock_batch["loss_mask"])
    mock_batch["images"] = None
    mock_batch["image_grid_thw"] = None
    mock_batch["pixel_values_videos"] = None
    mock_batch["video_grid_thw"] = None
    return mock_batch


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10

batch_list = []
forward_step_calling_count = 0
_vpp0_batch_cache = []   # cached batch_list per forward_group_id from vp_stage=0
_vpp_counters = {}       # counter for higher vpp chunks


def _reset_step_state():
    """Reset the forward-step counter and VPP caches; full hetero DP calls this after each step."""
    global forward_step_calling_count
    forward_step_calling_count = 0
    _vpp0_batch_cache.clear()
    _vpp_counters.clear()


def build_multimodal_runtime(args):
    """Engine bindings the multimodal model needs; call after Megatron parallel state exists."""
    from loongforge.models.multimodal.runtime import MultimodalRuntime
    from loongforge.engines.megatron.initialize import (
        change_parallel_state,
        get_encoder_dp_size,
        get_model_size,
    )

    scheduler_factory = None
    if args.overlap_moe_expert_parallel_comm:
        from loongforge.engines.megatron.distributed.model_chunk_schedule_plan import (
            TransformerModelChunkSchedulePlan,
        )

        scheduler_factory = TransformerModelChunkSchedulePlan

    vit_balance = None
    if args.use_vit_dp_balance:
        from loongforge.engines.megatron.distributed.dp_balance.vit_balance import (
            dp_balance_vit_encoder,
        )

        vit_balance = dp_balance_vit_encoder

    return MultimodalRuntime(
        encoder_dp_size=get_encoder_dp_size("image_encoder"),
        model_size=get_model_size(),
        parallel_state_switch=change_parallel_state,
        vit_balance=vit_balance,
        schedule_plan_factory=scheduler_factory,
    )


def build_hetero_step(args):
    """Create the full-hetero-DP step context; it carries the per-step encoder state."""
    return hetero_dp.HeteroStepContext(
        get_batch=get_batch,
        create_mock_batch=_create_mock_batch,
        reset_step_state=_reset_step_state,
        offload_enabled=args.full_hetero_dp_cpu_offload,
    )


def forward_step(data_iterator, model, return_schedule_plan: bool = False, hetero_step=None):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    args = get_args()
    timers = get_timers()
    model_config = get_model_config()
    # Get the batch.
    timers("batch-generator", log_level=2).start()
    change_parallel_state('text_decoder')

    global stimer
    global forward_step_calling_count
    global batch_list

    _ImageEncoderDataParallelSize = get_encoder_dp_size('image_encoder')

    model_add_encoder = get_attr_wrapped_model(model, 'add_encoder')
    is_higher_vpp_chunk = args.enable_full_hetero_dp and (not model_add_encoder) and mpu.is_pipeline_first_stage()

    if is_higher_vpp_chunk:
        vpp_counter = _vpp_counters.get('higher', 0)
        forward_group_id = vpp_counter // _ImageEncoderDataParallelSize
        inner_group_id = vpp_counter % _ImageEncoderDataParallelSize
        if inner_group_id == 0:
            batch_list.clear()
            batch_list.extend(_vpp0_batch_cache[forward_group_id])
        _vpp_counters['higher'] = vpp_counter + 1
    else:
        forward_group_id = forward_step_calling_count // _ImageEncoderDataParallelSize
        inner_group_id = forward_step_calling_count % _ImageEncoderDataParallelSize
        if inner_group_id == 0:
            batch_list.clear()
            with stimer(bdata=True):
                for i in range(_ImageEncoderDataParallelSize):
                    mb_index = forward_step_calling_count + i
                    if args.enable_full_hetero_dp and is_mock_microbatch(mb_index) and mpu.is_pipeline_first_stage():
                        mock_ref = batch_list[-1] if batch_list else get_batch(data_iterator)
                        if not batch_list:
                            batch_list.append(copy.deepcopy(mock_ref))
                        batch_list.append(_create_mock_batch(mock_ref))
                    else:
                        local_batch = copy.deepcopy(get_batch(data_iterator))
                        batch_list.append(local_batch)
            if args.enable_full_hetero_dp and mpu.is_pipeline_first_stage():
                _vpp0_batch_cache.append(list(batch_list))

    timers("batch-generator").stop()

    with stimer:
        (
            images,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            tokens,
            attn_mask,
            labels,
            cu_lengths,
            max_lengths,
            position_ids,
            loss_mask,
            packed_seq_params,
        ) = batch_list[inner_group_id].values()

        dump_model_input_example_once(
            tokens, labels, attn_mask, cu_lengths, packed_seq_params,
        )

        loss_func = getattr(model_config, "loss_func", default_loss_func)

        if not is_higher_vpp_chunk:
            forward_step_calling_count += 1

        if return_schedule_plan:
            assert args.overlap_moe_expert_parallel_comm, \
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                dict(
                images=images,
                image_grid_thw=image_grid_thw,
                ) if images is not None else None,
                dict(
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                ) if pixel_values_videos is not None else None,
                None,
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attn_mask,
                labels=labels,
                packed_seq_params=packed_seq_params,
                loss_mask=loss_mask,
                enable_encoder_hetero_dp=args.enable_encoder_hetero_dp,
                batch_list=batch_list,
                forward_group_id=forward_group_id,
                inner_group_id=inner_group_id,
                enable_full_hetero_dp=args.enable_full_hetero_dp,
                hetero_step=hetero_step,
            )
            return schedule_plan, partial(loss_func, loss_mask)
        else:
            output_tensor = model(
                dict(
                    images=images,
                    image_grid_thw=image_grid_thw,
                ) if images is not None else None,
                dict(
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                ) if pixel_values_videos is not None else None,
                None,
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attn_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
                enable_encoder_hetero_dp=args.enable_encoder_hetero_dp,
                batch_list=batch_list,
                forward_group_id=forward_group_id,
                inner_group_id=inner_group_id,
                enable_full_hetero_dp=args.enable_full_hetero_dp,
                hetero_step=hetero_step,
            )
            if hasattr(output_tensor, "loss") and output_tensor.loss is not None:
                output_tensor = output_tensor.loss

    return output_tensor, partial(loss_func, loss_mask)  # TODO: add loss_weights data


def train_valid_test_dataset_provider(train_val_test_num_samples, vp_stage=None, hetero_step=None):
    """Provides the datasets used by the trainer"""
    import loongforge.engines.megatron.distributed.dp_balance.patches  # noqa: F401
    args = get_args()
    tokenizer = get_tokenizer()
    loader_config = sft_loader_config(args)
    data_config = vlm_data_config(args)

    if args.is_tokenized_data:
        dpr = parallel_state.get_data_parallel_rank()
        dpw = parallel_state.get_data_parallel_world_size()
        train_ds = load_pretokenized_dataset(args.data_path, dpr)
        collator = build_sft_data_collator(
            DataCollatorForSeq2Seq,
            loader_config,
            tokenizer,
            pad_to_multiple_of=llm_pad_to_multiple_of(args),
        )
        train_data_iterator, valid_data_iterator, test_data_iterator = (
            build_sft_cyclic_iterators(
                train_ds,
                None,
                None,
                collator,
                loader_config,
                dpr,
                dpw,
                args.consumed_train_samples,
                args.consumed_valid_samples,
                args.skip_train,
                state=load_dataloader_state(args),
            )
        )

        # Build encoder-specific iterator for full_hetero_dp
        if getattr(args, 'enable_full_hetero_dp', False):
            encoder_iter = hetero_dp.build_encoder_data_iterator(
                train_ds,
                args.consumed_train_samples,
                collator,
                micro_batch_size=args.micro_batch_size,
                num_workers=args.num_workers,
                data_sharding=args.data_sharding,
            )
            if hetero_step is None:
                raise RuntimeError("full hetero DP needs the step context from the training method")
            hetero_step.encoder_data_iterator = encoder_iter

        return train_data_iterator, None, None

    else:
        dpr = parallel_state.get_data_parallel_rank()
        dpw = parallel_state.get_data_parallel_world_size()
        chat_template = get_chat_template() if args.training_phase == constants.TrainingPhase.SFT else None
        task_encoder = build_task_encoder(data_config, tokenizer, chat_template)
        train_dataset = get_train_dataset(
            task_encoder,
            data_config,
            dp_rank=dpr,
            dp_size=dpw,
            dp_group=parallel_state.get_data_parallel_group(),
        )

        collator = build_sft_data_collator(
            partial(
                VLMPretrainCollator,
                packed_padding=vlm_packed_padding(args),
                packing_sft_data=args.packing_sft_data,
                model_config=get_model_config(),
            ),
            loader_config,
            tokenizer,
            pad_to_multiple_of=llm_pad_to_multiple_of(args),
        )
        state = load_dataloader_state(args)
        train_dataloader = get_train_loader(train_dataset, collator, state)
        valid_dataloader = None
        if args.eval_interval and args.eval_iters > 0:
            valid_dataset = get_val_dataset(
                task_encoder,
                data_config,
                dp_rank=dpr,
                dp_size=dpw,
                dp_group=parallel_state.get_data_parallel_group(),
            )
            if valid_dataset is not None:
                valid_dataloader = get_train_loader(valid_dataset, collator)

        # Build encoder-specific iterator for full_hetero_dp (Energon path)
        if getattr(args, 'enable_full_hetero_dp', False):
            encoder_dataset = get_train_dataset(
                task_encoder,
                data_config,
                dp_rank=dpr,
                dp_size=dpw,
                dp_group=parallel_state.get_data_parallel_group(),
            )
            encoder_dataloader = get_train_loader(
                encoder_dataset, collator, load_dataloader_state(args)
            )
            encoder_iter = hetero_dp.build_encoder_energon_iterator(encoder_dataloader)
            if hetero_step is None:
                raise RuntimeError("full hetero DP needs the step context from the training method")
            hetero_step.encoder_data_iterator = encoder_iter

        return train_dataloader, valid_dataloader, None


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        mtp_ranks = get_mtp_ranks(pp_ranks, config=get_model_config().foundation)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks

def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Build the multimodal model with this run's args and model config."""
    args = get_args()
    return omni_model_provider(pre_process, post_process, vp_stage=vp_stage, args=args,
                               model_config=get_model_config(), runtime=build_multimodal_runtime(args))


def install_vlm_batch_layout(args):
    """Give DP load balancing the VLM packed layout; the engine keeps no model-family default."""
    if not args.use_vlm_dp_balance:
        return
    from loongforge.engines.megatron.distributed.dp_balance.rebalance.balance import VLM_BATCH_LAYOUT
    from loongforge.engines.megatron.distributed.dp_balance.rebalance.layout import install_batch_layout

    install_batch_layout(args.model_family, VLM_BATCH_LAYOUT)


def train(args):
    """Run VLM pretraining."""
    install_vlm_batch_layout(args)
    hetero_step = build_hetero_step(args) if args.enable_full_hetero_dp else None
    dataset_provider = partial(train_valid_test_dataset_provider, hetero_step=hetero_step)
    # Upstream reads this flag off the provider object; a partial does not inherit it.
    dataset_provider.is_distributed = True
    pretrain(
        args,
        dataset_provider,
        model_provider,
        ModelType.encoder_or_decoder,  # TODO: Heterogeneous TP/PP not supported yet
        partial(forward_step, hetero_step=hetero_step),
        get_embedding_ranks=get_embedding_ranks,
        hetero_step=hetero_step,
    )
