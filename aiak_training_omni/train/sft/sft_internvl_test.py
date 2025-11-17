""" sft for internvl model """
import torch.distributed as dist

import math
# from aiak_training_omni.data.internvl.dataset_packed import PackedDataset
import numpy as np
from torch.utils.data import ConcatDataset, WeightedRandomSampler, RandomSampler

# from aiak_training_omni.data.internvl.internvl_dataset import LazySupervisedDataset
# from aiak_training_omni.data.internvl.dataset_skip import skip_batches
import os
import torch
import re
import copy
from functools import partial
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_timers
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.core import mpu, tensor_parallel
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler

from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import StragglerDetector
from aiak_training_omni.utils import (get_args, get_tokenizer)
from megatron.core.enums import ModelType
from aiak_training_omni.utils import constants
from aiak_training_omni.models import get_model_provider, get_model_family
from aiak_training_omni.train.megatron_trainer import MegatronTrainer
from aiak_training_omni.train.trainer_builder import register_model_trainer
from aiak_training_omni.data.multimodal.internvl.internvl_task_encoder import InternVLTaskEncoder

from aiak_training_omni.data.multimodal.dataloader_provider import (
    get_train_dataset,
    get_train_loader
)
from aiak_training_omni.models.omni_models.omni_model_provider import (
    omni_model_provider,
)

IGNORE_TOKEN_ID = -100
IGNORE_INDEX = -100

stimer = StragglerDetector()


def model_provider(pre_process=True, post_process=True):
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        MCoreModel: The returned model
    """
    args = get_args()
    #model_family = get_model_family(args.model_family)
    model_provider = get_model_provider(args.model_family)
    assert model_provider is not None, f'model provider for {args.model_name} not found'
    return model_provider(pre_process, post_process)


def loss_func(loss_mask: torch.Tensor, loss_weight: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across the data parallel ranks
    """
    args = get_args()

    valid_mask = True
    if (loss_weight is not None and loss_weight.sum() == 0) or (loss_mask.sum() == 0):
        valid_mask = False
        output_tensor = output_tensor * 0.0  # skip update current microbatch

    losses = output_tensor.float()  # [B, s]
    loss_mask = loss_mask.view(-1).float()  # [B * s]

    if loss_weight is not None:
        shift_weights = loss_weight.view(-1)
        shift_weights_sum = shift_weights.sum()
        if args.loss_reduction_all_gather:
            torch.distributed.all_reduce(shift_weights_sum,
                                             op=dist.ReduceOp.SUM,
                                             group=mpu.get_data_parallel_group(with_context_parallel=True))
            shift_weights_sum = shift_weights_sum / mpu.get_data_parallel_world_size(with_context_parallel=True)
        loss = torch.sum(losses.view(-1) * shift_weights) / (shift_weights_sum if valid_mask else 1.)  # avoid divide 0
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / (loss_mask.sum() if valid_mask else 1.)

    if args.context_parallel_size > 1:
        cp_group = mpu.get_context_parallel_group()
        cp_size = torch.distributed.get_world_size(cp_group)
        torch.distributed.all_reduce(loss, group=cp_group)
        loss = loss / cp_size
    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                  f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    # reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_reduced_dict = {'lm loss': averaged_loss[0]}

    if args.variable_seq_lengths:
        # for variable seq length, we need to calculate the number of tokens on fly
        # model output tensor shape is [B, S, H]
        num_input_tokens = output_tensor.shape[0] * output_tensor.shape[1]
        input_tokens = torch.tensor(num_input_tokens, dtype=torch.int, device=output_tensor.device)
        # sum across all dp ranks
        torch.distributed.all_reduce(input_tokens, group=mpu.get_data_parallel_group())
        loss_reduced_dict["total_inputs"] = input_tokens.item() * args.context_parallel_size

    return (loss, loss_reduced_dict)


def get_packed_seq_params(attention_mask):
    """ get packed seq params """
    packed_seq_params = PackedSeqParams()
    packed_seq_params.qkv_format = "thd"
    packed_seq_params.cu_seqlens_q = attention_mask
    packed_seq_params.cu_seqlens_kv = attention_mask
    packed_seq_params.max_seqlen_q = (attention_mask[1:] - attention_mask[:-1]).max().item()
    packed_seq_params.max_seqlen_kv = packed_seq_params.max_seqlen_q

    return packed_seq_params, AttnMaskType.padding_causal


def get_batch(data_iterator):
    """Generate a batch"""
    # get batches based on the TP rank you are on
    args = get_args()
    if args.communicate_dataset and not mpu.is_pipeline_first_stage():
        return None, None, None, None, None, None, None, None, None, None

    if data_iterator is not None:
        sample = next(data_iterator)
        data = {}
        data["input_ids"] = sample["tokens"]
        data["position_ids"] = sample["position_ids"]
        data["attention_mask"] = sample["attn_mask"]
        data["pixel_values"] = sample["imgs"]
        data["labels"] = sample["labels"]
        data["image_flags"] = sample["image_flags"]
        data["loss_weight"] = sample["loss_weight"]
    else:
        data = None

    if data and data["loss_weight"] is not None:
        data["loss_weight"] = torch.tensor(data["loss_weight"], dtype=torch.float32)

    data_i = {}
    data_f = {}
    data_l = {}
    packed_seq_params = None
    attention_mask = None
    attn_mask_type = None

    if args.packing_sft_data:
        data_a = tensor_parallel.broadcast_data(["attention_mask"], data, torch.int32)
        attention_mask = data_a["attention_mask"].squeeze_(0)
        packed_seq_params, attn_mask_type = get_packed_seq_params(attention_mask)
    else:
        data_a = tensor_parallel.broadcast_data(["attention_mask"], data, torch.bool)
        packed_seq_params, attn_mask_type = None, AttnMaskType.padding_causal if data_a["attention_mask"].any(
        ) else AttnMaskType.causal
        attention_mask = ~(data_a["attention_mask"].unsqueeze(1).unsqueeze(1))

    if args.pipeline_model_parallel_size == 1 or args.communicate_dataset:
        data_i = tensor_parallel.broadcast_data(["input_ids", "position_ids", "labels", "image_flags"], data,
                                                torch.int64)
        data_f = tensor_parallel.broadcast_data(["pixel_values"], data, torch.float32)
        if args.packing_sft_data:
            data_l = tensor_parallel.broadcast_data(["loss_weight"], data, torch.float32)
    elif mpu.is_pipeline_first_stage():
        data_i = tensor_parallel.broadcast_data(["input_ids", "position_ids", "image_flags"], data, torch.int64)
        data_f = tensor_parallel.broadcast_data(["pixel_values"], data, torch.float32)
    elif mpu.is_pipeline_last_stage():
        data_i = tensor_parallel.broadcast_data(["input_ids", "labels", "image_flags"], data, torch.int64)
        if args.packing_sft_data:
            data_f = tensor_parallel.broadcast_data(["loss_weight"], data, torch.float32)

    input_ids = data_i["input_ids"] if "input_ids" in data_i else None
    position_ids = data_i["position_ids"] if "position_ids" in data_i else None
    labels = data_i["labels"] if "labels" in data_i else None
    image_flags = data_i["image_flags"] if "image_flags" in data_i else None
    pixel_values = data_f["pixel_values"] if "pixel_values" in data_f else None
    loss_weight = data_l["loss_weight"] if "loss_weight" in data_l else None

    if labels is not None:
        labels = torch.roll(labels, shifts=-1, dims=1)
        labels[:, -1] = -100

    if loss_weight is not None:
        loss_weight = torch.roll(loss_weight, shifts=-1, dims=1)
        loss_weight[:, -1] = 0.0

    loss_mask = (labels != -100).int() if labels is not None else None

    # slice batch along sequence dimension for context parallelism

    batch = (pixel_values,
             position_ids,
             input_ids,
             image_flags,
             attention_mask,
             labels,
             attn_mask_type,
             loss_mask,
             packed_seq_params,
             loss_weight)
    return batch


def filter_ignore_data(input_ids, image_flags, img_context_token_id, num_image_token):
    """ for qianfanvl """
    selected = (input_ids == img_context_token_id).sum().item()
    expected = (image_flags == 1).sum().item() * num_image_token
    return selected != expected


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    args = get_args()
    global stimer
    with stimer(bdata=True):
        (pixel_values, position_ids, input_ids, image_flags, attention_mask, labels, attn_mask_type, loss_mask,
         packed_seq_params, loss_weights) = get_batch(data_iterator)

    timers('batch-generator').stop()

    if args.communicate_dataset:
        if attention_mask is None:
            attention_mask = model.module.module.attention_mask
            labels = model.module.module.labels
            loss_weights = model.module.module.loss_weights
            loss_mask = (labels != -100).int()
            if args.packing_sft_data:
                packed_seq_params, attn_mask_type = get_packed_seq_params(attention_mask)
            else:
                packed_seq_params, attn_mask_type = None, AttnMaskType.padding_causal if attention_mask.any(
                ) else AttnMaskType.causal

    with stimer:
        output_tensor = model(
            pixel_values,
            input_ids,
            position_ids,
            attention_mask,
            image_flags,
            labels=labels,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )
    if args.communicate_dataset:
        ignore = model.module.module.ignore.to(attention_mask.device)
        pad_attention_mask = torch.nn.functional.pad(attention_mask, (0, 1000 - attention_mask.shape[0]), value=-1).to(
            attention_mask.dtype).to(attention_mask.device)
        pad_labels = torch.full((labels.shape[0], args.max_packed_tokens), -1, dtype=labels.dtype).to(labels.device)
        pad_labels[:, :labels.shape[1]] = labels
        pad_loss_weights = torch.nn.functional.pad(loss_weights, (0, args.max_packed_tokens - loss_weights.shape[1]),
                                                   value=-1).to(loss_weights.dtype).to(loss_weights.device)
    # for qianfanvl
    if mpu.is_pipeline_last_stage():
        if args.communicate_dataset:
            ignore_flag = model.module.module.ignore[0]
        else:
            # tokenizer = get_tokenizer().tokenizer
            # img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            num_image_token = int((args.force_image_size // args.patch_size) ** 2 * (args.down_sample_ratio ** 2))
            ignore_flag = filter_ignore_data(input_ids, image_flags, model.module.module.img_context_token_id,
                                             num_image_token)
        if ignore_flag:
            print(f"filter_ignore_data get True, skip current microbatch...")
            output_tensor = output_tensor * 0.0
    if args.communicate_dataset:
        return output_tensor, partial(loss_func, loss_mask,
                                      loss_weights), pad_attention_mask, pad_labels, pad_loss_weights, ignore
    # print(f"packed_seq_params:{packed_seq_params}")
    return output_tensor, partial(loss_func, loss_mask, loss_weights)


def len2weight(x, loss_reduction):
    """ len2weight """
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """ train_valid_test_datasets_provider """
    args = get_args()
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    elif args.communicate_dataset and not mpu.is_pipeline_first_stage():
        return None, None, None

    # 禁用 wandb，如果它导致问题
    if not hasattr(args, 'wandb_project') or args.wandb_project is None:
        print("Warning: Disabling wandb logging as wandb_project is not set")
        args.wandb_project = ""
        args.wandb_exp_name = ""
        args.wandb_save = False
        args.use_wandb = False

    # print(f"data_path: {args.data_path}")
    tokenizer = get_tokenizer().tokenizer
    task_encoder = InternVLTaskEncoder(args, tokenizer)
    train_dataset = get_train_dataset(task_encoder)
    train_dataloader = get_train_loader(train_dataset)
    return train_dataloader, None, None


# @register_model_trainer(
#     model_family=constants.VisionLanguageModelFamilies.names(),
#     training_phase=constants.TrainingPhase.SFT,
# )
# def default_sft_trainer(train_args):
#     """build trainer"""
#     trainer = MegatronTrainer(
#         train_args=train_args,
#         train_valid_test_dataset_provider=train_valid_test_datasets_provider,
#         model_provider=model_provider,
#         # model_provider=omni_model_provider,
#         model_type=ModelType.encoder_or_decoder,
#         forward_step_func=forward_step,
#     )

#     return trainer
