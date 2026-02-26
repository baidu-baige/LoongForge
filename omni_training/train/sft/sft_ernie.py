"""Sft script for Ernie"""

import torch
import torch.nn.functional as F
from functools import partial
from megatron.core.enums import ModelType
from megatron.training import get_timers

from omni_training.utils import get_args, print_rank_0
from omni_training.utils.constants import TrainingPhase, VisionLanguageModelFamilies
from omni_training.data.video.latent_dataset import ErnieImageDataset
from omni_training.train.megatron_trainer import MegatronTrainer
from omni_training.train.trainer_builder import register_model_trainer
from megatron.core import parallel_state
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.utils import StragglerDetector
from omni_training.utils import get_model_config
from omni_training.models.omni_models.omni_model_provider import (
    omni_model_provider
)
from omni_training.train.get_loss_func import default_loss_func

stimer = StragglerDetector()


def pad_to_len(data_i, loss_mask):
    """Pad to length for sp"""
    args = get_args()
    if args.tensor_model_parallel_size == 1:
        return data_i, loss_mask

    pad_to_multiple_of = 1
    pad_to_multiple_of *= (
        args.tensor_model_parallel_size if args.sequence_parallel else 1
    )
    pad_num = pad_to_multiple_of - data_i["input_ids"].shape[1] % pad_to_multiple_of
    data_i["input_ids"] = F.pad(data_i["input_ids"], (0, pad_num), "constant", 0)
    data_i["token_type_ids"] = F.pad(data_i["token_type_ids"], (0, pad_num), "constant", 0)
    data_i["position_ids"] = F.pad(data_i["position_ids"], ((0, 0, 0, pad_num)),
                                   "constant", data_i["position_ids"][0, -1, 0] + 1)
    data_i["labels"] = F.pad(data_i["labels"], (0, pad_num), "constant", -100)
    loss_mask = F.pad(loss_mask, (0, pad_num), "constant", 0)

    return data_i, loss_mask


def get_batch(data_iterator):
    """Generate a batch"""
    # get batches based on the TP rank you are on
    if data_iterator is not None:
        data = next(data_iterator)
        for key in data:
            ori_shape = data[key].shape
            data[key] = data[key].squeeze(dim=0)

        global_rank = torch.distributed.get_rank()
        # print(f"data_iterator is on current rank: {global_rank}")
    else:
        data = None

    data_i = {}
    data_f = {}
    # ['input_ids', 'token_type_ids', 'position_ids', 'images', 'grid_thw', 'image_type_ids', 'labels'])
    data_i = tensor_parallel.broadcast_data([
        "input_ids", 
        "token_type_ids",
        "position_ids",
        "grid_thw",
        "image_type_ids", 
        "labels", 
    ], data, torch.int64)
    data_f = tensor_parallel.broadcast_data(["images"], data, torch.uint8)

    # slice batch along sequence dimension for context parallelism
    assert mpu.get_context_parallel_world_size() == 1, "not implemented"
    data_i["attention_mask"] = data_i['input_ids'].logical_not()
    attn_mask_type = AttnMaskType.padding_causal if data_i["attention_mask"].any(
    ) else AttnMaskType.causal

    loss_mask = torch.roll(1 - data_i['token_type_ids'], shifts=-1, dims=1)
    loss_mask[: , -1] = 0
    data_i, loss_mask = pad_to_len(data_i, loss_mask)
    batch = (
        data_f['images'],
        data_i['input_ids'],
        data_i['token_type_ids'],
        data_i['position_ids'],
        data_i['attention_mask'],
        data_i['grid_thw'],
        data_i['image_type_ids'],
        data_i['labels'],
        loss_mask,
        attn_mask_type
    )
    return batch


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: Megatron Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    model_config = get_model_config()
    loss_func = getattr(model_config, "loss_func", default_loss_func)

    global stimer
    with stimer(bdata=True):
        images, input_ids, token_type_ids, position_ids, attention_mask, grid_thw, image_type_ids, \
            labels, loss_mask, attn_mask_type = get_batch(data_iterator)
    timers("batch-generator").stop()

    extra_input = {}
    model_config = get_model_config()
    image_mask = input_ids == model_config.foundation.im_patch_id
    with stimer:
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_mask_type=attn_mask_type,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            image_inputs={"images": images, "image_grid_thw": grid_thw, 
                "image_type_ids": image_type_ids, "image_mask": image_mask},
            labels=labels,
        )
    return loss, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets."""
    args = get_args()
    dataset = ErnieImageDataset(
        args, args.data_path[0], args.train_iters * args.global_batch_size
    )

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_world_size = parallel_state.get_data_parallel_world_size()

    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=False, num_replicas=dp_world_size, rank=dp_rank
    )
    # TODO: Batched inference is not supported yet.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=sampler,
        pin_memory=True,
    )
    print_rank_0(f"> finished creating {args.model_name} datasets ...")

    return iter(dataloader), None, None


@register_model_trainer(
    model_family=[VisionLanguageModelFamilies.ERNIE4_5_VL],
    training_phase=TrainingPhase.SFT, override=True)
def default_pretrain_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=omni_model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )
    return trainer
