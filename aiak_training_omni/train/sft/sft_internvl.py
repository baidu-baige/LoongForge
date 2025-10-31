""" sft for internvl model """
import torch.distributed as dist

import math
from aiak_training_omni.data.internvl.dataset_packed import PackedDataset
import numpy as np
from torch.utils.data import ConcatDataset, WeightedRandomSampler, RandomSampler

from aiak_training_omni.data.internvl.internvl_dataset import LazySupervisedDataset
from aiak_training_omni.data.internvl.dataset_skip import skip_batches
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
    model_family = get_model_family(args.model_name)
    model_provider = get_model_provider(model_family)
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
    # print(f"attention_mask:{attention_mask}")
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
        data = next(data_iterator)
    else:
        data = None

    if args.use_packed_ds and data is not None:
        data["loss_weight"] = torch.tensor(data["loss_weight"], dtype=torch.float32)

    data_i = {}
    data_f = {}
    data_l = {}
    packed_seq_params = None
    attention_mask = None
    attn_mask_type = None

    if args.use_packed_ds:
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
        if args.use_packed_ds:
            data_l = tensor_parallel.broadcast_data(["loss_weight"], data, torch.float32)
    elif mpu.is_pipeline_first_stage():
        data_i = tensor_parallel.broadcast_data(["input_ids", "position_ids", "image_flags"], data, torch.int64)
        data_f = tensor_parallel.broadcast_data(["pixel_values"], data, torch.float32)
    elif mpu.is_pipeline_last_stage():
        data_i = tensor_parallel.broadcast_data(["input_ids", "labels", "image_flags"], data, torch.int64)
        if args.use_packed_ds:
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
            if args.use_packed_ds:
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
    return output_tensor, partial(loss_func, loss_mask, loss_weights)


class WeightedConcatDataset(ConcatDataset):
    """ WeightedConcatDataset """

    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


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


def packed_collate_fn(
        features,
        data_collator,
        len2weight: callable,
        max_item_length: int,
        micro_num: int = 1,
        loss_reduction_all_gather: bool = False,
        pad_id: int = 0,
):
    """ packed_collate_fn """
    if not isinstance(features, list):
        features = [features]

    if len(features) > micro_num:
        raise NotImplementedError(f'{len(features)=} > {micro_num=}')

    # if len(features) < micro_num and WARNING_CNT['micro_num_warning'] < 5:
    #     logger.warning(
    #         f'{len(features)=} > {micro_num=}, '
    #         f'the features will be padded to satisfy micro_num requirement'
    #     )
    #     WARNING_CNT['micro_num_warning'] += 1

    # ensure that the len(features) is equal to the required micro_num
    num_features = len(features)
    while len(features) < micro_num:
        features.append(copy.deepcopy(features[0]))
        features[-1]['labels'] = torch.full_like(features[-1]['labels'], IGNORE_TOKEN_ID)

    indexes = []
    cu_seqlens = []
    cu_num_images_list = [0]

    worker_state_key_list = []
    worker_state_dict_list = []
    worker_state_custom_infos_list = []

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]

    num_samples = 0
    num_padding_tokens = 0
    for feat_idx, feat in enumerate(features):
        data_index = feat.pop('data_index')
        curr_cu_seqlens, curr_indexes, curr_loss_weight = PackedDataset.get_cu_seqlens_and_indexes(
            data_index=data_index,
            input_ids=feat['input_ids'],
            labels=feat['labels'],
            len2weight=len2weight,
        )

        feat['loss_weight'] = curr_loss_weight

        if feat_idx < num_features:
            num_samples += len(curr_cu_seqlens) - 1

        if curr_cu_seqlens[-1] < max_item_length:
            curr_cu_seqlens.append(max_item_length)
            curr_indexes.extend(list(range(max_item_length - curr_cu_seqlens[-2])))

        indexes.append(torch.tensor(curr_indexes, dtype=torch.long))
        cu_seqlens.append(torch.tensor(curr_cu_seqlens, dtype=torch.int32))

        worker_state_key_list.append(feat.pop('worker_state_key'))
        worker_state_dict_list.append(feat.pop('worker_state_dict'))
        worker_state_custom_infos_list.append(feat.pop('custom_infos', None))

        num_padding_tokens += (max_item_length - feat['input_ids'].size(0))
        cu_num_images_list.append(cu_num_images_list[-1] + feat['pixel_values'].size(0))

    batch = data_collator(features=features, max_item_length=max_item_length, pad_id=pad_id)
    # convert it to list in case it is converted into bf16
    batch['loss_weight'] = torch.where(batch['labels'] == IGNORE_TOKEN_ID, 0, batch['loss_weight']).tolist()
    batch['attention_mask'] = torch.stack(cu_seqlens)
    batch['loss_reduction_all_gather'] = loss_reduction_all_gather
    batch['statistics'] = torch.tensor(
        [
            num_samples,
            num_padding_tokens,
            batch['image_flags'].numel() - batch['image_flags'].sum().item(),
        ],
        dtype=torch.long,
    )
    batch.pop('type_ids')
    return batch


def concat_pad_data_collator(features, max_item_length=None, pad_id=0):
    """ concat_pad_data_collator """
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        if 'position_ids' in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
            feat['position_ids'] = temp_position_ids

        if 'loss_weight' in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
            feat['loss_weight'] = temp_loss_weight

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """ train_valid_test_datasets_provider """

    args = get_args()
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    elif args.communicate_dataset and not mpu.is_pipeline_first_stage():
        return None, None, None

    # 确保 args.save 有值，防止 wandb 初始化错误
    if not hasattr(args, 'save') or args.save is None:
        args.save = os.path.join(os.getcwd(), 'outputs')
        print(f"Warning: args.save was None, setting default value to {args.save}")
        # 确保目录存在
        os.makedirs(args.save, exist_ok=True)

    # 禁用 wandb，如果它导致问题
    if not hasattr(args, 'wandb_project') or args.wandb_project is None:
        print("Warning: Disabling wandb logging as wandb_project is not set")
        args.wandb_project = ""
        args.wandb_exp_name = ""
        args.wandb_save = False
        args.use_wandb = False

    num_image_token = int((args.force_image_size // args.patch_size) ** 2 * (args.down_sample_ratio ** 2))
    tokenizer = get_tokenizer().tokenizer
    datasets = []
    lengths = []
    data_rank = mpu.get_data_parallel_rank()
    data_world_size = mpu.get_data_parallel_world_size()

    # 辅助函数：检查文件是否存在，如果不存在，尝试其他可能的路径
    def find_existing_file(file_path):
        if os.path.exists(file_path):
            return file_path

        # 尝试添加.jsonl后缀
        if not file_path.endswith('.jsonl'):
            jsonl_path = file_path + '.jsonl'
            if os.path.exists(jsonl_path):
                print(f"Found file at {jsonl_path} instead of {file_path}")
                return jsonl_path

        # 尝试将.json替换为.jsonl
        if file_path.endswith('.json'):
            jsonl_path = file_path.replace('.json', '.jsonl')
            if os.path.exists(jsonl_path):
                print(f"Found file at {jsonl_path} instead of {file_path}")
                return jsonl_path

        print(f"Warning: Could not find file at {file_path} or any alternative paths")
        return file_path  # 返回原始路径，即使它不存在

    # 处理数据集配置
    all_ds_collections = {}

    if hasattr(args, 'data_path') and args.data_path and len(args.data_path) > 0:
        print("Loading dataset configuration from args.data_path")

        # 检查是否使用新的拼接格式（用@分隔的多个配置）
        data_path_str = args.data_path[0]
        config_parts = data_path_str.split('@')

        if len(config_parts) >= 5:
            print("Detected combined configuration format with @ separator")
            # 拆分不同类型的配置
            data_root = config_parts[0]
            data_annotations = config_parts[1]
            data_augment = config_parts[2]
            data_max_dynamic_patch = config_parts[3]
            data_repeat_time = config_parts[4]

            # 使用制表符分隔每个配置内的多个值
            data_roots = data_root.split('#') if data_root else []
            data_paths = data_annotations.split('#') if data_annotations else []
            data_augments = data_augment.split('#') if data_augment else []
            data_max_patches = data_max_dynamic_patch.split('#') if data_max_dynamic_patch else []
            data_repeat_times = data_repeat_time.split('#') if data_repeat_time else []

            print(f"Parsed combined configuration: {len(data_paths)} datasets")
        else:
            # 简单格式：每行一个数据集路径
            print("Using simple format: one dataset path per line")
            data_paths = data_path_str.split('\n')
            data_roots = ['/'] * len(data_paths)
            data_augments = ['false'] * len(data_paths)
            data_max_patches = [str(args.max_dynamic_patch)] * len(data_paths)
            data_repeat_times = ['1'] * len(data_paths)
    else:
        print("No data_path provided. No datasets will be loaded.")
        data_paths = []
        data_roots = []
        data_augments = []
        data_max_patches = []
        data_repeat_times = []

    # 确保所有列表长度一致，使用默认值填充不足的部分
    num_datasets = max(len(data_paths), 1)  # 至少需要一个数据路径

    if len(data_roots) < num_datasets:
        data_roots.extend(['/' for _ in range(num_datasets - len(data_roots))])

    if len(data_augments) < num_datasets:
        data_augments.extend(['false' for _ in range(num_datasets - len(data_augments))])

    if len(data_max_patches) < num_datasets:
        data_max_patches.extend([str(args.max_dynamic_patch) for _ in range(num_datasets - len(data_max_patches))])

    if len(data_repeat_times) < num_datasets:
        data_repeat_times.extend(['1' for _ in range(num_datasets - len(data_repeat_times))])

    # 组装数据集配置
    for i in range(len(data_paths)):
        if not data_paths[i].strip():
            continue

        # 使用全路径作为数据集名称
        path = data_paths[i].strip()
        # 清理路径，使其成为有效的标识符
        ds_name = path.replace('/', '_').replace('\\', '_').replace('.', '_').replace('-', '_').replace(' ', '_')
        # 确保名称不以数字开头（避免无效标识符）
        if ds_name and ds_name[0].isdigit():
            ds_name = 'ds_' + ds_name
        # 确保唯一性
        if ds_name in all_ds_collections:
            ds_name = f"{ds_name}_{i}"
        ds_config = {
            'root': data_roots[i] if i < len(data_roots) else '/',
            'annotation': data_paths[i],
            'data_augment': data_augments[i].lower() == 'true' if i < len(data_augments) else False,
            'max_dynamic_patch':
                int(data_max_patches[i])
                if i < len(data_max_patches) and data_max_patches[i].isdigit() else args.max_dynamic_patch,
            'repeat_time':
                int(data_repeat_times[i]) if i < len(data_repeat_times) and data_repeat_times[i].lstrip(
                    '-').isdigit() else 1
        }

        # 确保annotation字段指向jsonl文件
        if not ds_config['annotation'].endswith('.jsonl'):
            jsonl_path = ds_config['annotation'] + '.jsonl'
            print(f"Warning: Converting annotation path from {ds_config['annotation']} to {jsonl_path}")
            ds_config['annotation'] = jsonl_path

        all_ds_collections[ds_name] = ds_config
        print(f"Added dataset: {ds_name} with config: {ds_config}")

    print(f"Total datasets loaded: {len(all_ds_collections)}")

    # 处理每个数据集
    for ds_idx, ds_name in enumerate(all_ds_collections.keys()):
        ds_config = all_ds_collections[ds_name]

        # 打印数据集配置，用于调试
        print(f"Processing dataset {ds_name} with config: {ds_config}")

        # 设置默认值
        repeat_time = ds_config.get('repeat_time', 1)
        max_num = ds_config.get('max_dynamic_patch', args.max_dynamic_patch)

        # 检查必要的字段
        if 'annotation' not in ds_config:
            print(f"Warning: Dataset {ds_name} is missing required 'annotation' field. Skipping.")
            continue

        # 确保annotation字段指向jsonl文件
        if not ds_config['annotation'].endswith('.jsonl'):
            jsonl_path = ds_config['annotation'] + '.jsonl'
            print(f"Warning: Converting annotation path from {ds_config['annotation']} to {jsonl_path}")
            ds_config['annotation'] = jsonl_path

        try:
            # 确保annotation文件存在
            ds_config['annotation'] = find_existing_file(ds_config['annotation'])
            if not os.path.exists(ds_config['annotation']):
                print(f"Warning: Annotation file {ds_config['annotation']} does not exist. Skipping.")
                continue

            dataset = LazySupervisedDataset(
                args.conv_style,
                ds_config,
                tokenizer,
                None,
                ds_name=ds_name,
                num_image_token=num_image_token,
                image_size=args.force_image_size,
                is_train=ds_config.get('data_augment', False),
                pad2square=args.pad2square,
                group_by_length=args.group_by_length and not args.use_packed_ds,
                dynamic_image_size=args.dynamic_image_size,
                use_thumbnail=args.use_thumbnail,
                min_dynamic_patch=args.min_dynamic_patch,
                max_dynamic_patch=max_num,
                min_num_frame=args.min_num_frame,
                max_num_frame=args.max_num_frame,
                repeat_time=abs(repeat_time),
                normalize_type=args.normalize_type,
                # hyperparameters for packed training
                use_packed_ds=args.use_packed_ds,
                data_rank=data_rank,
                data_world_size=data_world_size,
                distributed_mode=args.use_packed_ds,
                force_shuffle=args.use_packed_ds,
                random_seed=ds_idx,
                ckpt_path=args.save,
                split_annotations=not args.no_split_annotations
            )

            # 只添加非空数据集
            if len(dataset) > 0:
                datasets.append(dataset)
                if args.use_data_resampling:
                    lengths.append(math.sqrt(len(dataset)))
                else:
                    lengths.append(len(dataset) if repeat_time > 0 else 0)
            else:
                print(f"Warning: Dataset {ds_name} is empty and will be skipped.")
        except Exception as e:
            print(f"Error creating dataset {ds_name}: {e}")
            continue

    # 确保至少有一个非空数据集
    if not datasets:
        raise ValueError("No valid datasets found. Please check your data paths.")

    train_iteration = 0
    if args.save_dataset_state:
        def find_saved_dataset_state(ckpt_dir, iteration):
            max_iter = 0
            # 遍历目标目录下的所有文件夹
            if os.path.exists(ckpt_dir):
                for dir_name in os.listdir(ckpt_dir):
                    dir_path = os.path.join(ckpt_dir, dir_name)
                    # 确保是目录
                    if os.path.isdir(dir_path):
                        # 从目录名中提取数字部分
                        match = re.match(r"iter_(\d+)", dir_name)
                        if match :
                            iter_num = int(match.group(1))
                            # 如果数字大于当前最大值，则更新最大值
                            if iter_num <= iteration and iter_num > max_iter:
                                max_iter = iter_num
            return max_iter

        train_iteration = 0
        if args.load:
            data_ckpt_dir = os.path.join(args.load, "dataset_ckpt")
            data_ckpt_path = os.path.join(data_ckpt_dir, f"iter_{args.iteration:07d}")
            data_ckpt_exist = os.path.exists(data_ckpt_path)
            if data_ckpt_exist:
                train_iteration = args.iteration
            else:
                train_iteration = find_saved_dataset_state(data_ckpt_dir, args.iteration)
    num_skip_batched = (args.iteration - train_iteration) * (args.global_batch_size / data_world_size)

    if args.use_packed_ds:
        total_length = sum(lengths)
        # 确保权重有效
        weights = [l / total_length for l in lengths] if total_length > 0 else None
        if weights and sum(weights) < 0.99:
            print(f"Warning: Dataset weights sum to {sum(weights)}, normalizing...")
            weights = [w / sum(weights) for w in weights]

        if args.save_dataset_state:
            exit_iteration = min(args.exit_interval, args.train_iters) if args.exit_interval else args.train_iters
            dataset_params = {
                "save_ckpt_interval": args.save_interval,
                "num_workers": args.num_workers,
                "global_batch_size": args.global_batch_size,
                "load_ckpt_path": args.load,
                "save_ckpt_path": args.save,
                "save_dataset_state": args.save_dataset_state,
                "train_iteration": train_iteration,
                "exit_iteration": exit_iteration,
            }
        else:
            dataset_params = {}

        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=weights,
            num_images_expected=args.num_images_expected,
            max_packed_tokens=args.max_packed_tokens,
            max_buffer_size=args.max_buffer_size,
            log_freq=args.log_freq,
            strict_mode=args.strict_mode,
            replacement=args.replacement,
            allow_overflow=args.allow_overflow,
            allow_deduplicated_ds_name=False,
            **dataset_params
        )
    elif args.use_data_resampling:
        total_length = sum(lengths)
        # 确保权重有效
        weights = [l / total_length for l in lengths] if total_length > 0 else None
        if weights and sum(weights) < 0.99:
            print(f"Warning: Dataset weights sum to {sum(weights)}, normalizing...")
            weights = [w / sum(weights) for w in weights]

        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)

    if args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=args.max_packed_tokens if args.strict_mode else 0,
            micro_num=args.micro_batch_size,
            len2weight=partial(len2weight, loss_reduction=args.loss_reduction),
            loss_reduction_all_gather=args.loss_reduction_all_gather,
        )
    else:
        collator = partial(concat_pad_data_collator, pad_id=tokenizer.pad_token_id)

    def seed_worker(_):

        def set_seed(seed: int, deterministic: bool = False):
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        worker_seed = torch.initial_seed() % 2 ** 32
        set_seed(worker_seed)

    args.dataloader_drop_last = False
    args.dataloader_persistent_workers = False
    args.dataloader_pin_memory = True
    dataloader_params = {
        "batch_size": args.micro_batch_size,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "persistent_workers": args.dataloader_persistent_workers,
        "prefetch_factor": args.dataloader_prefetch_factor
    }
    if not args.use_packed_ds:
        dataloader_params["sampler"] = RandomSampler(train_dataset)
        dataloader_params["drop_last"] = args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
    data_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
    new_data_loader = skip_batches(data_loader, num_skip_batched)

    data_iterator = iter(new_data_loader)
    return data_iterator, None, None

@register_model_trainer(model_family=[constants.VisionLanguageModelFamilies.INTERN_VL],
                        training_phase=constants.TrainingPhase.SFT)
def default_pretrain_trainer(train_args):
    """build trainer"""
    trainer = MegatronTrainer(
        train_args=train_args,
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_provider=model_provider,
        model_type=ModelType.encoder_or_decoder,
        forward_step_func=forward_step,
    )

    return trainer
