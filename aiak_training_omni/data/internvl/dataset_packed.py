# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
""" packed dataset for internvl """
import random
import bisect

import os
import logging
from typing import List, Union
import traceback
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

#from transformers.trainer_pt_utils import LabelSmoother
from megatron.core import mpu
from .constants import IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN

#IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_TOKEN_ID = -100
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

def is_dist_avail_and_initialized():
    """ is distributed training available and initialized """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """ get the size of the world """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """ get the rank of the current process """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


class PackedDataset(IterableDataset):
    """ packed dataset for internvl """

    def __init__(
        self,
        tokenizer,
        data_rank,
        data_world_size,
        datasets: List,
        dataset_weight: List[int] = None,
        num_images_expected: int = 6,
        max_packed_tokens: int = 32768,
        max_buffer_size: int = 100,
        log_freq: int = 1000000,
        strict_mode: bool = False,
        debug_mode: bool = False,
        replacement: bool = True,
        allow_overflow: bool = True,
        allow_empty_data: bool = False,
        allow_deduplicated_ds_name: bool = False,
        save_ckpt_interval: int = None,
        num_workers: int = 1,
        global_batch_size: int = 1024,
        load_ckpt_path: str = None,
        save_ckpt_path: str = None,
        save_dataset_state: bool = False,
        train_iteration: int = None,
        exit_iteration: int = None,
    ):
        super().__init__()
        self.save_dataset_state = save_dataset_state
        if self.save_dataset_state:
            self.global_batch_size = global_batch_size
            self.num_workers = 1 if num_workers == 0 else num_workers
            self.train_iteration = train_iteration
            self.exit_iteration = exit_iteration
            self.current_save_iteration = train_iteration
            self.save_ckpt_interval = save_ckpt_interval if save_ckpt_interval is not None else (
                    exit_iteration - train_iteration)
            self.save_ckpt_path = save_ckpt_path
            self.load_ckpt_path = load_ckpt_path
            assert self.save_ckpt_path is not None, \
                f"args.save is None, but use save_dataset_state need args.save is not None"
            assert self.train_iteration is not None, \
                f"args.iteration is None, but use save_dataset_state need args.iteration is not None"
            assert self.exit_iteration is not None, \
                (f"args.exit_interval and args.train_iters is None, "
                 f"but use save_dataset_state need args.exit_interval or args.train_iters is not None")
            assert self.save_ckpt_interval is not None, \
                f"args.save_interval is None, but use save_dataset_state need args.save_interval is not None"
        # save_step requires the call to update_save_step in __iter__ to initialize
        self.save_step = -1
        self.current_step = 0
        self.replacement_counts = [0] * len(datasets)
        self.dataset_pop = []
        self.tp = mpu.get_tensor_model_parallel_rank()
        self.pp = mpu.get_pipeline_model_parallel_rank()

        # buffer_list load data random state
        self.rng_states = {}
        self.rng = None

        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.num_images_expected = num_images_expected
        self.max_buffer_size = max_buffer_size
        self.log_freq = log_freq
        self.strict_mode = strict_mode
        self.debug_mode = debug_mode
        self.replacement = replacement
        self.allow_overflow = allow_overflow
        self.allow_empty_data = allow_empty_data

        self.max_packed_tokens = max_packed_tokens

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        # assert self.img_start_token_id != self.tokenizer.unk_token_id
        # assert self.img_token_id != self.tokenizer.unk_token_id
        # assert self.img_end_token_id != self.tokenizer.unk_token_id

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        self.dataset_type = [d.dataset_type for d in self.datasets]

        self.datasets_orig = datasets
        self.dataset_weight_orig = [w / sum(dataset_weight) for w in dataset_weight]

        self.datasets = [ds for ds in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]

        # lazy init
        self.worker_id = None
        self.worker_state_key = None
        self.dataset_iter_list = None
        self._state_dict = {
            'sample_info': {
                d.ds_name: 0
                for d in self.datasets
            },
        }

        self.worker_custom_infos = None

        ds_name_list = [d.ds_name for d in self.datasets]
        if not allow_deduplicated_ds_name:
            assert len(ds_name_list) == len(set(ds_name_list)), f'deduplicated ds_name: {ds_name_list}'

        for ds in self.datasets:
            if ds.max_num_images > self.num_images_expected:
                logger.warning(f'{ds.max_num_images=} of {ds.ds_name} is larger than {self.num_images_expected=}')
                ds.max_num_images = num_images_expected

            if ds.max_tokens > self.max_packed_tokens:
                logger.warning(f'{ds.max_tokens=} of {ds.ds_name} is larger than {self.max_packed_tokens=}')
                ds.max_tokens = self.max_packed_tokens

            self._state_dict[ds.ds_name] = {}
        self.dataset_index_map = {ds.ds_name: i for i, ds in enumerate(self.datasets)}
        if get_rank() == 0:
            logger.info(
                f'Loaded dataset to pack: {ds_name_list}, '
                f'{self.num_images_expected=}, {self.max_packed_tokens=}, '
                f'{self.replacement=}, {self.allow_overflow=}', )

            temp = []
            for ds, ds_w in zip(self.datasets, self.dataset_weight):
                temp.append(f'{ds.ds_name:<25}: {ds_w*100:.2f}%')
            temp = '\n'.join(temp)
            logger.info(f'Sampling prob for each dataset:\n{temp}')

        if self.allow_empty_data:
            logger.warning('allow_empty_data is enabled, note that empty data may be generated!')

    def load_state_dict(self, state_dict, custom_infos=None):
        """ load state dict """
        self.worker_custom_infos = state_dict['custom_infos']

        self._state_dict.update(state_dict)
        for ds in self.datasets:
            if ds.ds_name in self._state_dict:
                ds.load_state_dict(self._state_dict[ds.ds_name])
                logger.info(f'{ds.ds_name=} is resumed.')
            else:
                logger.warning(f'{ds.ds_name=} is not resumed.')

    def _should_log(self):
        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * get_rank() + worker_id
        num_workers = num_workers * get_world_size()

        return worker_id == 0

    def next_data(self, current_dataset_idx):
        """ get next data """
        while True:
            try:
                current_sample = next(self.dataset_iter_list[current_dataset_idx])
                break  # Exit loop if successful
            except StopIteration:
                if self.replacement:
                    # logger.info(f'[Worker id {self.worker_id}] '
                    # f'Dataset {self.datasets[current_dataset_idx].ds_name} is exhausted, restart it.')
                    try:
                        index = self.dataset_index_map[self.datasets[current_dataset_idx].ds_name]
                        self.dataset_iter_list[current_dataset_idx] = iter(self.datasets[current_dataset_idx])
                        self.replacement_counts[index] += 1
                        current_sample = next(self.dataset_iter_list[current_dataset_idx])
                        break
                    except Exception as e:
                        print(e)
                        # logger.error(f'{self.worker_id=} Fail to get any data '
                        # f'from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                        index = self.dataset_index_map[self.datasets[current_dataset_idx].ds_name]
                        self.datasets.pop(current_dataset_idx)
                        self.dataset_iter_list.pop(current_dataset_idx)
                        self.dataset_weight.pop(current_dataset_idx)
                        self.dataset_pop.append(index)
                        if len(self.datasets) == 0:
                            raise StopIteration
                        current_dataset_idx = np.random.choice(len(self.datasets))
                else:
                    # logger.error(f'{self.worker_id=} Fail to get any data '
                    # f'from {self.datasets[current_dataset_idx].ds_name}! length={len(self.datasets)}')
                    index = self.dataset_index_map[self.datasets[current_dataset_idx].ds_name]
                    self.datasets.pop(current_dataset_idx)
                    self.dataset_iter_list.pop(current_dataset_idx)
                    self.dataset_weight.pop(current_dataset_idx)
                    self.dataset_pop.append(index)
                    if len(self.datasets) == 0:
                        raise StopIteration
                    current_dataset_idx = np.random.choice(len(self.datasets))
            except Exception as e:
                print(e)
                logger.error('Unexpected error!')
                if len(self.datasets) == 0:
                    raise StopIteration
                current_dataset_idx = np.random.choice(len(self.datasets))

        current_ds_name = self.datasets[current_dataset_idx].ds_name
        init_index = self.dataset_index_map[current_ds_name]
        current_sample['type_ids'] = (torch.zeros_like(current_sample['input_ids']) + init_index)
        current_sample['replacement'] = (torch.zeros_like(current_sample['input_ids'])
                                         + self.replacement_counts[init_index])

        if self.worker_state_key not in self._state_dict[current_ds_name]:
            self._state_dict[current_ds_name][self.worker_state_key] = {}

        meta_info = current_sample.pop('meta_info', {})
        rng_state = current_sample.pop('rng_state', {})
        self._state_dict[current_ds_name][self.worker_state_key].update(**meta_info)
        self._state_dict['sample_info'][self.datasets[current_dataset_idx].ds_name] += 1
        if self.save_dataset_state:
            data_idx = current_sample['data_idx'][0].item()
            replacement_count = self.replacement_counts[init_index]
            # use tuple as key
            key = (init_index, data_idx, replacement_count)
            # when the data is put into buffer_list, save its corresponding rng state to self.rng_states
            if key not in self.rng_states:
                self.rng_states[key] = rng_state
            else:
                logger.info(f'worker_id={self.worker_id}, '
                            f'ds_name={self.datasets[current_dataset_idx].ds_name}, {current_sample=}'
                            f'{key=} is already in self.rng_states:{self.rng_states.keys()}')
                raise KeyError(f'{key=} is already in self.rng_states:{self.rng_states.keys()}')

        return current_sample

    def find_buffer(self, buffer_list, new_sample):
        """ find buffer """
        # NOTE: use `bisect` to search might be faster

        find = False
        find_idx = -1
        num_images_current = new_sample['pixel_values'].size(0)
        for buffer_idx, buffer in enumerate(buffer_list):
            num_images_buffer = buffer['pixel_values'].size(0)
            if num_images_buffer + num_images_current <= self.num_images_expected:
                num_merged_tokens = new_sample['input_ids'].size(0) + buffer['input_ids'].size(0)

                if num_merged_tokens <= self.max_packed_tokens:
                    find = True
                    find_idx = buffer_idx
                    break

                if self.allow_overflow and len(buffer_list) >= self.max_buffer_size // 2:
                    find = True
                    find_idx = buffer_idx

        if find:
            return buffer_list.pop(find_idx)
        return None

    def update_buffer(self, buffer, new_sample):
        """ update buffer """
        if buffer is None:
            new_sample['data_index'] = torch.zeros_like(new_sample['input_ids'])
            return new_sample

        new_sample['data_index'] = torch.ones_like(new_sample['input_ids']) + buffer['data_index'][-1].item()

        assert buffer.keys() == new_sample.keys(), f'{buffer.keys()=}, {new_sample.keys()=}'
        for k in buffer:
            buffer[k] = torch.cat([buffer[k], new_sample[k]])
        return buffer

    @staticmethod
    def check_valid(sample_to_check, min_active_tokens_ratio=1 / 256):
        """ check valid """
        num_ignore_tokens = (sample_to_check['labels'] == IGNORE_TOKEN_ID).sum()
        num_tokens = sample_to_check['labels'].numel()
        return (1 - num_ignore_tokens / num_tokens) > min_active_tokens_ratio

    @staticmethod
    def split_buffer(buffer, max_tokens, img_start_token_id, img_token_id, img_end_token_id):
        """ split buffer """
        if buffer['input_ids'].size(0) <= max_tokens:
            return [buffer]

        def _image_is_splitted(input_ids, cut_idx):
            is_image_start = input_ids[cut_idx].item() == img_start_token_id
            is_image_token = input_ids[cut_idx].item() == img_token_id
            is_image_end = input_ids[cut_idx].item() == img_end_token_id
            return is_image_start or is_image_token or is_image_end

        def _split(sample_to_split, left_idx, right_idx, left_img_idx, right_img_idx):
            assert (right_idx is None) == (right_img_idx is None)

            left_sample = {}
            right_sample = {} if right_idx is not None else None
            for k in sample_to_split:
                if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'data_index', 'type_ids',
                         'replacement', 'data_idx', 'real_idx']:
                    left_sample[k] = sample_to_split[k][:left_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_idx:]
                elif k in ['pixel_values', 'image_flags']:
                    left_sample[k] = sample_to_split[k][:left_img_idx]
                    if right_sample is not None:
                        right_sample[k] = sample_to_split[k][right_img_idx:]
                else:
                    raise NotImplementedError(f'find unsupported keys: {k} from {sample_to_split.keys()}')
            return left_sample, right_sample

        splitted_buffer = []
        while buffer['input_ids'].size(0) > max_tokens:
            img_start_idx_list = (buffer['input_ids'] == img_start_token_id).nonzero().squeeze(1).tolist()
            img_end_idx_list = (buffer['input_ids'] == img_end_token_id).nonzero().squeeze(1).tolist()
            assert len(img_start_idx_list) == len(img_end_idx_list)

            if _image_is_splitted(buffer['input_ids'], max_tokens):
                cut_idx = bisect.bisect_left(img_start_idx_list, max_tokens)
                if buffer['input_ids'][max_tokens] == img_start_token_id:
                    assert max_tokens == img_start_idx_list[cut_idx]
                    cut_left_idx = img_start_idx_list[cut_idx]
                    cut_left_img_idx = cut_idx
                else:
                    cut_left_idx = img_start_idx_list[cut_idx - 1]
                    cut_left_img_idx = cut_idx - 1
                cut_right_idx = cut_left_idx
                cut_right_img_idx = cut_left_img_idx
            else:
                cut_img_idx = bisect.bisect(img_start_idx_list, max_tokens)
                if cut_img_idx < len(img_start_idx_list):
                    cut_right_idx = img_start_idx_list[cut_img_idx]
                    cut_right_img_idx = cut_img_idx
                else:
                    cut_right_idx = None
                    cut_right_img_idx = None

                cut_left_idx = max_tokens
                cut_left_img_idx = cut_right_img_idx if cut_right_img_idx is not None else buffer['pixel_values'].size(
                    0)

            left, right = _split(
                sample_to_split=buffer,
                left_idx=cut_left_idx,
                left_img_idx=cut_left_img_idx,
                right_idx=cut_right_idx,
                right_img_idx=cut_right_img_idx,
            )

            assert (left['input_ids']
                    == img_end_token_id).sum() == (left['input_ids']
                                                   == img_start_token_id).sum() == left['pixel_values'].size(0)
            if right is not None:
                assert (right['input_ids']
                        == img_end_token_id).sum() == (right['input_ids']
                                                       == img_start_token_id).sum() == right['pixel_values'].size(0)

            if left['pixel_values'].size(0) >= 1 and PackedDataset.check_valid(left):
                splitted_buffer.append(left)

            if right is None or right['pixel_values'].size(0) == 0:
                break

            buffer = right
            if buffer['input_ids'].size(0) <= max_tokens and PackedDataset.check_valid(buffer):
                splitted_buffer.append(buffer)
                break

        logger.info(f'split a sample into {len(splitted_buffer)} samples, '
                    f'current max_tokens={max_tokens}')
        return splitted_buffer

    def update_buffer_list(self, buffer_list, buffer_max_len_list, buffer):
        """ update buffer list """
        # NOTE: in-place operation

        splitted_buffer = PackedDataset.split_buffer(
            buffer=buffer,
            max_tokens=self.max_packed_tokens,
            img_start_token_id=self.img_start_token_id,
            img_token_id=self.img_token_id,
            img_end_token_id=self.img_end_token_id,
        )

        for each_buffer in splitted_buffer:
            if each_buffer['pixel_values'].size(0) > self.num_images_expected:
                logger.error(f"Find a sample with {each_buffer['pixel_values'].size(0)} images, "
                             f'which exceeds {self.num_images_expected}')
                continue

            if each_buffer['input_ids'].size(0) >= self.max_packed_tokens:
                assert each_buffer['input_ids'].size(0) == self.max_packed_tokens
                buffer_max_len_list.append(each_buffer)
                continue

            find_idx = len(buffer_list)
            num_images_new_sample = each_buffer['pixel_values'].size(0)
            for buffer_idx in range(len(buffer_list)):
                if buffer_list[buffer_idx]['pixel_values'].size(0) < num_images_new_sample:
                    find_idx = buffer_idx
                    break
            buffer_list.insert(find_idx, each_buffer)

        for i in range(1, len(buffer_list)):
            assert buffer_list[i - 1]['pixel_values'].size(0) >= buffer_list[i]['pixel_values'].size(0)

        return buffer_list, buffer_max_len_list

    def pad_buffer(self, buffer):
        """ pad buffer """
        if buffer['pixel_values'].size(0) == self.num_images_expected:
            return buffer

        num_pad_images = self.num_images_expected - buffer['pixel_values'].size(0)
        pad_images = torch.stack([torch.zeros_like(buffer['pixel_values'][0]) for _ in range(num_pad_images)])
        pad_image_flags = torch.tensor([0] * num_pad_images, dtype=torch.long)

        buffer['pixel_values'] = torch.cat([buffer['pixel_values'], pad_images])
        buffer['image_flags'] = torch.cat([buffer['image_flags'], pad_image_flags])

        return buffer

    def save_state(self):
        """Saves the rng state for torch, numpy and random."""
        return dict({
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'random': random.getstate(),
        })

    def restore_state(self, state) -> None:
        """Restores the rng state for torch, numpy and random."""
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['numpy'])
        random.setstate(state['random'])

    def update_save_step(self):
        """ get next save step and update self.save_step """
        next_save_step = 0
        while (next_save_step == 0):
            if self.current_save_iteration >= self.exit_iteration:
                break
            self.current_save_iteration += self.save_ckpt_interval
            self.current_save_iteration = min(self.current_save_iteration, self.exit_iteration)
            num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers
            assert num_workers == self.num_workers, \
                f"num_workers should be equal to {self.num_workers}, but got {num_workers}"
            next_save_step = ((self.global_batch_size // self.data_world_size * self.current_save_iteration)
                              // num_workers)
            interval_leave = ((self.global_batch_size // self.data_world_size * self.current_save_iteration)
                              % num_workers)
            if interval_leave != 0:
                worker_id = 0 if get_worker_info() is None else get_worker_info().id
                if worker_id < interval_leave:
                    next_save_step += 1

        self.save_step = next_save_step

    def convert_buffer_to_idx(self, buffer_list=None):
        """ convert each data in the buffer list to its dataset_id and data_idx in the dataset """
        if buffer_list is None:
            return None

        return [torch.cat((buffer['type_ids'].unsqueeze(0), buffer['data_idx'].unsqueeze(0),
             buffer['replacement'].unsqueeze(0), buffer['real_idx'].unsqueeze(0)), 0)
                for buffer in buffer_list]

    def recover_buffer_data_form_idx(self, data_idxs):
        """ recover buffer base on dataset_id and data_idx in the dataset """
        buffer = None
        for i, index in enumerate(data_idxs):
            try:
                rng_state = self.rng_states[index[:3]]
            except KeyError as e:
                print(e)
                print(f"worker {self.worker_id} {index=} is not found in self.rng_states:{self.rng_states} ")
                raise e
            self.datasets[index[0]].set_rng_state(rng_state)
            data_idx = index[1]
            if len(index) == 4:
                data_idx = index[3]
            data = self.datasets[index[0]][data_idx]
            data['type_ids'] = torch.zeros_like(data['input_ids']) + index[0]
            data['data_idx'] = torch.zeros_like(data['input_ids']) + index[1]
            data['replacement'] = torch.zeros_like(data['input_ids']) + index[2]
            data.pop('meta_info', {})
            data.pop('rng_state', {})
            buffer = self.update_buffer(buffer, data)
        return buffer

    def get_unique_idx(self, idx):
        """ deduplicate idx and preserve the original order"""
        unique_idxs = list(dict.fromkeys(tuple(single.tolist()) for single in idx.T))
        return unique_idxs

    def convert_idx_to_buffer(self, buffer_idx_list=None):
        """ convert buffer_idx to buffer_list"""
        if buffer_idx_list is None:
            return None
        buffer_list = []
        for buffer_idx in buffer_idx_list:
            unique_idx = self.get_unique_idx(buffer_idx)
            buffer_list.append(self.recover_buffer_data_form_idx(unique_idx))

        return buffer_list

    def save_state_dict_to_disk(self, iteration):
        """ save dataset state dict to disk """
        if not (self.tp == 0 and self.pp == 0):
            return
        if self.worker_id % self.num_workers == 0:
            logging.info(f"worker {self.worker_id} begin to save dataset state dict to {self.save_ckpt_path} ...")
            logging.info(
                f"worker {self.worker_id} {self.current_save_iteration=}, {self.current_step=}, {self.save_step=}")
        data_ckpt_dir = os.path.join(self.save_ckpt_path,
                                    "dataset_ckpt",
                                    f"iter_{iteration:07d}")
        os.makedirs(data_ckpt_dir, exist_ok=True)
        torch.save(self._state_dict, os.path.join(data_ckpt_dir, f"{self.worker_state_key}.pt"))
        if self.worker_id % self.num_workers == 0:
            logging.info(f"worker {self.worker_id} successfully save dataset state dict to {self.save_ckpt_path}")

    def delete_idx_rng_state(self, data):
        """ when the data pop from buffer_list, delete it's rng state from self.rng_states"""
        idxs = torch.cat(
            (data['type_ids'].unsqueeze(0), data['data_idx'].unsqueeze(0), data['replacement'].unsqueeze(0)), 0)
        indexs = self.get_unique_idx(idxs)
        for index in indexs:
            del self.rng_states[index]

    def build_custom_infos(self, custom_infos=None):
        """ build dataset state infos for saving """
        if custom_infos is not None:
            custom_infos = {
                self.worker_state_key: {
                    'buffer_list': self.convert_buffer_to_idx(custom_infos['buffer_list']),
                    'buffer_max_len_list': custom_infos['buffer_max_len_list'],
                    'replacement_counts': self.replacement_counts,
                    'dataset_pop': self.dataset_pop,
                    'current_step': self.current_step,
                    'iter_idx': self.iter_idx + 1,
                    'rng_states': self.rng_states,
                    'rng': self.rng,
                    'current_rng_state': self.save_state()
                }
            }
        return custom_infos

    def save_and_find_next_save_step(self):
        """ save dataset state dict to disk and find the next save step"""
        if self.save_step == -1:
            raise Exception('PackedDataset params[save_step] is -1, no initialization')
        pre_save_step = self.save_step
        # Each time look for the next step to be saved, the number of iterations will be
        # increased for self.save_ckpt_interval(essentially increasing the total number of
        # samples that need to be trained).If the value after self.save_step is updated,
        # it is still the same as before the update, save state for this iteration
        # and continue to search for the next save step until find a different save_step.
        while self.save_step == pre_save_step:
            self.save_state_dict_to_disk(self.current_save_iteration)
            # update save steps
            self.update_save_step()
            if pre_save_step == self.exit_iteration:
                break

    def save_dataset_state_dict(self, buffer, custom_infos=None):
        """ save dataset state """
        self.current_step += 1
        if self.save_dataset_state:
            # delete the returned data rng status
            self.delete_idx_rng_state(buffer)
            if self.current_step == self.save_step:
                # build saved infos
                self._state_dict['custom_infos'] = self.build_custom_infos(custom_infos)
                # save state and find the next step that need to save
                self.save_and_find_next_save_step()

    def postprocess_buffer(self, buffer, custom_infos=None):
        """ postprocess buffer """

        # if custom_infos is not None:
        #     buffer['custom_infos'] = {self.worker_state_key: copy.deepcopy(custom_infos)}
        self.save_dataset_state_dict(buffer, custom_infos)
        buffer['worker_state_key'] = self.worker_state_key
        buffer['worker_state_dict'] = self._state_dict

        return buffer

    def print_log(self, iter_idx, buffer_list):
        """ print log """
        if iter_idx % self.log_freq != 0:
            return

        if self._should_log():
            logger.info(f"{iter_idx=}, {len(buffer_list)=}, {self._state_dict['sample_info']}")

    def load_state_dict_from_disk(self):
        """ load packed dataset state dict from disk """
        if self.train_iteration == 0:
            return None

        data_ckpt_path = os.path.join(self.load_ckpt_path,
                                      "dataset_ckpt",
                                      f"iter_{self.train_iteration:07d}",
                                      f"{self.worker_state_key}.pt")

        if os.path.exists(data_ckpt_path):
            try:
                dataset_state_dict = torch.load(data_ckpt_path, map_location="cpu")
                if self.worker_id % self.num_workers == 0:
                    logger.info(f"worker {self.worker_id} restoring dataset state from {data_ckpt_path} ...")
                return dataset_state_dict
            except Exception as e:
                logger.info("loading dataset state failed. Skipping. " + str(e))
        else:
            logger.info(f"dataset state {data_ckpt_path} does not exist")

        return None

    def __iter__(self):
        self.iter_idx = 0
        buffer_list = []
        buffer_max_len_list = []

        if self._should_log():
            logger.info(f'Begin to iter, {len(buffer_list)=}')

        worker_id = 0 if get_worker_info() is None else get_worker_info().id
        num_workers = 1 if get_worker_info() is None else get_worker_info().num_workers

        worker_id = num_workers * self.data_rank + worker_id
        num_workers = num_workers * self.data_world_size

        self.rng = np.random.default_rng(seed=worker_id)

        # reset states of each dataset
        self.worker_id = worker_id
        self.worker_state_key = f'work_state_{self.worker_id}'
        self.datasets = [d for d in self.datasets_orig]
        self.dataset_weight = [w for w in self.dataset_weight_orig]
        self.dataset_iter_list = [iter(d) for d in self.datasets]

        for ds in self.datasets:
            # if not isinstance(ds, (ImageTextPairDataset, InterleavedDataset)):
            ds.worker_id = worker_id
            ds.worker_state_key = f'work_state_{self.worker_id}'
            ds.num_workers = num_workers
            ds.init_dataset()
            if self._should_log() and worker_id == 0:
                logger.info(f'set worker_id and num_workers of {ds.__class__.__name__} {ds.ds_name}')
        if all(len(ds) == 0 for ds in self.datasets):
            raise Exception(f'All datasets of current worker[{self.worker_id}] are empty, please check your dataset. '
                            f'Or adjust the parameter num_workers，current num_workers is {self.num_workers}')

        # try resume dataset status
        if self.save_dataset_state:
            self.update_save_step()
            state_dict_load = self.load_state_dict_from_disk()
            if state_dict_load is not None:
                self.load_state_dict(state_dict_load)

        if self.worker_custom_infos is not None and self.worker_state_key in self.worker_custom_infos:
            custom_infos = self.worker_custom_infos[self.worker_state_key]
            # buffer list
            if 'buffer_list' in custom_infos and isinstance(custom_infos['buffer_list'], list):
                try:
                    self.rng_states = custom_infos['rng_states']
                    self.iter_idx = custom_infos['iter_idx']
                    self.replacement_counts = custom_infos['replacement_counts']
                    self.dataset_pop = custom_infos['dataset_pop'] if 'dataset_pop' in custom_infos else []
                    for index in self.dataset_pop:
                        self.dataset_weight[index] = 0
                    buffer_list = self.convert_idx_to_buffer(custom_infos['buffer_list'])
                    buffer_max_len_list = custom_infos['buffer_max_len_list']
                    self.current_step = custom_infos['current_step']
                    self.rng = custom_infos['rng']
                    self.restore_state(custom_infos['current_rng_state'])
                    if self._should_log() and worker_id == 0:
                        logger.info(f'[{self.worker_state_key}] load buffer list --> {len(buffer_list)=}')
                except Exception as e:
                    print(e)
                    logger.info(f'Worker {worker_id} custom_infos.keys:{custom_infos.keys()}')
                    raise e
            # other infos

            # reset
            self.worker_custom_infos = None

        logger.debug(f'{self.__class__.__name__} Rank {self.data_rank} '
                     f'Worker {worker_id} begin to load data')

        while True:
            self.dataset_weight = [w / sum(self.dataset_weight) for w in self.dataset_weight]
            current_dataset_idx = self.rng.choice(len(self.dataset_iter_list), p=self.dataset_weight)

            try:
                current_sample = self.next_data(current_dataset_idx)
            except Exception as e:
                logger.info(f'All datasets are exhausted, begin to empty the buffer_list ({len(buffer_list)=})')
                print(e)
                traceback.print_exc()
                while len(buffer_list) > 0:
                    if self.strict_mode:
                        yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(0)),
                                                      {'buffer_list': buffer_list,
                                                   'buffer_max_len_list': buffer_max_len_list})
                    else:
                        yield self.postprocess_buffer(buffer_list.pop(0),
                                                      {'buffer_list': buffer_list,
                                                   'buffer_max_len_list': buffer_max_len_list})
                logger.info(f'buffer_list is empty! ({len(buffer_list)=})')
                return

            buffer = self.find_buffer(buffer_list, current_sample)
            buffer = self.update_buffer(buffer, current_sample)
            buffer_list, buffer_max_len_list = self.update_buffer_list(buffer_list, buffer_max_len_list, buffer)

            while len(buffer_max_len_list) > 0:
                if buffer_max_len_list[0]['pixel_values'].size(0) != self.max_packed_tokens:
                    logger.debug(f'num tokens of a buffer exceed {self.max_packed_tokens=}, '
                                 f"yield a sample with {buffer_max_len_list[0]['pixel_values'].size(0)} images")
                if self.strict_mode and buffer_max_len_list[0]['pixel_values'].size(0) != self.num_images_expected:
                    # buffer_max_len_list.pop(0)
                    yield self.postprocess_buffer(self.pad_buffer(buffer_max_len_list.pop(0)),
                                                  {'buffer_list': buffer_list,
                                                   'buffer_max_len_list': buffer_max_len_list})
                else:
                    yield self.postprocess_buffer(buffer_max_len_list.pop(0),
                                                  {'buffer_list': buffer_list,
                                                   'buffer_max_len_list': buffer_max_len_list})

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'].size(0) > self.num_images_expected:
                logger.error(f"num images of a buffer ({buffer_list[0]['pixel_values'].size(0)}) "
                             f'is larger than num_images_expected({self.num_images_expected})')
                throw_buffer = buffer_list.pop(0)
                self.delete_idx_rng_state(throw_buffer)

            while len(buffer_list) > 0 and buffer_list[0]['pixel_values'].size(0) == self.num_images_expected:
                if self.debug_mode:
                    debug_data = self.postprocess_buffer(buffer_list.pop(0),
                                                         {'buffer_list': buffer_list,
                                                          'buffer_max_len_list': buffer_max_len_list})
                    while True:
                        yield debug_data.copy()

                yield self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list,
                                                                   'buffer_max_len_list': buffer_max_len_list})

            while len(buffer_list) > self.max_buffer_size:
                logger.debug(f'Failed to pack data to exactly {self.num_images_expected} images, '
                             f"yield a data sample with {buffer_list[0]['pixel_values'].size(0)} images.")
                if self.strict_mode:
                    yield self.postprocess_buffer(self.pad_buffer(buffer_list.pop(0)),
                                                  {'buffer_list': buffer_list,
                                                   'buffer_max_len_list': buffer_max_len_list})
                else:
                    yield self.postprocess_buffer(buffer_list.pop(0), {'buffer_list': buffer_list,
                                                                       'buffer_max_len_list': buffer_max_len_list})

            self.print_log(iter_idx=self.iter_idx, buffer_list=buffer_list)
            self.iter_idx += 1

    @staticmethod
    def get_cu_seqlens_and_indexes(
        data_index: torch.LongTensor,  # (seq_len,)
        input_ids: torch.LongTensor,  # (seq_len,)
        labels: torch.LongTensor,  # (seq_len,)
        len2weight: callable,
    ):
        """ get cu_seqlens and indexes """
        indexes = []
        cu_seqlens = [0]
        loss_weight = []

        start = data_index.min()
        end = data_index.max() + 1
        for i in range(start, end):
            num_tokens = (data_index == i).sum().item()
            indexes.extend(list(range(num_tokens)))
            cu_seqlens.append(cu_seqlens[-1] + num_tokens)
            assert num_tokens > 0

            curr_data_index = data_index[cu_seqlens[-2]:cu_seqlens[-2] + num_tokens]
            assert (curr_data_index == i).all(), data_index

            curr_labels = labels[cu_seqlens[-2]:cu_seqlens[-2] + num_tokens]
            num_effective_tokens = (curr_labels != IGNORE_TOKEN_ID).sum().item()
            loss_weight.extend([len2weight(num_effective_tokens)] * num_tokens)

        assert len(indexes) == data_index.size(0), f'{len(indexes)=}, {data_index.size(0)=}'

        loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
        return cu_seqlens, indexes, loss_weight