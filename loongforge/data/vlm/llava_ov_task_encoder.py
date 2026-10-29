# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from LLaVA-OneVision-1.5 under the Apache-2.0 License.

"""LLavA-OneVision TaskEncoder class."""

import re
from typing import List, Tuple, Union, Optional, TypeVar

import numpy as np
import torch
from megatron.energon import VQASample

from transformers import AutoProcessor
from typing_extensions import override

from loongforge.data.vlm.length_sort_dataset import LengthPoolSortDataset
from loongforge.data import IGNORE_INDEX, TrainingPhase

from .base.task_encoder import (
    BaseTaskSample,
)
from .vlm_task_encoder import (
    VLMTaskEncoder,
    VLMTaskSample,
)
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    SavableDataset,
)
from megatron.energon.flavors.crude import CrudeWebdataset
from megatron.energon.metadataset.loader_interface import DatasetBlendMode
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BlendDataset,
    EpochizeDataset,
    LogSampleDataset,
    ShuffleBufferDataset,
)

# from .batch_dataset import BatchDataset
from megatron.energon.wrappers.repeat_dataset import RepeatDataset

T_sample = TypeVar("T_sample")
T_batch = TypeVar("T_batch")

IMAGE_TOKEN = "<|image_pad|>"
VIDEO_TOKEN = "<|video_pad|>"
VISION_TAGS = ["<|vision_start|>", "<|vision_end|>"]
IMAGE_TOKEN_WITH_TAGS = VISION_TAGS[0] + IMAGE_TOKEN + VISION_TAGS[1]


class LLavaOv15TaskEncoder(VLMTaskEncoder):
    """A task encoder for LLava OV 1.5 that extends VLMTaskEncoder."""

    def __init__(self, config, tokenizer, chat_template=None):
        super().__init__(config, tokenizer, chat_template)
        self.processor = AutoProcessor.from_pretrained(
            self.config.hf_tokenizer_path, trust_remote_code=True
        )

        if config.image_resolution:
            setattr(self.processor, "image_resolution", config.image_resolution)
        # video
        self.frame_min_pixels = config.frame_min_pixels
        self.frame_max_pixels = config.frame_max_pixels
        self.video_max_pixels = config.video_max_pixels
        self.fps = config.fps
        self.fps_min_frames = config.fps_min_frames
        self.fps_max_frames = config.fps_max_frames
        # image
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels

    def encode_vqa(self, sample: VQASample) -> BaseTaskSample:
        """Encode pretrain sample in Qwen2VL style."""
        if self.config.training_phase == TrainingPhase.PRETRAIN:
            if self.config.add_question_in_pretrain:
                text = (sample.context + sample.answers).replace(
                    "<image>", IMAGE_TOKEN_WITH_TAGS
                )
            else:
                text = IMAGE_TOKEN_WITH_TAGS + sample.answers
            text = text + self.tokenizer.tokenizer.eos_token
            input_ids, target, imgs, image_grid_thw, attn_mask = self._process(
                sample.image, text
            )
        elif self.config.training_phase == TrainingPhase.SFT:

            if len(sample.answers) < 1:
                raise ValueError("sample.answers < 1!")

            # Add image resize check for PIL.Image
            if sample.image is not None:

                img_arr = np.array(sample.image)
                if np.sum(img_arr) == 0:
                    raise ValueError("Image pixels are all zero!")

            # Truncate answer to the last full sentence if it exceeds the max length.
            max_answer_length = self.config.training_rice_vl_max_answer_length
            if len(sample.answers) > max_answer_length:
                original_length = len(sample.answers)

                # Perform a preliminary cut at the maximum allowed length.
                preliminary_cut = sample.answers[:max_answer_length]

                # Clean up trailing punctuation and whitespace from the preliminary cut
                cleaned_cut = preliminary_cut.rstrip(".。 \t\n")

                # Find the last occurrence of a sentence-ending punctuation mark
                # followed by a space or the end of the string.
                # This pattern looks for sentence enders (. or 。)
                sentence_enders_pattern = r"[.。]"

                # Find all matches and get the end position of the last match
                matches = list(re.finditer(sentence_enders_pattern, cleaned_cut))

                if matches:
                    # Get the end position of the last match
                    last_end_index = matches[-1].end()
                    # Truncate at the end of the last full sentence.
                    sample.answers = cleaned_cut[:last_end_index]
                else:
                    # Fallback to a hard cut of the original preliminary string if no sentence ender is found.
                    sample.answers = preliminary_cut

                print(
                    f"Answer truncated to a full sentence. "
                    f"Original length: {original_length}, New length: {len(sample.answers)}"
                )

            text = self.processor.apply_chat_template(
                [
                    {"role": "user", "content": sample.context},
                    {"role": "assistant", "content": sample.answers},
                ],
                tokenize=False,
            ).replace("<image>", IMAGE_TOKEN_WITH_TAGS)
            if text[-1] == "\n":
                text = text[:-1]
            input_ids, _, imgs, image_grid_thw, attn_mask = self._process(
                sample.image, text, add_special_tokens=False
            )
            target = torch.ones_like(input_ids) * IGNORE_INDEX
            answers = self.tokenizer.tokenize(sample.answers, add_special_tokens=False)
            target[-len(answers) - 1 : -1] = torch.tensor(answers)
            target[-1] = input_ids[-1]
            # print(target[-1])
        else:
            raise NotImplementedError(
                f"Unknown training phase {self.config.training_phase}"
            )

        num_tiles = [len(image_grid_thw)]

        if self.config.enable_discard_sample:
            assert (
                len(input_ids) <= self.config.seq_length
            ), f"{sample.__key__} input length {len(input_ids)}"
        else:
            assert (
                image_grid_thw.prod() / 4 <= self.config.seq_length
            ), f"{sample.__key__} grid_thw: {image_grid_thw}"

        return VLMTaskSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavor__=None,
            __subflavors__=sample.__subflavors__,
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles,
            tokens=input_ids,
            labels=target,
            attn_mask=attn_mask,
            total_len=len(input_ids),
        )

    @override
    def build_train_datasets(
        self,
        *,
        datasets: List[
            Tuple[BaseCoreDatasetFactory[T_sample], Union[float, int, None]]
        ],
        worker_config: WorkerConfig,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
        blend_mode: DatasetBlendMode = DatasetBlendMode.NONE,
        repeat: bool = True,
    ) -> SavableDataset[T_batch]:
        """Combines train datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset, _ in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        global_workers = max(1, worker_config.num_workers) * worker_config.world_size
        rotation_lengths = [len(dataset) for dataset, _ in datasets]
        for i in range(1, len(rotation_lengths)):
            rotation_lengths[i] += rotation_lengths[i - 1]
        worker_rotation_offsets = [
            rotation_length % global_workers
            for rotation_length in [0] + rotation_lengths[:-1]
        ]

        if repeat:
            inner_datasets = [
                (
                    RepeatDataset(
                        dataset.build(worker_rotation_offset=worker_rotation_offset),
                        worker_config=worker_config,
                    ),
                    1.0 if weight is None else float(weight),
                )
                for (dataset, weight), worker_rotation_offset in zip(
                    datasets, worker_rotation_offsets
                )
            ]
        else:
            assert blend_mode in (
                DatasetBlendMode.NONE,
                DatasetBlendMode.SAMPLE_REPETITIONS,
            ) and all(
                isinstance(repetitions, int) for _dataset, repetitions in datasets
            ), "If repeat is False, the datasets must be repeated with integer weights."
            inner_datasets = [
                (
                    (
                        dataset.build(worker_rotation_offset=worker_rotation_offset)
                        if repetition is None or repetition == 1
                        else RepeatDataset(
                            dataset.build(
                                worker_rotation_offset=worker_rotation_offset
                            ),
                            repeats=int(repetition),
                            worker_config=worker_config,
                        )
                    ),
                    len(dataset) * (1 if repetition is None else int(repetition)),
                )
                for (dataset, repetition), worker_rotation_offset in zip(
                    datasets, worker_rotation_offsets
                )
            ]

        if len(inner_datasets) > 1:
            # The worker offset for each dataset is the cumsum of the dataset lengths, but modulo the
            # global number of workers.
            dataset = BlendDataset(
                *inner_datasets,
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = inner_datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        if shuffle_buffer_size is not None and shuffle_buffer_size > 1:
            dataset = ShuffleBufferDataset(
                dataset,
                size=shuffle_buffer_size,
                worker_config=worker_config,
            )
        dataset = self.build_cook_crude_sample(dataset, worker_config=worker_config)
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)

        # Insert pool sorting before entering BatchDataset
        if (
            getattr(self.config, "length_sort_pool_size", 0)
            and self.config.length_sort_pool_size > 0
        ):
            dataset = LengthPoolSortDataset(
                dataset,
                pool_size=self.config.length_sort_pool_size,
                key_fn=lambda s: getattr(s, "total_len", len(getattr(s, "tokens"))),
                ascending=not getattr(self.config, "length_sort_desc", False),
                worker_config=worker_config,
            )
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        if worker_config.should_log(level=1):
            dataset = LogSampleDataset(
                dataset, mode="train", worker_config=worker_config
            )
        return dataset
