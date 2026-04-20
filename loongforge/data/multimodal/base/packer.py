# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Packing strategys"""

from typing import List
import sys
import bisect


def _img_count(imgs) -> int:
    """Return number of images whether stored as tensor or list."""
    if imgs is None:
        return 0
    if hasattr(imgs, "shape"):
        try:
            return imgs.shape[0]
        except Exception:
            pass
    try:
        return len(imgs)
    except Exception:
        return 0


class Buffer:
    """A buffer to store samples"""

    def __init__(self, capacity, img_limit):
        """Buffer Init function"""
        self.capacity = capacity
        self.img_limit = img_limit
        self.packed_data_len = 0
        self.img_count = 0
        self.data = []

    def can_fit(self, sample):
        """Check if the sample can fit into the buffer"""
        satisfy_cap = self.packed_data_len + sample.tokens.shape[0] <= self.capacity
        satisfy_img = True
        if self.img_limit > 0:
            satisfy_img = self.img_count + _img_count(sample.imgs) <= self.img_limit
        return satisfy_img and satisfy_cap

    def insert(self, sample):
        """Insert a sample into the buffer"""
        self.data.append(sample)
        self.packed_data_len += sample.tokens.shape[0]
        self.img_count += _img_count(sample.imgs)


class Packer:
    """Pack samples into buffers according to the configured packing algorithm"""

    def __init__(self, args):
        """Packer Init function"""
        self.args = args

    def pack(
        self,
        samples: List,
        buffer_capacity: int,
        img_limit: int = sys.maxsize,
        buffers_num: int = 1,
    ):
        """
        Pack samples into buffers according to the configured packing algorithm

        Args:
            samples: List of samples to be packed
            buffer_capacity: Capacity of a single buffer
            img_limit: Maximum number of images allowed per buffer, defaults to sys.maxsize
            buffers_num: Number of buffers, defaults to 1

        Returns:
            List of packed buffers, each containing a group of samples
        """
        if self.args.energon_pack_algo == "balanced":
            return self.balanced_greedy_knapsack(samples, buffer_capacity, img_limit)
        elif self.args.energon_pack_algo == "sequential":
            return self.sequential_greedy_knapsack(
                samples, buffer_capacity, img_limit, buffers_num)
        elif self.args.energon_pack_algo == "sequential_max_images":
            return self.sequential_greedy_knapsack(
                samples, buffer_capacity, img_limit, buffers_num,
                prioritize_image_count=True)
        else:
            raise ValueError(f"Invalid energon_pack_algo: {self.args.energon_pack_algo}, \
                    only supports balanced/sequential/sequential_max_images")

    # Based on https://github.com/hiyouga/LLaMA-Factory/
    #          blob/641d0dab08d96a93c34657742213d8994d9ed476/src/llamafactory/data/processors/processor_utils.py#L19
    def search_for_fit(
        self, numbers: List[int], img_nums: List[int], capacity: int, img_num: int
    ) -> int:
        """
        Finds the largest sample index that fits both capacity and image number constraints.

        Args:
            numbers: Sorted list of sample lengths
            img_nums: Corresponding list of image counts per sample
            capacity: Remaining capacity of the current knapsack
            img_num: Remaining image number limit of the current knapsack

        Returns:
            int: Index of the largest fitting sample, or -1 if no sample fits
        """
        index = bisect.bisect(numbers, capacity)
        if index != 0:
            while index > 0:
                if img_nums[index - 1] <= img_num:
                    return index - 1
                else:
                    index -= 1
        return -1

    def balanced_greedy_knapsack(
        self, samples: List, buffer_capacity: int, img_limit: int
    ) -> List:
        """
        Pack samples into buffers using balanced greedy knapsack algorithm.

        This algorithm sorts samples by length and uses a greedy strategy to pack
        as many samples as possible into each buffer while satisfying both buffer
        capacity and image number constraints. This algorithm balances the load while
        packing samples.

        Args:
            samples: List of samples to be packed
            buffer_capacity: Maximum capacity of a single buffer
            img_limit: Maximum number of images allowed per buffer

        Returns:
            List: List of packed buffers, each containing a group of samples

        Raises:
            ValueError: If any sample exceeds buffer capacity or image limit
        """
        lengths = [sample.total_len for sample in samples]
        img_nums = [
            _img_count(sample.imgs) for sample in samples
        ]
        assert len(lengths) == len(
            samples
        ), "sample lengths and samples must have the same length."

        knapsacks = []

        if len(samples) == 0:
            return knapsacks

        # Sort sample lengths and samples together.
        sorted_lengths, sorted_samples, sorted_img_nums = zip(
            *sorted(zip(lengths, samples, img_nums), key=lambda x: x[0])
        )
        sorted_lengths = list(sorted_lengths)
        sorted_samples = list(sorted_samples)
        sorted_img_nums = list(sorted_img_nums)

        # Check if all samples fit in the knapsack capacity.
        if sorted_lengths[-1] > buffer_capacity:
            raise ValueError(
                f"Knapsack item '{sorted_samples[-1].__key__}' exceeds max sequence length:"
                f"{sorted_lengths[-1]} > {buffer_capacity}."
            )

        if max(sorted_img_nums) > img_limit:
            index = sorted_img_nums.index(max(sorted_img_nums))
            raise ValueError(
                f"Knapsack item '{sorted_samples[index].__key__}' exceeds image limit:"
                f"{sorted_img_nums[index]} > {img_limit}."
            )

        while sorted_lengths:
            current_knapsack = []
            remaining_capacity = buffer_capacity
            remaining_img_num = img_limit

            while True:
                idx = self.search_for_fit(
                    sorted_lengths,
                    sorted_img_nums,
                    remaining_capacity,
                    remaining_img_num,
                )
                if idx == -1:
                    break  # Can't fit more samples.

                remaining_capacity -= sorted_lengths[idx]
                remaining_img_num -= sorted_img_nums[idx]

                sorted_lengths.pop(idx)
                sorted_img_nums.pop(idx)
                sample = sorted_samples.pop(idx)
                current_knapsack.append(sample)

            knapsacks.append(current_knapsack)

        return knapsacks

    def sequential_greedy_knapsack(
        self,
        samples: List,
        buffer_capacity: int,
        img_limit: int = 0,
        buffers_num: int = 1,
        prioritize_image_count: bool = False
    ) -> List:
        """
        Pack samples into buffers using sequential greedy knapsack algorithm.

        This algorithm maintains multiple buffers and processes samples sequentially,
        placing each sample into the first buffer that can accommodate it. If no buffer
        can fit the current sample, the oldest buffer is added to the result list and
        a new buffer is created. When prioritize_image_count=True, the buffer prioritizes
        packing the maximum number of images, potentially reordering the original samples. 
        This algorithm aims to minimizes training sequence disruption during sample packing.

        Args:
            samples: List of samples to be packed
            buffer_capacity: Maximum capacity of a single buffer
            img_limit: Maximum number of images allowed per buffer, defaults to 0
            buffers_num: Number of buffers to maintain, defaults to 1
            prioritize_image_count: Whether to prioritize image count, defaults to False

        Returns:
            List: List of packed buffers, each containing a group of samples

        Raises:
            AssertionError: If sample token length exceeds buffer capacity or image count exceeds limit
        """
        buffers = [Buffer(buffer_capacity, img_limit) for _ in range(buffers_num)]
        packed_buffers = []
        for sample in samples:
            assert (sample.tokens.shape[0] <= buffer_capacity), f"sample token length: \
                {sample.tokens.shape[0]} > max buffer capacity: {buffer_capacity}, will skip this sample"
            # print(f"sample_len: {sample.total_len}, buffer_capacity: {buffer_capacity}, buffers_num: {buffers_num}")
            img_num = _img_count(sample.imgs)
            if img_num and img_limit:
                assert (img_num <= img_limit), f"sample img_num: {img_num} \
                > self.num_images_expected: {img_limit}, will skip this sample"

            packed = False
            for idx, buffer in enumerate(buffers):
                if buffer.can_fit(sample):
                    buffer.insert(sample)
                    packed = True
                    break

            # If there is no buffer to store the current sample,
            # add the buffer to packed_buffers and create a new buffer.
            if not packed:
                packed_buffers.append(buffers.pop(0).data)
                buffers.append(Buffer(buffer_capacity, img_limit))
                buffers[-1].insert(sample)
                # print(f"find: False, find_idx: 0, len: {sample.total_len}")

            # When prioritize-image-count=True, the buffer prioritizes packing the maximum 
            # number of images, potentially reordering the original samples.
            if prioritize_image_count:
                # sort from large to small
                buffers.sort(key=lambda x: x.img_count, reverse=True)
                # we pop the puffer if it is full
                for idx, buffer in enumerate(buffers):
                    if buffer.packed_data_len == buffer.capacity:
                        packed_buffers.append(buffers.pop(idx).data)
                        buffers.append(Buffer(buffer_capacity, img_limit))
                        break

                    if buffer.img_limit > 0:
                        if buffer.img_count == buffer.img_limit:
                            packed_buffers.append(buffers.pop(idx).data)
                            buffers.append(Buffer(buffer_capacity, img_limit))
                            break

        packed_buffers.extend(buffer.data for buffer in buffers if buffer.data)
        return packed_buffers