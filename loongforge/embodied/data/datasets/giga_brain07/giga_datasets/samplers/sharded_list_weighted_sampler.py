import itertools
import math
import os
from typing import Iterator

import numpy as np

from ..datasets import ConcatDataset
from .list_weighted_sampler import (
    ListWeightedSampler,
    _apply_coarse_group_embodiment_scaling_plan,
    _build_coarse_group_embodiment_scaling_plan,
    _debug_sampler,
    _is_embodiment_scaling_enabled,
    _normalize_embodiment_scaling_scope,
    _resolve_child_embodiment_names,
)


class ShardedListWeightedSampler(ListWeightedSampler):
    """ListWeightedSampler restricted to a data shard.

    ``shard_mode='global_range'`` keeps the legacy behavior: the full
    concatenated dataset is split into ``shard_world_size`` contiguous ranges.
    ``shard_mode='per_dataset'`` splits every top-level child dataset
    independently, so each shard can keep the original top-level sampling mix.
    ``shard_mode='group_balanced'`` splits top-level children inside coarse
    groups, keeping every shard close to the original group-level sampling mix
    while reducing the number of groups each machine has to touch.

    Each shard keeps the original ``ListWeightedSampler`` behavior inside its
    ranges, including per-child sampling weights, ``epoch_size`` and
    ``ratio_mode``.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        sampling_weights: list[float] | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
        infinite: bool = True,
        seed: int = 6666,
        ratio_mode: str = 'epoch',
        index_mode: str = 'array',
        epoch_size: int | None = None,
        oversample_mode: str = 'replacement',
        embodiment_scaling_exponent: float | None = None,
        embodiment_scaling_scope: str | None = None,
        embodiment_scaling_group_names: list[str] | tuple[str, ...] | None = None,
        embodiment_scaling_dataset_indices: list[int] | tuple[int, ...] | None = None,
        embodiment_scaling_balance_robot_types: bool = False,
        shard_world_size: int | None = None,
        shard_rank: int | None = None,
        num_processes: int | None = None,
        process_index: int | None = None,
        process_shard_size: int | None = None,
        process_batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        align_process_batches: bool = True,
        shard_mode: str = 'global_range',
        shard_group_names: list[str] | None = None,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError('dataset should be a ConcatDataset')

        full_sub_dataset_lengths = [len(d) for d in dataset.datasets]
        full_data_size = sum(full_sub_dataset_lengths)
        if full_data_size <= 0:
            raise ValueError('dataset should contain at least one sample')

        self.num_processes = self._resolve_positive_int(num_processes, ('WORLD_SIZE',), 1)
        self.process_index = self._resolve_non_negative_int(process_index, ('RANK',), 0)
        if self.process_index >= self.num_processes:
            raise ValueError(f'process_index should be < num_processes, got {self.process_index} >= {self.num_processes}')

        if shard_world_size is None:
            shard_world_size = self._infer_shard_world_size(self.num_processes, process_shard_size)
        self.shard_world_size = int(shard_world_size)
        if self.shard_world_size <= 0:
            raise ValueError('shard_world_size should be greater than 0')

        if process_shard_size is None:
            process_shard_size = self._infer_process_shard_size(self.num_processes, self.shard_world_size)
        self.process_shard_size = int(process_shard_size)
        if self.process_shard_size <= 0:
            raise ValueError('process_shard_size should be greater than 0')
        if self.process_shard_size * self.shard_world_size != self.num_processes:
            raise ValueError(
                'process_shard_size * shard_world_size should equal num_processes, '
                f'got {self.process_shard_size} * {self.shard_world_size} != {self.num_processes}'
            )

        if shard_rank is None:
            shard_rank = self.process_index // self.process_shard_size
        self.shard_rank = int(shard_rank)
        if self.shard_rank < 0 or self.shard_rank >= self.shard_world_size:
            raise ValueError(f'shard_rank should be in [0, {self.shard_world_size}), got {self.shard_rank}')

        self.shard_mode = str(shard_mode)
        if self.shard_mode not in {'global_range', 'per_dataset', 'group_balanced'}:
            raise ValueError('shard_mode should be one of "global_range", "per_dataset" or "group_balanced", ' f'got {self.shard_mode!r}')
        resolved_embodiment_scaling_scope = _normalize_embodiment_scaling_scope(embodiment_scaling_scope)

        if process_batch_size is None and batch_size is not None:
            gradient_accumulation_steps_for_batch = 1 if gradient_accumulation_steps is None else int(gradient_accumulation_steps)
            if gradient_accumulation_steps_for_batch <= 0:
                raise ValueError('gradient_accumulation_steps should be greater than 0')
            divisor = self.num_processes * gradient_accumulation_steps_for_batch
            if batch_size % divisor == 0:
                process_batch_size = batch_size // divisor

        self.process_batch_size = 0 if process_batch_size is None else int(process_batch_size)
        if self.process_batch_size < 0:
            raise ValueError('process_batch_size should be non-negative')
        if bool(align_process_batches) and self.process_batch_size <= 0:
            raise ValueError('process_batch_size is required when align_process_batches=True')
        self.align_process_batches = bool(align_process_batches)
        if self.align_process_batches:
            if batch_size is None:
                gradient_accumulation_steps = 1 if gradient_accumulation_steps is None else int(gradient_accumulation_steps)
            elif gradient_accumulation_steps is None:
                global_micro_batch_size = self.process_batch_size * self.num_processes
                if batch_size % global_micro_batch_size != 0:
                    raise ValueError(
                        'batch_size should be divisible by process_batch_size * num_processes '
                        f'when gradient_accumulation_steps is not set, got {batch_size} and {global_micro_batch_size}'
                    )
                gradient_accumulation_steps = batch_size // global_micro_batch_size
            else:
                gradient_accumulation_steps = int(gradient_accumulation_steps)
            if gradient_accumulation_steps <= 0:
                raise ValueError('gradient_accumulation_steps should be greater than 0')
            local_padding_batch_size = self.process_batch_size * self.process_shard_size * gradient_accumulation_steps
        else:
            local_padding_batch_size = batch_size

        shard_start, shard_end = self._split_range(full_data_size, self.shard_world_size, self.shard_rank)
        self.full_data_size = full_data_size
        self.shard_start = shard_start
        self.shard_end = shard_end
        sampling_weights_for_local_shard = sampling_weights
        embodiment_scaling_inner_ranges = None
        embodiment_scaling_inner_probabilities = None
        local_epoch_size_from_ranges = None
        full_sampling_weights = sampling_weights if sampling_weights is not None else getattr(dataset, 'sampling_weights', None)
        if full_sampling_weights is None:
            raise ValueError('sampling_weights is required, or dataset should provide sampling_weights')
        full_sampling_weights = [float(weight) for weight in full_sampling_weights]
        full_ranges = [(0, length) for length in full_sub_dataset_lengths]
        full_coarse_group_scaling_plan = None
        full_coarse_group_scaled_weights = full_sampling_weights
        if (
            _is_embodiment_scaling_enabled(
                embodiment_scaling_exponent,
                embodiment_scaling_balance_robot_types,
            )
            and resolved_embodiment_scaling_scope == 'coarse_group'
        ):
            full_coarse_group_scaling_plan = _build_coarse_group_embodiment_scaling_plan(
                dataset,
                full_sampling_weights,
                full_ranges,
                float(embodiment_scaling_exponent),
                embodiment_scaling_group_names,
                (None if embodiment_scaling_dataset_indices is None else {int(index) for index in embodiment_scaling_dataset_indices}),
                embodiment_scaling_balance_robot_types,
            )
            full_coarse_group_scaled_weights, _, _ = _apply_coarse_group_embodiment_scaling_plan(
                dataset,
                full_sampling_weights,
                full_ranges,
                full_coarse_group_scaling_plan,
            )
        if self.shard_mode == 'global_range':
            self.sub_dataset_ranges = self._global_range_to_sub_dataset_ranges(
                full_sub_dataset_lengths,
                shard_start,
                shard_end,
            )
            sampling_weights_for_local_shard = full_coarse_group_scaled_weights
        else:
            if self.shard_mode == 'per_dataset':
                if (
                    _is_embodiment_scaling_enabled(
                        embodiment_scaling_exponent,
                        embodiment_scaling_balance_robot_types,
                    )
                    and resolved_embodiment_scaling_scope == 'child'
                ):
                    self.sub_dataset_ranges, embodiment_scaling_inner_ranges = self._per_dataset_embodiment_to_sub_dataset_ranges(
                        dataset,
                        full_sub_dataset_lengths,
                        self.shard_world_size,
                        self.shard_rank,
                        embodiment_scaling_dataset_indices,
                    )
                    local_epoch_size_from_ranges = self._range_total_length(
                        self.sub_dataset_ranges,
                        embodiment_scaling_inner_ranges,
                    )
                else:
                    self.sub_dataset_ranges = self._per_dataset_to_sub_dataset_ranges(
                        full_sub_dataset_lengths,
                        self.shard_world_size,
                        self.shard_rank,
                    )
                sampling_weights_for_local_shard = full_coarse_group_scaled_weights
            else:
                group_names = self._resolve_group_names(dataset, shard_group_names)
                self.shard_group_names = group_names
                self.sub_dataset_ranges, sampling_weights_for_local_shard = self._group_balanced_to_sub_dataset_ranges(
                    full_sub_dataset_lengths,
                    full_coarse_group_scaled_weights,
                    group_names,
                    self.shard_world_size,
                    self.shard_rank,
                )
            if (
                _is_embodiment_scaling_enabled(
                    embodiment_scaling_exponent,
                    embodiment_scaling_balance_robot_types,
                )
                and resolved_embodiment_scaling_scope == 'coarse_group'
            ):
                (
                    _,
                    embodiment_scaling_inner_ranges,
                    embodiment_scaling_inner_probabilities,
                ) = _apply_coarse_group_embodiment_scaling_plan(
                    dataset,
                    sampling_weights_for_local_shard,
                    self.sub_dataset_ranges,
                    full_coarse_group_scaling_plan,
                )
        if (
            self.shard_mode == 'global_range'
            and _is_embodiment_scaling_enabled(
                embodiment_scaling_exponent,
                embodiment_scaling_balance_robot_types,
            )
            and resolved_embodiment_scaling_scope == 'coarse_group'
        ):
            (
                _,
                embodiment_scaling_inner_ranges,
                embodiment_scaling_inner_probabilities,
            ) = _apply_coarse_group_embodiment_scaling_plan(
                dataset,
                sampling_weights_for_local_shard,
                self.sub_dataset_ranges,
                full_coarse_group_scaling_plan,
            )
        if epoch_size is None:
            if self.shard_mode == 'group_balanced':
                local_epoch_size = shard_end - shard_start
                if local_epoch_size <= 0:
                    raise ValueError(
                        f'dataset size={full_data_size} is too small for shard_world_size={self.shard_world_size}; '
                        f'shard_rank={self.shard_rank} would receive no samples'
                    )
            elif local_epoch_size_from_ranges is not None:
                local_epoch_size = local_epoch_size_from_ranges
            else:
                local_epoch_size = None
        else:
            epoch_start, epoch_end = self._split_range(int(epoch_size), self.shard_world_size, self.shard_rank)
            local_epoch_size = epoch_end - epoch_start
            if local_epoch_size <= 0:
                raise ValueError(
                    f'epoch_size={epoch_size} is too small for shard_world_size={self.shard_world_size}; '
                    f'shard_rank={self.shard_rank} would receive no samples'
                )

        super().__init__(
            dataset=dataset,
            sampling_weights=sampling_weights_for_local_shard,
            batch_size=local_padding_batch_size,
            shuffle=shuffle,
            infinite=infinite,
            seed=seed,
            ratio_mode=ratio_mode,
            index_mode=index_mode,
            epoch_size=local_epoch_size,
            oversample_mode=oversample_mode,
            embodiment_scaling_exponent=embodiment_scaling_exponent,
            embodiment_scaling_scope=resolved_embodiment_scaling_scope,
            embodiment_scaling_group_names=embodiment_scaling_group_names,
            embodiment_scaling_dataset_indices=embodiment_scaling_dataset_indices,
            embodiment_scaling_balance_robot_types=embodiment_scaling_balance_robot_types,
            embodiment_scaling_inner_ranges=embodiment_scaling_inner_ranges,
            embodiment_scaling_inner_probabilities=embodiment_scaling_inner_probabilities,
            sub_dataset_ranges=self.sub_dataset_ranges,
        )

        self.consumed_total_size = super().__len__()
        self.aligned_total_size = self._compute_aligned_total_size()

    @staticmethod
    def _inner_range_total_length(ranges: list[tuple[int, int, str]] | None) -> int:
        if ranges is None:
            return 0
        return sum(end - start for start, end, _ in ranges)

    @staticmethod
    def _range_total_length(
        sub_dataset_ranges: list[tuple[int, int]],
        inner_ranges_by_dataset: list[list[tuple[int, int, str]] | None],
    ) -> int:
        total = 0
        for (start, end), inner_ranges in zip(sub_dataset_ranges, inner_ranges_by_dataset):
            if inner_ranges is None:
                total += end - start
            else:
                total += ShardedListWeightedSampler._inner_range_total_length(inner_ranges)
        return total

    @staticmethod
    def _resolve_positive_int(value: int | None, env_names: tuple[str, ...], default: int) -> int:
        if value is not None:
            return int(value)
        for env_name in env_names:
            env_value = os.environ.get(env_name)
            if env_value:
                return int(env_value)
        return int(default)

    @staticmethod
    def _resolve_non_negative_int(value: int | None, env_names: tuple[str, ...], default: int) -> int:
        resolved = ShardedListWeightedSampler._resolve_positive_int(value, env_names, default)
        if resolved < 0:
            raise ValueError(f'value should be non-negative, got {resolved}')
        return resolved

    @staticmethod
    def _infer_process_shard_size(num_processes: int, shard_world_size: int) -> int:
        if num_processes % shard_world_size != 0:
            raise ValueError(f'num_processes should be divisible by shard_world_size, got {num_processes} and {shard_world_size}')
        return num_processes // shard_world_size

    @staticmethod
    def _infer_shard_world_size(num_processes: int, process_shard_size: int | None) -> int:
        if process_shard_size is None:
            env_process_shard_size = os.environ.get('NPROC_PER_NODE') or os.environ.get('LOCAL_WORLD_SIZE')
            process_shard_size = int(env_process_shard_size) if env_process_shard_size else 1
        process_shard_size = int(process_shard_size)
        if process_shard_size <= 0:
            raise ValueError('process_shard_size should be greater than 0')
        if num_processes % process_shard_size != 0:
            raise ValueError(f'num_processes should be divisible by process_shard_size, got {num_processes} and {process_shard_size}')
        return num_processes // process_shard_size

    @staticmethod
    def _split_range(total_size: int, world_size: int, rank: int) -> tuple[int, int]:
        start = (total_size * rank) // world_size
        end = (total_size * (rank + 1)) // world_size
        return start, end

    @staticmethod
    def _global_range_to_sub_dataset_ranges(
        sub_dataset_lengths: list[int],
        shard_start: int,
        shard_end: int,
    ) -> list[tuple[int, int]]:
        ranges = []
        cursor = 0
        for length in sub_dataset_lengths:
            child_start = cursor
            child_end = cursor + length
            overlap_start = max(shard_start, child_start)
            overlap_end = min(shard_end, child_end)
            if overlap_start < overlap_end:
                ranges.append((overlap_start - child_start, overlap_end - child_start))
            else:
                ranges.append((0, 0))
            cursor = child_end
        return ranges

    @staticmethod
    def _per_dataset_to_sub_dataset_ranges(
        sub_dataset_lengths: list[int],
        world_size: int,
        rank: int,
    ) -> list[tuple[int, int]]:
        return [ShardedListWeightedSampler._split_range(length, world_size, rank) for length in sub_dataset_lengths]

    @staticmethod
    def _per_dataset_embodiment_to_sub_dataset_ranges(
        dataset: ConcatDataset,
        sub_dataset_lengths: list[int],
        world_size: int,
        rank: int,
        embodiment_scaling_dataset_indices: list[int] | tuple[int, ...] | None,
    ) -> tuple[list[tuple[int, int]], list[list[tuple[int, int, str]] | None]]:
        requested_indices = None if embodiment_scaling_dataset_indices is None else {int(index) for index in embodiment_scaling_dataset_indices}
        ranges = ShardedListWeightedSampler._per_dataset_to_sub_dataset_ranges(
            sub_dataset_lengths,
            world_size,
            rank,
        )
        inner_ranges_by_dataset: list[list[tuple[int, int, str]] | None] = [None] * len(sub_dataset_lengths)

        for dataset_index, child in enumerate(dataset.datasets):
            if requested_indices is not None and dataset_index not in requested_indices:
                continue
            if not isinstance(child, ConcatDataset):
                if requested_indices is not None:
                    raise ValueError(f'embodiment scaling requested for child dataset {dataset_index}, ' f'but it is not a ConcatDataset')
                continue

            embodiment_names = _resolve_child_embodiment_names(child)
            if embodiment_names is None:
                if requested_indices is not None:
                    raise ValueError(f'cannot infer embodiment names for child dataset {dataset_index}')
                continue

            selected_ranges = []
            cursor = 0
            for inner_child, embodiment_name in zip(child.datasets, embodiment_names):
                inner_length = len(inner_child)
                inner_start, inner_end = ShardedListWeightedSampler._split_range(
                    inner_length,
                    world_size,
                    rank,
                )
                if inner_start < inner_end:
                    selected_ranges.append((cursor + inner_start, cursor + inner_end, embodiment_name))
                cursor += inner_length

            positive_embodiments = {name for start, end, name in selected_ranges if end > start}
            if len(positive_embodiments) <= 1:
                if requested_indices is not None:
                    raise ValueError(f'child dataset {dataset_index} does not contain multiple embodiments in its sampled range')
                continue

            child_range_start = min(start for start, _, _ in selected_ranges)
            child_range_end = max(end for _, end, _ in selected_ranges)
            ranges[dataset_index] = (child_range_start, child_range_end)
            inner_ranges_by_dataset[dataset_index] = [
                (start - child_range_start, end - child_range_start, name) for start, end, name in selected_ranges
            ]

        return ranges, inner_ranges_by_dataset

    @staticmethod
    def _coarse_group_name(group_name: str) -> str:
        if group_name.startswith('robot_'):
            return 'robot'
        return group_name

    @staticmethod
    def _resolve_group_names(dataset: ConcatDataset, shard_group_names: list[str] | None) -> list[str]:
        group_names = shard_group_names
        if group_names is None:
            group_names = getattr(dataset, 'group_names', None)
        if group_names is None:
            raise ValueError('shard_mode="group_balanced" requires shard_group_names or dataset.group_names')
        if len(group_names) != len(dataset.datasets):
            raise ValueError('shard group name count should match number of child datasets, ' f'got {len(group_names)} and {len(dataset.datasets)}')
        return [ShardedListWeightedSampler._coarse_group_name(str(name)) for name in group_names]

    @staticmethod
    def _split_children_by_weight(
        child_indices: list[int],
        weights: list[float],
        world_size: int,
        rank: int,
    ) -> list[int]:
        return [
            child_index
            for child_index, _, _, _ in ShardedListWeightedSampler._split_child_weight_overlaps(
                child_indices,
                weights,
                world_size,
                rank,
            )
        ]

    @staticmethod
    def _split_child_weight_overlaps(
        child_indices: list[int],
        weights: list[float],
        world_size: int,
        rank: int,
    ) -> list[tuple[int, float, float, float]]:
        """Return child assignments for one rank.

        Each tuple is ``(child_index, overlap_weight, relative_start, relative_end)``.
        For positive weights, children are assigned by overlap between the
        child's cumulative weight interval and the rank's target interval.
        Relative bounds map that overlap back to a disjoint child data range.
        """
        if not child_indices:
            return []

        total_weight = sum(max(float(weights[i]), 0.0) for i in child_indices)
        if total_weight <= 0:
            start, end = ShardedListWeightedSampler._split_range(len(child_indices), world_size, rank)
            return [(child_index, 0.0, 0.0, 1.0) for child_index in child_indices[start:end]]

        target_start = total_weight * rank / world_size
        target_end = total_weight * (rank + 1) / world_size
        selected: list[tuple[int, float, float, float]] = []
        prefix = 0.0
        for child_index in child_indices:
            child_weight = max(float(weights[child_index]), 0.0)
            if child_weight == 0:
                continue
            child_start = prefix
            child_end = prefix + child_weight
            overlap_start = max(child_start, target_start)
            overlap_end = min(child_end, target_end)
            if overlap_start < overlap_end:
                selected.append(
                    (
                        child_index,
                        overlap_end - overlap_start,
                        (overlap_start - child_start) / child_weight,
                        (overlap_end - child_start) / child_weight,
                    )
                )
            prefix += child_weight

        return selected

    @staticmethod
    def _relative_range_to_child_range(length: int, relative_start: float, relative_end: float) -> tuple[int, int]:
        eps = 1e-9
        start = int(math.floor(length * relative_start + eps))
        end = int(math.floor(length * relative_end + eps))
        if relative_start <= eps:
            start = 0
        if relative_end >= 1.0 - eps:
            end = length
        start = max(0, min(start, length))
        end = max(start, min(end, length))
        return start, end

    @staticmethod
    def _group_balanced_to_sub_dataset_ranges(
        sub_dataset_lengths: list[int],
        sampling_weights: list[float] | None,
        group_names: list[str],
        world_size: int,
        rank: int,
    ) -> tuple[list[tuple[int, int]], list[float]]:
        if sampling_weights is None:
            raise ValueError('sampling_weights is required for shard_mode="group_balanced"')
        if len(sampling_weights) != len(sub_dataset_lengths):
            raise ValueError('sampling_weights length should match number of child datasets')
        if len(group_names) != len(sub_dataset_lengths):
            raise ValueError('group_names length should match number of child datasets')

        ranges = [(0, 0)] * len(sub_dataset_lengths)
        local_weights = [0.0] * len(sub_dataset_lengths)
        ordered_groups: list[str] = []
        children_by_group: dict[str, list[int]] = {}
        for index, group_name in enumerate(group_names):
            if group_name not in children_by_group:
                ordered_groups.append(group_name)
                children_by_group[group_name] = []
            children_by_group[group_name].append(index)

        for group_name in ordered_groups:
            child_indices = children_by_group[group_name]
            if len(child_indices) <= world_size:
                selected = child_indices
                use_child_range_split = True
                selected_weights = {child_index: float(sampling_weights[child_index]) for child_index in selected}
                selected_ranges = {
                    child_index: ShardedListWeightedSampler._split_range(
                        int(sub_dataset_lengths[child_index]),
                        world_size,
                        rank,
                    )
                    for child_index in selected
                }
            else:
                split_assignments = ShardedListWeightedSampler._split_child_weight_overlaps(
                    child_indices,
                    sampling_weights,
                    world_size,
                    rank,
                )
                selected = [child_index for child_index, _, _, _ in split_assignments]
                use_child_range_split = False
                selected_weights = {child_index: overlap_weight for child_index, overlap_weight, _, _ in split_assignments}
                selected_ranges = {
                    child_index: ShardedListWeightedSampler._relative_range_to_child_range(
                        int(sub_dataset_lengths[child_index]),
                        relative_start,
                        relative_end,
                    )
                    for child_index, _, relative_start, relative_end in split_assignments
                }
            selected_set = set(selected)
            original_group_weight = sum(float(sampling_weights[i]) for i in child_indices)
            selected_group_weight = sum(float(selected_weights[i]) for i in selected)
            scale = original_group_weight / selected_group_weight if selected_group_weight > 0 else 0.0

            for child_index in child_indices:
                if child_index not in selected_set:
                    continue
                if use_child_range_split:
                    ranges[child_index] = selected_ranges[child_index]
                else:
                    ranges[child_index] = selected_ranges[child_index]
                local_weights[child_index] = float(selected_weights[child_index]) * scale

        return ranges, local_weights

    def _compute_aligned_total_size(self) -> int:
        if not self.align_process_batches:
            return self.consumed_total_size
        if self.consumed_total_size % self.process_batch_size != 0:
            raise ValueError(
                'consumed_total_size should be divisible by process_batch_size, ' f'got {self.consumed_total_size} and {self.process_batch_size}'
            )
        consumed_batches = self.consumed_total_size // self.process_batch_size
        process_groups = int(math.ceil(consumed_batches / self.process_shard_size))
        return process_groups * self.num_processes * self.process_batch_size

    def __len__(self) -> int:
        return self.aligned_total_size

    def _first_valid_index(self) -> int:
        for offset, length in zip(self.offsets, self.sub_dataset_lengths):
            if length > 0:
                return offset
        raise ValueError(f'shard {self.shard_rank}/{self.shard_world_size} contains no samples')

    def _iter_consumed_indices(self) -> Iterator[int]:
        if self.ratio_mode == 'epoch':
            yield from self._build_epoch_indices().tolist()
        else:
            yield from self._build_batch_indices().tolist()

    def _iter_aligned_indices(self, consumed_indices: Iterator[int]) -> Iterator[int]:
        filler_index = self._first_valid_index()
        shard_process_start = self.shard_rank * self.process_shard_size
        shard_process_end = shard_process_start + self.process_shard_size
        process_groups = self.aligned_total_size // (self.num_processes * self.process_batch_size)
        _debug_sampler(
            'ShardedListWeightedSampler._iter_aligned_indices start shard=%s/%s process_index=%s process_groups=%s '
            'aligned_total_size=%s process_batch_size=%s',
            self.shard_rank,
            self.shard_world_size,
            self.process_index,
            process_groups,
            self.aligned_total_size,
            self.process_batch_size,
        )

        for _ in range(process_groups):
            for process_slot in range(self.num_processes):
                if shard_process_start <= process_slot < shard_process_end:
                    batch = list(itertools.islice(consumed_indices, self.process_batch_size))
                    if len(batch) != self.process_batch_size:
                        raise RuntimeError(
                            'not enough consumed indices to fill aligned process batches: ' f'expected {self.process_batch_size}, got {len(batch)}'
                        )
                    yield from batch
                else:
                    yield from itertools.repeat(filler_index, self.process_batch_size)

        leftover = next(consumed_indices, None)
        if leftover is not None:
            raise RuntimeError('aligned process batches did not consume all local shard indices')
        _debug_sampler(
            'ShardedListWeightedSampler._iter_aligned_indices finish shard=%s/%s process_index=%s',
            self.shard_rank,
            self.shard_world_size,
            self.process_index,
        )

    def __iter__(self) -> Iterator[int]:
        while True:
            _debug_sampler(
                'ShardedListWeightedSampler.__iter__ epoch_start epoch=%s index_mode=%s ratio_mode=%s consumed_total_size=%s '
                'aligned_total_size=%s shard=%s/%s range=[%s,%s)',
                self.epoch,
                self.index_mode,
                self.ratio_mode,
                self.consumed_total_size,
                self.aligned_total_size,
                self.shard_rank,
                self.shard_world_size,
                self.shard_start,
                self.shard_end,
            )
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1

            consumed_indices = self._iter_consumed_indices()
            if self.align_process_batches:
                yield from self._iter_aligned_indices(consumed_indices)
            else:
                yield from consumed_indices

            if not self.infinite:
                break
