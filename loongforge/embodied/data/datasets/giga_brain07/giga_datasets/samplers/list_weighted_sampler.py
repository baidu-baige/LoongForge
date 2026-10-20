import json
import logging
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler

from ..datasets import ConcatDataset


def _debug_sampler_enabled() -> bool:
    value = os.environ.get('GIGA_DEBUG_SAMPLER')
    if value is None:
        return False
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}


def _debug_sampler(message: str, *args) -> None:
    if not _debug_sampler_enabled():
        return
    if args:
        message = message % args
    worker_info = torch.utils.data.get_worker_info()
    worker_id = None if worker_info is None else worker_info.id
    worker_count = None if worker_info is None else worker_info.num_workers
    prefix = (
        f'[GIGA_DEBUG_SAMPLER host={socket.gethostname()} pid={os.getpid()} '
        f'rank={os.environ.get("RANK", "?")} local_rank={os.environ.get("LOCAL_RANK", "?")} '
        f'worker={worker_id}/{worker_count}] '
    )
    logging.info(prefix + message)


def _is_embodiment_scaling_enabled(
    exponent: float | None,
    balance_robot_types: bool = False,
) -> bool:
    return exponent is not None and (bool(balance_robot_types) or not math.isclose(float(exponent), 1.0))


def _normalize_embodiment_scaling_scope(scope: str | None) -> str:
    scope = 'child' if scope is None else str(scope)
    if scope not in {'child', 'coarse_group'}:
        raise ValueError('embodiment_scaling_scope should be either "child" or "coarse_group", ' f'got {scope!r}')
    return scope


def _coarse_group_name(group_name: str) -> str:
    group_name = str(group_name)
    if group_name.startswith('robot_'):
        return 'robot'
    return group_name


def _resolve_named_sequence(
    names: list[Any] | tuple[Any, ...] | None,
    expected_len: int,
    label: str,
) -> list[str] | None:
    if names is None:
        return None
    if len(names) != expected_len:
        raise ValueError(f'{label} length should match number of child datasets')
    return [str(name) for name in names]


def _safe_get_dataset_attr(dataset: Any, attr: str) -> Any:
    try:
        return getattr(dataset, attr)
    except Exception:
        return None


def _infer_lerobot_robot_type(dataset: Any) -> str | None:
    data_path = _safe_get_dataset_attr(dataset, 'data_path')
    if not isinstance(data_path, str):
        return None

    info_path = Path(data_path) / 'meta' / 'info.json'
    try:
        with info_path.open('r') as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    robot_type = info.get('robot_type')
    if robot_type is None:
        return None
    return str(robot_type)


def _infer_dataset_embodiment_name(dataset: Any) -> str | None:
    for attr in ('robot_type', 'embodiment_name', 'embodiment', 'embodiment_id'):
        value = _safe_get_dataset_attr(dataset, attr)
        if value is not None:
            return str(value)

    robot_type = _infer_lerobot_robot_type(dataset)
    if robot_type is not None:
        return robot_type

    child_datasets = _safe_get_dataset_attr(dataset, 'datasets')
    if not child_datasets:
        return None

    child_names = [_infer_dataset_embodiment_name(child) for child in child_datasets]
    if any(name is None for name in child_names):
        return None
    unique_names = set(child_names)
    if len(unique_names) == 1:
        return child_names[0]
    return None


def _resolve_child_embodiment_names(
    dataset: ConcatDataset,
) -> list[str] | None:
    expected_len = len(dataset.datasets)

    dataset_embodiment_names = _safe_get_dataset_attr(dataset, 'embodiment_names')
    resolved = _resolve_named_sequence(dataset_embodiment_names, expected_len, 'dataset.embodiment_names')
    if resolved is not None:
        return resolved

    dataset_robot_types = _safe_get_dataset_attr(dataset, 'robot_types')
    resolved = _resolve_named_sequence(dataset_robot_types, expected_len, 'dataset.robot_types')
    if resolved is not None:
        return resolved

    inferred = [_infer_dataset_embodiment_name(child) for child in dataset.datasets]
    if all(name is not None for name in inferred):
        return [str(name) for name in inferred]

    group_names = _safe_get_dataset_attr(dataset, 'group_names')
    return _resolve_named_sequence(group_names, expected_len, 'dataset.group_names')


def _resolve_child_robot_types(dataset: ConcatDataset) -> list[str] | None:
    expected_len = len(dataset.datasets)
    dataset_robot_types = _safe_get_dataset_attr(dataset, 'robot_types')
    resolved = _resolve_named_sequence(dataset_robot_types, expected_len, 'dataset.robot_types')
    if resolved is not None:
        return resolved

    inferred = [_infer_lerobot_robot_type(child) for child in dataset.datasets]
    if all(name is not None for name in inferred):
        return [str(name) for name in inferred]
    return None


def _resolve_top_level_group_names(dataset: ConcatDataset) -> list[str]:
    group_names = _safe_get_dataset_attr(dataset, 'group_names')
    resolved = _resolve_named_sequence(group_names, len(dataset.datasets), 'dataset.group_names')
    if resolved is None:
        raise ValueError('embodiment_scaling_scope="coarse_group" requires dataset.group_names')
    return resolved


def _resolve_embodiment_scaling_group_names(
    group_names: list[str] | tuple[str, ...] | None,
    coarse_group_names: list[str],
) -> set[str]:
    if group_names is None:
        return set(coarse_group_names)
    return {str(name) for name in group_names}


def _child_inner_embodiment_ranges(
    child: Any,
    child_start: int,
    child_end: int,
) -> list[tuple[int, int, str]] | None:
    if child_start >= child_end:
        return []

    if not isinstance(child, ConcatDataset):
        embodiment_name = _infer_dataset_embodiment_name(child)
        if embodiment_name is None:
            return None
        return [(0, child_end - child_start, embodiment_name)]

    embodiment_names = _resolve_child_embodiment_names(child)
    if embodiment_names is None:
        return None

    cursor = 0
    inner_ranges: list[tuple[int, int, str]] = []
    for inner_child, embodiment_name in zip(child.datasets, embodiment_names):
        inner_start = cursor
        inner_end = cursor + len(inner_child)
        overlap_start = max(child_start, inner_start)
        overlap_end = min(child_end, inner_end)
        if overlap_start < overlap_end:
            inner_ranges.append((overlap_start - child_start, overlap_end - child_start, embodiment_name))
        cursor = inner_end
    return inner_ranges


def _child_inner_robot_type_ranges(
    child: Any,
    child_start: int,
    child_end: int,
) -> list[tuple[int, int, str]] | None:
    if child_start >= child_end:
        return []
    if not isinstance(child, ConcatDataset):
        robot_type = _infer_lerobot_robot_type(child)
        if robot_type is None:
            return None
        return [(0, child_end - child_start, robot_type)]

    robot_types = _resolve_child_robot_types(child)
    if robot_types is None:
        return None

    cursor = 0
    inner_ranges: list[tuple[int, int, str]] = []
    for inner_child, robot_type in zip(child.datasets, robot_types):
        inner_start = cursor
        inner_end = cursor + len(inner_child)
        overlap_start = max(child_start, inner_start)
        overlap_end = min(child_end, inner_end)
        if overlap_start < overlap_end:
            inner_ranges.append((overlap_start - child_start, overlap_end - child_start, robot_type))
        cursor = inner_end
    return inner_ranges


def _build_coarse_group_embodiment_scaling_plan(
    dataset: ConcatDataset,
    sampling_weights: list[float],
    sub_dataset_ranges: list[tuple[int, int]],
    exponent: float,
    embodiment_scaling_group_names: list[str] | tuple[str, ...] | None,
    embodiment_scaling_dataset_indices: set[int] | None = None,
    balance_robot_types: bool = False,
) -> dict[str, Any]:
    group_names = _resolve_top_level_group_names(dataset)
    coarse_group_names = [_coarse_group_name(name) for name in group_names]
    selected_group_names = _resolve_embodiment_scaling_group_names(
        embodiment_scaling_group_names,
        coarse_group_names,
    )

    group_total_weights: dict[str, float] = {}
    lengths_by_group_and_embodiment: dict[str, dict[str, int]] = {}
    lengths_by_group_embodiment_and_robot_type: dict[str, dict[str, dict[str, int]]] = {}

    for dataset_index, (child, weight, child_range, coarse_group) in enumerate(
        zip(dataset.datasets, sampling_weights, sub_dataset_ranges, coarse_group_names)
    ):
        if coarse_group not in selected_group_names:
            continue
        if embodiment_scaling_dataset_indices is not None and dataset_index not in embodiment_scaling_dataset_indices:
            continue

        start, end = child_range
        if end <= start:
            continue

        inner_ranges = _child_inner_embodiment_ranges(child, start, end)
        if inner_ranges is None:
            raise ValueError('cannot infer embodiment names for ' f'child dataset {dataset_index} in coarse group {coarse_group!r}')
        robot_type_ranges = None
        if balance_robot_types:
            robot_type_ranges = _child_inner_robot_type_ranges(child, start, end)
            if robot_type_ranges is None or len(robot_type_ranges) != len(inner_ranges):
                raise ValueError(
                    'cannot infer robot types for balanced sampling in ' f'child dataset {dataset_index} in coarse group {coarse_group!r}'
                )

        group_total_weights[coarse_group] = group_total_weights.get(coarse_group, 0.0) + float(weight)
        lengths_by_embodiment = lengths_by_group_and_embodiment.setdefault(coarse_group, {})
        for range_index, (range_start, range_end, embodiment_name) in enumerate(inner_ranges):
            range_length = range_end - range_start
            if range_length <= 0:
                continue
            lengths_by_embodiment[embodiment_name] = lengths_by_embodiment.get(embodiment_name, 0) + int(range_length)
            if balance_robot_types:
                robot_start, robot_end, robot_type = robot_type_ranges[range_index]
                if (robot_start, robot_end) != (range_start, range_end):
                    raise ValueError('embodiment and robot-type ranges should be aligned')
                lengths_by_robot_type = lengths_by_group_embodiment_and_robot_type.setdefault(coarse_group, {}).setdefault(embodiment_name, {})
                lengths_by_robot_type[robot_type] = lengths_by_robot_type.get(robot_type, 0) + int(range_length)

    density_by_group_and_embodiment: dict[str, dict[Any, float]] = {}
    for group_name, lengths_by_embodiment in lengths_by_group_and_embodiment.items():
        positive_lengths = {name: length for name, length in lengths_by_embodiment.items() if length > 0}
        if not positive_lengths or (len(positive_lengths) == 1 and not balance_robot_types):
            continue

        scaled_totals = {name: length ** float(exponent) for name, length in positive_lengths.items()}
        scaled_sum = sum(scaled_totals.values())
        group_weight = group_total_weights.get(group_name, 0.0)
        if scaled_sum <= 0 or group_weight <= 0:
            continue

        if balance_robot_types:
            densities = {}
            robot_type_lengths_by_embodiment = lengths_by_group_embodiment_and_robot_type[group_name]
            for name in positive_lengths:
                positive_robot_types = {robot_type: length for robot_type, length in robot_type_lengths_by_embodiment[name].items() if length > 0}
                robot_type_count = len(positive_robot_types)
                for robot_type, length in positive_robot_types.items():
                    densities[(name, robot_type)] = group_weight * (scaled_totals[name] / scaled_sum) / robot_type_count / length
            density_by_group_and_embodiment[group_name] = densities
        else:
            density_by_group_and_embodiment[group_name] = {
                name: group_weight * (scaled_totals[name] / scaled_sum) / positive_lengths[name] for name in positive_lengths
            }

    return {
        'coarse_group_names': coarse_group_names,
        'selected_group_names': selected_group_names,
        'group_total_weights': group_total_weights,
        'density_by_group_and_embodiment': density_by_group_and_embodiment,
        'embodiment_scaling_dataset_indices': embodiment_scaling_dataset_indices,
        'balance_robot_types': bool(balance_robot_types),
    }


def _apply_coarse_group_embodiment_scaling_plan(
    dataset: ConcatDataset,
    sampling_weights: list[float],
    sub_dataset_ranges: list[tuple[int, int]],
    plan: dict[str, Any],
) -> tuple[list[float], list[list[tuple[int, int, str]] | None], list[list[float] | None]]:
    new_sampling_weights = [float(weight) for weight in sampling_weights]
    inner_ranges_by_dataset: list[list[tuple[int, int, str]] | None] = [None] * len(dataset.datasets)
    inner_probabilities_by_dataset: list[list[float] | None] = [None] * len(dataset.datasets)

    coarse_group_names: list[str] = plan['coarse_group_names']
    density_by_group_and_embodiment: dict[str, dict[Any, float]] = plan['density_by_group_and_embodiment']
    group_total_weights: dict[str, float] = plan['group_total_weights']
    embodiment_scaling_dataset_indices: set[int] | None = plan.get('embodiment_scaling_dataset_indices')
    balance_robot_types = bool(plan.get('balance_robot_types', False))

    pending_by_group: dict[str, list[tuple[int, list[tuple[int, int, str]], list[float]]]] = {}
    selected_target_totals: dict[str, float] = {}

    for dataset_index, (child, child_range, coarse_group) in enumerate(zip(dataset.datasets, sub_dataset_ranges, coarse_group_names)):
        densities = density_by_group_and_embodiment.get(coarse_group)
        if not densities:
            continue
        if embodiment_scaling_dataset_indices is not None and dataset_index not in embodiment_scaling_dataset_indices:
            continue

        start, end = child_range
        if end <= start:
            new_sampling_weights[dataset_index] = 0.0
            continue

        inner_ranges = _child_inner_embodiment_ranges(child, start, end)
        if inner_ranges is None:
            raise ValueError('cannot infer embodiment names for ' f'child dataset {dataset_index} in coarse group {coarse_group!r}')
        robot_type_ranges = None
        if balance_robot_types:
            robot_type_ranges = _child_inner_robot_type_ranges(child, start, end)
            if robot_type_ranges is None or len(robot_type_ranges) != len(inner_ranges):
                raise ValueError(
                    'cannot infer robot types for balanced sampling in ' f'child dataset {dataset_index} in coarse group {coarse_group!r}'
                )

        target_weights = []
        for range_index, (range_start, range_end, embodiment_name) in enumerate(inner_ranges):
            range_length = range_end - range_start
            density_key: Any = embodiment_name
            if balance_robot_types:
                robot_start, robot_end, robot_type = robot_type_ranges[range_index]
                if (robot_start, robot_end) != (range_start, range_end):
                    raise ValueError('embodiment and robot-type ranges should be aligned')
                density_key = (embodiment_name, robot_type)
            target_weights.append(float(densities.get(density_key, 0.0)) * range_length)

        child_target_total = sum(target_weights)
        if child_target_total <= 0:
            new_sampling_weights[dataset_index] = 0.0
            continue

        pending_by_group.setdefault(coarse_group, []).append((dataset_index, inner_ranges, target_weights))
        selected_target_totals[coarse_group] = selected_target_totals.get(coarse_group, 0.0) + child_target_total

    for coarse_group, pending_items in pending_by_group.items():
        selected_total = selected_target_totals.get(coarse_group, 0.0)
        group_total = group_total_weights.get(coarse_group, 0.0)
        if selected_total <= 0 or group_total <= 0:
            continue
        scale = group_total / selected_total
        for dataset_index, inner_ranges, target_weights in pending_items:
            scaled_target_weights = [weight * scale for weight in target_weights]
            child_target_total = sum(scaled_target_weights)
            new_sampling_weights[dataset_index] = child_target_total

            positive_embodiments = {
                name
                for (range_start, range_end, name), target_weight in zip(inner_ranges, scaled_target_weights)
                if range_end > range_start and target_weight > 0
            }
            positive_range_count = sum(
                range_end > range_start and target_weight > 0
                for (range_start, range_end, _), target_weight in zip(inner_ranges, scaled_target_weights)
            )
            should_override_inner_probabilities = positive_range_count > 1 if balance_robot_types else len(positive_embodiments) > 1
            if child_target_total > 0 and should_override_inner_probabilities:
                inner_ranges_by_dataset[dataset_index] = inner_ranges
                inner_probabilities_by_dataset[dataset_index] = [weight / child_target_total for weight in scaled_target_weights]

    return new_sampling_weights, inner_ranges_by_dataset, inner_probabilities_by_dataset


def _embodiment_scaled_range_probabilities(
    range_lengths: list[int],
    embodiment_names: list[str],
    exponent: float,
) -> list[float]:
    exponent = float(exponent)
    if exponent < 0:
        raise ValueError('embodiment_scaling_exponent should be non-negative')
    if len(range_lengths) != len(embodiment_names):
        raise ValueError('range_lengths and embodiment_names should have the same length')

    ordered_embodiments: list[str] = []
    lengths_by_embodiment: dict[str, int] = {}
    for name, length in zip(embodiment_names, range_lengths):
        if length <= 0:
            continue
        if name not in lengths_by_embodiment:
            ordered_embodiments.append(name)
            lengths_by_embodiment[name] = 0
        lengths_by_embodiment[name] += int(length)

    scaled_totals = {name: (lengths_by_embodiment[name] ** exponent if lengths_by_embodiment[name] > 0 else 0.0) for name in ordered_embodiments}
    scaled_sum = sum(scaled_totals.values())
    if scaled_sum <= 0:
        return [0.0] * len(range_lengths)

    probabilities = []
    for name, length in zip(embodiment_names, range_lengths):
        embodiment_length = lengths_by_embodiment.get(name, 0)
        if length <= 0 or embodiment_length <= 0:
            probabilities.append(0.0)
        else:
            probabilities.append((scaled_totals[name] / scaled_sum) * (length / embodiment_length))
    return probabilities


class ListWeightedSampler(Sampler):
    """Sampler for ``ConcatDataset`` using explicit per-subdataset weights.

    Sampling weights represent sampling probabilities among child datasets.
    The sampled total size follows the concatenated dataset size (optionally
    padded to ``batch_size``). Sampling ratios can be enforced at the epoch
    level or within each ``batch_size`` block.

    ``embodiment_scaling_exponent`` keeps the top-level ``sampling_weights``
    unchanged with ``embodiment_scaling_scope='child'`` and only reshapes
    sampling inside child ``ConcatDataset`` objects that contain multiple robot
    embodiments. With ``embodiment_scaling_scope='coarse_group'``, child
    weights inside selected coarse groups are recomputed from global
    per-embodiment totals while preserving each coarse group's total weight.
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
        embodiment_scaling_inner_ranges: list[list[tuple[int, int, str]] | None] | None = None,
        embodiment_scaling_inner_probabilities: list[list[float] | None] | None = None,
        sub_dataset_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        if not isinstance(dataset, ConcatDataset):
            raise TypeError('dataset should be a ConcatDataset')

        if sampling_weights is None:
            sampling_weights = getattr(dataset, 'sampling_weights', None)
        if sampling_weights is None:
            raise ValueError('sampling_weights is required, or dataset should provide sampling_weights')

        if len(sampling_weights) != len(dataset.datasets):
            raise ValueError('sampling_weights length should match number of child datasets')

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.ratio_mode = ratio_mode
        self.index_mode = index_mode
        self.epoch_size = epoch_size
        self.oversample_mode = str(oversample_mode)
        self.epoch = 0

        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError('batch_size should be greater than 0')
        if self.ratio_mode not in ('epoch', 'batch'):
            raise ValueError("ratio_mode should be either 'epoch' or 'batch'")
        if self.ratio_mode == 'batch' and self.batch_size is None:
            raise ValueError("batch_size is required when ratio_mode is 'batch'")
        if self.index_mode != 'array':
            raise ValueError("index_mode should be 'array'")
        if self.epoch_size is not None and self.epoch_size <= 0:
            raise ValueError('epoch_size should be greater than 0')
        if self.oversample_mode not in {'replacement', 'cycle'}:
            raise ValueError("oversample_mode should be either 'replacement' or 'cycle'")

        self.sampling_weights = [float(w) for w in sampling_weights]
        self.embodiment_scaling_exponent = embodiment_scaling_exponent
        self.embodiment_scaling_scope = _normalize_embodiment_scaling_scope(embodiment_scaling_scope)
        self.embodiment_scaling_group_names = (
            None if embodiment_scaling_group_names is None else [str(name) for name in embodiment_scaling_group_names]
        )
        self.embodiment_scaling_dataset_indices = (
            None if embodiment_scaling_dataset_indices is None else {int(index) for index in embodiment_scaling_dataset_indices}
        )
        self.embodiment_scaling_balance_robot_types = bool(embodiment_scaling_balance_robot_types)
        if self.embodiment_scaling_balance_robot_types and self.embodiment_scaling_scope != 'coarse_group':
            raise ValueError('embodiment_scaling_balance_robot_types is only supported with ' 'embodiment_scaling_scope="coarse_group"')
        self._explicit_inner_embodiment_ranges = embodiment_scaling_inner_ranges
        if self._explicit_inner_embodiment_ranges is not None and len(self._explicit_inner_embodiment_ranges) != len(dataset.datasets):
            raise ValueError('embodiment_scaling_inner_ranges length should match number of child datasets')
        self._explicit_inner_embodiment_probabilities = embodiment_scaling_inner_probabilities
        if self._explicit_inner_embodiment_probabilities is not None and len(self._explicit_inner_embodiment_probabilities) != len(dataset.datasets):
            raise ValueError('embodiment_scaling_inner_probabilities length should match number of child datasets')
        if _is_embodiment_scaling_enabled(
            self.embodiment_scaling_exponent,
            self.embodiment_scaling_balance_robot_types,
        ):
            if float(self.embodiment_scaling_exponent) < 0:
                raise ValueError('embodiment_scaling_exponent should be non-negative')
            if self.embodiment_scaling_scope == 'child' and self.embodiment_scaling_group_names is not None:
                raise ValueError('embodiment_scaling_group_names is only supported with ' 'embodiment_scaling_scope="coarse_group"')
            if self.embodiment_scaling_dataset_indices is not None:
                invalid_indices = [index for index in self.embodiment_scaling_dataset_indices if index < 0 or index >= len(dataset.datasets)]
                if invalid_indices:
                    raise ValueError(f'invalid embodiment_scaling_dataset_indices: {invalid_indices}')
        if any(w < 0 for w in self.sampling_weights):
            raise ValueError('sampling_weights should be non-negative')

        self.full_sub_dataset_lengths = [len(d) for d in self.dataset.datasets]
        if sub_dataset_ranges is None:
            self.sub_dataset_ranges = [(0, length) for length in self.full_sub_dataset_lengths]
            allow_empty_ranges = False
        else:
            if len(sub_dataset_ranges) != len(self.dataset.datasets):
                raise ValueError('sub_dataset_ranges length should match number of child datasets')
            self.sub_dataset_ranges = []
            for i, (start, end) in enumerate(sub_dataset_ranges):
                start = int(start)
                end = int(end)
                full_length = self.full_sub_dataset_lengths[i]
                if start < 0 or end < start or end > full_length:
                    raise ValueError(f'invalid sub_dataset_ranges[{i}]={(start, end)}, ' f'expected 0 <= start <= end <= {full_length}')
                self.sub_dataset_ranges.append((start, end))
            allow_empty_ranges = True

        self.sub_dataset_lengths = [end - start for start, end in self.sub_dataset_ranges]
        empty_dataset_indices = [i for i, length in enumerate(self.sub_dataset_lengths) if length == 0]
        if empty_dataset_indices and not allow_empty_ranges:
            raise ValueError(f'child datasets at indices {empty_dataset_indices} have zero length')
        if allow_empty_ranges:
            self.sampling_weights = [0.0 if length == 0 else weight for weight, length in zip(self.sampling_weights, self.sub_dataset_lengths)]
        self._inner_embodiment_ranges = self._build_inner_embodiment_ranges()
        self._inner_embodiment_probabilities = self._build_inner_embodiment_probabilities()
        has_explicit_coarse_group_plan = (
            self._explicit_inner_embodiment_ranges is not None and self._explicit_inner_embodiment_probabilities is not None
        )
        if (
            _is_embodiment_scaling_enabled(
                self.embodiment_scaling_exponent,
                self.embodiment_scaling_balance_robot_types,
            )
            and self.embodiment_scaling_scope == 'coarse_group'
            and not has_explicit_coarse_group_plan
        ):
            plan = _build_coarse_group_embodiment_scaling_plan(
                self.dataset,
                self.sampling_weights,
                self.sub_dataset_ranges,
                float(self.embodiment_scaling_exponent),
                self.embodiment_scaling_group_names,
                self.embodiment_scaling_dataset_indices,
                self.embodiment_scaling_balance_robot_types,
            )
            (
                self.sampling_weights,
                self._inner_embodiment_ranges,
                self._inner_embodiment_probabilities,
            ) = _apply_coarse_group_embodiment_scaling_plan(
                self.dataset,
                self.sampling_weights,
                self.sub_dataset_ranges,
                plan,
            )

        weight_sum = sum(self.sampling_weights)
        if weight_sum <= 0:
            raise ValueError('sum of sampling_weights should be greater than 0')

        probabilities = [w / weight_sum for w in self.sampling_weights]

        original_total_size = int(self.epoch_size) if self.epoch_size is not None else sum(self.sub_dataset_lengths)
        if self.batch_size is not None:
            original_total_size = int(math.ceil(original_total_size / self.batch_size)) * self.batch_size

        if self.ratio_mode == 'epoch':
            self.num_samples_per_sub_dataset = self._allocate_counts(probabilities, original_total_size)
            self.num_samples_per_batch = None
        else:
            num_batches = original_total_size // self.batch_size
            self.num_samples_per_batch = self._allocate_counts(probabilities, self.batch_size)
            self.num_samples_per_sub_dataset = [count * num_batches for count in self.num_samples_per_batch]

        self.total_size = sum(self.num_samples_per_sub_dataset)

        cumulative_sizes = [0] * len(self.dataset.datasets)
        for i in range(1, len(self.dataset.datasets)):
            cumulative_sizes[i] = cumulative_sizes[i - 1] + self.full_sub_dataset_lengths[i - 1]
        self.offsets = [offset + start for offset, (start, _) in zip(cumulative_sizes, self.sub_dataset_ranges)]

    def _allocate_counts(self, probabilities: list[float], total_size: int) -> list[int]:
        # Allocate integer sample counts while preserving the desired total size.
        float_counts = [p * total_size for p in probabilities]
        counts = [int(math.floor(v)) for v in float_counts]
        remain = total_size - sum(counts)

        if remain > 0:
            frac_order = sorted(range(len(float_counts)), key=lambda i: float_counts[i] - counts[i], reverse=True)
            for i in frac_order[:remain]:
                counts[i] += 1
        return counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.total_size

    def _build_inner_embodiment_ranges(self) -> list[list[tuple[int, int, str]] | None]:
        ranges_by_dataset: list[list[tuple[int, int, str]] | None] = [None] * len(self.dataset.datasets)
        if not _is_embodiment_scaling_enabled(
            self.embodiment_scaling_exponent,
            self.embodiment_scaling_balance_robot_types,
        ):
            return ranges_by_dataset
        if self._explicit_inner_embodiment_ranges is not None:
            return self._explicit_inner_embodiment_ranges
        if self.embodiment_scaling_scope == 'coarse_group':
            return ranges_by_dataset

        for dataset_index, child in enumerate(self.dataset.datasets):
            if self.embodiment_scaling_dataset_indices is not None and dataset_index not in self.embodiment_scaling_dataset_indices:
                continue
            if not isinstance(child, ConcatDataset):
                if self.embodiment_scaling_dataset_indices is not None:
                    raise ValueError(f'embodiment scaling requested for child dataset {dataset_index}, ' f'but it is not a ConcatDataset')
                continue

            embodiment_names = _resolve_child_embodiment_names(child)
            if embodiment_names is None:
                if self.embodiment_scaling_dataset_indices is not None:
                    raise ValueError(f'cannot infer embodiment names for child dataset {dataset_index}')
                continue

            child_start, child_end = self.sub_dataset_ranges[dataset_index]
            cursor = 0
            inner_ranges: list[tuple[int, int, str]] = []
            for inner_child, embodiment_name in zip(child.datasets, embodiment_names):
                inner_start = cursor
                inner_end = cursor + len(inner_child)
                overlap_start = max(child_start, inner_start)
                overlap_end = min(child_end, inner_end)
                if overlap_start < overlap_end:
                    inner_ranges.append((overlap_start - child_start, overlap_end - child_start, embodiment_name))
                cursor = inner_end

            positive_embodiments = {name for start, end, name in inner_ranges if end > start}
            if len(positive_embodiments) <= 1:
                if self.embodiment_scaling_dataset_indices is not None:
                    raise ValueError(f'child dataset {dataset_index} does not contain multiple embodiments in its sampled range')
                continue
            ranges_by_dataset[dataset_index] = inner_ranges

        return ranges_by_dataset

    def _build_inner_embodiment_probabilities(self) -> list[list[float] | None]:
        probabilities_by_dataset: list[list[float] | None] = [None] * len(self.dataset.datasets)
        if not _is_embodiment_scaling_enabled(
            self.embodiment_scaling_exponent,
            self.embodiment_scaling_balance_robot_types,
        ):
            return probabilities_by_dataset
        if self._explicit_inner_embodiment_probabilities is None:
            return probabilities_by_dataset
        for dataset_index, probabilities in enumerate(self._explicit_inner_embodiment_probabilities):
            inner_ranges = self._inner_embodiment_ranges[dataset_index]
            if probabilities is None:
                continue
            if inner_ranges is None:
                raise ValueError(
                    'embodiment_scaling_inner_probabilities requires matching ' f'embodiment_scaling_inner_ranges for child dataset {dataset_index}'
                )
            if len(probabilities) != len(inner_ranges):
                raise ValueError(
                    'embodiment_scaling_inner_probabilities entries should match ' f'embodiment_scaling_inner_ranges[{dataset_index}] length'
                )
            probability_sum = sum(float(probability) for probability in probabilities)
            if probability_sum <= 0:
                raise ValueError(f'embodiment_scaling_inner_probabilities[{dataset_index}] ' 'should have a positive sum')
            probabilities_by_dataset[dataset_index] = [float(probability) / probability_sum for probability in probabilities]
        return probabilities_by_dataset

    def _sample_uniform_local_indices(self, data_size: int, num_samples: int) -> np.ndarray:
        if num_samples == 0:
            return np.array([], dtype=np.int64)
        if data_size == 0:
            raise ValueError('cannot sample from an empty sub-dataset range')

        if self.oversample_mode == 'cycle' and num_samples > data_size:
            indices = []
            remaining = num_samples
            while remaining > 0:
                indices_i = np.arange(data_size)
                if self.shuffle:
                    indices_i = np.random.permutation(indices_i)
                num_data = min(data_size, remaining)
                indices.append(indices_i[:num_data])
                remaining -= num_data
            return np.concatenate(indices)

        if self.shuffle:
            replace = num_samples > data_size
            return np.random.choice(data_size, num_samples, replace=replace)

        if num_samples > data_size:
            base_indices = np.arange(data_size)
            num_repeats = int(np.ceil(num_samples / data_size))
            return np.tile(base_indices, num_repeats)[:num_samples]

        return np.arange(num_samples)

    def _sample_local_indices(self, dataset_index: int, num_samples: int) -> np.ndarray:
        data_size = self.sub_dataset_lengths[dataset_index]
        inner_ranges = self._inner_embodiment_ranges[dataset_index]
        if inner_ranges is None:
            return self._sample_uniform_local_indices(data_size, num_samples)

        range_lengths = [end - start for start, end, _ in inner_ranges]
        embodiment_names = [name for _, _, name in inner_ranges]
        probabilities = self._inner_embodiment_probabilities[dataset_index]
        if probabilities is None:
            probabilities = _embodiment_scaled_range_probabilities(
                range_lengths,
                embodiment_names,
                float(self.embodiment_scaling_exponent),
            )
        num_samples_per_range = self._allocate_counts(probabilities, num_samples)

        all_indices = []
        for (start, end, _), range_num_samples in zip(inner_ranges, num_samples_per_range):
            if range_num_samples == 0:
                continue
            local_indices = self._sample_uniform_local_indices(end - start, range_num_samples)
            all_indices.append(local_indices + start)

        if not all_indices:
            return np.array([], dtype=np.int64)

        indices = np.concatenate(all_indices)
        if self.shuffle and len(indices) > 0:
            np.random.shuffle(indices)
        return indices

    def _build_epoch_indices(self) -> np.ndarray:
        tic = time.time()
        _debug_sampler(
            'ListWeightedSampler._build_epoch_indices start total_size=%s sub_dataset_lengths=%s num_samples_per_sub_dataset=%s',
            self.total_size,
            self.sub_dataset_lengths,
            self.num_samples_per_sub_dataset,
        )
        all_indices = []
        for i in range(len(self.dataset.datasets)):
            local_indices = self._sample_local_indices(i, self.num_samples_per_sub_dataset[i])
            if len(local_indices) == 0:
                continue
            all_indices.append(local_indices + self.offsets[i])

        if not all_indices:
            indices = np.array([], dtype=np.int64)
        else:
            indices = np.concatenate(all_indices)

        if self.shuffle and len(indices) > 0:
            np.random.shuffle(indices)
        _debug_sampler(
            'ListWeightedSampler._build_epoch_indices finish elapsed=%.3fs num_indices=%s',
            time.time() - tic,
            len(indices),
        )
        return indices

    def _build_batch_indices(self) -> np.ndarray:
        num_batches = self.total_size // self.batch_size
        local_index_pools = [self._sample_local_indices(i, self.num_samples_per_sub_dataset[i]) for i in range(len(self.dataset.datasets))]
        pool_offsets = [0] * len(self.dataset.datasets)
        batches = []

        for _ in range(num_batches):
            batch_parts = []
            for i in range(len(self.dataset.datasets)):
                num_samples = self.num_samples_per_batch[i]
                if num_samples == 0:
                    continue
                start = pool_offsets[i]
                end = start + num_samples
                batch_parts.append(local_index_pools[i][start:end] + self.offsets[i])
                pool_offsets[i] = end

            batch_indices = np.concatenate(batch_parts)
            if self.shuffle and len(batch_indices) > 0:
                batch_indices = np.random.permutation(batch_indices)
            batches.append(batch_indices)

        if self.shuffle and len(batches) > 0:
            batch_order = np.random.permutation(len(batches))
            batches = [batches[i] for i in batch_order]

        if not batches:
            return np.array([], dtype=np.int64)
        return np.concatenate(batches)

    def __iter__(self) -> Iterator[int]:
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1

            if self.ratio_mode == 'epoch':
                indices = self._build_epoch_indices()
                yield from indices.tolist()
            else:
                indices = self._build_batch_indices()
                yield from indices.tolist()

            if not self.infinite:
                break
