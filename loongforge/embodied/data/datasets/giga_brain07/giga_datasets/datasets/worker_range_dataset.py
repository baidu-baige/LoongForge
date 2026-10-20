import bisect
import os
from typing import Any

import torch

from .base_dataset import BaseDataset, _get_data_worker_context
from .dataset import load_dataset, register_dataset


@register_dataset
class WorkerRangeDataset(BaseDataset):
    """Remap DataLoader worker reads into worker-local contiguous ranges.

    The wrapped dataset keeps its original length so samplers can keep emitting
    global indices. Inside a DataLoader worker, incoming indices are remapped
    into that worker's contiguous sub-range of ``[range_start, range_end)``.
    This bounds the portion of the dataset each worker reads while leaving the
    sampler in the main process unchanged.
    """

    def __init__(
        self,
        dataset: Any | None = None,
        data_or_config: Any | None = None,
        range_start: int | None = None,
        range_end: int | None = None,
        shard_world_size: int | None = None,
        shard_rank: int | None = None,
        num_processes: int | None = None,
        process_index: int | None = None,
        process_shard_size: int | None = None,
        worker_scope: str = 'process',
        mode: str = 'whole',
        sub_dataset_ranges: list[tuple[int, int]] | None = None,
        config_path: str | None = None,
        data_path: str | None = None,
        transform: Any = None,
    ) -> None:
        super().__init__(config_path=config_path, data_path=data_path, transform=transform)
        if dataset is None:
            if data_or_config is None:
                raise ValueError('Either dataset or data_or_config is required')
            dataset = load_dataset(data_or_config)

        self.dataset = dataset
        self.range_start = None if range_start is None else int(range_start)
        self.range_end = None if range_end is None else int(range_end)
        self.shard_world_size = None if shard_world_size is None else int(shard_world_size)
        self.shard_rank = None if shard_rank is None else int(shard_rank)
        self.num_processes = None if num_processes is None else int(num_processes)
        self.process_index = None if process_index is None else int(process_index)
        self.process_shard_size = None if process_shard_size is None else int(process_shard_size)
        if worker_scope not in ('process', 'node'):
            raise ValueError("worker_scope should be either 'process' or 'node'")
        self.worker_scope = worker_scope
        if mode in ('dataset', 'whole_dataset', 'global'):
            mode = 'whole'
        if mode not in ('whole', 'per_child'):
            raise ValueError("mode should be either 'whole' or 'per_child'")
        self.mode = mode
        self.sub_dataset_ranges = None if sub_dataset_ranges is None else [(int(start), int(end)) for start, end in sub_dataset_ranges]

    @classmethod
    def load(cls, data_or_config: Any) -> 'WorkerRangeDataset':
        from .dataset import load_config

        config = dict(load_config(data_or_config))
        for key in list(config):
            if key.startswith('_'):
                config.pop(key)
        return cls(**config)

    def __getattr__(self, name: str) -> Any:
        dataset = self.__dict__.get('dataset', None)
        if dataset is not None:
            return getattr(dataset, name)
        raise AttributeError(name)

    def __len__(self) -> int:
        return len(self.dataset)

    def open(self) -> None:
        if hasattr(self.dataset, 'open'):
            self.dataset.open()

    def close(self) -> None:
        dataset = self.__dict__.get('dataset', None)
        if dataset is not None and hasattr(dataset, 'close'):
            dataset.close()
        super().close()

    def reset(self) -> None:
        if hasattr(self.dataset, 'reset'):
            self.dataset.reset()
        else:
            self.close()

    def filter(self, *args: Any, **kwargs: Any) -> None:
        self.dataset.filter(*args, **kwargs)

    def save(self, save_path: str, **kwargs: Any) -> None:
        raise NotImplementedError('WorkerRangeDataset.save is not implemented')

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
        resolved = WorkerRangeDataset._resolve_positive_int(value, env_names, default)
        if resolved < 0:
            raise ValueError(f'value should be non-negative, got {resolved}')
        return resolved

    @staticmethod
    def _split_range(total_size: int, world_size: int, rank: int) -> tuple[int, int]:
        start = (total_size * rank) // world_size
        end = (total_size * (rank + 1)) // world_size
        return start, end

    def _infer_machine_range(self, total_size: int) -> tuple[int, int]:
        num_processes = self._resolve_positive_int(self.num_processes, ('WORLD_SIZE',), 1)
        process_index = self._resolve_non_negative_int(self.process_index, ('RANK',), 0)
        if process_index >= num_processes:
            raise ValueError(f'process_index should be < num_processes, got {process_index} >= {num_processes}')

        process_shard_size = self.process_shard_size
        if process_shard_size is None:
            env_process_shard_size = os.environ.get('NPROC_PER_NODE') or os.environ.get('LOCAL_WORLD_SIZE')
            process_shard_size = int(env_process_shard_size) if env_process_shard_size else 1
        process_shard_size = int(process_shard_size)
        if process_shard_size <= 0:
            raise ValueError('process_shard_size should be greater than 0')
        if num_processes % process_shard_size != 0:
            raise ValueError(f'num_processes should be divisible by process_shard_size, got {num_processes} and {process_shard_size}')

        shard_world_size = self.shard_world_size
        if shard_world_size is None:
            shard_world_size = num_processes // process_shard_size
        shard_world_size = int(shard_world_size)
        if shard_world_size <= 0:
            raise ValueError('shard_world_size should be greater than 0')
        if num_processes % shard_world_size != 0:
            raise ValueError(f'num_processes should be divisible by shard_world_size, got {num_processes} and {shard_world_size}')

        shard_rank = self.shard_rank
        if shard_rank is None:
            shard_rank = process_index // process_shard_size
        shard_rank = int(shard_rank)
        if shard_rank < 0 or shard_rank >= shard_world_size:
            raise ValueError(f'shard_rank should be in [0, {shard_world_size}), got {shard_rank}')
        return self._split_range(total_size, shard_world_size, shard_rank)

    def _base_range(self) -> tuple[int, int]:
        total_size = len(self.dataset)
        if total_size <= 0:
            raise ValueError('WorkerRangeDataset requires a non-empty dataset')

        if self.range_start is None and self.range_end is None:
            if self.shard_world_size is None and self.shard_rank is None:
                start, end = 0, total_size
            else:
                start, end = self._infer_machine_range(total_size)
        elif self.range_start is not None and self.range_end is not None:
            start, end = self.range_start, self.range_end
        else:
            raise ValueError('range_start and range_end should be provided together')

        if start < 0 or end <= start or end > total_size:
            raise ValueError(f'invalid WorkerRangeDataset range [{start}, {end}) for dataset length {total_size}')
        return start, end

    def _worker_rank_world_size(self) -> tuple[int, int, int, int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)
        if num_workers <= 0:
            raise ValueError(f'num_workers should be positive, got {num_workers}')

        if self.worker_scope == 'node':
            local_world_size = self._resolve_positive_int(
                self.process_shard_size,
                ('LOCAL_WORLD_SIZE', 'NPROC_PER_NODE'),
                1,
            )
            local_rank_env = os.environ.get('LOCAL_RANK') or os.environ.get('SLURM_LOCALID')
            if local_rank_env is not None:
                local_rank = int(local_rank_env)
            else:
                process_index = self._resolve_non_negative_int(self.process_index, ('RANK',), 0)
                local_rank = process_index % local_world_size
            if local_rank < 0 or local_rank >= local_world_size:
                raise ValueError(f'local_rank should be in [0, {local_world_size}), got {local_rank}')
            worker_rank = local_rank * num_workers + worker_id
            worker_world_size = local_world_size * num_workers
        else:
            worker_rank = worker_id
            worker_world_size = num_workers

        return worker_rank, worker_world_size, worker_id, num_workers

    def _range_for_worker(self, base_start: int, base_end: int) -> tuple[int, int, int, int]:
        base_size = base_end - base_start
        if base_size <= 0:
            raise ValueError(f'worker base range should be non-empty, got [{base_start}, {base_end})')

        worker_rank, worker_world_size, worker_id, num_workers = self._worker_rank_world_size()
        shard_workers = min(worker_world_size, base_size)
        worker_rank = worker_rank % shard_workers
        worker_start = base_start + (base_size * worker_rank) // shard_workers
        worker_end = base_start + (base_size * (worker_rank + 1)) // shard_workers
        if worker_end <= worker_start:
            raise ValueError(
                'WorkerRangeDataset worker range is empty: '
                f'base=[{base_start}, {base_end}), worker_id={worker_id}, num_workers={num_workers}, '
                f'{_get_data_worker_context()}'
            )
        return worker_start, worker_end, worker_id, num_workers

    def _worker_range(self) -> tuple[int, int, int, int]:
        base_start, base_end = self._base_range()
        worker_start, worker_end, worker_id, _ = self._range_for_worker(base_start, base_end)
        return base_start, worker_start, worker_end, worker_id

    def _top_level_children_and_offsets(self) -> tuple[list[Any], list[int]]:
        children = getattr(self.dataset, 'datasets', None)
        if not children:
            raise TypeError('WorkerRangeDataset mode="per_child" requires a concat-style dataset with child datasets')
        children = list(children)

        cumulative_sizes = []
        total_size = 0
        for child in children:
            total_size += len(child)
            cumulative_sizes.append(total_size)

        if len(cumulative_sizes) != len(children):
            raise ValueError('concat cumulative size count should match child dataset count, ' f'got {len(cumulative_sizes)} and {len(children)}')
        return children, cumulative_sizes

    def _per_child_base_ranges(self, children: list[Any], cumulative_sizes: list[int]) -> list[tuple[int, int]]:
        if self.sub_dataset_ranges is not None:
            if len(self.sub_dataset_ranges) != len(children):
                raise ValueError(
                    'sub_dataset_ranges length should match number of child datasets, ' f'got {len(self.sub_dataset_ranges)} and {len(children)}'
                )
            ranges = self.sub_dataset_ranges
        else:
            base_start, base_end = self._base_range()
            ranges = []
            previous_size = 0
            for cumulative_size in cumulative_sizes:
                overlap_start = max(base_start, previous_size)
                overlap_end = min(base_end, cumulative_size)
                if overlap_start < overlap_end:
                    ranges.append((overlap_start - previous_size, overlap_end - previous_size))
                else:
                    ranges.append((0, 0))
                previous_size = cumulative_size

        validated_ranges = []
        for child_index, ((start, end), child) in enumerate(zip(ranges, children)):
            child_length = len(child)
            if start < 0 or end < start or end > child_length:
                raise ValueError(f'invalid sub_dataset_ranges[{child_index}]={(start, end)}, ' f'expected 0 <= start <= end <= {child_length}')
            validated_ranges.append((start, end))
        return validated_ranges

    def _map_per_child_index(self, index: int) -> int:
        children, cumulative_sizes = self._top_level_children_and_offsets()
        if not cumulative_sizes or cumulative_sizes[-1] <= 0:
            raise ValueError('WorkerRangeDataset requires a non-empty dataset')

        child_index = bisect.bisect_right(cumulative_sizes, index)
        if child_index == len(cumulative_sizes):
            raise IndexError(f'index {index} is out of range for dataset length {cumulative_sizes[-1]}')
        child_global_offset = 0 if child_index == 0 else cumulative_sizes[child_index - 1]
        local_index = index - child_global_offset

        child_ranges = self._per_child_base_ranges(children, cumulative_sizes)
        child_base_start, child_base_end = child_ranges[child_index]
        if child_base_end <= child_base_start:
            raise ValueError(
                'WorkerRangeDataset mode="per_child" selected a child with an empty base range: '
                f'child_index={child_index}, range=[{child_base_start}, {child_base_end}), index={index}'
            )

        worker_start, worker_end, _, _ = self._range_for_worker(child_base_start, child_base_end)
        worker_size = worker_end - worker_start
        mapped_local_index = worker_start + ((local_index - child_base_start) % worker_size)
        return child_global_offset + mapped_local_index

    def map_index(self, index: int) -> int:
        if self.mode == 'per_child':
            return self._map_per_child_index(int(index))

        base_start, worker_start, worker_end, _ = self._worker_range()
        worker_size = worker_end - worker_start
        return worker_start + ((int(index) - base_start) % worker_size)

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        if isinstance(index, (list, tuple)):
            return self.__getitems__(list(index))

        mapped_index = self.map_index(int(index))
        data = self.dataset[mapped_index]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitems__(self, indices: list[int]) -> list[Any]:
        mapped_indices = [self.map_index(int(index)) for index in indices]
        getitems = getattr(self.dataset, '__getitems__', None)
        if getitems is not None:
            data = getitems(mapped_indices)
        else:
            try:
                data = self.dataset[mapped_indices]
            except Exception:
                data = [self.dataset[index] for index in mapped_indices]
        if self.transform is not None:
            data = self.transform(data)
        return data
