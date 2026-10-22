import bisect
import logging
import os
from typing import Any

from .. import utils
from .base_dataset import BaseDataset, _describe_dataset_for_error, _get_data_worker_context

DATASETS = dict()


def get_abs_path(abs_or_rel_path: str, data_dir: str | None = None) -> str:
    """Resolve an absolute path.

    Args:
        abs_or_rel_path: Either an absolute path or a path relative to candidate data directories.
        data_dir: Preferred data directory. Also tries ``utils.get_data_dir()``.

    Returns:
        The resolved absolute path that exists on disk.

    Raises:
        IOError: If the path cannot be resolved to an existing file or directory.
    """
    if os.path.isabs(abs_or_rel_path):
        assert os.path.exists(abs_or_rel_path)
        return abs_or_rel_path
    rel_path = abs_or_rel_path
    data_dirs = [data_dir] if data_dir is not None else []
    data_dirs.append(utils.get_data_dir())
    for data_dir in data_dirs:
        abs_path = os.path.abspath(os.path.join(data_dir, rel_path))
        if os.path.exists(abs_path):
            return abs_path
    raise IOError('{} is not exist'.format(rel_path))


def get_rel_path(abs_path: str, data_dir: str | None = None) -> str:
    """Convert absolute path into a path relative to a known base directory.

    Args:
        abs_path: An absolute path.
        data_dir: Preferred data directory. Also tries ``utils.get_data_dir()``.

    Returns:
        A relative path if ``abs_path`` starts with one of the root directories; otherwise returns ``abs_path`` unchanged.
    """
    data_dirs = []
    if data_dir is not None:
        if os.path.isabs(data_dir):
            data_dirs.extend([data_dir, os.path.relpath(data_dir)])
        else:
            data_dirs.extend([data_dir, os.path.abspath(data_dir)])
    data_dirs.append(utils.get_data_dir())
    for data_dir in data_dirs:
        if abs_path.startswith(data_dir):
            rel_path = abs_path[len(data_dir) :]
            if rel_path[0] == '/':
                rel_path = rel_path[1:]
            return rel_path
    return abs_path


def sort_paths(paths: list[str]) -> list[str]:
    """Order config paths so those with ``data_index`` precede others.

    Ensures downstream merging aligns by ``data_index`` keys before other fields.

    Args:
        paths: A list of configuration file or directory paths.

    Returns:
        Re-ordered list of paths.
    """
    new_paths = []
    idx = 0
    for path in paths:
        config = load_config(path)
        key_names = config['_key_names']
        if 'data_index' in key_names:
            # Ensure datasets with explicit alignment key come first for stable merging
            new_paths.insert(idx, path)
            idx += 1
        else:
            new_paths.append(path)
    return new_paths


def load_config(data_or_config: str | dict) -> dict:
    """Load and normalize a dataset configuration.

    Args:
        data_or_config: A directory path, a path to ``config.json``, or an in-memory config dict.

    Returns:
        A normalized configuration containing ``config_path`` and ``data_path`` fields.
    """
    if isinstance(data_or_config, str):
        if os.path.isdir(data_or_config):
            config_path = os.path.join(data_or_config, 'config.json')
        else:
            config_path = data_or_config
        config = utils.load_file(config_path)
        config['config_path'] = config_path
        if 'data_path' not in config:
            config['data_path'] = os.path.dirname(config_path)
        else:
            config_data_path = config['data_path']
            if isinstance(config_data_path, list):
                config['data_path'] = [
                    get_abs_path(path, os.path.dirname(config_path)) if isinstance(path, str) else path for path in config_data_path
                ]
            else:
                config['data_path'] = get_abs_path(config_data_path)
    elif isinstance(data_or_config, dict):
        config = data_or_config
    else:
        assert False
    return config


def register_dataset(dataset: type[BaseDataset]) -> type[BaseDataset]:
    """Class decorator to register a dataset class by its name.

    Args:
        dataset (type[BaseDataset]): Dataset class to register.

    Returns:
        type[BaseDataset]: The same dataset class, unchanged.

    Raises:
        AssertionError: If a dataset with the same name was already registered.
    """
    dataset_name = dataset.__name__
    assert dataset_name not in DATASETS, dataset_name
    DATASETS[dataset_name] = dataset
    return dataset


def load_dataset(data_or_config: Any) -> BaseDataset:
    """Instantiate a dataset from configuration.

    Args:
        data_or_config: A config dict, directory path, or file path. If a list, delegates to ``ConcatDataset``.

    Returns:
        An initialized dataset instance.
    """
    if isinstance(data_or_config, list):
        return ConcatDataset.load(data_or_config)
    config = load_config(data_or_config)
    class_name = config['_class_name']
    return DATASETS[class_name].load(config)


@register_dataset
class Dataset(BaseDataset):
    """Dataset that merges multiple aligned datasets by key."""

    def __init__(self, datasets: list[BaseDataset]) -> None:
        """Initialize a merged dataset.

        Args:
            datasets: A non-empty list of datasets. The first dataset defines length and indexing base.
        """
        super(Dataset, self).__init__()
        assert len(datasets) > 0
        self.datasets = datasets

    @classmethod
    def load(cls, data_or_config: Any):
        """Load a merged dataset from config or delegate to specific dataset
        types.

        If ``_class_name`` in the config is not ``Dataset``, delegates to ``load_dataset``.
        """
        if isinstance(data_or_config, list):
            return ConcatDataset.load(data_or_config)
        config = load_config(data_or_config)
        class_name = config['_class_name']
        if class_name != 'Dataset':
            return load_dataset(config)
        data_path = config['data_path']
        config_paths = config['config_paths']
        config_paths = [get_abs_path(config_path, data_path) for config_path in config_paths]
        config_paths = sort_paths(config_paths)
        datasets = [load_dataset(config_path) for config_path in config_paths]
        return cls(datasets)

    def save(self, save_path: str, store_rel_path: bool = True) -> None:
        """Save a composite dataset's config that references child configs.

        Args:
            save_path: Directory or file path to write ``config.json``.
            store_rel_path: Store child config paths as relative to ``save_path`` for portability.
        """
        if os.path.isdir(save_path):
            save_config_path = os.path.join(save_path, 'config.json')
        else:
            save_config_path = save_path
            save_path = os.path.dirname(save_config_path)
        config_paths = [dataset.config_path for dataset in self.datasets]
        config_paths = sort_paths(config_paths)
        if store_rel_path:
            config_paths = [get_rel_path(config_path, save_path) for config_path in config_paths]
        config = {
            '_class_name': 'Dataset',
            'config_paths': config_paths,
        }
        utils.save_file(save_config_path, config)

    def open(self) -> None:
        for dataset in self.datasets:
            dataset.open()

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()
        super(Dataset, self).close()

    def reset(self) -> None:
        for dataset in self.datasets:
            dataset.reset()
        super(Dataset, self).reset()

    def filter(self, mode: str, **kwargs: Any) -> None:
        """Apply an in-place filter on the first child dataset.

        Notes:
            Only the first child dataset is filtered, serving as the primary index.
        """
        self.datasets[0].filter(mode, **kwargs)

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        try:
            self.open()
            if isinstance(index, (list, tuple)):
                # Vectorized fetch: preserve order from the provided index list
                data_dict = [self._get_data(idx) for idx in index]
            else:
                data_dict = self._get_data(index)
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            return data_dict
        except Exception as error:
            raise RuntimeError(
                'Failed to fetch merged dataset sample: '
                f'dataset={_describe_dataset_for_error(self)}, index={index!r}, {_get_data_worker_context()}'
            ) from error

    def _get_data(self, index: int) -> dict:
        # Start with primary dataset; it defines canonical indexing and may contain 'data_index'
        try:
            data_dict = self.datasets[0][index]
        except Exception as error:
            raise RuntimeError(
                'Failed to fetch primary child dataset sample: '
                f'child_index=0, child={_describe_dataset_for_error(self.datasets[0])}, '
                f'parent_index={index!r}, {_get_data_worker_context()}'
            ) from error
        if 'data_index' in data_dict:
            data_index = data_dict['data_index']
        else:
            data_index = index
        # Merge aligned records from the remaining datasets using the resolved data_index
        for i in range(1, len(self.datasets)):
            try:
                data_dict.update(self.datasets[i][data_index])
            except Exception as error:
                raise RuntimeError(
                    'Failed to fetch aligned child dataset sample: '
                    f'child_index={i}, child={_describe_dataset_for_error(self.datasets[i])}, '
                    f'parent_index={index!r}, resolved_data_index={data_index!r}, {_get_data_worker_context()}'
                ) from error
        return data_dict


class ConcatDataset(BaseDataset):
    """Dataset that concatenates multiple datasets end-to-end."""

    def __init__(self, datasets: list[BaseDataset]) -> None:
        """Initialize a concatenated dataset.

        Args:
            datasets: A non-empty list of datasets to concatenate.
        """
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0
        self.datasets = datasets
        self._cumulative_sizes: list[int] | None = None

    @classmethod
    def load(cls, data_or_config_list: list[Any]) -> 'ConcatDataset':
        """Load and concatenate datasets specified by a list of
        configs/paths."""
        datasets = [load_dataset(d) for d in data_or_config_list]
        return cls(datasets)

    def open(self) -> None:
        for dataset in self.datasets:
            dataset.open()

    def close(self) -> None:
        for dataset in self.datasets:
            dataset.close()
        self._invalidate_index_cache()
        super(ConcatDataset, self).close()

    def reset(self) -> None:
        for dataset in self.datasets:
            dataset.reset()
        self._invalidate_index_cache()
        super(ConcatDataset, self).reset()

    def _invalidate_index_cache(self) -> None:
        self._cumulative_sizes = None

    def _get_cumulative_sizes(self) -> list[int]:
        if self._cumulative_sizes is None:
            cumulative_sizes = []
            total_size = 0
            for dataset in self.datasets:
                total_size += len(dataset)
                cumulative_sizes.append(total_size)
            self._cumulative_sizes = cumulative_sizes
        return self._cumulative_sizes

    def filter(self, mode: str, **kwargs: Any) -> None:
        """Filter each child dataset or perform an overall collection-level op.

        If ``mode`` starts with ``overall`` then an aggregation is performed across
        all first-child data lists; otherwise forward filter to each dataset with
        ``dataset_index`` as an additional kwarg.
        """
        if mode.startswith('overall'):
            all_data_list = []
            for i, dataset in enumerate(self.datasets):
                dataset.datasets[0].open()
                all_data_list.append(dataset.datasets[0].data_list)
            if mode == 'overall_func':
                func = kwargs.pop('func')
                if isinstance(func, str):
                    func = utils.import_function(func)
                all_data_list = func(all_data_list, **kwargs)
            else:
                assert False
            for i, dataset in enumerate(self.datasets):
                logging.info(f'filter dataset {i} from {dataset.datasets[0].data_size} to {len(all_data_list[i])}')
                dataset.datasets[0].data_list = all_data_list[i]
                dataset.datasets[0].data_size = len(all_data_list[i])
        else:
            for i, dataset in enumerate(self.datasets):
                dataset.filter(mode=mode, dataset_index=i, **kwargs)
        self._invalidate_index_cache()

    def __len__(self) -> int:
        return self._get_cumulative_sizes()[-1]

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        if isinstance(index, (list, tuple)):
            data_dict = [self._get_data(idx) for idx in index]
        else:
            data_dict = self._get_data(index)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def _get_data(self, index: int) -> Any:
        cumulative_sizes = self._get_cumulative_sizes()
        dataset_index = bisect.bisect_right(cumulative_sizes, index)
        if dataset_index == len(cumulative_sizes):
            raise IndexError
        previous_size = 0 if dataset_index == 0 else cumulative_sizes[dataset_index - 1]
        local_index = index - previous_size
        child = self.datasets[dataset_index]
        try:
            return child[local_index]
        except Exception as error:
            raise RuntimeError(
                'Failed to fetch concat child dataset sample: '
                f'global_index={index!r}, child_index={dataset_index}, local_index={local_index!r}, '
                f'child_start={previous_size}, child_end={cumulative_sizes[dataset_index]}, '
                f'child={_describe_dataset_for_error(child)}, {_get_data_worker_context()}'
            ) from error


@register_dataset
class WeightedConcatDataset(ConcatDataset):
    """Concat dataset with explicit per-subdataset sampling weights.

    This class keeps the same sample access behavior as ``ConcatDataset`` while
    carrying ``sampling_weights`` metadata that can be consumed by custom
    samplers.
    """

    def __init__(
        self,
        datasets: list[BaseDataset],
        sampling_weights: list[float],
        group_names: list[str] | None = None,
        embodiment_names: list[str] | None = None,
        robot_types: list[str] | None = None,
    ) -> None:
        super(WeightedConcatDataset, self).__init__(datasets)
        assert len(sampling_weights) == len(datasets), 'sampling_weights length should match datasets length'
        self.sampling_weights = [float(w) for w in sampling_weights]
        assert all(w >= 0 for w in self.sampling_weights), 'sampling_weights should be non-negative'
        assert sum(self.sampling_weights) > 0, 'sum of sampling_weights should be greater than 0'
        if group_names is not None:
            assert len(group_names) == len(datasets), 'group_names length should match datasets length'
            self.group_names = [str(name) for name in group_names]
        if embodiment_names is not None:
            assert len(embodiment_names) == len(datasets), 'embodiment_names length should match datasets length'
            self.embodiment_names = [str(name) for name in embodiment_names]
        if robot_types is not None:
            assert len(robot_types) == len(datasets), 'robot_types length should match datasets length'
            self.robot_types = [str(name) for name in robot_types]

    @classmethod
    def load(cls, data_or_config: Any) -> 'WeightedConcatDataset':
        """Load a weighted concat dataset from config.

        Supported config keys:
            - ``datasets`` or ``data_or_config_list``: list of child dataset
              configs/paths.
            - ``sampling_weights`` or ``weights``: list of sampling weights.
            - ``group_names``/``embodiment_names``/``robot_types``: optional
              per-child metadata used by weighted samplers.
        """
        config = load_config(data_or_config)
        data_or_config_list = config.get('datasets', config.get('data_or_config_list', None))
        sampling_weights = config.get('sampling_weights', config.get('weights', None))
        group_names = config.get('group_names', None)
        embodiment_names = config.get('embodiment_names', None)
        robot_types = config.get('robot_types', None)
        assert data_or_config_list is not None, 'datasets or data_or_config_list is required'
        assert sampling_weights is not None, 'sampling_weights or weights is required'

        data_dir = config.get('data_path', None)
        resolved_data_or_config_list = []
        for d in data_or_config_list:
            if isinstance(d, str) and data_dir is not None:
                d = get_abs_path(d, data_dir)
            resolved_data_or_config_list.append(d)

        datasets = [load_dataset(d) for d in resolved_data_or_config_list]
        return cls(
            datasets,
            sampling_weights=sampling_weights,
            group_names=group_names,
            embodiment_names=embodiment_names,
            robot_types=robot_types,
        )

    def save(self, save_path: str, store_rel_path: bool = True) -> None:
        """Save weighted concat dataset config to ``config.json``."""
        if os.path.isdir(save_path):
            save_config_path = os.path.join(save_path, 'config.json')
        else:
            save_config_path = save_path
            save_path = os.path.dirname(save_config_path)

        dataset_paths = [dataset.config_path for dataset in self.datasets]
        if store_rel_path:
            dataset_paths = [get_rel_path(config_path, save_path) if config_path is not None else None for config_path in dataset_paths]

        config = {
            '_class_name': 'WeightedConcatDataset',
            'datasets': dataset_paths,
            'sampling_weights': self.sampling_weights,
        }
        if hasattr(self, 'group_names'):
            config['group_names'] = self.group_names
        if hasattr(self, 'embodiment_names'):
            config['embodiment_names'] = self.embodiment_names
        if hasattr(self, 'robot_types'):
            config['robot_types'] = self.robot_types
        utils.save_file(save_config_path, config)
