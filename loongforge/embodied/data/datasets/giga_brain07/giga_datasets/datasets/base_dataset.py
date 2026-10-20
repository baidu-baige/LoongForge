import os
from typing import Any

import torch
from tqdm import tqdm

from ..utils import Timer


def _get_data_worker_context() -> str:
    env_names = (
        'RANK',
        'LOCAL_RANK',
        'WORLD_SIZE',
        'LOCAL_WORLD_SIZE',
        'NODE_RANK',
        'GROUP_RANK',
        'ROLE_RANK',
        'SLURM_PROCID',
        'SLURM_NODEID',
        'SLURM_LOCALID',
    )
    env_context = ', '.join(f'{name}={os.environ[name]}' for name in env_names if name in os.environ)
    hostname = os.environ.get('HOSTNAME') or getattr(os.uname(), 'nodename', 'unknown')

    worker_info = torch.utils.data.get_worker_info()
    worker_context = 'worker=main' if worker_info is None else f'worker_id={worker_info.id}, num_workers={worker_info.num_workers}'

    parts = [f'host={hostname}', f'pid={os.getpid()}', worker_context]
    if env_context:
        parts.append(env_context)
    return ', '.join(parts)


def _describe_dataset_for_error(dataset: Any) -> str:
    parts = [dataset.__class__.__name__]
    for attr_name in ('data_path', 'config_path', 'root'):
        try:
            attr_value = getattr(dataset, attr_name, None)
        except Exception as error:
            attr_value = f'<failed to read {attr_name}: {error}>'
        if attr_value is not None:
            parts.append(f'{attr_name}={attr_value!s}')
    return ' '.join(parts)


class BaseProcessor:
    """Base interface for streaming data post-processing.

    Subclasses are expected to override ``process`` to define per-sample logic
    and may override ``__call__`` to support batch-level transforms. This class
    provides a minimal protocol for processors used by :class:`BaseDataset`.
    """

    def __call__(self, data_dict: Any) -> Any:
        """Identity by default; override to transform a sample or batch.

        Args:
            data_dict (Any): The sample or a collated mini-batch to be processed.

        Returns:
            Any: The processed data. Defaults to input as-is.
        """
        return data_dict

    def process(self, *args: Any, **kwargs: Any) -> None:
        """Process a single sample; called inside ``BaseDataset.process``.

        Args:
            *args (Any): Positional fields for a single sample. When a dataset
                returns a tuple, it will be unpacked and passed here.
            **kwargs (Any): Keyword fields for a single sample if returned as a dict.

        Notes:
            Implement this in subclasses to define per-sample logic.
        """
        pass

    def close(self) -> None:
        """Finalize resources if needed after processing completes.

        Subclasses can override to close files, flush writers, or release handles opened during processing.
        """
        pass


class BaseDataset(torch.utils.data.Dataset):
    """Dataset base class with lazy open/close semantics and transforms.

    This class defines a common interface for datasets that support delayed I/O
    initialization via :meth:`open`, cooperative resource cleanup via
    :meth:`close`, and optional per-sample transforms via ``transform``.
    """

    def __init__(self, config_path: str | None = None, data_path: str | None = None, transform: Any = None) -> None:
        """Initialize a base dataset instance.

        Args:
            config_path (str | None): Optional path to a dataset configuration file.
            data_path (str | None): Optional path to the dataset directory. If ``config_path``
                is provided, this is inferred as ``dirname(config_path)``.
            transform (Any): Optional transform or processor applied on returned samples.
        """
        self._config_path = config_path
        self._data_path = data_path
        self.transform = transform

    @property
    def config_path(self) -> str | None:
        if self._config_path is not None:
            return self._config_path
        elif self._data_path is not None:
            return os.path.join(self._data_path, 'config.json')
        else:
            return None

    @property
    def data_path(self) -> str | None:
        if self._data_path is not None:
            return self._data_path
        elif self._config_path is not None:
            return os.path.dirname(self._config_path)
        else:
            return None

    @classmethod
    def load(cls, data_or_config: Any) -> 'BaseDataset':
        """Load dataset from a config path or a dict.

        Args:
            data_or_config (Any): A configuration dict or a path to the configuration file.

        Returns:
            BaseDataset: An instantiated dataset.
        """
        raise NotImplementedError

    def save(self, save_path: str, **kwargs: Any) -> None:
        """Save dataset configuration and optional metadata.

        Args:
            save_path (str): Path to save configuration file. Can be a directory or a file.
            **kwargs (Any): Additional metadata to store.
        """
        raise NotImplementedError

    def open(self) -> None:
        """Open underlying storage lazily before reading data.

        Notes:
            Subclasses should implement actual I/O initialization here.
        """
        pass

    def close(self) -> None:
        self._config_path = None
        self._data_path = None
        self.transform = None

    def reset(self) -> None:
        """Reset dataset state by closing all resources."""
        self.close()

    def filter(self, *args: Any, **kwargs: Any) -> None:
        """In-place filter to reduce or re-order underlying data.

        Notes:
            Implement dataset-specific filtering strategies in subclasses.
        """
        raise NotImplementedError

    def set_transform(self, transform: Any) -> None:
        """Set a transform or processor to be applied at ``__getitem__``.

        Args:
            transform (Any): A callable that accepts and returns a sample or batch.
        """
        self.transform = transform

    def process(self, processor: BaseProcessor, num_workers: int = 0) -> None:
        """Iterate through dataset and call processor on each sample.

        Args:
            processor (BaseProcessor): A processor that defines ``process`` to handle each sample.
            num_workers (int): Number of worker processes for the internal ``DataLoader``.
        """
        ori_transform = self.transform
        self.transform = processor
        dataloader = torch.utils.data.DataLoader(
            self,
            collate_fn=lambda x: x,
            batch_size=1,
            num_workers=num_workers,
        )
        with Timer('Process Cost'):
            # Iterate over all samples; ``processor.process`` handles per-sample logic
            for batch in tqdm(dataloader, total=len(dataloader)):
                for data in batch:
                    if isinstance(data, tuple):
                        processor.process(*data)
                    else:
                        processor.process(data)
            processor.close()
        self.transform = ori_transform

    def __len__(self) -> int:
        """Return the number of samples contained in this dataset."""
        raise NotImplementedError

    def __del__(self) -> None:
        """Ensure resources are released on garbage collection."""
        self.close()

    def __getitem__(self, index: int | list[int] | tuple[int, ...]) -> Any:
        """Retrieve sample(s) at ``index`` and apply optional transform.

        Args:
            index (int | list[int] | tuple[int, ...]): An integer index or a list/tuple of indices for multi-fetch.

        Returns:
            Any: The sample dict or a list of sample dicts after optional transform.
        """
        try:
            self.open()
            if isinstance(index, (list, tuple)):
                data_dict = [self._get_data(idx) for idx in index]
            else:
                data_dict = self._get_data(index)
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            return data_dict
        except Exception as error:
            raise RuntimeError(
                'Failed to fetch dataset sample: ' f'dataset={_describe_dataset_for_error(self)}, index={index!r}, {_get_data_worker_context()}'
            ) from error

    def _get_data(self, index: int) -> Any:
        """Fetch raw data for a single index.

        Args:
            index (int): The sample index to fetch.

        Returns:
            Any: Raw per-sample data without transforms applied.

        Notes:
            To be implemented by subclasses.
        """
        raise NotImplementedError
