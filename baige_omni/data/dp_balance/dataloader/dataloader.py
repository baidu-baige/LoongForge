# Copyright 2026 The BaigeOmni Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from PyTorch under the BSD 3-Clause License.
# Copyright (c) 2016-present, Facebook, Inc and respective contributors.

r"""Definition of the DataLoader and associated iterators that subclass _BaseDataLoaderIter.

To support these two classes, in `./_utils` we define many utility methods and
functions to be run in multiprocessing. E.g., the data loading worker loop is
in `./_utils/worker.py`.
"""

import functools
import itertools
import logging
import multiprocessing as python_multiprocessing
import os
import queue
import threading
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from torch.utils.data import _utils
from torch.utils.data.datapipes.datapipe import (
    IterDataPipe,
    MapDataPipe,
    _IterDataPipeSerializationWrapper,
    _MapDataPipeSerializationWrapper,
)
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.sampler import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

__all__ = [
    "DataLoader",
    "get_worker_info",
    "default_collate",
    "default_convert",
]


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_worker_init_fn_t = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of `collate_fn`,
# but there is currently no way to have that
# type parameter set to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
_collate_fn_t = Callable[[List[_T]], Any]

_resort_fn_t = Callable[[int], Any]


# These functions used to be defined in this file. However, it was moved to
# _utils/collate.py. Although it is rather hard to access this from user land
# (one has to explicitly directly `import torch.utils.data.dataloader`), there
# probably is user code out there using it. This aliasing maintains BC in this
# aspect.
default_collate: _collate_fn_t = _utils.collate.default_collate
default_convert = _utils.collate.default_convert

get_worker_info = _utils.worker.get_worker_info

logger = logging.getLogger(__name__)


class _DatasetKind:
    """
    Dataset type enumeration class for distinguishing different types of datasets

    Attributes:
        Map: Map-style dataset, supports random access
        Iterable: Iterable-style dataset, supports sequential access
    """
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        """
        Create corresponding data fetcher based on dataset type

        Args:
            kind: Dataset type, using _DatasetKind enumeration values
            dataset: Dataset object
            auto_collation: Whether to automatically batch data
            collate_fn: Data batching function
            drop_last: Whether to drop the last incomplete batch

        Returns:
            Corresponding data fetcher instance
        """
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )
        else:
            return _utils.fetch._IterableDatasetFetcher(
                dataset, auto_collation, collate_fn, drop_last
            )


class _InfiniteConstantSampler(Sampler):
    r"""Analogous to ``itertools.repeat(None, None)``.

    Used as sampler for :class:`~torch.utils.data.IterableDataset`.
    """

    def __iter__(self):
        """
        Implement iterator interface, indefinitely yields None values

        This method mimics the behavior similar to itertools.repeat(None, None),
        used as a sampler for :class:`~torch.utils.data.IterableDataset`.

        Yields:
            None: Indefinitely yielded null values
        """
        while True:
            yield None


def _get_distributed_settings():
    """
    Get distributed training related settings

    Returns:
        tuple: A tuple containing two elements:
            - int: Current distributed training world size, returns 1 if distributed training is not enabled
            - int: Current process rank, returns 0 if distributed training is not enabled
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    """
    Worker initialization function for distributed data sharding

    This function is used in distributed training environments to set up data sharding
    for each worker process, ensuring data is evenly distributed between distributed
    processes and worker processes.

    Args:
        worker_init_fn: Original worker initialization function, can be None
        world_size: World size for distributed training (total number of processes)
        rank_id: Current process rank (0 to world_size-1)
        worker_id: Local ID of worker process (0 to num_workers-1)

    Note:
        - Data sharding strategy: first shard across distributed processes, then across worker processes
        - Uses default SHARDING_PRIORITIES for backward compatibility (BC)
    """
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    assert info is not None
    total_workers = info.num_workers
    datapipe = info.dataset
    assert isinstance(datapipe, (IterDataPipe, MapDataPipe))
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    # For BC, use default SHARDING_PRIORITIES
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, total_workers, global_worker_id
    )
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


def _share_dist_seed(generator, pg):
    _shared_seed = torch.empty((), dtype=torch.int64).random_(generator=generator)
    if isinstance(pg, dist.ProcessGroup):
        dist.broadcast(_shared_seed, src=0, group=pg)
    return _shared_seed.item()


class DataLoader(Generic[_T_co]):
    r"""
    Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (Callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into device/CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (Callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)
        multiprocessing_context (str or multiprocessing.context.BaseContext, optional): If
            ``None``, the default `multiprocessing context`_ of your operating system will
            be used. (default: ``None``)
        generator (torch.Generator, optional): If not ``None``, this RNG will be used
            by RandomSampler to generate random indexes and multiprocessing to generate
            ``base_seed`` for workers. (default: ``None``)
        prefetch_factor (int, optional, keyword-only arg): Number of batches loaded
            in advance by each worker. ``2`` means there will be a total of
            2 * num_workers batches prefetched across all workers. (default value depends
            on the set value for num_workers. If value of num_workers=0 default is ``None``.
            Otherwise, if value of ``num_workers > 0`` default is ``2``).
        persistent_workers (bool, optional): If ``True``, the data loader will not shut down
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. (default: ``False``)
        pin_memory_device (str, optional): the device to :attr:`pin_memory` to if ``pin_memory`` is
            ``True``.


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.

    .. warning:: See :ref:`reproducibility`, and :ref:`dataloader-workers-random-seed`, and
                 :ref:`data-loading-randomness` notes for random seed related questions.

    .. _multiprocessing context:
        https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    """

    dataset: Dataset[_T_co]
    batch_size: Optional[int]
    num_workers: int
    pin_memory: bool
    drop_last: bool
    timeout: float
    sampler: Union[Sampler, Iterable]
    pin_memory_device: str
    prefetch_factor: Optional[int]
    _iterator: Optional["_BaseDataLoaderIter"]
    __initialized = False

    def __init__(
        self,
        dataset: Dataset[_T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        resort_fn: Optional[_resort_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ):
        """Initialize a DataLoader instance

        DataLoader is used to load data from datasets, supporting batch processing,
        multiprocess data loading, memory pinning, and other features.

        Args:
            dataset: Dataset to load, supports map-style and iterable-style datasets
            batch_size: Size of each batch, None disables automatic batching
            shuffle: Whether to shuffle data order (only effective for map-style datasets)
            sampler: Custom sampler for defining data sampling strategy
            batch_sampler: Custom batch sampler, mutually exclusive with batch_size and other parameters
            num_workers: Number of worker processes for data loading, 0 means loading in main process
            collate_fn: Function to merge a list of samples into a mini-batch
            resort_fn: Function for resorting/reordering data
            pin_memory: Whether to copy data to CUDA pinned memory
            drop_last: Whether to drop the last incomplete batch
            timeout: Timeout for data loading operations
            worker_init_fn: Worker process initialization function
            multiprocessing_context: Multiprocessing context
            generator: Random number generator
            prefetch_factor: Number of batches to prefetch per worker process
            persistent_workers: Whether to keep worker processes alive
            pin_memory_device: Memory pinning device name

        Raises:
            ValueError: Raised when parameter values are invalid or parameters conflict with each other

        Note:
            - For IterableDataset, custom samplers and batch_samplers are not supported
            - sampler and shuffle parameters are mutually exclusive
            - batch_sampler is mutually exclusive with batch_size, shuffle, sampler, and drop_last
            - persistent_workers requires num_workers > 0
            - prefetch_factor only takes effect when num_workers > 0
        """
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError(
                "num_workers option should be non-negative; "
                "use num_workers=0 to disable multiprocessing."
            )

        if timeout < 0:
            raise ValueError("timeout option should be non-negative")

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError(
                "prefetch_factor option could only be specified in multiprocessing."
                "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."
            )
        elif num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor option should be non-negative")

        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers option needs num_workers > 0")

        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.pin_memory_device = pin_memory_device
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   _DataPipeSerializationWrapper container makes it easier to serialize without redefining pickler
        if isinstance(self.dataset, IterDataPipe):
            self.dataset = _IterDataPipeSerializationWrapper(self.dataset)
        elif isinstance(self.dataset, MapDataPipe):
            self.dataset = _MapDataPipeSerializationWrapper(self.dataset)

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and IterableDataset ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if isinstance(dataset, IterDataPipe):
                if shuffle is not None:
                    dataset = torch.utils.data.graph_settings.apply_shuffle_settings(
                        dataset, shuffle=shuffle
                    )
            # We cannot check `shuffle is not None` here, since previously `shuffle=False` was the default.
            elif shuffle not in {False, None}:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}"
                )

            if sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={sampler}"
                )
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    f"batch_sampler option, but got batch_sampler={batch_sampler}"
                )
        else:
            shuffle = bool(shuffle)
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError(
                    "batch_sampler option is mutually exclusive "
                    "with batch_size, shuffle, sampler, and "
                    "drop_last"
                )
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError(
                    "batch_size=None option disables auto-batching "
                    "and is mutually exclusive with drop_last"
                )

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
                else:
                    sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.resort_fn = resort_fn
        self.persistent_workers = persistent_workers

        self.__initialized = True
        self._IterableDataset_len_called = (
            None  # See NOTE [ IterableDataset and __len__ ]
        )

        self._iterator = None

        self.check_worker_number_rationality()

        torch.set_vital("Dataloader", "enabled", "True")  # type: ignore[attr-defined]

    def _get_iterator(self) -> "_BaseDataLoaderIter":
        """
        Return the corresponding data loader iterator based on the number of worker processes.

        Selects either single-process or multi-process data loader iterator based on num_workers parameter:
        - When num_workers is 0, use single-process iterator to load data directly in the main process
        - When num_workers is greater than 0, use multi-process iterator to load data in parallel using subprocesses

        Returns:
            _BaseDataLoaderIter: Data loader iterator instance, specific type depends on the number of worker processes
        """
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)

    @property
    def multiprocessing_context(self):
        """Get the multiprocessing context object of the data loader

        Returns:
            The multiprocessing context object currently used by this data loader
        """
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        """Set the multiprocessing context object of the data loader

        This method configures the multiprocessing context used by the data loader,
        supporting string-form start method names or direct context objects.
        Only effective when num_workers > 0.

        Args:
            multiprocessing_context: Multiprocessing context object, can be a string
                (e.g., 'spawn', 'fork') or a multiprocessing.context.BaseContext object.
                If None, clears the setting.

        Raises:
            ValueError: When setting multiprocessing context with num_workers=0,
                or when the provided start method name is invalid
            TypeError: When the provided object is neither a string nor a valid context object
        """
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = torch.multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    multiprocessing_context = torch.multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                    multiprocessing_context, python_multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        "multiprocessing_context option should be a valid context "
                        "object or a string specifying the start method, but got "
                        f"multiprocessing_context={multiprocessing_context}"
                    )
            else:
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multi-process loading (num_workers > 0), but got "
                    f"num_workers={self.num_workers}"
                )

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        """
        Override attribute setter to prevent modifying key configuration attributes after DataLoader initialization

        This method serves as a safety protection mechanism for DataLoader, ensuring that some critical
        configuration attributes cannot be accidentally modified after object initialization,
        thus maintaining the consistency of DataLoader behavior.

        Args:
            attr: Attribute name to set
            val: Attribute value to set

        Raises:
            ValueError: Raised when attempting to modify key configuration attributes after initialization

        Note:
            - Protected attributes include: batch_size, batch_sampler, sampler, drop_last,
              dataset, persistent_workers
            - Modifying these attributes may affect the internal state and behavior of the data loader,
              so they are not allowed to be modified after initialization
        """
        if self.__initialized and attr in (
            "batch_size",
            "batch_sampler",
            "sampler",
            "drop_last",
            "dataset",
            "persistent_workers",
        ):
            raise ValueError(
                f"{attr} attribute should not be set after {self.__class__.__name__} is initialized"
            )

        super().__setattr__(attr, val)

    # We quote '_BaseDataLoaderIter' since it isn't defined yet and the definition can't be moved up
    # since '_BaseDataLoaderIter' references 'DataLoader'.
    def __iter__(self) -> "_BaseDataLoaderIter":
        """
        Return a data loader iterator for iterating through dataset batches.

        Determines iterator creation strategy based on persistent workers setting:
        - When persistent workers are enabled and num_workers > 0, reuses the same iterator object
        - Otherwise, creates a new iterator for each call

        Returns:
            _BaseDataLoaderIter: Data loader iterator object that supports next() operation

        Note:
            - In persistent worker mode, the iterator is created only once during the DataLoader lifecycle
            - In single worker mode, a new iterator is created each time to avoid state reset
        """
        # When using a single worker the returned iterator should be
        # created everytime to avoid resetting its state
        # However, in the case of a multiple workers iterator
        # the iterator is only created once in the lifetime of the
        # DataLoader object so that workers can be reused
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()

    @property
    def _auto_collation(self):
        """Check if the data loader has enabled automatic batch processing mode.

        When batch_sampler is configured, the data loader is in auto-collation mode,
        which automatically groups and organizes samples into batches.

        Returns:
            bool: True if automatic batch processing mode is enabled, False otherwise
        """
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        """
        Get the actual sampler used for generating indices for `_DatasetFetcher`

        This property returns the actual sampler used to generate indices for each
        data read operation. In auto-collation mode, returns `.batch_sampler`,
        otherwise returns `.sampler`.

        Note: Due to backward compatibility reasons, `.sampler` and `.batch_sampler`
              attributes cannot be directly modified.

        Returns:
            Sampler object: Corresponding sampler based on auto-collation mode
                - If in auto-collation mode: returns batch_sampler
                - If not in auto-collation mode: returns sampler
        """
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self) -> int:
        """
        Return the number of batches in the data loader

        Calculation method varies based on dataset type (_DatasetKind):
        - For IterableDataset: Calculate dataset length and adjust based on batch_size and drop_last parameters
        - For MapDataset: Directly return the length of the index sampler

        Note: For `IterableDataset`, `__len__` may be inaccurate during multiprocess data loading
        because samples may be duplicated. Use this result with caution in actual usage.

        Returns:
            int: Number of batches in the data loader
        """
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.

            # Cannot statically verify that dataset is Sized
            length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
            if (
                self.batch_size is not None
            ):  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil

                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)

    def check_worker_number_rationality(self):
        """
        Check if the number of workers in the data loader is reasonable

        Evaluates whether the number of workers the data loader will create is reasonable
        based on current system resources. The main rule is: if the number of workers
        to be created by the data loader exceeds the number of logical CPUs available
        to the current process, a warning is issued.

        Example:
            - System has 2 physical CPUs, each with 16 cores, each core supports 2 threads,
              total logical CPUs = 64
            - Logical CPUs available to the current process = 32
            - Reasonable maximum number of workers = 32
            - If the data loader's num_workers=40, a warning will be triggered

        Notes:
            - cpuset restrictions are only considered when os.sched_getaffinity is available (Linux systems)
            - If os.sched_getaffinity is unavailable, os.cpu_count() is used but cpuset is not considered
            - Thread settings are not currently considered as each worker process is single-threaded
            - Thread flags like OMP_NUM_THREADS are not set; callers are responsible for correctly
              configuring third-party modules that depend on these flags
        """
        def _create_warning_msg(num_worker_suggest, num_worker_created, cpuset_checked):
            suggested_max_worker_msg = (
                (
                    (
                        "Our suggested max number of worker in current system is {}{}, which is smaller "
                        "than what this DataLoader is going to create."
                    ).format(
                        num_worker_suggest,
                        (
                            ""
                            if cpuset_checked
                            else " (`cpuset` is not taken into account)"
                        ),
                    )
                )
                if num_worker_suggest is not None
                else (
                    "DataLoader is not able to compute a suggested max number of worker in current system."
                )
            )

            warn_msg = (
                f"This DataLoader will create {num_worker_created} worker processes in "
                f"total. {suggested_max_worker_msg} "
                "Please be aware that excessive worker creation might get DataLoader "
                "running slow or even freeze, "
                "lower the worker number to avoid potential slowness/freeze if necessary."
            )
            return warn_msg

        if not self.num_workers or self.num_workers == 0:
            return

        # try to compute a suggested max number of worker based on system's resource
        max_num_worker_suggest = None
        cpuset_checked = False
        if hasattr(os, "sched_getaffinity"):
            try:
                max_num_worker_suggest = len(os.sched_getaffinity(0))
                cpuset_checked = True
            except Exception:
                pass
        if max_num_worker_suggest is None:
            # os.cpu_count() could return Optional[int]
            # get cpu count first and check None in order to satisfy mypy check
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                max_num_worker_suggest = cpu_count

        if max_num_worker_suggest is None:
            warnings.warn(
                _create_warning_msg(
                    max_num_worker_suggest, self.num_workers, cpuset_checked
                )
            )
            return

        if self.num_workers > max_num_worker_suggest:
            warnings.warn(
                _create_warning_msg(
                    max_num_worker_suggest, self.num_workers, cpuset_checked
                )
            )


class _BaseDataLoaderIter:
    """Base class for data loader iterators

    This class is an abstract base class for PyTorch DataLoader iterators, defining the basic
    interface and behavior for data loading. It provides common functionality for both
    single-process and multi-process data loading, with concrete implementations provided by subclasses.

    Attributes:
        _dataset: Dataset object to iterate over
        _dataset_kind: Dataset type (Map or Iterable)
        _num_workers: Number of worker processes
        _pin_memory: Whether to enable memory pinning
        _timeout: Data loading timeout duration
        _collate_fn: Data batching function
        _sampler_iter: Sampler iterator
        _persistent_workers: Whether to keep worker processes alive
        _num_yielded: Count of yielded data batches
        _profile_name: Name used for performance profiling

    Note:
        - This is an abstract base class and cannot be instantiated directly
        - The _next_data() method must be specifically implemented by subclasses
        - Supports iterator protocol (__iter__ and __next__)
        - Requires handling inter-process communication and synchronization in multi-process environments
    """
    def __init__(self, loader: DataLoader) -> None:
        """
        Initialize base data loader iterator instance

        Copy configuration parameters from the given DataLoader instance and set iterator state.
        This constructor is responsible for:
        1. Handling different types of datasets (especially sharing random seeds for IterDataPipe datasets)
        2. Configuring distributed environment settings
        3. Setting up memory pinning and device configuration
        4. Initializing samplers and random seed generators

        Args:
            loader: Source DataLoader instance from which all configuration parameters are copied

        Note:
            - For IterDataPipe type datasets, creates process group and shares random seeds in distributed environment
            - Memory pinning device configuration: if no device is specified and
            CUDA is available, uses CUDA device by default
            - Sampler state and base seed are determined during initialization to ensure deterministic iteration
        """
        self._dataset = loader.dataset
        self._loader = loader
        self._shared_seed = None
        self._pg = None
        if isinstance(self._dataset, IterDataPipe):
            if dist.is_available() and dist.is_initialized():
                self._pg = dist.new_group(backend="gloo")
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )
        self._dataset_kind = loader._dataset_kind
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        self._auto_collation = loader._auto_collation
        self._drop_last = loader.drop_last
        self._index_sampler = loader._index_sampler
        self._num_workers = loader.num_workers
        ws, rank = _get_distributed_settings()
        self._world_size = ws
        self._rank = rank
        # for other backends, pin_memory_device need to set. if not set
        # default behaviour is CUDA device. if pin_memory_device is selected
        # and pin_memory is not set, the default behaviour false.
        if len(loader.pin_memory_device) == 0:
            self._pin_memory = loader.pin_memory and torch.cuda.is_available()
            self._pin_memory_device = None
        else:
            if not loader.pin_memory:
                warn_msg = (
                    "pin memory device is set and pin_memory flag is not used then device pinned memory won't be used"
                    "please set pin_memory to true, if you need to use the device pin memory"
                )
                warnings.warn(warn_msg)

            self._pin_memory = loader.pin_memory
            self._pin_memory_device = loader.pin_memory_device
        self._timeout = loader.timeout
        self._collate_fn = loader.collate_fn
        self._sampler_iter = iter(self._index_sampler)
        self._base_seed = (
            torch.empty((), dtype=torch.int64)
            .random_(generator=loader.generator)
            .item()
        )
        self._persistent_workers = loader.persistent_workers
        self._num_yielded = 0
        self._profile_name = f"enumerate(DataLoader)#{self.__class__.__name__}.__next__"

    def __iter__(self) -> "_BaseDataLoaderIter":
        """
        Implement iterator protocol, return the iterator object itself

        This method enables the data loader iterator to be used in iteration scenarios
        like for loops. As part of the iterator protocol, it needs to return the
        iterator object itself.

        Returns:
            _BaseDataLoaderIter: Data loader iterator instance
        """
        return self

    def _reset(self, loader, first_iter=False):
        """Reset the internal state of the data loader iterator

        Used to reinitialize key state variables at the beginning of iteration,
        including sampler iterators and counters. For IterDataPipe type datasets,
        also resets distributed random seeds to ensure data shuffling consistency.

        Args:
            loader: Associated DataLoader instance
            first_iter: Whether this is the first iteration (default False)

        Note:
            - In multiprocess environments, this method ensures all processes use the same random seed
            - Reset operation does not affect the original DataLoader configuration
        """
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        if isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(
                self._dataset, shared_rng
            )

    def _next_index(self):
        """Get the next index from the sampler iterator

        This method is responsible for fetching the next data index from the sampler
        iterator, which is a key step in the data loading process.

        Returns:
            Index value of the next data sample

        Raises:
            StopIteration: Raised when the iterator is exhausted
        """
        return next(self._sampler_iter)  # may raise StopIteration

    def _next_data(self):
        """Fetch the next data batch

        This is an abstract method that needs to be implemented by concrete DataLoader
        iterator subclasses. Responsible for loading and returning the next data sample
        or batch from the data source.

        Returns:
            Next data batch, specific type and format determined by subclass implementation

        Raises:
            StopIteration: Raised when data loading completes or iterator is exhausted
            NotImplementedError: If the subclass does not implement this method

        Note:
            - This method is called by __next__ method and is the core logic of data loading
            - Subclasses need to handle specific details like data preprocessing, batch processing,
              multiprocess communication, etc.
        """
        raise NotImplementedError

    def __next__(self) -> Any:
        """Get the next data batch

        Core method implementing the iterator protocol, returns the next data batch
        during data loading. Before returning data, it checks the sampler state
        and performs necessary initialization resets. For iterable datasets,
        it also checks if the number of batches exceeds the preset length and issues warnings.

        Returns:
            Any: Preprocessed data batch, specific format determined by data loader configuration

        Raises:
            StopIteration: Raised when dataset iteration completes with no more data
        """
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset(self._loader)  # type: ignore[call-arg]
            data = self._next_data()
            self._num_yielded += 1
            if (
                self._dataset_kind == _DatasetKind.Iterable
                and self._IterableDataset_len_called is not None
                and self._num_yielded > self._IterableDataset_len_called
            ):
                warn_msg = (
                    f"Length of IterableDataset {self._dataset} was reported to be {self._IterableDataset_len_called}"
                    f"(when accessing len(dataloader)), but {self._num_yielded} samples have been fetched. "
                )
                if self._num_workers > 0:
                    warn_msg += (
                        "For multiprocessing data-loading, this could be caused by not properly configuring the "
                        "IterableDataset replica at each worker. Please see "
                        "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples."
                    )
                warnings.warn(warn_msg)
            return data

    def __len__(self) -> int:
        """
        Return the number of batches in the data loader iterator

        This method implements len() function support, returning the number of
        available batches in the current batch sampler. For Map-style datasets,
        returns the effective number of batches calculated based on sampler
        and dataset configuration.

        Returns:
            int: Total number of available batches in the data loader iterator
        """
        return len(self._index_sampler)

    def __getstate__(self):
        """
        Implement pickle serialization protocol, define object state during serialization

        Currently, data loader iterator serialization is not supported because
        sharing iterators across multiple threads (like HOGWILD) requires handling
        data pushing in separate threads and sharing data queues, but the lack of
        non-blocking APIs makes end signal handling complex.

        Raises:
            NotImplementedError: Current class does not support serialization operations
        """
        # TODO: add limited pickling support for sharing an iterator
        # across multiple threads for HOGWILD.
        # Probably the best way to do this is by moving the sample pushing
        # to a separate thread and then just sharing the data queue
        # but signalling the end is tricky without a non-blocking API
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    """Single-process data loader iterator

    Inherits from _BaseDataLoaderIter, implements a single-threaded data loading strategy
    that loads data directly in the main process. Suitable for num_workers=0 scenarios,
    avoids multiprocessing overhead but has lower performance.

    Attributes:
        _dataset_fetcher: Dataset fetcher instance responsible for fetching actual data samples based on indices
        All attributes inherited from the base class

    Note:
        - Must be used with num_workers=0 configuration
        - No timeout restriction (timeout=0)
        - Supports distributed data sharding for DataPipe types
        - Suitable for simple scenarios or debugging environments
    """
    def __init__(self, loader):
        """
        Initialize single-process data loader iterator

        Inherits base class initialization, verifies single-process configuration constraints,
        sets up dataset sharding and data fetcher. Suitable for scenarios where data loading
        happens directly in the main process.

        Args:
            loader: Data loader configuration source from which all settings and state are inherited

        Raises:
            AssertionError: If timeout is not 0 or num_workers is not 0, which violates single-process requirements

        Note:
            - Must be used with num_workers=0 DataLoader
            - Supports DataPipe dataset sharding to maintain compatibility with multi-process version
            - No timeout restrictions as all operations are executed synchronously in the main thread
        """
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            # For BC, use default SHARDING_PRIORITIES
            torch.utils.data.graph_settings.apply_sharding(
                self._dataset, self._world_size, self._rank
            )

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_data(self):
        """
        Get the next data batch

        Implements the core logic for single-process data loading: gets index from sampler,
        uses dataset fetcher to retrieve data, and enables memory pinning based on configuration.

        Returns:
            Any: Processed data batch, specific format determined by data loader configuration

        Raises:
            StopIteration: Raised when dataset iteration completes

        Note:
            - Data fetching process happens entirely in the main process, no multiprocess communication involved
            - Memory pinning operations only transfer data between CPU and GPU devices, does not affect data content
            - Exception handling depends on the implementation of the underlying data fetcher
        """
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    r"""Iterates once over the DataLoader's dataset, as specified by the sampler."""

    # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
    #
    # Preliminary:
    #
    # Our data model looks like this (queues are indicated with curly brackets):
    #
    #                main process                              ||
    #                     |                                    ||
    #               {index_queue}                              ||
    #                     |                                    ||
    #              worker processes                            ||     DATA
    #                     |                                    ||
    #            {worker_result_queue}                         ||     FLOW
    #                     |                                    ||
    #      pin_memory_thread of main process                   ||   DIRECTION
    #                     |                                    ||
    #               {data_queue}                               ||
    #                     |                                    ||
    #                data output                               \/
    #
    # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
    #      `pin_memory=False`.
    #
    #
    # Terminating multiprocessing logic requires very careful design. In
    # particular, we need to make sure that
    #
    #   1. The iterator gracefully exits the workers when its last reference is
    #      gone or it is depleted.
    #
    #      In this case, the workers should be gracefully exited because the
    #      main process may still need to continue to run, and we want cleaning
    #      up code in the workers to be executed (e.g., releasing GPU memory).
    #      Naturally, we implement the shutdown logic in `__del__` of
    #      DataLoaderIterator.
    #
    #      We delay the discussion on the logic in this case until later.
    #
    #   2. The iterator exits the workers when the loader process and/or worker
    #      processes exits normally or with error.
    #
    #      We set all workers and `pin_memory_thread` to have `daemon=True`.
    #
    #      You may ask, why can't we make the workers non-daemonic, and
    #      gracefully exit using the same logic as we have in `__del__` when the
    #      iterator gets deleted (see 1 above)?
    #
    #      First of all, `__del__` is **not** guaranteed to be called when
    #      interpreter exits. Even if it is called, by the time it executes,
    #      many Python core library resources may already be freed, and even
    #      simple things like acquiring an internal lock of a queue may hang.
    #      Therefore, in this case, we actually need to prevent `__del__` from
    #      being executed, and rely on the automatic termination of daemonic
    #      children.
    #
    #      Thus, we register an `atexit` hook that sets a global flag
    #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
    #      reverse order of registration, we are guaranteed that this flag is
    #      set before library resources we use are freed (which, at least in
    #      CPython, is done via an `atexit` handler defined in
    #      `multiprocessing/util.py`
    #      https://github.com/python/cpython/blob/
    #      c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
    #      registered when an object requiring this mechanism is first
    #      created, e.g., `mp.Queue`
    #      https://github.com/python/cpython/blob/
    #      c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
    #      https://github.com/python/cpython/blob/
    #      c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
    #      )
    #
    #      So in `__del__`, we check if `_utils.python_exit_status` is set or
    #      `None` (freed), and perform no-op if so.
    #
    #      However, simply letting library clean-up codes run can also be bad,
    #      because such codes (i.e., `multiprocessing.util._exit_function()`)
    #      include join putting threads for `mp.Queue`, which can be blocking.
    #      Hence, the main process putting threads are called with
    #      `cancel_join_thread` at creation.  See later section
    #      [ 3b. A process won't hang when putting into a queue; ]
    #      for more details.
    #
    #      Here are two example cases where library clean-up codes can run
    #      before `__del__` is called:
    #
    #        1. If we hold onto a reference to the iterator, it more often
    #           than not tries to do `multiprocessing` library cleaning before
    #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
    #           and thus prevents our cleaning-up code to run first.
    #
    #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
    #           When a process ends, it shuts the all its daemonic children
    #           down with a SIGTERM (instead of joining them without a timeout).
    #           Simiarly for threads, but by a different mechanism. This fact,
    #           together with a few implementation details of multiprocessing, forces
    #           us to make workers daemonic. All of our problems arise when a
    #           DataLoader is used in a subprocess, and are caused by multiprocessing
    #           code which looks more or less like this:
    #
    #               try:
    #                   your_function_using_a_dataloader()
    #               finally:
    #                   multiprocessing.util._exit_function()
    #
    #           The joining/termination mentioned above happens inside
    #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
    #           throws, the stack trace stored in the exception will prevent the
    #           frame which uses `DataLoaderIter` to be freed. If the frame has any
    #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
    #           its  `__del__`, which starts the shutdown procedure, will not be
    #           called. That, in turn, means that workers aren't notified. Attempting
    #           to join in `_exit_function` will then result in a hang.
    #
    #           For context, `_exit_function` is also registered as an `atexit` call.
    #           So it is unclear to me (@ssnl) why this is needed in a finally block.
    #           The code dates back to 2008 and there is no comment on the original
    #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
    #           the finally block and the `atexit` registration) that explains this.
    #
    #
    #      Finally, another choice is to just shutdown workers with logic in 1
    #      above whenever we see an error in `next`. This isn't ideal because
    #        a. It prevents users from using try-catch to resume data loading.
    #        b. It doesn't prevent hanging if users have references to the
    #           iterator.
    #
    #   3. All processes exit if any of them die unexpectedly by fatal signals.
    #
    #      As shown above, the workers are set as daemonic children of the main
    #      process. However, automatic cleaning-up of such child processes only
    #      happens if the parent process exits gracefully (e.g., not via fatal
    #      signals like SIGKILL). So we must ensure that each process will exit
    #      even the process that should send/receive data to/from it were
    #      killed, i.e.,
    #
    #        a. A process won't hang when getting from a queue.
    #
    #           Even with carefully designed data dependencies (i.e., a `put()`
    #           always corresponding to a `get()`), hanging on `get()` can still
    #           happen when data in queue is corrupted (e.g., due to
    #           `cancel_join_thread` or unexpected exit).
    #
    #           For child exit, we set a timeout whenever we try to get data
    #           from `data_queue`, and check the workers' status on each timeout
    #           and error.
    #           See `_DataLoaderiter._get_batch()` and
    #           `_DataLoaderiter._try_get_data()` for details.
    #
    #           Additionally, for child exit on non-Windows platforms, we also
    #           register a SIGCHLD handler (which is supported on Windows) on
    #           the main process, which checks if any of the workers fail in the
    #           (Python) handler. This is more efficient and faster in detecting
    #           worker failures, compared to only using the above mechanism.
    #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
    #
    #           For `.get()` calls where the sender(s) is not the workers, we
    #           guard them with timeouts, and check the status of the sender
    #           when timeout happens:
    #             + in the workers, the `_utils.worker.ManagerWatchdog` class
    #               checks the status of the main process.
    #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
    #               check `pin_memory_thread` status periodically until `.get()`
    #               returns or see that `pin_memory_thread` died.
    #
    #        b. A process won't hang when putting into a queue;
    #
    #           We use `mp.Queue` which has a separate background thread to put
    #           objects from an unbounded buffer array. The background thread is
    #           daemonic and usually automatically joined when the process
    #           *exits*.
    #
    #           In case that the receiver has ended abruptly while
    #           reading from the pipe, the join will hang forever.  The usual
    #           solution for this in Python is calling  `q.cancel_join_thread`,
    #           which prevents automatically joining it when finalizing
    #           (exiting).
    #
    #           Nonetheless, `cancel_join_thread` must only be called when the
    #           queue is **not** going to be read from or write into by another
    #           process, because it may hold onto a lock or leave corrupted data
    #           in the queue, leading other readers/writers to hang.
    #
    #           Hence,
    #             + For worker processes, we only do so (for their output
    #               queues, i.e., `worker_result_queue`) before exiting.
    #             + For `pin_memory_thread`, its output queue `data_queue` is a
    #               `queue.Queue` that does blocking `put` if the queue is full.
    #               So there is no above problem, but as a result, in
    #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
    #               that breaks not only upon success, but also when the main
    #               process stops reading, i.e., is shutting down.
    #             + For loader process, we `cancel_join_thread()` for all
    #               `_index_queues` because the whole purpose of workers and
    #               `pin_memory_thread` is to serve the loader process.  If
    #               loader process is already exiting, we don't really care if
    #               the queues are corrupted.
    #
    #
    # Now let's get back to 1:
    #   how we gracefully exit the workers when the last reference to the
    #   iterator is gone.
    #
    # To achieve this, we implement the following logic along with the design
    # choices mentioned above:
    #
    # `workers_done_event`:
    #   A `multiprocessing.Event` shared among the main process and all worker
    #   processes. This is used to signal the workers that the iterator is
    #   shutting down. After it is set, they will not send processed data to
    #   queues anymore, and only wait for the final `None` before exiting.
    #   `done_event` isn't strictly needed. I.e., we can just check for `None`
    #   from the input queue, but it allows us to skip wasting resources
    #   processing data if we are already shutting down.
    #
    # `pin_memory_thread_done_event`:
    #   A `threading.Event` for a similar purpose to that of
    #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
    #   that separate events are needed is that `pin_memory_thread` reads from
    #   the output queue of the workers. But the workers, upon seeing that
    #   `workers_done_event` is set, only wants to see the final `None`, and is
    #   not required to flush all data in the output queue (e.g., it may call
    #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
    #   happens to exhaust coincidentally, which is out of the control of the
    #   main process). Thus, since we will exit `pin_memory_thread` before the
    #   workers (see below), two separete events are used.
    #
    # NOTE: In short, the protocol is that the main process will set these
    #       `done_event`s and then the corresponding processes/threads a `None`,
    #       and that they may exit at any time after receiving the `None`.
    #
    # NOTE: Using `None` as the final signal is valid, since normal data will
    #       always be a 2-tuple with the 1st element being the index of the data
    #       transferred (different from dataset index/key), and the 2nd being
    #       either the dataset key or the data sample (depending on which part
    #       of the data model the queue is at).
    #
    # [ worker processes ]
    #   While loader process is alive:
    #     Get from `index_queue`.
    #       If get anything else,
    #          Check `workers_done_event`.
    #            If set, continue to next iteration
    #                    i.e., keep getting until see the `None`, then exit.
    #            Otherwise, process data:
    #                If is fetching from an `IterableDataset` and the iterator
    #                    is exhausted, send an `_IterableDatasetStopIteration`
    #                    object to signal iteration end. The main process, upon
    #                    receiving such an object, will send `None` to this
    #                    worker and not use the corresponding `index_queue`
    #                    anymore.
    #       If timed out,
    #          No matter `workers_done_event` is set (still need to see `None`)
    #          or not, must continue to next iteration.
    #   (outside loop)
    #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
    #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
    #                                          main process won't read from it;
    #                                          other workers will also call
    #                                          `cancel_join_thread`.)
    #
    # [ pin_memory_thread ]
    #   # No need to check main thread. If this thread is alive, the main loader
    #   # thread must be alive, because this thread is set as daemonic.
    #   While `pin_memory_thread_done_event` is not set:
    #     Get from `worker_result_queue`.
    #       If timed out, continue to get in the next iteration.
    #       Otherwise, process data.
    #       While `pin_memory_thread_done_event` is not set:
    #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
    #         If timed out, continue to put in the next iteration.
    #         Otherwise, break, i.e., continuing to the out loop.
    #
    #   NOTE: we don't check the status of the main thread because
    #           1. if the process is killed by fatal signal, `pin_memory_thread`
    #              ends.
    #           2. in other cases, either the cleaning-up in __del__ or the
    #              automatic exit of daemonic thread will take care of it.
    #              This won't busy-wait either because `.get(timeout)` does not
    #              busy-wait.
    #
    # [ main process ]
    #   In the DataLoader Iter's `__del__`
    #     b. Exit `pin_memory_thread`
    #          i.   Set `pin_memory_thread_done_event`.
    #          ii   Put `None` in `worker_result_queue`.
    #          iii. Join the `pin_memory_thread`.
    #          iv.  `worker_result_queue.cancel_join_thread()`.
    #
    #     c. Exit the workers.
    #          i.   Set `workers_done_event`.
    #          ii.  Put `None` in each worker's `index_queue`.
    #          iii. Join the workers.
    #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
    #
    #        NOTE: (c) is better placed after (b) because it may leave corrupted
    #              data in `worker_result_queue`, which `pin_memory_thread`
    #              reads from, in which case the `pin_memory_thread` can only
    #              happen at timing out, which is slow. Nonetheless, same thing
    #              happens if a worker is killed by signal at unfortunate times,
    #              but in other cases, we are better off having a non-corrupted
    #              `worker_result_queue` for `pin_memory_thread`.
    #
    #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
    #         can be omitted
    #
    # NB: `done_event`s isn't strictly needed. E.g., we can just check for
    #     `None` from `index_queue`, but it allows us to skip wasting resources
    #     processing indices already in `index_queue` if we are already shutting
    #     down.

    def __init__(self, loader):
        """
        Initialize multiprocessing data loading iterator.

        This method is responsible for setting up various queues, worker processes,
        and memory pinning threads required for multiprocess data loading.

        Args:
            loader: Parent DataLoader instance containing all necessary configuration parameters such as:
                - prefetch_factor: Prefetch factor
                - num_workers: Number of worker processes
                - multiprocessing_context: Multiprocessing context
                - worker_init_fn: Worker process initialization function
                - pin_memory: Whether to pin memory
                - pin_memory_device: Memory pinning device

        Note:
            - Creates multiple worker processes to handle data loading
            - If pin_memory is enabled, creates an additional thread for memory pinning
            - Registers cleanup functions with atexit to ensure proper process termination
        """
        super().__init__(loader)

        self._prefetch_factor = loader.prefetch_factor

        assert self._num_workers > 0
        assert self._prefetch_factor > 0

        if loader.multiprocessing_context is None:
            multiprocessing_context = torch.multiprocessing
        else:
            multiprocessing_context = loader.multiprocessing_context

        self._worker_init_fn = loader.worker_init_fn

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Additional worker init function will take care of sharding in MP and Distributed
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            self._worker_init_fn = functools.partial(
                _sharding_worker_init_fn,
                self._worker_init_fn,
                self._world_size,
                self._rank,
            )

        # No certainty which module multiprocessing_context is
        self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
        self._worker_pids_set = False
        self._shutdown = False
        self._workers_done_event = multiprocessing_context.Event()
        self.resort_fn = loader.resort_fn

        self._index_queues = []
        self._workers = []
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(
                    self._dataset_kind,
                    self._dataset,
                    index_queue,
                    self._worker_result_queue,
                    self._workers_done_event,
                    self._auto_collation,
                    self._collate_fn,
                    self._drop_last,
                    self._base_seed,
                    self._worker_init_fn,
                    i,
                    self._num_workers,
                    self._persistent_workers,
                    self._shared_seed,
                ),
            )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)

        if self._pin_memory:
            self._pin_memory_thread_done_event = threading.Event()

            # Queue is not type-annotated
            self._data_queue = queue.Queue()  # type: ignore[var-annotated]
            if self._pin_memory_device == "xpu":
                current_device = torch.xpu.current_device()  # type: ignore[attr-defined]
            elif self._pin_memory_device == torch._C._get_privateuse1_backend_name():
                custom_device_mod = getattr(
                    torch, torch._C._get_privateuse1_backend_name()
                )
                current_device = custom_device_mod.current_device()
            else:
                current_device = torch.cuda.current_device()  # choose cuda for default
            from ._utils.pin_memory import pin_memory_loop
            pin_memory_thread = threading.Thread(
                target=pin_memory_loop,
                args=(
                    self._worker_result_queue,
                    self._data_queue,
                    current_device,
                    self._pin_memory_thread_done_event,
                    self.resort_fn,
                    self._pin_memory_device,
                ),
            )
            pin_memory_thread.daemon = True
            pin_memory_thread.start()
            # Similar to workers (see comment above), we only register
            # pin_memory_thread once it is started.
            self._pin_memory_thread = pin_memory_thread
        else:
            self._data_queue = self._worker_result_queue  # type: ignore[assignment]

        # In some rare cases, persistent workers (daemonic processes)
        # would be terminated before `__del__` of iterator is invoked
        # when main process exits
        # It would cause failure when pin_memory_thread tries to read
        # corrupted data from worker_result_queue
        # atexit is used to shutdown thread and child processes in the
        # right sequence before main process exits
        if self._persistent_workers and self._pin_memory:
            import atexit

            for w in self._workers:
                atexit.register(_MultiProcessingDataLoaderIter._clean_up_worker, w)

        # .pid can be None only before process is spawned (not the case, so ignore)
        _utils.signal_handling._set_worker_pids(
            id(self), tuple(w.pid for w in self._workers)
        )  # type: ignore[misc]
        _utils.signal_handling._set_SIGCHLD_handler()
        self._worker_pids_set = True
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        """
        Reset the internal state of the data loader, preparing for a new epoch iteration.

        This method resets all internal counters related to task distribution and reception,
        and decides whether to restart worker processes based on whether this is the first iteration.

        Args:
            loader: Parent DataLoader instance
            first_iter: Whether this is the first iteration
            (if True, no resume iteration signal will be sent to workers)

        Note:
            - This method resets task indices, worker states and task queues
            - For non-first iterations, resume iteration signals will be sent to all workers
            - Prefetches specified number of tasks (_prefetch_factor * _num_workers)
        """
        super()._reset(loader, first_iter)
        self._send_idx = 0  # idx of the next task to be sent to workers
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}
        self._tasks_outstanding = (
            0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        )
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.
        self._workers_status = [True for i in range(self._num_workers)]
        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(self._num_workers):
                self._index_queues[idx].put(
                    _utils.worker._ResumeIteration(self._shared_seed)
                )
            resume_iteration_cnt = self._num_workers
            while resume_iteration_cnt > 0:
                return_idx, return_data = self._get_data()
                if isinstance(return_idx, _utils.worker._ResumeIteration):
                    assert return_data is None
                    resume_iteration_cnt -= 1
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()

    def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):
        """
        Attempt to get data from the data queue with optional timeout

        Attempts to fetch data from `self._data_queue` within the specified timeout.
        Can also be used as an inner loop for non-timeout data fetching,
        using the sender's state as the loop condition.

        If any worker process exits abnormally, a `RuntimeError` exception is raised.
        This error may come from:
        - SIGCHLD handler in `_utils/signal_handling.py` (non-Windows platforms only)
        - Manual checks during error and timeout scenarios (primary detection mechanism on Windows)

        If the number of file descriptors is detected to be approaching system limits,
        a related error prompt is raised.

        Args:
            timeout: Timeout duration, defaults to multiprocess status check interval

        Returns:
            tuple: A tuple containing two elements
                - bool: Whether data was successfully fetched
                - any: The fetched data if successful, None otherwise

        Raises:
            RuntimeError: Raised when worker processes exit abnormally or file descriptors exceed limits
        """
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(
                    f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
                ) from e
            if isinstance(e, queue.Empty):
                return (False, None)

            import errno
            import tempfile

            try:
                # Raise an exception if we are this close to the FDs limit.
                # Apparently, trying to open only one file is not a sufficient
                # test.
                # See NOTE [ DataLoader on Linux and open files limit ]
                fds_limit_margin = 10
                fs = [tempfile.NamedTemporaryFile() for i in range(fds_limit_margin)]
            except OSError as e:
                if e.errno == errno.EMFILE:
                    raise RuntimeError(
                        "Too many open files. Communication with the"
                        " workers is no longer possible. Please increase the"
                        " limit using `ulimit -n` in the shell or change the"
                        " sharing strategy by calling"
                        " `torch.multiprocessing.set_sharing_strategy('file_system')`"
                        " at the beginning of your code"
                    ) from None
            raise

    # NOTE [ DataLoader on Linux and open files limit ]
    #
    # On Linux when DataLoader is used with multiprocessing we pass the data between
    # the root process and the workers through SHM files. We remove those files from
    # the filesystem as soon as they are created and keep them alive by
    # passing around their file descriptors through AF_UNIX sockets. (See
    # docs/source/multiprocessing.rst and 'Multiprocessing Technical Notes` in
    # the wiki (https://github.com/pytorch/pytorch/wiki).)
    #
    # This sometimes leads us to exceeding the open files limit. When that happens,
    # and the offending file descriptor is coming over a socket, the `socket` Python
    # package silently strips the file descriptor from the message, setting only the
    # `MSG_CTRUNC` flag (which might be a bit misleading since the manpage says that
    # it _indicates that some control data were discarded due to lack of space in
    # the buffer for ancillary data_). This might reflect the C implementation of
    # AF_UNIX sockets.
    #
    # This behaviour can be reproduced with the script and instructions at the
    # bottom of this note.
    #
    # When that happens, the standard Python `multiprocessing` (and not
    # `torch.multiprocessing`) raises a `RuntimeError: received 0 items of ancdata`
    #
    # Sometimes, instead of the FD being stripped, you may get an `OSError:
    # Too many open files`, both in the script below and in DataLoader. However,
    # this is rare and seems to be nondeterministic.
    #
    #
    #   #!/usr/bin/env python3
    #   import sys
    #   import socket
    #   import os
    #   import array
    #   import shutil
    #   import socket
    #
    #
    #   if len(sys.argv) != 4:
    #       print("Usage: ", sys.argv[0], " tmp_dirname iteration (send|recv)")
    #       sys.exit(1)
    #
    #   if __name__ == '__main__':
    #       dirname = sys.argv[1]
    #       sock_path = dirname + "/sock"
    #       iterations = int(sys.argv[2])
    #       def dummy_path(i):
    #           return dirname + "/" + str(i) + ".dummy"
    #
    #
    #       if sys.argv[3] == 'send':
    #           while not os.path.exists(sock_path):
    #               pass
    #           client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #           client.connect(sock_path)
    #           for i in range(iterations):
    #               fd = os.open(dummy_path(i), os.O_WRONLY | os.O_CREAT)
    #               ancdata = array.array('i', [fd])
    #               msg = bytes([i % 256])
    #               print("Sending fd ", fd, " (iteration #", i, ")")
    #               client.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, ancdata)])
    #
    #
    #       else:
    #           assert sys.argv[3] == 'recv'
    #
    #           if os.path.exists(dirname):
    #               raise Exception("Directory exists")
    #
    #           os.mkdir(dirname)
    #
    #           print("Opening socket...")
    #           server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    #           server.bind(sock_path)
    #
    #           print("Listening...")
    #           for i in range(iterations):
    #               a = array.array('i')
    #               msg, ancdata, flags, addr = server.recvmsg(1, socket.CMSG_SPACE(a.itemsize))
    #               assert(len(ancdata) == 1)
    #               cmsg_level, cmsg_type, cmsg_data = ancdata[0]
    #               a.frombytes(cmsg_data)
    #               print("Received fd ", a[0], " (iteration #", i, ")")
    #
    #           shutil.rmtree(dirname)
    #
    # Steps to reproduce:
    #
    # 1. Run two shells and set lower file descriptor limit in the receiving one:
    # (shell1) ulimit -n 1020
    # (shell2) ulimit -n 1022
    #
    # 2. Run the script above with the `recv` option in the first shell
    # (shell1) ./test_socket.py sock_tmp 1017 recv
    #
    # 3. Run the script with the `send` option in the second shell:
    # (shell2) ./test_socket.py sock_tmp 1017 send

    def _get_data(self):
        """
        Get data from the data queue with timeout support and pin_memory thread monitoring

        This method is responsible for retrieving processed data from worker processes
        or pin_memory threads, providing three different data fetching strategies:
        1. Timeout mode (timeout > 0): Attempt to fetch data within specified timeout
        2. PinMemory mode: Monitor pin_memory thread status to ensure thread is running normally
        3. Infinite wait mode: Continuously attempt to fetch data until success

        Also periodically checks worker process status. Windows platform relies on this
        mechanism for worker failure detection, while non-Windows platforms also use
        SIGCHLD signal handlers for supplementary detection.

        Returns:
            Data retrieved from the data queue, specific format determined by dataset and collate_fn

        Raises:
            RuntimeError:
                - Failed to fetch data within specified time in timeout mode
                - Pin_memory thread exited unexpectedly
                - Worker process terminated abnormally
        """
        if self._timeout > 0:
            success, data = self._try_get_data(self._timeout)
            if success:
                return data
            else:
                raise RuntimeError(
                    f"DataLoader timed out after {self._timeout} seconds"
                )
        elif self._pin_memory:
            while self._pin_memory_thread.is_alive():
                success, data = self._try_get_data()
                if success:
                    return data
            else:
                # while condition is false, i.e., pin_memory_thread died.
                raise RuntimeError("Pin memory thread exited unexpectedly")
            # In this case, `self._data_queue` is a `queue.Queue`,. But we don't
            # need to call `.task_done()` because we don't use `.join()`.
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _next_data(self):
        """
        Get the next data batch, implementing core logic for multiprocess data loading

        This method is the core iteration logic of the multiprocess data loader, responsible for:
        1. Maintaining task index queues and reception queues
        2. Handling worker status management and failure detection
        3. Supporting both ordered and out-of-order data returns
        4. Handling special cases for IterableDataset and persistent workers

        Specific flow:
        - Continuously check validity of current receive index
        - Clean up completed worker tasks
        - Check if iteration termination conditions are met
        - Fetch and process data, supporting out-of-order caching

        Returns:
            Any: Processed data batch, wrapped by _process_data method

        Raises:
            StopIteration: Raised when all data has been processed
            RuntimeError: May be raised when worker processes or pin_memory thread encounter exceptions

        Note:
            - This method handles edge cases like unexpected worker termination and IterableDataset exhaustion
            - Supports worker state persistence in _persistent_workers mode
            - Automatically handles caching and reordering of out-of-order data
            - Works collaboratively with _get_data, _try_put_index and other methods
        """
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if (
                    len(info) == 2 or self._workers_status[worker_id]
                ):  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data)

    def _try_put_index(self):
        """
        Attempt to distribute the next data index to available worker processes

        This method is responsible for fetching the next index from the sampler
        and assigning it to the next available worker process. This is a core part
        of the multiprocess data loader's prefetching mechanism, ensuring that
        worker processes continuously have tasks to process.

        Process:
          1. Check if current outstanding tasks exceed the prefetch limit
          2. Get the next index from the sampler
          3. Find the next available worker process
          4. Send the index to the selected worker process
          5. Update task tracking information

        Raises:
            AssertionError: Raised when outstanding tasks exceed prefetch factor limit

        Note:
            - Returns immediately if sampler is exhausted (StopIteration)
            - Also returns immediately if no available worker processes are found
            - Each successful task distribution updates _send_idx and _tasks_outstanding counts
        """
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers

        try:
            index = self._next_index()
        except StopIteration:
            return
        for _ in range(self._num_workers):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        self._index_queues[worker_queue_idx].put((self._send_idx, index))  # type: ignore[possibly-undefined]
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1

    def _process_data(self, data):
        """
        Process data received from worker processes

        This method is a core part of the multiprocess data loader's data processing flow, responsible for:
        1. Updating the received index count
        2. Triggering new task distribution to maintain worker load balancing
        3. Checking and handling exception data wrappers
        4. Returning valid data for use by upper-level iterators

        Args:
            data: Data received from worker process queue, may contain data batches or exception wrappers

        Returns:
            Processed data object, if input is an exception it will be re-raised

        Note:
            - Each successful data reception increments the _rcvd_idx counter
            - Maintains continuous task distribution via _try_put_index()
            - ExceptionWrapper type exceptions are re-raised in the current process
        """
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        """
        Mark a worker process as unavailable.

        Called when a worker process completes its task (e.g., exhausts IterableDataset data).
        Should only be used when _MultiProcessingDataLoaderIter will continue running.

        Args:
            worker_id: ID of the worker process to mark as unavailable
            shutdown: Whether shutdown is in progress, defaults to False. When set to True,
                      allows calling this method even if worker status is False, as long as
                      persistent workers are enabled

        Note:
            - This method sends termination signal to the specific worker process but doesn't immediately join it
            - Actual join operation is deferred to the `_shutdown_workers` method
            - This prevents exceptions from being ignored that might be raised after worker completion
        """

        assert self._workers_status[worker_id] or (
            self._persistent_workers and shutdown
        )

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        assert self._workers_done_event.is_set() == shutdown

    def _shutdown_workers(self):
        """
        Shutdown worker processes and memory pin thread of the multiprocess data loader.

        This method is responsible for safely shutting down all worker processes and
        memory pin threads, handling cleanup operations for both normal shutdown and
        exception scenarios. Follows the NOTE [ Data Loader Multiprocessing Shutdown Logic ].

        Execution flow:
        1. First shutdown pin_memory_thread to avoid reading from potentially corrupted worker_result_queue
        2. Set workers_done_event to notify all worker processes to stop
        3. Mark all worker processes as unavailable
        4. Wait for worker processes to exit normally, force termination if timeout
        5. Close all queues and clean up worker pids

        Note:
            - This method skips shutdown logic during Python exit
            - Windows platform lacks SIGCHLD handling mechanism, cannot detect worker process errors
        """
        if (
            _utils is None
            or _utils.python_exit_status is True
            or _utils.python_exit_status is None
        ):
            # See (2) of the note. If Python is shutting down, do no-op.
            return
        # Normal exit when last reference is gone / iterator is depleted.
        # See (1) and the second half of the note.
        if not self._shutdown:
            self._shutdown = True
            try:
                # Normal exit when last reference is gone / iterator is depleted.
                # See (1) and the second half of the note.

                # Exit `pin_memory_thread` first because exiting workers may leave
                # corrupted data in `worker_result_queue` which `pin_memory_thread`
                # reads from.
                if hasattr(self, "_pin_memory_thread"):
                    # Use hasattr in case error happens before we set the attribute.
                    self._pin_memory_thread_done_event.set()
                    # Send something to pin_memory_thread in case it is waiting
                    # so that it can wake up and check `pin_memory_thread_done_event`
                    self._worker_result_queue.put((None, None))
                    self._pin_memory_thread.join()
                    self._worker_result_queue.cancel_join_thread()
                    self._worker_result_queue.close()

                # Exit workers now.
                self._workers_done_event.set()
                for worker_id in range(len(self._workers)):
                    # Get number of workers from `len(self._workers)` instead of
                    # `self._num_workers` in case we error before starting all
                    # workers.
                    # If we are using workers_status with persistent_workers
                    # we have to shut it down because the worker is paused
                    if self._persistent_workers or self._workers_status[worker_id]:
                        self._mark_worker_as_unavailable(worker_id, shutdown=True)
                for w in self._workers:
                    # We should be able to join here, but in case anything went
                    # wrong, we set a timeout and if the workers fail to join,
                    # they are killed in the `finally` block.
                    w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
                for q in self._index_queues:
                    q.cancel_join_thread()
                    q.close()
            finally:
                # Even though all this function does is putting into queues that
                # we have called `cancel_join_thread` on, weird things can
                # happen when a worker is killed by a signal, e.g., hanging in
                # `Event.set()`. So we need to guard this with SIGCHLD handler,
                # and remove pids from the C side data structure only at the
                # end.
                #
                # FIXME: Unfortunately, for Windows, we are missing a worker
                #        error detection mechanism here in this function, as it
                #        doesn't provide a SIGCHLD handler.
                if self._worker_pids_set:
                    _utils.signal_handling._remove_worker_pids(id(self))
                    self._worker_pids_set = False
                for w in self._workers:
                    if w.is_alive():
                        # Existing mechanisms try to make the workers exit
                        # peacefully, but in case that we unfortunately reach
                        # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                        # we kill the worker.
                        w.terminate()

    @staticmethod
    def _clean_up_worker(w):
        """
        Clean up a single worker process

        This method is used to gracefully terminate a specified worker process.
        First attempts to wait for the process to exit normally, and if the process
        does not exit within the specified time, forcefully terminates it.

        Args:
            w: multiprocessing.Process instance, the worker process object to clean up

        Note:
            - Defined as static method to avoid holding references
            to the main iterator instance, preventing circular references
            - Uses MP_STATUS_CHECK_INTERVAL as wait timeout
            - Uses try-finally structure to ensure the process is eventually terminated
        """
        try:
            w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
        finally:
            if w.is_alive():
                w.terminate()

    def __del__(self):
        """
        Destructor method for multiprocess data loader iterator.

        Automatically called when the iterator object is garbage collected,
        used to clean up all worker processes and resources.
        This method calls _shutdown_workers to safely shut down worker processes
        and memory pinning threads.

        Note:
            - This is a Python destructor method that is automatically called during object destruction
            - Ensures proper resource cleanup in multiprocess environments, avoiding zombie processes
            - Follows the NOTE [ Data Loader Multiprocessing Shutdown Logic ] shutdown logic
        """
        self._shutdown_workers()
