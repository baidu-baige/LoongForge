# Copyright 2026 The OmniTraining Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-Energon under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION.

"""energon dataloader wrapper"""

import gc
import multiprocessing
from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from megatron.energon.cache import CachePool
from megatron.energon.deprecation import deprecated
from megatron.energon.flavors.base_dataset import (
    FlexState,
    SavableDataset,
    State,
    add_sample_restore_key,
)
from megatron.energon.savable_loader import (SavableDatasetCheckpoint,
                                             SavableCheckpoint,
                                             SavableDatasetState,
                                             SavableDatasetWrapper,
                                             SimpleSavableDatasetWrapper,
                                             SavableDataLoaderState, _init_worker)
from megatron.energon.wrappers.base import BaseWrapperDataset
from megatron.energon.wrappers.batch_dataset import BatchDataset
from megatron.energon.wrappers.gc_dataset import GC_DEFAULT_EVERY_N_ITER, GcDataset, gc_init_worker
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.log_sample_dataset import default_get_batch_keys
from megatron.energon.wrappers.watchdog_dataset import WatchdogDataset

import torch
from omni_training.data.dp_balance.dataloader.dataloader import DataLoader

T = TypeVar("T")

class SavableDataLoader(DataLoader[T], Generic[T]):
    """DataLoader that supports saving and restoring the state of the dataset.
    When restoring, the dataloader and dataset must be instantiated with the exactly same
    parameters.

    How this works (for no worker processes)
    ----------------------------------------

    1. The state of the dataset is saved using :meth:`megatron.energon.SavableDataset.save_state`
    2. (for compatibility) The state of the dataset is converted to using inner arrays using
       :meth:`megatron.energon.SavableDataset.merge_states`.
    3. The state can be restored using :meth:`megatron.energon.SavableDataset.restore_state` given the
       previously saved (and merged) state.

    How this works (for worker processes)
    -------------------------------------

    - First issue is, that worker processes work with internal queues between processes to pass
      loaded samples to the main process (also to perform collating). This means that the whole
      state of the dataset is not directly accessible from the main process.
    - To solve this issue, the dataset regularly saves a checkpoint of its state to be able to
      resume from that state (and skip the samples that have already been yielded).
    - To have a consistent state, the sample index from the latest yielded samples is saved for all
      worker instances. Thus, the main process knows exactly which sample indexes should come next
      from which worker.
    - Internally, pytorch iterates through the workers in order to retrieve the next worker's
      samples. Unfortunately, that next worker index cannot be restored in pytorch's dataloader,
      thus the workers are shifted internally by that offset
      (see :attr:`megatron.energon.WorkerConfig.worker_id_offset`).

    1. The dataset is wrapped in a :class:`megatron.energon.SavableDatasetWrapper`. This allows the main
       process to communicate with the worker and send commands to the workers and retrieve the
       results.
    2. The state of the dataset is saved using
       :meth:`megatron.energon.SavableDatasetWrapper.get_checkpoint`. This gives the last checkpoint
       from the requested sample index and stores the offset (i.e. number of samples to skip) from
       that checkpoint.
    3. The state is merged using :meth:`megatron.energon.SavableDatasetWrapper.merge_checkpoints`. This
       merges the states of all workers and returns a single state that can be used to restore the
       state of the dataset.
    4. The state can be restored using :meth:`megatron.energon.SavableDatasetWrapper.restore_state`
       before a worker is started, such that all workers initially receive the same state array.
       The worker firstly sets the worker index offset, then uses its (shifted) own index to get its
       required state from the merged state array.

    """

    #: The worker config
    worker_config: WorkerConfig
    #: The wrapped dataset. For multiprocessing, this is a :class:`megatron.energon.SavableDatasetWrapper`
    dataset: Union[SavableDatasetWrapper[T], SimpleSavableDatasetWrapper[T]]

    #: The global ID counter
    _next_id: ClassVar[int] = 0
    #: Class instance id
    id: int = 0

    #: The queues used to send commands to the workers
    cmd_queues: List[torch.multiprocessing.Queue]
    #: The queues used to receive results from the workers
    result_queues: List[torch.multiprocessing.Queue]

    #: Instance of the current data iterator. There shall be only one active iterator, such that the
    # dataset is not iterated multiple times in parallel. The state will continue between epochs.
    _epoch_iterator: Optional[Iterator[T]] = None
    #: Whether the dataloader has running workers.
    _has_workers: bool = False
    #: The index of the current worker. -1 if not started yet.
    _worker_sample_counters: List[int]
    #: Id of the next worker to retrieve data from
    _next_worker_id: int = 0
    #: Global index of the last yielded sample
    _global_sample_idx: int = 0
    #: Current iterator index of the last yielded sample
    _sample_idx: int = 0

    def __init__(
        self,
        dataset: SavableDataset[T],
        *,
        checkpoint_every_sec: float = 60,
        checkpoint_every_min_n_samples: Optional[int] = None,
        n_checkpoints: Optional[int] = None,
        gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
        gc_freeze_at_start: bool = True,
        prefetch_factor: int = 2,
        cache_pool: Optional[CachePool] = None,
        watchdog_timeout_seconds: Optional[float] = 60,
        watchdog_initial_timeout_seconds: Optional[float] = None,
        fail_on_timeout: bool = False,
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load.
            worker_config: The worker config to use
            checkpoint_every_sec: This is the time in seconds after which a checkpoint is saved.
                It may take the same duration to restore a checkpoint, but introduces additional
                overhead during reading data from the dataset, so this should be chosen accordingly.
                Only applies if using workers.
            checkpoint_every_min_n_samples: Overwrites the minimum number of samples between
                checkpoints. Defaults to `number of workers * 2`. Only applies if using workers.
            n_checkpoints: The number of checkpoints to keep in memory. Only applies if using
                workers. If None, computes a suitable value.
            gc_collect_every_n_steps: The number of steps after which the garbage collector is
                called. As we're usually handling large (but few) tensors here, and the python
                garbage collection is already full of objects just by importing, this can improve
                the memory footprint quite a lot, and may even be necessary to avoid memory
                overflow.
            gc_freeze_at_start: If true, the garbage collector is frozen at the start of the worker
                processes. This improves the garbage collection performance by a lot.
                In rare cases, this may cause issues and can be disabled. Keep enabled if you
                experience no issues.
            cache_pool: If set, the cache pool to use for the dataset.
            watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds.
            If None, the timeout is the same as watchdog_timeout_seconds.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
        """
        self.worker_config = dataset.worker_config
        self.id = self.next_id()

        dataset = WatchdogDataset(
            dataset,
            worker_config=self.worker_config,
            timeout_seconds=watchdog_timeout_seconds,
            initial_timeout_seconds=watchdog_initial_timeout_seconds,
            fail_on_timeout=fail_on_timeout,
        )

        if gc_collect_every_n_steps > 0:
            dataset = GcDataset(
                dataset,
                worker_config=self.worker_config,
                every_n_iter=gc_collect_every_n_steps,
                freeze=gc_freeze_at_start,
            )

        self.cmd_queues = [multiprocessing.Queue() for _ in range(self.worker_config.num_workers)]
        self.result_queues = [
            multiprocessing.Queue() for _ in range(self.worker_config.num_workers)
        ]

        num_procs = max(self.worker_config.num_workers, 1)

        if n_checkpoints is None:
            n_checkpoints = prefetch_factor * num_procs + 1

        if self.worker_config.num_workers > 0:
            if checkpoint_every_min_n_samples is None:
                checkpoint_every_min_n_samples = self.worker_config.num_workers * 2

            dataset = SavableDatasetWrapper(
                dataset,
                self.worker_config,
                checkpoint_every_sec=checkpoint_every_sec,
                checkpoint_every_min_n_samples=checkpoint_every_min_n_samples,
                n_checkpoints=n_checkpoints,
                cmd_queues=self.cmd_queues,
                result_queues=self.result_queues,
                cache_pool=cache_pool,
            )
        else:
            dataset = SimpleSavableDatasetWrapper(
                dataset, self.worker_config, cache_pool=cache_pool
            )

        self._worker_sample_counters = [-1] * num_procs

        kwargs = {}
        if self.worker_config.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["multiprocessing_context"] = "fork"

        # Assert that prefetch_factor works well with num_checkpoints.
        # This ensures that the oldest checkpoint is old enough to cover
        # all the buffered samples in the torch dataloader.
        assert prefetch_factor * num_procs + 1 <= n_checkpoints, (
            "When increasing prefetch_factor, also increase n_checkpoints, so that "
            "the number of checkpoints is at least as large as num_workers * prefetch_factor + 1"
        )

        # Compute seeds for each worker, based on current rank
        seed_per_worker = [
            self.worker_config.worker_seed(i) for i in range(self.worker_config.num_workers)
        ]

        super().__init__(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.worker_config.num_workers,
            pin_memory=True,
            worker_init_fn=partial(_init_worker, seed_per_worker),
            **kwargs,
        )

        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "SavableLoader.__init__",
                    "r": self.worker_config.rank,
                    "w": None,
                    "id": self.id,
                    "config": dataset.config(),
                }
            )

    @staticmethod
    def next_id() -> int:
        """Get the next unique id for a dataloader."""
        next_id = SavableDataLoader._next_id
        SavableDataLoader._next_id += 1
        return next_id

    def __len__(self):
        # We override this, because otherwise we'll see warnings
        return self.dataset.len_rank()

    def _epoch_iter(self):
        """Iterator for one epoch, i.e. until the inner dataset raises StopIteration."""
        iter_idx = 0
        id = self.next_id()
        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "SavableDataLoader.iter",
                    "r": self.worker_config.rank,
                    "w": None,
                    "id": self.id,
                    "iter_id": id,
                }
            )
        try:
            for worker_id, sample_idx, sample in super().__iter__():
                self._worker_sample_counters[worker_id] = sample_idx
                # If the next sample will be from the first worker, we can safely resume
                self._next_worker_id = (worker_id + 1) % max(self.num_workers, 1)
                # self._debugf.write(
                #     f"[w={worker_id}, s={sample_idx}] {self._sample_str(sample)}\n"
                # )
                # self._debugf.flush()
                if self.worker_config.should_log(level=1):
                    keys = default_get_batch_keys(sample)
                    self.worker_config.worker_log(
                        {
                            **{
                                "t": "SavableDataLoader.yield",
                                "r": self.worker_config.rank,
                                "w": None,
                                "id": self.id,
                                "iter_id": id,
                                "worker_id": worker_id,
                                "worker_idx": sample_idx,
                                "idx": self._sample_idx,
                                "iter_idx": iter_idx,
                                "global_idx": self._global_sample_idx,
                            },
                            **({} if keys is None else {"keys": keys}),
                        }
                    )
                self._sample_idx += 1
                self._global_sample_idx += 1
                iter_idx += 1
                yield sample
            self._epoch_iterator = None
            self._next_worker_id = 0
        finally:
            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "SavableDataLoader.StopIteration",
                        "r": self.worker_config.rank,
                        "w": None,
                        "id": self.id,
                        "iter_id": self.id,
                    }
                )

    def __iter__(self):
        if self.num_workers > 0:
            # Always keep same iterator alive, as long as it yields data
            if self._epoch_iterator is None:
                self._epoch_iterator = self._epoch_iter()
                self._sample_idx = 0
                self._has_workers = True
                # print("New Iterator", self._persistent_iterator)
            return self._epoch_iterator
        else:
            return self._epoch_iter()

    def _worker_command(self, *cmd_args) -> List[Any]:
        """Executes a command in all workers and returns the results."""
        # print(f"cmd: {cmd_args}")
        for cmd_queue in self.cmd_queues:
            cmd_queue.put(cmd_args)
        # print(f"waiting for res")
        assert len(self.result_queues) == self.worker_config.num_workers
        res = {k: v for results_queue in self.result_queues for k, v in results_queue.get().items()}
        res = [res[i] for i in range(len(res))]
        # print(f"res: {res}")
        for r in res:
            if isinstance(r, Exception):
                raise r
        return res

    def _get_batch_size(self) -> Optional[int]:
        """Try to infer micro batch size from the dataset"""
        if isinstance(self.dataset, (SavableDatasetWrapper, SimpleSavableDatasetWrapper)):
            dataset = self.dataset.dataset
        else:
            dataset = self.dataset

        if (
            isinstance(dataset, BaseWrapperDataset)
            and (bds := dataset._find_wrapped_dataset(BatchDataset)) is not None
        ):
            assert isinstance(bds, BatchDataset)
            return bds.batch_size
        else:
            return None

    def save_state_rank(self) -> Optional[SavableDataLoaderState]:
        """
        Saves the state of the dataset for the current rank. Allows for restoring the state later
        using `restore_state_rank`, given the result of this method.

        Returns:
            The state of the dataset.
        """
        # Fetch current rank's worker's state
        if self.num_workers == 0:
            # No workers configured
            assert isinstance(self.dataset, SimpleSavableDatasetWrapper)
            worker_states = [self.dataset.save_state()]
            assert self._next_worker_id == 0
        elif self._has_workers:
            # Fetch from worker processes
            worker_states = self._worker_command("get_checkpoint", self._worker_sample_counters)
        else:
            # Workers configured, but not started yet.
            # If a state has already been restored, it will be returned.
            assert isinstance(self.dataset, SavableDatasetWrapper)
            worker_states = self.dataset.get_initial_checkpoint()

        if worker_states is None:
            return None

        # Merge the states
        merged_state = SavableDataLoaderState(
            worker_states=worker_states,
            next_worker_id=self._next_worker_id,
            micro_batch_size=self._get_batch_size(),
        )

        # Not distributed -> return the merged state
        return merged_state

    def restore_state_rank(self, state: Optional[SavableDataLoaderState]) -> None:
        """
        Restores the saved state for the current rank.

        Args:
            state: The state to restore, as saved by `save_state_rank`.
        """
        assert not self._has_workers, "Cannot restore state while workers are running"
        if state is None:
            # Assume initial state
            return
        assert isinstance(state, SavableDataLoaderState)

        old_micro_batch_size = state.micro_batch_size
        micro_batch_size = self._get_batch_size()

        if self.num_workers == 0:
            # No workers configured
            assert isinstance(self.dataset, SimpleSavableDatasetWrapper)
            assert micro_batch_size == old_micro_batch_size, (
                "Changing micro batch size is not allowed without workers"
            )

            assert len(state.worker_states) == 1
            assert isinstance(state.worker_states[0], FlexState)
            self.dataset.restore_state(state.worker_states[0])
        else:
            # Workers configured
            assert isinstance(self.dataset, SavableDatasetWrapper)
            assert all(isinstance(s, SavableDatasetCheckpoint) for s in state.worker_states)

            # Check batch sizes (before and after)
            if micro_batch_size != old_micro_batch_size:
                assert micro_batch_size is not None and old_micro_batch_size is not None, (
                    "Cannot resume with different batching mode "
                    "(batching to non-batching or vice versa)"
                )

                if micro_batch_size > old_micro_batch_size:
                    raise ValueError(
                        "Resuming with larger micro batch size is not allowed: "
                        f"{micro_batch_size} > {state.micro_batch_size}"
                    )
                elif (
                    micro_batch_size < old_micro_batch_size
                    and old_micro_batch_size % micro_batch_size != 0
                ):
                    raise ValueError(
                        "Resuming with smaller micro batch size only allowed if the old "
                        f"micro batch size is a multiple of the new one: {micro_batch_size} < {state.micro_batch_size}"
                    )
                batch_size_ratio = old_micro_batch_size // micro_batch_size
                for worker_state in state.worker_states:
                    assert isinstance(worker_state, SavableDatasetCheckpoint)
                    # When resuming with a smaller micro batch size, the offset must be scaled
                    # up to the new micro batch size to skip the same number of samples as before.
                    worker_state.offset *= batch_size_ratio

            self.dataset.restore_checkpoint(state.worker_states, worker_offset=state.next_worker_id)

            # Initialize the worker-sample counters so that every worker owns a valid
            # "last emitted sample" index.  Workers that have not emitted anything yet keep
            # the default value ``-1``.

            assert isinstance(state.worker_states, list)

            self._worker_sample_counters = [
                (
                    ws.state.sample_index - 1
                    if (isinstance(ws, SavableDatasetCheckpoint) and ws.state is not None)
                    else -1
                )
                for ws in state.worker_states
            ]

            self._next_worker_id = state.next_worker_id

    @deprecated(
        "`save_state` is deprecated and was renamed to `save_state_global` and will be removed "
        "in a future update. If you actually do not want to gather the states to a rank, use "
        "`save_state_rank` instead."
    )
    def save_state(self, dst_rank: int) -> Optional[Sequence[Optional[SavableDataLoaderState]]]:
        """Deprecated. Use `save_state_global` (or `save_state_rank`) instead."""

        return self.save_state_global(dst_rank)

    def save_state_global(
        self, global_dst_rank: int
    ) -> Optional[Sequence[Optional[SavableDataLoaderState]]]:
        """
        Saves the state of the dataset globally, collecting the state from all ranks using torch
        distributed. Allows for restoring the state later using `restore_state_global`, given the
        result of this method.
        Typical scenario: Save the state to disk only on the `dst_rank`, the other ranks do not
        save the state. Later, restore the state either only loaded on the `dst_rank` or
        loading on all ranks separately using `restore_state_global`.

        Note: If you want to save/restore the state per rank separately, use `save_state_rank` and
        the corresponding `restore_state_rank`. Also, these do not rely on torch distributed.

        Args:
            global_dst_rank: The state will be gathered to this rank. The rank refers to the
                global rank, not the rank within the data parallel group.

        Returns:
            The state of the dataset (or `None`, if not on `dst_rank`).
        """
        # Fetch current rank's worker's state
        merged_state = self.save_state_rank()

        # Gather the merged states
        if self.worker_config.world_size > 1:
            output: Optional[Sequence[Optional[SavableDataLoaderState]]]
            if self.worker_config.global_rank() == global_dst_rank:
                output = [None] * self.worker_config.world_size
            else:
                # Check if the global_dst_rank is in the same group at all
                if self.worker_config.data_parallel_group is not None:
                    try:
                        _ = torch.distributed.get_group_rank(
                            self.worker_config.data_parallel_group, global_dst_rank
                        )
                    except RuntimeError:
                        raise ValueError(
                            f"global_dst_rank {global_dst_rank} is not in the group of the current rank's worker config"
                        )

                output = None

            torch.distributed.gather_object(
                merged_state,
                output,
                global_dst_rank,
                group=self.worker_config.data_parallel_group,
            )

            return output
        else:
            # Not distributed -> return the merged state
            return [merged_state]

    @deprecated(
        "`restore_state` was renamed to `restore_state_global` and will be removed in a future update."
    )
    def restore_state(
        self,
        state: Optional[Sequence[Optional[SavableDataLoaderState]]],
    ) -> None:
        """Deprecated. Use `restore_state_global` (or `restore_state_rank`) instead."""

        return self.restore_state_global(state)

    def restore_state_global(
        self,
        state: Optional[Sequence[Optional[SavableDataLoaderState]]],
        *,
        src_rank: Optional[int] = None,
    ) -> None:
        """
        Restores the saved state from `save_state_global` (in torch distributed setup).
        The global state needs be loaded on every rank that has a data loader instance.

        Optionally, one can specify a src_rank and only provide the state once.
        In case of multiple data parallel groups, you must provide the state once
        in each data parallel group. In this case the `src_rank` is the rank within the
        data parallel group.

        Args:
            state: The state to restore, as saved by `save_state_global`.
            src_rank: The rank from which the state is broadcasted (within the data parallel group, if using DP groups).
        """

        assert self._epoch_iterator is None, "Cannot restore state while workers are running"

        # Only restore multi-rank if state is actually a list and we are in a distributed setup.
        # Otherwise treat as single rank state.
        if src_rank is None or self.worker_config.world_size == 1:
            assert isinstance(state, list), "State must be a list in distributed setup"
            assert len(state) == self.worker_config.world_size, (
                "State must be a list of size world_size"
            )

            # All ranks have the state
            # Select the state of the current rank
            rank_state = state[self.worker_config.rank]
        else:
            if self.worker_config.data_parallel_group is not None:
                # Only the src_rank has the state within this dp group
                try:
                    global_src_rank = torch.distributed.get_global_rank(
                        self.worker_config.data_parallel_group, src_rank
                    )
                except RuntimeError:
                    raise ValueError(
                        f"src_rank {src_rank} is not in the group of the current rank's worker config"
                    )
            else:
                # If no DP group is given, we assume the global rank is
                # the same as the data parallel rank
                global_src_rank = src_rank

            if self.worker_config.rank != src_rank:
                # Send the state to all other ranks
                assert state is None
                # Must still be a list of Nones
                state = [None] * self.worker_config.world_size
            else:
                assert isinstance(state, list), "State must be a list in distributed setup"
                assert len(state) == self.worker_config.world_size, (
                    "State must be a list of size world_size"
                )

            local_object = [None]
            torch.distributed.scatter_object_list(
                local_object,
                state,
                src=global_src_rank,
                group=self.worker_config.data_parallel_group,
            )
            rank_state = local_object[0]

        self.restore_state_rank(rank_state)

    def can_restore_sample(self) -> bool:
        """Returns whether the dataset can be restored to a sample."""
        return self.dataset.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        """Restores a sample from a key. This is useful to debug the dataset."""
        return self.dataset.restore_sample(restore_key)

    def config(self):
        """Get the configuration, which defines the dataset. Useful in conjunction with `save_state`
        and `restore_state` to match the configuration as well."""
        return {
            "type": type(self).__qualname__,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": None if self.num_workers == 0 else self.prefetch_factor,
            "dataset": self.dataset.config(),
        }


class BasicDataLoader(DataLoader[T], Generic[T]):
    """DataLoader that supports debugging the dataset without saving capability (e.g. for val/eval)."""

    #: The worker config
    worker_config: WorkerConfig
    #: The wrapped dataset. For multiprocessing, this is a :class:`megatron.energon.SavableDatasetWrapper`
    dataset: Union[SavableDatasetWrapper[T], SavableDataset[T]]

    id: int
    _sample_idx: int = 0

    def __init__(
        self,
        dataset: SavableDataset[T],
        gc_collect_every_n_steps: int = GC_DEFAULT_EVERY_N_ITER,
        gc_freeze_at_start: bool = True,
        prefetch_factor: int = 2,
        cache_pool: Optional[CachePool] = None,
        watchdog_timeout_seconds: Optional[float] = 60,
        watchdog_initial_timeout_seconds: Optional[float] = None,
        fail_on_timeout: bool = False,
    ):
        """
        Create the dataloader supporting saving and restoring the state.

        Args:
            dataset: The dataset to load.
            gc_collect_every_n_steps: The number of steps after which the garbage collector is
                called. As we're usually handling large (but few) tensors here, and the python
                garbage collection is already full of objects just by importing, this can improve
                the memory footprint quite a lot, and may even be necessary to avoid memory
                overflow.
            gc_freeze_at_start: If true, the garbage collector is frozen at the start of the worker
                processes. This improves the garbage collection performance by a lot.
                In rare cases, this may cause issues and can be disabled. Keep enabled if you
                experience no issues.
            cache_pool: If set, the cache pool to use for the dataset.
            watchdog_timeout_seconds: The timeout in seconds. If None, the watchdog is disabled.
            watchdog_initial_timeout_seconds: The initial timeout in seconds.
            If None, the timeout is the same as watchdog_timeout_seconds.
            fail_on_timeout: If True, stops the whole process upon timeout, after printing a stack trace.
        """
        self.worker_config = dataset.worker_config

        self.id = SavableDataLoader.next_id()

        dataset = WatchdogDataset(
            dataset,
            worker_config=self.worker_config,
            timeout_seconds=watchdog_timeout_seconds,
            initial_timeout_seconds=watchdog_initial_timeout_seconds,
            fail_on_timeout=fail_on_timeout,
        )

        if gc_collect_every_n_steps > 0:
            dataset = GcDataset(
                dataset,
                worker_config=self.worker_config,
                every_n_iter=gc_collect_every_n_steps,
                freeze=gc_freeze_at_start,
            )

        dataset = SimpleSavableDatasetWrapper(
            dataset, worker_config=self.worker_config, cache_pool=cache_pool
        )

        self._worker_sample_counters = [0] * max(self.worker_config.num_workers, 1)

        kwargs = {}
        if self.worker_config.num_workers > 0:
            # These must not be specified for num_workers =0
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["multiprocessing_context"] = "fork"

        seed_per_worker = [
            self.worker_config.worker_seed(i) for i in range(self.worker_config.num_workers)
        ]

        gc.collect()  # This ensures that we don't include any old worker refs in the newly forked worker processes

        super().__init__(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.worker_config.num_workers,
            pin_memory=True,
            worker_init_fn=partial(_init_worker, seed_per_worker),
            **kwargs,
        )
        if self.worker_config.should_log(level=1):
            self.worker_config.worker_log(
                {
                    "t": "BasicDataLoader.__init__",
                    "r": self.worker_config.rank,
                    "w": None,
                    "id": self.id,
                    "config": self.config(),
                }
            )

    def __len__(self):
        # We override this, because otherwise we'll see warnings
        return self.dataset.len_rank()

    def __iter__(self):
        def _inner_generator(iterator):
            iter_idx = 0
            id = SavableDataLoader.next_id()

            if self.worker_config.should_log(level=1):
                self.worker_config.worker_log(
                    {
                        "t": "BasicDataLoader.iter",
                        "r": self.worker_config.rank,
                        "w": None,
                        "id": self.id,
                        "iter_id": id,
                    }
                )

            try:
                for worker_id, sample_idx, sample in iterator:
                    # If the next sample will be from the first worker, we can safely resume
                    if self.worker_config.should_log(level=1):
                        keys = default_get_batch_keys(sample)
                        self.worker_config.worker_log(
                            {
                                **{
                                    "t": "BasicDataLoader.yield",
                                    "r": self.worker_config.rank,
                                    "w": None,
                                    "id": self.id,
                                    "iter_id": self.id,
                                    "worker_id": worker_id,
                                    "worker_idx": sample_idx,
                                    "idx": iter_idx,
                                    "iter_idx": iter_idx,
                                    "global_idx": self._sample_idx,
                                },
                                **({} if keys is None else {"keys": keys}),
                            }
                        )
                    self._sample_idx += 1
                    iter_idx += 1
                    yield sample
            finally:
                if self.worker_config.should_log(level=1):
                    self.worker_config.worker_log(
                        {
                            "t": "BasicDataLoader.StopIteration",
                            "r": self.worker_config.rank,
                            "w": None,
                            "id": self.id,
                            "iter_id": id,
                        }
                    )

        return _inner_generator(super().__iter__())

    def config(self):
        """Get the configuration, which defines the dataset. Useful in conjunction with `save_state`
        and `restore_state` to match the configuration as well."""
        return {
            "type": type(self).__qualname__,
            "num_workers": self.worker_config.num_workers,
            "persistent_workers": self.persistent_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": None if self.num_workers == 0 else self.prefetch_factor,
            "dataset": self.dataset.config(),
        }

    def can_restore_sample(self) -> bool:
        """Returns whether the dataset can restore samples."""
        return self.dataset.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        """Restores a sample from a key. This is useful to debug the dataset."""
        return self.dataset.restore_sample(restore_key)

