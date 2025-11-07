"""LengthPoolSortDataset"""

from typing import Callable, Iterator, List, TypeVar, Optional
import random
from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample")


class LengthPoolSortDataset(SavableDataset[T_sample]):
    """
    局部池化长度排序:
      - 累积 pool_size 个样本，按 key_fn(sample) 排序后依次输出
      - 剩余不足 pool_size 的尾部再排序输出
    """

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        pool_size: int,
        key_fn: Callable[[T_sample], int],
        ascending: bool,
        worker_config: WorkerConfig,
        tail_shuffle: bool = True,
        shuffle_seed: Optional[int] = None,  # 若 None 使用 worker_config.global_seed
    ):
        super().__init__(worker_config=worker_config)
        assert pool_size > 0
        self.dataset = dataset
        self.pool_size = pool_size
        self.key_fn = key_fn
        self.ascending = ascending
        self.tail_shuffle = tail_shuffle
        base_seed = (
            shuffle_seed
            if shuffle_seed is not None
            else getattr(worker_config, "global_seed", 1234)
        )
        # 独立 RNG, 不污染全局
        self._rng = random.Random(base_seed)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        pool: List[T_sample] = []
        for batch_idx, sample in enumerate(self.dataset):
            pool.append(sample)
            if len(pool) >= self.pool_size:
                pool.sort(key=self.key_fn, reverse=not self.ascending)
                shuffle_seed = 42 + batch_idx
                random.Random(shuffle_seed).shuffle(pool)
                for s in pool:
                    yield s
                pool.clear()
        if pool:
            pool.sort(key=self.key_fn, reverse=not self.ascending)
            if self.tail_shuffle:
                # 仅对尾池可复现打乱
                self._rng.shuffle(pool)
            for s in pool:
                yield s
            pool.clear()

    # ---- 抽象方法实现委托 ----
    def worker_has_samples(self) -> bool:
        """worker_has_samples"""
        return self.dataset.worker_has_samples()

    def can_restore_sample(self) -> bool:
        """can_restore_sample"""
        return self.dataset.can_restore_sample()

    def assert_can_restore(self) -> None:
        """assert_can_restore"""
        self.dataset.assert_can_restore()

    def restore_sample(self, index):
        """restore_sample"""
        return self.dataset.restore_sample(index)

    def save_state(self):
        """save_state"""
        return self.dataset.save_state()

    def merge_states(self, states):
        """merge_states"""
        return self.dataset.merge_states(states)

    def restore_state(self, state):
        """restore_state"""
        self.dataset.restore_state(state)

    def config(self):
        """config"""
        return {
            "type": type(self).__qualname__,
            "pool_size": self.pool_size,
            "ascending": self.ascending,
            "tail_shuffle": self.tail_shuffle,
            "dataset": self.dataset.config(),
        }
