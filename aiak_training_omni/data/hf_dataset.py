"""huggingface base dataset"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union, List, Tuple

from .blended_hf_dataset_config import BlendedHuggingFaceDatasetConfig

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset


class HuggingFaceDataset(ABC):
    """The highest level wrapper class for huggingface dataset.

    Assumes that methods such as __len__, __getitem__, and __iter__ will not be overridden.  

    Args:
        dataset_name (Optional[str]): The name of the dataset, for bookkeeping
        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping
        config (BlendedHuggingFaceDatasetConfig): The config
    """

    def __init__(
        self,
        dataset_name: Optional[str],
        dataset_path: Optional[str],
        config: BlendedHuggingFaceDatasetConfig,
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.config = config
        
        # TODO: add unique identifiers for dataset description?

    @abstractmethod
    def split(self, split: Optional[List[Tuple[float, float]]]) -> List[Optional[Union["Dataset", "IterableDataset"]]]:
        """split dataset"""
        raise NotImplementedError
