"""blended huggingface dataset builder"""

import logging
from typing import Callable, List, Optional, Union, Tuple

import torch

from datasets import concatenate_datasets, interleave_datasets, Dataset, IterableDataset

from megatron.core.utils import log_single_rank
from megatron.core.datasets.utils import Split, normalize
from megatron.core.parallel_state import get_virtual_pipeline_model_parallel_rank

from .blended_hf_dataset_config import BlendedHuggingFaceDatasetConfig
from .hf_dataset import HuggingFaceDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BlendedHuggingFaceDatasetBuilder(object):
    """Builder class for the SFT Dataset. 

    Args:
        cls (HuggingFaceDataset): The class to instantiate, wich build on huggingface. i.e. SFTDataset

        sizes (List[Optional[int]]): The minimum total number of samples to draw, or None, per split

        is_built_on_rank (Callable): A callable which returns True if the dataset should be built on the current rank
            and False otherwise. It should be Megatron Core parallelism aware i.e. global rank, local group rank,
            and virtual rank may inform its return value.

        config (BlendedHuggingFaceDatasetConfig): The config object which informs dataset creation
    """

    def __init__(
        self,
        cls: HuggingFaceDataset,
        sizes: List[int],
        is_built_on_rank: Callable,
        config: BlendedHuggingFaceDatasetConfig,
    ):
        self.cls = cls
        self.sizes = sizes # Note: the sizes not use now
        self.is_built_on_rank = is_built_on_rank
        self.config = config

        log_single_rank(
            logger,
            logging.INFO,
            f"Building huggingface dataset splits with cls={cls.__name__}, and config={self.config}",
        )

        if not self.config.mock:
            for split in Split:
                size_is_none = self.sizes[split.value] is None
                if self.config.blend_per_split is None:
                    weights_are_none = self.config.blend[1] is None
                else:
                    if self.config.blend_per_split[split.value] is None:
                        continue
                    weights_are_none = self.config.blend_per_split[split.value][1] is None
                if size_is_none:
                    assert (
                        weights_are_none
                    ), f"size_is_none => weights_are_none fails for {split.name} split"

        if torch.distributed.is_initialized():
            gb_rank = torch.distributed.get_rank()
            vp_rank = get_virtual_pipeline_model_parallel_rank()
            if gb_rank == 0 and (vp_rank == 0 or vp_rank is None):
                assert (
                    self.is_built_on_rank()
                ), "is_built_on_rank must return True when global rank = 0 and vp rank = 0"

    def build(self) -> List[Optional[Union[Dataset, IterableDataset]]]:
        """Build all dataset splits according to the provided blend(s)
        
        Returns:
            List[Optional[Union[Dataset, IterableDataset]]]: A list containing a dataset instance (or None) per split
        """
        assert not self.config.mock, "Not support mock dataset for SFT!"
        
        if self.config.blend:
            return self._build_splits_from_same_blend()

        return self._build_splits_from_separate_blend()

    def _build_splits_from_same_blend(self) -> List[Optional[Union[Dataset, IterableDataset]]]:
        """
        Each split comes from the same distribution
        
        Returns:
            List[Optional[Union[Dataset, IterableDataset]]]: A list containing a dataset instance (or None)
        """
        prefixes, weights = self.config.blend
        dataset_names = self.config.dataset

        if weights is not None:
            weights = normalize(weights)

        split = self.config.split_matrix

        # blend consists of a single prefix
        if len(prefixes) == 1:
            return self._build_huggingface_dataset_splits(prefixes[0], dataset_names[0], split)

        # blend consists of multiple prefixes
        huggingface_datasets = [[] for _ in range(len(Split))]
        all_datasets_split = []
        for i in range(len(prefixes)):
           all_datasets_split.append(self._build_huggingface_dataset_splits(prefixes[i], dataset_names[i], split))
        
        for dataset_split in all_datasets_split:
            for j in range(len(dataset_split)):
                huggingface_datasets[j].append(dataset_split[j])

        # run mix strategy
        blended_datasets = [None] * len(Split)
        for i in range(len(Split)):
            if split[i] is not None:
                blended_datasets[i] = self._build_blend_huggingface_dataset_splits(huggingface_datasets[i], weights)

        return blended_datasets

    def _build_splits_from_separate_blend(self) -> List[Optional[Union[Dataset, IterableDataset]]]:
        """
        Each split comes from a separate distribution
        
        Returns:
            List[Optional[Union[Dataset, IterableDataset]]]: A list containing a dataset instance (or None)
        """
        blended_datasets = [None] * len(Split)
        for i in range(len(Split)):            
            split_spoof = [None] * len(Split)
            split_spoof[i] = (0.0, 1.0)
            
            blend = self.config.blend_per_split[i]
            dataset_names = self.config.dataset_per_split[i]

            if blend is None:
                continue
            
            prefixes, weights = blend
            if weights is not None:
                weights = normalize(weights)

            # Blend consists of a single prefix
            if len(prefixes) == 1:
                blended_datasets[i] = self._build_huggingface_dataset_splits(
                    prefixes[0], dataset_names[0], split_spoof
                )[i]
                
                continue

            # Blend consists of multiple prefixes
            all_datasets = []
            for p in range(len(prefixes)):
                all_datasets.append(
                    self._build_huggingface_dataset_splits(
                        prefixes[p], dataset_names[p], split_spoof
                    )[i]
                )

            blended_datasets[i] = self._build_blend_huggingface_dataset_splits(all_datasets, weights)

        return blended_datasets

    def _build_huggingface_dataset_splits(
        self,
        dataset_path: str,
        dataset_name: str,
        split: Optional[List[Tuple[float, float]]],
    ) -> List[Optional[Union[Dataset, IterableDataset]]]:
        """Build each Dataset split
        
        Args:
            dataset_path (str): The path to the dataset
            dataset_name (str): The name of the dataset
            split (Optional[List[Tuple[float, float]]]): The split of the dataset
        
        Returns:
            List[Optional[Union[Dataset, IterableDataset]]]: A list containing a dataset instance (or None)
        """
        split_datasets = [None] * len(Split)

        # only build on the needed rank
        if torch.distributed.is_initialized():
            if not self.is_built_on_rank():
                return split_datasets

        # build low-level dataset
        hf_dataset = self.cls(dataset_name, dataset_path, self.config)
        split_datasets = hf_dataset.split(split)
        return split_datasets

    def _build_blend_huggingface_dataset_splits(
        self,
        datasets: List[Optional[Union[Dataset, IterableDataset]]],
        weights: Optional[List[float]],
    ) -> Optional[Union[Dataset, IterableDataset]]:
        """Build each blend Dataset split
        
        Args:
            datasets (List[Optional[Union[Dataset, IterableDataset]]]): The list of datasets
            weights (Optional[List[float]]): The list of weights
        
        Returns:
            Optional[Union[Dataset, IterableDataset]]: A mixed dataset instance (or None)
            
        """
        if torch.distributed.is_initialized():
            if not self.is_built_on_rank():
                return None
            
        if len(datasets) == 1:
            return datasets[0]

        if self.config.mix_strategy == "concat":
            if weights is not None:
                log_single_rank(
                    logger,
                    logging.WARN,
                    "Building sft dataset with concat mode that will ignore the weights.",
                )
            return concatenate_datasets(datasets)

        if self.config.mix_strategy == "interleave_under":
            return interleave_datasets(
                datasets=datasets,
                probabilities=weights,
                seed=self.config.random_seed,
                stopping_strategy="first_exhausted",
            )
        
        if self.config.mix_strategy == "interleave_over":
            return interleave_datasets(
                datasets=datasets,
                probabilities=weights,
                seed=self.config.random_seed,
                stopping_strategy="all_exhausted",
            )

        raise ValueError(f"Unsupported mix strategy: {self.config.mix_strategy}")
