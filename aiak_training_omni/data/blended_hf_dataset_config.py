""" Configuration object for blended huggingface datasets """

from typing import List, Optional
from dataclasses import dataclass

from megatron.core.datasets.utils import Split
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig


@dataclass
class BlendedHuggingFaceDatasetConfig(BlendedMegatronDatasetConfig):
    """Configuration object for Megatron Core SFT datasets"""

    dataset: Optional[List[str]] = None
    """The dataset consisting of a list of dataset name. Not to be used with 'dataset_per_split'."""

    dataset_per_split: Optional[List[Optional[List[str]]]] = None
    """A set of dataset, one for each split distribution"""

    streaming: bool = False
    """Whether to stream the dataset or not"""

    streaming_buffer_size: int = 16384
    """The size of the buffer to randomly sample examples from in dataset streaming"""

    mix_strategy: str = "concat"
    """The strategy to mix the dataset, options: concat, interleave_under, interleave_over"""

    num_preprocess_workers: Optional[int] = None
    """The number of workers to preprocess the dataset"""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.tokenizer is not None
        
        assert self.dataset_config_file is not None, "dataset_config_file must be provided"

        if self.blend_per_split is not None and any(self.blend_per_split):
            assert self.dataset_per_split is not None, \
                "dataset_per_split must be provided since blend_per_split is defined"

            assert len(self.dataset_per_split) == len(self.blend_per_split), \
                f"datset_per_split must contain {len(self.blend_per_split)} blends"

            for split in Split:
                if self.blend_per_split[split.value] is not None:
                    assert self.dataset_per_split[split.value] is not None, \
                        f"dataset_per_split must be provided for {split.name} split"

                    assert len(self.blend_per_split[split.value][0]) == len(self.dataset_per_split[split.value]), \
                        (f"dataset_per_split must contain {len(self.blend_per_split[split.value][0])} "
                        f"datasets for {split.name} split")
        else:
            assert self.split is not None, "split must be provided in absence of blend_per_split"
            assert self.dataset is not None, "dataset must be provided in absence of blend_per_split"
            assert len(self.dataset) == len(self.blend[0]), f"dataset must contain {len(self.blend[0])} blends"
