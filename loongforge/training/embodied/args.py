# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Embodied-only Torch arguments: trainer routing and data loading."""

from dataclasses import dataclass, field
from typing import Optional

from loongforge.training.cli_utils import parse_positive_int


@dataclass(frozen=True)
class _EmbodiedRoutingArgs:
    """Trainer selection for embodied recipes."""

    trainer_type: str = field(
        default="FinetuneTrainer",
        metadata={
            "help": "Trainer type (e.g. FinetuneTrainer, GrootN1d6Trainer); "
                    "selects the recipe, hooks and runner in loongforge/train.py."
        },
    )


@dataclass(frozen=True)
class _EmbodiedDataArgs:
    """Data loading control (cross-model; per-model processing lives in DataConfig)."""

    dataset_format: str = field(
        default="lerobot_datasets",
        metadata={
            "help": "Dataset backend to use "
                    "(e.g. lerobot_datasets, hdf5_datasets, dummy_datasets)."
        },
    )

    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Filesystem path or repo id of the dataset to train on."},
    )

    dataset_strategy: Optional[str] = field(
        default="default",
        metadata={
            "help": "Under --dataset-format lerobot_datasets: the model-specific "
                    "dataset build strategy ('default', 'vlda', 'fastwam', "
                    "'lingbot_va', 'gr00t_n1_7', 'cosmos3_droid', "
                    "'dreamzero', or 'wall_oss_0_5'); unknown values raise "
                    "ValueError."
        },
    )

    split: str = field(
        default="train",
        metadata={
            "help": "Dataset split to load (RLDS), e.g. 'train' or 'train[:95%%]'."
        },
    )

    data_root_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Root directory containing the datasets referenced by "
                    "--dataset-mix."
        },
    )

    robot_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "Robot embodiment type (e.g. libero_franka); selects "
                    "action/state layout."
        },
    )

    task_name: str = field(
        default="perform the task",
        metadata={
            "help": "Language instruction used as the prompt when the dataset has "
                    "none (HDF5)."
        },
    )

    lerobotdataset_version: str = field(
        default="v3.0",
        metadata={
            "choices": ["v2.0", "v2.1", "v3.0"],
            "help": "On-disk LeRobot dataset format version to parse.",
        },
    )

    video_backend: str = field(
        default="torchcodec",
        metadata={
            "choices": ["torchcodec", "decord", "opencv", "pyav", "torchvision_av"],
            "help": "Backend used to decode episode videos into frames.",
        },
    )

    streaming: bool = field(
        default=False,
        metadata={
            "help": "Use a streaming/iterable dataset instead of map-style random "
                    "access (lower memory, no global shuffle)."
        },
    )

    per_device_batch_size: int = field(
        default=4,
        metadata={"help": "Micro-batch size processed per GPU per forward pass."},
    )

    num_workers: int = field(
        default=4,
        metadata={"help": "Number of DataLoader worker processes per rank."})

    dataloader_prefetch_factor: int = field(
        default=2,
        metadata={
            "cli_type": parse_positive_int,
            "help": "Number of batches prefetched by each DataLoader worker.",
        },
    )

    dataloader_seed_workers: bool = field(
        default=False,
        metadata={"help": "Set DataLoader worker_init_fn and generator from --seed. "
                          "Default leaves both unset for baseline precision comparison."})

    dataloader_multiprocessing_context: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["fork", "spawn", "forkserver"],
            "help": "Multiprocessing start method for DataLoader workers.",
        },
    )

    sampler_shuffle: bool = field(
        default=True,
        metadata={"help": "Shuffle sample order in the sampler."},
    )

    distributed_sampler_mode: str = field(
        default="cyclic",
        metadata={
            "choices": ["cyclic", "block"],
            "help": "How the distributed sampler partitions indices across ranks: "
                    "'cyclic' (round-robin) or 'block' (contiguous shards).",
        },
    )

    sampler_epoch_mode: str = field(
        default="advance",
        metadata={
            "choices": ["advance", "fixed"],
            "help": "Whether the sampler's epoch advances between epochs. "
                    "'advance' (default) uses seed+epoch, so every epoch reshuffles. "
                    "'fixed' pins the epoch to 0, so every epoch reuses the SAME "
                    "permutation -- this reproduces standalone FastWAM, whose "
                    "ResumableEpochSampler epoch is never advanced. Only used by "
                    "model types that register their own sampler builder (currently "
                    "fastwam); see data/embodied/fastwam/samplers.py.",
        },
    )

    batch_drop_last: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, drop the last incomplete batch so every rank sees the same "
                "number of full-size batches. Applied to both sampler and DataLoader. "
                "Default False preserves all samples."
            ),
        },
    )

    num_samples: int = field(
        default=100,
        metadata={
            "help": "Number of synthetic samples to generate. Only effective "
                    "when --dataset-format=dummy_datasets."
        },
    )
