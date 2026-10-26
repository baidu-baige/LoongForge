# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Build the VLM data config from the training arguments.

The data layer receives this frozen config instead of the whole argument
namespace, so it never depends on the backend's argument schema.
"""

from loongforge.data.vlm.dataset_config import VLMDataConfig


def vlm_data_config(args) -> VLMDataConfig:
    """Collect the Energon/encoder data recipe from the training arguments."""
    encoder_fields = {
        name: getattr(args, name, None)
        for name in VLMDataConfig.__dataclass_fields__
        if name not in {"data_path", "valid_data_path", "micro_batch_size", "num_workers",
                        "packing_buffer_size", "data_shuffle_buffer_size",
                        "data_max_samples_per_sequence"}
    }
    return VLMDataConfig(
        data_path=args.data_path,
        valid_data_path=getattr(args, "valid_data_path", None),
        micro_batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        packing_buffer_size=getattr(args, "packing_buffer_size", 0),
        data_shuffle_buffer_size=getattr(args, "data_shuffle_buffer_size", 0),
        data_max_samples_per_sequence=getattr(args, "data_max_samples_per_sequence", 0),
        **encoder_fields,
    )
