# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Megatron engine helpers that upstream Megatron does not provide."""

import os
from importlib.metadata import version
from typing import Optional
from packaging.version import Version as PkgVersion

_transformers_version = None


def get_transformers_version():
    """Get transformers version (cached)."""
    import transformers

    global _transformers_version

    if _transformers_version is None:
        # prefer module __version__ to avoid importlib.metadata lookup issues in editable installs
        ver = getattr(transformers, "__version__", None)
        if ver is None:
            ver = version("transformers")
        _transformers_version = PkgVersion(ver)

    return _transformers_version


def get_config_from_file(config_file: str):
    """
    Split a full config file path into:
    - config_path (directory)
    - config_name (file name without extension)
    """
    config_file = os.path.abspath(config_file)

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    config_dir = os.path.dirname(config_file)
    file_name = os.path.basename(config_file)

    if not file_name.endswith((".yaml", ".yml")):
        raise ValueError("Config file must end with .yaml or .yml")

    # remove extension
    config_name = os.path.splitext(file_name)[0]

    return config_dir, config_name


def convert_custom_pipeline_to_layout(
    custom_pipeline_layers: Optional[str] = None,
    custom_virtual_pipeline_layers: Optional[str] = None,
    num_virtual_stages_per_pipeline_rank: Optional[int] = None,
    mtp_num_layers: Optional[int] = None,
) -> str:
    """
    Convert custom-pipeline-layers format to pipeline-model-parallel-layout format.
    
    Args:
        custom_pipeline_layers: Comma-separated string of decoder layer counts per PP stage.
                                Example: "6,5,7,9" means 4 PP stages with 6,5,7,9 decoder layers.
                                Cannot be used together with custom_virtual_pipeline_layers.
        custom_virtual_pipeline_layers: Optional. Comma-separated string specifying exact layer 
                                       counts for each VPP stage. Format: all PP's VPP0, then all PP's VPP1.
                                       Example: "3,2,3,4,3,3,4,5" for PP=4, VPP=2
                                       Cannot be used together with custom_pipeline_layers.
        num_virtual_stages: Number of virtual pipeline stages (VPP) per PP rank.
                        If None, defaults to 1 (no virtual pipeline).
        mtp_num_layers: Number of MTP layers. If None, no MTP layers are added.
                        MTP layers are placed in the last stage with loss.
    
    Returns:
        Layout string in pipeline-model-parallel-layout format.
    """
    # Handle None case: default to 1 (no virtual pipeline)
    vp_size = num_virtual_stages_per_pipeline_rank if num_virtual_stages_per_pipeline_rank is not None else 1

    splits = []
    # Determine PP size based on which parameter is provided
    if custom_virtual_pipeline_layers is not None:
        if custom_virtual_pipeline_layers.find(',') != -1:
            splits = [int(s) for s in custom_virtual_pipeline_layers.split(',') if s.strip()]
        pp_size = len(splits) // vp_size
    elif custom_pipeline_layers is not None:
        if custom_pipeline_layers.find(',') != -1:
            splits = [int(s) for s in custom_pipeline_layers.split(',') if s.strip()]
        pp_size = len(splits)
    else:
        raise ValueError(
            "One of custom_pipeline_layers or custom_virtual_pipeline_layers must be provided.")

    # Symbol mapping
    from megatron.core.transformer.enums import LayerType
    symbols = {
        LayerType.embedding: 'E',
        LayerType.decoder: 't',
        LayerType.mtp: 'm',
        LayerType.loss: 'L',
    }

    # Build layout for each PP rank and VPP rank
    # Layout order: All PP ranks' VPP0, then all PP ranks' VPP1, etc.
    layout_stages = []

    for vp_rank in range(vp_size):
        for pp_rank in range(pp_size):
            stage_layers = []

            # First stage of first PP rank: add embedding
            if pp_rank == 0 and vp_rank == 0:
                stage_layers.append(symbols[LayerType.embedding])

            # Determine number of decoder layers for this stage
            if custom_virtual_pipeline_layers is not None:
                # Directly get the layer count from custom_virtual_pipeline_layers
                num_layers_this_stage = splits[vp_rank * pp_size + pp_rank]
            else:
                # Calculate layers per VPP stage using the formula:
                # num_layers_to_build = ([q] * (vp_size - r) + [q + 1] * r)[vp_rank]
                num_layers = splits[pp_rank]
                q = num_layers // vp_size
                r = num_layers % vp_size

                if vp_rank < (vp_size - r):
                    num_layers_this_stage = q
                else:
                    num_layers_this_stage = q + 1

            # Add decoder layers for this VPP stage
            stage_layers.extend([symbols[LayerType.decoder]] * num_layers_this_stage)

            # Last stage of last PP rank: add MTP (if applicable) and loss
            if pp_rank == pp_size - 1 and vp_rank == vp_size - 1:
                if mtp_num_layers is not None and mtp_num_layers > 0:
                    stage_layers.extend(
                        [symbols[LayerType.mtp]] * mtp_num_layers)
                stage_layers.append(symbols[LayerType.loss])

            layout_stages.append(''.join(stage_layers))

    # Join all stages with '|'
    layout_str = '|'.join(layout_stages)

    return layout_str
