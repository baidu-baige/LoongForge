# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

""" Parallel config """
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    """
    Parallel config for distributed training
    """
    tp_size: int = 1
    pp_size: int = 1
    ep_size: Optional[int] = None
    etp_size: Optional[int] = None
    vpp_size: Optional[int] = None
    encoder_tp_size: Optional[int] = None
    custom_pipeline_layers: Optional[str] = None
    decoder_first_pipeline_num_layers: Optional[int] = None
    decoder_last_pipeline_num_layers: Optional[int] = None
    moe_grouped_gemm: bool = False
    vpp_scheduler: Optional[str] = None
    tp_ranks: Optional[list[int]] = None
    pp_ranks: Optional[list[int]] = None
    ep_ranks: Optional[list[int]] = None
    etp_ranks: Optional[list[int]] = None
    safetensors: bool = True
    max_workers: int = 1
    fp8_force_no_requant: bool = False
    force_pow_2_scales: bool = False
    amax_epsilon: float = 0.0
    mtp_num_layers: int = 0
    lora_alpha: int = None
    lora_dim: int = None
    enable_full_hetero_dp: bool = False
    hf_checkpoint_device: str = "cpu"
    sub_file_tag: int = None

