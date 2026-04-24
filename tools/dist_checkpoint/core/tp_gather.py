# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
TPGather module: Gather state_dicts from all TP ranks to TP rank 0

This module collects state_dicts from all tensor parallel ranks within
the same pipeline stage and returns them as a list to TP rank 0.
"""
import os
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in os.sys.path:
    import sys
    sys.path.insert(0, project_root)

try:
    from megatron.core import parallel_state
    from megatron.training import print_rank_0
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False
    parallel_state = None
    print_rank_0 = None

from tools.dist_checkpoint.core.topo_sharder import TopoSharder


class TPGather:
    """
    TP group state_dict gatherer

    Responsibilities:
    1. Use parallel_state to get TP communication group
    2. Gather state_dicts from all TP ranks to TP rank 0
    3. Return list ordered by tp_rank to TP rank 0
    """

    def __init__(self, topo_sharder: TopoSharder):
        """
        Initialize TPGather

        Args:
            topo_sharder: Topology sharder that provides parallel information

        Raises:
            ImportError: If megatron.core is not available
            RuntimeError: If parallel_state is not initialized

        Notes:
            - Get all parallel information from topo_sharder
            - parallel_state must be initialized via TopoSharder first
        """
        if not MEGATRON_AVAILABLE:
            raise ImportError(
                "megatron.core is not available. "
                "Please install Megatron-LM to use TPGather."
            )

        if not parallel_state.model_parallel_is_initialized():
            raise RuntimeError(
                "parallel_state is not initialized. "
                "Please initialize TopoSharder before creating TPGather."
            )

        self.topo_sharder = topo_sharder

        # Get parallel information from topo_sharder
        self.tp_rank, self.pp_rank, self.ep_rank, self.etp_rank, self.dp_rank = (
            topo_sharder.get_current_rank_coordinates()
        )
        self.tp_size = topo_sharder.tp_size
        self.pp_size = topo_sharder.pp_size
        self.ep_size = topo_sharder.ep_size
        self.etp_size = topo_sharder.etp_size

        # Get global rank
        self.rank = dist.get_rank()

    def gather_state_dicts(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Gather state_dicts from all TP ranks to TP rank 0 using NCCL backend.

        Uses per-tensor gather with immediate CPU offload to minimize GPU memory usage.

        Args:
            state_dict: Complete state_dict of current rank

        Returns:
            List[Dict]: TP rank 0 returns list of state_dicts (index corresponds to tp_rank)
                       Other ranks return None

        Notes:
            - Uses NCCL dist.gather for fast GPU communication
            - Immediately moves gathered tensors to CPU to release GPU memory
            - Peak GPU memory = tp_size * largest_single_tensor (not entire model)
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()
        tp_rank_0_global = parallel_state.get_tensor_model_parallel_src_rank()

        # Get sorted keys for consistent ordering
        local_keys = sorted([k for k, v in state_dict.items() if isinstance(v, torch.Tensor)])

        # Prepare result structure for tp_rank 0
        result = None
        if self.tp_rank == 0:
            result = [{} for _ in range(self.tp_size)]

        # Process tensors one by one to minimize peak GPU memory
        for key in local_keys:
            tensor = state_dict[key]

            if self.tp_rank == 0:
                # Allocate GPU buffer for gathering this tensor from all ranks
                gathered_gpu = [torch.empty_like(tensor) for _ in range(self.tp_size)]

                # Gather using NCCL
                dist.gather(tensor, gathered_gpu, dst=tp_rank_0_global, group=tp_group)

                # Immediately move to CPU and release GPU memory
                for tp_idx in range(self.tp_size):
                    result[tp_idx][key] = gathered_gpu[tp_idx].cpu()

                # Explicitly delete GPU buffer and free memory
                del gathered_gpu
                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()
            else:
                # Non-zero ranks just send their tensor
                dist.gather(tensor, None, dst=tp_rank_0_global, group=tp_group)

        # Handle non-tensor values (copy to all for rank 0)
        if self.tp_rank == 0 and result is not None:
            for key, value in state_dict.items():
                if key not in local_keys:
                    for tp_idx in range(self.tp_size):
                        result[tp_idx][key] = value

        # Final GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
