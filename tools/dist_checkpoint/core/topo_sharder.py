# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
TopoSharder module: Initialize parallel_state and provide rank topology

This module initializes Megatron-Core parallel_state with ParallelConfig
and provides a unified interface to query rank information.
"""
import os
from typing import List, Tuple, Optional, Dict
import torch
import torch.distributed as dist

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

from tools.dist_checkpoint.config.parallel_config import ParallelConfig

# Type definitions
RankTopoTuple = Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]
RankTopoList = List[RankTopoTuple]


class TopoSharder:
    """
    Initialize parallel_state and provide rank topology information

    Responsibilities:
    - Initialize Megatron-Core parallel_state with ParallelConfig
    - Read parallel ranks from initialized parallel_state
    - Provide unified query interface for (tp_rank, pp_rank, ep_rank, etp_rank)
    - Build complete rank topology list for all ranks in world
    """

    def __init__(self, parallel_config: ParallelConfig):
        """
        Initialize TopoSharder with ParallelConfig

        If parallel_state is not yet initialized, this will call
        parallel_state.initialize_model_parallel() internally.

        Args:
            parallel_config: ParallelConfig object containing parallel configuration

        Raises:
            ImportError: If megatron.core is not available
            RuntimeError: If torch.distributed is not initialized
        """
        if not MEGATRON_AVAILABLE:
            raise ImportError(
                "megatron.core is not available. "
                "Please install Megatron-LM to use TopoSharder."
            )

        # Initialize model parallel if not already initialized
        if not parallel_state.model_parallel_is_initialized():
            # Only pass ep_size and etp_size if they are explicitly set (not None)
            init_kwargs = {
                'tensor_model_parallel_size': parallel_config.tp_size,
                'pipeline_model_parallel_size': parallel_config.pp_size,
            }
            if parallel_config.ep_size is not None:
                init_kwargs['expert_model_parallel_size'] = parallel_config.ep_size
            if parallel_config.etp_size is not None:
                init_kwargs['expert_tensor_parallel_size'] = parallel_config.etp_size

            parallel_state.initialize_model_parallel(**init_kwargs)

        # Read world size from torch.distributed
        self.world_size = dist.get_world_size()

        # Read parallel sizes
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        # Read expert parallel sizes from parallel_config (not from parallel_state to avoid auto-calculated values)
        # Use the values from parallel_config if explicitly set, otherwise try to read from parallel_state
        if parallel_config.ep_size is not None:
            self.ep_size = parallel_state.get_expert_model_parallel_world_size()
        else:
            self.ep_size = 1

        if parallel_config.etp_size is not None:
            self.etp_size = parallel_state.get_expert_tensor_parallel_world_size()
        else:
            self.etp_size = None

        # Calculate derived sizes
        self.model_parallel_size = self.tp_size * self.pp_size
        self.data_parallel_size = self.world_size // self.model_parallel_size

        # Cache for rank topology list
        self._rank_topo_list: Optional[RankTopoList] = None

        # Print initialization info (only rank 0 will print)
        print_rank_0(f"[TopoSharder] Initialized from parallel_state:")
        print_rank_0(f"  World size: {self.world_size}")
        print_rank_0(f"  TP size: {self.tp_size}, PP size: {self.pp_size}")
        print_rank_0(f"  EP size: {self.ep_size}, ETP size: {self.etp_size}")
        print_rank_0(f"  DP size: {self.data_parallel_size}")

    def get_current_rank_coordinates(self) -> RankTopoTuple:
        """
        Get coordinates for current rank

        Returns:
            (tp_rank, pp_rank, ep_rank, etp_rank, dp_rank) for current process
        """
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        dp_rank = parallel_state.get_data_parallel_rank()

        # Get expert parallel ranks if available
        if self.ep_size > 1:
            ep_rank = parallel_state.get_expert_model_parallel_rank()
        else:
            ep_rank = None

        if self.etp_size is not None:
            etp_rank = parallel_state.get_expert_tensor_parallel_rank()
        else:
            etp_rank = None

        return (tp_rank, pp_rank, ep_rank, etp_rank, dp_rank)

    def build_rank_mapping_table(self) -> dict:
        """
        Build a mapping table from (pp, tp, ep) coordinates to list of global ranks.

        All ranks report their coordinates using gather_object, rank 0 builds lookup table.
        When DP > 1, multiple ranks share the same (pp, tp, ep) coordinates, so value is a list.

        Returns:
            dict: {(pp_rank, tp_rank, ep_rank): [global_rank, ...], ...}
                   Only valid on rank 0. Other ranks receive empty dict.
        """
        rank = dist.get_rank()

        # Get current rank's coordinates
        tp_rank, pp_rank, ep_rank, etp_rank, dp_rank = self.get_current_rank_coordinates()

        # Prepare coordinate data: (global_rank, pp, tp, ep)
        coord_tuple = (rank, pp_rank, tp_rank, ep_rank)

        # Gather all coordinates to rank 0
        gather_list = [None] * self.world_size if rank == 0 else None
        dist.gather_object(coord_tuple, gather_list if rank == 0 else None, dst=0)

        if rank == 0:
            # Build lookup table: (pp, tp, ep) -> list of global_ranks
            lookup_table = {}
            for global_rank, pp, tp, ep in gather_list:
                key = (pp, tp, ep)
                if key not in lookup_table:
                    lookup_table[key] = []
                lookup_table[key].append(global_rank)
            return lookup_table
        else:
            return {}


