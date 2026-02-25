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
        self.tp_rank, self.pp_rank, self.ep_rank, self.etp_rank = (
            topo_sharder.get_current_rank_coordinates()
        )
        self.tp_size = topo_sharder.tp_size
        self.pp_size = topo_sharder.pp_size
        self.ep_size = topo_sharder.ep_size
        self.etp_size = topo_sharder.etp_size

        # Get global rank
        self.rank = dist.get_rank()

        # Log output (current rank only)
        print(f"[TPGather] Rank {self.rank} initialized:")
        print(f"  Coordinates: tp={self.tp_rank}, pp={self.pp_rank}, "
              f"ep={self.ep_rank}, etp={self.etp_rank}")
        print(f"  TP size: {self.tp_size}, PP size: {self.pp_size}")

    def gather_state_dicts(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Gather state_dicts from all TP ranks to TP rank 0 using dist.gather

        Assumes all ranks have the same model architecture and thus identical keys.
        Uses tensor communication (no pickle serialization overhead).

        Args:
            state_dict: Complete state_dict of current rank (on GPU)

        Returns:
            List[Dict]: TP rank 0 returns list of state_dicts (index corresponds to tp_rank)
                       Other ranks return None
                       List format: [state_dict_tp0, state_dict_tp1, ...]

        Notes:
            - Only TP rank 0 receives and stores the complete list
            - Other TP ranks send data and return None
            - Uses sorted keys to ensure consistent ordering across all ranks
            - Filters out non-tensor values (metadata, etc.)
            - No serialization involved - pure tensor communication via dist.gather
        """
        tp_group = parallel_state.get_tensor_model_parallel_group()

        # Get the global rank of TP rank 0 for dist.gather
        tp_rank_0_global = parallel_state.get_tensor_model_parallel_src_rank()

        # Step 1: Sort keys and filter tensor-only keys
        # (all ranks have identical model architecture)
        local_keys = sorted([k for k, v in state_dict.items() if isinstance(v, torch.Tensor)])
        print(f"[TPGather] Rank {self.rank} (tp={self.tp_rank}) gathering {len(local_keys)} tensors")

        # Step 2: Gather each tensor from all TP ranks
        if self.tp_rank == 0:
            # TP rank 0: prepare receive buffers for each key
            gathered_dict = {}
            for key in local_keys:
                # Create a list of empty tensors to receive data from all ranks
                gathered_dict[key] = [torch.zeros_like(state_dict[key]) for _ in range(self.tp_size)]

            # Gather each tensor one by one
            for idx, key in enumerate(local_keys):
                dist.gather(state_dict[key], gathered_dict[key], dst=tp_rank_0_global, group=tp_group)
                if (idx + 1) % max(1, len(local_keys) // 10) == 0:
                    print(f"[TPGather] Rank {self.rank} gathered {idx + 1}/{len(local_keys)} tensors")

            print(f"[TPGather] Rank {self.rank} (tp=0) gathered all {len(local_keys)} tensors")

            # Step 3: Reconstruct as list of state_dicts
            result = []
            for tp_idx in range(self.tp_size):
                state_dict_tp = {key: gathered_dict[key][tp_idx] for key in local_keys}
                result.append(state_dict_tp)

            return result
        else:
            # Other TP ranks: only send data, no receiving
            for idx, key in enumerate(local_keys):
                dist.gather(state_dict[key], None, dst=tp_rank_0_global, group=tp_group)
                if (idx + 1) % max(1, len(local_keys) // 10) == 0:
                    print(f"[TPGather] Rank {self.rank} sent {idx + 1}/{len(local_keys)} tensors")

            print(f"[TPGather] Rank {self.rank} (tp={self.tp_rank}) sent all {len(local_keys)} tensors")
            return None
