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
        Gather state_dicts from all TP ranks to TP rank 0

        Args:
            state_dict: Complete state_dict of current rank

        Returns:
            List[Dict]: TP rank 0 returns list (index corresponds to tp_rank)
                       Other ranks return None
                       List format: [state_dict_tp0, state_dict_tp1, ...]

        Notes:
            - Only TP rank 0 receives and returns the complete list
            - Other TP ranks only send data and return None
            - List index i corresponds to state_dict of tp_rank=i
            - No concatenation is performed; PPDistSaver handles that
        """
        # Get TP communication group from parallel_state
        tp_group = parallel_state.get_tensor_model_parallel_group()

        # Get the global rank of tp_rank=0 in my TP group
        # This is needed because gather_object's dst parameter expects global rank within the group
        tp_rank_0_global = parallel_state.get_tensor_model_parallel_src_rank()

        # Prepare gather list based on tp_rank
        if self.tp_rank == 0:
            gather_list = [None] * self.tp_size
            print(f"[TPGather] Rank {self.rank} (tp=0) gathering state_dicts "
                  f"from {self.tp_size} TP ranks...")
        else:
            gather_list = None
            print(f"[TPGather] Rank {self.rank} (tp={self.tp_rank}) "
                  f"sending state_dict to tp=0 (global rank {tp_rank_0_global})...")

        # Use gather_object: dst is the global rank of tp_rank=0 in this TP group
        dist.gather_object(
            obj=state_dict,
            object_gather_list=gather_list,
            dst=tp_rank_0_global,
            group=tp_group
        )

        # Return result and log
        if self.tp_rank == 0:
            print(f"[TPGather] Rank {self.rank} (tp=0) collected "
                  f"{len(gather_list)} state_dicts")
            # Validate list completeness
            for i, sd in enumerate(gather_list):
                if sd is None:
                    print(f"[TPGather] WARNING: state_dict at index {i} is None!")
        else:
            print(f"[TPGather] Rank {self.rank} (tp={self.tp_rank}) "
                  f"send complete")

        return gather_list  # TP rank 0 returns list, others return None


def test():
    """
    Test TPGather functionality

    Test scenario:
    - TP=4, PP=2, world_size=8
    - Each rank creates a mock state_dict containing its rank information
    - Verify that TP rank 0 correctly collects all state_dicts

    Run with:
        torchrun --nproc_per_node=8 tools/dist_checkpoint/core/tp_gather.py
    """
    print("\n" + "="*80)
    print("TPGather Test Starting...")
    print("="*80 + "\n")

    # Step 1: Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set GPU device for each rank to avoid "Duplicate GPU detected" error
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"[Test] Rank {rank}/{world_size} initialized on GPU {local_rank}")
    else:
        print(f"[Test] Rank {rank}/{world_size} initialized on CPU")

    # Step 2: Create parallel configuration
    from tools.dist_checkpoint.config.parallel_config import ParallelConfig

    parallel_config = ParallelConfig(
        tp_size=4,
        pp_size=2,
        ep_size=1,  # Use 1 instead of None for expert parallel
        etp_size=None,
        vpp_size=None,
        custom_pipeline_layers=None
    )

    # Step 3: Initialize TopoSharder
    print(f"[Test] Rank {rank} initializing TopoSharder...")
    topo_sharder = TopoSharder(parallel_config)

    # Step 4: Create TPGather (pass topo_sharder)
    print(f"[Test] Rank {rank} initializing TPGather...")
    tp_gather = TPGather(topo_sharder)

    # Step 5: Create mock state_dict
    # Each rank's state_dict contains its rank information for verification
    state_dict = {
        "layer.weight": torch.tensor([float(rank * 100 + tp_gather.tp_rank * 10)]),
        "layer.bias": torch.tensor([float(rank)]),
        "metadata": {
            "global_rank": rank,
            "tp_rank": tp_gather.tp_rank,
            "pp_rank": tp_gather.pp_rank,
        }
    }

    print(f"[Test] Rank {rank} created state_dict: {state_dict['metadata']}")

    # Step 6: Execute gather
    print(f"[Test] Rank {rank} calling gather_state_dicts...")
    gathered = tp_gather.gather_state_dicts(state_dict)

    # Step 7: Verify results
    if gathered is not None:
        print(f"\n[Test] Rank {rank} (tp=0) received gathered results:")
        print(f"  Number of state_dicts: {len(gathered)}")

        # Verify list length
        assert len(gathered) == tp_gather.tp_size, \
            f"Expected {tp_gather.tp_size} state_dicts, got {len(gathered)}"

        # Verify each state_dict
        for i, sd in enumerate(gathered):
            assert sd is not None, f"state_dict at index {i} is None"
            assert "metadata" in sd, f"state_dict at index {i} missing metadata"

            metadata = sd["metadata"]
            print(f"  [{i}] tp_rank={metadata['tp_rank']}, "
                  f"pp_rank={metadata['pp_rank']}, "
                  f"global_rank={metadata['global_rank']}")

            # Verify tp_rank order
            assert metadata["tp_rank"] == i, \
                f"state_dict at index {i} has wrong tp_rank: {metadata['tp_rank']}"

            # Verify pp_rank consistency
            assert metadata["pp_rank"] == tp_gather.pp_rank, \
                f"state_dict at index {i} has wrong pp_rank: {metadata['pp_rank']}"

        print(f"[Test] Rank {rank} (tp=0) ✅ All validations passed!")

    else:
        print(f"[Test] Rank {rank} (tp={tp_gather.tp_rank}) returned None as expected")

    # Step 8: Synchronize all ranks
    dist.barrier()

    if rank == 0:
        print("\n" + "="*80)
        print("✅ TPGather Test Completed Successfully!")
        print("="*80 + "\n")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    test()
