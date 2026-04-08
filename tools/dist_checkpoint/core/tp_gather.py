# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""
TPGather module: Gather state_dicts from all TP ranks to TP rank 0

This module collects state_dicts from all tensor parallel ranks within
the same pipeline stage and returns them as a list to TP rank 0.
"""
import os
from typing import Dict, List, Optional

from tools.convert_checkpoint.common.common_checkpoint import EXTRA_DATA, LAYER_PREFIX, MOE_EXPERT, MOE_GROUPED_GEMM_EXPERT, MTP_LAYER_PREFIX, MTP_NAME_PREFIX_FOR_LAYER
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

    def __init__(self, topo_sharder: TopoSharder, c_config, args, c_vision_patch_config=None):
        """
        Initialize TPGather

        Args:
            topo_sharder: Topology sharder that provides parallel information
            c_config: Common configuration
            args: Arguments containing model parameters
            c_vision_patch_config: Vision patch configuration (optional)

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
        self.dense_prefix, self.moe_prefix = TPGather.get_layer_prefix_list(c_config, args)
        if c_vision_patch_config is not None:
            v_dense_keys, v_moe_keys = TPGather.get_layer_prefix_list(c_vision_patch_config, args)
            self.dense_prefix += v_dense_keys
            self.moe_prefix += v_moe_keys

    @staticmethod
    def calculate_tensor_memory_size(tensor: torch.Tensor) -> int:
        """
        Args:
            tensor: PyTorch Tensor
        Returns:
            int: Tensor memory size in bytes
        """
        if tensor is None:
            return 0
        return tensor.element_size() * tensor.numel()

    def gather_state_dicts_balanced(self, state_dict: Dict[str, torch.Tensor], layer_prefix_list, ranks, rank_groups):
        """ Split state_dict by key prefix and assign to ranks round-robin
        Args:
            state_dict: Dictionary with key->tensor
            layer_prefix_list: List of layer prefixes
            ranks: List of ranks
            rank_groups: Dict mapping group_key -> list of ranks

        Returns:
            Dict: Dictionary mapping each key to its assigned rank_id or None
        """

        if not state_dict:
            return {}
        if ranks is None or len(ranks) == 0:
            return {}

        cur_rank = dist.get_rank()

        # Initialize result list with empty dicts for each prefix
        layer_groups = [{} for _ in layer_prefix_list]

        # Group keys by matching prefix
        for key, tensor in state_dict.items():
            matched_idx = 0
            for idx, prefix in enumerate(layer_prefix_list):
                if key.startswith(prefix):
                    matched_idx = idx
                    break
            layer_groups[matched_idx][key] = tensor

        # Filter out empty dicts and assign to ranks round-robin
        result = {}
        layer_groups = [d for d in layer_groups if d]
        for idx, group in enumerate(layer_groups):
            rank_id = ranks[idx % len(ranks)]

            # Find which rank_group contains rank_id
            target_rank_group = None
            for group_key, rank_list in rank_groups.items():
                if rank_id in rank_list:
                    target_rank_group = rank_list
                    break

            # Check if cur_rank is in the target_rank_group
            should_assign = (target_rank_group is not None and cur_rank in target_rank_group)

            for key in group:
                result[key] = rank_id if should_assign else None

        return result

    def split_state_dict_by_moe(
        self,
        state_dict: Dict[str, torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Split state_dict into moe_state_dict and dense_state_dict based on moe_prefix.

        Args:
            state_dict: Complete state_dict of current rank

        Returns:
            tuple: (moe_state_dict, dense_state_dict)
                - moe_state_dict: Keys starting with any prefix in self.moe_prefix
                - dense_state_dict: All other keys
        """
        moe_state_dict = {}
        dense_state_dict = {}

        for key, value in state_dict.items():
            if key.endswith(f".{EXTRA_DATA}"):
                continue
            # Check if key starts with any moe_prefix
            if len(self.moe_prefix) == 0:
                is_moe = False
            else:
                is_moe = any(key.startswith(prefix) for prefix in self.moe_prefix)
            if is_moe:
                moe_state_dict[key] = value
            else:
                dense_state_dict[key] = value

        return moe_state_dict, dense_state_dict

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
        """
        # Split state_dict into moe and dense parts
        moe_state_dict, dense_state_dict = self.split_state_dict_by_moe(state_dict)

        tp_group = parallel_state.get_tensor_model_parallel_group()
        etp_group = parallel_state.get_expert_tensor_parallel_group()
        cur_rank_id = dist.get_rank()

        # Get sorted keys for consistent ordering
        dense_local_keys = sorted([k for k, v in dense_state_dict.items() if isinstance(v, torch.Tensor)])
        if len(moe_state_dict) > 0:
            moe_local_keys = sorted([k for k, v in moe_state_dict.items() if isinstance(v, torch.Tensor)])
        else:
            moe_local_keys = []

        dense_ranks_for_tp, dense_rank_groups_for_tp, moe_ranks_for_etp, moe_rank_groups_for_etp = self.get_rank_group()

        # Prepare result structure for tp_rank 0
        result = [{} for _ in range(self.tp_size)]

        # Process tensors one by one to ranks
        dense_key_tp_dict = self.gather_state_dicts_balanced(
            dense_state_dict, self.dense_prefix, dense_ranks_for_tp, dense_rank_groups_for_tp)
        if len(moe_state_dict) > 0:
            moe_key_tp_dict = self.gather_state_dicts_balanced(
                moe_state_dict, self.moe_prefix, moe_ranks_for_etp, moe_rank_groups_for_etp)

        for key in dense_local_keys:
            tensor = dense_state_dict[key]
            rank_id = dense_key_tp_dict[key]
            if rank_id is None:
                pass
            elif cur_rank_id == rank_id:
                gathered_gpu = [torch.empty_like(tensor) for _ in range(self.tp_size)]
                dist.gather(tensor, gathered_gpu, dst=rank_id, group=tp_group)
                for tp_idx in range(self.tp_size):
                    result[tp_idx][key] = gathered_gpu[tp_idx].cpu()
                del gathered_gpu
            else:
                dist.gather(tensor, None, dst=rank_id, group=tp_group)

        tp_rank_0 = cur_rank_id - self.tp_rank
        if self.etp_rank is not None:
            etp_rank_0 = cur_rank_id - self.etp_rank
            tp_rank_for_etp_rank0 = etp_rank_0 - tp_rank_0
        else:
            tp_rank_for_etp_rank0 = 0
        for key in moe_local_keys:
            tensor = moe_state_dict[key]
            rank_id = moe_key_tp_dict[key]
            cur_etp_size = self.etp_size if self.etp_size is not None else self.tp_size
            if rank_id is None:
                pass
            elif cur_etp_size == 1:
                result[self.tp_rank][key] = tensor
                continue
            elif cur_rank_id == rank_id:
                gathered_gpu = [torch.empty_like(tensor) for _ in range(cur_etp_size)]
                dist.gather(tensor, gathered_gpu, dst=rank_id, group=etp_group)
                for etp_idx in range(cur_etp_size):
                    tp_idx = etp_idx + tp_rank_for_etp_rank0
                    result[tp_idx][key] = gathered_gpu[etp_idx].cpu()
                del gathered_gpu
            else:
                dist.gather(tensor, None, dst=rank_id, group=etp_group)

        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()

        # Final GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    @staticmethod
    def get_layer_prefix_list(c_config, args):
        name_map = c_config.get("name_map")["mcore"]
        layer_prefix = name_map[LAYER_PREFIX]
        cargs = c_config.get_args("common")
        num_layers = cargs["num_layers"]
        mtp_num_layers = args.mtp_num_layers if args.mtp_num_layers is not None else cargs.get("mtp_num_layers", 0)

        dense_prefix = []
        moe_prefix = []
        for layer_id in range(num_layers):
            base_prefix = f"{layer_prefix}.{layer_id}."
            dense_prefix.append(base_prefix)
            moe_prefix.extend(TPGather.get_expert_prefix(c_config, base_prefix))
        if mtp_num_layers > 0:
            mtp_layer_prefix = name_map[MTP_LAYER_PREFIX] if MTP_LAYER_PREFIX in name_map else layer_prefix
            for mtp_layer_id in range(mtp_num_layers):
                base_prefix = f"{mtp_layer_prefix}.{mtp_layer_id}."
                dense_prefix.append(base_prefix)
                moe_prefix.extend(TPGather.get_expert_prefix(c_config, base_prefix, is_mtp=True))

        return dense_prefix, moe_prefix

    @staticmethod
    def get_expert_prefix(c_config, base_prefix, is_mtp=False):
        name_map = c_config.get("name_map")["mcore"]
        if is_mtp:
            name_prefix = name_map[MTP_NAME_PREFIX_FOR_LAYER] if MTP_NAME_PREFIX_FOR_LAYER in name_map else None
        else:
            name_prefix = None

        result = []
        if MOE_GROUPED_GEMM_EXPERT in name_map:
            expert_tag = name_map[MOE_GROUPED_GEMM_EXPERT] if name_prefix is None else f"{name_prefix}.{name_map[MOE_GROUPED_GEMM_EXPERT]}"
            result.append(f"{base_prefix}{expert_tag}.")
        if MOE_EXPERT in name_map:
            expert_tag = name_map[MOE_EXPERT] if name_prefix is None else f"{name_prefix}.{name_map[MOE_EXPERT]}"
            result.append(f"{base_prefix}{expert_tag}.")

        return result

    def get_rank_group(self):
        world_size = dist.get_world_size()
        rank_id = dist.get_rank()
        if self.ep_size is None:
            ranks_in_group = self.tp_size
        else:
            if self.etp_size is None:
                etp_size = self.tp_size
            else:
                etp_size = self.etp_size
            if etp_size * self.ep_size >= self.tp_size:
                ranks_in_group = etp_size * self.ep_size
            else:
                ranks_in_group = self.tp_size
        ranks_in_pp = world_size // self.pp_size
        pp_rank = rank_id // ranks_in_pp
        if self.ep_size is not None:
            ep_rank = ((rank_id % ranks_in_pp) % (etp_size * self.ep_size)) // etp_size

        group_size = ranks_in_pp // ranks_in_group

        # Generate groups and assign pp_rank to each group
        # Groups with same pp_rank are combined together
        dense_ranks_for_tp = []
        dense_rank_groups_for_tp = {}
        moe_ranks_for_etp = []
        moe_rank_groups_for_etp = {}

        start = pp_rank * ranks_in_pp
        end = start + ranks_in_pp
        pp_group = list(range(start, end))
        for group_rank in range(group_size):
            group_start = group_rank * ranks_in_group
            group_end = group_start + ranks_in_group
            group = pp_group[group_start: group_end]
            tp_group_count = ranks_in_group // self.tp_size
            if self.ep_size is None: 
                for tp_group_rank in range(tp_group_count):
                    tp_group_start = tp_group_rank * self.tp_size
                    tp_group_end = tp_group_start + self.tp_size
                    tp_group = group[tp_group_start: tp_group_end]
                    dense_ranks_for_tp.extend(tp_group)
                    dense_rank_groups_for_tp[f"{pp_rank}_{group_rank}_{tp_group_rank}"] = tp_group
            else:
                tp_group = group[0: self.tp_size]
                dense_ranks_for_tp.extend(tp_group)
                dense_rank_groups_for_tp[f"{pp_rank}_{group_rank}_0"] = tp_group

                ep_group_size = self.ep_size * etp_size
                ep_group_count = ranks_in_pp // ep_group_size
                for ep_group_rank in range(ep_group_count):
                    ep_group_start = ep_group_rank * ep_group_size
                    ep_group_end = ep_group_start + ep_group_size
                    ep_group = group[ep_group_start: ep_group_end]

                    ep_start = ep_rank * etp_size
                    ep_end = ep_start + etp_size
                    ep_ranks = ep_group[ep_start: ep_end]
                    moe_ranks_for_etp.extend(ep_ranks)
                    moe_rank_groups_for_etp[f"{pp_rank}_{group_rank}_{ep_group_rank}_{ep_rank}"] = ep_ranks

        return dense_ranks_for_tp, dense_rank_groups_for_tp, moe_ranks_for_etp, moe_rank_groups_for_etp