# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing utilities for diffusion models"""

from typing import List


def first_fit(seqlens: List[int], pack_size: int) -> List[List[int]]:
    """
    Packs sequences of varying lengths into bins using the First-Fit algorithm.

    Args:
      seqlens: A list of integers, representing the lengths of the sequences to be packed.
      pack_size: The maximum capacity of each bin.

    Returns:
      A list of lists, where each inner list represents a bin and contains the indices
        of the sequences assigned to that bin.
    """
    bins = []  # each bin: {"indices": [...], "used": total_size}
    for i, s in enumerate(seqlens):
        # Find first bin that fits
        found = False
        for bin_info in bins:
            if bin_info["used"] + s <= pack_size:
                bin_info["indices"].append(i)
                bin_info["used"] += s
                found = True
                break
        if not found:  # open a new bin
            bins.append({"indices": [i], "used": s})
    # Return only the indices
    return [bin_info["indices"] for bin_info in bins]
