# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Utilities for monitoring and profiling checkpoint load/save operations.

Provides:
1. Timing utilities to measure load/save performance
2. GPU memory monitoring to track VRAM changes during operations
"""

import time
import functools
from typing import Callable, Any, Dict, Optional, TypeVar, Tuple, List
from contextlib import contextmanager
import torch
import torch.distributed as dist
from megatron.training import print_rank_0

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class MemoryTracker:
    """Track GPU memory usage before and after operations."""

    def __init__(self):
        """Initialize memory tracker."""
        self.initial_memory = 0
        self.peak_memory = 0
        self.final_memory = 0
        self.memory_allocated_diff = 0
        self.memory_reserved_diff = 0
        self.snapshots: List[Dict] = []  # Store intermediate snapshots for detailed analysis

    def snapshot(self) -> Dict[str, float]:
        """Get current GPU memory snapshot in MB."""
        if not torch.cuda.is_available():
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'free': 0.0,
            }

        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': total - allocated,
            'total': total,
        }

    def start(self) -> None:
        """Record initial memory state."""
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.initial_memory_snapshot = self.snapshot()
        self.initial_memory = self.initial_memory_snapshot['allocated']
        self.peak_memory = self.initial_memory

    def update(self) -> None:
        """Update peak memory tracking."""
        if not torch.cuda.is_available():
            return

        current = self.snapshot()
        self.peak_memory = max(self.peak_memory, current['allocated'])

    def checkpoint(self, label: str = "") -> Dict[str, float]:
        """Record intermediate memory checkpoint for detailed analysis.

        Args:
            label: Label for this checkpoint (e.g., "after_gather", "before_save")

        Returns:
            Current memory snapshot with label
        """
        if not torch.cuda.is_available():
            return {}

        snapshot = self.snapshot()
        snapshot['label'] = label
        snapshot['timestamp'] = len(self.snapshots)
        self.snapshots.append(snapshot)
        self.peak_memory = max(self.peak_memory, snapshot['allocated'])
        return snapshot

    def end(self) -> Dict[str, float]:
        """Record final memory state and compute differences."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.final_memory_snapshot = self.snapshot()
        self.final_memory = self.final_memory_snapshot['allocated']

        self.memory_allocated_diff = self.final_memory - self.initial_memory
        self.memory_reserved_diff = (
            self.final_memory_snapshot['reserved'] -
            self.initial_memory_snapshot['reserved']
        )

        return {
            'initial_allocated_mb': self.initial_memory,
            'peak_allocated_mb': self.peak_memory,
            'final_allocated_mb': self.final_memory,
            'allocated_diff_mb': self.memory_allocated_diff,
            'reserved_diff_mb': self.memory_reserved_diff,
            'initial_reserved_mb': self.initial_memory_snapshot['reserved'],
            'final_reserved_mb': self.final_memory_snapshot['reserved'],
        }


class Timer:
    """Simple timer context manager for measuring elapsed time."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and optionally print results."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if self.verbose and dist.is_initialized():
            print_rank_0(f"{self.name} took {self.elapsed:.2f} seconds")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed


@contextmanager
def profile_checkpoint_operation(
    operation_name: str = "Checkpoint Operation",
    track_memory: bool = True,
    print_rank_0_only: bool = True
):
    """
    Context manager to profile checkpoint operations with timing and memory tracking.

    Args:
        operation_name: Name of the operation for logging
        track_memory: Whether to track GPU memory changes
        print_rank_0_only: Whether to only print output on rank 0

    Yields:
        Dict with timing and memory information

    Example:
        with profile_checkpoint_operation("Model Checkpoint Load") as profile_stats:
            load_hf_checkpoint_online(model, optimizer, scheduler, args)
            print(profile_stats)
    """
    stats = {
        'operation': operation_name,
        'timing': {},
        'memory': {},
        'rank': dist.get_rank() if dist.is_initialized() else 0,
    }

    # Start tracking
    timer = Timer(operation_name, verbose=False)
    memory_tracker = MemoryTracker() if track_memory else None

    if memory_tracker:
        memory_tracker.start()

    start_time = timer.__enter__()

    try:
        # Store memory_tracker in stats so nested code can update peak
        if memory_tracker:
            stats['_memory_tracker'] = memory_tracker
        yield stats
    finally:
        timer.__exit__(None, None, None)
        stats['timing']['total_seconds'] = timer.elapsed

        if memory_tracker:
            memory_stats = memory_tracker.end()
            stats['memory'] = memory_stats
            # Remove the tracker reference before returning
            stats.pop('_memory_tracker', None)

        # Print results
        if print_rank_0_only and dist.is_initialized():
            print_rank_0(_format_profile_stats(stats))
        elif not print_rank_0_only:
            print(_format_profile_stats(stats))


def time_checkpoint_operation(func: F) -> F:
    """
    Decorator to time checkpoint load/save operations.

    Wraps the function with timing and memory tracking.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that includes timing information

    Example:
        @time_checkpoint_operation
        def load_hf_checkpoint_online(model, optimizer, scheduler, args):
            # ... implementation
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with profile_checkpoint_operation(
            operation_name=f"{func.__name__}",
            track_memory=True
        ) as stats:
            result = func(*args, **kwargs)
        return result

    return wrapper


def _format_profile_stats(stats: Dict[str, Any]) -> str:
    """Format profile statistics for display."""
    lines = [
        "\n" + "="*80,
        f"[Rank {stats['rank']}] Profile: {stats['operation']}",
        "="*80,
    ]

    # Timing information
    if stats['timing']:
        lines.append("Timing:")
        for key, value in stats['timing'].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.2f} seconds")
            else:
                lines.append(f"  {key}: {value}")

    # Memory information
    if stats['memory']:
        lines.append("\nMemory (GPU):")
        memory = stats['memory']
        lines.append(f"  Initial Allocated: {memory['initial_allocated_mb']:.1f} MB")
        lines.append(f"  Peak Allocated: {memory['peak_allocated_mb']:.1f} MB")
        lines.append(f"  Final Allocated: {memory['final_allocated_mb']:.1f} MB")
        lines.append(f"  Allocated Change: {memory['allocated_diff_mb']:+.1f} MB")
        lines.append(f"  Reserved Change: {memory['reserved_diff_mb']:+.1f} MB")

    lines.append("="*80 + "\n")
    return "\n".join(lines)


class RankProfiler:
    """Profile operations across all ranks and collect statistics."""

    def __init__(self, operation_name: str = "Checkpoint Operation"):
        """
        Initialize rank profiler.

        Args:
            operation_name: Name of the operation
        """
        self.operation_name = operation_name
        self.rank_stats = {}

    @contextmanager
    def profile(self):
        """
        Context manager to profile operation on current rank.

        Yields:
            Dict to store operation statistics
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        stats = {
            'rank': rank,
            'operation': self.operation_name,
            'timing': {},
            'memory': {},
        }

        timer = Timer(f"[Rank {rank}] {self.operation_name}", verbose=False)
        memory_tracker = MemoryTracker()

        memory_tracker.start()
        timer.__enter__()

        try:
            yield stats
        finally:
            timer.__exit__(None, None, None)
            stats['timing']['elapsed_seconds'] = timer.elapsed
            stats['memory'] = memory_tracker.end()

            self.rank_stats[rank] = stats

    def gather_and_report(self) -> Dict[int, Dict[str, Any]]:
        """
        Gather statistics from all ranks and print report.

        Returns:
            Dict mapping rank -> statistics
        """
        if not dist.is_initialized():
            return self.rank_stats

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Gather all rank statistics to rank 0
        all_stats = [None] * world_size
        if rank == 0:
            all_stats = [self.rank_stats.get(r, {}) for r in range(world_size)]

        # Broadcast to all ranks
        if dist.is_initialized() and world_size > 1:
            import pickle
            if rank == 0:
                stats_bytes = pickle.dumps(all_stats)
            else:
                stats_bytes = None

            # Simple gather: each rank sends to rank 0
            obj_list = [None] * world_size if rank == 0 else None
            dist.gather_object(self.rank_stats, obj_list, dst=0)

            if rank == 0:
                all_stats = obj_list

        # Print report on rank 0
        if rank == 0:
            self._print_report(all_stats)

        return all_stats[rank] if rank < len(all_stats) else self.rank_stats

    def _print_report(self, all_stats: list):
        """Print profiling report from all ranks."""
        lines = [
            "\n" + "="*80,
            f"[Profiler Report] {self.operation_name}",
            "="*80,
        ]

        for rank_stats in all_stats:
            if not rank_stats:
                continue

            rank = rank_stats.get('rank', '?')
            timing = rank_stats.get('timing', {})
            memory = rank_stats.get('memory', {})

            lines.append(f"\nRank {rank}:")
            if timing:
                lines.append(f"  Elapsed: {timing.get('elapsed_seconds', 0):.2f}s")

            if memory:
                mem = memory
                lines.append(f"  Memory Allocated: {mem['initial_allocated_mb']:.1f} → {mem['final_allocated_mb']:.1f} MB "
                           f"(Δ {mem['allocated_diff_mb']:+.1f} MB)")
                lines.append(f"  Memory Reserved: {mem['initial_reserved_mb']:.1f} → {mem['final_reserved_mb']:.1f} MB "
                           f"(Δ {mem['reserved_diff_mb']:+.1f} MB)")

        lines.append("="*80 + "\n")
        print_rank_0("\n".join(lines))


def get_memory_stats(label: str = "", update_peak: bool = True) -> Dict[str, float]:
    """
    Get current GPU memory statistics.

    Args:
        label: Optional label for the statistics
        update_peak: If True, try to update any active memory tracker's peak value

    Returns:
        Dict with memory statistics
    """
    tracker = MemoryTracker()
    stats = tracker.snapshot()

    if dist.is_initialized():
        rank = dist.get_rank()
        print(f"[Rank {rank}] Memory {label}: "
              f"Allocated={stats['allocated']:.1f}MB, "
              f"Reserved={stats['reserved']:.1f}MB, "
              f"Free={stats['free']:.1f}MB")
    else:
        print(f"Memory {label}: "
              f"Allocated={stats['allocated']:.1f}MB, "
              f"Reserved={stats['reserved']:.1f}MB, "
              f"Free={stats['free']:.1f}MB")

    return stats
