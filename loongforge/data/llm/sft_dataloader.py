# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""utils for sft"""

import logging
from collections import deque, defaultdict, namedtuple
from dataclasses import dataclass

from typing import TYPE_CHECKING, Optional, Union, Type
import os
import torch
from torch.utils.data import DataLoader
from transformers.utils import PaddingStrategy

from datasets import load_from_disk
from datasets.distributed import split_dataset_by_node

from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler

from loongforge.data import IGNORE_INDEX
from loongforge.data.llm.sft_data_collator import DataCollatorForSupervisedDataset


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
logger = logging.getLogger(__name__)


# A scheduling slot consumed by `_schedule_step_aligned`. Each slot occupies
# `len(chunks)` chunks of step capacity and contributes one or more real
# source groups to the per-sample loss normalization.
#
# - chunks:     flat dataset indices, length k = sum(components).
# - components: real-group sizes inside the slot, in chunk-placement order.
#               * Real slot: components == [k] (single real group).
#               * Synth slot: components == [c1, c2, ...] with sum == k,
#                 stitched from short real groups in the synthesis path.
#
# MLA's KV chain resets at chunk_idx_in_group==0, so each component (real
# group) inside a synth slot keeps an independent KV chain — no special
# handling needed in the attention layer.
Slot = namedtuple("Slot", ["chunks", "components"])


def _cyclic_iter(iter):
    """cyclic iteration"""
    while True:
        for x in iter:
            yield x


def llm_pad_to_multiple_of(args) -> int:
    """Sequence multiple an LLM SFT batch is padded to before the model sees it."""
    if args.enable_chunkpipe and getattr(args, "sft_chunkpipe_mode", False):
        return args.chunksize
    factor = args.tensor_model_parallel_size if args.sequence_parallel else 1
    if args.context_parallel_size > 1:
        factor *= 2 * args.context_parallel_size
    # TransformerEngine FP8 GEMMs want a multiple of 128.
    # https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L151
    if args.fp8:
        factor *= 128
    return factor


def build_sft_data_collator(
    cls: Type[DataCollatorForSupervisedDataset],
    config: "SFTLoaderConfig",
    tokenizer,
    pad_to_multiple_of: int,
    **kwargs,
) -> DataCollatorForSupervisedDataset:
    """build data collator for sft

    The training method passes ``pad_to_multiple_of`` from
    ``llm_pad_to_multiple_of(args)``.
    """
    assert hasattr(
        tokenizer, "tokenizer"
    ), f"Only support HFTokenizer for sft, but got {config.tokenizer_type}."

    padding = (
        PaddingStrategy.LONGEST
        if config.variable_seq_lengths
        else PaddingStrategy.MAX_LENGTH
    )

    # When chunkpipe is enabled, all base chunks are already padded to chunksize.
    # If SFT chunkpipe + MTP is enabled, the collator temporarily strips bridge
    # tokens, pads only the base part to pad_to_multiple_of, then appends bridge
    # tokens back.
    max_length = config.chunksize if config.enable_chunkpipe else config.seq_length

    if config.enable_chunkpipe and config.sft_chunkpipe_mode:
        kwargs["chunkpipe_base_length"] = config.chunksize
        kwargs["chunkpipe_mtp_num_layers"] = config.mtp_num_layers

    data_collator = cls(
        tokenizer=tokenizer.tokenizer,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
        padding=padding,
        max_length=max_length,
        **kwargs,
    )
    return data_collator


class _IterableWithState:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.step = 0
        self._iterator = iter(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
            self.step += 1
            return batch
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
            self.step += 1
            return batch

    def save_state(self):
        """dataloader save state"""
        return {"step": self.step}

    def load_state(self, state):
        """dataloader load state"""
        target = state.get("step", 0)
        if target <= self.step:
            return
        for _ in range(target - self.step):
            next(self._iterator)
        self.step = target


class SavableCyclicIterator:
    """
    Cyclic iterator that:
      - exposes `.iterable` with save_state/load_state (via _IterableWithState)
      - yields batches infinitely
    Compatible with Megatron's maybe_save_dataloader_state().
    """

    def __init__(self, dataloader):
        self.iterable = _IterableWithState(dataloader)
        self._iterator = self._cyclic_iter(self.iterable)

    def _cyclic_iter(self, iterable_with_state):
        while True:
            for batch in iterable_with_state:
                yield batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def save_state(self):
        """dataloader save state"""
        return self.iterable.save_state()

    def load_state(self, state):
        """dataloader load state"""
        return self.iterable.load_state(state)


class ChunkPipeGroupBatchSampler:
    """Batch sampler that shuffles chunk groups while preserving intra-group order
    and aligning chunk groups to training step boundaries.

    In chunkpipe SFT, a long sequence is split into multiple consecutive chunks
    that must be yielded in order. Each chunk carries a `chunk_group_size` field
    indicating how many consecutive chunks belong to the same source sequence
    (1 for binpacked short-sequence chunks).

    All chunks of a long sequence must fall within the same training step
    (same gradient-accumulation window), because KV cache is carried between
    chunks and would be invalidated by a gradient update.

    This sampler:
      1. Scans the dataset to identify groups (consecutive chunks with the same group size).
      2. Shuffles groups for training randomness.
      3. Shards groups across data-parallel ranks via LPT multiway partition
         (whole groups, never split), balancing total chunk count per rank so
         that all ranks produce nearly the same number of complete steps.
      4. Schedules groups into fixed-capacity step windows (capacity = num_microbatches
         x micro_batch_size chunks) using FFD (first-fit decreasing) to maximize
         packing density, ensuring no group is split across step boundaries.
      5. Aligns the per-rank step count to the cross-rank minimum, so every rank
         yields the same number of micro-batches per epoch (required for
         collective-sync correctness under DDP).
      6. Yields one micro-batch at a time, keeping group members consecutive.
    """

    def __init__(
        self,
        dataset,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        num_microbatches,
        seed=0,
        enable_synthesis=False,
    ):
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.num_microbatches = num_microbatches
        self.seed = seed
        # Heterogeneous-DP synthesis path: when attn_dp != expert_dp the
        # partitioner switches from LPT to `_equal_size_partition`, which
        # guarantees per-size group-count parity across DP ranks (required for
        # MoE All2All lockstep) and synthesizes virtual groups from the short
        # sequence pool when residuals trigger threshold rules.
        self.enable_synthesis = enable_synthesis

        # Step capacity in chunks: each step has num_microbatches micro-batches,
        # each micro-batch holds micro_batch_size chunks.
        self.step_capacity = num_microbatches * micro_batch_size

        # Build groups by scanning chunk_group_size
        self.groups = []
        idx = 0
        group_sizes = dataset["chunk_group_size"]
        max_group_size = 0
        while idx < total_samples:
            size = group_sizes[idx]
            self.groups.append(list(range(idx, idx + size)))
            max_group_size = max(max_group_size, size)
            idx += size

        assert max_group_size <= self.step_capacity, (
            f"Max chunk_group_size ({max_group_size}) exceeds step capacity "
            f"({self.step_capacity} = num_microbatches {num_microbatches} "
            f"x micro_batch_size {micro_batch_size}). "
            f"Increase global_batch_size or decrease seq_length/chunksize ratio."
        )

        # Determine initial epoch and completed iteration offset for checkpoint
        # resume. ChunkPipe SFT only supports resuming from full training
        # iteration boundaries.
        self._epoch, self._resume_step_in_epoch = self._locate_resume_position(consumed_samples)

        # Per-microbatch G FIFO queue. Semantics: each entry is the GLOBAL
        # source group count G_total for the step that the corresponding
        # micro-batch belongs to (sum of step_gs across all DP ranks). This
        # replaces the previous per-rank G_local convention; using G_total
        # uniformly across ranks is required for the per-sample loss path to
        # match the target normalization 1/G_total when attention_dp != expert_dp,
        # and also fixes a latent precision drift in the equal-DP case where
        # G_local could fluctuate across ranks.
        #
        # Populated yield-time in __iter__, consumed by attach_step_metadata,
        # which pops one entry per batch and writes it onto the batch dict; get_batch
        # then broadcasts it from TP rank 0. The deques are created once and never
        # cleared, so FIFO alignment with the DataLoader's batch production holds:
        # sampler yield and metadata pop both run in the main process in strict order.
        self._step_g_queue = deque()

        # Per-microbatch composite-group descriptor queue. Each entry is the
        # `component_sizes` list of the composite group that the corresponding
        # micro-batch's chunk belongs to (e.g. [3, 2] for a composite of total
        # size 5 made of a real group of size 3 followed by a real group of
        # size 2). For trivial composites (every real group is its own
        # composite, the equal-DP path), this is a single-element list
        # [group_size]. Consumed by the SFT scheduler (via get_batch's TP rank-0
        # broadcast of the value attached to the batch) to drive
        # the outer composite loop and to attribute each chunk to its real
        # group within the composite.
        self._composite_queue = deque()

    def __len__(self):
        return self.total_samples

    def attach_step_metadata(self, base_iter):
        """Yield batches from ``base_iter`` with this sampler's per-micro-batch metadata attached.

        The sampler appends one entry right before yielding each micro-batch and the DataLoader
        returns batches in sampler order, so the i-th pop matches the i-th batch. Carrying the
        values on the batch keeps every iterator, VP chunk and split independent.
        """
        for batch in base_iter:
            batch["chunkpipe_step_num_groups"] = self._step_g_queue.popleft()
            batch["chunkpipe_component_sizes"] = list(self._composite_queue.popleft())
            yield batch

    def _lpt_partition(self, group_indices):
        """LPT (Longest Processing Time) multiway partition into DP buckets.

        Distributes the given group indices across `data_parallel_size` buckets
        such that the total chunk count per bucket is balanced. Largest groups
        are assigned first to the bucket with the smallest current load; ties
        are broken by bucket id for determinism. LPT guarantees
        `max_load - min_load <= max(group_size) <= step_capacity`, i.e. the
        cross-rank imbalance is bounded by one step window.

        All ranks run this locally and obtain the same assignment because the
        inputs (`self.groups` and `group_indices`) and the algorithm are
        deterministic — no collective communication is required.

        Args:
            group_indices: iterable of indices into self.groups.

        Returns:
            List[List[Slot]]: length `data_parallel_size`; `buckets[r]` is the
            list of Real slots assigned to rank r. Each Slot wraps a single
            real group (chunks=group, components=[len(group)]).
        """
        buckets = [[] for _ in range(self.data_parallel_size)]
        loads = [0] * self.data_parallel_size
        sorted_indices = sorted(
            group_indices,
            key=lambda gi: (-len(self.groups[gi]), gi),
        )
        for gi in sorted_indices:
            r = min(range(self.data_parallel_size), key=lambda r: (loads[r], r))
            group = self.groups[gi]
            buckets[r].append(Slot(chunks=list(group), components=[len(group)]))
            loads[r] += len(group)
        return buckets

    def _partition_groups(self, group_order):
        """Dispatch entry: choose LPT (homogeneous DP) or equal-size (synthesis)
        partition based on `self.enable_synthesis`.

        Both branches return `List[List[Slot]]` — buckets[r] is the ordered
        list of slots assigned to rank r. Real slots wrap a single real group;
        synth slots stitch multiple short real groups into a virtual group of
        a target size, used only on the synthesis path.
        """
        if self.enable_synthesis:
            return self._equal_size_partition(group_order)
        return self._lpt_partition(group_order)

    def _equal_size_partition(self, group_order):
        """Heterogeneous-DP partition: per-size group-count parity across ranks.

        For each size class k:
          - The first q*D groups (q = N_k // D) are dealt round-robin to D
            ranks, giving each rank exactly q real slots of size k.
          - The remaining r = N_k % D residual groups are either:
              * dropped (default for low-residual classes), or
              * synthesized into (D - r) virtual slots stitched from groups
                in a global short-pool, so all D ranks receive a size-k slot.
                Synthesis is all-or-nothing per size class: if the short-pool
                cannot fill (D - r) virtual slots, the consumed pool material
                is returned and the r real residuals are dropped too.

        Sizes are processed in descending order so that long sequences (rare,
        high per-sample value) get first pick of the short-pool. Decisions
        are deterministic given the shuffled `group_order`, so all ranks
        compute identical buckets locally without collective communication.

        Args:
            group_order: iterable of group ids in shuffled order. Order
                determines which groups become quotient / residual / pool
                material per size class (deterministic, seeded per-epoch).

        Returns:
            List[List[Slot]] of length D.
        """
        D = self.data_parallel_size

        # Pool: size class -> deque of group ids, in shuffled order. All
        # groups start in the pool; size-k processing dequeues from pool[k]
        # (taking exactly N_k entries), and residual synthesis dequeues from
        # smaller-size pools.
        pool = defaultdict(deque)
        for gi in group_order:
            pool[len(self.groups[gi])].append(gi)

        buckets = [[] for _ in range(D)]
        # Per-size temporary slot lists; appended to buckets after the size
        # class is fully processed (so synthesis failures roll back cleanly
        # without leaving partial slots).
        for k in sorted(pool.keys(), reverse=True):
            queue_k = pool[k]
            N_k = len(queue_k)
            if N_k == 0:
                continue
            # N_k = q*D + r, where q is the number of complete round-robin cycles
            # and r is the residual group count (ranks that won't get a slot of this size).
            q, r = divmod(N_k, D)

            # 1. q*D quotient groups → round-robin to D ranks (q each).
            quotient_slots_per_rank = [[] for _ in range(D)]
            for _ in range(q):
                for d in range(D):
                    gid = queue_k.popleft()
                    grp = self.groups[gid]
                    quotient_slots_per_rank[d].append(
                        Slot(chunks=list(grp), components=[len(grp)])
                    )

            if r == 0:
                for d in range(D):
                    buckets[d].extend(quotient_slots_per_rank[d])
                continue

            # 2. Residual decision.
            # must_synth: N_k < D (q==0) means fewer sequences of chunk size-k than DP ranks.
            # Without synthesis, all sequences of chunk size-k would be dropped; synthesize to avoid it.
            must_synth = (q == 0)
            # threshold_synth: if the residual count exceeds half of all size-k slots
            # (i.e., more than half of the groups would be dropped without synthesis), trigger synthesis.
            threshold_synth = (r / (q * D + r) > 0.5)

            if not (must_synth or threshold_synth):
                # Drop r residuals (don't return to pool — already labeled as
                # this size class and won't be reused at smaller k).
                for _ in range(r):
                    queue_k.popleft()
                for d in range(D):
                    buckets[d].extend(quotient_slots_per_rank[d])
                continue

            # 3. Synthesize (D - r) virtual slots of total length k each.
            target = D - r
            synth_slots, consumed = self._greedy_pack(pool, k, target,
                                                      exclude_size=k)
            if len(synth_slots) < target:
                # All-or-nothing rollback: return ingredients to pool, drop
                # the r real residuals as well. Quotient slots are kept.
                self._return_to_pool(pool, consumed)
                for _ in range(r):
                    queue_k.popleft()
                if must_synth:
                    logger.warning(
                        "[chunkpipe] size-%d class entirely dropped: "
                        "short-sequence pool insufficient to synthesize "
                        "%d virtual slots (got %d). Consider adding more "
                        "data or reducing data_parallel_size.",
                        k, target, len(synth_slots),
                    )
                for d in range(D):
                    buckets[d].extend(quotient_slots_per_rank[d])
                continue

            # 4. Inject the size class to all D ranks: r real + (D - r) synth.
            #    Rank-slot mapping is fixed by id (rank 0..r-1 → real,
            #    rank r..D-1 → synth) so all ranks compute identical bucket
            #    layouts locally.
            real_residuals = [queue_k.popleft() for _ in range(r)]
            for d in range(r):
                grp = self.groups[real_residuals[d]]
                quotient_slots_per_rank[d].append(
                    Slot(chunks=list(grp), components=[len(grp)])
                )
            for i, comp_gids in enumerate(synth_slots):
                d = r + i
                chunks = []
                components = []
                for cgid in comp_gids:
                    grp = self.groups[cgid]
                    chunks.extend(grp)
                    components.append(len(grp))
                quotient_slots_per_rank[d].append(
                    Slot(chunks=chunks, components=components)
                )
            for d in range(D):
                buckets[d].extend(quotient_slots_per_rank[d])

        return buckets

    def _greedy_pack(self, pool, length, count, exclude_size):
        """Greedy first-fit-decreasing pack: stitch (count) virtual groups
        whose component sizes sum exactly to `length`, drawing from `pool`.

        Each virtual group is built independently: for every slot we try
        size classes in descending order (within the < length range,
        excluding `exclude_size` to prevent self-feeding) and consume
        groups greedily until the running sum equals `length`. If the slot
        cannot be completed exactly, its locally-consumed groups are
        returned to the pool and the function returns immediately so the
        caller can perform an all-or-nothing rollback.

        Args:
            pool: Dict[size, deque[gid]] — global short-pool, mutated.
            length: target total chunk count of each virtual slot (== k).
            count: number of virtual slots to produce (== D - r).
            exclude_size: size class currently being processed; its pool is
                excluded as ingredient (the residuals from this class are
                what we're trying to pad).

        Returns:
            (synth_slots, consumed):
              synth_slots: List[List[gid]] of length <= count. Each inner
                list is the component gid sequence of a virtual slot.
              consumed: List[(size, gid)] — every gid consumed across all
                successfully-built slots, for caller rollback on failure.
        """
        synth_slots = []
        consumed = []

        sizes_desc = lambda: sorted(
            (k for k, dq in pool.items() if k != exclude_size and k < length and dq),
            reverse=True,
        )

        for _ in range(count):
            slot_components = []
            slot_consumed = []
            slot_sum = 0
            done = False
            while not done:
                progressed = False
                for sz in sizes_desc():
                    if sz > length - slot_sum:
                        continue
                    while pool[sz] and slot_sum + sz <= length:
                        gid = pool[sz].popleft()
                        slot_components.append(gid)
                        slot_consumed.append((sz, gid))
                        slot_sum += sz
                        progressed = True
                        if slot_sum == length:
                            done = True
                            break
                    if done:
                        break
                if not progressed:
                    break
            if slot_sum != length:
                # Cannot complete this virtual slot — return its ingredients
                # to the pool and abort. Caller decides rollback semantics.
                for sz, gid in reversed(slot_consumed):
                    pool[sz].appendleft(gid)
                return synth_slots, consumed
            synth_slots.append(slot_components)
            consumed.extend(slot_consumed)
        return synth_slots, consumed

    @staticmethod
    def _return_to_pool(pool, consumed):
        """Restore consumed ingredients back to the head of their size deque,
        preserving the original shuffled order so subsequent (smaller-k)
        size classes can re-attempt synthesis with the same material.
        """
        for sz, gid in reversed(consumed):
            pool[sz].appendleft(gid)

    def _get_epoch_aligned_steps(self, epoch):
        """Return aligned step count for the exact shuffled schedule of one epoch."""
        total_groups = len(self.groups)
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        group_order = torch.randperm(total_groups, generator=g).tolist()
        buckets = self._partition_groups(group_order)
        _, _, _, _, aligned_count = self._schedule_buckets(buckets)
        return aligned_count

    def _locate_resume_position(self, consumed_samples):
        """Locate resume epoch and completed step offset from consumed samples."""
        if consumed_samples <= 0:
            return 0, 0

        global_step_capacity = self.step_capacity * self.data_parallel_size
        assert consumed_samples % global_step_capacity == 0, (
            "SFT chunkpipe only supports resuming from an iteration checkpoint. "
            f"consumed_samples={consumed_samples} is not divisible by "
            f"global step capacity={global_step_capacity}."
        )

        completed_steps = consumed_samples // global_step_capacity
        epoch = 0
        while True:
            epoch_steps = self._get_epoch_aligned_steps(epoch)
            if completed_steps < epoch_steps:
                return epoch, completed_steps
            completed_steps -= epoch_steps
            epoch += 1

    def _schedule_buckets(self, buckets):
        """Schedule all buckets and cross-rank-align their step counts.

        Runs `_schedule_step_aligned` independently on every bucket (locally,
        all ranks compute the same results) and truncates the current rank's
        schedule to the minimum step count across all buckets. This guarantees
        every rank yields the same number of complete steps per epoch, which
        is required for DDP collectives to stay in lock-step.

        Also computes G_total per step (sum of source group counts across all
        DP ranks at the same step index), used by the per-sample loss path to
        normalize independent of per-rank G_local fluctuations. G_local at
        a step is the sum of `len(slot.components)` across all slots placed
        in that step — real slots contribute 1, synth slots contribute the
        number of real components they stitch. Computing G_total locally is
        exact because partition + FFD scheduling are deterministic given the
        same input — no collective communication required.

        Args:
            buckets: List[List[Slot]] from `_partition_groups`.

        Returns:
            Tuple (my_steps, my_step_gs, my_step_group_struct, g_total_per_step, aligned_count):
              my_steps: List[List[int]] of complete steps for the current rank,
                        already truncated to aligned_count.
              my_step_gs: List[int], number of source groups placed in each
                          step of my_steps (G_local for this rank, aligned).
              my_step_group_struct: List[List[List[int]]], per-step list of
                                    per-slot component_sizes for the current
                                    rank, aligned. Outer = steps, middle =
                                    slots within step, inner = real-group
                                    sizes inside the slot (sums to slot k).
              g_total_per_step: List[int] of length aligned_count, sum across
                                all ranks' step_gs at each step index.
              aligned_count: int, the common step count across all ranks.
        """
        step_counts = []
        step_gs_per_rank = []
        my_steps = None
        my_step_gs = None
        my_step_group_struct = None
        for r, bucket in enumerate(buckets):
            steps_r, step_gs_r, struct_r = self._schedule_step_aligned(bucket)
            step_counts.append(len(steps_r))
            step_gs_per_rank.append(step_gs_r)
            if r == self.data_parallel_rank:
                my_steps = steps_r
                my_step_gs = step_gs_r
                my_step_group_struct = struct_r
        aligned_count = min(step_counts) if step_counts else 0
        assert aligned_count > 0, (
            f"ChunkPipe sampler: 0 complete steps per epoch. "
            f"total_chunks={len(self.groups)}, step_capacity={self.step_capacity}, DP={self.data_parallel_size}.\n"
            f"Possible causes:\n"
            f"  1. Dataset too small (need at least {self.step_capacity * self.data_parallel_size} chunks total)\n"
            f"  2. --global-batch-size too large (reduces step_capacity={self.step_capacity})\n"
            f"Suggestion: add more data into dataset or reduce --global-batch-size."
        )
        g_total_per_step = [
            sum(step_gs_per_rank[r][s] for r in range(self.data_parallel_size))
            for s in range(aligned_count)
        ]
        return (
            my_steps[:aligned_count],
            my_step_gs[:aligned_count],
            my_step_group_struct[:aligned_count],
            g_total_per_step,
            aligned_count,
        )

    def _schedule_step_aligned(self, rank_slots):
        """Arrange slots into a list of complete step windows.

        For each step window of `step_capacity` chunks:
          1. Greedily place multi-chunk slots (long sequences) that fit, in
             size-descending order (FFD).
          2. Fill remaining capacity with single-chunk slots.

        Slots are atomic — a slot's chunks are never split across a step
        boundary. This guarantees that every chunk group (real or synthetic)
        starts at a fresh `chunk_idx_in_group==0` boundary that the MLA layer
        relies on for KV chain reset.

        Under-filled steps (when no single-slots remain and no deferred
        multi-slot fits the current vacancy) are dropped but scheduling
        continues — deferred multi-slots may still combine into complete
        windows in subsequent iterations.

        Args:
            rank_slots: List[Slot] assigned to this DP rank.

        Returns:
            Tuple (steps, step_gs, step_group_struct):
              steps: List[List[int]] — each inner list is exactly
                `step_capacity` dataset indices, representing one complete
                training step.
              step_gs: List[int] — step_gs[i] is the total number of real
                source groups placed in steps[i] (sum over slots of
                len(slot.components); used as G_local for this rank).
              step_group_struct: List[List[List[int]]] — step_group_struct[i]
                is the list of per-slot component_sizes placed in steps[i]
                in chunk order. Each inner list sums to that slot's k; the
                concatenation across slots sums to step_capacity.
        """
        # FFD (first-fit decreasing): sort multi-chunk slots by size descending
        # so that large slots are placed first and small slots act as "glue"
        # to fill remaining capacity. Within-step order of slots does not
        # affect training correctness (gradients are accumulated across all
        # chunks in the step).
        multi_slots = deque(sorted(
            (s for s in rank_slots if len(s.chunks) > 1),
            key=lambda s: len(s.chunks), reverse=True,
        ))
        single_slots = deque(s for s in rank_slots if len(s.chunks) == 1)

        steps = []
        step_gs = []
        step_group_struct = []

        while multi_slots or single_slots:
            remaining = self.step_capacity
            step_indices = []
            step_group_count = 0
            placed_components = []

            # Phase 1: greedily place multi-chunk slots
            deferred = deque()
            while multi_slots:
                slot = multi_slots.popleft()
                slot_k = len(slot.chunks)
                if slot_k <= remaining:
                    step_indices.extend(slot.chunks)
                    placed_components.append(list(slot.components))
                    remaining -= slot_k
                    # Each slot contributes len(components) real groups to G.
                    step_group_count += len(slot.components)
                else:
                    deferred.append(slot)
            # Put back slots that didn't fit for future steps
            multi_slots = deferred

            # Phase 2: fill remaining slots with single-chunk slots
            while single_slots and remaining > 0:
                slot = single_slots.popleft()
                step_indices.extend(slot.chunks)
                placed_components.append(list(slot.components))
                remaining -= 1
                step_group_count += len(slot.components)

            if len(step_indices) < self.step_capacity:
                # Under-filled step: drop and keep scheduling. Same termination
                # argument as before — every outer iteration consumes at least
                # one slot when multi_slots is non-empty.
                continue

            steps.append(step_indices)
            step_gs.append(step_group_count)
            step_group_struct.append(placed_components)

        return steps, step_gs, step_group_struct

    def __iter__(self):
        total_groups = len(self.groups)

        # Shuffle groups deterministically — different seed per epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        group_order = torch.randperm(total_groups, generator=g).tolist()

        # Shard groups across DP ranks. Homogeneous DP (attn_dp == expert_dp)
        # → LPT multiway partition (load-balanced by chunk count). Heterogeneous
        # DP → equal-size partition with short-pool synthesis (per-size group
        # parity for MoE All2All lockstep). All ranks run the chosen partition
        # locally and obtain identical assignments without any collective.
        buckets = self._partition_groups(group_order)

        # Schedule every bucket and truncate the current rank's schedule to
        # the cross-rank minimum step count. This guarantees every rank yields
        # the same number of micro-batches per epoch, preventing DDP collective
        # desync when one rank's iterator exhausts before others. Also derives
        # G_total per step (cross-rank sum) for the per-sample loss path.
        my_steps, _, my_step_group_struct, g_total_per_step, _ = self._schedule_buckets(buckets)

        # Resume only from full training iteration boundaries. Skipped steps do
        # not append queue entries, so chunkpipe metadata remains aligned with
        # the first real batch consumed after resume.
        skip_steps = self._resume_step_in_epoch
        self._resume_step_in_epoch = 0

        # Yield one micro-batch at a time. Every micro-batch within a step
        # shares the same G_total = total source groups in that step across
        # all DP ranks. We append G_total to _step_g_queue and per-microbatch
        # composite descriptor to _composite_queue immediately before yield
        # (never clear them) so that attach_step_metadata's popleft order matches
        # yield order strictly — robust to DataLoader prefetching and epoch
        # boundaries. Skipped micro-batches (via resume) do NOT enter the
        # queues, preserving FIFO alignment.
        #
        # Composite descriptor: each chunk maps to its enclosing slot's
        # `components` list — for a real slot this is `[k]`, for a synth
        # slot this is `[c1, c2, ...]` with sum(components)==k. Multiple
        # chunks in the same slot share the same descriptor list (they're
        # parts of the same composite); the consumer side uses
        # chunk_idx_in_group==0 to decide when to (re-)apply the descriptor.
        for step_idx, (step, G_total, step_components) in enumerate(
            zip(my_steps, g_total_per_step, my_step_group_struct)
        ):
            if step_idx < skip_steps:
                continue
            # Per-chunk attribution to enclosing slot. step_components[i] is
            # the components list of the i-th slot in this step; expand it
            # to one entry per chunk in the slot, sharing the same list.
            chunk_to_components = []
            for slot_components in step_components:
                slot_k = sum(slot_components)
                for _ in range(slot_k):
                    chunk_to_components.append(slot_components)
            for mb_start in range(0, len(step), self.micro_batch_size):
                batch = step[mb_start:mb_start + self.micro_batch_size]
                self._step_g_queue.append(G_total)
                # First chunk in this micro-batch maps to its slot's
                # components. Under chunkpipe SFT micro_batch_size is
                # typically 1 so the micro-batch's single chunk is
                # unambiguously attributed.
                self._composite_queue.append(chunk_to_components[mb_start])
                self.consumed_samples += self.micro_batch_size * self.data_parallel_size
                yield batch

        # Epoch complete — next __iter__ call will use a different shuffle
        self._epoch += 1
        self._resume_step_in_epoch = 0


@dataclass(frozen=True)
class SFTLoaderConfig:
    """Data-recipe fields needed to assemble the SFT loader.

    Built by the training method from its arguments; the loader never reads
    backend arguments or global CLI state.
    """

    tokenizer_type: str
    variable_seq_lengths: bool
    enable_chunkpipe: bool
    chunksize: int
    seq_length: int
    sft_chunkpipe_mode: bool
    mtp_num_layers: int
    sft_data_streaming: bool
    streaming_buffer_size: int
    seed: int
    micro_batch_size: int
    global_batch_size: int
    data_sharding: bool
    chunkpipe_enable_synthesis: bool
    num_workers: int
    dataloader_save: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "SFTLoaderConfig":
        """Build the loader recipe from the training arguments."""
        return cls(
            tokenizer_type=args.tokenizer_type,
            variable_seq_lengths=args.variable_seq_lengths,
            enable_chunkpipe=args.enable_chunkpipe,
            chunksize=args.chunksize,
            seq_length=args.seq_length,
            sft_chunkpipe_mode=getattr(args, "sft_chunkpipe_mode", False),
            mtp_num_layers=args.mtp_num_layers or 0,
            sft_data_streaming=args.sft_data_streaming,
            streaming_buffer_size=args.streaming_buffer_size,
            seed=args.seed,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            data_sharding=args.data_sharding,
            chunkpipe_enable_synthesis=getattr(args, "chunkpipe_enable_synthesis", False),
            num_workers=args.num_workers,
            dataloader_save=args.dataloader_save,
        )


def load_pretokenized_dataset(data_path, dp_rank):
    """Load the dataset pre-sharded for this data-parallel rank on disk."""
    save_path = os.path.join(data_path[0], "preprocess", str(dp_rank))
    print(f"[rank{dp_rank}] loading preprocessed dataset from {save_path}")
    return load_from_disk(save_path)


def _build_cylic_iterator(
    dataset: Union["Dataset", "IterableDataset"],
    consumed_samples: int,
    data_collator: DataCollatorForSupervisedDataset,
    config: "SFTLoaderConfig",
    dp_rank: int,
    dp_size: int,
):
    """build data iterator for sft"""
    if dataset is None:
        return None

    _dataloader_kwargs = {}
    if config.sft_data_streaming:
        # split distributed dataset for streaming
        dataset = split_dataset_by_node(
            dataset=dataset,
            rank=dp_rank,
            world_size=dp_size,
        )

        dataset = dataset.shuffle(
            buffer_size=config.streaming_buffer_size,
            seed=config.seed,
        )

        _dataloader_kwargs = dict(
            batch_size=config.micro_batch_size,
        )
    else:
        # build distribued sampler for non-streaming dataset
        if config.enable_chunkpipe:
            num_microbatches = config.global_batch_size // (
                config.micro_batch_size * dp_size
            )
            _batch_sampler = ChunkPipeGroupBatchSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=config.micro_batch_size,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
                num_microbatches=num_microbatches,
                seed=config.seed,
                enable_synthesis=config.chunkpipe_enable_synthesis,
            )
        else:
            _batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,  # not support for streaming now!
            micro_batch_size=config.micro_batch_size,
            data_parallel_rank=dp_rank,
            data_parallel_size=dp_size,
            data_sharding=config.data_sharding,
        )

        _dataloader_kwargs = dict(
            batch_sampler=_batch_sampler,
            persistent_workers=True if config.num_workers > 0 else False,
        )

    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        num_workers=config.num_workers,
        pin_memory=True,
        **_dataloader_kwargs,
    )

    if config.dataloader_save is not None:
        base_iter = SavableCyclicIterator(dataloader)
    else:
        base_iter = iter(_cyclic_iter(dataloader))

    if config.enable_chunkpipe and not config.sft_data_streaming:
        base_iter = _batch_sampler.attach_step_metadata(base_iter)

    return base_iter


def build_sft_cyclic_iterators(
    train_ds: Optional[Union["Dataset", "IterableDataset"]],
    valid_ds: Optional[Union["Dataset", "IterableDataset"]],
    test_ds: Optional[Union["Dataset", "IterableDataset"]],
    data_collator: Optional[DataCollatorForSupervisedDataset],
    config: "SFTLoaderConfig",
    dp_rank: int,
    dp_size: int,
    consumed_train_samples: int,
    consumed_valid_samples: int,
    skip_train: bool,
    state=None,
):
    """build data iterators for sft

    ``state`` restores the streaming iterator's saved position. Non-streaming SFT
    resumes from ``consumed_samples`` in the sampler instead, so the saved step is
    not applied there.
    """
    train_iter = _build_cylic_iterator(
        train_ds, consumed_train_samples, data_collator, config, dp_rank, dp_size
    )
    valid_iter = _build_cylic_iterator(
        valid_ds, 0 if skip_train else consumed_valid_samples, data_collator, config, dp_rank, dp_size
    )
    test_iter = _build_cylic_iterator(test_ds, 0, data_collator, config, dp_rank, dp_size)
    if state is not None and config.sft_data_streaming and hasattr(train_iter, "load_state"):
        train_iter.load_state(state)
    return train_iter, valid_iter, test_iter


def build_full_hetero_encoder_data_iterator(
    dataset,
    consumed_samples,
    data_collator,
    *,
    micro_batch_size,
    num_workers,
    data_sharding,
    data_parallel_rank,
    data_parallel_size,
    pp_rank,
    tp_size,
    model_size,
    num_real_microbatch,
    prefetch_count,
):
    """Build a DataLoader iterator for the encoder in full_hetero_dp mode.

    Uses EncoderStridedSampler to yield only microbatches assigned to this PP rank,
    avoiding unnecessary disk IO for microbatches handled by other ranks.
    """
    from loongforge.data.vlm.encoder_strided_sampler import EncoderStridedSampler, PrefetchIterator

    batch_sampler = EncoderStridedSampler(
        dataset,
        total_samples=len(dataset),
        consumed_samples=consumed_samples,
        micro_batch_size=micro_batch_size,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        data_sharding=data_sharding,
        pp_rank=pp_rank,
        tp_size=tp_size,
        model_size=model_size,
        num_real_microbatch=num_real_microbatch,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=data_collator,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    return PrefetchIterator(iter(_cyclic_iter(dataloader)), prefetch_count=prefetch_count)
