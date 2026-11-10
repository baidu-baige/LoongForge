# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""ChunkPipe cross-chunk recurrent-state cache, backend-agnostic.

Delta-rule linear-recurrence layers (KDA, Gated DeltaNet) carry a compact
recurrent state along the sequence.  When ChunkPipe splits a sequence into
pipeline chunks, that state must cross chunk boundaries, exactly like the
key/value cache does for softmax/MLA attention.  This mixin provides the slot
management and the store/load/gradient-hook primitives shared by every such
layer, so each attention module only declares its state tensors and passes them
through its operator's ``initial_state`` / ``output_final_state`` interface.

The gradient accounting differs from the softmax/MLA chunk KV cache in one
important way: a linear-recurrence chunk's *output* state is consumed only by
its successor chunk, never by its own output activation.  It is therefore not
part of the current chunk's loss graph, so its operator's backward (which needs
the successor's ``dh0`` as the ``dht`` input) would never run on its own.  The
forward pass stores detached state values; the backward recompute loads the
previous chunk's state as a ``requires_grad`` leaf (an accumulate hook captures
this chunk's ``dh0``), and the current chunk's output state is *tied into the
loss graph with zero weight* so a hook can deliver the successor's ``dh0`` to
it.  Gradients are consumed in LIFO order, enforced by ``check_kv_cache_grad_consumed``.
"""

import torch

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)


class ChunkPipeStateCacheMixin:
    """Mixin providing ChunkPipe cross-chunk recurrent-state caching.

    Subclass contract (see ``KimiK3Attention`` and ``GatedDeltaNet``):

    - At the end of ``__init__``, when ``config.enable_chunkpipe`` is set, call
      ``setup_chunkpipe_cache_config()``, then ``register_state_cache(name, shape, dtype)``
      once per carried state.
    - In ``forward``, per chunk::

          prev = self.load_prev_state(name)   # None means zero-init (first chunk)
          output, final = op(..., initial_state=prev, output_final_state=True)
          self.store_current_state(name, final)
          tie = self.inject_state_grad(name, final)   # None if no injection
          if tie is not None:
              output = output + tie.to(output.dtype)  # zero-sum: delivers the
                                                      # successor chunk's dh0

    All carried states of one chunk share a single cache slot, keyed by the
    chunk's forward microbatch index in ``micro_batch_to_cache_chunk_map``.
    """

    # ------------------------------------------------------------------
    # Slot management (mirrors attention.setup_chunkpipe_kv_cache_config)
    # ------------------------------------------------------------------
    def setup_chunkpipe_cache_config(self) -> None:
        """Initialize chunkpipe cache sizing and slot bookkeeping."""
        self.num_chunks_per_seq = self.config.chunk_num_per_seq

        pipeline_world_size = get_pipeline_model_parallel_world_size()
        pipeline_rank = get_pipeline_model_parallel_rank()
        # K3 and Qwen3-Next do not use virtual pipeline parallelism, so this is
        # the plain non-VPP sizing (same as the standard KV chunk cache).
        self.kv_cache_chunk_size = (
            (pipeline_world_size - pipeline_rank - 1) * 2 + self.num_chunks_per_seq
        )

        self.micro_batch_to_cache_chunk_map = {}
        self.empty_chunk_indices = list(range(self.kv_cache_chunk_size))
        self._state_caches = {}
        self._state_cache_grads = {}

    def is_enable_grad_chunkpipe(self) -> bool:
        """Whether gradients should be re-attached during this chunk's forward."""
        if not self.config.enable_chunkpipe:
            raise RuntimeError(
                "This method is valid only for ChunkPipe, please check "
                "config.enable_chunkpipe=True"
            )

        # Backward recompute: gradients must be re-attached to carry across chunks.
        if not self.config.chunkpipe_forward:
            return True

        if not self.training:
            return False

        # Forward: only the trailing keep_activations_chunks keep a live graph.
        if self.config.sft_chunkpipe_mode:
            current_chunk_idx = self.config.chunkpipe_chunk_idx_in_group
            effective_group = self.config.chunkpipe_current_group_size
        else:
            current_chunk_idx = (
                self.config.chunkpipe_forward_microbatch % self.num_chunks_per_seq
            )
            effective_group = self.num_chunks_per_seq
        return current_chunk_idx + self.config.keep_activations_chunks >= effective_group

    def clear_chunk_key_value_cache(self) -> None:
        """Reset all cache slots for reuse.

        Named to match ``attention.clear_chunk_key_value_cache`` so the chunkpipe
        scheduler's ``clear_key_value_cache`` helper works unchanged on models
        whose attention is a ``ChunkPipeStateCacheMixin`` subclass.
        """
        if not self.config.enable_chunkpipe:
            raise RuntimeError(
                "Chunk key-value cache operations require chunkpipe to be enabled. "
                "Please check your configuration."
            )
        self.empty_chunk_indices = list(range(self.kv_cache_chunk_size))
        self.micro_batch_to_cache_chunk_map.clear()

    def delete_chunk_key_value_cache(self, micro_batch_index: int) -> None:
        """Free the cache slot owned by ``micro_batch_index``.

        Named to match ``attention.delete_chunk_key_value_cache`` (see
        ``clear_chunk_key_value_cache``).
        """
        if not self.config.enable_chunkpipe:
            raise RuntimeError(
                "Chunk key-value cache operations require chunkpipe to be enabled. "
                "Please check your configuration."
            )
        if micro_batch_index not in self.micro_batch_to_cache_chunk_map:
            return
        cache_chunk_index = self.micro_batch_to_cache_chunk_map.pop(micro_batch_index)
        self.empty_chunk_indices.append(cache_chunk_index)

    # ------------------------------------------------------------------
    # State cache primitives
    # ------------------------------------------------------------------
    def register_state_cache(self, name: str, shape: tuple, dtype: torch.dtype) -> None:
        """Allocate one carried state's cache (``kv_cache_chunk_size`` slots)."""
        device = next(self.parameters()).device
        self._state_caches[name] = torch.zeros(
            (self.kv_cache_chunk_size, *shape), device=device, dtype=dtype
        )
        self._state_cache_grads[name] = {}

    def _microbatch_idx(self) -> int:
        """The microbatch index of the current chunk (forward or backward)."""
        return (
            self.config.chunkpipe_forward_microbatch
            if self.config.chunkpipe_forward
            else self.config.chunkpipe_backward_microbatch
        )

    def _current_chunk_idx(self) -> int:
        """0-based index of the current chunk within its sequence/group."""
        if self.config.sft_chunkpipe_mode:
            return self.config.chunkpipe_chunk_idx_in_group
        return self._microbatch_idx() % self.num_chunks_per_seq

    def _is_last_chunk(self) -> bool:
        """Whether the current chunk is the last one in its sequence/group."""
        if self.config.sft_chunkpipe_mode:
            return (
                self.config.chunkpipe_chunk_idx_in_group
                >= self.config.chunkpipe_current_group_size - 1
            )
        return (self._microbatch_idx() + 1) % self.num_chunks_per_seq == 0

    def load_prev_state(self, name: str):
        """Return the previous chunk's state, or ``None`` for the first chunk.

        During backward recompute the returned state is a detached leaf with a
        gradient-accumulation hook, so the operator's ``dh0`` lands in
        ``_state_cache_grads[name][prev_chunk_idx]``.
        """
        current_chunk_idx = self._current_chunk_idx()
        if current_chunk_idx == 0:
            return None

        start_microbatch_idx = self._microbatch_idx() - current_chunk_idx
        prev_chunk_idx = current_chunk_idx - 1
        cache_slot = self.micro_batch_to_cache_chunk_map[
            start_microbatch_idx + prev_chunk_idx
        ]
        state = self._state_caches[name][cache_slot]

        if self.is_enable_grad_chunkpipe():
            state = state.detach().requires_grad_(True)
            state.register_hook(self._make_state_accumulate_hook(name, prev_chunk_idx))
        return state

    def store_current_state(self, name: str, state: torch.Tensor) -> None:
        """Store the current chunk's output state for its successor chunk."""
        # Only the forward pass stores; backward recompute reuses stored values.
        if not self.config.chunkpipe_forward:
            return
        # The last chunk has no successor.
        if self._is_last_chunk():
            return

        microbatch_idx = self.config.chunkpipe_forward_microbatch
        # All carried states of one chunk share a single slot: allocate it on the
        # first state and reuse it for the rest.
        if microbatch_idx not in self.micro_batch_to_cache_chunk_map:
            if not self.empty_chunk_indices:
                raise RuntimeError(
                    f"No available chunk state cache slots for '{name}'. "
                    "Consider increasing cache size."
                )
            self.micro_batch_to_cache_chunk_map[microbatch_idx] = (
                self.empty_chunk_indices.pop(0)
            )
        cache_slot = self.micro_batch_to_cache_chunk_map[microbatch_idx]
        self._state_caches[name][cache_slot] = state.detach()

    def inject_state_grad(self, name: str, state: torch.Tensor):
        """Inject the successor chunk's gradient into the current chunk's state.

        The current chunk's output boundary state is consumed only by its
        successor chunk, so it is not part of the current chunk's loss graph and
        its operator backward would never run w.r.t. it.  Register a hook that,
        at backward time, replaces the state's gradient with the one the next
        chunk accumulated (``_state_cache_grads[name][chunk_idx]``), and return
        a zero scalar tied to ``state`` so it participates in the loss graph.

        Callers must add the returned scalar to the layer output (with dtype
        matching the output).  Returns ``None`` when nothing needs injecting
        (forward pass, first/last chunk, or grad disabled).
        """
        if not self.is_enable_grad_chunkpipe() or self._is_last_chunk():
            return None
        chunk_idx = self._current_chunk_idx()
        state.register_hook(self._make_state_inject_hook(name, chunk_idx))
        return state.sum() * 0.0

    def check_kv_cache_grad_consumed(self) -> None:
        """Assert no orphan gradients remain after a group's backward.

        Checks every gradient bucket this mixin's layers can hold: the recurrent
        state grads (``_state_cache_grads``, used by KDA/GDN) and, when the layer
        also carries an MLA chunk KV cache, its compressed-KV / key-extra grads
        (``kv_compressed_cache_grad`` / ``key_extra_cache_grad``).  Leftover
        entries indicate a LIFO-order violation, mirroring the base class.
        """
        for name, grads in self._state_cache_grads.items():
            if grads:
                orphan_keys = list(grads.keys())
                grads.clear()
                raise AssertionError(
                    f"[ChunkPipe] Orphan state cache grads {orphan_keys} remain after "
                    f"group backward for state '{name}'. "
                    f"layer={getattr(self, 'layer_number', None)}"
                )
        for attr in ("kv_compressed_cache_grad", "key_extra_cache_grad"):
            grads = getattr(self, attr, None)
            if grads:
                orphan_keys = list(grads.keys())
                grads.clear()
                raise AssertionError(
                    f"[ChunkPipe][MLA] Orphan {attr} entries {orphan_keys} remain after "
                    f"group backward. layer={getattr(self, 'layer_number', None)}"
                )

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------
    def _make_state_accumulate_hook(self, name: str, chunk_idx: int):
        """Accumulate the gradient a later chunk produces w.r.t. this chunk's state."""

        def hook_fn(grad):
            grads = self._state_cache_grads[name]
            if chunk_idx not in grads:
                grads[chunk_idx] = grad
            else:
                grads[chunk_idx] = grads[chunk_idx] + grad
            return grad

        return hook_fn

    def _make_state_inject_hook(self, name: str, chunk_idx: int):
        """Replace the state's gradient with the successor chunk's ``dh0``.

        The tie (``state.sum() * 0``) drives a zero gradient into ``state``; this
        hook discards it and returns the gradient the next chunk accumulated
        w.r.t. this chunk's state instead.
        """

        def hook_fn(_grad):
            return self._state_cache_grads[name].pop(chunk_idx)

        return hook_fn


__all__ = ["ChunkPipeStateCacheMixin"]
