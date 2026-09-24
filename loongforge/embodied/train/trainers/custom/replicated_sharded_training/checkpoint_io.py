# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local checkpoint I/O for the replicated-sharded training backend.

Compute replicas are complete on every rank and travel with the model state
dict, so only the owner-distributed half is written here: the fp32 masters and
the per-parameter optimizer state that follows them. The shared checkpoint
backend owns the file layout and the aux files (scheduler, RNG, dataloader) and
delegates this half through ``optimizer.zero_checkpoint_io``.
"""

from __future__ import annotations

import json
import os

import torch

BACKEND = "replicated_zero1"
FORMAT_VERSION = 1


class ReplicatedShardedCheckpointIO:
    """Save and restore the owner-distributed half of the replicated shard."""

    def __init__(self, manager, optimizer):
        """Bind to the parameter manager and the optimizer over its masters."""
        self._manager = manager
        self._optimizer = optimizer

    # -- save -----------------------------------------------------------------

    def save_local_state(self, state_dir, metadata_path, ctx) -> None:
        """Write this rank's masters and optimizer state, then the shared metadata."""
        if (ctx.rank, ctx.world_size) != (self._manager.rank, self._manager.world_size):
            raise RuntimeError("checkpoint context does not match parameter registry")
        manager_state = self._manager.state_dict()
        local_state = {
            "format_version": FORMAT_VERSION,
            "backend": BACKEND,
            "manager": manager_state,
            "optimizer": self._optimizer.state_dict(),
            "optimizer_param_names": self._optimizer_param_names(),
        }
        self._saved_manifest = manager_state["manifest"]
        self._saved_world_size = ctx.world_size
        self._validate_rank_state(local_state, ctx.rank)
        torch.save(
            local_state,
            os.path.join(state_dir, f"rank_{ctx.rank}.pt"),
        )
        ctx.barrier()
        if ctx.is_main:
            with open(metadata_path, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "format_version": FORMAT_VERSION,
                        "backend": BACKEND,
                        "world_size": ctx.world_size,
                        "manifest": self._manager.parameter_manifest(),
                    },
                    file,
                    indent=2,
                )
        ctx.barrier()

    # -- load -----------------------------------------------------------------

    def load_local_state(self, state_dir, metadata, ctx) -> None:
        """Restore this rank's share, re-partitioning when the layout changed."""
        saved_world_size = int(metadata["world_size"])
        if (ctx.rank, ctx.world_size) != (self._manager.rank, self._manager.world_size):
            raise RuntimeError("checkpoint context does not match parameter registry")
        if saved_world_size < 1:
            raise RuntimeError("invalid checkpoint world_size")
        self._manager.registry.validate_manifest(
            metadata.get("manifest"),
            allow_reshard=saved_world_size != ctx.world_size,
            saved_world_size=saved_world_size,
        )
        self._saved_manifest = metadata.get("manifest")
        self._saved_world_size = saved_world_size
        if saved_world_size == ctx.world_size:
            self._load_same_layout(state_dir, ctx)
        else:
            self._load_resharded(state_dir, ctx, saved_world_size)
        # The checkpoint also carries compute parameters, but the fp32 master is
        # the source of truth here. Publishing it after the load keeps a stale or
        # rounded model shard from surviving the resume.
        self._manager.start_serial_parameter_sync()
        self._manager.finish_serial_parameter_sync()

    def _load_same_layout(self, state_dir, ctx) -> None:
        """Load this rank's file as-is; ownership is asserted to be unchanged."""
        local_state = self._read_rank_file(state_dir, ctx.rank)
        self._validate_rank_state(local_state, ctx.rank)
        names = local_state.get("optimizer_param_names")
        if names is not None and names != self._optimizer_param_names():
            raise RuntimeError("checkpoint optimizer parameter order mismatch")
        self._manager.load_state_dict(local_state["manager"])
        self._optimizer.load_state_dict(local_state["optimizer"])

    def _load_resharded(self, state_dir, ctx, saved_world_size) -> None:
        """Re-partition state saved under a different world size.

        Both halves are addressed by registry name, so each rank scans the saved
        rank files, keeps the names its new ownership plan assigns to it, and
        drops the rest. Peak host memory is one saved rank file plus this rank's
        own share.

        RNG and dataloader position are per-rank and cannot be re-partitioned;
        the shared resume path skips them, which restarts the data stream.
        """
        wanted = set(self._manager.master)
        template = self._optimizer.state_dict()
        inner_templates = self._inner_states(template)
        inner_count = len(inner_templates)
        masters: dict = {}
        state_by_name: list = [{} for _ in range(inner_count)]

        for saved_rank in range(saved_world_size):
            saved = self._read_rank_file(state_dir, saved_rank)
            self._validate_rank_state(saved, saved_rank)
            names_per_optimizer = saved.get("optimizer_param_names")
            if names_per_optimizer is None:
                raise RuntimeError(
                    "This checkpoint predates name-keyed Zero-1 optimizer state and "
                    f"can only be resumed at its original world size "
                    f"({saved_world_size})."
                )
            if len(names_per_optimizer) != inner_count:
                raise RuntimeError(
                    "Zero-1 inner optimizer count mismatch: "
                    f"checkpoint={len(names_per_optimizer)}, current={inner_count}."
                )
            for name, value in saved["manager"]["master"].items():
                if name in wanted:
                    masters[name] = value
            for index, (names, saved_optimizer) in enumerate(
                zip(names_per_optimizer, self._inner_states(saved["optimizer"]))
            ):
                for position, name in enumerate(names):
                    if name in wanted and position in saved_optimizer["state"]:
                        state_by_name[index][name] = saved_optimizer["state"][position]
            del saved

        self._manager.load_master_tensors(masters)

        for index, names in enumerate(self._optimizer_param_names()):
            state = {}
            for position, name in enumerate(names):
                entry = state_by_name[index].get(name)
                if entry is not None:
                    state[position] = entry
            inner_templates[index]["state"] = state
        self._optimizer.load_state_dict(template)

    # -- helpers --------------------------------------------------------------

    def _validate_rank_state(self, saved, rank):
        """Check all rank-local contracts before any master/optimizer mutation."""
        state = saved["manager"]
        if state.get("world_size") != self._saved_world_size:
            raise RuntimeError("rank-local checkpoint world_size mismatch")
        manifest = state.get("manifest")
        if manifest != self._saved_manifest:
            raise RuntimeError("rank-local/shared parameter manifest mismatch")
        self._manager.registry.validate_manifest(
            manifest, allow_reshard=True, saved_world_size=self._saved_world_size,
        )
        if manifest is None:
            return
        rows = {row["name"]: row for row in manifest["parameters"]}
        expected = {name for name, row in rows.items() if row["owner_rank"] == rank}
        if set(state["master"]) != expected:
            raise RuntimeError("saved master keys do not match manifest owners")
        for name, tensor in state["master"].items():
            if list(tensor.shape) != rows[name]["shape"] or tensor.dtype != torch.float32:
                raise RuntimeError(f"invalid saved FP32 master: {name}")
        names = saved.get("optimizer_param_names")
        if names is not None:
            flattened = [name for group in names for name in group]
            if len(flattened) != len(set(flattened)) or set(flattened) != expected:
                raise RuntimeError("saved optimizer names do not match manifest owners")

    @staticmethod
    def _read_rank_file(state_dir, rank):
        """Load and validate one saved rank file."""
        rank_path = os.path.join(state_dir, f"rank_{rank}.pt")
        if not os.path.exists(rank_path):
            raise FileNotFoundError(
                f"replicated-sharded optimizer state for rank {rank} not found: {rank_path}"
            )
        saved = torch.load(rank_path, map_location="cpu", weights_only=False)
        if saved.get("backend") != BACKEND:
            raise RuntimeError(f"Unsupported rank-local ZeRO state in {rank_path}.")
        if saved.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(f"Unsupported rank-local ZeRO format in {rank_path}.")
        return saved

    @staticmethod
    def _inner_states(state_dict) -> list:
        """Return per-inner-optimizer state dicts for plain and combined optimizers."""
        if "optimizers" in state_dict:
            return state_dict["optimizers"]
        return [state_dict]

    def _optimizer_param_names(self) -> list:
        """Map each inner optimizer's parameter indices to registry names.

        ``state_dict`` keys the per-parameter state by the index a parameter has
        in its optimizer's ``param_groups`` traversal. Those indices are
        owner-local, so they mean nothing to a run with a different world size.
        Recording the registry name per index makes the state re-partitionable.
        """
        name_by_id = {
            id(value): key for key, value in self._manager.master.items()
        }
        inner = getattr(self._optimizer, "optimizers", [self._optimizer])
        names_per_optimizer = []
        for child in inner:
            names = []
            for group in child.param_groups:
                for parameter in group["params"]:
                    names.append(name_by_id.get(id(parameter)))
            names_per_optimizer.append(names)
        return names_per_optimizer


__all__ = ["BACKEND", "FORMAT_VERSION", "ReplicatedShardedCheckpointIO"]
