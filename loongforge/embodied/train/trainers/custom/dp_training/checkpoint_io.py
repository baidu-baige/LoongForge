# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local checkpoint I/O for the custom_dp training backend.

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

BACKEND = "custom_dp"
FORMAT_VERSION = 1


class RankLocalCheckpointIO:
    """Save and restore the owner-distributed half of the custom_dp state."""

    def __init__(self, dp_optimizer):
        """Bind to the ``CustomDpOptimizer``; its inner optimizer holds the state."""
        self._dp = dp_optimizer
        self._optimizer = dp_optimizer.optim

    # -- save -----------------------------------------------------------------

    def save_local_state(self, state_dir, metadata_path, ctx) -> None:
        """Write this rank's masters and optimizer state, then the shared metadata."""
        if (ctx.rank, ctx.world_size) != (self._dp.rank, self._dp.world_size):
            raise RuntimeError("checkpoint context does not match parameter plan")
        local_state = {
            "format_version": FORMAT_VERSION,
            "backend": BACKEND,
            "manager": self._dp.master_state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "optimizer_param_names": self._optimizer_param_names(),
        }
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
                        "manifest": self._dp.parameter_manifest(),
                    },
                    file,
                    indent=2,
                )
        ctx.barrier()

    # -- load -----------------------------------------------------------------

    def load_local_state(self, state_dir, metadata, ctx) -> None:
        """Restore this rank's share by parameter name.

        With an unchanged world size only this rank's file is read. Otherwise each
        rank scans every saved file, keeps the names its new ownership plan assigns
        to it, and drops the rest; peak host memory is one saved rank file plus this
        rank's own share. RNG and dataloader position are per-rank and cannot be
        re-partitioned; the shared resume path skips them when the layout changed.
        """
        saved_world_size = int(metadata["world_size"])
        if (ctx.rank, ctx.world_size) != (self._dp.rank, self._dp.world_size):
            raise RuntimeError("checkpoint context does not match parameter plan")
        manifest = metadata.get("manifest")
        resharding = saved_world_size != ctx.world_size
        self._dp.plan.validate_manifest(
            manifest, allow_reshard=resharding, saved_world_size=saved_world_size
        )
        rows = {row["name"]: row for row in manifest["parameters"]}
        wanted = set(self._dp.master)
        template = self._optimizer.state_dict()
        inner_templates = self._inner_states(template)
        masters: dict = {}
        residuals: dict = {}
        state_by_name: dict = {}

        for saved_rank in range(saved_world_size) if resharding else [ctx.rank]:
            saved = self._read_rank_file(state_dir, saved_rank)
            self._validate_rank_state(saved, saved_rank, rows)
            manager = saved["manager"]
            masters.update((n, v) for n, v in manager["master"].items() if n in wanted)
            residuals.update(
                (n, v)
                for n, v in (manager.get("param_sync_residuals") or {}).items()
                if n in wanted
            )
            # A rank only builds the inner optimizers it has parameters for, so the
            # inner index is not stable across ranks; the name is.
            for names, inner in zip(
                saved["optimizer_param_names"], self._inner_states(saved["optimizer"]),
                strict=True,
            ):
                for position, name in enumerate(names):
                    if name in wanted and position in inner["state"]:
                        state_by_name[name] = inner["state"][position]
            del saved

        self._dp.load_masters(masters, residuals)
        for names, inner in zip(self._optimizer_param_names(), inner_templates):
            inner["state"] = {
                position: state_by_name[name]
                for position, name in enumerate(names)
                if name in state_by_name
            }
        self._optimizer.load_state_dict(template)
        # The checkpoint also carries compute parameters, but the fp32 master is
        # the source of truth here. Publishing it after the load keeps a stale or
        # rounded model shard from surviving the resume.
        self._dp.publish_serially()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _validate_rank_state(saved, rank, rows):
        """Check one rank file against the shared manifest ``rows``, before any mutation."""
        master = saved["manager"]["master"]
        expected = {name for name, row in rows.items() if row["owner_rank"] == rank}
        if set(master) != expected:
            raise RuntimeError("saved master keys do not match manifest owners")
        for name, tensor in master.items():
            if list(tensor.shape) != rows[name]["shape"] or tensor.dtype != torch.float32:
                raise RuntimeError(f"invalid saved FP32 master: {name}")
        flattened = [name for group in saved["optimizer_param_names"] for name in group]
        if len(flattened) != len(set(flattened)) or set(flattened) != expected:
            raise RuntimeError("saved optimizer names do not match manifest owners")

    @staticmethod
    def _read_rank_file(state_dir, rank):
        """Load and validate one saved rank file."""
        rank_path = os.path.join(state_dir, f"rank_{rank}.pt")
        if not os.path.exists(rank_path):
            raise FileNotFoundError(
                f"custom_dp optimizer state for rank {rank} not found: {rank_path}"
            )
        saved = torch.load(rank_path, map_location="cpu", weights_only=False)
        if saved.get("backend") != BACKEND:
            raise RuntimeError(f"Unsupported rank-local ZeRO state in {rank_path}.")
        if saved.get("format_version") != FORMAT_VERSION:
            raise RuntimeError(f"Unsupported rank-local ZeRO format in {rank_path}.")
        # Every file this backend writes carries the names, which is what makes the
        # state re-partitionable; a file without them cannot be trusted.
        if "optimizer_param_names" not in saved:
            raise RuntimeError(f"rank-local ZeRO state has no optimizer names: {rank_path}")
        return saved

    @staticmethod
    def _inner_states(state_dict) -> list:
        """Return per-inner-optimizer state dicts for plain and combined optimizers."""
        if "optimizers" in state_dict:
            return state_dict["optimizers"]
        return [state_dict]

    def _optimizer_param_names(self) -> list:
        """Map each inner optimizer's parameter indices to parameter names.

        ``state_dict`` keys the per-parameter state by the index a parameter has
        in its optimizer's ``param_groups`` traversal. Those indices are
        owner-local, so they mean nothing to a run with a different world size.
        Recording the parameter name per index makes the state re-partitionable.
        """
        names_per_optimizer = []
        for child in getattr(self._optimizer, "optimizers", [self._optimizer]):
            names_per_optimizer.append(
                [
                    self._dp.name_for(parameter)
                    for group in child.param_groups
                    for parameter in group["params"]
                ]
            )
        return names_per_optimizer


__all__ = ["BACKEND", "FORMAT_VERSION", "RankLocalCheckpointIO"]
