# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Configuration shared by every FSDP group created during one wrap pass."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from ..context import DistributedContext
from .builders import build_fsdp_device_mesh, build_ignored_params, build_mp_policy


@dataclass
class FSDPWrapContext:
    """FSDP wrapping configuration.

    ``fsdp_kwargs`` holds the arguments passed verbatim to ``fully_shard`` for
    every group, so mesh, mixed precision policy and ignored params are built
    once per wrap pass rather than per group. The reshard fields are resolved
    per group instead (see ``resolve_reshard_policy``).
    """

    fsdp_kwargs: dict
    reshard_default: bool | int | None
    reshard_overrides: dict[str, bool | int]
    no_wrap_classes: set[str] = field(default_factory=set)

    @classmethod
    def create(cls, model: nn.Module, training_args, ctx: DistributedContext):
        """Build the context for one wrap pass, doing all one-time work up front.

        Args:
            model: Model about to be wrapped, still raw. Only inspected here, but
                ``build_ignored_params`` moves its 0-dim parameters onto the
                compute device as a side effect, so this is not a pure query.
            training_args: Supplies the reshard policy (default plus per-class
                overrides) and ``fsdp_no_wrap_modules``.
            ctx: Distributed context used for mesh construction and device
                placement.

        Returns:
            Context whose ``fsdp_kwargs`` is safe to shallow-copy per group —
            ``fully_shard_unit`` does exactly that before adding
            ``reshard_after_forward``, so the shared mesh, policy and ignored set
            stay identical across groups.

        Note:
            Collective: mesh construction must be reached by all ranks with the
            same arguments. Build this once per pass rather than per group; a
            second call would create a second device mesh for the same ranks.
        """
        fsdp_kwargs = {
            "mesh": build_fsdp_device_mesh(training_args, ctx),
            "mp_policy": build_mp_policy(training_args, model),
            "ignored_params": build_ignored_params(model, ctx),
        }

        return cls(
            fsdp_kwargs=fsdp_kwargs,
            reshard_default=training_args.fsdp_reshard_default,
            reshard_overrides=training_args.fsdp_reshard_module_overrides or {},
            no_wrap_classes=set(training_args.fsdp_no_wrap_modules or []),
        )

    @property
    def ignored_params(self) -> set:
        """Return parameters excluded from FSDP sharding."""
        return self.fsdp_kwargs["ignored_params"]
