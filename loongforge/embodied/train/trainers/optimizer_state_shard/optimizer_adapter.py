# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Bridge between the optimizer and the optimizer-state-shard runtime.

The runtime must not know that Muon exists. What it needs is two capabilities:
map an updated ``Parameter`` object back to a registry name, and know which
masters an optimizer finishes inside ``step()`` rather than at the end. Both are
expressed here.
"""

from __future__ import annotations

from loongforge.embodied.train.trainers.optimizer_state_shard.registry import (
    MasterParameterView,
)


class OptimizerAdapter:
    """Translate optimizer-side parameter updates into registry names."""

    def __init__(self, registry):
        """Bind to a registry and index its masters by object identity."""
        self._registry = registry
        self._name_by_id: dict[int, str] = {}
        self._in_step_names: set[str] | None = None
        self.refresh()

    def refresh(self):
        """Rebuild the identity index after the masters are (re)created."""
        self._name_by_id = {
            id(parameter): name for name, parameter in self._registry.master.items()
        }

    def optimizer_view(self):
        """Return a module-like view over the fp32 master parameters."""
        return MasterParameterView(self._registry.master.items())

    def name_for(self, parameter):
        """Return the registry name of ``parameter``, or ``None`` if unmanaged."""
        return self._name_by_id.get(id(parameter))

    def set_in_step_names(self, names):
        """Record which masters the optimizer updates during ``step()``.

        Masters an optimizer finishes early can start publishing before
        ``step()`` returns; the rest must wait for it. Without this information
        every master is treated as publishable early, which is correct but
        serializes the launch queue behind the slowest one.
        """
        self._in_step_names = None if names is None else set(names)

    @property
    def in_step_names(self):
        """Return the in-step-update name set, or ``None`` if unknown."""
        return self._in_step_names

    def updates_in_step(self, record) -> bool:
        """True when ``record``'s master is expected to be updated in-step."""
        if self._in_step_names is None:
            return True
        return record.name in self._in_step_names

    def supports_update_callback(self, optimizer) -> bool:
        """True when ``optimizer`` can report each parameter update as it lands."""
        return hasattr(optimizer, "param_update_callback")

    def attach_update_callback(self, optimizer, callback) -> bool:
        """Install ``callback`` on ``optimizer``; return whether it was accepted.

        Publication correctness never depends on this: the drain at the end of the
        step publishes anything the callback did not.
        """
        if not self.supports_update_callback(optimizer):
            return False
        optimizer.param_update_callback = callback
        return True


__all__ = ["OptimizerAdapter"]
