# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Wall-X under the Apache-2.0 License.

"""Base class for operator proxies with lazy backend resolution."""

import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


# Keep registration explicit and deterministic for cross-rank diagnostics.
_PROXY_REGISTRY = OrderedDict()


def register_proxy(name, proxy):
    """Register a process-wide Wall-OSS operator proxy by diagnostic name."""
    if name in _PROXY_REGISTRY and _PROXY_REGISTRY[name] is not proxy:
        raise RuntimeError(f"Duplicate Wall-OSS operator proxy name: {name}")
    _PROXY_REGISTRY[name] = proxy
    return proxy


def backend_inventory():
    """Resolve and return the current backend for every registered proxy."""
    return {name: proxy.backend for name, proxy in _PROXY_REGISTRY.items()}


def log_backend_inventory(rank=0, world_size=1):
    """Log resolved backends and verify that all distributed ranks agree."""
    inventory = backend_inventory()
    encoded = ";".join(f"{name}={backend}" for name, backend in inventory.items())
    logger.warning(
        "[WallOpsBackendInventory] rank=%d/%d %s",
        rank,
        world_size,
        encoded or "<empty>",
    )
    if world_size <= 1:
        return inventory
    try:
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            logger.warning(
                "[WallOpsBackendInventory] distributed context unavailable; "
                "rank consistency was not checked"
            )
            return inventory
        gathered = [None] * world_size
        dist.all_gather_object(gathered, inventory)
        if any(item != gathered[0] for item in gathered[1:]):
            raise RuntimeError(
                "Wall-OSS operator backend mismatch across ranks: "
                + repr(gathered)
            )
        logger.warning(
            "[WallOpsBackendInventory] rank consistency verified across %d ranks",
            world_size,
        )
    except Exception:
        logger.exception("[WallOpsBackendInventory] rank consistency check failed")
        raise
    return inventory


class OpsProxy:
    """Callable proxy that lazily resolves to the best available backend.

    Two-level fallback:
      Level 1: CUDA inline kernel (compiled ``_cuda_ext_bin``, requires CUDA)
      Level 2: Pure PyTorch (always available)

    Subclasses define ``_pytorch_fallback`` and optionally override
    ``_get_cuda_kernel`` to return a callable when the CUDA extension is
    importable. When no CUDA kernel is available, the proxy silently falls
    back to the PyTorch implementation, so callers can use the same import
    regardless of whether the extension has been built.
    """

    def __init__(self, name=None):
        """Initialize a lazy operator proxy."""
        self.name = name or self.__class__.__name__
        self._resolved_fn = None
        self._backend = None
        self._resolve_lock = threading.Lock()
        self._call_logged = False
        register_proxy(self.name, self)

    def _resolve(self):
        """Resolve the best available backend. Called once on first use (thread-safe)."""
        if self._resolved_fn is not None:
            return
        with self._resolve_lock:
            # Double-check after acquiring lock
            if self._resolved_fn is not None:
                return

            # Level 1: try CUDA inline kernel (subclass override)
            cuda_fn = self._get_cuda_kernel()
            if cuda_fn is not None:
                self._backend = "cuda_inline"
                self._resolved_fn = cuda_fn
                return

            # Level 2: PyTorch fallback
            self._backend = "pytorch"
            self._resolved_fn = self._pytorch_fallback
            logger.warning(
                "%s: CUDA backend unavailable; using explicit PyTorch fallback",
                self.name,
            )

    def _get_cuda_kernel(self):
        """Override in subclass to provide the CUDA kernel. Returns None if unavailable."""
        return None

    def _pytorch_fallback(self, *args, **kwargs):
        """Implement the operator using the PyTorch fallback backend."""
        raise NotImplementedError(f"{self.__class__.__name__} has no PyTorch fallback")

    def __call__(self, *args, **kwargs):
        """Invoke the lazily resolved operator backend."""
        if self._resolved_fn is None:
            self._resolve()
        if not self._call_logged:
            with self._resolve_lock:
                if not self._call_logged:
                    logger.warning(
                        "[WallOpsBackendCall] op=%s backend=%s",
                        self.name,
                        self._backend,
                    )
                    self._call_logged = True
        return self._resolved_fn(*args, **kwargs)

    @property
    def backend(self) -> str:
        """Return the name of the currently selected backend."""
        if self._resolved_fn is None:
            self._resolve()
        return self._backend

    # ------------------------------------------------------------------
    # Multi-backend API (for testing / benchmarking)
    # ------------------------------------------------------------------

    def available_backends(self):
        """Return list of available backend names for this operator."""
        backends = ["pytorch"]
        if self._get_cuda_kernel() is not None:
            backends.append("cuda_inline")
        return backends

    def _get_backend_fn(self, backend):
        """Return the callable for a specific backend, or None if unavailable."""
        if backend == "pytorch":
            return self._pytorch_fallback
        elif backend == "cuda_inline":
            return self._get_cuda_kernel()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def call_with_backend(self, backend, *args, **kwargs):
        """Call this operator using a specific backend.

        This does NOT change the default backend used by ``__call__``.
        The ``backend`` property still reflects the auto-resolved default.

        Args:
            backend: One of "cuda_inline", "pytorch".

        Returns:
            Operator output from the specified backend.

        Raises:
            RuntimeError: If the requested backend is not available.
        """
        fn = self._get_backend_fn(backend)
        if fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: backend '{backend}' not available. "
                f"Available: {self.available_backends()}"
            )
        return fn(*args, **kwargs)

    def __repr__(self):
        """Return a concise representation containing the backend name."""
        backend = self._backend or "unresolved"
        return f"<{self.__class__.__name__} backend={backend}>"
