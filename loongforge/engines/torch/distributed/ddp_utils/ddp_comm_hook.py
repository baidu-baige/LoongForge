# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DDP gradient communication hooks."""

import logging
from typing import Callable

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import (
    allreduce_hook,
    bf16_compress_hook,
    fp16_compress_hook,
)

from ..utils import is_rank_zero

logger = logging.getLogger(__name__)

# Comm hooks eligible to be resolved by ``resolve_comm_hook``. All of them
# share the ``(process_group, bucket) -> Future`` signature.
_SUPPORTED_COMM_HOOKS = {
    "allreduce_hook": allreduce_hook,
    "fp16_compress_hook": fp16_compress_hook,
    "bf16_compress_hook": bf16_compress_hook,
}


def resolve_comm_hook(hook: Callable | str, use_logging: bool = False) -> Callable:
    """Resolve a comm hook name/callable, optionally wrapped with logging.

    Args:
        hook: Either a DDP comm hook callable (e.g. ``allreduce_hook``,
            ``fp16_compress_hook``, ``bf16_compress_hook``) or its name.
        use_logging: If True, wrap the resolved hook so it logs bucket info
            on rank 0 before/after it runs.

    Returns:
        A comm hook with the ``(process_group, bucket)`` signature.
    """
    if isinstance(hook, str):
        try:
            hook = _SUPPORTED_COMM_HOOKS[hook]
        except KeyError:
            raise ValueError(
                f"Unsupported comm hook name {hook!r}, expected one of "
                f"{sorted(_SUPPORTED_COMM_HOOKS)}."
            ) from None

    if not use_logging:
        return hook

    hook_name = hook.__name__

    def logging_comm_hook(process_group, bucket):
        if is_rank_zero():
            tensor = bucket.buffer()
            logger.info(
                "DDP %s: bucket_index=%d numel=%d dtype=%s",
                hook_name,
                bucket.index(),
                tensor.numel(),
                tensor.dtype,
            )
        fut = hook(process_group, bucket)
        if is_rank_zero():
            fut.add_done_callback(
                lambda fut: logger.info(
                    "DDP %s done: bucket_index=%d", hook_name, bucket.index()
                )
            )
        return fut

    return logging_comm_hook
