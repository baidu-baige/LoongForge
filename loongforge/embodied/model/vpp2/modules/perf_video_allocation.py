# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in empty-fill suppression for the serial frozen eager video producer.

This changes a process-global PyTorch allocation policy temporarily. Supported
only on the single-main-thread producer, before native action execution, with
no concurrent custom tensor-producing threads/streams. Native operators retain
their original output contracts; the fused RoPE fully writes its empty output.
Deterministic algorithms, precision and mathematical operators are unchanged.
"""
import threading

import torch
from torch.utils import deterministic

_lock = threading.Lock()
_records = {}


def diagnostic_snapshot():
    """Exit-only task diagnostics; never synchronizes or touches tensors."""
    return [dict(record) for record in _records.values()]


def scoped_prefill(mot, kwargs):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Scoped video allocation requires the main training thread')
    if (torch.is_grad_enabled() or not torch.are_deterministic_algorithms_enabled()
            or not deterministic.fill_uninitialized_memory):
        raise RuntimeError('Scoped video allocation requires no_grad and the original deterministic fill policy')
    if (mot.perf_video_execution != 'eager' or mot.perf_action_execution != 'native_graph'
            or not mot.perf_video_rope_fused
            or mot.use_video_tokens or mot.use_fixed_video_layers is not None):
        raise ValueError('Scoped video allocation requires the validated serial eager-video/native-action path')
    if any(kwargs[key] is not None for key in [
        'max_layers', 'video_feature_callback',
    ]):
        raise ValueError('Scoped video allocation does not support callbacks or partial prefill')
    video = mot.mixtures['video']
    tokens = kwargs['video_tokens']
    if (video.training or tokens.device.type != 'cuda' or tokens.dtype != torch.bfloat16
            or any(p.requires_grad or p.device != tokens.device for p in video.parameters())):
        raise ValueError('Scoped video allocation requires a frozen BF16 video expert on one CUDA device')
    if (torch.cuda.is_current_stream_capturing()
            or torch.cuda.current_stream(tokens.device) != torch.cuda.default_stream(tokens.device)
            or torch.compiler.is_compiling()):
        raise RuntimeError('Scoped video allocation requires the uncaptured default stream')
    if not _lock.acquire(blocking=False):
        raise RuntimeError('Concurrent scoped video allocation is unsupported')
    before = deterministic.fill_uninitialized_memory
    try:
        # Keep only scalar diagnostics: neither retain the model through exit
        # nor lose the record if the trainer releases its model before atexit.
        if '_perf_video_fill_subject_id' not in mot.__dict__:
            mot.__dict__['_perf_video_fill_subject_id'] = len(_records)
        key = mot.__dict__['_perf_video_fill_subject_id']
        mot.__dict__['_perf_video_fill_calls'] = mot.__dict__.get('_perf_video_fill_calls', 0) + 1
        mot.__dict__['_perf_video_fill_active'] = True
        _records[key] = {'calls': mot.__dict__['_perf_video_fill_calls'],
                         'restores': mot.__dict__.get('_perf_video_fill_restores', 0), 'active': True}
        deterministic.fill_uninitialized_memory = False
        return mot._prefill_video_cache_eager(**kwargs)
    finally:
        deterministic.fill_uninitialized_memory = before
        mot.__dict__['_perf_video_fill_active'] = False
        mot.__dict__['_perf_video_fill_restores'] = mot.__dict__.get('_perf_video_fill_restores', 0) + 1
        if '_perf_video_fill_subject_id' in mot.__dict__:
            _records[mot.__dict__['_perf_video_fill_subject_id']] = {
                'calls': mot.__dict__['_perf_video_fill_calls'],
                'restores': mot.__dict__['_perf_video_fill_restores'], 'active': False}
        _lock.release()
