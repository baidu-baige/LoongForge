# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Batch-scaling probe: is server-side batching (M3) worth building?

The M3 go/no-go question is not "is the GPU busy" but "where does a request's wall
time go, and how does the model's own compute scale with batch size". A serial
server is the binding resource, so making each request cheaper raises the ceiling
whether or not the GPU is saturated.

Two measurements, no eval-path changes:

1. **Split** -- profile a real ``predict_action`` and report wall vs CUDA kernel
   time (plus the top kernels). If host/overhead dominates, batching amortises it.
2. **Scaling** -- capture the real ``FastWAM.infer_action`` call, replay it with the
   batch dimension expanded to 1/2/4/8, and report the per-sample time. This uses
   the batch-general inner entry point, so it does not need the public
   ``predict_action`` batch contract (which asserts batch == 1).

Usage:
    python -m loongforge.embodied.eval.orchestrator.batch_probe \
        --config examples/embodied/fastwam/eval/configs/libero/libero_10.yaml \
        --calls 5 --batches 1,2,4,8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _build_model(config_path: str) -> Tuple[Any, Dict[str, Any]]:
    """Build the eval model exactly the way the policy server does.

    The camera layout lives on the *spec*, not the model (the server's warmup reads
    it from there too), and FastWAM's ``horizontal2`` layout rejects a single view.
    """
    from loongforge.embodied.eval.factories.registry import build_model_spec
    from loongforge.embodied.eval.servers.eval_server_config import parse_eval_server_config

    server_args, raw_model_dict = parse_eval_server_config(config_path)
    spec = build_model_spec(server_args, raw_model_dict)
    return spec.model, (getattr(spec, "metadata", {}) or {})


def _dummy_call(model: Any, metadata: Dict[str, Any]) -> None:
    """One representative call, shaped like the server's warmup."""
    n_views = max(1, int(metadata.get("num_camera_views", 1) or 1))
    state_dim = int(metadata.get("state_dim", 0) or 0) or (14 if n_views == 3 else 8)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    model.predict_action(
        images=[[image] * n_views],
        instructions=["pick up the object and place it in the basket"],
        state=np.zeros(state_dim, dtype=np.float32),
        dataset_stats=None,
    )


def _profile(model: Any, metadata: Dict[str, Any], calls: int) -> Dict[str, Any]:
    """Wall time vs CUDA kernel time for a real predict_action."""
    _dummy_call(model, metadata)  # warmup (lazy imports, cudnn autotune)
    torch.cuda.synchronize()
    walls: List[float] = []
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            start = time.perf_counter()
            _dummy_call(model, metadata)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - start) * 1000.0)
    cuda_ms = sum(event.device_time_total for event in prof.key_averages()) / 1000.0  # us -> ms
    top = sorted(
        ((event.key, event.device_time_total / 1000.0) for event in prof.key_averages() if event.device_time_total),
        key=lambda kv: -kv[1],
    )[:5]
    mean_wall = statistics.mean(walls)
    return {
        "wall_ms": round(mean_wall, 1),
        "cuda_ms": round(cuda_ms, 1),
        "host_ms": round(mean_wall - cuda_ms, 1),
        "gpu_share": round(cuda_ms / mean_wall, 3) if mean_wall else None,
        "top_kernels_ms": [(name, round(ms, 1)) for name, ms in top],
    }


def _capture_infer_call(model: Any, metadata: Dict[str, Any]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Run one predict_action and capture the kwargs it hands to FastWAM.infer_action."""
    from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM

    captured: Dict[str, Any] = {}
    original = FastWAM.infer_action

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not captured:
            captured["args"] = tuple(t.detach().clone() if torch.is_tensor(t) else t for t in args)
            captured["kwargs"] = {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
        return original(self, *args, **kwargs)

    FastWAM.infer_action = spy
    try:
        _dummy_call(model, metadata)
    finally:
        FastWAM.infer_action = original
    return captured["args"], captured["kwargs"]


def _expand(value: Any, batch: int) -> Any:
    """Tile a batch-1 tensor (or a prompt) up to batch; leave the rest alone."""
    if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == 1:
        return value.repeat(batch, *([1] * (value.ndim - 1)))
    if isinstance(value, str):
        return [value] * batch  # the text encoder takes a Sequence[str]
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return [value[0]] * batch
    return value


def _as_tensor(out: Any) -> Any:
    """Best-effort: pull the action tensor out of whatever infer_action returned."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for value in out.values():
            if torch.is_tensor(value):
                return value
    if isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
        return out[0]
    return None


def _scaling(model: Any, metadata: Dict[str, Any], batches: List[int]) -> List[Dict[str, Any]]:
    """Replay the captured inner call with an expanded batch and time per sample."""
    from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM

    args, kwargs = _capture_infer_call(model, metadata)
    core = model.core
    rows: List[Dict[str, Any]] = []
    for batch in batches:
        big_args = tuple(_expand(a, batch) for a in args)
        big_kwargs = {k: _expand(v, batch) for k, v in kwargs.items()}
        try:
            out = FastWAM.infer_action(core, *big_args, **big_kwargs)  # warmup
            torch.cuda.synchronize()
            # Sanity: did the batch actually flow through, or did the model quietly
            # process one sample? K identical inputs with per-sample noise must give
            # K different actions, and the leading dim must be K.
            actions = _as_tensor(out)
            shape = tuple(actions.shape) if actions is not None else None
            differ = bool(actions is not None and actions.shape[0] >= 2 and not torch.allclose(actions[0], actions[1]))
            start = time.perf_counter()
            for _ in range(3):
                FastWAM.infer_action(core, *big_args, **big_kwargs)
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) / 3 * 1000.0
        except Exception as exc:  # the inner path may still not be batch-general
            import traceback

            frames = traceback.extract_tb(exc.__traceback__)
            where = f"{frames[-1].filename.split('/')[-1]}:{frames[-1].lineno}" if frames else "?"
            rows.append({"batch": batch, "error": f"{type(exc).__name__} at {where}: {exc}"})
            continue
        rows.append(
            {
                "batch": batch,
                "total_ms": round(total_ms, 1),
                "per_sample_ms": round(total_ms / batch, 1),
                "out_shape": shape,
                "samples_differ": differ,
            }
        )
    return rows


def _equivalence(
    model: Any,
    metadata: Dict[str, Any],
    seeds: List[int],
    batch_size: Optional[int] = None,
    same_seed: bool = False,
) -> Dict[str, Any]:
    """Compare batch=K with K separate batch=1 calls.

    Two modes:

    * ``same_seed=True`` (the controlled batching test): every reference and
      every batch row uses the same seed. With the same captured input this
      isolates the batched compute path: references are identical, batch rows
      should also be identical, and the only possible difference is the batched
      GEMM reduction order.
    * ``same_seed=False`` (per-row seed / fake-batch guard): row ``i`` uses
      ``seeds[i]`` in both arms. If the rows differed from their own reference
      but matched another row's reference, the batch routing would be wrong.
    """
    from loongforge.embodied.model.fastwam.mot.fastwam import FastWAM

    batch = int(batch_size or len(seeds))
    if same_seed:
        row_seeds = [int(seeds[0])] * batch
    else:
        row_seeds = [int(seed) for seed in seeds]
        if len(row_seeds) != batch:
            raise ValueError(
                f"per-row seed count {len(row_seeds)} must match batch size {batch}"
            )

    args, kwargs = _capture_infer_call(model, metadata)
    core = model.core
    references = []
    for row_seed in row_seeds:
        call = dict(kwargs)
        call["seed"] = row_seed
        out = FastWAM.infer_action(core, *args, **call)
        action = _as_tensor(out)
        references.append(action[0] if action.ndim == 3 else action)

    batch_kwargs = {
        key: _expand(value, batch)
        for key, value in kwargs.items()
        if key != "seed"
    }
    batch_kwargs["seed"] = row_seeds
    batch_args = tuple(_expand(value, batch) for value in args)
    batched = _as_tensor(FastWAM.infer_action(core, *batch_args, **batch_kwargs))

    per_row = [float((batched[index] - references[index]).abs().max()) for index in range(batch)]
    cross = [
        float((batched[i] - references[j]).abs().max())
        for i in range(batch)
        for j in range(batch)
        if i != j
    ]
    reference_spread = max(
        (
            float((references[i] - references[j]).abs().max())
            for i in range(batch)
            for j in range(i + 1, batch)
        ),
        default=0.0,
    )
    batch_spread = max(
        (
            float((batched[i] - batched[j]).abs().max())
            for i in range(batch)
            for j in range(i + 1, batch)
        ),
        default=0.0,
    )
    return {
        "same_seed": same_seed,
        "batch_size": batch,
        "seeds": row_seeds,
        "out_shape": tuple(batched.shape),
        "per_row_max_abs_diff": per_row,
        "max_abs_diff": max(per_row),
        "reference_spread": reference_spread,
        "batch_row_spread": batch_spread,
        "batch_rows_identical": batch_spread == 0.0,
        "min_cross_seed_diff": min(cross) if cross else None,
        # Only meaningful in per-row-seed mode: identical rows are expected when
        # the same seed is broadcast to the whole batch.
        "rows_distinct": bool((not same_seed) and cross and min(cross) > 0),
    }


def main() -> None:
    """Run main."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="a server eval config (server.ckpt_path etc.)")
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument(
        "--equivalence",
        action="store_true",
        help="compare batch=K with per-row seeds against K batch=1 calls",
    )
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="rows in the equivalence batch (default: number of seeds)",
    )
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="use one seed for every reference and batch row (controlled batching test)",
    )
    args = parser.parse_args()

    model, metadata = _build_model(args.config)
    if args.equivalence:
        seeds = [int(seed) for seed in args.seeds.split(",") if seed]
        print(
            json.dumps(
                _equivalence(
                    model,
                    metadata,
                    seeds,
                    batch_size=args.batch_size or None,
                    same_seed=args.same_seed,
                ),
                indent=2,
            )
        )
        return
    split = _profile(model, metadata, args.calls)
    scaling = _scaling(model, metadata, [int(b) for b in args.batches.split(",") if b])
    print(json.dumps({"split": split, "scaling": scaling}, indent=2))


if __name__ == "__main__":
    main()
