# Parallel Evaluation

This page describes the multi-GPU evaluation orchestration (multiple sim workers across
GPUs, one inference service per GPU) and the switches that make an evaluation faster.

> Measured speedups quoted on this page are end-to-end numbers from the reference
> setup (8×A800-80GB): LIBERO with FastWAM, RoboTwin with pi05. They depend on model,
> benchmark, checkpoint and protocol. Re-measure on your machine with
> `orchestrator/calibration/calibrate.py` (see [Calibration](#5-calibration-trial-runs)).

## 1. Two entry points

The evaluator picks the path from the config:

| Config contains | Path | Behaviour |
|---|---|---|
| none of `coordinator` / `servers` / `pools` | single run | One inference service and one sim worker for the whole config (the original entry) |
| `servers` and `pools` | coordinated | One inference service per GPU, `n_sims_per_server` sim workers per GPU, global FIFO queue |

Both paths use the same runners and the same output schema, so an existing
`examples/embodied/*/eval/configs/*.yaml` keeps working unchanged.

## 2. Configuration reference

| Key | Default | Effect |
|---|---|---|
| `servers[].gpus` | — | GPUs that host one inference service each |
| `pools.gpu_sim.n_sims_per_server` | 1 | Sim workers per GPU (`N`) — **GPU-sim families**: RoboTwin, SimplerEnv, ManiSkill |
| `pools.cpu_sim.n_sims_per_server` | 1 | Sim workers per GPU (`N`) — **CPU-sim families**: LIBERO, CALVIN |
| `pools.gpu_sim.placement` | `co_locate` | GPU-sim only. `co_locate`: sims share the GPU with their inference service. `separate`: sims use `sim_gpus`, the service cards stay dedicated |
| `coordinator.n_workers` | 1 | Fallback `N` when the family's pool section is absent |
| `run.unit_timeout_sec` | `max_episodes × timeouts.per_episode_sec` | Wall-clock backstop for one unit. Set it explicitly whenever `per_episode_sec` is large (a sensor-model batch can need 7200 s), otherwise a unit that hangs in native code blocks its slot for tens of hours |
| `benchmark.episodes_per_chunk` | computed: ~4 units per slot (`ceil(E / (4·S))`) | Episodes per eval unit (one worker process) |
| `benchmark.plan_order` | `task` | `lpt` dispatches the longest units first; `lpt_long` keeps every shard of the long tasks ahead of all short work (see Section 4) |
| `benchmark.long_task_steps` | unset | Step limit that marks a task as "long" for `lpt_long` (e.g. `1200`); `-1`/unset disables the split |
| `benchmark.long_task_episodes_per_chunk` | unset | Finer chunk for the long tasks only (e.g. `10`), which shortens the last unit's tail |
| `benchmark.sim_cpu_threads` | unset | (RoboTwin only) Caps `OMP/MKL/OpenBLAS` + `torch` threads per sim worker. `1` removes thread oversubscription when many sims share a card |

The pool key must match the benchmark family: LIBERO and CALVIN read `pools.cpu_sim`,
the GPU-sim families read `pools.gpu_sim`. A pool written under the other key is ignored
(the coordinator prints a warning) and `N` falls back to `coordinator.n_workers or 1`,
which only shows up as "the run is slower than the calibration table says".

`N` is per service, so the number of concurrent sims is `len(servers[].gpus) × N`
(e.g. `N=48` on 8 GPUs = 384 sims). `N` and `episodes_per_chunk` are the two knobs that
matter most; see [Section 4](#4-choosing-n-and-episodes_per_chunk).

Server-side batching (`server.batching`, below) is the third knob, and the only one that
raises the ceiling for an inference-bound model:

| Key | Default | Effect |
|---|---|---|
| `server.batching.enabled` | `true` when the block is non-empty | Master switch |
| `server.batching.max_batch_size` | 8 | Requests grouped into one model call |
| `server.batching.max_wait_ms` | 5 | How long the batcher waits to fill a batch (adds that latency to every request) |

The batcher only groups requests that are already queued, so a large `max_wait_ms` does
not help when the arrival rate is low — watch the `Batcher stats: ... mean_batch=` line
in the service log. On FastWAM, batch 8 is worth 7.35× per sample in isolation and
`mean_batch` settles at 2.5–3.0 in the eval (~3× throughput); see
[the FastWAM section](benchmarks/robotwin.md#fastwam-batching-is-the-lever-n-is-not).

## 3. Chunk execution

Three switches control how action chunks are consumed:

| Switch | Default | Meaning |
|---|---|---|
| `benchmark.return_action_chunk` | `false` (LIBERO: `true`) | `false`: the inference service caches the chunk and returns one action per env step. `true`: the client receives the whole chunk and executes it locally |
| `benchmark.chunk_execute_steps` | 0 (all) | How many actions of a returned chunk are executed before re-planning |
| `benchmark.skip_intra_chunk_obs` | `false` | (RoboTwin only) Inside a chunk, build a proprio-only observation instead of rendering the cameras. Camera rendering is the dominant sim cost, so this is what makes long episodes much cheaper |

With `skip_intra_chunk_obs` the evaluation becomes open-loop inside a chunk: the policy
sees fresh observations only every `chunk_execute_steps` steps. This is the same cadence
as the original server-side chunk cache, but the observation path differs (no camera
frames inside the chunk), so per-episode results are **not** bit-identical to a
per-step closed-loop run — see [Result alignment](#8-result-alignment).

## 4. Choosing N and episodes_per_chunk

Let `S = G × N` be the number of slots and `E` the number of episodes in the run.

```
c* = max(1, floor(E / S))
```

The rule keeps the number of eval units close to the number of slots: one wave, one
process start per unit, no idle slots. If `episodes_per_chunk` is left unset, the planner
targets about four units per slot (`ceil(E / (4·S))`) so that late waves can refill freed
slots; set it explicitly to `c*` when you know the whole run fits in one wave. Two
corrections:

- heterogeneous tasks: keep `E / (S·c) ≥ 2` so LPT has units to refill freed slots;
- repair granularity: a larger `c` re-runs more episodes when a unit fails.

In a single-task run with `E ≤ S` (e.g. 50 episodes on 64 slots), `c = 1` and `N`
above `ceil(E/G)` changes nothing — the number of concurrent sims is capped by `E`.

### Long-tail correction (`lpt_long`)

With mixed task lengths the makespan is set by the last shard of the longest task, not by
the average: a 1,700-step task takes ~200 s per episode, so one 25-episode unit is ~83 min.
`lpt` sorts tasks but then interleaves, which puts the long tasks' later shards deep in the
queue. `lpt_long` splits the queue into two round-robins — every shard of the long tasks
first, then everything else — and can give only those tasks a finer chunk:

```yaml
benchmark:
  plan_order: lpt_long
  long_task_steps: 1200              # tasks at or above this official step limit
  long_task_episodes_per_chunk: 10   # 25 -> 10 shortens the last unit from 83 to 33 min
```

Measured on 50 RoboTwin tasks × 100 episodes (N=10, 8×A800, same seeds, one run each):

| plan_order | chunk (long) | wall clock | speedup |
|---|---|---|---|
| `lpt` (previous default) | 25 | 1.88 h | 21.8× |
| `lpt_long`, `long_task_steps=1200` | 10 | **1.77 h** | **23.2×** |

Raising `long_task_steps` (fewer tasks in the head) keeps more slots free for short tasks
in the first wave and avoids stacking many heavy scenes on one card; keep
`long_task_episodes_per_chunk ≥ episodes_per_task / 4` so repair rounds stay cheap.

## 5. Calibration (trial runs)

A trial run measures throughput at several `N` values before committing to a full
evaluation. Use the calibration entry point:

```bash
python -m loongforge.embodied.eval.orchestrator.calibration.calibrate \
    --config run.yaml --n 6,8,10,12 --gpus 0,1,2,3,4,5,6,7 --out /tmp/calib
```

Read the curve with two metrics:

| Metric | What it includes | When to use |
|---|---|---|
| wall throughput (`N × episodes / unit wall`) | model load is excluded, **per-unit process startup is included** | small runs where each slot executes one unit (a single wave) |
| steady throughput (`N / rollout per episode`) | rollout only | full runs where each slot executes many episodes |

Process startup dominates small units (on RoboTwin a unit spends ~150–340 s starting
torch/SAPIEN and building the first scene, versus ~20 s of rollout per episode), so the
two metrics can differ by several times. Pick `N` on the metric that matches the run you
are about to do.

Benchmark-specific calibration data and the recommended settings for the reference
environment are in the [LIBERO](benchmarks/libero.md#calibration-results) and
[RoboTwin](benchmarks/robotwin.md#calibration-results) pages.

## 6. Outputs and validation

The coordinated path writes `coordinator_result.json` next to the per-unit outputs:

| Field | Meaning |
|---|---|
| `n_units`, `n_units_launched` | Planned units and units actually dispatched (repairs included) |
| `elapsed_sec` | Worker phase; model loading is excluded |
| `repair_rounds`, `failed_units` | Coverage repairs (up to 3 rounds) and units still failing |
| `validation.ok` | `true` only when every planned episode has exactly one record |
| `validation.infra_failed` | Episodes that failed at the infrastructure level (they did not run) |

`run.py` exits with code `2` when `validation.ok` is false or a service could not be
restarted, so a pipeline never reads a partial score as a success.

## 7. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| CUDA OOM when the wave starts | `N` too high for the per-sim memory of that benchmark | Lower `N`; failed units are retried automatically, but a systematic overflow needs a smaller `N` |
| `validation.ok=false` after 3 repair rounds | Units that consistently fail | Inspect `failed_units`; check the per-unit log and the inference service |
| Long gap before the first episode | Model load and first scene build | Expected; it is excluded from `elapsed_sec` but included in the wall clock |
| Slots idle in the second half | Mixed task lengths | Use `plan_order: lpt_long` with `long_task_steps` + `long_task_episodes_per_chunk` |

## 8. Result alignment

Acceleration changes the evaluated set and/or the execution path, so per-episode results
are **not bit-identical** to a vanilla run:

- **Seed/scene set**: RoboTwin skips unstable scenes and continues with later seeds;
  LIBERO's `per_task` scene is `(seed, reset count in the process)`, so sharding without
  scene snapshots changes the scene.
- **Chunk execution**: `return_action_chunk` + `skip_intra_chunk_obs` re-plan and
  re-observe every `chunk_execute_steps` steps instead of every step.
- **Batching**: batch composition changes reduction order; differences are noise-level.

For large evaluations (≥50–100 episodes per task or suite) these differences behave like
sampling noise: the aggregate success rate stays inside the binomial interval (about
±7pt at 1σ for n=50, ±5pt for n=100). Do not use single-episode or few-episode runs for
bit-level regression comparisons.

To stay as close to vanilla as possible, disable the corresponding switches:
`return_action_chunk: false`, `skip_intra_chunk_obs: false`, `env_seed_mode: per_episode`
(or `per_task` with scene snapshots), one worker per task, batching off.

## 9. Compatibility

- Existing `examples/embodied/*/eval/*.sh` scripts and configs are unchanged.
- Defaults are unchanged except LIBERO's `return_action_chunk`, which defaults to `true`
  on this branch (it was validated against the legacy cache path over 80 episodes: no
  success flips, 79/80 identical step counts, ~9.8× fewer requests).
- For RoboTwin the accelerated switches are **opt-in**: use
  `configs/robotwin/robotwin_pi05_parallel.yaml`. Per-episode equivalence with the legacy
  path has not been established, so the defaults stay at the legacy behaviour.
