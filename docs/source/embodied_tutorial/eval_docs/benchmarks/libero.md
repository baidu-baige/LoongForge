# LIBERO Evaluation

LIBERO is a tabletop robot manipulation benchmark with 4 task suites (Spatial, Object, Goal, Long Horizon), totaling 40 tasks, evaluated on a Franka arm. This page is a step-by-step reproduction guide for running LIBERO evaluation through the LoongForge eval module.

Five models are verified on LIBERO: **pi05**, **xvla**, **GR00T-N1.6** (public weights), **LingBot-VA** (weights trained with LoongForge from `lingbot-va-base`), and **FastWAM** (weights trained with LoongForge).

## Step 0: Download weights

| Model | Weights |
|---|---|
| pi05 | [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) (`model.safetensors` + `dataset_statistics.json`) |
| xvla | [2toINF/X-VLA-LIBERO](https://huggingface.co/2toINF/X-VLA-LIBERO) |
| GR00T-N1.6 | [0xAnkitSingh/GR00T-N1.6-LIBERO](https://huggingface.co/0xAnkitSingh/GR00T-N1.6-LIBERO) (weight dir + Eagle3 processor + `libero_panda` stats) |
| LingBot-VA | [robbyant/lingbot-va-posttrain-libero-long](https://huggingface.co/robbyant/lingbot-va-posttrain-libero-long) (one root with `transformer/` + `vae/` + `text_encoder/` + `tokenizer/`; libero_10 only) |
| FastWAM | [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM#inference-with-released-checkpoints) checkpoint + [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) (T5/VAE + `google/umt5-xxl` tokenizer) |

## Step 1: Environment setup

### Standard environment

Install the base LIBERO environment following the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) instructions, then install the additional dependencies the eval client needs:

```bash
pip install websockets msgpack pyyaml
pip install numpy==1.24.4   # downgrade numpy for simulator compatibility
```

⚠️ Common issues:

- LIBERO historically defaults to Python 3.8; the syntax differences up to 3.10 are substantial. Python 3.10 avoids many issues.
- `numpy==1.24.4` is pinned for compatibility with the simulation environment — do not upgrade.
- The LIBERO runner uses MuJoCo with offscreen GL. Keep `MUJOCO_GL=osmesa` / `PYOPENGL_PLATFORM=osmesa` (set by the shipped run scripts).

The benchmark dataset is downloaded following the official LIBERO data instructions; the eval module does not repackage it.

## Step 2: Run evaluation

Run from inside the **benchmark** environment. The run scripts and eval YAMLs ship with `/path/to/...` placeholders — fill them in before running:

```bash
cd /path/to/LoongForge-VLA

# pi05
examples/embodied/pi05/eval/run_libero_eval.sh

# xvla
examples/embodied/xvla/eval/run_libero_eval.sh

# GR00T-N1.6
examples/embodied/gr00t_n1_6/eval/run_libero_eval.sh

# LingBot-VA (default config: libero_10_full.yaml, 10 tasks x 50 episodes)
examples/embodied/lingbot_va/eval/run_libero_eval.sh

# FastWAM (default config: libero_spatial.yaml; set CONFIG for another suite)
examples/embodied/fastwam/eval/run_libero_eval.sh
```

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CONFIG` | eval YAML config path | `<model>/eval/configs/libero/object_smoke.yaml` (xvla: `libero_weight_object_smoke.yaml`) |
| `REPO_ROOT` | LoongForge-VLA repo root | `/path/to/LoongForge-VLA` |
| `BENCHMARK_PYTHON` | LIBERO env interpreter | `/path/to/libero/bin/python` |
| `CUDA_VISIBLE_DEVICES` | GPU id for the policy server | `0` |
| `MUJOCO_GL` / `PYOPENGL_PLATFORM` | MuJoCo offscreen GL backend | `osmesa` |
| `LD_LIBRARY_PATH` | NVIDIA libs (xvla script) | `/path/to/nvidia_lib:/usr/lib64` |

Key config fields (see `examples/embodied/<model>/eval/configs/libero/*.yaml`):

- `benchmark.suite` — `libero_object` | `libero_spatial` | `libero_goal` | `libero_10`
- `benchmark.max_tasks` / `benchmark.episodes_per_task` — raise for full-suite / multi-episode eval (e.g. `max_tasks: 0`, `episodes_per_task: 10`)
- `benchmark.continuous_gripper` — `true` for pi05 continuous gripper
- `benchmark.control_mode` — LIBERO OSC: `auto` | `absolute` | `delta`; xvla uses `absolute`
- pi05: `model.action_dim: 7`, `model.action_horizon: 50`, plus a matching `server.dataset_statistics_path`
- xvla: `model.domain_id: 3` (or omit to auto-resolve), `server.chunk_execute_steps: 10`, recommended `benchmark.max_steps: 800`
- LingBot-VA: use `libero_10_full.yaml` as shipped (official LIBERO client protocol: seed 7, `max_steps: 800`, `num_steps_wait: 5`, native 128 rendering); see [LingBot-VA quick start](../../quick_start_lingbot_va.md)
- GR00T-N1.6: `model.model_type: Gr00tN1d6`, `benchmark.control_mode: delta`, `server.embodiment_tag: libero_panda`, plus `model.base_model_path` / `model.model_name` (Eagle3 processor)
- FastWAM: one YAML per suite; `model.variant: uncond` for LoongForge-trained checkpoints, `benchmark.episodes_per_task: 50`, `benchmark.controller_force_update: false`; set `DIFFSYNTH_MODEL_BASE_PATH` to the directory containing `Wan-AI/Wan2.2-TI2V-5B`

## Verification

Task-success status:

| Model | LIBERO status | Notes |
|---|---|---|
| pi05 | ✅ task success | no full-suite regression run; verified via single-episode smoke only |
| xvla | ✅ task success | libero_object 94/100 (10 tasks × 10 eps); per-task spread 8/10 … 10/10 |
| GR00T-N1.6 | ✅ task success | 10 tasks × 10 eps per suite: object 100/100, spatial 95/100, goal 95/100, libero_10 89/100 |
| LingBot-VA | ✅ task success | libero_10, 10 tasks × 50 eps: 423/500 (84.6%) with LoongForge-trained 5000-step weights |
| FastWAM | ✅ task success | 4 suites × 500 eps with LoongForge-trained 20k weights: spatial 97.00%, object 99.60%, goal 97.20%, libero_10 94.20% (total 1940/2000 = 97.00%) |

Report rates as successes/episodes over a full suite; single-episode smoke runs are
not success rates and are not listed here.

Caveat on the GR00T-N1.6 LIBERO numbers: the shipped configs do not set
`server.chunk_execute_steps`, so the whole 16-step chunk is executed open-loop,
whereas the official GR00T LIBERO client uses `--n_action_steps 8`. The numbers
above are therefore not strictly the official protocol.

## Parallel evaluation

The parallel entry point runs one inference service per GPU plus `N` sim workers per GPU
behind a global queue. Add `servers` and `pools` to the YAML; the LIBERO client is
unchanged. Background and the shared configuration reference:
[Parallel evaluation](../parallel_eval.md).

```bash
examples/embodied/pi05/eval/run_libero_eval_parallel.sh
```

Start from `configs/libero/libero_pi05_parallel.yaml`. LIBERO-specific notes:

- LIBERO is a **CPU-sim family**, so its worker pool is `pools.cpu_sim.n_sims_per_server`
  (a `pools.gpu_sim` block is ignored and `N` falls back to 1 — the coordinator warns).
- `benchmark.return_action_chunk` defaults to `true` on this branch: the client owns the
  action chunk, which needs one request per chunk instead of one per environment step.
  It was validated against the legacy server-cache path over 80 episodes (no success
  flips, 79/80 identical step counts, ~9.8× fewer requests). Pass
  `--no-return-action-chunk` (or set it to `false`) to restore the per-step path.
- `benchmark.env_seed_mode: per_task` binds a scene to "seed + reset count in this
  process", so a task must stay in one worker. Episode-level sharding under `per_task`
  requires scene snapshots (`benchmark.scene_snapshot_dir`); otherwise keep the default
  `per_episode`.
- `server.chunk_execute_steps` selects how many actions of a chunk are executed before
  re-planning (the reference runs use 10).

## Scene snapshots

Under `per_task` seeding the scene an episode sees depends on how many resets already ran
in that process, so one task cannot be split across workers. A scene snapshot is the RNG
state captured just before each reset; replaying it lets any episode run in any worker
again. Generate them once per `(suite, seed, episodes_per_task)` and point
`benchmark.scene_snapshot_dir` at the directory.

```bash
# one task
python -m loongforge.embodied.eval.orchestrator.scene_snapshots \
    --suite libero_10 --task 3 --seed 7 --episodes 50 --out /path/to/snapshots

# every task of one suite, 10 at a time
python -m loongforge.embodied.eval.orchestrator.scene_snapshots \
    --suite libero_10 --seed 7 --episodes 50 --out /path/to/snapshots --jobs 10

# every suite in one directory (each task is an independent process)
python -m loongforge.embodied.eval.orchestrator.scene_snapshots \
    --suites libero_10,libero_object,libero_goal,libero_spatial \
    --seed 7 --episodes 50 --out /path/to/snapshots --jobs 40
```

Cost and constraints:

- ~90–105 s per task (create the env + `episodes` resets + write ~300 KB). One suite of
  10 tasks is ~15 min serially; 40 tasks is ~1 h serially and **~4 min at `--jobs 40`** on a
  128-core machine (measured: 40 tasks x 50 snapshots in 245 s). The work is pure CPU and the tasks are independent.
- **All suites share one directory**: file names carry the suite
  (`{suite}__task{k}__seed{seed}.pkl`), so a four-suite parallel run can point
  `scene_snapshot_dir` at the parent directory.
- `--seed` must equal the run's `run.seed` (the template uses `7`): under `per_task`
  seeding every episode of a task reuses that one seed, and the snapshots are looked up
  by file name, so a mismatch is a missing file (the runner now says so).
- `--episodes` must be **>= `benchmark.episodes_per_task`**, otherwise the runner raises
  `scene snapshot ... has N states but episode_idx=...`. Generation reads every file back
  and exits non-zero on a length mismatch, so the failure surfaces before the run.
- Files carry `SNAPSHOT_VERSION`; a mismatch is rejected at load time. Regenerate after
  upgrading if the format changes, and after changing the benchmark assets or reset logic
  (snapshots are tied to the version that produced them).

The generator prints one `[i/N]` line per task plus a final `written / failed` summary; a
failed task does not stop the others and makes the exit code non-zero.

## Calibration results

Reference environment: 8×A800-80GB, one inference service per GPU, FastWAM epilogue
(`chunk_execute_steps=10`), 512 concurrent clients at `N=64`:

| N (sims per service) | 16 | 24 | 48 | 80 |
|---|---|---|---|---|
| steady capacity (eps/min) | 70.7 | 77.8 | 81.8 | 81.9 |

The curve plateaus around `N≈48` on this machine: beyond that, extra sims mostly add
memory and CPU contention. Capacity is a property of the model, the benchmark and the
machine — re-measure with `calibrate.py`, and note that the server-side batcher changes
the request cadence and therefore the knee.

## Measured speedup

Full-suite comparison against the official evaluation script (FastWAM, 2000 episodes,
8 GPUs, `per_task` + scene snapshots, `chunk_execute_steps=10`, `N=64`, batching on):

| Measurement | Wall | Speedup vs official |
|---|---|---|
| Official script (8-card equivalent) | 173.7 min | 1.0× |
| Parallel evaluation, worker phase | 35.5 min | **4.9×** |
| Parallel evaluation, including model loading | ~41 min | **~4.3×** |

Other configurations land lower (`episodes_per_chunk=1`: 43.6 min worker phase; `=4`:
36.1 min), so `episodes_per_chunk` and the client-owned chunk path both matter. Numbers
are for this checkpoint and protocol; the aggregate success rate is aligned within
binomial noise (96.35% vs 96.65% reference).

