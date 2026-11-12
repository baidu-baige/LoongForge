# RoboTwin Evaluation

RoboTwin is a dual-arm tabletop manipulation benchmark. The LoongForge eval module reuses RoboTwin's official evaluator, which calls the policy through a plugin (the `action_bridge` protocol). Three model × bridge combinations reach task success.

## Step 0: Download weights

| Model | Weights |
|---|---|
| pi05 | [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2) (openpi `norm_stats`, see §2 Run evaluation below) |
| xvla | [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2) |
| LingBot-VA | [robbyant/lingbot-va-posttrain-robotwin](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin) (one root with `transformer/` + `vae/` + `text_encoder/` + `tokenizer/`) |

## Step 1: Environment setup

### Standard environment

Install RoboTwin following the official repository instructions (SAPIEN-based), then install the additional dependencies:

```bash
pip install websockets msgpack msgpack-numpy pyyaml
```

The official evaluator video logging launches `ffmpeg` directly — link the `imageio-ffmpeg` executable into the env `bin` directory.

⚠️ Common issues:

- **Vulkan / SAPIEN.** Verify a real NVIDIA Vulkan ICD with `vulkaninfo` (expect `deviceName = NVIDIA ...`); set `LD_LIBRARY_PATH` / `VK_ICD_FILENAMES` before SAPIEN is imported.
- `numpy` pin: internal env uses `numpy 1.26.4` / `torch 2.4.1` / `sapien 3.0.0b1`.

## Step 2: Run evaluation

Run from inside the **benchmark** environment. The run scripts and eval YAMLs ship with `/path/to/...` placeholders — fill them in before running:

```bash
cd /path/to/LoongForge-VLA
examples/embodied/pi05/eval/run_robotwin_eval.sh    # pi05, action_bridge: pi05_aloha_14d
examples/embodied/xvla/eval/run_robotwin_eval.sh    # xvla, action_bridge: ee6d_dual
examples/embodied/lingbot_va/eval/run_robotwin_eval.sh    # LingBot-VA, action_bridge: lingbot_va_ee_quat_16d (default config: robotwin_50tasks.yaml)
```

Environment variables: `CONFIG`, `BENCHMARK_PYTHON`, `CUDA_VISIBLE_DEVICES`, plus the SAPIEN Vulkan variables (`LD_LIBRARY_PATH` / `VK_ICD_FILENAMES`).

Key config fields (see `examples/embodied/<model>/eval/configs/robotwin/adjust_bottle_smoke.yaml`):

- `benchmark.action_bridge` — selects the official-evaluator protocol:

| `action_bridge` | Model | Protocol |
|---|---|---|
| `pi05_aloha_14d` | pi05 | openpi Aloha joint protocol; `model.action_dim: 14`, `action_horizon: 32` |
| `ee6d_dual` | xvla | X-VLA dual-arm end-effector protocol; `model.domain_id: 6` |
| `lingbot_va_ee_quat_16d` | LingBot-VA | 16-D dual-arm end-effector pose relative to the episode's initial pose, no proprio input; the community protocol also needs `benchmark.instruction_type: seen` (already set in `robotwin_50tasks.yaml`, see [LingBot-VA quick start](../../quick_start_lingbot_va.md)) |

- pi05 + RoboTwin additionally needs `server.dataset_statistics_path` pointing at a stats file derived from the weights' openpi `norm_stats.json`; a ready-made copy ships at `examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json`. To regenerate from another openpi-style weights:

  ```python
  import json
  from pathlib import Path

  raw = json.loads(Path("<weight_dir>/assets/.../norm_stats.json").read_text())["norm_stats"]
  out = {"observation.state": raw["state"], "action": raw["actions"]}
  Path("dataset_stats.json").write_text(json.dumps(out, indent=2))
  ```

## Verification

| Model | Status | Notes |
|---|---|---|
| pi05 | ✅ task success | `adjust_bottle demo_clean`, `action_bridge=pi05_aloha_14d` |
| xvla | ✅ task success | `adjust_bottle demo_clean`, `action_bridge=ee6d_dual`, `domain_id=6` |
| LingBot-VA | ✅ task success | 50 tasks × 50 eps with LoongForge-trained 50000-step weights: Easy 2189/2500 (87.6%), Hard 2143/2500 (85.7%) |

## Outputs

In addition to the standard outputs (see the user guide §4 Outputs), RoboTwin collects the official evaluator logs, deploy config, result file, and `mp4` videos under `artifacts/robotwin/<task_name>/<task_config>/`, and writes one `results.jsonl` row per completed episode.

## Parallel evaluation

The parallel entry point runs one inference service per GPU plus `N` sim workers per GPU
behind a global queue. It is configured by adding `servers` and `pools` to the YAML; the
benchmark client is unchanged. Background: [Parallel evaluation](../parallel_eval.md).

```bash
examples/embodied/pi05/eval/run_robotwin_eval_parallel.sh
```

Start from `configs/robotwin/robotwin_pi05_parallel.yaml`. The RoboTwin-specific switches:

| Key | Value in the parallel config | Why |
|---|---|---|
| `pools.gpu_sim.n_sims_per_server` | 8 | Fits the per-sim memory of the reference tasks |
| `benchmark.episodes_per_chunk` | `floor(E/S)` | One wave, one process start per unit |
| `benchmark.plan_order` | `lpt_long` | Front-loads the long tasks; with `long_task_steps: 1200` + `long_task_episodes_per_chunk: 10` the measured wall clock drops 1.88 → 1.77 h on the full 50×100 sweep |
| `benchmark.long_task_steps` | 1200 | RoboTwin step limits run 400–1700; the ≥1200 tasks dominate the tail |
| `benchmark.long_task_episodes_per_chunk` | 10 | One unit of a 1,500-step task is ~33 min instead of ~83 min |
| `benchmark.return_action_chunk` | `true` | The client owns the action chunk |
| `benchmark.skip_intra_chunk_obs` | `true` | No camera render inside a chunk — the largest single win |
| `benchmark.sim_cpu_threads` | 1 | Avoids thread oversubscription with many sims per card |
| `resources.gpu_sim` | `vram_total_mb` / `vram_per_sim_mb` / `vram_model_mb` | Startup check: `N x per_sim + model <= total`, otherwise the run refuses to start. Fill it with measured peaks — a steady-state average will not catch a card that runs at 99% |

`return_action_chunk` + `skip_intra_chunk_obs` make the policy open-loop inside a chunk
(one observation per `chunk_execute_steps` actions). Keep them off if you need
per-episode agreement with a per-step closed-loop run — see
[Result alignment](../parallel_eval.md#8-result-alignment). For this reason both switches
default to `false` in the runner.

### Offline expert check

RoboTwin's official loop runs an expert rollout before the policy rollout to reject
unsolvable seeds. That pass can be cached and reused:

1. **Scan** — `benchmark.offline_expert_check_scan: <dir>` runs the expert pass only (no
   inference service) and writes `<task>.pkl`. `offline_expert_check_scan_shards: k`
   splits each task's candidate seeds into `k` residue classes (`start+i`, stride `k`)
   and merges the parts at the end of the run.
2. **Consume** — `benchmark.offline_expert_check: <dir>` skips the expert pass and reads
   seeds by index. The planner writes the per-unit cache offset; `start_seed` remains a
   seed value and does not index the cache.
3. **Staleness** — the cache is only valid for the environment that produced it, and
   "environment" means the **full compute stack**: GPU model, driver and CUDA version,
   Python and library versions (torch, SAPIEN/svulkan, cuRobo), the benchmark assets and
   the task/scene definition. GPU or driver changes can alter which seeds are stable, so
   a cache produced on another machine may admit a seed that is unstable here. This
   implementation does **not** record or verify an environment digest, so re-scan
   whenever any of the above changes, and treat the cache as a property of one machine
   configuration. If a cached seed turns out to be unstable at consume time the unit
   fails with `CACHE STALE` and the coordinator retries it; re-scan rather than silently
   skipping.

Scan cost scales with the number of candidate seeds and the expert rollout length; use
one shard per available slot to keep the scan short.

The cache is per `task_config`: `demo_clean` and `demo_randomized` have different scenes,
so they admit different seeds and record different `arm_tag`/`origin_z` values. The file
name carries only the task (`<task>.pkl`), so scan a randomized run into its **own**
directory — pointing it at the clean cache overwrites it.

## FastWAM: batching is the lever, N is not

FastWAM is inference-bound, not sim-bound: one `predict_action` takes 810 ms while the
GPU idles between requests. Server batching removes that idle time. A single-GPU probe
(`orchestrator/batch_probe.py`) measures 810 / 845 / 843 / 881 ms per call at batch
1 / 2 / 4 / 8 — flat latency, so **7.35x per sample at batch 8**.

Inside the eval the batcher cannot always fill: with 8 sims per service the measured
`mean_batch` is 2.5–3.0 (`max_batch` 7–8), which is still ~3x the inference throughput of
batch 1. `max_wait_ms` cannot raise it — `pending` is usually 0, so arrivals, not the
wait window, are the limit.

That makes N and the batch size compete for the same VRAM budget, and N stops paying:

| | N=8, batch 8 | N=10, batch 4 |
|---|---|---|
| 10 tasks × 100 episodes, wall clock | **43 min** | 58 min |
| episodes/min | **25.3** | 18.9 |
| peak VRAM per card | 73 GB | 81 GB (at the limit) |
| unit failures | 0/145 | 22 units OOM'd once, 1 repair round |
| success rate | 84.1% | 85.8% (same within noise) |

More sims only lengthen the queue: per-episode latency rose 1.2–2.2x while request
throughput rose 6%. **Use N=8 for FastWAM**; the template encodes it together with
`batching.max_batch_size: 8` and `run.unit_timeout_sec: 2400` (a hung unit would
otherwise hold its slot for `chunk x timeouts.per_episode_sec` = 20 h).

Full clean sweep (50 tasks × 100 episodes, seed 100000, offline expert check, 8×A800):

| | Value |
|---|---|
| coverage | 5,000/5,000, 590/590 units, 0 retries, `validation.ok=true` |
| success rate | **92.10%** vs **91.88%** in the paper (+0.22 pt) |
| worker phase / end-to-end | **1.84 h / 1.93 h** |
| throughput | 45.3 episodes/min (43.2 end-to-end) |
| baseline (`demo_clean` only, 16 slots, same protocol) | 28.6 h → **15.5x** (14.8x end-to-end) |

Per task the mean deviation is +0.22 pt; 49 of 50 tasks are within ±7 pt. The one real
gap is `open_microwave` (35% vs 62%), reproduced across three independent runs — that
task needs the official `deploy_policy` entry point to tell a protocol effect from a
checkpoint weakness. The randomized column has not been run yet.

## Calibration results

Reference environment: 8×A800-80GB, pi05, `adjust_bottle`, official 400-step limit, offline
expert check, `E = 16N`, two episodes per unit.

| N | unit wall | startup | rollout / episode | wall throughput (per GPU) | steady throughput (per GPU) | peak VRAM |
|---|---|---|---|---|---|---|
| 6 | 204.5 s | 167 s | 17.0 s | 3.52 ep/min | 21.1 ep/min | — |
| 8 | 266.6 s | 221 s | 20.8 s | 3.60 ep/min | 23.1 ep/min | — |
| 10 | 323.8 s | 275 s | 22.0 s | 3.71 ep/min | 27.3 ep/min | ~58–60 GB |
| 12 | 389.0 s | 341 s | 21.9 s | 3.70 ep/min | 32.9 ep/min | ~75 GB |

Observations from this run: steady throughput keeps rising through `N=12` (no plateau
yet), while peak memory grows roughly linearly with `N` and approaches the 80 GB card
limit at `N=12`; process startup dominates the wall metric at these unit sizes. Memory
depends on the benchmark scene, the model and the renderer settings — measure on your
setup with `calibrate.py` and keep headroom.

## Measured speedup

Single-task comparison against the original entry point (pi05, same task, 50 episodes,
official step limits, 8 GPUs):

| Task | Baseline | Parallel config | Speedup |
|---|---|---|---|
| `adjust_bottle` (400 steps) | 1040 s total / 699 s episode phase | 513 s / 199 s | 2.0× / 3.5× |
| `open_microwave` (1500 steps) | 4984 s / 4701 s | 756 s / 453 s | 6.6× / 10.4× |

The second number excludes model loading and service startup (about 5 min for eight
services); sim-worker process startup is included. The two runs use different seed sets
(see [Result alignment](../parallel_eval.md#8-result-alignment)), so the failure counts
are not directly comparable. Full-scale numbers (50 tasks × 100 episodes) are in
[FastWAM: batching is the lever](#fastwam-batching-is-the-lever-n-is-not) above and in
the pi05 result table of the PR description.

> **Reproducibility note.** This pipeline is not bit-reproducible on GPU: two identical
> runs (same config, fixed `model.noise_seed`, `max_batch_size: 1`) still differ by 1–3
> steps on the same seed. Per-episode regression against another run is therefore not a
> usable equivalence test; compare aggregate success rates instead (n≥50 has ±7pt
> binomial noise at 1σ, ±14pt at 95%). This is also why the chunk switches above stay
> off by default: their per-episode effect cannot be separated from that noise floor.


