# LIBERO-Pro Evaluation

[LIBERO-Pro](https://github.com/Zxy-MLlab/LIBERO-PRO) ([paper](https://arxiv.org/abs/2510.03827)) is a robustness extension of LIBERO. It keeps the four stock suites (Spatial, Object, Goal, Long/10) and adds five perturbation dimensions per suite:

| Official column | Perturbation | Registered suite |
|---|---|---|
| **Obj** | object appearance / color / size | `libero_<base>_object` |
| **Pos** | initial object positions | `libero_<base>_swap` |
| **Sem** | paraphrased instructions | `libero_<base>_lan` |
| **Task** | redesigned task logic / goals | `libero_<base>_task` |
| **Env** | working environment replacement | `libero_<base>_env` |

That is **20 suites × 10 tasks**, evaluated with **50 episodes per task** (each `.pruned_init` holds 50 initial states). LIBERO-Pro ships as a drop-in replacement of the `libero` package, so the LoongForge LIBERO runner is reused unchanged. The shipped template runs in **parallel mode**: one inference service per GPU, N CPU-sim workers per GPU, and a single global queue over all 20 suites.

pi05 is verified on all five dimensions (see [Verification](#verification)).

## ⚠️ Differences vs standard LIBERO

If you already have a working standard-LIBERO evaluation, only the following need attention:

| Item | Standard LIBERO | LIBERO-Pro |
|---|---|---|
| `benchmark.name` | `libero` | `libero` — keep it; do **not** write `libero_pro` (the name is used as the suite name) |
| `benchmark.variant` | omitted → defaults to `standard` | **`pro`** — required to read the instruction from the BDDL; with the default, `_lan` / `_task` fall back to the filename-derived instruction |
| `benchmark.suites` | `libero_spatial` / `object` / `goal` / `10` | derived names `libero_<base>_{object,swap,lan,task,env}` — **20 suites** |
| `benchmark.episodes_per_task` | typically 10–50 | **50** (official protocol) |
| `benchmark.max_tasks` | usually a subset or 10 | **0** (all 10 tasks per suite) |
| Python env | stock `libero` | a **separate** env with LIBERO-Pro installed (same package name; the two cannot coexist) |
| Data | 4 stock suites | HF dataset (~10 MB) for Obj/Pos/Sem/Task **plus locally generated `_env` suites** (`GEN_ENV=1`) |
| `LIBERO_CONFIG_PATH` | stock LIBERO config | must point at the LIBERO-Pro `config.yaml` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | usually unset | **required** for `.pruned_init` under torch ≥ 2.6 (the runner sets it for you) |
| Instruction source | task filename | **BDDL `(:language ...)`** for `_lan` / `_task` (hence `variant: pro`) |
| Outputs | `results.jsonl` / `summary.csv` / `suite_summary.csv` | the same, written per unit under `run.output_dir/worker_<i>/` and merged by the coordinator |

Everything else — model settings (`action_horizon`, `q99`, `empty_cameras`, norm stats), the runner, the payload builder and the policy server — is identical to standard LIBERO.

## Step 0: Weights and normalization statistics

| Item | Source |
|---|---|
| Weights | [lerobot/pi05_libero](https://huggingface.co/lerobot/pi05_libero) (`model.safetensors` + tokenizer) |
| Tokenizer | `google/paligemma-3b-pt-224` |
| **Norm stats** | `gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json` ([HTTPS](https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json)) |

The checkpoint's **own** norm stats are required. Convert the openpi layout to the LeRobot keys the runner expects:

```python
import json
src = json.load(open("norm_stats.json"))["norm_stats"]
out = {
    "observation.state": {k: src["state"][k] for k in ("mean", "std", "q01", "q99")},
    "action":            {k: src["actions"][k] for k in ("mean", "std", "q01", "q99")},
}
json.dump(out, open("pi05_libero_norm_stats.json", "w"), indent=2)
```

## Step 1: Environment setup

You only need to install the environment in accordance with [LIBERO's official requirements](https://github.com/Lifelong-Robot-Learning/LIBERO) (MuJoCo / robosuite / bddl stack). LIBERO-Pro is a drop-in replacement of the `libero` package, so no extra simulator setup is needed — install it into a dedicated benchmark env (stock LIBERO, LIBERO-Plus and LIBERO-Pro share the package name and cannot coexist):

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git <LIBERO-PRO>
pip install --no-build-isolation --config-settings editable_mode=compat -e <LIBERO-PRO>
pip install websockets msgpack msgpack-numpy pyyaml
```

The repository has no top-level `libero/__init__.py`, so use `editable_mode=compat` (or a plain source-root `.pth`), otherwise `import libero` fails. LIBERO-Pro vendors the LIBERO asset tree, so no extra assets are needed.

### 1.1 Key dependency versions used for the reported results

| Component | Version |
|---|---|
| Python | 3.12.3 |
| `libero` (LIBERO-Pro, editable) | 0.1.0 |
| torch / torchvision | 2.9.0 / 0.24.0 |
| numpy | 1.26.4 |
| mujoco | 3.3.1 |
| robosuite | 1.4.0 |
| bddl | 1.0.1 |
| robomimic | 0.2.0 |
| transformers | 4.57.3 |
| opencv-python | 4.10.0.84 |
| websockets | 17.1 |
| msgpack / msgpack-numpy | 1.1.0 / 0.4.8 |
| PyYAML | 6.0.3 |
| imageio | 2.37.4 |
| h5py | 3.16.0 |
| scipy | 1.17.1 |

> These are the benchmark/client-side versions. The policy server runs in a separate LoongForge environment.

### 1.2 Load the four pre-generated dimensions

The HF dataset [zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) (~10 MB) ships `bddl_files/<suite>/` and `init_files/<suite>/` for the **Obj / Pos / Sem / Task** dimensions:

```bash
hf download zhouxueyang/LIBERO-Pro --repo-type dataset --local-dir <PRO-DATA>
for base in libero_spatial libero_object libero_goal libero_10; do
  for dim in lan object swap task; do
    cp -r <PRO-DATA>/bddl_files/${base}_${dim} <LIBERO-PRO>/libero/libero/bddl_files/
    cp -r <PRO-DATA>/init_files/${base}_${dim} <LIBERO-PRO>/libero/libero/init_files/
  done
done
```

### 1.3 Generate the Env dimension

`libero_*_env` bddl/init are produced at evaluation time by the official pipeline. **The launcher `examples/embodied/pi05/eval/run_libero_pro_eval.sh` already contains this generation logic**: run it with `GEN_ENV=1` and it generates any missing `libero_*_env` suites before the evaluation (requires `LIBERO_PRO_ROOT`; suites whose `*_env` directories already exist and are non-empty are skipped). You can also generate them yourself once, with the same official pipeline:

```python
import os, shutil, sys, yaml
root = "<LIBERO-PRO>"; base = f"{root}/libero/libero"; sys.path.insert(0, root)
import perturbation

cfg = yaml.safe_load(open(f"{root}/evaluation_config.yaml"))
ood = {k: os.path.join(root, v.lstrip("./")) for k, v in cfg["ood_task_configs"].items()}
for suite in ("libero_goal", "libero_spatial", "libero_10", "libero_object"):
    tmp = perturbation.process_bddl_file_mixed(
        input_dir=f"{base}/bddl_files/{suite}", task_suite_name=suite,
        flags=perturbation.PerturbFlags(use_environment=True), configs=ood, seed=42)
    perturbation.EvalEnvCreator(
        input_dir=tmp, script_path=f"{root}/notebooks/generate_init_states.py",
        base_output_dir=f"{base}/init_files").create_env()
    for kind in ("bddl_files", "init_files"):
        shutil.rmtree(f"{base}/{kind}/{suite}_env", ignore_errors=True)
        os.rename(f"{base}/{kind}/{suite}_temp", f"{base}/{kind}/{suite}_env")
```

### 1.4 LIBERO path config

Point LIBERO at the LIBERO-Pro checkout (do not reuse the stock LIBERO config):

```yaml
# <PRO-CONFIG>/config.yaml
benchmark_root: <LIBERO-PRO>/libero/libero
bddl_files:    <LIBERO-PRO>/libero/libero/bddl_files
init_states:   <LIBERO-PRO>/libero/libero/init_files
datasets:      <LIBERO-PRO>/libero/libero/../datasets
assets:        <LIBERO-PRO>/libero/libero/assets
```

Set `env.libero_config_path` in the eval YAML to the directory containing it.

⚠️ The runner loads `.pruned_init` pickles (`numpy` inside), which `torch >= 2.6` rejects under the new `weights_only=True` default. The runner sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` itself; export it too when running the runner standalone.

## Step 2: Run evaluation

The launcher starts the parallel coordinator: one inference service per GPU listed in `servers[].gpus`, N sim workers per GPU from `pools.cpu_sim.n_sims_per_server`, and one global queue over **all 20 suites** in `benchmark.suites`:

```bash
cd /path/to/LoongForge
BENCHMARK_PYTHON=/path/to/libero_pro/bin/python \
LIBERO_PRO_ROOT=<LIBERO-PRO> \
  bash examples/embodied/pi05/eval/run_libero_pro_eval.sh
```

The coordinator assigns each service a port starting at `LOONGFORGE_COORD_PORT_BASE` and pins each worker's renderer to its slot GPU. Do **not** export `CUDA_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID` around the launcher — that would collapse every service and renderer onto one card.

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CONFIG` | eval YAML | `examples/embodied/pi05/eval/configs/libero/libero_pro.yaml` |
| `BENCHMARK_PYTHON` | LIBERO-Pro env interpreter (client) | `/path/to/libero_pro/bin/python` |
| `LIBERO_PRO_ROOT` | LIBERO-Pro checkout (preflight) | unset |
| `GEN_ENV` | `1` generates missing `libero_*_env` suites first | `0` |
| `LOONGFORGE_COORD_PORT_BASE` | base port for the per-GPU inference services | framework default |
| `DRY_RUN` | `1` prints the preflight result and command only | `0` |

Key config fields (`configs/libero/libero_pro.yaml`):

- `benchmark.variant: pro` — read the instruction from the BDDL `(:language ...)` block instead of the filename. The default is `standard`, so it **must be set explicitly**; otherwise `_lan` / `_task` fall back to the filename-derived instruction.
- `benchmark.suites` — the full 20-suite matrix (trim the list to run a single row, e.g. the five `libero_spatial_*` cells)
- `benchmark.episodes_per_task: 50` — official protocol; `max_tasks: 0` means all 10 tasks
- `benchmark.tasks_per_unit: 5`, `benchmark.episodes_per_chunk: 2` — group tasks per worker process and chunk episodes so slots stay busy (reference: 2,500 episodes over 8×16 slots)
- `benchmark.plan_order: lpt` — longest-processing-time-first scheduling
- `servers: [{gpus: [0, ..., 7]}]` — one inference service per GPU (shorten the list for fewer cards)
- `pools.cpu_sim.n_sims_per_server: 16` — sim workers per GPU; LIBERO is CPU-sim, so `gpu_sim` is ignored
- `server.batching: {max_batch_size: 8, max_wait_ms: 5}` — server-side batching used by the reference runs
- `env.mujoco_gl: egl` — keep EGL (not OSMesa) in parallel mode so the coordinator can pin each worker to its slot GPU
- `benchmark.max_steps: -1` — base-suite budget (Spatial 220 / Object 280 / Goal 300 / Long 520)
- `benchmark.continuous_gripper: true`, `prefer_controller_ee: false` — pi05 LIBERO conventions
- `model.action_horizon: 10` — **must match** the checkpoint (openpi `pi05_libero` uses 10)
- `model.eval_image_normalize_mode: siglip`, `model.eval_normalization_mode: q99` — openpi `LiberoInputs` pipeline
- `model.state_encoding: libero_ee8` — 8D proprio (pos + axis-angle + gripper qpos)
- `model.empty_cameras: 1` — the official checkpoint expects 3 visual slots (base + wrist + 1 masked pad); use `0` for a 2-camera checkpoint
- `server.chunk_execute_steps: 10` — execute 10 of the predicted chunk before replanning
- `server.dataset_statistics_path` — the checkpoint's norm stats from Step 0 (not a dataset-derived stats file)
- `run.unit_timeout_sec: 5400` — per-unit wall-clock budget; timed-out units are retried

## Verification

pi05 (`lerobot/pi05_libero`), 50 episodes/task, seed 7, **10,000 episodes, 0 infrastructure failures**.
Each cell is **LoongForge / official π0.5** success rate (%), reference values from the
[LIBERO-Pro leaderboard](https://github.com/Zxy-MLlab/LIBERO-PRO#-libero-pro-model-leaderboard):

| suite | Obj | Pos | Sem | Task | Env | avg |
|---|---|---|---|---|---|---|
| Spatial | **97.4** / 97 | **50.4** / 20 | **95.6** / 97 | **45.8** / 1 | **53.6** / 46 | **68.6** / 52.2 |
| Object | **94.2** / 98 | **26.6** / 17 | **99.4** / 96 | **10.2** / 1 | **75.2** / 73 | **61.1** / 57.0 |
| Goal | **74.8** / 97 | **27.6** / 38 | **95.0** / 97 | **21.2** / 0 | **31.2** / 46 | **50.0** / 55.6 |
| Long (10) | **65.2** / 92 | **10.0** / 8 | **93.8** / 93 | **17.6** / 1 | **65.8** / 46 | **50.5** / 48.0 |
| **Overall** | | | | | | **57.5** / **53.2** |

> The official per-suite `avg` is the arithmetic mean of its five columns; the published
> leaderboard `Total` is **0.53**, consistent with 53.2% here.