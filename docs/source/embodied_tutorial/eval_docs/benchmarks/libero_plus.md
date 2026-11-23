# LIBERO-Plus Evaluation

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) ([paper](https://arxiv.org/abs/2510.13626)) is a robustness extension of LIBERO. It keeps the four stock suites (Spatial, Object, Goal, Long/10) and expands each into 7 perturbation categories, for **10,030 tasks** in total:

| Official column | Perturbation | Tasks (4 suites) |
|---|---|---|
| Camera | camera viewpoints | 1,599 |
| Robot | robot initial states | 1,550 |
| Language | paraphrased instructions | 1,537 |
| Light | light conditions | 1,142 |
| Background | background textures | 1,076 |
| Noise | sensor noise | 1,601 |
| Layout | object layouts | 1,525 |
| **Total** | | **10,030** |

The official protocol evaluates **1 initial state per task** (`num_trials_per_task: 1`). LIBERO-Plus ships as a drop-in replacement of the `libero` package, so the LoongForge LIBERO runner is reused unchanged; `benchmark.variant: plus` only adds the perturbation-level summary. The shipped template runs in **parallel mode**: one inference service per GPU, N CPU-sim workers per GPU, and a single global queue covering all four suites.

pi05 is verified on all four suites (see [Verification](#verification)); the runner is model-agnostic, so any model already working on stock LIBERO can run LIBERO-Plus.

## ⚠️ Differences vs standard LIBERO

If you already have a working standard-LIBERO evaluation, only the following need attention:

| Item | Standard LIBERO | LIBERO-Plus |
|---|---|---|
| `benchmark.name` | `libero` | `libero` — keep it; do **not** write `libero_plus` (the name is used as the suite name) |
| `benchmark.variant` | omitted → defaults to `standard` | **`plus`** — required to emit `libero_plus_summary.csv`; leaving it at the default silently drops the perturbation summary |
| `benchmark.episodes_per_task` | typically 10–50 | **1** (official protocol: one initial state per task) |
| `benchmark.max_tasks` | usually a subset | **0** (all ~2,400–2,600 tasks per suite) |
| Python env | stock `libero` | a **separate** env with LIBERO-Plus installed (same package name; the two cannot coexist) |
| Assets | shipped by the repo | **`assets.zip` (6.4 GB) is required** — the repository has no `assets/` |
| `LIBERO_CONFIG_PATH` | stock LIBERO config | must point at the LIBERO-Plus `config.yaml` |
| `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | usually unset | **required** for `.pruned_init` under torch ≥ 2.6 (the runner sets it for you) |
| Outputs | `results.jsonl` / `summary.csv` / `suite_summary.csv` | the same, **plus `libero_plus_summary.csv`** (7 categories + Total) |

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

You only need to install the environment in accordance with [LIBERO's official requirements](https://github.com/Lifelong-Robot-Learning/LIBERO) (MuJoCo / robosuite / bddl stack). LIBERO-Plus is a drop-in replacement of the `libero` package, so no extra simulator setup is needed — install it into a dedicated benchmark env (stock LIBERO, LIBERO-Plus and LIBERO-Pro share the package name and cannot coexist), together with its additional client requirements:

```bash
git clone --depth 1 https://github.com/sylvestf/LIBERO-plus.git <LIBERO-PLUS>
pip install --no-build-isolation --config-settings editable_mode=compat -e <LIBERO-PLUS>
pip install websockets msgpack msgpack-numpy pyyaml wand "scikit-image==0.26.0" "usd-core>=25.5"
```

`wand` and `scikit-image` are LIBERO-Plus `extra_requirements`; `envs/env_wrapper.py` imports them unconditionally. Keep `numpy==1.26.4` when installing scikit-image (newer releases may pull NumPy 2.x, which the simulator stack does not support). The repository has no top-level `libero/__init__.py`, so use `editable_mode=compat` (or a plain source-root `.pth`), otherwise `import libero` fails. Offscreen rendering needs the EGL/OSMesa and ImageMagick runtime packages:

```bash
apt-get install -y libegl1 libegl-mesa0 libosmesa6 libfontconfig1-dev libmagickwand-dev
```

### 1.1 Key dependency versions used for the reported results

| Component | Version |
|---|---|
| Python | 3.12.3 |
| `libero` (LIBERO-Plus, editable) | 0.1.0 |
| torch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 |
| numpy | 1.26.4 |
| mujoco | 3.3.1 |
| robosuite | 1.4.0 |
| bddl | 1.0.1 |
| robomimic | 0.2.0 |
| transformers | 4.57.3 |
| opencv-python | 4.10.0.84 |
| scikit-image | 0.26.0 |
| wand | 0.7.2 |
| usd-core | 26.11 |
| websockets | 17.4 |
| msgpack / msgpack-numpy | 1.1.0 / 0.4.8 |
| PyYAML | 6.0.3 |
| imageio | 2.37.4 |
| h5py | 3.16.0 |
| apt (`libegl1` / `libegl-mesa0` / `libosmesa6` / `libmagickwand-dev`) | 1.7.0 / 25.2.8 / 25.1.7 / 8:6.9.12.98+dfsg1 |

> These are the benchmark/client-side versions. The policy server runs in a separate LoongForge environment.

### 1.2 Download and unpack the assets (required)

The repository does **not** ship `assets/`; the 6.4 GB archive contains the meshes, textures and scenes used by the 10,030 perturbed tasks. Without it, environment creation fails.

```bash
HF_HUB_DISABLE_XET=1 hf download Sylvest/LIBERO-plus assets.zip \
  --repo-type dataset --local-dir <ASSETS>
```

The archive stores a deep prefix (`inspire/hdd/.../LIBERO-plus-0/assets/`), so extract to a staging directory on the same filesystem and move the `assets/` directory into place:

```bash
unzip -q <ASSETS>/assets.zip -d <ASSETS>/staging
mv <ASSETS>/staging/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets \
   <LIBERO-PLUS>/libero/libero/assets
```

Expected result: `<LIBERO-PLUS>/libero/libero/assets/{new_objects,textures,scenes,stable_scanned_objects,...}`.

### 1.3 LIBERO path config

Point LIBERO at the LIBERO-Plus checkout (do not reuse the stock LIBERO config):

```yaml
# <PLUS-CONFIG>/config.yaml
benchmark_root: <LIBERO-PLUS>/libero/libero
bddl_files:    <LIBERO-PLUS>/libero/libero/bddl_files
init_states:   <LIBERO-PLUS>/libero/libero/init_files
datasets:      <LIBERO-PLUS>/libero/libero/../datasets
assets:        <LIBERO-PLUS>/libero/libero/assets
```

Set `env.libero_config_path` in the eval YAML to the directory containing it.

⚠️ The runner loads `.pruned_init` pickles (`numpy` inside), which `torch >= 2.6` rejects under the new `weights_only=True` default. The runner sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` itself; export it too when running the runner standalone.

## Step 2: Run evaluation

The launcher starts the parallel coordinator: one inference service per GPU listed in `servers[].gpus`, N sim workers per GPU from `pools.cpu_sim.n_sims_per_server`, and one global queue over every suite in `benchmark.suites` (the shipped template runs all four):

```bash
cd /path/to/LoongForge
BENCHMARK_PYTHON=/path/to/libero_plus/bin/python \
  bash examples/embodied/pi05/eval/run_libero_plus_eval.sh
```

The coordinator assigns each service a port starting at `LOONGFORGE_COORD_PORT_BASE` and pins each worker's renderer to its slot GPU. Do **not** export `CUDA_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID` around the launcher — that would collapse every service and renderer onto one card.

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `CONFIG` | eval YAML (or pass it as the first positional argument) | `examples/embodied/pi05/eval/configs/libero/libero_plus.yaml` |
| `BENCHMARK_PYTHON` | LIBERO-Plus env interpreter (client) | `/path/to/libero_plus/bin/python` |
| `LOONGFORGE_COORD_PORT_BASE` | base port for the per-GPU inference services | framework default |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | keep the client offline | `1` |
| `DRY_RUN` | `1` prints the environment and command only | `0` |

Key config fields (`configs/libero/libero_plus.yaml`):

- `benchmark.variant: plus` — enables `libero_plus_summary.csv`; standard LIBERO omits it
- `benchmark.suites: [libero_spatial, libero_object, libero_goal, libero_10]` — all four suites in one coordinated plan
- `benchmark.max_tasks: 0`, `benchmark.episodes_per_task: 1` — all tasks, official 1-trial protocol
- `benchmark.tasks_per_unit: 5`, `benchmark.episodes_per_chunk: 1` — group tasks per worker process to amortise the torch/robosuite import (the official protocol is one episode per task, so this is the main startup-cost lever)
- `benchmark.plan_order: lpt` — longest-processing-time-first scheduling
- `servers: [{gpus: [0, ..., 7]}]` — one inference service per GPU (shorten the list for fewer cards)
- `pools.cpu_sim.n_sims_per_server: 16` — sim workers per GPU; LIBERO is CPU-sim, so `gpu_sim` is ignored
- `server.batching: {max_batch_size: 8, max_wait_ms: 5}` — server-side batching used by the reference runs
- `benchmark.max_steps: -1` — suite budget (Spatial 220 / Object 280 / Goal 300 / Long 520)
- `benchmark.continuous_gripper: true`, `prefer_controller_ee: false` — pi05 LIBERO conventions
- `model.action_horizon: 10` — **must match** the checkpoint (openpi `pi05_libero` uses 10; 50 collapses success)
- `model.eval_image_normalize_mode: siglip`, `model.eval_normalization_mode: q99` — openpi `LiberoInputs` pipeline
- `model.state_encoding: libero_ee8` — 8D proprio (pos + axis-angle + gripper qpos)
- `model.empty_cameras: 1` — the official checkpoint expects 3 visual slots (base + wrist + 1 masked pad); use `0` for a 2-camera checkpoint
- `server.chunk_execute_steps: 10` — execute 10 of the predicted chunk before replanning
- `server.dataset_statistics_path` — the checkpoint's norm stats from Step 0 (not a dataset-derived stats file)
- `run.unit_timeout_sec: 7200` — per-unit wall-clock budget; timed-out units are retried

Outputs: each unit writes `results.jsonl` / `summary.csv` / `suite_summary.csv` under `run.output_dir/worker_<i>/`; the coordinator merges them and, for `variant: plus`, also writes **`libero_plus_summary.csv`** (7 perturbation categories + suite `Total` + `all/Total`).

## Verification

pi05 (`lerobot/pi05_libero`), all tasks of each suite, 1 episode/task, seed 7, **10,030 episodes, 0 infrastructure failures**. Reference values are the π0.5 row of [ELAN4D](https://arxiv.org/pdf/2605.30484) Table 1.

The table merges both reported dimensions: **perturbation** rows are the unweighted mean over the four suites, **suite** rows are per-suite, and **Overall** is pooled over all 10,030 tasks.

| Dimension | Metric | LoongForge | paper π0.5 | Δ |
|---|---|---|---|---|
| Perturbation | Camera Viewpoints | 57.4% | 59.7% | −2.3 |
| Perturbation | Robot Initial States | 82.6% | 65.5% | +17.1 |
| Perturbation | Language Instructions | 83.7% | 75.3% | +8.4 |
| Perturbation | Light Conditions | 93.5% | 87.0% | +6.5 |
| Perturbation | Background Textures | 93.2% | 82.4% | +10.8 |
| Perturbation | Sensor Noise | 84.5% | 72.1% | +12.4 |
| Perturbation | Objects Layout | 86.4% | 80.3% | +6.1 |
| Task suite | Spatial | 89.1% | 79.9% | +9.2 |
| Task suite | Object | 87.1% | 87.8% | −0.7 |
| Task suite | Goal | 75.4% | 69.0% | +6.4 |
| Task suite | Long (10) | 75.6% | 64.9% | +10.7 |
| Overall | All tasks (pooled) | 81.7% | 73.6% | +8.1 |