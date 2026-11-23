# RoboLab Evaluation

[RoboLab](https://research.nvidia.com/labs/srl/projects/robolab) ([paper](https://arxiv.org/abs/2604.09860), [leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html)) is a manipulation-policy benchmark built on [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab): **120 tasks, 10 episodes per task** (`num_envs=10` stepped in lockstep in a single run), with success detected automatically by the simulator.

LoongForge integrates RoboLab as a benchmark family (`benchmark.name: robolab`) with a fully native model side: **cosmos3 weight loading, VAE encoding, MoT forward, CFG + UniPC sampling and action post-processing all happen inside LoongForge's `model/cosmos3`**; the inference process only needs LoongForge and public dependencies (torch / transformers / diffusers). Isaac Sim runs inside the benchmark python, the policy server runs in the LoongForge python, and the two talk over a local websocket.

This guide uses cosmos3 as the example. On the leaderboard Cosmos3-Nano-Policy scores **441/1200 = 36.8%**.

## Step 0: Weights

| Item | Source / notes |
|---|---|
| Weights | [nvidia/Cosmos3-Nano-Policy-DROID](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID), diffusers directory layout |
| Architecture config | `<ckpt>/config.json` (contains `text_config`) + `transformer/config.json` |
| Tokenizer | Bundled in the checkpoint root (Qwen2 tokenizer files); `model.qwen3_vl_path` points at the checkpoint root |
| VAE | **`Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`** (not the diffusers `vae/` directory inside the checkpoint) |
| Normalization stats | None. Actions are predicted directly in model space; no dataset normalization |

Both artifacts are required (the policy weights and the VAE are separate downloads):

```bash
# HuggingFace
hf download nvidia/Cosmos3-Nano-Policy-DROID --local-dir <CKPT>
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir <WAN>
```

Key `model:` / `server:` fields:

```yaml
model:
  model_type: cosmos3
  qwen3_vl_path: <CKPT>              # checkpoint root
  vae_path: <WAN>/Wan2.2_VAE.pth
  action_dim: 8
  action_horizon: 32                 # official chunk length, executed open-loop
  state_dim: 8
server:
  python: /usr/bin/python3           # LoongForge-side interpreter; needs torch/transformers/diffusers
  ckpt_path: <CKPT>
```

Inference defaults (`Cosmos3ModelConfig`) match the official RoboLab policy server: `guidance=3.0`, `num_steps=4`, `inference_shift=5.0`, `resolution="480"`, `conditioning_fps=15`, `image_height/width=540/640`, `use_state=True`, `history_length=1`, `domain_name=droid_lerobot`.

## Step 1: Environment

### 1.1 Install RoboLab (benchmark python)

Install it following the RoboLab README (requires `uv` and a system `ffmpeg`; the Isaac Sim version is selected with a mutually exclusive extra):

```bash
sudo apt install ffmpeg
git clone <robolab_repo>
cd robolab
uv venv --python 3.11
source .venv/bin/activate
uv sync --extra isaac50            # or isaac51 / isaac60, see the RoboLab README
uv run pytest tests/               # install self-check: runs one full episode
```

**Policy server environment**: the LoongForge image environment is enough.

### 1.2 Container / headless requirements

Running Isaac Sim in a container needs a set of environment fixes. The example script `run_robolab_eval.sh` exports the proxy, `NO_PROXY`, cache directories and EULA (each overridable with the same-named environment variable); `LD_PRELOAD` is machine-specific, so the script only leaves it commented out — enable it when needed:

| Variable | Purpose |
|---|---|
| `http_proxy` / `https_proxy` | **Required** (a non-working address such as `http://127.0.0.1:9` is fine). With no proxy set at all, Isaac Sim blocks for a long time in the physx/usdrt scene-filling stage and looks like a hung scene load |
| `no_proxy=localhost,127.0.0.1` | **Required.** Otherwise the websocket to the local policy server also goes through the proxy and fails with `did not receive a valid HTTP response` |
| `XDG_CACHE_HOME` / `TMPDIR` | Simulator and HuggingFace cache directories; point them at a large disk |
| `LD_PRELOAD=<libglapi_shim.so>` | Shim for mismatched glvnd / libglapi versions inside the container (defines `_glapi_tls_Current`) |
| `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64` | Isaac Sim needs the real GL / CUDA libraries |
| `OMNI_KIT_ACCEPT_EULA=Y` | Accept the Omniverse EULA before the first run |

The first run takes about 10 minutes of cold caching (texture / shader / cooked mesh); later starts take about 1–2 minutes. Do not kill it assuming it hung.

**cuDNN clash on the policy server**: the variables above are inherited by the policy-server subprocess. If `/usr/lib/x86_64-linux-gnu` comes first in `LD_LIBRARY_PATH`, the system cuDNN does not match the cuDNN bundled with torch and the server fails at startup with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` or `undefined symbol: __nvrtcNvvmHandle`. Fix it by prepending torch's own libraries in the YAML's `env.ld_library_path` (that field only affects the policy server, not the simulator):

```yaml
env:
  ld_library_path: <site-packages>/torch/lib:<site-packages>/nvidia/cudnn/lib
```

### 1.3 GPU memory

| Component | Memory |
|---|---|
| Cosmos3-Nano (16B, bf16) | about 33 GB (including VAE and activations) |
| One Isaac Sim process | about 17–35 GB (depends on the task and whether video is enabled) |

## Step 2: Run the evaluation

The example script `examples/embodied/cosmos3/eval/run_robolab_eval.sh` only does three things: set the environment variables above, point `PYTHONPATH` at LoongForge, and start the coordinator with the RoboLab venv. The coordinator launches policy servers itself from `server.python` in the YAML. The repository ships only the config used for the actual parallel run, `configs/robolab/cosmos3.yaml`; derive a smoke config from it (see §2.1).

### 2.1 Smoke test (single task, single GPU)

Copy `cosmos3.yaml` and change only the following (keep the other sections, including `model:` and `server.ckpt_path`, unchanged, and fill in the path placeholders from Step 0):

```yaml
benchmark:
  name: robolab
  task_name: BananaInBowlTask     # replaces the tasks list; one task = one shard
  num_envs: 1
  episodes_per_task: 1
  max_steps: 0
  headless: true
  device: cuda:0
  return_action_chunk: true
# Remove the servers / pools sections: the coordinator then uses 1 server + 1 simulator
# on GPU0 (i.e. the card selected by CUDA_VISIBLE_DEVICES)
run:
  output_dir: /tmp/robolab_smoke
```

```bash
cd /path/to/LoongForge
BENCHMARK_PYTHON=/path/to/RoboLab/.venv/bin/python \
CONFIG=/tmp/robolab_smoke.yaml CUDA_VISIBLE_DEVICES=2 \
  bash examples/embodied/cosmos3/eval/run_robolab_eval.sh
```

The first run includes cold caching and takes about 10 minutes; `validation.ok=true` plus a successful episode means the pipeline works.

### 2.2 Full run (120 tasks × 10 episodes)

```bash
cd /path/to/LoongForge
BENCHMARK_PYTHON=/path/to/RoboLab/.venv/bin/python \
  bash examples/embodied/cosmos3/eval/run_robolab_eval.sh
# equivalent to CONFIG=examples/embodied/cosmos3/eval/configs/robolab/cosmos3.yaml
```

The layout in `cosmos3.yaml`: **6 policy servers (2 each on GPU0/1/2) + 6 Isaac Sim workers (GPU3–7, `n_sims_per_server: 1`)**. Before running, replace the `/path/to/...` placeholders (checkpoint, VAE, LoongForge root, output directory) with real paths.

Do not set `CUDA_VISIBLE_DEVICES` outside the script for multi-GPU runs; GPU placement is fully controlled by the YAML:

- `servers[].gpus`: the GPU of each policy server; its length is the number of servers (duplicates mean several processes on one card);
- `pools.gpu_sim.placement`: `separate` runs simulators on `sim_gpus`, `co_locate` puts them on the same card as their server;
- `pools.gpu_sim.n_sims_per_server`: how many simulator workers each server gets.

### 2.3 Key config fields

| Field | Meaning |
|---|---|
| `benchmark.tasks` / `task_name` | Task list / single task; one task is one shard, `plan_robolab` splits on this |
| `benchmark.num_envs` / `episodes_per_task` | Environments stepped in lockstep / episodes per task (must divide evenly) |
| `benchmark.return_action_chunk` | Client holds the 32-step chunk; otherwise the server caches it and returns one step at a time. Can also be set under `run:` (takes precedence) |
| `model.action_horizon` | Must match the checkpoint (32); `predict_action` returns `action_horizon - history_length` rows |
| `server.chunk_execute_steps` | Steps executed before replanning; unset for cosmos3 = the whole 31-step chunk |
| `server.start_timeout_sec` | Loading a 16B model takes time; 900 or more is recommended |
| `run.output_dir` | Result directory; the coordinator refuses to overwrite an existing result. `run.resume: true` reuses it (it currently re-evaluates all tasks and appends results, de-duplicated by `episode_id`; it does **not** re-run only the missing tasks) |

### 2.4 Results and report

`run_robolab_eval.sh` **summarizes automatically after the run**: when `coordinator_result.json` has `validation.ok=true`, it picks the leaderboard row from the config's `model_type` (`cosmos3` / `pi05`) and prints overall, difficulty and capability group success rates with the leaderboard comparison. If the run failed or the coverage check did not pass, it prints a skip notice and keeps the original exit code.

You can also run the report manually (e.g. to compare several result directories):

```bash
python3 -m loongforge.embodied.eval.metrics.robolab_report <output_dir> [cosmos3|pi05]
```

The RoboLab runner writes the difficulty/capability tags (`difficulty_label`, `task_attributes`) into every episode record, so new results are self-describing. **Older results** lack these two fields and need `ROBOLAB_ROOT=/path/to/robolab`, from which the script falls back to `robolab/tasks/_metadata/task_metadata.json`.

`<output_dir>/coordinator_result.json` holds the coverage check:

```json
{"validation": {"ok": true, "expected": 1200, "actual": 1200,
                "missing": 0, "duplicates": 0, "infra_failed": 0,
                "successes": 455, "failures": {"not_successful_within_max_steps": 745}}}
```

Only `ok=true` and `infra_failed=0` make the measurement valid; `not_successful_within_max_steps` is a model failure, not an infrastructure failure.

## Verification

`nvidia/Cosmos3-Nano-Policy-DROID`, 120 tasks × 10 episodes, seed 0, **1200 episodes, 0 infrastructure failures**, 6 servers + 6 simulators. The reference is the Cosmos3-Nano-Policy row of the [RoboLab leaderboard](https://research.nvidia.com/labs/srl/projects/robolab/leaderboard.html).

| Group | LoongForge (cosmos3 native) | Leaderboard |
|---|---|---|
| **all** | **455/1200 = 37.9%** | 441 = 36.8% |
| simple | 274/640 = 42.8% | 40.6% |
| moderate | 126/390 = 32.3% | 35.4% |
| complex | 55/170 = 32.4% | 25.3% |
| visual | 230/840 = 27.4% | 26.1% |
| relational | 214/420 = 51.0% | 49.3% |
| procedural | 54/220 = 24.5% | 27.3% |

`all` means the aggregate success rate over **all 120 tasks, 10 episodes each, 1200 episodes in total**; the simple/moderate/complex rows split by task difficulty and visual/relational/procedural by task attributes, all subsets of those 1200 episodes. The leaderboard's `all` is likewise the same 120 tasks × 1200 episodes.

Difference: +1.2 pt versus the leaderboard (z=0.59), not significant.

## Verification: pi05

The same benchmark and protocol also run with `pi05_droid_jointpos` (π0.5) through the native pi05 path (`model_type: pi05`). Config: `examples/embodied/pi05/eval/configs/robolab/pi05.yaml`; launcher: `examples/embodied/pi05/eval/run_robolab_eval.sh`. Because the 3.3B model needs far less memory, the layout is **8 GPUs, each hosting one policy server and one simulator (`co_locate`)**.

`pi05_droid_jointpos`, 120 tasks × 10 episodes, seed 0, **1200 episodes, 0 infrastructure failures**. The reference is the π0.5 row of the leaderboard.

| Group | LoongForge (pi05 native) | Leaderboard (π0.5) |
|---|---|---|
| **all** | **325/1200 = 27.1%** | 336/1200 = 28.0% |
| simple | 183/640 = 28.6% | 29.7% |
| moderate | 113/390 = 29.0% | 31.5% |
| complex | 29/170 = 17.1% | 13.5% |
| visual | 178/840 = 21.2% | 23.8% |
| relational | 138/420 = 32.9% | 33.8% |
| procedural | 50/220 = 22.7% | 21.8% |

Difference: overall −0.9 pt (z=−0.50), not significant; every group has |z| < 1.3. `moderate` is the largest negative gap (−2.6 pt) and `complex` is positive (+3.5 pt).

