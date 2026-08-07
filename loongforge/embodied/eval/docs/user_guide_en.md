# LoongForge-VLA Offline Evaluation — User Guide

LoongForge-VLA offline evaluation runs a policy against a simulation benchmark from a single YAML config. You point the CLI at one `--config <yaml>` file; it launches the benchmark simulator and an independent model server, runs the rollout, and writes results.

---

## 1. What's supported

| | LIBERO | CALVIN | SimplerEnv (WidowX) | RoboTwin | ManiSkill |
|---|---|---|---|---|---|
| **pi05** | ✅ task success ([lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)) | connectivity only | connectivity only | ✅ task success ([motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2)) | connectivity only |
| **xvla** | ✅ task success ([2toINF/X-VLA-LIBERO](https://huggingface.co/2toINF/X-VLA-LIBERO)) | connectivity only | ✅ task success ([2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX)) | ✅ task success ([2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2)) | connectivity only |
| **GR00T-N1.6** | — | — | ✅ task success ([nvidia/GR00T-N1.6-bridge](https://huggingface.co/nvidia/GR00T-N1.6-bridge)) | — | — |

- **Task success**: at least one episode passed the benchmark's official success criterion.
- **Weights**: parentheses show the Hugging Face weight (`org/name`) that achieved the run.
- **Connectivity only**: the pipeline runs with `random_init: true` and no score yet — the domain weights are missing or the benchmark assets block a full run (e.g. CALVIN needs the original-format validation dataset even where weights are public).
- **—**: not supported yet — coming soon.
- **Detailed results**: see the per-benchmark [benchmark pages](benchmarks/libero.md).

Any model that implements the shared `predict_action(images, instructions, state=None, dataset_stats=None)` interface can be added — see [§6](#6-adding-a-new-model).

---

## 2. Quick start

The example below runs LIBERO with pi05:

1. **Set up the LIBERO environment** — create a dedicated conda environment for the benchmark simulator, install LIBERO following the official [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) instructions, then install the eval client dependencies:

    ```bash
    conda create -n libero python=3.10 -y
    conda activate libero
    pip install websockets msgpack pyyaml
    pip install numpy==1.24.4   # pinned for simulator compatibility
    ```

    For the eval client deps, common issues, and the verified version list, check the [LIBERO guide](benchmarks/libero.md#step-1-environment-setup); the verified environment version lists are in [benchmark_envs.md](benchmark_envs.md).
2. **Get the weights and edit the config** — download [lerobot/pi05_libero_finetuned_v044](https://huggingface.co/lerobot/pi05_libero_finetuned_v044), then fill the `/path/to/...` placeholders in `examples/embodied/pi05/eval/configs/libero/object_smoke.yaml`:

    ```yaml
    server:
      python: /path/to/loongforge/bin/python            # model server env (loongforge)
      ckpt_path: /path/to/pi05_libero_finetuned_v044
      dataset_statistics_path: /path/to/pi05_libero_finetuned_v044/dataset_statistics.json
      tokenizer_path: /path/to/paligemma-3b-pt-224
      loongforge_root: /path/to/LoongForge-VLA
      log: /path/to/.../policy_server.log
    env:
      eval_root: /path/to/LoongForge-VLA/loongforge/embodied/eval
      libero_config_path: /path/to/libero_config
      ld_library_path: /path/to/nvidia_lib
    run:
      output_dir: /path/to/.../reports/pi05/libero/object_smoke
    ```

    Unchanged fields (e.g. `benchmark:`, `model:`, `timeouts:`) can be left as-is.

3. **Run** — execute the script inside the **benchmark** environment (the script launches the orchestrator with `BENCHMARK_PYTHON`; the policy server is started from `server.python`):

    ```bash
    cd /path/to/LoongForge
    examples/embodied/pi05/eval/run_libero_eval.sh
    ```

The LIBERO simulator runs in the benchmark environment; the policy server is launched from the YAML `server.python` field, pointing at the LoongForge environment.

The run scripts also accept environment overrides (`CONFIG`, `REPO_ROOT`, `BENCHMARK_PYTHON`, `CUDA_VISIBLE_DEVICES`, and for SAPIEN benchmarks `LD_LIBRARY_PATH` / `VK_ICD_FILENAMES`), or you can invoke the orchestrator directly:

```bash
python -m loongforge.embodied.eval.orchestrator.run --config /path/to/config.yaml
```

Run the orchestrator command inside the **benchmark** environment; the config's `server.python` points at the **model server** environment. Other models/benchmarks: the [benchmark pages](benchmarks/libero.md).

---

## 3. Configuration reference

A config has five sections: `benchmark`, `model`, `server`, `run`, `timeouts`.

```yaml
benchmark:
  name: libero               # libero | calvin | simplerenv | robotwin | maniskill
  suite: libero_object       # benchmark-specific
  max_tasks: 1
  episodes_per_task: 1
  max_steps: 300
  num_steps_wait: 10

model:
  backend: loongforge        # loongforge | mock
  model_type: pi05           # REQUIRED (no default) — pi05 | xvla | Gr00tN1d6
  action_dim: 7
  action_horizon: 50
  # Optional model capability fields (have defaults; usually omitted):
  #   state_encoding   proprio encoding the model consumes
  #   action_encoding  the model's action encoding
  #   domain_id        xvla multi-embodiment id (auto by benchmark if omitted)

server:
  host: 127.0.0.1
  port: 12093
  health_port: 12094
  python: /path/to/model-server-env/bin/python
  log: /path/to/policy_server.log
  start_timeout_sec: 900
  ckpt_path: /path/to/weight_dir
  dataset_statistics_path: /path/to/dataset_statistics.json
  tokenizer_path: /path/to/paligemma-3b-pt-224
  use_bf16: false
  loongforge_root: /path/to/LoongForge-VLA
  random_init: false         # true = run with random weights (connectivity check)

run:
  output_dir: /path/to/reports/pi05/libero/object_smoke
  seed: 7
  save_trace: true
  save_replay: true

timeouts:
  policy_call_ms: 600000
  per_step_sec: 600
  per_episode_sec: 900
```

Key fields:

- `benchmark.name` — selects the benchmark runner.
- `model.model_type` — **required**; selects the model factory / PayloadBuilder (`pi05` | `xvla` | `Gr00tN1d6`). There is no default — the eval server fails fast if it is missing.
- `model.backend` — `loongforge` for a real model, `mock` for a pipeline-only check (no model weights; the server returns mock actions to validate the eval chain).
- `model:` — model-structure fields (`action_dim`, `action_horizon`, …) plus optional capability fields (`state_encoding` / `action_encoding` / `domain_id`). The fields are **model-specific** — pi05, xvla, and GR00T-N1.6 declare different structural fields (see the per-model configs under `examples/embodied/<model>/eval/configs/` and the [model integration guide](model_integration.md)). Defaults are sensible per model, so you rarely set the capability fields by hand.
- `server.ckpt_path` — a directory with `model.safetensors` (or the weight file). Set `server.random_init: true` to run without weights.
- `server.dataset_statistics_path` — action-normalization stats the model uses internally (e.g. pi05).
- `server.python` — the model server interpreter.
- `run.output_dir` — where results are written.
- Set `run.save_replay: false` and `run.save_trace: false` to keep only the CSV/JSONL summaries.

Every config comes as a **public template** (with `/path/to/...` placeholders, meant to be edited) plus a matching launch script. Optional knobs (larger suites, more episodes) are documented in each config's header comments — raise `max_tasks` / `episodes_per_task` in the same file.

---

## 4. Outputs

LIBERO / CALVIN / SimplerEnv / ManiSkill write under `run.output_dir`:

| File | Meaning |
|---|---|
| `results.jsonl` | per-episode results |
| `summary.csv` | task-level aggregate |
| `suite_summary.csv` | suite-level aggregate |
| `artifacts/.../*.gif` | replay (when `save_replay: true`; e.g. LIBERO `replay_*.gif`, SimplerEnv/ManiSkill `ep*_*.gif`) |
| `artifacts/.../*trace*.json` | per-step action trace (when `save_trace: true`; e.g. LIBERO/CALVIN `trace_*.json`, SimplerEnv/ManiSkill `*_trace.json`) |
| `policy_server.log` | model server stdout/stderr |

RoboTwin additionally collects the official evaluator logs, deploy config, result file, and any `mp4` videos under `artifacts/robotwin/<task_name>/<task_config>/`, and writes one `results.jsonl` row per completed episode so aggregation matches the other benchmarks.

`run.output_dir` is a stable run tag; by default the orchestrator writes a timestamped subdirectory so previous results are never overwritten. Set `run.timestamped_output: false` to reuse a fixed directory.

---

## 5. Troubleshooting

**Vulkan / SAPIEN (SimplerEnv, RoboTwin, ManiSkill).** These render with SAPIEN and need a working NVIDIA Vulkan ICD. Check with `vulkaninfo` (not just `nvidia-smi`):

```bash
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64 \
VK_ICD_FILENAMES=/path/to/nvidia_icd.json \
vulkaninfo
```

Expect `deviceName = NVIDIA ...` / `driverName = NVIDIA`. If you see only `llvmpipe` / `lavapipe`, camera images and replays are unreliable. Set `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, and `XDG_RUNTIME_DIR` **before** SAPIEN is imported; the runners re-exec the process so these take effect.

**MuJoCo (LIBERO, CALVIN).** Keep the `MUJOCO_GL` / `PYOPENGL_PLATFORM` and benchmark config-path settings from the shipped configs.

**Disk space.** Replay GIFs are the largest artifact. Set `run.save_replay: false` (and `save_trace: false`) if a run fails while writing artifacts.

---

## 6. Adding a new model

In addition to the built-in models, you can add a new model by implementing the shared `predict_action(images, instructions, state=None, dataset_stats=None)` interface — the eval server calls it over RPC, and model-specific preprocessing / normalization live inside it. Then plug into three small pieces (benchmark runners and adapters are reused unchanged):

1. **Model factory** (`factories/<model>_factory.py`): loads the model + weights and returns it behind the shared `predict_action(images, instructions, state=None, dataset_stats=None)` interface.
2. **PayloadBuilder** (`payload_builders/<model>.py`): turns a benchmark observation into the model's `predict_action` inputs, and declares the model's capabilities (`state_encoding` / `action_encoding` / `domain_id`).
3. **ActionDecoder** (`action_decoders/`, optional): converts the model's raw actions into the benchmark's action space. If the model's action encoding already matches the benchmark, this is a no-op and nothing is needed.

Action normalization/unnormalization always lives inside the model's `predict_action`; eval only passes stats through. The full step-by-step checklist (with pi05 and xvla as worked examples) and the exact interface contract are in the [model integration guide](model_integration.md).
