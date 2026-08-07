# CALVIN Evaluation

CALVIN is a long-horizon Franka language-conditioned manipulation benchmark: each sequence chains five subtasks. The LoongForge eval module uses the original-format CALVIN validation dataset for official online long-horizon rollout.

Current status: **connectivity checks only**. The main blocker is the benchmark dataset — the official online rollout needs the original-format CALVIN validation assets (`validation/` tree, see Step 1). On top of that, no CALVIN-domain weights are released for pi05 (xvla's CALVIN weights are public: [2toINF/X-VLA-Calvin-ABC_D](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D)). Shipped configs run with `server.random_init: true`, not scores.

## Step 0: Download weights

| Model | Weights |
|---|---|
| pi05 | none released (connectivity only) |
| xvla | [2toINF/X-VLA-Calvin-ABC_D](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D) (open-source, connectivity only) |

## Step 1: Environment setup

### Standard environment

Install CALVIN following the official [CALVIN repository](https://github.com/mees/calvin) instructions, then install the additional dependencies:

```bash
pip install websockets msgpack pyyaml
```

The evaluation needs the original-format CALVIN validation assets, not just a LeRobot-format dataset:

- `validation/` tree containing `validation/.hydra/merged_config.yaml`, `calvin_models/conf`, and `eval_sequences.json`
- LeRobot-format CALVIN datasets are useful for training/statistics but are not sufficient by themselves for the official online rollout

⚠️ Common issue: CALVIN uses MuJoCo; keep `MUJOCO_GL` / `PYOPENGL_PLATFORM` settings from the shipped configs.

## Step 2: Run evaluation

Run from inside the **benchmark** environment. The run scripts and eval YAMLs ship with `/path/to/...` placeholders — fill them in before running:

```bash
cd /path/to/LoongForge-VLA
examples/embodied/pi05/eval/run_calvin_eval.sh    # pi05 (connectivity only)
examples/embodied/xvla/eval/run_calvin_eval.sh    # xvla (connectivity only)
```

Environment variables: `CONFIG`, `BENCHMARK_PYTHON` (CALVIN env interpreter), `CUDA_VISIBLE_DEVICES`.

Key config fields (see `examples/embodied/<model>/eval/configs/calvin/smoke.yaml`):

- `benchmark.dataset_path` — path to a tree containing `validation/`
- `server.random_init` — `true` in shipped configs (connectivity check). With CALVIN-domain weights set `false`, fill `ckpt_path` / `dataset_statistics_path`; for xvla set `model.domain_id: 2`

## Verification

| Model | Status | Notes |
|---|---|---|
| pi05 | connectivity only | needs CALVIN-domain weights + `dataset_statistics.json` |
| xvla | connectivity only | needs X-VLA CALVIN (ABC_D) weights ([2toINF/X-VLA-Calvin-ABC_D](https://huggingface.co/2toINF/X-VLA-Calvin-ABC_D)) |

Connectivity run: 1 sequence, first subtask capped at 30 steps.
