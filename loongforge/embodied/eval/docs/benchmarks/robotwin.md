# RoboTwin Evaluation

RoboTwin is a dual-arm tabletop manipulation benchmark. The LoongForge eval module reuses RoboTwin's official evaluator, which calls the policy through a plugin (the `action_bridge` protocol). Two model × bridge combinations reach task success.

## Step 0: Download weights

| Model | Weights |
|---|---|
| pi05 | [motus-robotics/pi0.5_robotwin2](https://huggingface.co/motus-robotics/pi0.5_robotwin2) (openpi `norm_stats`, see [§2](#step-2-run-evaluation)) |
| xvla | [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2) |

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
```

Environment variables: `CONFIG`, `BENCHMARK_PYTHON`, `CUDA_VISIBLE_DEVICES`, plus the SAPIEN Vulkan variables (`LD_LIBRARY_PATH` / `VK_ICD_FILENAMES`).

Key config fields (see `examples/embodied/<model>/eval/configs/robotwin/adjust_bottle_smoke.yaml`):

- `benchmark.action_bridge` — selects the official-evaluator protocol:

| `action_bridge` | Model | Protocol |
|---|---|---|
| `pi05_aloha_14d` | pi05 | openpi Aloha joint protocol; `model.action_dim: 14`, `action_horizon: 32` |
| `ee6d_dual` | xvla | X-VLA dual-arm end-effector protocol; `model.domain_id: 6` |

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

## Outputs

In addition to the standard outputs (see the [user guide §4](../user_guide_en.md#4-outputs)), RoboTwin collects the official evaluator logs, deploy config, result file, and `mp4` videos under `artifacts/robotwin/<task_name>/<task_config>/`, and writes one `results.jsonl` row per completed episode.
