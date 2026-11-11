# 离线评测

LoongForge 在 `loongforge/embodied/eval` 下提供了离线评测模块，通过 WebSocket / msgpack-numpy RPC 将基准测试客户端和模型策略服务器作为两个独立进程连接。目前支持 LoongForge **pi0.5** 和 **X-VLA** 作为策略后端，覆盖五个基准测试：**LIBERO**、**CALVIN**、**SimplerEnv**、**RoboTwin (2.0)** 和 **ManiSkill**。用户侧只需要一个 YAML 配置文件和一个启动脚本，无需修改 Python 入口代码。本指南涵盖支持范围、运行方法、各基准测试的配置方式，以及如何扩展新模型。

---

## 1. 支持矩阵

### 1.1 当前可运行的组合

| 模型 | LIBERO | CALVIN | SimplerEnv (WidowX) | RoboTwin 2.0 | ManiSkill |
|---|---|---|---|---|---|
| **pi0.5** | 任务成功（微调后） | 尚未评分 — 无匹配的开放权重 | 尚未评分 — 无匹配的开放权重 | 任务成功（`pi0.5_robotwin2` + `pi05_aloha_14d`） | 尚未评分 — 无匹配的开放权重 |
| **X-VLA** | 任务成功（~94% object suite） | 尚未评分 — 无匹配的开放权重 | 任务成功（X-VLA-WidowX + SimplerEnv 补丁） | 任务成功（X-VLA-RoboTwin2 + `ee6d_dual`） | 尚未评分 — 无匹配的开放权重 |

"任务成功"表示至少有一个 episode 通过了官方或本地的成功判定标准。"尚未评分"的组合仍附带可运行的 YAML（设置了 `server.random_init: true`），无需加载任何模型权重即可验证 RPC 通路。

### 1.2 可直接运行的组合

以下每个组合都有发布的 YAML，只需将其中的 `/path/to/...` 替换为实际路径即可。

| 组合 | 公开权重 | 配置文件 |
|---|---|---|
| pi0.5 + LIBERO | pi0.5 LIBERO 微调（openpi `pi05_libero` 系列或本地 `model.safetensors` + `dataset_statistics.json`） | `examples/embodied/pi05/eval/configs/libero/object_smoke.yaml` |
| pi0.5 + RoboTwin | pi0.5 RoboTwin-2.0 联合微调 + 统计信息（见 *4.2 章节*） | `examples/embodied/pi05/eval/configs/robotwin/adjust_bottle_smoke.yaml` |
| X-VLA + LIBERO | [2toINF/X-VLA-LIBERO](https://huggingface.co/2toINF/X-VLA-LIBERO) | `examples/embodied/xvla/eval/configs/libero/libero_weight_object_smoke.yaml` |
| X-VLA + RoboTwin | [2toINF/X-VLA-RoboTwin2](https://huggingface.co/2toINF/X-VLA-RoboTwin2) | `examples/embodied/xvla/eval/configs/robotwin/adjust_bottle_smoke.yaml` |
| X-VLA + SimplerEnv | [2toINF/X-VLA-WidowX](https://huggingface.co/2toINF/X-VLA-WidowX) | `examples/embodied/xvla/eval/configs/simplerenv/widowx_stack_cube_smoke.yaml` |

X-VLA + SimplerEnv 达成任务成功需要上游 SimplerEnv 的一次性补丁 — 参见 *6.3 章节*。

### 1.3 各组合关键协议字段

| 组合 | 必需配置 |
|---|---|
| pi0.5 + LIBERO | `action_dim: 7`、`action_horizon: 50`；匹配的 `dataset_statistics.json`（q01/q99） |
| pi0.5 + RoboTwin | `action_dim: 14`、`action_horizon: 32`；`action_bridge: pi05_aloha_14d`；`server.dataset_statistics_path`（见 *4.2 章节*） |
| X-VLA + LIBERO | `domain_id: 3`；`action_postprocess: ee6d_to_axis_angle`；`server.state_format: ee6d`；`max_steps: 800` |
| X-VLA + RoboTwin | `domain_id: 6`；`action_bridge: ee6d_dual` |
| X-VLA + SimplerEnv | `domain_id: 0`；`max_steps: 1200`；`control_mode: arm_pd_ee_target_base_pose_gripper_pd_joint_pos`；`action_postprocess: ee6d_to_simpler_abs_euler`；需要 SimplerEnv 补丁 |

---

## 2. 快速开始

### 2.1 环境

涉及两个 conda 环境，有意保持分离：

- **基准测试侧**（每个基准测试一个） — LIBERO、CALVIN、SimplerEnv、RoboTwin 或 ManiSkill。示例：`/path/to/envs/libero/bin/python`。
- **模型侧** — 与训练使用相同的 LoongForge 环境。示例：`/path/to/envs/loongforge/bin/python`。

GPU 要求：模型推理需要至少 1 张 NVIDIA GPU（显存 ≥16 GB，推荐 A100/A800）。基准测试侧渲染（SimplerEnv、RoboTwin、ManiSkill）同样要求 GPU 支持 Vulkan。

各基准测试的依赖和已知兼容版本记录在 [benchmark_envs.md](https://github.com/baidu-baige/LoongForge/blob/master/loongforge/embodied/eval/benchmark_envs.md) 中。

### 2.2 运行 pi0.5 + LIBERO

```bash
cd /path/to/LoongForge

# 1. 编辑 YAML 中的 /path/to/...：
#    examples/embodied/pi05/eval/configs/libero/object_smoke.yaml
# 2. 启动：
examples/embodied/pi05/eval/run_libero_eval.sh
```

启动脚本封装了单一 Python 入口：

```bash
"${BENCHMARK_PYTHON}" -m loongforge.embodied.eval.orchestrator.run \
  --config "${CONFIG}"
```

可覆盖的环境变量：`REPO_ROOT`、`CONFIG`、`BENCHMARK_PYTHON`、`CUDA_VISIBLE_DEVICES`、`LD_LIBRARY_PATH`、`VK_ICD_FILENAMES`。

### 2.3 运行 X-VLA + LIBERO

```bash
cd /path/to/LoongForge
# 编辑 configs/libero/libero_weight_object_smoke.yaml 中的 /path/to/...
examples/embodied/xvla/eval/run_libero_eval.sh
```

默认 YAML 运行一个任务 × 一个 episode。若要运行完整 object suite，在同一 YAML 中增大 `max_tasks` 和 `episodes_per_task` — 文件头注释列出了可选项。

---

## 3. 配置

### 3.1 YAML 结构

最小骨架（pi0.5 + LIBERO）：

```yaml
benchmark:
  name: libero
  suite: libero_object      # 可选：libero_spatial | libero_goal | libero_10
  max_tasks: 1
  episodes_per_task: 1
  max_steps: 300
  num_steps_wait: 10

model:
  backend: loongforge       # 或 `mock` 用于连通性检查
  model_type: pi05          # 或 `xvla`
  name: loongforge-pi05
  action_dim: 7
  state_dim: 7
  action_horizon: 50
  max_action_dim: 32
  max_state_dim: 32
  compile_model: false

server:
  host: 127.0.0.1
  port: 12093
  health_port: 12094
  python: /path/to/envs/loongforge/bin/python
  log: /path/to/reports/pi05/libero/object_smoke/policy_server.log
  start_timeout_sec: 900
  ckpt_path: /path/to/checkpoint_or_model_dir
  dataset_statistics_path: /path/to/dataset_statistics.json
  tokenizer_path: /path/to/paligemma-3b-pt-224
  use_bf16: false
  loongforge_root: /path/to/LoongForge

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

### 3.2 字段说明

| 字段 | 含义 |
|---|---|
| `benchmark.name` | 选择运行器：`libero`、`calvin`、`simplerenv`、`robotwin`、`maniskill`。 |
| `model.backend` | `loongforge` 启动真实模型服务器。`mock` 启动仅协议服务器用于连通性检查。 |
| `model.*` | 与各 `ModelConfig` dataclass 匹配的模型结构字段（`action_dim`、`action_horizon`、`compile_model` 等）。 |
| `server.ckpt_path` | 包含 `model.safetensors` 的目录，或权重文件本身。 |
| `server.random_init: true` | 跳过 checkpoint 加载；在无匹配的开放权重时有用。 |
| `server.dataset_statistics_path` | 传递给模型 `predict_action()` 用于模型内部反归一化。 |
| `server.chunk_execute_steps`（仅 X-VLA） | 开环执行步数截断（open-loop horizon truncation）。`10` = 官方 X-VLA LIBERO 风格；`0` = 工厂默认值（X-VLA 为 10）；`-1` = 不截断。 |
| `server.python` | 用于启动策略服务器的 Python 解释器。 |
| `server.log` | 策略服务器日志文件路径。惯例为运行目录下的 `policy_server.log`。 |
| `run.output_dir` | 运行目录。默认为带时间戳的子目录（见 *5.2 章节*）。 |
| `run.timestamped_output` | 默认 `true`；设为 `false` 可重用固定的 `output_dir`（如调试时）。 |
| `run.save_trace` / `run.save_replay` | 磁盘紧张时同时禁用，仅保留 `results.jsonl`、`summary.csv`、`suite_summary.csv` 和策略日志。 |

协议控制参数（`benchmark.action_bridge`、`benchmark.domain_id`、`benchmark.action_postprocess`、`server.state_format` 等）仅在 YAML 中配置，且针对各基准测试不同。具体使用见 *1.3 章节* 和 *4 章节* 各基准测试指南。

### 3.3 配置文件位置

```text
examples/embodied/<model>/eval/
  configs/
    <benchmark>/*.yaml     # 每个基准测试一个发布的 YAML
  run_<benchmark>_eval.sh  # 启动脚本
```

所有发布的 YAML 使用 `/path/to/...` 占位符，请勿提交包含机器特定绝对路径的配置文件。

---

## 4. 各基准测试指南

每个小节介绍默认 YAML 运行的内容、完整扫描时需要调整的参数，以及该基准测试的特殊注意事项。

### 4.1 LIBERO

发布的 YAML：

- pi0.5（任务成功）：`examples/embodied/pi05/eval/configs/libero/object_smoke.yaml`
- X-VLA（任务成功）：`examples/embodied/xvla/eval/configs/libero/libero_weight_object_smoke.yaml`

默认 suite 为 `libero_object`。在同一 YAML 中修改 `suite`（`libero_object`、`libero_spatial`、`libero_goal`、`libero_10`）、`max_tasks` 和 `episodes_per_task` 即可进行完整扫描。X-VLA 通常需要 `max_steps: 800` 和 `chunk_execute_steps: 10`。

#### 场景快照（`per_task` + episode 级分片）

`benchmark.env_seed_mode: per_task` 下，一个 episode 看到的场景取决于「本进程已 reset 过多少次」，所以一个 task 不能拆到多个 worker。场景快照记录的是每次 reset 之前的 RNG 状态，回放它就能让任意 episode 在任意 worker 里复现同一个场景。按 `(suite, seed, episodes_per_task)` 生成一次，然后让 `benchmark.scene_snapshot_dir` 指向该目录。

```bash
# 单个 task
python -m loongforge.embodied.eval.orchestrator.scene_snapshots \
    --suite libero_10 --task 3 --seed 7 --episodes 50 --out /path/to/snapshots

# 一个 suite 的全部 task，10 路并行
python -m loongforge.embodied.eval.orchestrator.scene_snapshots \
    --suite libero_10 --seed 7 --episodes 50 --out /path/to/snapshots --jobs 10

# 多个 suite 写到同一目录（每个 task 是独立进程）
python -m loongforge.embodied.eval.orchestrator.scene_snapshots \
    --suites libero_10,libero_object,libero_goal,libero_spatial \
    --seed 7 --episodes 50 --out /path/to/snapshots --jobs 40
```

注意事项：

- **成本**：约 90–105 s/task（建 env + `episodes` 次 reset + 写 ~300 KB）。单 suite 10 个 task 串行约 15 min；4 个 suite 40 个 task 串行约 1 h，`--jobs 40` 在 128 核机器上 **~4 min**（实测 40 task × 50 快照 245 s）。纯 CPU、task 之间完全独立。
- **多 suite 必须放同一目录**：文件名自带 suite（`{suite}__task{k}__seed{seed}.pkl`），所以 `scene_snapshot_dir` 指向父目录即可被多 suite 并行评测直接使用。
- **`--seed` 必须与评测配置里的 `run.seed` 一致**（模板是 `7`）：`per_task` 模式下每个 task 的所有 episode 都用这一个 seed 建环境，快照按文件名查找，不一致就是「文件不存在」（runner 会直接报错并给出生成命令）。
- **`--episodes` 必须 ≥ `benchmark.episodes_per_task`**，否则运行时会报 `scene snapshot ... has N states but episode_idx=...`。生成阶段会把每个文件读回校验长度，不匹配直接非零退出，避免跑到一半才失败。
- 文件带 `SNAPSHOT_VERSION`，加载时校验版本；换版本或改了 benchmark 资源 / reset 逻辑后需要重新生成。
- 生成过程每完成一个 task 打印一行 `[i/N]`，结束时汇总 `written / failed`；单个 task 失败不影响其余 task，但退出码非零。

### 4.2 RoboTwin

RoboTwin 通过官方 `script/eval_policy.py` 启动；桥接代码位于 `loongforge/embodied/eval/bridges/robotwin_policy.py`。使用的协议在 YAML 中通过 `benchmark.action_bridge` 设置；当模型指定了 `model.robotwin_action_bridge` 时，该值覆盖 `benchmark.action_bridge`。

| `action_bridge` | 角色 | 控制方式 | 备注 |
|---|---|---|---|
| `strict_14d` | 默认 14D 关节动作 | `take_action` 关节 qpos | 模型必须输出 ≥14D；无 `adapt_to_pi`。 |
| `duplicate_7d` | 7D → 14D | `take_action` 关节 qpos | 仅连通性检查。非真实分数。 |
| `pi05_aloha_14d` | pi0.5 RoboTwin 正式协议 | 关节 qpos | openpi Aloha：`adapt_to_pi` 解码 state → 模型 → delta→abs → `adapt_to_pi` 编码。 |
| `ee6d_dual` | X-VLA RoboTwin 正式协议 | `take_action(..., action_type='ee')` | 20D ee6d，三个视角（head/left/right），本体感知在桥接内部从上一次发送的 EE 动作构建。 |

`strict_14d` 和 `duplicate_7d` 共享相同的默认代码路径；只有 `pi05_aloha_14d` 和 `ee6d_dual` 有专用协议处理。

#### pi0.5 + RoboTwin — 任务成功

关键 YAML 字段：

```yaml
benchmark:
  name: robotwin
  task_name: adjust_bottle
  task_config: demo_clean
  action_bridge: pi05_aloha_14d
  max_steps: 300
model:
  model_type: pi05
  action_dim: 14
  state_dim: 14
  action_horizon: 32
server:
  ckpt_path: /path/to/pi0.5_robotwin2
  dataset_statistics_path: examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json
```

统计文件 `pi05_robotwin2_dataset_stats.json` 来源于 pi0.5 RoboTwin-2.0 权重包中的 openpi `norm_stats.json`，仅将顶层键名重命名以适配 LoongForge 的 q99 反归一化：

| 项目 | 值 |
|---|---|
| 源文件（权重包内） | `assets/pi0.5_clean_randomize_joint_training/norm_stats.json` |
| 源结构 | `{"norm_stats": {"state": {mean,std,q01,q99}, "actions": {...}}}` |
| LoongForge 结构 | `{"observation.state": {...}, "action": {...}}` |
| 数值 | 与源文件完全相同；仅键名重命名。 |
| 仓库内副本 | `examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json` |

生成脚本（openpi 格式统计文件 → LoongForge 统计信息）：

```python
import json
from pathlib import Path

src = Path("/path/to/pi0.5_robotwin2/assets/pi0.5_clean_randomize_joint_training/norm_stats.json")
raw = json.loads(src.read_text())["norm_stats"]
out = {"observation.state": raw["state"], "action": raw["actions"]}
Path("pi05_robotwin2_dataset_stats.json").write_text(json.dumps(out, indent=2))
```

启动：

```bash
cd /path/to/LoongForge
CONFIG=examples/embodied/pi05/eval/configs/robotwin/adjust_bottle_smoke.yaml \
  examples/embodied/pi05/eval/run_robotwin_eval.sh
```

#### X-VLA + RoboTwin — 任务成功

```bash
cd /path/to/LoongForge
CONFIG=examples/embodied/xvla/eval/configs/robotwin/adjust_bottle_smoke.yaml \
  examples/embodied/xvla/eval/run_robotwin_eval.sh
```

与官方 `evaluation/robotwin-2.0` 协议对齐：`domain_id: 6`、`action_bridge: ee6d_dual`，权重如 `/path/to/X-VLA-RoboTwin2`。

#### 无匹配的开放权重时运行 RoboTwin

发布的 YAML 也支持连通性检查。编辑同一 YAML：

- `server.random_init: true`、`max_steps: 5` — 随机 14D 输出，验证官方评测器可达。
- `action_bridge: duplicate_7d`（配合 `model.action_dim: 7`） — 仅连通性检查。非真实分数。
- `action_bridge: strict_14d` — 原始 14D 关节，无 `adapt_to_pi`。非 openpi 正式协议。

### 4.3 SimplerEnv

X-VLA + SimplerEnv 是任务成功组合。pi0.5 无匹配的开放权重，因此其 YAML 以 `server.random_init: true` 运行，仅用于连通性验证。

发布的 YAML：

- pi0.5（仅连通性）：`examples/embodied/pi05/eval/configs/simplerenv/widowx_stack_cube_smoke.yaml`
- X-VLA（任务成功）：`examples/embodied/xvla/eval/configs/simplerenv/widowx_stack_cube_smoke.yaml`

通过同一 YAML 中的 `task_name`、`robot_setup` 和 `scene_name` 切换 Bridge 任务（eggplant、carrot、spoon 等）。

此外，运行器会在导入 SAPIEN 之前重新执行当前进程（re-exec），以确保 `LD_LIBRARY_PATH` 和 `VK_ICD_FILENAMES` 生效。

### 4.4 CALVIN

CALVIN 是一个长 horizon Franka 语言操作基准测试（每个序列 5 个子任务）。指标：`success_count`、平均长度、每任务成功率。`benchmark.dataset_path` 需要指向包含 `validation/` 的目录树。

pi0.5 和 X-VLA 目前均无匹配的开放权重，因此发布的 YAML 使用 `server.random_init: true` 仅用于连通性验证：

- `examples/embodied/pi05/eval/configs/calvin/smoke.yaml`
- `examples/embodied/xvla/eval/configs/calvin/smoke.yaml`

有匹配的 CALVIN 领域开放权重时，设置 `server.random_init: false` 并填写 `server.ckpt_path` / `server.dataset_statistics_path`。对于 X-VLA，正式协议为 `domain_id: 2`、`action_postprocess: ee6d_to_calvin_abs`。

### 4.5 ManiSkill

ManiSkill 是基于 SAPIEN 的 GPU 友好操作套件。pi0.5 和 X-VLA 目前均无匹配的开放权重，因此发布的 YAML 以 `server.random_init: true` 运行 PickCube：

- `examples/embodied/pi05/eval/configs/maniskill/pick_cube_smoke.yaml`
- `examples/embodied/xvla/eval/configs/maniskill/pick_cube_smoke.yaml`

默认值：`PickCube-v1`、`pd_ee_delta_pose`、7D 动作。在 YAML 中修改 `task_name` 和 `obs_mode`（rgbd 或 state）。

本体感知说明：对于 Panda 机器人，当 `qpos` 为 9D（7 臂 + 2 手指）时，适配器输出 8D `model_state` = 7 关节 + 两个手指关节的均值（与 RLinf / openpi ManiSkill 布局一致）。结构化 `state` 仍保留完整 `qpos`。仅打包 `base_camera` 视角。

> **注意：这不同于 RLinf PutOnPlate。** RLinf 的 pi0.5 ManiSkill SFT 训练在 `PutOnPlateInScene25Main-v3`（WidowX Bridge real2sim）上，而非标准 `PickCube-v1` + Panda。LoongForge 不提供该环境；PickCube YAML 仅用于验证 ManiSkill 运行器和 RPC 通路。

---

## 5. 输出

### 5.1 文件布局

在 `run.output_dir` 下，直接 rollout 的基准测试（LIBERO、CALVIN、SimplerEnv、ManiSkill）写入：

| 文件 | 含义 |
|---|---|
| `results.jsonl` | 每个 episode 一行。 |
| `summary.csv` | 任务级聚合。 |
| `suite_summary.csv` | Suite 级聚合。 |
| `artifacts/.../replay_*.gif` | 可选的回放 GIF。 |
| `artifacts/.../trace_*.json` | 可选的逐步动作 trace。 |
| `policy_server.log` | 策略服务器 stdout/stderr。 |

磁盘紧张或做连通性检查时，设置 `run.save_replay: false` 和 `run.save_trace: false`，仅保留结果文件和策略日志。

RoboTwin 使用官方评测器，并额外收集部署配置、`_result.txt`、桥接 `trace.json` 以及可用的 `mp4` 视频，存放在 `run.output_dir/artifacts/robotwin/<task_name>/<task_config>/` 下。视频保持官方 `mp4` 格式；不强制转换为 GIF。

### 5.2 目录约定

```text
loongforge/embodied/eval/reports/
  <model>/
    <benchmark>/
      <run_name>/
        policy_server.log
        results.jsonl
        summary.csv
        suite_summary.csv
        artifacts/
          ...
```

`run.output_dir` 应指向一个稳定的运行标签目录，如 `reports/pi05/robotwin/adjust_bottle_smoke`。`run.timestamped_output` 默认为 `true`，会在父目录下创建 `<yyyymmdd_hhmmss>_<run_tag>`。`reports/` 仅本地使用，不提交。调试时若要重用固定目录，设置 `run.timestamped_output: false`。

### 5.3 RoboTwin 逐 episode 聚合

官方 RoboTwin 评测器循环直到收集 `test_num` 个**有效** episode（expert_check 可能跳过某些 seed），并在 `_result.txt` 中写入单一成功率。LoongForge **不会**将其折叠为一行。

运行器解析官方日志中形如 `Success rate: suc/test_num => …, current seed: S` 的行，并为每个完成的 episode 写入一条 `results.jsonl` 记录，`success` ∈ {0, 1}。每条记录还附带总体 `success_rate` 和 `n_episodes`，使 `summary.csv` 与 LIBERO 风格的逐 episode 聚合一致。

如果日志解析未找到 episode 行，运行器回退到从 `_result.txt` 生成单行记录。

---

## 6. 故障排除

### 6.1 SAPIEN / Vulkan（SimplerEnv、RoboTwin、ManiSkill）

基于 SAPIEN 的基准测试需要可用的 Vulkan ICD，不仅仅是可见的 GPU。如果 `vulkaninfo` 仅报告 `llvmpipe` / `lavapipe`，视觉观测、相机和回放可能崩溃；state-only rollout 通过不代表视觉通路正常。

快速检查：

```bash
LD_LIBRARY_PATH=/path/to/nvidia_lib:/usr/lib64:${LD_LIBRARY_PATH:-} \
VK_ICD_FILENAMES=/path/to/nvidia_lib/10_nvidia.json \
vulkaninfo
```

期望看到 `deviceName = NVIDIA ...` 和 `driverName = NVIDIA`。在导入 SAPIEN / svulkan2 / ManiSkill / 渲染器**之前**设置 `LD_LIBRARY_PATH`、`VK_ICD_FILENAMES` 和 `XDG_RUNTIME_DIR`；SimplerEnv 运行器正是因此会重新执行 Python 进程（re-exec）。Python 启动后再修改 `LD_LIBRARY_PATH` 通常无效。

### 6.2 磁盘空间

长时间 LIBERO 扫描曾因保存回放 artifact 时遇到 `No space left on device` 错误 — 模型服务器本身正常。设置 `run.save_replay: false` 和 `run.save_trace: false`，仅保留 `results.jsonl`、`summary.csv`、`suite_summary.csv` 和 `policy_server.log`。

### 6.3 SimplerEnv 绝对 EE 控制（X-VLA）

上游 `simpler-env/SimplerEnv` 默认不提供绝对 EE 控制，但 X-VLA 输出绝对 EE 位姿。两种解决方案：

- 使用 [`255isWhite/SimplerEnv`](https://github.com/255isWhite/SimplerEnv) fork（官方 X-VLA SIMPLER 评测使用的 fork），或
- 应用 `examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md` 中记录的两个本地补丁。

若两者都未应用，环境要么在构造时因缺少控制模式报错，要么静默地将绝对动作当作增量动作应用从而永远无法成功。

---

## 7. 添加新模型

本节面向需要集成新模型的开发者。如需集成 pi0.5 / X-VLA 之外的模型，核心思路是：复用共享的 `predict_action` 接口和 `GenericPredictActionPolicy`，保持基准测试协议和适配器不变，将模型差异放在一个轻量工厂中。不要 fork 基准测试运行器或直接修改 LoongForge 训练代码。详细的集成步骤、模型语义对比（动作空间、归一化归属、chunk 长度等）和最小检查清单，请参见：

- [model_integration_guide.md](https://github.com/baidu-baige/LoongForge/blob/master/loongforge/embodied/eval/model_integration_guide.md) — 新模型语义检查清单，含 pi0.5 与 X-VLA 的并排对比
- [predict_action_interface.md](https://github.com/baidu-baige/LoongForge/blob/master/loongforge/embodied/eval/predict_action_interface.md) — `predict_action` 接口契约（签名、shape、反归一化归属、后处理 vs 模型）

---

## 8. 并行评测（多卡编排）

在配置中加入 `servers` 和 `pools`，评测会切换为「每张卡一个推理服务 + 每卡 N 个仿真 worker + 全局队列」：

```yaml
servers:
  - gpus: [0, 1, 2, 3, 4, 5, 6, 7]
pools:
  # GPU-sim 家族（RoboTwin / SimplerEnv / ManiSkill）用 gpu_sim；
  # CPU-sim 家族（LIBERO / CALVIN）改成 cpu_sim，且不支持 placement。
  gpu_sim:
    n_sims_per_server: 8
    placement: co_locate
```

现有配置若不含 `servers` / `pools` / `coordinator`，仍走原来的单 run 路径，行为不变。可直接复制 `configs/libero/libero_pi05_parallel.yaml`、`configs/robotwin/robotwin_pi05_parallel.yaml`，用 `run_*_eval_parallel.sh` 启动。

### 8.1 关键配置

| 配置 | 作用 |
|---|---|
| `pools.gpu_sim.n_sims_per_server` | **GPU-sim 家族**（RoboTwin / SimplerEnv / ManiSkill）每卡仿真数（N） |
| `pools.cpu_sim.n_sims_per_server` | **CPU-sim 家族**（LIBERO / CALVIN）每卡仿真数（N） |
| `pools.gpu_sim.placement` | 仅 GPU-sim 家族：`co_locate`（仿真与推理服务同卡，推荐）或 `separate` |
| `coordinator.n_workers` | 未配置对应池时 N 的回退值（默认 1） |
| `benchmark.episodes_per_chunk` | 一个评测单元连续跑多少集 |
| `benchmark.plan_order` | `lpt_long`：长任务的**所有** chunk 先派发，其余再轮转（RoboTwin 推荐） |
| `benchmark.long_task_steps` | 长任务判定阈值（官方 step_limit，如 `1200`） |
| `benchmark.long_task_episodes_per_chunk` | 只对长任务使用的更细分片（如 `10`），把最后一个 unit 的尾段从 ~83 min 压到 ~33 min |
| `benchmark.sim_cpu_threads` | 仅 RoboTwin；多仿真共卡时设为 1，避免线程超订 |

**池的 key 必须和 benchmark 家族匹配**：LIBERO / CALVIN 读 `pools.cpu_sim`，GPU-sim 家族读 `pools.gpu_sim`。写错 key 会被忽略（coordinator 会打印一行 warning），N 回退到 `coordinator.n_workers or 1`，表现只是「比校准表慢很多」。另外 **N 是每个 service（每张卡）的仿真数**，总并发 = 卡数 × N（如 8 卡 N=48 → 384 个仿真）。

单元数应接近 slot 数：`episodes_per_chunk ≈ max(1, floor(总集数 / (卡数 × N)))`，这样一波跑完、每单元只付一次进程启动。未显式设置时，planner 的默认值是 `ceil(E / (4 × slot 数))`（约每 slot 4 个单元），让后续波次可以回填空出的 slot；已知一波能跑完时请显式设为上式。单任务且总集数 ≤ slot 数时该值取 1，此时继续增大 N 不会提高并发。

任务长度差异大时（RoboTwin 的 step_limit 从 400 到 1700），makespan 由**最长任务的最后一个 shard** 决定，而不是平均值：一个 1,500 步任务的 25 集 unit 要跑 ~83 min。`lpt` 只排任务顺序、之后仍会打散，长任务的后几个 shard 会被排到队列深处。用 `lpt_long` 把长任务的全部 shard 前置，并只给它们更细的 chunk：

```yaml
benchmark:
  plan_order: lpt_long
  long_task_steps: 1200
  long_task_episodes_per_chunk: 10
```

50 任务 × 100 集实测：`lpt` + chunk 25 → **1.88 h（21.8×）**；`lpt_long`(1200) + 长任务 chunk 10 → **1.77 h（23.2×）**，成功率不变（85.9% vs 86.0%，同种子）。

### 8.2 加速开关

| 开关 | 默认 | 说明 |
|---|---|---|
| `benchmark.return_action_chunk` | `false`（LIBERO 为 `true`） | 客户端持有动作块，请求数从每步一次降到每块一次 |
| `benchmark.chunk_execute_steps` | 0（全部） | 一块执行多少步后重新规划 |
| `benchmark.skip_intra_chunk_obs` | `false` | 仅 RoboTwin；块内不渲染相机，是长任务加速的主要来源 |

`return_action_chunk` + `skip_intra_chunk_obs` 会让策略在块内开环（每 `chunk_execute_steps` 步才重新感知），因此逐集结果不会与逐步闭环完全一致，详见 8.5。

### 8.3 标定（试跑）

```bash
python -m loongforge.embodied.eval.orchestrator.calibration.calibrate \
    --config run.yaml --n 6,8,10,12 --gpus 0,1,2,3,4,5,6,7 --out /tmp/calib
```

两个口径：**墙钟吞吐**（不含模型加载、含每单元进程启动，适合一波跑完的小规模）和**稳态吞吐**（只算 rollout，适合每个 slot 连续跑很多集的全量评测）。RoboTwin 上每单元进程启动约 150–340 s，而单集 rollout 只有约 20 s，两个口径可能相差数倍，按实际场景选择。

参考实测（8×A800）：

- **LIBERO + FastWAM**（每卡一个推理服务，`chunk_execute_steps=10`）：N=16/24/48/80 时稳态容量 70.7 / 77.8 / 81.8 / 81.9 eps/min，本机在 N≈48 附近出平台。
- **RoboTwin + pi05**（`adjust_bottle`，官方 400 步，`E=16N`，每单元 2 集）：

| N | 单元墙钟 | 启动 | rollout/集 | 墙钟吞吐（每卡） | 稳态吞吐（每卡） | 峰值显存 |
|---|---|---|---|---|---|---|
| 6 | 204.5 s | 167 s | 17.0 s | 3.52 ep/min | 21.1 ep/min | — |
| 8 | 266.6 s | 221 s | 20.8 s | 3.60 ep/min | 23.1 ep/min | — |
| 10 | 323.8 s | 275 s | 22.0 s | 3.71 ep/min | 27.3 ep/min | ~58–60 GB |
| 12 | 389.0 s | 341 s | 21.9 s | 3.70 ep/min | 32.9 ep/min | ~75 GB |

稳态吞吐到 N=12 仍在上升；峰值显存随 N 近似线性增长，在本机 80 GB 卡上 N=12 已接近上限。显存上限取决于场景、模型与渲染设置，请在自己的环境实测并留余量。

### 8.4 加速实测

| 场景 | 基线 | 并行配置 | 加速 |
|---|---|---|---|
| LIBERO + FastWAM，2000 集 | 官方脚本 173.7 min（8 卡折合） | 35.5 min（worker 阶段）/ ~41 min（含加载） | **4.9× / ~4.3×** |
| RoboTwin + pi05，`adjust_bottle` 50 集 | 1040 s（评测阶段 699 s） | 513 s（199 s） | 2.0× / 3.5× |
| RoboTwin + pi05，`open_microwave` 50 集 | 4984 s（4701 s） | 756 s（453 s） | 6.6× / 10.4× |

RoboTwin 的第二个数去掉了模型加载与服务启动（8 个服务约 5 min），仿真 worker 的进程启动仍计入。

> **可复现性说明**：该链路上不是逐位可复现的——同一配置连续跑两次（固定 `model.noise_seed`、`max_batch_size: 1`），同一 seed 的步数仍会差 1–3 步。因此逐集回归不能作为等价性判据，只能比较聚合成功率（n≥50 时二项噪声约 ±7pt @1σ、±14pt @95%）。这也是 RoboTwin 的动作块开关默认关闭的原因：其逐集影响无法与噪声底分开。

### 8.5 结果对齐

开启加速后，评测集或执行路径会变化，逐集结果不会与 vanilla 逐位一致：

- RoboTwin 会跳过不稳定场景并顺延后续 seed；LIBERO 的 `per_task` 场景由「seed + 进程内第几次 reset」决定，分片且未启用场景快照时场景会变；
- 动作块执行让策略每 `chunk_execute_steps` 步才重新感知；
- batching 改变批内归约顺序，差异在噪声级。

评测集足够大时（每任务/套件 ≥50–100 集），这些差异表现为抽样噪声（n=50 约 ±7pt @1σ，n=100 约 ±5pt @1σ），可视为随机性的一部分；**不要**用单集或少集结果做逐位回归对比。需要严格对齐时关闭 `return_action_chunk`、`skip_intra_chunk_obs`，使用 `env_seed_mode: per_episode`（或 `per_task` + 场景快照），并关闭 batching。

### 8.6 RoboTwin 离线专家检查

RoboTwin 官方流程会先跑专家 rollout 过滤不可解场景，可缓存复用：

1. **扫描**：`benchmark.offline_expert_check_scan: <dir>` 只跑专家分支（不启动推理服务），`offline_expert_check_scan_shards: k` 按残差类分片，跑完自动 merge；
2. **消费**：`benchmark.offline_expert_check: <dir>` 跳过专家分支、按索引取 seed；`start_seed` 仍是 seed 值，不用于索引；
3. **失效**：缓存只在**生成它的环境**有效。「环境」指完整计算栈：GPU 型号、驱动与 CUDA 版本、Python 与依赖版本（torch、SAPIEN/svulkan、cuRobo）、benchmark 资产以及任务/场景定义。GPU 或驱动变化可能改变 seed 的稳定性，异地生成的缓存可能放入一个本地不稳定的 seed。当前实现**不记录也不校验环境摘要**，上述任一项变化都必须重新扫描，缓存应视为某一机器配置的产物；消费时若缓存 seed 不稳定，单元以 `CACHE STALE` 失败并重试，应重新扫描而非静默跳过。
