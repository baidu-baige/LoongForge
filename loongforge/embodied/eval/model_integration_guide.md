# 新模型接入评测系统要点说明（以 pi05 与 xvla 为例）

本文档总结将新 VLA 模型接入 `loongforge/embodied/eval` 评测系统的完整流程与关键配置项。
除实现 factory 与编写 YAML 配置外，接入过程中还存在一系列模型语义层面的配置点需要逐项确认。
pi05 与 xvla 在这些配置点上的取值几乎完全不同，本文以二者为例逐项说明，并给出配置对照表。

**范围说明：** 下文语义清单覆盖当前已接入的全部 benchmark（LIBERO / RoboTwin / SimplerEnv / CALVIN / ManiSkill）。
早期接入以 LIBERO 为验收主路径，故部分条目以 LIBERO 为例展开；跨 benchmark 的协议差异（如 RoboTwin `action_bridge`）单独标出。
配置布局约定见 §0.1；任务成功状态见 `README.md` / `user_guide_en.md`。

## 0. 接入流程概览

1. 在 `factories/` 目录下新增 `<model>_factory.py`，使用 `@register_factory("<model_type>")`
   注册，`build()` 方法返回 `PredictActionModelSpec(model, metadata)`。模型需实现统一的
   `predict_action(images, instructions, state, dataset_stats, ...)` 接口
   （接口规范见 `predict_action_interface.md`）。
2. 编写评测配置与脚本（`examples/embodied/<model>/eval/`），并逐项核对本文
   第 1 至第 10 节所列配置点。
3. 首先执行 smoke（1 task × 1 episode 或链路 smoke），验证 RPC/动作语义，
   再在有域权重时做 task-success 与全量评测。
4. 修改 runner / adapter / bridge / generic policy 等**公共**代码后，须对已成功组合
   （至少 pi05×LIBERO）做回归，避免新协议破坏既有成绩。

### 0.1 配置与脚本布局（pi05 / xvla 约定）

- 每个 benchmark **只保留一对** YAML：公开模板 `*.yaml`（`/path/to/...` 占位）+
  内部一键 `*_internal.yaml`（本机绝对路径）。可选 knobs 写在文件头注释，
  **不要**额外再放 full-suite / 备用 smoke 文件（用户明确要求时除外）。
- 脚本成对：`run_<benchmark>_eval.sh` ↔ 公开 YAML；
  `run_<benchmark>_eval_internal.sh` ↔ `_internal` YAML。
- **task-success vs 链路 smoke：** 仅在有匹配域权重且已验证时写 task-success；
  无域权重时用 `server.random_init: true`、空 `ckpt_path`、短 `max_steps`，
  并在注释中标明 **not task-success**（见 pi05/xvla 的 CALVIN、ManiSkill；
  以及 pi05 的 SimplerEnv）。

参考目录：

- `examples/embodied/pi05/eval/configs/<benchmark>/`
- `examples/embodied/xvla/eval/configs/<benchmark>/`

## 1. 动作空间与维度（action_dim / action_mode）

动作维度**随 benchmark 协议变化**，不能只按 LIBERO 写死。

- **pi05 + LIBERO / 单臂 EE 类：** 输出 7 维（位置增量 3、axis-angle 增量 3、夹爪 1），
  与 LIBERO 环境动作空间一致，可直接下发。
- **pi05 + RoboTwin：** 关节协议 14 维，`model.action_dim: 14`，`action_horizon: 32`，
  配合 `benchmark.action_bridge: pi05_aloha_14d`（见 §3.1），**不是** 7 维。
- **xvla：** 模型侧 EE6D 20 维（`model.action_mode: ee6d`、`real_action_dim: 20`）。
  - 单臂（LIBERO / SimplerEnv / CALVIN）：runner 侧 `action_postprocess` 转成环境 7 维。
  - 双臂 RoboTwin：`action_bridge: ee6d_dual`，以 EE 位姿经 `take_action(..., action_type='ee')` 执行。

确认要点：模型输出维度总数、各维语义排布（位置 / 旋转 / 夹爪）、旋转表示
（axis-angle、6D rotation、四元数）、以及目标环境控制接口（关节 vs EE）。

## 2. 控制模式（绝对位姿控制与增量控制）

- **pi05 + LIBERO：** 增量（delta），robosuite OSC 默认 `use_delta=True`，无需额外设置。
- **pi05 + RoboTwin（`pi05_aloha_14d`）：** 模型侧 delta 关节 → bridge 内转为绝对关节后再下发
  （openpi Aloha：`adapt_to_pi` 解码 state → 推理 → delta→abs → `adapt_to_pi` 编码）。
- **xvla + LIBERO：** 绝对 EEF 位姿。配置 `benchmark.control_mode: absolute`
  （或默认 `auto` 且存在 `action_postprocess`）时 runner 将 OSC 设为 `use_delta=False`。
- **xvla + SimplerEnv WidowX：** 绝对 EE，依赖 SimplerEnv 注册
  `arm_pd_ee_target_base_pose_*` 控制器（见 `examples/embodied/xvla/eval/SIMPLERENV_PATCH_en.md`）。
- **xvla + RoboTwin：** EE 绝对位姿路径（`ee6d_dual`），不走关节 delta。

需要特别说明的是，绝对位姿不可通过与当前位姿「线性相减」草率变成增量：
axis-angle 旋转不满足线性相减，且增量模式常有动作缩放。控制模式配错是
xvla 在 LIBERO 上初始成功率为 0 的首要原因。

## 3. 动作后处理（action_postprocess）与 RoboTwin bridge

### 3.1 `benchmark.action_postprocess`（runner 侧，注册表）

注册表：`servers/predict_action_interface.py` → `ACTION_POSTPROCESS_REGISTRY`。

| key | 用途 |
|---|---|
| `ee6d_to_axis_angle` | xvla × LIBERO：20D EE6D → 7D（pos + axis-angle + grip） |
| `ee6d_to_simpler_abs_euler` | xvla × SimplerEnv WidowX：rot6d→euler + 偏置 + grip 映射 |
| `ee6d_to_calvin_abs` | xvla × CALVIN 官方绝对位姿协议 |
| `ee6d_to_euler` / `ee6d_to_quat` | 其他 EE 变体 |

- pi05：一般不配（LIBERO 直通；RoboTwin 走 bridge）。
- xvla：按 benchmark 选择上表 key；未配置时为恒等直通。

若新模型输出与环境原生动作空间不一致，须在注册表中登记转换函数，
**不要**在 training-side `predict_action` 里写环境特判。

### 3.2 `benchmark.action_bridge`（仅 RoboTwin）

实现：`bridges/robotwin_policy.py`。

| bridge | 用途 |
|---|---|
| `pi05_aloha_14d` | **pi05 RoboTwin 正式协议**（adapt_to_pi + delta→abs） |
| `ee6d_dual` | **xvla RoboTwin 正式协议**（20D EE、三视角、`action_type='ee'`） |
| `strict_14d` | 原始 14D 关节、无 adapt_to_pi（非 openpi 正式协议） |
| `duplicate_7d` | 7D→14D **仅链路 smoke**，不可作成绩 |

协议逻辑放在命名 bridge mode 中，避免改模型默认行为影响其他 benchmark。

## 4. 本体感知输入（state_format 与数据排布）

- **pi05 + LIBERO：** 常用 8 维标准 LIBERO 本体感知（或模型约定布局），
  一般不强制 `state_format`。
- **pi05 + RoboTwin：** adapter / bridge 使用 14D 关节 `model_state`；
  `pi05_aloha_14d` 在 bridge 内再做 adapt_to_pi 解码。
- **xvla：** 配置 `server.state_format: ee6d`，输入 20 维，与动作空间同构。
  旋转 6D 必须列主序 `[R00,R10,R20, R01,R11,R21]`
  （旋转矩阵前两列拼接，对齐 X-VLA `Mat_to_Rotate6D`）。
  行主序 `mat[:, :2].flatten()` 会导致输入分布错误——这是 xvla LIBERO
  初始成功率为 0 的第二项原因。

此外需确认 proprio 语义来源：原版 X-VLA client 将上一步预测动作回填；
当前实现多用环境实测状态，LIBERO 上已验证可行。若新模型敏感，需消融。

边界：`canonical_obs["state"]` 供 trace/debug；仅 `canonical_obs["model_state"]`
  经 RPC 进入 `predict_action(state=...)`。

## 5. 归一化方式（外部统计文件与模型内部归一化）

- **pi05：** 依赖外部 `dataset_statistics.json`（q01/q99 反归一化），
  必须配置 `server.dataset_statistics_path`（RoboTwin 可用
  `examples/embodied/pi05/eval/assets/pi05_robotwin2_dataset_stats.json`）。
- **xvla：** 归一化在模型内部 action space（`action_hub.py` 的 EE6DActionSpace），
  配置 `dataset_statistics_path: ""`。

确认要点：反归一化只在一处执行（模型内部 `predict_action`），
generic policy **不做** unnorm。

## 6. 模型专属请求字段（domain_id 等）

- **pi05：** 无 `domain_id`。原 `unnorm_key` 已废弃，统一走 dataset_statistics。
- **xvla：** 多域模型，必须配置 `benchmark.domain_id`。runner 写入 RPC payload，
  factory 包装函数将 int 转为 `LongTensor`。配错时通常**不报错**，但动作分布错误。

仓库内已用的 xvla domain_id（以官方 eval / 现网 YAML 为准，勿臆造）：

| Benchmark | domain_id |
|---|---|
| SimplerEnv WidowX / Bridge | **0** |
| CALVIN | 2 |
| LIBERO | 3 |
| RoboTwin2 | 6 |
| VLABench（若接入） | 8 |

新模型的 task embedding / domain embedding / 特殊 prompt 等同理：
YAML → runner payload → factory → 模型 四层贯通。

## 7. 动作 chunk 长度与执行策略

- **pi05 + LIBERO：** `action_horizon` 50，runner 按配置消费。
- **pi05 + RoboTwin：** `action_horizon: 32`（与 openpi RoboTwin 训练一致）。
- **xvla：** 模型单次输出 30 步，原版 client 只执行前 10 步后 replan；
  由 `server.chunk_execute_steps` 控制（默认 0→截断为 10；可改 N 或 -1 关闭截断）。

确认要点：训练时的开环执行长度；chunk 消费过长/过短都会显著影响闭环。

## 8. 评测参数（max_steps / num_steps_wait 等）

须按**原始评测协议 + 当前 smoke 意图**配置，不要混用：

- **pi05 × LIBERO：** smoke 常用 `max_steps: 300`；全量 libero_10 等长时序套件
  建议更高（如 520）。`max_steps` 过小会把进行中的 episode 判负。
- **xvla × LIBERO：** 原版 horizon 800、`num_steps_wait: 10`；smoke 可 1 task × 1 ep，
  仍建议保留 800 以免长任务被截断。
- **xvla × SimplerEnv WidowX：** 官方 `max_steps: 1200`（task-success 配置）。
- **链路 smoke（random_init）：** 刻意短步数（如 20–30），只证 RPC，不作成绩。

runner 语义：`benchmark.max_steps > 0` 时优先于套件默认值（不受其上限约束）。

## 9. 图像视角数量与顺序

`GenericPredictActionPolicy._build_image_input` 规则（**非**固定 2 视角）：

1. 必须有 `primary` 或 `head`；
2. 若同时存在 `left` 与 `right` → **3 视角** `[head, left, right]`（RoboTwin / X-VLA 官方）；
3. 否则最多再附加一个 `wrist`（或单独的 left/right）作为第 2 视角。

- **pi05 × LIBERO：** 通常 2 视角（agentview + wrist）。
- **pi05 / xvla × RoboTwin：** 3 视角；模型侧须支持**动态** `num_images = len(images[0])`，
  **禁止** hardcode `num_images=2` 或 `3`（否则破坏其他 benchmark）。
- **xvla × SimplerEnv：** 官方单视角（第 3 人称）。

确认要点：训练时的相机数量、顺序、分辨率、是否翻转
（LIBERO agentview 上下翻转由 adapter 统一处理）。

YAML 的 `model.num_image_views` **不是**通用 ModelConfig 字段；视角由 policy 按
obs 动态打包，不要靠 YAML 虚构字段控制视角数。

## 10. 工程参数

- **加载超时（`start_timeout_sec`）：** xvla 冷启动可 >900s，task-success 配置常用 2400；
  pi05 900 通常足够；`random_init` 链路 smoke 可更短。
- **processor / tokenizer：** pi05 需外部 paligemma（`tokenizer_path`）；
  xvla 的 processor 在 checkpoint 目录（`processor_path` / `tokenizer_path` 常同目录）。
- **`server.random_init`：** 无权重时的链路 smoke；与空 `ckpt_path` 配对。
- **端口：** 每次运行独立 `port` / `health_port`，避免 chunk 缓存交叉污染。
- **GPU：** 通常一卡一 policy server；评测任务串行更稳妥。
- **环境分离：** orchestrator 用 **benchmark conda**；`server.python` 用 **模型 server** 环境。
- **SAPIEN（SimplerEnv / RoboTwin / ManiSkill）：** 除 `nvidia-smi` 外用 `vulkaninfo`
  确认 `deviceName=NVIDIA`（非 llvmpipe）；设置 `LD_LIBRARY_PATH`、`VK_ICD_FILENAMES`。

## 11. pi05 与 xvla 配置对照表

### 11.1 LIBERO（语义验收主路径）

| 配置项 | pi05 | xvla |
|---|---|---|
| model_type | pi05 | xvla |
| action_dim（环境侧） | 7 | 7（后处理产出） |
| real_action_dim（模型侧） | 7（pad 至 32） | 20（EE6D） |
| action_mode | — | ee6d |
| 控制模式 | 增量（`control_mode: auto`→delta） | 绝对（`control_mode: absolute` 或 auto+postprocess） |
| action_postprocess | — | ee6d_to_axis_angle |
| chunk_execute_steps | — | 10（`server.chunk_execute_steps`） |
| state_format | 默认 / 不强制 | ee6d（20 维，rot6d 列主序） |
| dataset_statistics_path | 必须 | "" |
| domain_id | — | 3 |
| chunk 执行长度 | 50 | 10（factory 截断） |
| max_steps（协议参考） | smoke 300；长套件更高 | 800 |
| num_steps_wait | 10 | 10 |
| 图像视角 | 2（agentview+wrist） | 2（同 LIBERO 相机） |
| tokenizer/processor | 外部 paligemma | checkpoint 内置 |
| start_timeout_sec | 900 | 2400 |

### 11.2 其他已验证 / 链路组合（摘要）

| 组合 | 关键配置 | 状态 |
|---|---|---|
| pi05 × RoboTwin | `action_dim: 14`, `action_horizon: 32`, `action_bridge: pi05_aloha_14d`, stats JSON | task-success |
| xvla × RoboTwin | `domain_id: 6`, `action_bridge: ee6d_dual`, 3 视角 | task-success |
| xvla × SimplerEnv WidowX | `domain_id: 0`, `ee6d_to_simpler_abs_euler`, abs EE + patch, `max_steps: 1200` | task-success |
| pi05 × SimplerEnv / CALVIN / ManiSkill | `random_init: true`（无域权重） | 仅链路 smoke |
| xvla × CALVIN / ManiSkill | `random_init: true`（官方 Calvin 权重未接线时） | 仅链路 smoke |

## 12. 接入验收流程

1. **动作数值检查：** 记录初始若干步模型输出与当前位姿，确认量级、坐标系、
   夹爪取值域（±1 或 [0,1]）后再跑长评测。
2. **与原版推理对照：** 官方 eval client（如 X-VLA 的 libero / simpler / robotwin client、
   openpi RoboTwin）为行为基准，核对 proprio、控制器、chunk、视角顺序。
3. **分层扩大规模：** 本地 interface 校验 → mock/RPC → random-init 链路 →
   真实权重短 episode → 可信 task-success / 全量。
4. **公共代码回归：** 改 runner、adapter、bridge、`_build_image_input`、factory 共享逻辑后，
   必须用已成功组合（推荐 pi05×LIBERO 小 smoke）回归。
5. **文档与 YAML 头注释：** 区分 mock / random-init / real checkpoint / task-success；
   写明 open weight URL 与内部路径，勿把链路 smoke 报成成绩。

## 13. 可配置项现状（已实现与待实现）

设计原则：凡属模型训练协议的内容（动作空间、控制模式、本体感知格式、
归一化、chunk 策略、相机视角、域 ID），实现为 YAML 可配置（带默认值）；
凡属通用管线（RPC、chunk 缓存、延迟、结果记录），对模型无感。

### 13.1 已实现的可配置项

以下项已在 pi05 / xvla 多 benchmark 接入中落地（LIBERO xvla object 全量约 94%；
RoboTwin / SimplerEnv 等见 README 状态表）。

| 配置项 | 实现位置 | 功能说明 | 取值示例 |
|---|---|---|---|
| `benchmark.action_postprocess` | runner + `ACTION_POSTPROCESS_REGISTRY` | 动作后处理；未配置恒等 | 空 / `ee6d_to_axis_angle` / `ee6d_to_simpler_abs_euler` / `ee6d_to_calvin_abs` |
| `benchmark.control_mode` | LIBERO runner `_resolve_libero_use_delta` | OSC absolute/delta；**仅 LIBERO**（与 SimplerEnv/ManiSkill 的仿真 `control_mode` 字符串无关） | `auto`（默认）/ `absolute` / `delta` |
| `server.chunk_execute_steps` | xvla factory 包装函数 | 开环执行长度截断 | `0`→默认 10；`N>0`→截到 N；`<0`→不截断 |
| `benchmark.action_bridge` | `bridges/robotwin_policy.py` | RoboTwin 协议 mode | `pi05_aloha_14d` / `ee6d_dual` / `strict_14d` / `duplicate_7d` |
| `server.state_format` | adapter `_build_model_state` | proprio 布局 | 空 / `ee6d` |
| `benchmark.domain_id` | runner payload → factory wrap | 模型专属字段透传 | 空 / `0`/`2`/`3`/`6` |
| `server.dataset_statistics_path` | GenericPredictActionPolicy | 外部 stats vs 模型内归一化 | 路径 / `""` |
| `server.random_init` | factory / server | 无权重链路 smoke | `false` / `true` |
| `benchmark.max_steps` | runner | 显式值优先于套件默认 | 300 / 800 / 1200 / 短 smoke |
| `benchmark.num_steps_wait` | runner | 环境稳定等待 | 10 |
| `benchmark.continuous_gripper` | adapter | 夹爪连续或二值 | true |
| `benchmark.task_ids` | runner `run_batch` | 指定任务子集 | 空 / 列表 |
| `model:` 结构字段 | 各 ModelConfig | 仅写 dataclass 已声明字段（pi05: `action_dim`…；xvla: `action_mode`/`real_action_dim`…） | — |
| `server.start_timeout_sec` | server 管理 | 加载超时 | 900 / 2400 |
| `server.port` / `health_port` | YAML | 独立端口，防缓存污染 | 每 run 独立 |
| 视角打包 | `GenericPredictActionPolicy._build_image_input` | 1–3 视角动态打包 | 见 §9 |

`control_mode`（LIBERO）语义：

- `auto`（默认，兼容旧 YAML）：有 `action_postprocess` → absolute（`use_delta=False`），否则 delta。
- `absolute`：强制绝对 EE（xvla LIBERO 正式配置可显式写出）。
- `delta`：强制增量（即使配置了 postprocess）。

`chunk_execute_steps`：写在 `server:` 段，由 `EvalServerArgs` 传入 factory；未写时 xvla 仍截断为 10。

### 13.2 待实现 / 仍可改进的可配置项

以下仍为固定或半固定实现，新模型命中时再参数化：

1. **本体感知来源**（`measured | predicted`）。
   现状：多用环境实测；原版 X-VLA 用预测回填，应保留消融开关。
2. **夹爪二值化阈值与映射。**
   现状：各 `action_postprocess` key 内已按官方协议写死阈值；新协议优先新增 registry key，而非全局 threshold。
3. **视角策略进一步 YAML 化。**
   现状：3 视角已由 generic policy 在 left+right 同时存在时自动打包；
   若未来有非 head/left/right 命名或 4+ 视角，再扩展配置。

## 14. 与 skill / 其他文档的关系

- 接入操作 skill：`skills/vla-model-eval-adapter/SKILL.md`（流程、官方 config 对齐、
  YAML 布局、eval-only 边界）。
- 接口契约：`predict_action_interface.md`。
- 用户入口与状态表：`README.md`、`user_guide_en.md`、`benchmark_envs.md`。
- 本文聚焦**模型语义 checklist** 与 pi05/xvla 对照，不重复粘贴全量命令行。
