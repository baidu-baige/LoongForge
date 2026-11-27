# ego2robot：第一视角人手视频转 Panda + 灵巧手数据

[English](README.md)

ego2robot 把第一视角双手操作视频转换成机器人训练数据：先从视频中去掉人手，
再让 Panda 双臂加灵巧手在 MuJoCo 里复现同样的动作，把机器人合成到干净背景上，
最后写成 LeRobot v3.0 数据集。

支持的手：`allegro`、`leap`、`wuji`、`inspire`、`shadow`、`svh`、`ability`、
`dexhand021`、`sharpa`。旧版的 Panda 平行夹爪和其他机械臂构型仍可通过
`--robot_type` 使用（默认 `--end_effector parallel_gripper`）。

## 流程

```
load      读取并重采样 zarr episode
align     构建 18 维状态和平滑动作
mask      SAM3 时序分割手、前臂和身体
inpaint   ProPainter 去掉人手，得到干净背景
[depth]   Depth Anything 3 场景深度（可选，--with_depth）
retarget  Panda 基座放置、手臂 IK、灵巧手重定向、渲染、验收
lerobot   LeRobot v3.0 数据集（H.264）
validate  数据集检查
demo      对比预览视频
```

灵巧手模式下，`retarget` 根据 episode 的 21 点手部轨迹同时求解双臂和双手。
每个结果都要通过独立验收（`independent_validation.json`）。验收失败时运行报错
停止，该 episode 不会写入数据集。

## 安装

参考环境是 NVIDIA PyTorch 25.06 容器（Ubuntu 24.04、Python 3.12、CUDA 12.9）：

```bash
docker build -t ego2robot -f docker/Dockerfile .
docker run --gpus all -it -v $PWD:/workspace/ego2robot ego2robot
```

不用容器时，先按 CUDA 版本装好 torch 和 torchvision，再执行：

```bash
pip install -r requirements.txt
pip install --no-deps lerobot==0.5.0   # lerobot 声明 numpy>=2，固定版本栈用的是 1.26.4
```

### 机器人模型

仓库里不带机器人模型。用脚本按固定 commit 下载并生成规范化的 MuJoCo 模型
（需要 `git` 和网络，约 1 GB）：

```bash
python tools/fetch_hand_assets.py          # 写入 hand_assets/
```

脚本提供 Panda 手臂（mujoco_menagerie）和 8 只手。生成的文件含绝对路径，
移动目录后需要重新运行。夹爪机器人（jaco、sawyer、SO-ARM101）复用
`tools/ego2robot/models` 里的模型。

**Sharpa** 不由脚本下载。请自行获取官方 Do-As-I-Do 发布
（<https://github.com/malik-group/do-as-i-do>，commit `824591b8`），并设置：

```bash
export EGO2ROBOT_SHARPA_UPSTREAM=/path/to/do-as-i-do
export PYTHONPATH=$EGO2ROBOT_SHARPA_UPSTREAM/retargeting:$PYTHONPATH
```

### 感知模型（仅完整流程需要）

自带示例从这些步骤之后开始，不需要它们。

| 步骤 | 项目（测试过的 commit） | 设置 |
|---|---|---|
| mask | facebookresearch/sam3（`f1647e8d`）+ 权重 | `pip install -e sam3`；`--sam3_checkpoint` 或 `EGO2ROBOT_SAM3_CKPT` |
| inpaint | sczhou/ProPainter（`32678252`）+ 权重 | `EGO2ROBOT_PROPAINTER_DIR`（默认 `third_party/ProPainter`） |
| depth | ByteDance-Seed/depth-anything-3（`3d835ec1`）+ DA3-BASE 权重 | `pip install -e depth-anything-3`；`--model_id` 或 `EGO2ROBOT_DA3_MODEL_DIR` |

## 示例：case_01

`examples/case_01/` 是一条 183 帧的抓放 episode（640x480），包括 `input/` 里的
原始 zarr、load、align、mask、inpaint 各步的输出和一份手臂种子。重定向到某只手：

```bash
bash tools/ego2dex/examples/run_case_01.sh allegro   # 输出：tools/ego2dex/examples/output/case_01_allegro/
```

结果目录里的主要文件：

| 文件 | 内容 |
|---|---|
| `robot_on_bg.mp4` | 合成到去手背景上的机器人视频 |
| `contact_sheet.jpg`、`review_*.jpg` | 用于检查的静帧 |
| `independent_validation.json` | 验收结果和误差指标 |
| `trajectory.npz`、`scene.xml` | 求解出的关节轨迹和 MuJoCo 场景 |
| `motor_actions.npz` | 手的逐帧电机指令 |

示例设置了 `EGO2ROBOT_ARM_INITIALIZER=examples/case_01/arm_seed`，这是这条
episode 一次已通过验收的 Sharpa 运行的手臂轨迹，发布结果就是从它求解的。有了它，
9 只手的结果与参考结果逐位一致。把该变量设为空字符串，手臂就从初始姿态开始求解。

## 新素材

输入是 EgoVerse 格式的 zarr v3 episode 目录（图像、21 点手部轨迹、头部位姿、
相机内参），`examples/case_01/input` 就是一个例子。先跑 Sharpa，再把它的手臂
轨迹作为其他手的种子。不同手的数据集状态维度不同，每只手用单独的 `--output_dir`：

```bash
python cli.py run-all --robot_type panda --end_effector sharpa \
  --input_dir examples/case_01/input --output_dir out_sharpa \
  --sam3_checkpoint <sam3.pt> --device cuda:0

export EGO2ROBOT_ARM_INITIALIZER=out_sharpa/06_retarget_sharpa/_render
python cli.py run-all --robot_type panda --end_effector allegro \
  --input_dir examples/case_01/input --output_dir out_allegro \
  --sam3_checkpoint <sam3.pt> --device cuda:0
```

种子目录可以是单个结果，也可以每个 episode 一个子目录。通过验收的 Sharpa 结果
里有 `arm_seed.npz`（静态基座位姿和手臂关节），复用时不需要 Sharpa 网格。只用
已通过验收的结果；手以及所有任务、接触、视角检查都会重新求解。
不设种子时，手臂从初始姿态开始求解。

结果写到 `<output_dir>/06_retarget_<hand>/_render/<ep>/`，数据集写到
`<output_dir>/07_lerobot/`。用 `--episodes <ep>` 可以只跑指定的 episode。数据集的帧率
跟随 `--target_fps`，写完后 `run-all` 会自动做 `validate` 检查。episodes 表里的
`acceptance/*` 列记录旋转策略和两个验收结果（如 `acceptance/original_passed`、
`acceptance/stable_passed`、`acceptance/repaired_frames`），可以看出 episode 是按原始旋转通过，
还是修复之后才通过。

灵巧手常用的 `run-all` 选项：`--acceptance_profile`（默认 `mild`，可选
`strict`、`loose_b`）、`--hand_initializer`（默认 `v28-v29`，可选 `v28`、`off`）、
`--hand_rotation_policy`（默认 `stabilize`，可选 `original`）、
`--hand_depth_mode`（默认 `alpha`，可选 `depth-aware`）。单步运行见
`python cli.py <step> --help`（`load`、`align`、`mask`、`inpaint`、`depth`、
`retarget`、`lerobot`、`validate`、`demo`、`export`）。

### 没有手部轨迹的视频（Path B）

`python cli.py path-b` 用 WiLoR 手部重建（可选 Dyn-HaMR 优化）把普通视频转成
EgoVerse 格式的 episode。需要 WiLoR 源码（`--wilor_repo`）、它的权重和检测器，
以及 MANO 右手模型（`--mano_dir` 或 `EGO2ROBOT_MANO_DIR`）。详见
`python cli.py path-b --help`。

## 环境变量

| 变量 | 作用 |
|---|---|
| `EGO2ROBOT_HAND_ASSETS` | 规范化手模型目录（默认 `hand_assets/normalized`） |
| `EGO2ROBOT_MENAGERIE_DIR` | mujoco_menagerie 目录（默认 `hand_assets/vendor/mujoco_menagerie`） |
| `EGO2ROBOT_SHARPA_UPSTREAM` | Sharpa 用的官方 Do-As-I-Do 目录 |
| `EGO2ROBOT_ARM_INITIALIZER` | 手臂种子目录（见上文） |
| `EGO2ROBOT_ELASTIC_WORKERS` | 注册手求解的并行进程数，不含 Sharpa（默认 1） |
| `EGO2ROBOT_ACCEPTANCE_PROFILE`、`EGO2ROBOT_HAND_INITIALIZER` | 同名选项的默认值 |
| `EGO2ROBOT_SAM3_CKPT`、`EGO2ROBOT_PROPAINTER_DIR`、`EGO2ROBOT_PROPAINTER_STAGE_ROOT`、`EGO2ROBOT_DA3_MODEL_DIR` | 感知模型 |
| `MUJOCO_GL` | `osmesa`（容器默认）或 `egl` |

发布结果所用的求解器设置在 `steps/config.py`（`HAND_SOLVER_DEFAULTS`），
每项都可以用对应的环境变量覆盖。

## 说明

- 渲染：结果用 OSMesa 验证过。EGL 更快，但在部分 GPU 上出现过坏帧，切换前
  请先和 OSMesa 对比几帧。
- 求解耗时为每条 episode 每只手几分钟到几小时，取决于 episode 长度和 CPU。
  注册手可设置 `EGO2ROBOT_ELASTIC_WORKERS` 使用更多核。
- 缺少相机内参的 episode 会借用同目录下其他 episode 的内参。
- 如果 mask 步骤在 triton 里报 `UnicodeDecodeError`，是宿主 `ldconfig -p` 输出含
  非 UTF-8 字节。在 `PATH` 前面放一个过滤包装（容器里可挂到 `/sbin/ldconfig`）：
  `/sbin/ldconfig.real "$@" | LC_ALL=C tr -cd '\11\12\15\40-\176'`。
- 手模型来自公开描述文件，没有对照硬件标定检查过。

## 许可证

Apache-2.0，见 `LICENSE`。机器人模型、感知模型和 Python 包保留各自的许可证，
见 `THIRD_PARTY_NOTICES.md`。ProPainter 仅限非商业使用。
