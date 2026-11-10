# ego2robot — EgoVerse → 16 种双臂 morphology LeRobot 数据集管线

把开源 EgoVerse 第一视角（ego）双手操作视频转成 **16 种双臂机器人 morphology 的 LeRobot v3.0 训练数据**。每种 morphology 使用对应的 MuJoCo 模型、关节布局、末端/TCP 标定和夹爪映射，复现 Qwen-RobotManip 的 retargeting 思路(Eq.1/2/3)。

核心链路:`人做什么操作 → 机器人在 MuJoCo 仿真中重演同一动作 → 把机器人渲染合成到"抹除人手后的真实背景" → 打包成 LeRobot 数据集`。所有路径均可由 CLI 参数 / 环境变量覆盖,不依赖服务器硬编码绝对路径。

## Pipeline

```
load (zarr 读入/清洗/重采样)
  -> align (动作构造 state_16 维 + 平滑)
  -> mask (SAM3 可见人体 + 手部/前臂时序分割)
  -> inpaint (ProPainter 抹掉人手, 得到干净背景)
  [-> depth (Depth Anything 3 场景深度)]        # 可选: run-all --with_depth
  [-> gauss-recon (DVLT 米制场景深度)]         # 可选: run-all --with_gauss
  -> retarget (目标 morphology 双基座搜索 + morphology-aware IK + 深度遮挡/alpha 合成)
  -> lerobot (LeRobot v3.0 数据集写出, H.264)
  -> validate (健检全绿)
  -> demo (3x3 分屏演示视频)
```

## 目录结构

```
ego2robot/
├── cli.py                  # 统一入口: python cli.py <子命令>
├── pipeline.py             # run-all 编排(固定子目录 01_load..08_demo)
├── requirements.txt        # 核心依赖
├── steps/
│   ├── config.py           # 硬编码路径集中处; EGO2ROBOT_* 环境变量覆盖
│   ├── loader.py           # Step1 数据加载/清洗
│   ├── path_b.py           # Path B: 纯视频 -> WiLoR/DynHaMR -> EgoVerse-like Zarr
│   ├── action_alignment.py # Step2 动作构造
│   ├── hand_mask.py        # Step3 SAM3 人体/手部/前臂时序分割
│   ├── inpaint.py          # Step4 ProPainter 背景补全
│   ├── depth_estimate.py   # Step5 DA3 深度估计(可选)
│   ├── gauss_recon.py      # 可选 DVLT 米制场景深度
│   ├── gauss/              # DVLT 推理、SLAM 投影与导出
│   ├── robot_registry.py   # 16 种 morphology 注册与 MJCF/夹爪组合
│   ├── robot_retarget.py   # Step6 兼容入口和 CLI
│   ├── retarget/           # Step6 目标、IK、base search、碰撞、渲染模块
│   │   ├── targets.py      # 人手目标构造和 TCP 转换
│   │   ├── ik.py           # Mink/DLS 逆运动学
│   │   ├── base_search.py  # 基座搜索和碰撞筛选
│   │   ├── rendering.py    # MuJoCo 模型、相机和夹爪渲染
│   │   ├── episode.py      # 单 episode 编排和产物写出
│   │   └── cli.py          # retarget CLI 实现
│   ├── quality/            # L1/L2/L3 数据质检（独立步骤）
│   ├── lerobot_writer.py   # Step7 LeRobot v3.0 数据集写出
│   ├── export_hdf5.py      # Step7b 可选 HDF5 格式导出
│   └── ...
└── examples/
    └── tea_demo_3x3.mp4    # 示例输出(3x3 分屏含深度格)
```

### Path B：只有视频、没有手部关键点

论文中的 Path B 先用 WiLoR 逐帧重建手部，再可选用 DynHaMR 做时序优化。
当前提供的 `wilor_final.ckpt` 只有权重，不包含 WiLoR Python 源码，需要另外指定
WiLoR 源码目录：

```bash
git clone --depth 1 https://github.com/rolpotamias/WiLoR.git <WiLoR_dir>
```

下载 WiLoR checkpoint 和 detector 权重：

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
```

将 WiLoR 所需的右手 MANO 模型下载到 `--mano_dir`（或
`EGO2ROBOT_MANO_DIR`）指定的目录：

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/mano_data/mano/MANO_RIGHT.pkl \
  -O <MANO_models_dir>/MANO_RIGHT.pkl
```

```bash
python cli.py path-b \
  --video /path/to/video.mp4 \
  --wilor_repo /path/to/WiLoR \
  --checkpoint <WiLoR_checkpoint> \
  --detector <WiLoR_detector> \
  --mano_dir <MANO_models_dir> \
  --output_dir out_path_b --episode demo_000
```

命令会写出 `out_path_b/demo_000/`，其中包含视频帧、左右手 MANO 21 点、固定相机的
头部位姿和内参，格式与现有 EgoVerse Zarr 兼容。然后直接接现有流水线：

```bash
python cli.py run-all --input_dir out_path_b --output_dir out_demo \
  --robot_type panda --episodes demo_000
```

离线验收可用 `--predictions` 指向 WiLoR/DynHaMR 输出目录。NPZ/JSON 至少要包含
`pred_keypoints_3d`（或 `keypoints_3d`/`joints_3d`），顺序需使用 WiLoR 采用的
OpenPose 兼容 21 点手部顺序（wrist、thumb、index、middle、ring、pinky），也可提供
`frame`、`right`、`score` 字段。DynHaMR 可通过带 `{input}`、`{output}` 占位符的命令接入。

支持只有一只手的视频。缺失侧在 episode 中保持无效，retarget 时跳过该侧的手部目标
IK，不再生成镜像关键点轨迹。

也可以直接接入官方 Dyn-HaMR 仓库。官方入口是
`dyn-hamr/run_opt.py`，不是 `infer.py`：

```bash
git clone --recursive https://github.com/ZhengdiYu/Dyn-HaMR.git <DynHaMR_dir>

python cli.py path-b \
  --video <video.mp4> \
  --wilor_repo <WiLoR_dir> \
  --checkpoint <WiLoR_checkpoint> \
  --detector <WiLoR_detector> \
  --mano_dir <MANO_models_dir> \
  --dynhamr_repo <DynHaMR_dir> \
  --dynhamr_gpu 0 \
  --output_dir path_b_dynhamr --episode source_000
```

bridge 会自动准备 Dyn-HaMR 所需的视频帧、track JSON、静态相机和 shot
文件，调用官方 `dyn-hamr/run_opt.py`，再把优化后的 MANO 参数转换回
`dynhamr_predictions.npz`。Dyn-HaMR 自身仍需要 `_DATA` 模型资源和依赖，
请在其仓库执行 `scripts/prepare.sh` 并按官方 README 准备 HaMeR/BMC/HMP。

官方 WiLoR 运行还需要 `ultralytics`、`smplx`、`pytorch-lightning`、`yacs`、
`scikit-image` 等依赖。将 `--mano_dir` 指向 MANO 项目的 `models` 目录
（或设置 `EGO2ROBOT_MANO_DIR`）；适配器启动时会自动把 `MANO_RIGHT.pkl`
链接到 WiLoR 的 `mano_data/`。请将 `--mano_dir` 指向 MANO 的 `models` 目录，
并使用与 checkpoint 对应版本的 detector。

使用已经准备好的模型资源时，可以直接按下面两步运行：

```bash
cd <ego2robot_dir>

# 1) 视频 -> WiLoR + Dyn-HaMR -> EgoVerse-like Zarr
python cli.py path-b \
  --video <video.mp4> \
  --output_dir <path_b_output> \
  --episode source_000 \
  --wilor_repo <WiLoR_dir> \
  --checkpoint <WiLoR_checkpoint> \
  --detector <WiLoR_detector> \
  --mano_dir <MANO_models_dir> \
  --dynhamr_repo <DynHaMR_dir> \
  --dynhamr_gpu 0

# 2) Zarr -> mask/inpaint -> retarget -> LeRobot -> demo
python cli.py run-all \
  --input_dir <path_b_output> \
  --output_dir <pipeline_output> \
  --episodes source_000 \
  --robot_type panda \
  --sam3_checkpoint <SAM3_checkpoint> \
  --device cuda:0 \
  --skip_depth
```

Path B 输出的是 Zarr v3 episode。`left.obs_keypoints` 和
`right.obs_keypoints` 保存合成相机世界坐标中的 21 点 MANO 手部轨迹，
`images.front_1` 保存原始 JPEG 视频帧。后续 loader 会将其转换为编号步骤目录。
机器人与背景合成视频位于
`06_retarget/<episode>_robot_on_bg.mp4`，最终 3x3 演示视频位于
`08_demo/<episode>_demo.mp4`。

## 环境要求

- DVLT 要求 Python **>=3.11**；现有容器为 Python **3.12.3**、Torch **2.8.0a0+5228986c39.nv25.06**（CUDA 12.9）、NumPy **1.26.4**。这是环境实测信息，不是从零安装验证；推理建议 CUDA GPU。
- 安装依赖和下载模型权重需要访问互联网；无网络环境下需提前离线准备

### 安装

以 [requirements.txt](requirements.txt) 为依赖清单，但其 NumPy 2.4.6 / Torch 2.5.1 等既有锁版本与运行环境不同，不是已验证的联合安装方案。DVLT 元数据也锁了 NumPy 2.4.4 / Torch 2.5.1。新环境须先隔离解析兼容性，已有环境不要盲目升级 NumPy 2；详见 [安装指南](GAUSS_SETUP_ZH.md)。

```bash
# 在已有兼容环境中单独安装 DVLT 源码，避免其锁版本覆盖环境
python -m pip install -e /path/to/dvlt-source --no-deps
python -m pip check
```

Gauss 补充依赖为 `accelerate`、`huggingface-hub`、`hydra-core`、`omegaconf`、`open3d`、`zstandard`、`numcodecs`（已加入 requirements）。SAM3 和可选 DA3 源码包按各自上游说明安装；系统需要支持 libx264 的 `ffmpeg` 以及 MuJoCo 渲染库。ProPainter 子进程不隔离 Python 包依赖。历史环境还在仓库外修复过 ProPainter 预发行 Torch 版本解析、补装 `scikit-image/yapf/future`，以及从 openai-clip 源码分发包补齐 SAM3 的 `assets/bpe_simple_vocab_16e6.txt.gz`；迁移时不能假设这些修复自动存在。

### 模型资产(离线环境按需放置)

| 资产 | 位置(默认) | 说明 |
|---|---|---|
| SAM3 权重 `sam3.pt` | `--sam3_checkpoint` 或 `EGO2ROBOT_SAM3_CKPT` | [SAM3 官方仓库](https://github.com/facebookresearch/sam3) 的 checkpoint 获取说明；另需源码包和 tokenizer 资源 |
| DVLT 源码 + `config.json`、`model.safetensors` | 单步 `--checkpoint <dvlt_ckpt>` / 整链 `--gauss_checkpoint <dvlt_ckpt>` | 本地 README 指向 [项目页](https://research.nvidia.com/labs/dvl/projects/dvlt/) 和 [nvidia/dvlt](https://huggingface.co/nvidia/dvlt)；源码单独获取并用 `--no-deps` 安装 |
| 条件 DINOv2 ViT-B/14 reg4 `.pth` state dict | `--dinov2_weights` / 整链 `--gauss_dinov2_weights` | [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)；仅 checkpoint 缺少 patch embedding 权重时需要 |
| `mujoco_menagerie/*` | `EGO2ROBOT_MENAGERIE_DIR` | 基于 menagerie 的 morphology 所需 XML+mesh |
| `models/jaco`, `models/sawyer_gripper` | 仓库内置 | Jaco 和 Sawyer 的本地夹爪资产 |
| `models/SO-ARM100/Simulation/SO101` | 仓库内置 | 官方 SO-ARM101 MuJoCo 模型和 mesh 资产 |
| ProPainter 源码 + `weights/ProPainter.pth`、`weights/raft-things.pth`、`weights/recurrent_flow_completion.pth` | 放在 `EGO2ROBOT_PROPAINTER_DIR` 下 | 三项均必需，inference 脚本使用 [v0.1.0 release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0) |
| DA3-BASE `config.json` + `model.safetensors` 及 Python 包 | `depth --model_id` 或 `EGO2ROBOT_DA3_MODEL_DIR` | 仅 depth / `--with_depth` 需要；按 [官方模型清单](https://github.com/ByteDance-Seed/depth-anything-3) 获取 |
| 可选 Path B：WiLoR `wilor_final.ckpt`、`detector.pt`、MANO `MANO_RIGHT.pkl` | `path-b --wilor_repo`、`--checkpoint`、`--detector`、`--mano_dir` | 另需 WiLoR 源码/依赖及许可允许的 MANO 资产，见前文 Path B |
| 可选 DynHaMR 源码和 `_DATA` 资源 | `path-b --dynhamr_repo` | 按 Dyn-HaMR README / `scripts/prepare.sh` 准备 HaMeR/BMC/HMP 及独立依赖 |

以上来源依据本地源码/资产说明，本次未联网验证下载可用性，不随仓库分发权重。DVLT 目录必须包含 `config.json` 和 `model.safetensors`；配置中的 `load_patch_embed_weights=false` 不足以证明 tensor 完整。完整清单见 [安装指南](GAUSS_SETUP_ZH.md)。

### 完整链路一键跑

```bash
python cli.py run-all --with_depth --robot_type panda \
  --input_dir <egoverse_zarr_dir> \
  --output_dir <output_root> \
  --sam3_checkpoint /path/sam3.pt \
  --episodes <ep1> <ep2> ...
```

如需减少底座靠近相机造成的视觉穿模，可选用 `--base_pullback`（单位：米）。该参数默认是 `0`，会在 base 搜索后沿头部视线反方向移动两个底座，并用移动后的底座重新执行最终 IK：

```bash
python cli.py run-all --robot_type xarm7 --base_pullback 0.05 \
  --input_dir <egoverse_zarr_dir> --output_dir <output_root> \
  --sam3_checkpoint /path/sam3.pt
```

逐帧 Mink IK 后默认还会执行整段轨迹 refinement。它保留 TCP 跟踪目标，并联合优化
关节速度、加速度、home pose 偏差和关节限位余量；如果平均或 P95 TCP 位置误差明显
变差，会自动保留原始 Mink 轨迹。可用 `--no_trajectory_refine` 关闭，或用
`--trajectory_refine_velocity_weight`、`--trajectory_refine_acceleration_weight`、
`--trajectory_refine_home_weight` 和 `--trajectory_refine_joint_margin_weight` 调整权重。

`--robot_type` 可选：`panda`、`xarm7`、`arx_l5`、`piper`、`yam`、`fr3`、`ur5e`、`ur10e`、`kinova_gen3`、`sawyer`、`iiwa`、`jaco`、`viperx`、`widowx`、`aloha_agilex`、`so_arm101`。SO-ARM101 使用仓库内置的官方 `new_calib` MuJoCo 模型和 5-DOF 位置优先 IK 配置。每种 morphology 建议使用独立的 `--output_dir`，避免混合不同状态维度。输出目录始终保留步骤编号：`01_load/ ... 04_inpaint/ [05_depth/] [05_gauss/] 06_retarget/ 07_lerobot/ 08_demo/`。未启用的深度目录省略，后续目录名保持不变。

在 `depth-aware` 模式下，SO-ARM101 还会从米制场景深度估计静态桌面支撑平面，并约束左右 base 的底面落在该平面上。如果无法获得可靠平面或相机内参，程序会输出回退提示并保留原有的手部锚定高度搜索；其他 morphology 不启用该约束。

### 单步运行

```bash
python cli.py load    --input_dir <zarr> --output_dir out01 --target_fps 30
python cli.py align   --input_dir out01 --output_dir out02
python cli.py mask    --zarr_dir <zarr> --output_dir out03 \
  --sam3_checkpoint /path/sam3.pt --max_frames 120 --device cuda:0   # 调参可加 --max_frames 试跑再全量
python cli.py mask    --zarr_dir <zarr> --output_dir out03 \
  --sam3_checkpoint /path/sam3.pt --device cuda:0
python cli.py inpaint --zarr_dir <zarr> --mask_dir out03 --output_dir out04 \
  --mask_dilation 4 --fp16 --ref_stride 10 --neighbor_length 10 \
  --subvideo_length 80 --raft_iter 20
python cli.py depth   --input_dir out04 --output_dir out05 --model_id /path/da3  --batch 4 --device cuda
python cli.py retarget --robot_type xarm7 --zarr_dir <zarr> --bg_dir out04 --state_dir out02 --output_dir out06 \
  --scene_depth_dir out05 --depth_mode depth-aware --depth_epsilon 0.02 --base_pullback 0.05
# 回退到原有 mask/alpha 合成（不读取/渲染深度）:
# python cli.py retarget --zarr_dir <zarr> --bg_dir out04 --state_dir out02 --output_dir out06 \
#   --depth_mode alpha
# 可选独立质检步骤，完整说明见 steps/quality/README_ZH.md:
python cli.py quality --retarget_dir out06 --state_dir out02 \
  --output_dir out_quality --skip_vlm
python cli.py lerobot --ik_dir out06 --state_dir out02 --bg_video_dir out06 --output_dir out07
python cli.py validate --dataset_dir out07 --skip_loader_check
python cli.py demo --zarr_dir <zarr> --mask_dir out03 --inpaint_dir out04 --ik_dir out06 \
  --depth_dir out05 --output_dir out08 --episodes <ep>
```

## 输出数据格式

### 机器人构型声明(state/action 维度)

`retarget --robot_type` 决定目标 morphology。IK 输出中的 `state`、`state_names` 和 `robot_type` 会被 `lerobot` 自动读取；不同 morphology 的状态维度可以不同，但同一个 LeRobot 数据集内不能混用。

### 输出流

| 字段 | 类型 | 说明 |
|---|---|---|
| `observation.state` / `action` | float32 `[D]` | 关节位置 / 关节增量(delta), 单位 rad |
| `observation.images.ego` | video | 640×360 H.264 30fps, 机器人+背景合成帧 |
| `observation.masks.ego` | video | 同尺寸机器人实例 mask(灰度) |
| `observation.images.wrist_l` / `observation.images.wrist_r` | video | 同尺寸左右腕部相机 RGB 视频 |
| `timestamp/frame_index/episode_index/index/task_index` | 标量 | LeRobot 帧对齐字段 |
| `ik/ok_ratio`, `ik/pos_err_mean_mm`, `ik/rot_err_mean_deg` | 质量列 | meta/episodes parquet 内每集 IK 质量 |

### 格式选择

- `lerobot`(默认): LeRobot v3.0 目录结构
- `hdf5`(可选): `python cli.py export --input <lerobot数据集目录> --output out.hdf5 [--images]`,
  输出 robomimic/act 风格 h5(per-episode `observations/state`, `actions`, 可选 `observations/images`)

## 示例

`examples/tea_demo.mp4`: 窄桌 prepare-tea 单集全链结果(3x3 分屏, 含 DA3 深度格);
对应的全套中间产物可按下面命令用任意 EgoVerse 样例复现:

```bash
python cli.py run-all --with_depth --robot_type panda --input_dir <egoverse> --output_dir ./out --episodes <ep>
```

## Gaussian 场景重建（DVLT，`gauss-recon` / `--with_gauss`）

新增能力包括 DVLT 前馈多视角重建、米制场景深度、可视化与缓存，并接入 retarget 深度遮挡合成。**当前没有 gsplat trainer、Gaussian splatting renderer 或 novel-view synthesis 入口**。缓存用于离线分析和未来扩展。

DVLT 使用原始 Zarr 图像，可能仍含人手；背景视频只用于输出尺寸/裁剪检测。尺度优先由世界系手部关键点拟合，失败或缺失时回退轨迹长度比，再通过 SLAM `obs_head_pose` 反投影及逐帧投影。原始 DVLT 位姿/世界点仍在模型坐标系，相似变换仅作诊断。静态/近静态相机及动态人手可能导致退化、孔洞和错误几何；不保证优于 DA3 或新视角质量。

```bash
python cli.py gauss-recon --zarr_dir <zarr_root> --state_dir <out>/02_align \
  --bg_dir <out>/04_inpaint --output_dir <out>/05_gauss \
  --checkpoint <dvlt_ckpt> --episodes <ep> --sample_fps 1

python cli.py run-all --input_dir <zarr_root> --output_dir <out> --episodes <ep> \
  --sam3_checkpoint <sam3.pt> --robot_type panda \
  --with_gauss --gauss_checkpoint <dvlt_ckpt> --scene_backend gauss
```

除 `--check` 外，单步必须传 `--zarr_dir`、`--state_dir`、`--bg_dir`、`--output_dir`。state/bg 内容缺失时虽会回退 Zarr 帧数/原图尺寸，但参数不可省略，且回退可能影响后续帧数或裁剪对齐。

`--with_depth` 和 `--with_gauss` 可同时生成 `05_depth/`、`05_gauss/`；默认 `--scene_backend auto` 在 Gauss 启用时优先选 Gauss，否则使用已启用的 DA3。显式 `da3` 应搭配 `--with_depth`，显式 `gauss` 应搭配 `--with_gauss`；选择器不启用阶段，当前实现会在 Gauss 未启用而 DA3 启用时回退 DA3。`--skip_depth` 或 `--depth_mode alpha` 禁用两个阶段和深度合成。

| 单步参数 | run-all 对应参数 | 默认值 |
|---|---|---|
| `--checkpoint` | `--gauss_checkpoint` | 单步 `nvidia/dvlt`；整链 `None`，建议显式传本地目录 |
| `--dinov2_weights` | `--gauss_dinov2_weights` | `None`，仅缺 patch embedding 时需要 |
| `--sample_fps` | `--gauss_sample_fps` | `2.0`（上例为 `1`） |
| `--fy` | `--gauss_fy` | `None`，DVLT 焦距按输出高度缩放 |
| `--conf_threshold` | `--gauss_conf_threshold` | `0.2` |
| `--points_per_frame` | `--gauss_points_per_frame` | `8192` |
| `--inference_steps` | `--gauss_inference_steps` | `None`，由模型决定 |
| `--device` | `--device` | `cuda:0` |

单步还支持 `--img_size 504`、`--patch_size 14`、`--hole_fill_iters 6`。整链自动同步焦距到 retarget（`meta.json` 的 `fy_recommended_retarget`，显式 `--gauss_fy` 优先）；分步执行需保持同一焦距。

每集 `<out>/05_gauss/<ep>/` 输出 `scene_depth.npz`（float16 `[N,H,W]`、`unit="meters"`、`backend="gauss"`、`K` 和裁剪元数据）、`<ep>_dvlt.pt`（原始 DVLT/尺度/对齐缓存）、`recon_vis.mp4`（原图与深度对照）、`depth_vis.mp4`、`meta.json`（scale/coverage/fy 等）。

```bash
python cli.py gauss-recon --help
python cli.py run-all --help
python cli.py validate --help
python cli.py gauss-recon --check --checkpoint <dvlt_ckpt>
# 仅当对应数据、深度与 IK 产物存在时运行
python cli.py validate --dataset_dir <out>/07_lerobot \
  --depth_dir <out>/05_gauss --depth_ik_dir <out>/06_retarget
```

`--check` 只检查导入、CUDA 状态和 checkpoint 文件列表，不加载模型或验证推理；缺文件可能只打印警告。`validate` 深度项为抽样诊断，缺文件会跳过且使用固定渲染视场，不能证明 Gauss 几何正确；必要时追加 `--skip_loader_check` 并报告跳过 loader。环境、权重来源、NPZ 字段检查及迁移限制见 [GAUSS_SETUP_ZH.md](GAUSS_SETUP_ZH.md)。本次维护不重建已删除的中间产物，历史整链记录不等于此次 smoke/e2e 重跑。

## 环境变量


| 变量 | 作用 |
|---|---|
| `EGO2ROBOT_PROPAINTER_DIR` | ProPainter 仓库 |
| `EGO2ROBOT_PROPAINTER_STAGE_ROOT` | 暂存 PNG 目录(建议 tmpfs) |
| `EGO2ROBOT_MENAGERIE_DIR` | mujoco_menagerie 缓存 |
| `EGO2ROBOT_SAM3_CKPT` | SAM3 checkpoint |
| `EGO2ROBOT_DA3_MODEL_DIR` | DA3 权重目录 |
| `EGO2ROBOT_VLM_MODEL` | quality L3 使用的 Qwen3.5 模型 ID/路径（或传 `--vlm_model` / `--model_dir`） |
| `MUJOCO_GL` | 渲染后端, 推荐 `osmesa`(避免 EGL 深度漏面) |

## 已知限制(迁移注意)

- 渲染: EGL 在新版 mujoco 有大面积"漏底"(渲染黑点), **请在 GPU 机器用 `MUJOCO_GL=osmesa`**。
- 数据资产管理: intrinsics 缺失的 episode 会从同批次其他 episode 借用(同一相机配置)。
- `--with_depth` 时，DA3 对 inpaint 后的背景估计场景深度，retarget 同时渲染机器人深度，
  仅在机器人比背景更靠近相机的位置覆盖背景。可用 `--skip_depth` 或
  `--depth_mode alpha` 回退到原有 mask/alpha 合成。
- 若下手(Optionally)需要 装 le 视频回显 md5: rundemo 已验证 30fps 10 秒片段。

## 外部参考

- Qwen-RobotManip: <https://github.com/QwenLM/Qwen2-RobotManip>
- EgoVerse: huggingface `EgoVerse`(或采内部镜像)
- LeRobot v3.0: https://github.com/huggingface/lerobot
