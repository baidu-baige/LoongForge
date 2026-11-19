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
  --geocalib_repo <DynHaMR_dir>/third-party/vipe \
  --geocalib_weights <GeoCalib_pinhole_checkpoint> \
  --unidepth_repo <DynHaMR_dir>/third-party/vipe \
  --unidepth_model <UniDepthV2_checkpoint_directory> \
  --output_dir out_path_b --episode demo_000
```

命令会写出 `out_path_b/demo_000/`，其中包含视频帧、左右手 MANO 21 点、估计的移动相机
位姿和视频估计内参，格式与现有 EgoVerse Zarr 兼容。数据输入仅为视频；
不读取测试 Zarr 的内参、手部坐标或头部位姿。默认 GeoCalib 从视频抽取 8 帧，
联合估计针孔相机，再用 WiLoR 相对手形和其预测 2D 求解正深度的刚体位姿。
手部初始化、Dyn-HaMR 和导出使用相同的 hand K。
默认不使用图像宽度焦距先验；旧实验可显式设置
`--camera_focal_prior_max_deviation 0.35`（并配合
`--camera_focal_prior_ratio 0.4`）复现。生产流程默认只使用 GeoCalib 的视频内参。
如需利用视频多视图几何提出共享焦距，可显式启用
`--camera_self_calibration --camera_focal_prior_max_deviation -1`；该实验后端使用
两个时间块估计、第三个时间块验证，并在证据不足时保留 GeoCalib。通过后手部
PnP、相机运动和深度换算共享同一个 K。
该先验属于启发式保护。GeoCalib 模块可使用 ViPE 中的
`vipe/priors/geocalib`，另需 `pip install kornia`；checkpoint 可用
`--geocalib_weights` 指定本地文件，默认 `pinhole` 会获取其官方权重。
没有 GeoCalib 时会报错；显式 `--camera_estimator assumed` 可使用未标定的近似相机，
结果会标记为 `assumed`，不能视作真实标定或准确米制坐标。

相机运动默认使用 `--camera_motion colmap`：CPU SIFT / 序列匹配、增量 SfM 和
bundle adjustment 在每个候选内全程固定内参，WiLoR 手部轮廓及其时间邻域用于屏蔽动态
手部特征。再通过背景地图点的前后向光流检查和 PnP，为其他视频帧估计位姿。
无法定位的首尾和长缺口明确标为无效，相机与手部输出均为 NaN；仅插值短的内部
缺口。整体定位覆盖率不足时显式报错。

显式 `--pose_intrinsics_policy video_consistency` 可增加视频几何验证：触发旧焦距保护时，
分别用原始 K 和保护 K 运行固定内参重建。两套候选在共同注册的帧、相同的背景
SIFT 像素上比较模型深度与地图深度的一致性，同时检查有效帧覆盖率、注册率、
可靠背景观测的空间/时间分布和重投影误差。v2 要求共同背景点的每帧中位数至少
60、P10 至少 30，中位覆盖不少于 6/12 个画面网格；至少 75% 的共同采样帧和
60% 的采样时间跨度有足够支撑。只有深度一致性至少改善 10%、至少 60% 视图
改善且质量门限全部通过时，
才使用保护 K；证据不足时保留原始 K。DA3Metric 复用同一次网络推理，分别按候选
焦距换算深度。最终选择的 K 同时用于相机 PnP、米制尺度与相机 NPZ，候选得分、
失败原因和所选内参写入 `camera_metadata.json`。该策略在当前数据上进行了开发集
回归，尚无独立测试集验证；显式 `--pose_intrinsics_policy fixed` 可保留原始内参
策略。图像宽度先验不等于真实标定。候选选择不读取测试 Zarr。
手部相机拟合默认使用固定形状 PnP；可显式使用
`--camera_fit pnp_ransac` 对少量异常 2D 关节点做 RANSAC 内点精修。
该选项已经通过合成离群点测试，但在当前冻结视频回放中没有稳定优于默认
`pnp`，因此不作为默认配置。
COLMAP 原生轨迹到米制世界坐标的尺度估计按时间块做共识：每个有效时间块
先独立估计尺度，再用块中位数汇总；块间尺度不一致时拒绝该尺度并保留明确的
质量信息，避免局部背景或单段深度漂移支配整段视频。
地图总点数及其比例保留为诊断信息，候选选择依据共同可靠观测判断地图支撑。
保护 K 候选如果在 8° 初始三角化角度下注册不足，会复用同一组背景匹配依次尝试
4°、2°，保持覆盖率要求；这些重试不会修改内参，也不会把定位失败帧填成有效。

安装 `pycolmap>=4.2,<5`，并准备 UniDepthV2 small 权重，例如：

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("lpiccinelli/unidepth-v2-vits14", local_dir="/path/to/UniDepthV2-ViTS14", allow_patterns=["config.json", "model.safetensors"])'
```

`--unidepth_repo` 指向 ViPE 或其内置 UniDepth 源码目录。UniDepth 接收对应候选的相机
内参，使用静态 COLMAP 背景点的深度比值稳健估计整段共享尺度，不读取源 Zarr 的内参、
头部或手部数据。该尺度来自模型预测，保留不确定性记录，不能当成物理标定。

也可以用 DA3Metric-Large 替换 UniDepthV2 的米制深度尺度后端。它用于把
COLMAP 的相对平移恢复到模型估计的米制尺度；GeoCalib 仍负责内参，COLMAP
仍负责相机位姿。先安装官方 Depth Anything 3 源码包并准备
`depth-anything/DA3METRIC-LARGE` 权重，然后运行：

```bash
python -m pip install -e /path/to/depth-anything-3 --no-deps
python cli.py path-b \
  --video /path/to/video.mp4 \
  --wilor_repo /path/to/WiLoR \
  --checkpoint <WiLoR_checkpoint> \
  --detector <WiLoR_detector> \
  --mano_dir <MANO_models_dir> \
  --geocalib_repo <DynHaMR_dir>/third-party/vipe \
  --geocalib_weights <GeoCalib_pinhole_checkpoint> \
  --camera_motion colmap \
  --metric_depth_backend da3_metric \
  --da3_repo /path/to/depth-anything-3 \
  --da3_metric_model /path/to/DA3Metric-Large \
  --output_dir out_path_b --episode demo_000
```

DA3Metric 输出的是 canonical metric depth。Path B 按官方焦距换算公式在网络
处理分辨率下完成换算，再恢复到视频分辨率，并在静态 COLMAP 背景点上估计
一段共享尺度。`--da3_repo` 在当前环境已经安装官方包时可以省略；
`--da3_metric_process_res` 控制 DA3 处理分辨率，默认 504。保留
`--metric_depth_backend unidepth` 可以继续使用 UniDepthV2 基线。

实验选项 `--hand_depth_alignment` 可使用视频模型深度校正手部平移，再进行时序处理。
它在 wrist 和 MCP 关键点周围采样深度，用 hand K 反投影后，稳健估计整只手的平移，
保留相对手型。深度不足、关节估计不一致、校正过大或 2D 重投影变差时保留原预测。
COLMAP 模式复用已加载的深度模型，但手部深度换算独立使用 hand K；
static 模式启用此选项时也需要提供 metric-depth 权重。

```bash
# 在前面的 path-b 命令中追加这些参数进行受控消融：
--hand_depth_alignment \
--hand_depth_patch_size 7 \
--hand_depth_max_mad 0.03 \
--hand_depth_max_delta 0.15
```

此选项默认关闭。旧代码的 5 段开发集消融未显示整体精度提升，因此迁移后继续
保留为实验开关。`--hand_depth_stride` 默认 1，即处理每个有观测的帧；
较大的值会保留跳过帧的原始预测。Zarr 属性和相机元数据中的
`hand_depth_alignment` 记录通过数量、拒绝原因和平移变化统计。
模型推理异常会显式报错。阈值判定仅使用推理输出；原始 Zarr 仅在预测冻结后由
独立 evaluator 读取。

输出相机轨迹是第一张成功定位帧为原点的 c2w（`camera_reference_frame_index`，
通常为 0），供 Dyn-HaMR、世界系平滑、
Zarr 关键点和 `obs_head_pose` 一致使用；优化时固定相机尺度。
显式 `--camera_motion static` 使用固定相机假设；不启用手部深度校正时无需深度模型。
`--pose_stride`（3）、`--pose_max_keyframes`（300）、`--pose_depth_samples`（8）、
`--pose_threads`（4）、`--pose_mask_margin`（25 像素）控制重建。
`<episode>_camera_work/` 保存数据库、稀疏地图及质量记录；
`<episode>_camera.npz` 保存完整轨迹、原生尺度轨迹、米制深度、观测标记和尺度。
Zarr 中的 `path_b_camera_pose_valid`、`path_b_camera_pose_observed` 会传递到
`load` / `align`。无效帧的有限相机值仅供优化器内部占位，对应观测置信度为零，
对外导出的相机位姿保持 NaN；
retarget 只用有效相机帧计算基座位置和朝向。未定位帧保留时间轴，输出原背景和
空白机器人视图，并在 IK/质量文件的 `camera_pose_valid` 中标为无效；
整段没有有效相机位姿时会明确报错。
`<episode>_wilor_raw.npz` 保存原始 WiLoR 预测供重放。
旧的 DA3-BASE 相机后端仍已移除。DA3Metric-Large 只通过
`--metric_depth_backend da3_metric` 参与米制尺度估计及可选手部平移校正，
不提供 Path B 相机位姿。

原始 WiLoR 2D 是模型自投影，并非独立观测。PnP 保留预测手形的尺度，但不能消除
单目尺度/姿态歧义；输出仍需外部评估。`--camera_fit translation` 可作仅平移消融。
`--max_gap` 默认仅插值不超过 5 帧的内部缺口，长缺口和首尾缺失保持无效；
EE、wrist、3D/2D 关键点的缺失状态一致，保存每手 confidence/observed。
Loader 为 PathB 保留视频时间线，保存 `frame_valid` 和 `source_frame_indices`。
然后直接接现有流水线：

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
  --geocalib_weights <GeoCalib_pinhole_checkpoint> \
  --unidepth_model <UniDepthV2_checkpoint_directory> \
  --dynhamr_gpu 0 \
  --output_dir path_b_dynhamr --episode source_000
```

bridge 会自动准备完整视频帧、左右手 track JSON、相机和 shot 文件，
调用官方 `dyn-hamr/run_opt.py` 的 Root/Smooth 优化，默认迭代上限为
50/300，并采用官方提前停止逻辑。两只已观测的手会一起优化；缺失观测的
2D 置信度保持为零。最终 MANO 参数按正确的手部均值约定转换回相机坐标，
保留优化后的平移，不再逐帧对齐回 WiLoR 手腕。

官方 `fa9cd74` 版本需要以下兼容补丁，以延迟加载默认未启用的 HMP/BMC
资源、去掉未使用的 `mano` 导入、修正缺失的默认数据配置，并让固定相机
读取提供的标定文件。补丁还会保留缺失观测的零置信度和不可见标记；
MANO 初始化可以插值，但缺失的 2D 观测不会参与优化损失。加载器保留片头和片尾
没有观测的帧，使优化结果与原视频保持相同的帧号：

```bash
git -C <DynHaMR_dir> apply <ego2robot_dir>/steps/dynhamr_compat.patch
python -m pip install tensorboard loguru torchgeometry==0.1.2 gdown
```

加载器集成测试从已有仓库读取原始 `fa9cd74` 源码，在临时目录中应用补丁，
不会修改该仓库。在 `tools/ego2robot` 目录运行：

```bash
EGO2ROBOT_DYNHAMR_REPO=<DynHaMR_dir> python -m unittest tests.test_dynhamr_loader -v
```

测试需要 CPU 运行依赖（`torch`、`imageio`、OpenCV、NumPy、SciPy），无需模型
资源；未设置仓库环境变量时会跳过这些集成测试。

使用 WiLoR 初始化时不需要重复运行 HaMeR 或 SLAM。默认未启用 HMP motion
prior，BMC 的默认损失权重为零；要开启这些可选阶段，仍需按官方 README
准备相应权重、统计资源与依赖。相机内参来自视频估计；默认 COLMAP 模式向优化器
提供估计的动态相机轨迹，显式 static 模式则使用固定相机假设。
`--dynhamr_gpu -1` 可让优化器在 CPU 运行，WiLoR 仍按自身设备设置运行。

优化日志、输入帧/JSON 和官方结果保存在
`<output_dir>/<episode>_dynhamr/`，便于回读验证。`--predictions` 也会保留
WiLoR NPZ 中的 2D 和 MANO 参数，可从已有完整预测重跑官方优化器。
未启用 Dyn-HaMR 时，也支持只有 3D/2D 关键点的离线预测；原始缓存只保存每条
检测都具有的可选字段。官方 Dyn-HaMR 仍要求完整的 MANO 初始化参数。

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
  --geocalib_weights <GeoCalib_pinhole_checkpoint> \
  --unidepth_model <UniDepthV2_checkpoint_directory> \
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
`right.obs_keypoints` 保存重建世界坐标中的 21 点 MANO 手部轨迹（static 模式使用相机坐标），
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
| DA3Metric-Large `config.json` + `model.safetensors` 及官方 Python 包 | `path-b --da3_metric_model`、`--da3_repo` | Path B 可选米制尺度后端；需配合 `--metric_depth_backend da3_metric` |
| 可选 Path B：WiLoR `wilor_final.ckpt`、`detector.pt`、MANO `MANO_RIGHT.pkl` | `path-b --wilor_repo`、`--checkpoint`、`--detector`、`--mano_dir` | 另需 WiLoR 源码/依赖及许可允许的 MANO 资产，见前文 Path B |
| 可选 DynHaMR 源码和 MANO 资源 | `path-b --dynhamr_repo` | WiLoR 初始化的 Root/Smooth 路径可复用已有 MANO；其他可选阶段按 Dyn-HaMR README 准备对应权重和依赖 |

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
