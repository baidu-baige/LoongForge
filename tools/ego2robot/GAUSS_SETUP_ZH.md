# Gauss 场景重建（DVLT）安装与验证

## 能力与边界

`gauss-recon` / `run-all --with_gauss` 提供 DVLT 前馈多视角重建、米制场景深度、可视化与缓存，并接入 retarget 深度遮挡合成。当前没有 gsplat trainer、Gaussian splatting renderer 或 novel-view synthesis 入口；不需要安装 gsplat。

输入图像来自原始 EgoVerse Zarr，仍可能包含人手；`04_inpaint` 的背景视频只用于确定输出尺寸与裁剪范围，并未作为 DVLT 图像输入。尺度优先由世界系手部关键点拟合，失败或无关键点时回退轨迹长度比；再通过 SLAM `obs_head_pose` 反投影、逐帧重投影。DVLT 位姿的相似变换用于诊断，不能将缓存中的原始 `world_points` 当作 SLAM 米制点云。静态/近静态相机、动态人手、缺乏视差可能导致尺度退化、孔洞和错误几何，不保证优于 DA3，也不保证新视角质量。

## 环境与依赖

本地 DVLT `pyproject.toml` 要求 **Python >=3.11**。2026-11-04 在现有容器中读取到 Python **3.12.3**、Torch **2.8.0a0+5228986c39.nv25.06**（CUDA **12.9**）、NumPy **1.26.4**。这是既有环境信息，不代表从零安装已验证。历史 runtime 记录在该环境做过整链运行；本次文档维护不重建已删除的中间产物。

以 [requirements.txt](requirements.txt) 为依赖清单，但它的既有锁版本（如 NumPy 2.4.6、Torch 2.5.1）与运行环境不同，**不是已验证的 SAM3/ProPainter/DVLT 联合安装锁文件**。DVLT 自身元数据也锁了 NumPy 2.4.4、Torch 2.5.1、torchvision 0.20.1 等版本；不要直接在工作环境执行带依赖的 editable install 覆盖它们。本次保留原锁版本，只补充 Gauss 缺失项。

在隔离的新环境中先根据 CUDA 驱动、SAM3 与 ProPainter 的实际要求解析版本兼容性；在已有可运行环境中只补缺失项并检查导入。不要为了 Gauss 盲目升级到 NumPy 2。`pip check` 可以揭示声明冲突，但不能替代推理验证。ProPainter 子进程本身不隔离 Python 依赖，仍需关注同一环境中的 OpenCV/NumPy 兼容性。

```bash
# 已有兼容环境；这些安装步骤不是本次验证执行的命令
python -m pip install accelerate huggingface-hub hydra-core omegaconf open3d zstandard numcodecs
python -m pip install -e /path/to/dvlt-source --no-deps
python -m pip check
```

还需 torch/torchvision、safetensors、Pillow、OpenCV、SciPy 等基础依赖（参见 requirements 和 DVLT 元数据）。`open3d` 是 DVLT 几何模块的导入依赖；`zstandard`/`numcodecs` 用于 Zarr shard 解码。系统 PATH 需有支持 libx264 的 `ffmpeg`；retarget 还需 MuJoCo 所用的 OSMesa/EGL 库，现有环境使用 `MUJOCO_GL=osmesa`。显存和产物大小随采样帧数、分辨率而变，先用较低 `--sample_fps` 评估。

历史环境额外修复（均在仓库外，不随本工具分发）：ProPainter 补装 `scikit-image`、`yapf`、`future`，修复其 torch 预发行版本字符串解析；SAM3 wheel 缺失的 `assets/bpe_simple_vocab_16e6.txt.gz` 从 openai-clip 源码分发包补齐。迁移时应先检查安装包是否仍有这些问题，不能假设修复自动存在。

## 模型、源码及来源

以下路径均为示例，请替换为自己的路径。源码包和权重分别准备，离线环境需提前下载。外部来源依据现有源码/资产说明列出，本次没有联网验证下载可用性。

| 阶段 | 必需文件 / 资产 | 路径或参数 | 来源 |
|---|---|---|---|
| SAM3 mask | `sam3.pt`，另需 SAM3 Python 包及 tokenizer 资源 | `--sam3_checkpoint /path/to/sam3.pt` 或 `EGO2ROBOT_SAM3_CKPT` | [SAM3 官方仓库](https://github.com/facebookresearch/sam3)，按其 checkpoint 获取及访问许可说明准备 |
| DVLT Gauss | 源码包 + `config.json`、`model.safetensors` | `pip install -e /path/to/dvlt-source --no-deps`；单步 `--checkpoint /path/to/dvlt_ckpt`，整链 `--gauss_checkpoint` | 本地 DVLT README 指向 [项目页](https://research.nvidia.com/labs/dvl/projects/dvlt/) 和 [nvidia/dvlt](https://huggingface.co/nvidia/dvlt)；源码从项目维护者提供的分发渠道获取，不假设 Git clone 地址 |
| 条件 DINOv2 | ViT-B/14 with 4 registers 对应的 `.pth` state dict | 单步 `--dinov2_weights`；整链 `--gauss_dinov2_weights` | [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2) 的对应模型；仅 DVLT checkpoint 缺少 patch embedding 时需要 |
| ProPainter inpaint | `weights/ProPainter.pth`、`weights/raft-things.pth`、`weights/recurrent_flow_completion.pth` | 放在 `EGO2ROBOT_PROPAINTER_DIR` 指向的源码目录下 | [ProPainter v0.1.0 release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)，本地 inference 脚本使用该版本下载目录 |
| DA3（仅 depth / `--with_depth`） | `config.json`、`model.safetensors` + `depth-anything-3` Python 包 | `depth --model_id /path/to/DA3-BASE`；整链用 `EGO2ROBOT_DA3_MODEL_DIR` | [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3)，按官方模型清单获取 DA3-BASE；不是固定“4 个文件” |
| MuJoCo retarget | 对应 morphology 的 XML/mesh | `EGO2ROBOT_MENAGERIE_DIR`；部分模型在本仓库 `models/` | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 与仓库内置模型 |
| Path B（纯视频，可选） | WiLoR 源码、`wilor_final.ckpt`、`detector.pt`、`MANO_RIGHT.pkl` | `--wilor_repo`、`--checkpoint`、`--detector`、`--mano_dir`（或 `EGO2ROBOT_MANO_DIR`） | [WiLoR](https://github.com/rolpotamias/WiLoR) 及其模型分发说明；遵循 MANO 模型许可 |
| Path B DynHaMR（可选时序优化） | Dyn-HaMR 源码、`_DATA` 中 HaMeR/BMC/HMP 等资源 | `--dynhamr_repo`、`--dynhamr_gpu` | [Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR)，按 README 和 `scripts/prepare.sh` 准备独立依赖与资产 |

WiLoR 还依赖 `ultralytics`、`smplx`、`pytorch-lightning`、`yacs`、`scikit-image` 等；这些不是 Zarr→Gauss 单步所需模型。Path B 完整命令见 [中文 README](README_ZH.md) / [English README](README.md)。DVLT checkpoint 目录应为：

```text
dvlt_ckpt/
├── config.json
└── model.safetensors
```

本地 config 指定 `load_patch_embed_weights=false` 和 `dinov2_vitb14_reg`。该配置本身不能证明权重完整；加载器在加载结果报告缺少 patch embedding 时才尝试额外 DINOv2。`--check` 只检查导入和目录文件，不实例化模型、不检查所有 tensor，也不能证明推理成功。它对缺少 checkpoint 文件可能只打印警告，须检查输出而非只看退出码。

## 单集与完整管线

从工具目录执行，替换尖括号占位符。除 `--check` 外，**`--zarr_dir`、`--state_dir`、`--bg_dir`、`--output_dir` 四项均必须传入**。state 文件不存在时回退 Zarr 帧数，bg 视频不存在时回退原图尺寸；这是内容缺失回退，不是可省略 CLI 参数，也不保证与后续 retarget 帧数/裁剪一致。

```bash
python cli.py gauss-recon \
  --zarr_dir <zarr_root> \
  --state_dir <out>/02_align \
  --bg_dir <out>/04_inpaint \
  --output_dir <out>/05_gauss \
  --checkpoint <dvlt_ckpt> \
  --episodes <ep> --sample_fps 1

python cli.py run-all \
  --input_dir <zarr_root> --output_dir <out> --episodes <ep> \
  --sam3_checkpoint <sam3.pt> --robot_type panda \
  --with_gauss --gauss_checkpoint <dvlt_ckpt> --scene_backend gauss
```

`--with_depth` 与 `--with_gauss` 可同时启用，分别写 `05_depth/` 与 `05_gauss/`。`--scene_backend auto` 默认优先使用已启用的 Gauss，否则使用已启用的 DA3；显式 `da3` 应配合 `--with_depth`，显式 `gauss` 应配合 `--with_gauss`。选择器不会自动运行未启用的阶段；当前代码在 Gauss 未启用而 DA3 启用时仍会回退 DA3，即使选择 `gauss`。无选中深度时回退 alpha；`--skip_depth` 或 `--depth_mode alpha` 均禁用两个重建阶段及深度合成。

| 单步参数 | run-all 对应参数 | 默认值 |
|---|---|---|
| `--checkpoint` | `--gauss_checkpoint` | 单步 `nvidia/dvlt`；整链 `None`（应显式提供本地路径） |
| `--dinov2_weights` | `--gauss_dinov2_weights` | `None`，条件依赖 |
| `--sample_fps` | `--gauss_sample_fps` | `2.0`；上面单集示例降至 `1` |
| `--fy` | `--gauss_fy` | `None`，DVLT 预测焦距按输出高度缩放 |
| `--conf_threshold` | `--gauss_conf_threshold` | `0.2` |
| `--points_per_frame` | `--gauss_points_per_frame` | `8192` |
| `--inference_steps` | `--gauss_inference_steps` | `None`，由模型决定 |
| `--device` | `--device` | `cuda:0` |
| `--img_size` / `--patch_size` | 不暴露 | `504` / `14` |
| `--hole_fill_iters` | 不暴露 | `6` |

整链会将 Gauss 焦距同步到 retarget（默认读取 `meta.json` 的 `fy_recommended_retarget`，显式 `--gauss_fy` 优先）。单步重建后自行运行 retarget 时也应保持相同焦距。`--fps` 默认取 Zarr 属性或 30；帧索引对应原始 Zarr 前 N 帧，不使用 `01_load` 的 head-relative 数据。

## 输出与验证

每集位于 `<out>/05_gauss/<ep>/`：

| 文件 | 内容 |
|---|---|
| `scene_depth.npz` | `depth (N,H,W)` float16 米，`model="dvlt"`、`unit="meters"`、`backend="gauss"`、`height/width/crop_top/crop_bottom`、`K (3,3)` |
| `<ep>_dvlt.pt` | 原始 DVLT `c2w/K/depth/world_points/conf`、采样索引和尺度/对齐元数据，供离线分析及未来扩展 |
| `recon_vis.mp4` | 原始 RGB 与深度并排抽样展示，不是 novel-view 渲染 |
| `depth_vis.mp4` | 米制深度伪彩色视频 |
| `meta.json` | scale、coverage、align、K、fy_used、fy_recommended_retarget 等 |

```bash
python cli.py gauss-recon --help
python cli.py run-all --help
python cli.py validate --help
python cli.py gauss-recon --check --checkpoint <dvlt_ckpt>

# 只有保留了对应深度及 IK 产物才运行深度诊断
python cli.py validate --dataset_dir <out>/07_lerobot \
  --depth_dir <out>/05_gauss --depth_ik_dir <out>/06_retarget
# 未安装可用 LeRobot loader 时，可追加 --skip_loader_check；需明确记录跳过项
```

`validate` 的深度部分是抽样统计诊断，会跳过缺失文件，且使用固定渲染尺寸/视场，不能代替对 Gauss 内参、几何及遮挡质量的完整验收。独立检查已有产物：

```python
from pathlib import Path
import json
import numpy as np

root = Path("/path/to/out/05_gauss/episode")
with np.load(root / "scene_depth.npz") as data:
    depth = data["depth"]
    assert depth.ndim == 3 and depth.dtype == np.float16
    assert str(data["unit"].item()) == "meters"
    assert str(data["backend"].item()) == "gauss"
    assert data["K"].shape == (3, 3)
    print("shape", depth.shape, "finite", np.isfinite(depth).mean(),
          "median meters", np.nanmedian(depth))
meta = json.loads((root / "meta.json").read_text())
print({key: meta[key] for key in
       ("scale", "coverage", "fy_used", "fy_recommended_retarget")})
```

深度范围应结合真实场景判断；高 coverage 可能来自填洞，不等于几何正确。如果中间产物已清理，不为文档维护重建全链，应明确报告 smoke/e2e 和深度产物检查未重跑。不要提交权重、缓存、视频或访问凭据。
