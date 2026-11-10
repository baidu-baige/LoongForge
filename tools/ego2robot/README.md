# ego2robot - EgoVerse to 16 Dual-Arm Morphologies LeRobot Dataset Pipeline

Convert first-person (ego) bimanual manipulation videos from the open-source EgoVerse dataset into **LeRobot v3.0 training data for 16 dual-arm robot morphologies**. Each morphology uses its own MuJoCo model, joint layout, end-effector/TCP calibration, and gripper mapping, following the retargeting approach from Qwen-RobotManip (Eq. 1/2/3).

The core flow is: `what the human does -> the robot replays the same motion in MuJoCo -> the robot is composited into the real background after the human hands are removed -> a LeRobot dataset is written`. All paths can be overridden with CLI arguments or environment variables; no server-specific absolute paths are required.

## Pipeline

```
load (read/clean/resample zarr data)
  -> align (build 16-D state + smooth actions)
  -> mask (SAM3 temporal segmentation of visible body, hands, and forearms)
  -> inpaint (ProPainter removes hands to produce a clean background)
  [-> depth (Depth Anything 3 scene depth)]        # optional: run-all --with_depth
  [-> gauss-recon (DVLT metric scene depth)]       # optional: run-all --with_gauss
  -> retarget (dual-base search + morphology-aware IK + depth-aware/alpha compositing)
  -> lerobot (write LeRobot v3.0 dataset, H.264)
  -> validate (health checks)
  -> demo (3x3 split-screen demo video)
```

## Directory Layout

```
ego2robot/
├── cli.py                  # Unified entry point: python cli.py <subcommand>
├── pipeline.py             # run-all orchestration (fixed 01_load..08_demo subdirectories)
├── requirements.txt        # Core dependencies
├── steps/
│   ├── config.py           # Centralized paths; EGO2ROBOT_* environment overrides
│   ├── loader.py           # Step 1 data loading/cleaning
│   ├── path_b.py           # Path B: pure video -> WiLoR/DynHaMR -> EgoVerse-like Zarr
│   ├── action_alignment.py # Step 2 action construction
│   ├── hand_mask.py        # Step 3 SAM3 body/hand/forearm temporal segmentation
│   ├── inpaint.py          # Step 4 ProPainter background completion
│   ├── depth_estimate.py   # Step 5 optional DA3 depth estimation
│   ├── gauss_recon.py      # Optional DVLT metric scene depth
│   ├── gauss/              # DVLT inference, SLAM projection and export
│   ├── robot_registry.py   # Registry of 16 morphologies and MJCF/gripper combinations
│   ├── robot_retarget.py   # Step 6 compatibility facade and CLI entry point
│   ├── retarget/            # Step 6 target, IK, base-search, collision, and render modules
│   │   ├── targets.py       # Human-hand target construction and TCP conversion
│   │   ├── ik.py            # Mink/DLS inverse kinematics
│   │   ├── base_search.py   # Base placement and collision screening
│   │   ├── rendering.py     # MuJoCo models, cameras, and gripper rendering
│   │   ├── episode.py       # Per-episode orchestration and artifact writing
│   │   └── cli.py           # Retarget-specific CLI implementation
│   ├── lerobot_writer.py   # Step 7 LeRobot v3.0 dataset writer
│   ├── export_hdf5.py      # Step 7b optional HDF5 exporter
│   └── ...
└── examples/
    └── tea_demo_3x3.mp4    # Example output (3x3 split screen with depth tiles)
```

### Path B: pure video without hand annotations

The paper's Path B starts with WiLoR hand reconstruction and optional DynHaMR
temporal refinement. The checkpoint does not contain the WiLoR Python source,
so provide the source checkout separately:

```bash
git clone --depth 1 https://github.com/rolpotamias/WiLoR.git <WiLoR_dir>
```

Download the WiLoR checkpoint and detector weights:

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt
```

Download the required MANO right-hand model into the directory passed via
`--mano_dir` (or `EGO2ROBOT_MANO_DIR`):

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

The command writes `out_path_b/demo_000/`, an EgoVerse-like Zarr episode
containing video frames, left/right MANO 21-point tracks, synthetic fixed-camera
head pose, and camera intrinsics. Feed that directory to the existing pipeline:

```bash
python cli.py run-all --input_dir out_path_b --output_dir out_demo \
  --robot_type panda --episodes demo_000
```

For an offline protocol check, `--predictions` accepts a directory of NPZ/JSON
files containing `pred_keypoints_3d` (or `keypoints_3d`/`joints_3d`) in the
OpenPose-compatible 21-point hand order used by WiLoR (wrist, thumb, index,
middle, ring, pinky). Each file may also contain `frame`, `right`, and `score`
fields. A custom DynHaMR command
can be supplied with `{input}` and `{output}` placeholders.

Single-hand videos are supported. The missing hand remains invalid in the
episode and is skipped by hand-target IK; no synthetic mirrored hand track is
created.

The official Dyn-HaMR repository can be connected with `--dynhamr_repo`:

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

The bridge prepares the official video/track/camera layout, runs
`dyn-hamr/run_opt.py` with a static-camera fallback, and converts the final
MANO parameters back to `dynhamr_predictions.npz`. The Dyn-HaMR checkout still
needs its own `_DATA` assets and dependencies; run its `scripts/prepare.sh`
and follow its README for HaMeR/BMC/HMP resources.

The official WiLoR runtime also needs its Python dependencies (`ultralytics`,
`smplx`, `pytorch-lightning`, `yacs`, and `scikit-image`). Set `--mano_dir` to
the MANO `models` directory (or use `EGO2ROBOT_MANO_DIR`); the adapter links
`MANO_RIGHT.pkl` into the WiLoR checkout at startup. The default is
the MANO `models` directory. The checkpoint and detector files from the same
WiLoR release are compatible with this adapter.

The complete direct CLI sequence is:

```bash
cd <ego2robot_dir>

# 1) Video -> WiLoR + Dyn-HaMR -> EgoVerse-like Zarr
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

# 2) Zarr -> masks/inpaint -> retarget -> LeRobot -> demo
python cli.py run-all \
  --input_dir <path_b_output> \
  --output_dir <pipeline_output> \
  --episodes source_000 \
  --robot_type panda \
  --sam3_checkpoint <SAM3_checkpoint> \
  --device cuda:0 \
  --skip_depth
```

The Path-B episode is a Zarr v3 group. Its `left.obs_keypoints` and
`right.obs_keypoints` arrays are 21-point MANO tracks in the synthetic camera
world frame; `images.front_1` stores the original JPEG frames. The downstream
loader converts this group to the numbered pipeline directories. The final
robot/background composite is written to
`06_retarget/<episode>_robot_on_bg.mp4`, and the 3x3 demo is written to
`08_demo/<episode>_demo.mp4`.

## Requirements

- DVLT requires Python **>=3.11**. The existing container uses Python **3.12.3**, Torch **2.8.0a0+5228986c39.nv25.06** (CUDA 12.9), and NumPy **1.26.4**; this is an observed runtime, not a verified clean-install recipe. A CUDA GPU is recommended for inference.
- Internet access is required to install dependencies and download model weights; in offline environments, prepare them separately

### Installation

Use [requirements.txt](requirements.txt) as a dependency inventory, not as a verified combined SAM3/ProPainter/DVLT lock. Its existing NumPy 2.4.6 / Torch 2.5.1 pins differ from the working runtime; DVLT's own metadata pins NumPy 2.4.4 / Torch 2.5.1. Resolve compatibility in an isolated environment before installation, and preserve a working environment rather than blindly upgrading to NumPy 2. See [the setup guide](GAUSS_SETUP_ZH.md) for the observed environment and migration caveats.

```bash
# In an already compatible environment, install the separate DVLT source
python -m pip install -e /path/to/dvlt-source --no-deps
python -m pip check
```

Gauss additionally needs `accelerate`, `huggingface-hub`, `hydra-core`, `omegaconf`, `open3d`, `zstandard`, and `numcodecs` (now listed in requirements). SAM3 and optional DA3 require their own source packages; follow their upstream installation instructions. System `ffmpeg` with libx264 and the MuJoCo rendering libraries are also required. A ProPainter subprocess does not isolate package dependencies. Historical local fixes included ProPainter prerelease Torch-version parsing and missing `scikit-image`/`yapf`/`future`, plus SAM3's missing `assets/bpe_simple_vocab_16e6.txt.gz` copied from the openai-clip source distribution; these fixes are outside this repository.

### Model Assets (Place as Needed in Offline Environments)

| Asset | Default location | Description |
|---|---|---|
| SAM3 weights `sam3.pt` | `--sam3_checkpoint` or `EGO2ROBOT_SAM3_CKPT` | [SAM3 upstream](https://github.com/facebookresearch/sam3): follow its checkpoint access instructions; also install the source package/tokenizer assets |
| DVLT source + `config.json`, `model.safetensors` | `--checkpoint <dvlt_ckpt>` / run-all `--gauss_checkpoint <dvlt_ckpt>` | Local DVLT README links to the [project](https://research.nvidia.com/labs/dvl/projects/dvlt/) and [nvidia/dvlt](https://huggingface.co/nvidia/dvlt); obtain source separately and install with `--no-deps` |
| Conditional DINOv2 ViT-B/14 reg4 `.pth` state dict | `--dinov2_weights` / run-all `--gauss_dinov2_weights` | [DINOv2 upstream](https://github.com/facebookresearch/dinov2); only needed when the DVLT checkpoint lacks patch embedding weights |
| `mujoco_menagerie/*` | `EGO2ROBOT_MENAGERIE_DIR` | XML and mesh assets for the menagerie-backed morphologies |
| `models/jaco`, `models/sawyer_gripper` | Included in this repository | Local Jaco and Sawyer gripper assets |
| `models/SO-ARM100/Simulation/SO101` | Included in this repository | Official SO-ARM101 MuJoCo model and mesh assets |
| ProPainter source + `weights/ProPainter.pth`, `weights/raft-things.pth`, `weights/recurrent_flow_completion.pth` | Under `EGO2ROBOT_PROPAINTER_DIR` | All three files are required; the inference script uses the [v0.1.0 release](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0) |
| DA3-BASE `config.json` + `model.safetensors` and Python package | `depth --model_id` or `EGO2ROBOT_DA3_MODEL_DIR` | Only for DA3 depth / `--with_depth`; follow the [official model list](https://github.com/ByteDance-Seed/depth-anything-3) |
| Optional Path B: WiLoR `wilor_final.ckpt`, `detector.pt`, MANO `MANO_RIGHT.pkl` | `path-b --wilor_repo`, `--checkpoint`, `--detector`, `--mano_dir` | Separate WiLoR source/dependencies and licensed MANO assets; see Path B above |
| Optional DynHaMR source and `_DATA` resources | `path-b --dynhamr_repo` | Follow Dyn-HaMR README / `scripts/prepare.sh` for HaMeR/BMC/HMP assets and its own dependencies |

Sources above are taken from local source/asset documentation; download availability was not tested during this maintenance. No weights are bundled. The DVLT directory must contain `config.json` and `model.safetensors`; `load_patch_embed_weights=false` alone does not prove tensor completeness. See [the setup guide](GAUSS_SETUP_ZH.md) for the complete asset checklist.

### Run the Full Pipeline

```bash
python cli.py run-all --with_depth --robot_type panda \
  --input_dir <egoverse_zarr_dir> \
  --output_dir <output_root> \
  --sam3_checkpoint /path/sam3.pt \
  --episodes <ep1> <ep2> ...
```

To reduce visual interpenetration caused by bases being too close to the camera, use `--base_pullback` (in meters). The default is `0`; after base search, both bases are moved along the direction away from the head-view ray and the final IK is recomputed with the moved bases:

```bash
python cli.py run-all --robot_type xarm7 --base_pullback 0.05 \
  --input_dir <egoverse_zarr_dir> --output_dir <output_root> \
  --sam3_checkpoint /path/sam3.pt
```

Available `--robot_type` values are: `panda`, `xarm7`, `arx_l5`, `piper`, `yam`, `fr3`, `ur5e`, `ur10e`, `kinova_gen3`, `sawyer`, `iiwa`, `jaco`, `viperx`, `widowx`, `aloha_agilex`, and `so_arm101`. SO-ARM101 uses the repository's official `new_calib` MuJoCo model and a 5-DOF position-priority IK configuration. Use a separate `--output_dir` for each morphology to avoid mixing different state dimensions. Output directories always keep their step numbers: `01_load/ ... 04_inpaint/ [05_depth/] [05_gauss/] 06_retarget/ 07_lerobot/ 08_demo/`. Optional depth directories are omitted when disabled; later directory names remain unchanged.

In `depth-aware` mode, SO-ARM101 also estimates a static tabletop support plane from metric scene depth and constrains both base soles to that plane. If a reliable plane or camera intrinsics are unavailable, it reports the fallback and retains the previous hand-anchored height search. Other morphologies do not use this constraint.

### Run Individual Steps

```bash
python cli.py load    --input_dir <zarr> --output_dir out01 --target_fps 30
python cli.py align   --input_dir out01 --output_dir out02
python cli.py mask    --zarr_dir <zarr> --output_dir out03 \
  --sam3_checkpoint /path/sam3.pt --max_frames 120 --device cuda:0   # Add --max_frames for a quick trial
python cli.py mask    --zarr_dir <zarr> --output_dir out03 \
  --sam3_checkpoint /path/sam3.pt --device cuda:0
python cli.py inpaint --zarr_dir <zarr> --mask_dir out03 --output_dir out04 \
  --mask_dilation 4 --fp16 --ref_stride 10 --neighbor_length 10 \
  --subvideo_length 80 --raft_iter 20
python cli.py depth   --input_dir out04 --output_dir out05 --model_id /path/da3  --batch 4 --device cuda
python cli.py retarget --robot_type xarm7 --zarr_dir <zarr> --bg_dir out04 --state_dir out02 --output_dir out06 \
  --scene_depth_dir out05 --depth_mode depth-aware --depth_epsilon 0.02 --base_pullback 0.05
# Fall back to the original mask/alpha compositing (no depth read or rendering):
# python cli.py retarget --zarr_dir <zarr> --bg_dir out04 --state_dir out02 --output_dir out06 \
#   --depth_mode alpha
python cli.py lerobot --ik_dir out06 --state_dir out02 --bg_video_dir out06 --output_dir out07
python cli.py validate --dataset_dir out07 --skip_loader_check
python cli.py demo --zarr_dir <zarr> --mask_dir out03 --inpaint_dir out04 --ik_dir out06 \
  --depth_dir out05 --output_dir out08 --episodes <ep>
```

## Output Data Format

### Robot Configuration Declaration (State/Action Dimensions)

`retarget --robot_type` selects the target morphology. The IK output's `state`, `state_names`, and `robot_type` are read automatically by `lerobot`. State dimensions may differ between morphologies, but a single LeRobot dataset must not mix them.

### Output Streams

| Field | Type | Description |
|---|---|---|
| `observation.state` / `action` | float32 `[D]` | Joint positions / joint increments (delta), in radians |
| `observation.images.ego` | video | 640x360 H.264 at 30 fps, composited robot and background frames |
| `observation.masks.ego` | video | Same-size grayscale robot instance mask |
| `observation.images.wrist_l` / `observation.images.wrist_r` | video | Same-size RGB video from the left/right wrist cameras |
| `timestamp/frame_index/episode_index/index/task_index` | scalar | LeRobot frame-alignment fields |
| `ik/ok_ratio`, `ik/pos_err_mean_mm`, `ik/rot_err_mean_deg` | quality columns | Per-episode IK quality in meta/episodes parquet |

### Format Options

- `lerobot` (default): LeRobot v3.0 directory layout
- `hdf5` (optional): `python cli.py export --input <lerobot dataset directory> --output out.hdf5 [--images]`; writes robomimic/ACT-style H5 with per-episode `observations/state`, `actions`, and optional `observations/images`

## Example

`examples/tea_demo.mp4`: complete single-episode prepare-tea result on a narrow table (3x3 split screen, including a DA3 depth tile). Reproduce the full set of intermediate artifacts from any EgoVerse sample with:

```bash
python cli.py run-all --with_depth --robot_type panda --input_dir <egoverse> --output_dir ./out --episodes <ep>
```

## Gaussian Scene Reconstruction with DVLT

`gauss-recon` / `run-all --with_gauss` adds DVLT feed-forward reconstruction, metric scene depth, visualization and caching for retarget depth-aware compositing. **There is no gsplat trainer, Gaussian splatting renderer, or novel-view synthesis entry point.** Raw caches support offline analysis and future extensions.

DVLT reads raw Zarr images, which may still contain hands; the inpainted background is used only to determine output size/cropping. Scale is fitted from world-frame hand keypoints, falling back to a trajectory-length ratio when unavailable or unsuccessful. Depth is unprojected through SLAM `obs_head_pose` and reprojected per frame. Raw DVLT poses/world points remain in the model coordinate frame; pose similarity registration is diagnostic. Static or near-static cameras and moving hands can produce degenerate scale, holes and incorrect geometry; this is not a guarantee of better depth than DA3 or usable novel views.

```bash
python cli.py gauss-recon --zarr_dir <zarr_root> --state_dir <out>/02_align \
  --bg_dir <out>/04_inpaint --output_dir <out>/05_gauss \
  --checkpoint <dvlt_ckpt> --episodes <ep> --sample_fps 1

python cli.py run-all --input_dir <zarr_root> --output_dir <out> --episodes <ep> \
  --sam3_checkpoint <sam3.pt> --robot_type panda \
  --with_gauss --gauss_checkpoint <dvlt_ckpt> --scene_backend gauss
```

Except for `--check`, all four arguments `--zarr_dir`, `--state_dir`, `--bg_dir`, and `--output_dir` are required. Missing state/background files fall back to the Zarr frame count/source image size; this does not make the arguments optional or guarantee downstream frame/crop alignment.

`--with_depth` and `--with_gauss` can both run, writing `05_depth/` and `05_gauss/`. Default `--scene_backend auto` prefers enabled Gauss, otherwise enabled DA3. Pair explicit `da3` with `--with_depth`, and explicit `gauss` with `--with_gauss`; the selector does not enable stages. The current implementation still falls back to enabled DA3 if Gauss is disabled, even with selector `gauss`. `--skip_depth` or `--depth_mode alpha` disables both stages and depth compositing.

| Standalone option | run-all option | Default |
|---|---|---|
| `--checkpoint` | `--gauss_checkpoint` | Standalone `nvidia/dvlt`; pipeline `None`, provide a local directory explicitly |
| `--dinov2_weights` | `--gauss_dinov2_weights` | `None`, only for missing patch embedding weights |
| `--sample_fps` | `--gauss_sample_fps` | `2.0` (standalone example uses `1`) |
| `--fy` | `--gauss_fy` | `None`, DVLT focal length scaled to output height |
| `--conf_threshold` | `--gauss_conf_threshold` | `0.2` |
| `--points_per_frame` | `--gauss_points_per_frame` | `8192` |
| `--inference_steps` | `--gauss_inference_steps` | `None`, model choice |
| `--device` | `--device` | `cuda:0` |

Standalone also exposes `--img_size 504`, `--patch_size 14`, and `--hole_fill_iters 6`. The pipeline synchronizes focal length with retarget using `meta.json`'s `fy_recommended_retarget`, unless `--gauss_fy` is explicit. Keep focal lengths consistent when running steps separately.

Each `<out>/05_gauss/<ep>/` contains `scene_depth.npz` (float16 `[N,H,W]`, `unit="meters"`, `backend="gauss"`, `K` and crop metadata), `<ep>_dvlt.pt` (raw DVLT/scale/alignment cache), `recon_vis.mp4` (original RGB versus depth), `depth_vis.mp4`, and `meta.json` (scale/coverage/focal lengths).

```bash
python cli.py gauss-recon --help
python cli.py run-all --help
python cli.py validate --help
python cli.py gauss-recon --check --checkpoint <dvlt_ckpt>
# Only when dataset, scene depth and IK artifacts exist
python cli.py validate --dataset_dir <out>/07_lerobot \
  --depth_dir <out>/05_gauss --depth_ik_dir <out>/06_retarget
```

`--check` checks imports, CUDA status and checkpoint filenames; it does not load model tensors or test inference, and missing files may only produce warnings. Depth validation is a sampled diagnostic with fixed rendering field of view and skips missing files; it does not certify Gauss geometry. Add `--skip_loader_check` only when appropriate and record that the LeRobot loader was skipped. See [the setup guide](GAUSS_SETUP_ZH.md) for installation, asset sources and NPZ field checks. This documentation maintenance does not regenerate deleted intermediates; historical end-to-end results are not a new smoke/e2e run.

## Environment Variables

| Variable | Purpose |
|---|---|
| `EGO2ROBOT_PROPAINTER_DIR` | ProPainter repository |
| `EGO2ROBOT_PROPAINTER_STAGE_ROOT` | Temporary PNG directory (tmpfs is recommended) |
| `EGO2ROBOT_MENAGERIE_DIR` | mujoco_menagerie cache |
| `EGO2ROBOT_SAM3_CKPT` | SAM3 checkpoint |
| `EGO2ROBOT_DA3_MODEL_DIR` | DA3 weights directory |
| `EGO2ROBOT_VLM_MODEL` | Qwen3.5 model id/path for quality L3 (or pass `--vlm_model` / `--model_dir`) |
| `MUJOCO_GL` | Rendering backend; `osmesa` is recommended to avoid EGL depth leaks |

## Known Limitations and Migration Notes

- Rendering: EGL can produce large black gaps with newer MuJoCo versions. **Use `MUJOCO_GL=osmesa` on GPU machines.**
- Data assets: episodes without intrinsics borrow them from another episode in the same batch (with the same camera configuration).
- With `--with_depth`, DA3 estimates scene depth from the inpainted background, while retargeting renders robot depth. Pixels are replaced only where the robot is closer to the camera than the background. Use `--skip_depth` or `--depth_mode alpha` to fall back to the original mask/alpha compositing.
- For video playback and MD5 checks, `run-demo` has been verified on 30 fps, 10-second clips.

## External References

- Qwen-RobotManip: <https://github.com/QwenLM/Qwen2-RobotManip>
- EgoVerse: `EgoVerse` on Hugging Face (or an internal mirror)
- LeRobot v3.0: <https://github.com/huggingface/lerobot>
