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
files containing `pred_keypoints_3d` (or `keypoints_3d`/`joints_3d`). Each file
may also contain `frame`, `right`, and `score` fields. A custom DynHaMR command
can be supplied with `{input}` and `{output}` placeholders.

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

- Python **3.10**, CUDA GPU (an A800 or similar; GPU is recommended for SAM3 inference, depth, and rendering)
- Internet access is required to install dependencies and download model weights; in offline environments, prepare them separately

### Installation

```bash
# 1) Create a virtual environment
python3.10 -m venv ~/.venvs/ego2robot && source ~/.venvs/ego2robot/bin/activate
pip config set global.index-url https://pypi.org/simple/

# 2) Install dependencies (core)
pip install -r requirements.txt   # Install torch as needed: pip install torch==2.7.1+cu118 -f <mirror>

# SAM3 (if the git dependency in requirements is not installed automatically)
git clone https://github.com/facebookresearch/sam3 && pip install -e ./sam3

# 3) DA3 (needed by the depth step; the official package is not on PyPI)
git clone https://github.com/ByteDance-Seed/depth-anything-3 && pip install -e ./depth-anything-3
```

### Model Assets (Place as Needed in Offline Environments)

| Asset | Default location | Description |
|---|---|---|
| SAM3 weights `sam3.pt` | Set `EGO2ROBOT_SAM3_CKPT` | Temporal segmentation of body, hands, and forearms |
| `mujoco_menagerie/*` | `EGO2ROBOT_MENAGERIE_DIR` | XML and mesh assets for the menagerie-backed morphologies |
| `models/jaco`, `models/sawyer_gripper` | Included in this repository | Local Jaco and Sawyer gripper assets |
| `models/SO-ARM100/Simulation/SO101` | Included in this repository | Official SO-ARM101 MuJoCo model and mesh assets |
| ProPainter repository and weights | `EGO2ROBOT_PROPAINTER_DIR` | Video background completion (called as an inpaint subprocess) |
| DA3-BASE weights (4 files) | `EGO2ROBOT_DA3_MODEL_DIR` | Depth step weights |

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

Available `--robot_type` values are: `panda`, `xarm7`, `arx_l5`, `piper`, `yam`, `fr3`, `ur5e`, `ur10e`, `kinova_gen3`, `sawyer`, `iiwa`, `jaco`, `viperx`, `widowx`, `aloha_agilex`, and `so_arm101`. SO-ARM101 uses the repository's official `new_calib` MuJoCo model and a 5-DOF position-priority IK configuration. Use a separate `--output_dir` for each morphology to avoid mixing different state dimensions. Output directories always keep their step numbers: `01_load/ ... 04_inpaint/ [05_depth/] 06_retarget/ 07_lerobot/ 08_demo/`. When depth is disabled, `05_depth/` is omitted and the later directory names remain unchanged.

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
