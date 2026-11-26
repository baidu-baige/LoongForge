# ego2robot: egocentric human video to Panda + dexterous hand data

[中文](README_ZH.md)

ego2robot converts first-person bimanual manipulation videos into robot
training data. The human hands are removed from the video, a Panda dual arm
with a dexterous hand replays the motion in MuJoCo, the robot is composited
onto the cleaned background, and the result is written as a LeRobot v3.0
dataset.

Supported hands: `allegro`, `leap`, `wuji`, `inspire`, `shadow`, `svh`,
`ability`, `dexhand021` and `sharpa`. The Panda parallel gripper and the
other arm morphologies of the earlier release remain available through
`--robot_type` with the default `--end_effector parallel_gripper`.

## Pipeline

```
load      read and resample the zarr episode
align     build the 18-D state and smoothed actions
mask      SAM3 temporal segmentation of hands, forearms and body
inpaint   ProPainter removes the hands to give a clean background
[depth]   Depth Anything 3 scene depth (optional, --with_depth)
retarget  Panda base placement, arm IK, hand retargeting, rendering, acceptance
lerobot   LeRobot v3.0 dataset (H.264)
validate  dataset checks
demo      side-by-side preview video
```

For the dexterous hands, `retarget` solves both arms and hands from the
21-point hand tracks of the episode. Every result goes through an independent
acceptance check (`independent_validation.json`). If an episode fails it, the
run stops with an error and the episode is not written to the dataset.

## Installation

The reference environment is the NVIDIA PyTorch 25.06 container (Ubuntu 24.04,
Python 3.12, CUDA 12.9):

```bash
docker build -t ego2robot -f docker/Dockerfile .
docker run --gpus all -it -v $PWD:/workspace/ego2robot ego2robot
```

Outside the container, install torch and torchvision for your CUDA version,
then:

```bash
pip install -r requirements.txt
pip install --no-deps lerobot==0.5.0   # lerobot declares numpy>=2; the pinned stack uses 1.26.4
```

### Robot models

Robot models are not stored in this repository. Download them at pinned
commits and build the normalized MuJoCo models (needs `git` and network
access, about 1 GB):

```bash
python tools/fetch_hand_assets.py          # writes hand_assets/
```

This provides the Panda arm (mujoco_menagerie) and 8 hands. Generated files
contain absolute paths; re-run the script after moving the directory.
The gripper robots (jaco, sawyer, SO-ARM101) reuse the models in
`tools/ego2robot/models`.

**Sharpa** is not downloaded by the script. Obtain the official Do-As-I-Do
release (<https://github.com/malik-group/do-as-i-do>, commit `824591b8`) and
point the pipeline at it:

```bash
export EGO2ROBOT_SHARPA_UPSTREAM=/path/to/do-as-i-do
export PYTHONPATH=$EGO2ROBOT_SHARPA_UPSTREAM/retargeting:$PYTHONPATH
```

### Perception models (full pipeline only)

The bundled example starts after these steps, so it does not need them.

| Step | Project (tested commit) | Setting |
|---|---|---|
| mask | facebookresearch/sam3 (`f1647e8d`) + checkpoint | `pip install -e sam3`; `--sam3_checkpoint` or `EGO2ROBOT_SAM3_CKPT` |
| inpaint | sczhou/ProPainter (`32678252`) + weights | `EGO2ROBOT_PROPAINTER_DIR` (default `third_party/ProPainter`) |
| depth | ByteDance-Seed/depth-anything-3 (`3d835ec1`) + DA3-BASE weights | `pip install -e depth-anything-3`; `--model_id` or `EGO2ROBOT_DA3_MODEL_DIR` |

## Example: case_01

`examples/case_01/` is one 183-frame pick-and-place episode (640x480): the raw
zarr episode in `input/`, the outputs of the load, align, mask and inpaint
steps, and an arm seed. Retarget it onto one hand:

```bash
bash examples/run_case_01.sh allegro        # output: examples/output/case_01_allegro/
```

Main outputs in the result directory:

| File | Content |
|---|---|
| `robot_on_bg.mp4` | robot composited onto the inpainted background |
| `contact_sheet.jpg`, `review_*.jpg` | still frames for review |
| `independent_validation.json` | acceptance result and error metrics |
| `trajectory.npz`, `scene.xml` | solved joint trajectory and MuJoCo scene |
| `motor_actions.npz` | per-frame motor commands of the hand |

The example sets `EGO2ROBOT_ARM_INITIALIZER=examples/case_01/arm_seed`, an
arm trajectory from an accepted Sharpa run of this episode, the one the
published results were solved from. With it, the 9 hands reproduce the
reference results bit for bit. Set the variable to an empty string to solve
the arm from the initial pose instead.

## New episodes

Input is a directory of EgoVerse-style zarr v3 episodes (images, 21-point
hand tracks, head pose, camera intrinsics); `examples/case_01/input` is one.
Run Sharpa first, then reuse its arm trajectory as the seed for the other
hands. Use one `--output_dir` per hand, because the state dimension of the
dataset differs between hands:

```bash
python cli.py run-all --robot_type panda --end_effector sharpa \
  --input_dir examples/case_01/input --output_dir out_sharpa \
  --sam3_checkpoint <sam3.pt> --device cuda:0

export EGO2ROBOT_ARM_INITIALIZER=out_sharpa/06_retarget_sharpa/_render
python cli.py run-all --robot_type panda --end_effector allegro \
  --input_dir examples/case_01/input --output_dir out_allegro \
  --sam3_checkpoint <sam3.pt> --device cuda:0
```

The seed directory may hold one result or one subdirectory per episode. An
accepted Sharpa result contains `arm_seed.npz` (static base poses and arm
joints), so the Sharpa meshes are not needed when it is reused. Only results
that passed acceptance are used; hands and all task, contact and view checks
are solved again. Without a seed the arm solve starts from the initial pose.

Results are written to `<output_dir>/06_retarget_<hand>/_render/<ep>/` and the
dataset to `<output_dir>/07_lerobot/`. `--episodes <ep>` restricts a run to
selected episodes.

Useful `run-all` options for the hands: `--acceptance_profile`
(`mild` default, `strict`, `loose_b`), `--hand_initializer` (`v28-v29`
default, `v28`, `off`), `--hand_rotation_policy` (`stabilize` default,
`original`), `--hand_depth_mode` (`alpha` default, `depth-aware`).
Individual steps can be run with `python cli.py <step> --help`
(`load`, `align`, `mask`, `inpaint`, `depth`, `retarget`, `lerobot`,
`validate`, `demo`, `export`).

### Video without hand tracks (Path B)

`python cli.py path-b` turns a plain video into an EgoVerse-style episode with
WiLoR hand reconstruction and optional Dyn-HaMR refinement. It needs the WiLoR
source (`--wilor_repo`), its checkpoint and detector, and the MANO right-hand
model (`--mano_dir` or `EGO2ROBOT_MANO_DIR`). See `python cli.py path-b --help`.

## Environment variables

| Variable | Purpose |
|---|---|
| `EGO2ROBOT_HAND_ASSETS` | normalized hand models (default `hand_assets/normalized`) |
| `EGO2ROBOT_MENAGERIE_DIR` | mujoco_menagerie checkout (default `hand_assets/vendor/mujoco_menagerie`) |
| `EGO2ROBOT_SHARPA_UPSTREAM` | official Do-As-I-Do checkout for Sharpa |
| `EGO2ROBOT_ARM_INITIALIZER` | arm seed directory (see above) |
| `EGO2ROBOT_ELASTIC_WORKERS` | parallel workers for the registered-hand solve, not Sharpa (default 1) |
| `EGO2ROBOT_ACCEPTANCE_PROFILE`, `EGO2ROBOT_HAND_INITIALIZER` | defaults for the options of the same name |
| `EGO2ROBOT_SAM3_CKPT`, `EGO2ROBOT_PROPAINTER_DIR`, `EGO2ROBOT_PROPAINTER_STAGE_ROOT`, `EGO2ROBOT_DA3_MODEL_DIR` | perception models |
| `MUJOCO_GL` | `osmesa` (default in the container) or `egl` |

The solver settings used for the published results are in
`steps/config.py` (`HAND_SOLVER_DEFAULTS`); each can be overridden through
its environment variable.

## Notes

- Rendering: results were verified with OSMesa. EGL is faster, but on some
  GPUs it produced corrupted frames. Compare a few frames against OSMesa before
  switching.
- Solve time is minutes to hours per episode and hand, depending on the
  episode length and the CPU. Set `EGO2ROBOT_ELASTIC_WORKERS` to use more cores
  for the registered hands.
- Episodes without camera intrinsics borrow them from another episode in the
  same directory.
- If the mask step fails inside triton with a `UnicodeDecodeError`, the
  host's `ldconfig -p` output contains non-UTF-8 bytes. Put a wrapper first on
  `PATH` (or bind it over `/sbin/ldconfig` in the container) that filters it:
  `/sbin/ldconfig.real "$@" | LC_ALL=C tr -cd '\11\12\15\40-\176'`.
- The hand models come from public descriptions and have not been checked
  against hardware calibration.

## License

Apache-2.0, see `LICENSE`. Robot models, perception models and Python packages
keep their own licenses; see `THIRD_PARTY_NOTICES.md`. ProPainter is licensed
for non-commercial use only.
