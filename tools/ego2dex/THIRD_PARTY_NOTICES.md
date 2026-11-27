# Third-Party Notices

ego2robot itself is licensed under Apache-2.0 (see `LICENSE`). It does not
redistribute robot models, network weights or the third-party projects below.
`tools/fetch_hand_assets.py` downloads the robot descriptions at pinned commits,
and the other projects are installed by the user. Each keeps its own license;
check it before using the generated data, in particular for commercial use.

## Robot descriptions (downloaded by `tools/fetch_hand_assets.py`)

| Model | Source (commit) | License |
|---|---|---|
| Panda arm | google-deepmind/mujoco_menagerie `71f066ad`, `franka_emika_panda/` | Apache-2.0 |
| Allegro | dexsuite/dex-urdf `f5e7132f`, `robots/hands/allegro_hand` | BSD (SimLab) |
| LEAP | dex-urdf, `robots/hands/leap_hand` | MIT |
| Shadow | dex-urdf, `robots/hands/shadow_hand` | Apache-2.0 |
| Inspire | dex-urdf, `robots/hands/inspire_hand` | see its `LICENSE.txt` (derived from vendor files) |
| Ability | dex-urdf, `robots/hands/ability_hand` | see its `LICENSE.txt` (derived from vendor files) |
| SVH | dex-urdf, `robots/hands/schunk_hand` | GPL-3.0 |
| SVH reference | fzi-forschungszentrum-informatik/schunk_svh_ros_driver `121dbf87` | GPL-3.0 |
| Wuji | wuji-technology/wuji-hand-description `bbc2f64e` | MIT |
| DexHand021 | DexRobot/dexrobot_mujoco `9ab14230` | Apache-2.0 |

dex-urdf as a whole is MIT; the per-hand directories above carry their own
license files, which take precedence. The script reads the SVH ROS description
only to confirm that joint J5 is fixed in the official model.

## Sharpa hand (obtained by the user)

The Sharpa path uses the official Do-As-I-Do release,
<https://github.com/malik-group/do-as-i-do> at `824591b8` (MIT). The Sharpa
hand model shipped there states no license of its own; obtain permission from
its owner before redistributing it or data rendered with it.

## Perception models (installed by the user)

| Project | Commit | License | Used by |
|---|---|---|---|
| facebookresearch/sam3 | `f1647e8d` | SAM License | `mask` step |
| sczhou/ProPainter | `32678252` | S-Lab License 1.0 (**non-commercial**) | `inpaint` step |
| ByteDance-Seed/depth-anything-3 | `3d835ec1` | Apache-2.0 (code; check the license of the weights you download) | `depth` step |
| rolpotamias/WiLoR | `fcb91131` | see repository | Path B only |
| ZhengdiYu/Dyn-HaMR | - | see repository | Path B only (optional) |
| MANO hand model | - | MANO license (registration required) | Path B only |

ProPainter is licensed for non-commercial use only. For commercial use, replace
the `inpaint` step with an inpainting model whose license allows it.

## Python packages

The packages in `requirements.txt` are installed from PyPI under their own
licenses. Notable ones: MuJoCo (Apache-2.0), mink (Apache-2.0), LeRobot
(Apache-2.0), python-fcl (BSD), trimesh (MIT), Transformers (Apache-2.0),
quadprog (GPL-2.0-or-later), PyAV (BSD).

## Example data

`examples/case_01/` contains one short pick-and-place episode (183 frames,
640x480) with its intermediate pipeline outputs, included to reproduce the
documented results.
