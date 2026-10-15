# LingBot-VLA-v2 Examples

Run these scripts from the repository root.

| Script | Purpose |
| --- | --- |
| `run_lingbot_vla_v2_finetune.sh` | Fine-tune LingBot-VLA-v2 on 8 GPUs with `custom_dp` (ZeRO-1 style, Muon). |

## Artifacts

Checkpoints, teachers, the tokenizer and data are shared on BOS under
`${BOS_VLA_ARTIFACTS_ROOT}/lingbot_vla_v2/` and laid out as:

```
lingbot_vla_v2/
├── models/
│   ├── lingbot-vla-v2-6b/          # VLM + expert checkpoint, with depth/ and dino_video/ teachers
│   ├── Qwen3-VL-4B-Instruct/       # tokenizer
│   └── moge-2-vitb-normal/         # depth teacher (model.pt)
├── datasets/
│   └── lerobot_robotwin_eef_aug_500/<task>-aloha-agilex_randomized_500-1000
├── assets/norm_stats/robotwin.json
└── configs/robot_configs/{robotwin,agilex_cobot_magic}.yaml
```

Sync them to `$LOCAL_VLA_ARTIFACTS_ROOT` (default
`/ssd2/loongforge_embodied_ci/vla_artifacts`) with:

```bash
bash tests/embodied/config/prepare.sh
```

The model YAML leaves every checkpoint/teacher path `null`; the launcher fills
them from `$LOCAL_VLA_ARTIFACTS_ROOT/lingbot_vla_v2` and exits early, pointing
at `prepare.sh`, if any is missing. Starting `train.py` directly without the
launcher therefore needs those paths passed as `model.*` overrides.

## Overrides

```bash
export LOCAL_VLA_ARTIFACTS_ROOT=/path/to/vla_artifacts
export TASK_NAME=click_bell                  # task dir under DATASET_DIR
export DATA_PATH=/path/to/dataset_list.txt   # or supply your own "<robot_config> <dataset_dir>" list
export CKPT_PATH=/path/to/lingbot-vla-v2-6b  # depth/ and dino_video/ are read from here
export TOKENIZER_PATH=/path/to/Qwen3-VL-4B-Instruct
export MOGE_PATH=/path/to/moge-2-vitb-normal/model.pt
export OUTPUT_DIR=/path/to/outputs
```
