# Dataset Conversion for VLM

## 1.Data Format & Processing
To accommodate the diversity of multimodal datasets, this project uses the **Energon loader** for high-performance data handling.  
Energon requires datasets to be stored in standard **WebDataset** format. WebDataset keeps files in their native formats (jpg, mp4, …), so almost any multimodal dataset can be converted by simply archiving the files, which Energon can then read directly.

Reference docs  
- Energon: https://nvidia.github.io/Megatron-Energon/  
- WebDataset: https://huggingface.co/docs/hub/datasets-webdataset  

We provide `tools/data_preprocess/vlm/convert_to_webdataset.py` to turn a `.json/.jsonl` annotation file plus raw media (images/videos) into an Energon-ready WebDataset directory (including the required index and `dataset.yaml`).

## 2.Supported Sample Types (`--sample_type`)
The script writes different `dataset.yaml` files and organizes the tar-internal fields according to `--sample_type`:

| sample_type | Scenario | Description |
|-------------|----------|-------------|
| `vqa` | single-image VQA | Creates `VQASample` map; image field is `jpg`, text read from `json[…]` |
| `caption` | single-image captioning | Creates `CaptioningSample` map; image field is `jpg`, text from `json[…]` |
| `multi_mix_qa` | multi-image / mixed-media QA | Uses `CrudeWebdataset`; downstream cooker parses via `subflavors.sample_type` |
| `multi_vid_vqa` | multi-video VQA | Same as above |
| `packed_captioning` / `packed_vqa` / `packed_multi_mix_qa` | offline-packed data | Usually produced by the `offline_packing` pipeline (see Sec. 2) |
| other string | custom scenario | Still writes `CrudeWebdataset`, but you must implement the corresponding `sample_type` parser downstream |

Notes  
- `--media` is **only** written into dataset metadata to distinguish image/video/mix. Whether a sample actually contains images/videos is determined by the presence of `image(s)` / `video(s)` in each entry.  
- If an entry has **neither** `image(s)` **nor** `video(s)`, it is stored as a text-only sample (only `json`).

## 3.Conversion Script Usage
Accepted inputs  
- `--json_file`: `.json` (list[dict]) or `.jsonl` (one dict per line)  
- `--image_dir` / `--video_dir`: root directories for media (entries store relative paths)

```bash
python tools/data_preprocess/vlm/convert_to_webdataset.py \
  --output_dir /tmp/vlm/wds \
  --json_file /tmp/vlm/data.jsonl \
  --image_dir /tmp/vlm/ \
  --video_dir /tmp/vlm/ \
  --media mix \
  --columns_messages messages \
  --maxcount 10000 \
  --maxsize 3000000000 \
  --sample_type multi_mix_qa
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--output_dir` | yes | – | Output folder (`pretrain-*.tar` + Energon meta) |
| `--json_file` | yes | – | Input `.json/.jsonl` |
| `--image_dir` | no | – | Image root (needed when entries contain `image(s)` or `sample_type=vqa/caption`) |
| `--video_dir` | no | – | Video root (needed when entries contain `video(s)`) |
| `--media` | no | `image` | `image` / `video` / `mix` |
| `--columns_messages` | no | `messages` | Key for dialogue/text field in each entry |
| `--maxcount` | no | `10000` | Max samples per shard (tar) |
| `--maxsize` | no | `3000000000` | Max bytes per shard (tar) |
| `--sample_type` | yes | – | Data type (see table above) |

Outputs  
- `pretrain-0.tar`, `pretrain-1.tar`, … (each tar stores files under `__key__` per WebDataset spec, e.g. `xxx.jpg`, `xxx.json`, `xxx.0_a.mp4`, …).  
- Energon meta folder (usually `.wds/`) containing `dataset.yaml` and index files.  
Point `--data-path` to `--output_dir` during training.

## 4.Input JSON Convention (common fields)
Each entry may contain (all paths relative, joined with `--image_dir`/`--video_dir` at runtime):

- Image: `image: "a/b.jpg"` or `images: ["a/b.jpg", "c/d.jpg"]`  
- Video: `video: "a/b.mp4"` or `videos: ["a/b.mp4", "c/d.mp4"]`  
- Text/chat: read from `messages` by default (change with `--columns_messages`)

Text field requirements for different `sample_type` (aligned with generated `dataset.yaml`):

- `vqa`: `messages` must satisfy reading `json[0][content]` and `json[1][content]` (usually a list of length ≥ 2 with `content`).  
- `caption`: `messages` must satisfy reading `json[captions][0][content]` (e.g. dict contains `captions: [{content: ...}]`).  
- `multi_mix_qa` / `multi_vid_vqa`, etc.: script writes a structured `json` (`texts/media/name`); downstream cooker parses according to the corresponding `sample_type`.

## 5.Offline Packing Data Processing
See [Offline Data Packing Guide](offline_data_packing.md) for details.