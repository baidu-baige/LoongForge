# VLM 数据集转换

## 1. 数据集格式与处理

考虑到多模态数据集的多样性，本项目采用 Energon 加载器来提升数据处理性能，该加载器要求数据集以标准 WebDataset 格式存储。WebDataset 以原生文件格式（jpg、mp4 等）存储数据，允许各种原生多模态数据集简单地压缩并转换为 WebDataset 格式，然后由 Energon 读取。

参考文档：

* Energon：[https://nvidia.github.io/Megatron-Energon/](https://nvidia.github.io/Megatron-Energon/)
* WebDataset：[https://huggingface.co/docs/hub/datasets-webdataset](https://huggingface.co/docs/hub/datasets-webdataset)

本目录提供 `tools/data_preprocess/vlm/convert_to_webdataset.py` 用于将 `.json/.jsonl` 标注文件 + 原始媒体文件（图像/视频）转换为 Energon 可直接读取的 WebDataset 目录（同时生成 Energon 所需的索引和 `dataset.yaml`）。

## 2. 支持的数据类型（`--sample_type`）

根据 `--sample_type` 写入不同的 `dataset.yaml` 文件，决定了 tar 文件内样本的字段组织方式：

常见预训练数据集可使用 caption 格式，SFT 数据集可使用 VQA 格式。对于多图像或混合 SFT 需求，推荐使用 `multi_mix_qa` 格式。

| sample_type | 适用场景 | 说明 |
|-------------|---------|------|
| `vqa` | 单图 VQA | 生成 `VQASample` 映射，图像字段为 `jpg`，文本从 `json[...]` 中提取 |
| `caption` | 单图描述 | 生成 `CaptioningSample` 映射，图像字段为 `jpg`，文本从 `json[...]` 中提取 |
| `multi_mix_qa` | 多图/视频混合 QA | 使用 `CrudeWebdataset`，通过 `subflavors.sample_type` 传递给下游 cooker 进行解析 |
| `multi_vid_vqa` | 多视频 VQA | 同上 |
| `packed_captioning` / `packed_vqa` / `packed_multi_mix_qa` | 离线打包后的数据 | 通常由 `offline_packing` 工作流生成（见第 2 节） |
| 其他字符串 | 自定义场景 | 仍写入 `CrudeWebdataset`，但需确保下游实现了对应的 `sample_type` 解析逻辑 |

注意事项：

* `--media` 仅用于写入数据集元数据（用于区分 image/video/mix）。实际样本是否包含图像/视频取决于每条数据是否包含 `image(s)` / `video(s)` 字段。
* 如果一条数据既没有 `image(s)` 也没有 `video(s)`，将被写入为"纯文本样本"（仅包含 `json`）。

## 3. 转换脚本使用方法

支持的输入文件：

* `--json_file`：`.json`（list[dict]）或 `.jsonl`（每行一个 dict）
* `--image_dir` / `--video_dir`：原始媒体文件根目录（条目中存储的是相对路径）

```bash
python tools/data_preprocess/vlm/convert_to_webdataset.py \
  --output_dir /workspace/wds_data/ \
  --json_file tests/datasets/vlm/mllm_demo.json \
  --image_dir tests/datasets/vlm/ \
  --video_dir tests/datasets/vlm/ \
  --media mix \
  --columns_messages messages \
  --maxcount 10000 \
  --maxsize 3000000000 \
  --sample_type multi_mix_qa
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | - | 输出目录（生成 `pretrain-*.tar` + Energon 元数据目录） |
| `--json_file` | - | 输入 `.json/.jsonl` 文件 |
| `--image_dir` | - | 图像根目录（样本包含 `image(s)` 或 `sample_type=vqa/caption` 时必需） |
| `--video_dir` | - | 视频根目录（样本包含 `video(s)` 时必需） |
| `--media` | `image` | `image/video/mix` |
| `--columns_messages` | `messages` | 条目中对话/文本字段的键名 |
| `--maxcount` | `10000` | 每个分片（tar）的最大样本数 |
| `--maxsize` | `3000000000` | 每个分片（tar）的最大字节数 |
| `--sample_type` | - | 数据类型（见上表） |

输出说明：

* 输出目录包含 `pretrain-0.tar`、`pretrain-1.tar`...（每个 tar 根据 WebDataset 规范按照 `__key__` 存储若干文件，如 `xxx.jpg`/`xxx.json`/`xxx.0_a.mp4` 等）
* 同时生成 Energon 所需的元数据目录（通常名为 `.wds/`），包含 `dataset.yaml` 和索引文件；训练时 `--data-path` 通常指向 `--output_dir`

## 4. 输入 JSON 约定（常用字段）

每条数据支持以下字段组合（均为相对路径，将与 `--image_dir/--video_dir` 拼接以读取二进制文件）：

* 图像：`image: "a/b.jpg"` 或 `images: ["a/b.jpg", "c/d.jpg"]`
* 视频：`video: "a/b.mp4"` 或 `videos: ["a/b.mp4", "c/d.mp4"]`
* 文本/对话：默认读取 `messages`（可通过 `--columns_messages` 修改）

不同 `sample_type` 的文本字段要求（与脚本生成的 `dataset.yaml` 对齐）：

* `vqa`：`messages` 应支持读取 `json[0][content]` 和 `json[1][content]`（通常为长度 >= 2 的列表，元素包含 `content`）
* `caption`：`messages` 应支持读取 `json[captions][0][content]`（例如 dict 包含 `captions: [{content: ...}]`）
* `multi_mix_qa` / `multi_vid_vqa` 等：脚本写入结构化的 `json`（包含 `texts/media/name`），下游根据对应的 `sample_type` cooker 进行解析

## 5. 离线 Packing 数据处理

在多模态场景中，提供了序列离线 packing 处理方法

详情请参阅 [离线数据打包指南](https://loongforge.readthedocs.io/en/latest/features/offline_data_packing.html)
