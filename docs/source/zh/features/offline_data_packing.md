# 离线数据打包
本模块提供"离线序列打包"流水线：它接收一个**样本级**目录（每个样本包含一个 `*.json` 文件及其媒体文件），根据 `max_token_len` 对样本进行分组和重排序，最终生成**打包后的 WebDataset**（`pretrain-*.tar` 加 Energon 元数据文件）。
通过将变长序列拼接至目标长度，减少填充并提高训练吞吐量。

入口脚本：
`tools/data_preprocess/vlm/offline_packing/scripts/pack_wds.sh`（4 个步骤，见下文）。

## 1. 支持的打包场景（`sample.sample_type`）

目前支持单样本字幕、VQA 和多模态混合 QA 格式的打包。

|场景|`sample_type`|说明|
|---|---|---|
|离线打包字幕|`packed_captioning`|生成 `CrudeWebdataset`；下游代码必须解析此 `sample_type`。|
|离线打包单图 QA|`packed_vqa`|同上。|
|离线打包图像+视频混合 QA|`packed_multi_mix_qa`|同上（输入 JSON 必须声明媒体类型和文件列表）。|

## 2. 输入要求（`data.wds_dir`）
实现**不会直接读取 tar 分片**；它期望一个扁平的、可随机访问的目录：

* 多个 `*.json` 文件（每个文件 = 一个样本 / 一个 WDS json 载荷）。
* 媒体文件（图像/视频）位于同一目录中，或通过从该目录解析的相对路径引用。

如果数据已经是 `convert_to_webdataset.py` 生成的 `pretrain-*.tar` 分片，请先解包：

```bash
mkdir -p /path/to/wds_flat
for t in /path/to/wds/pretrain-*.tar; do tar -xf "$t" -C /path/to/wds_flat; done
```

注意事项：

* `get_sample_len.py` 从 `data.template_text_key` 指定的字段读取消息列表；它也接受常用键 `messages` 和 `texts`。
* 如果 JSON 文件来自 `tools/data_preprocess/vlm/convert_to_webdataset.py`（多场景默认写入 `texts`），通常需要将 `data.template_text_key` 设置为 `texts`。
* `packed_vqa` / `packed_captioning`：如果 JSON 不包含显式的 `media_files/name` 字段，代码会尝试查找具有相同主干名称的媒体文件（例如 `0001.json` → `0001.jpg`）。
* `packed_multi_mix_qa`：JSON 必须声明 `media`/`media_type`（`image` 或 `video`）并提供 `name`/`media_files` 列表（允许嵌套列表）。

## 3. 快速开始
```bash
cd tools/data_preprocess/vlm/offline_packing

# 1) 编辑 config.yaml（或复制 packed_vqa_demo.yaml）
# 2) 运行 4 步流水线（默认读取 config.yaml）
bash scripts/pack_wds.sh
```

切换到其他配置：

* 方式 1：将其覆盖/复制到 `config.yaml`
* 方式 2：使用 `--config your.yaml` 手动运行每个脚本（见下一节）

## 4. 流水线详情（对应 `pack_wds.sh`）

### 步骤 1：计算每个样本的 Token 长度（`get_sample_len.py`）
* 输入：`data.wds_dir` 下的 `*.json` + 媒体文件
* 处理：根据 `sample.sample_type` + `model.model_type` 选择模板（`utils.TEMPLATES`），使用 `AutoProcessor` 对文本+视觉输入进行分词，记录每个样本的 Token 长度
* 输出：`{data.wds_dir}/.temp/sample_len_report.txt`（`sample_id: token_len`）

手动运行：
```bash
python get_sample_len.py --config config.yaml
```

### 步骤 2：长度分桶与打包分组（`do_hashbacket.py`）
* 输入：`sample_len_report.txt`
* 处理：构建哈希桶，在 `sample.max_token_len` 约束下将样本打包入"箱子"
* 输出：`{data.packed_json_dir}/bins_boxs.pkl`（每个箱子 = 将被拼接成一个打包样本的样本 ID 列表）

手动运行：
```bash
python do_hashbacket.py --config config.yaml
```

### 步骤 3：生成打包中间 JSON（`prepare_raw_samples.py`）
* 输入：`bins_boxs.pkl` + 原始 `*.json`/媒体
* 处理：按箱子聚合样本，生成包含 `prompts`/`captions`/`media_files`/`media_type` 等字段的打包 JSON
* 输出：`{data.packed_json_dir}/row_packing_jsons/*.json`

手动运行：
```bash
python prepare_raw_samples.py --config config.yaml
```

### 步骤 4：将打包 JSON 写回 WebDataset（`packed_to_wds.py`）
* 输入：`row_packing_jsons/*.json` + 媒体（在 `{data.wds_dir}` 或 `{data.packed_json_dir}/row_packing_images` 下查找）
* 输出：`data.packed_wds_dir/pretrain-*.tar`（如未配置则为 `{data.packed_json_dir}/packed_wds`）加 Energon 元数据（`.wds/dataset.yaml` + 索引）

手动运行：
```bash
python packed_to_wds.py --config config.yaml
```

## 5. 配置（`config.yaml`）
关键字段：

* `data.wds_dir` — 输入样本目录（`*.json` + 媒体）
* `data.template_text_key` — JSON 中的消息字段名（`messages` 或 `texts`）
* `data.packed_json_dir` — 中间 pkl/json 工作目录
* `data.packed_wds_dir` — 最终打包 WDS 输出目录
* `sample.max_token_len` — 目标打包长度（如 8192 / 16384）
* `sample.sample_type` — 见第 1 节
* `model.model_type` — 用于选择模板的模型标识符
* `model.processor_kwargs.*` — 传递给 `transformers.AutoProcessor.from_pretrained` 的 HF 处理器参数
* `packed_wds.maxcount` / `maxsize` — tar 分片拆分策略

示例（摘录，完整字段见 `config.yaml`）：

```yaml
data:
  wds_dir: "/mnt/cluster/.../wds_flat/"
  template_text_key: "messages"
  packed_json_dir: "/mnt/cluster/.../packed_json/"
  packed_wds_dir: "/mnt/cluster/.../packed_wds/"

sample:
  max_token_len: 8192
  sample_type: packed_multi_mix_qa
```

## 6. 切换模型 / 调整图像处理
步骤 1 的 Token 计数取决于实际的 `AutoProcessor` 逻辑，因此可以通过配置更换模型或图像预处理参数：

* 更换模型：将 `model.processor_kwargs.pretrained_model_name_or_path` 设置为所需的 HF 模型/处理器；相应更新 `model.model_type`。
* 调整图像 Token 预算 / 分辨率：在 `model.processor_kwargs` 下添加处理器支持的参数（例如 Qwen-VL 的 `min_pixels`/`max_pixels`）。
* 模板对齐：如果添加了新的 `model.model_type`，确保 `tools/data_preprocess/vlm/offline_packing/utils.py` 中的 `TEMPLATES[sample_type][model_type]` 包含对应条目；否则步骤 1 将报错"No template found for model_type ..."。
* 媒体预处理：在 `media_preprocess` 下可以为每种模态指定预处理函数名（实现在 `tools/data_preprocess/vlm/offline_packing/media_preprocess_utils.py`），以控制缩放/裁剪/帧读取行为。
