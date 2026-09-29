# 离线数据打包
本模块提供"离线序列打包"流水线：它直接读取源 **WebDataset tar 分片**，根据 `max_token_len` 对样本进行分组和重排序，最终生成**打包后的 WebDataset**（`pretrain-*.tar` 加 Energon 元数据文件）。
通过将变长序列拼接至目标长度，减少填充并提高训练吞吐量。

入口脚本：
`tools/vlm_data_preprocess/offline_packing/scripts/pack_wds.sh`（4 个步骤，见下文）。

## 1. 支持的打包场景（`sample.sample_type`）

WDS-native V1 路径（`wds_pack.cli.scan_manifest`）仅接受以下两种 `sample.sample_type`，其它取值会在扫描阶段被拒绝。

|场景|`sample_type`|说明|
|---|---|---|
|离线打包图像/视频/文本混合 QA|`packed_multi_mix_qa`|输入 WDS JSON 必须声明 `media`/`media_type`；打包按媒体类型同质进行。使用手写模板 `TEMPLATES[sample_type][model_type]`。|
|离线打包对话（HF chat 模板）|`packed_chat_mix`|使用模型自带的 HF chat 模板渲染样本，而非 `TEMPLATES`；需要设置 `model.use_hf_chat_template: true` 或 `model.chat_template_path`。|

## 2. 输入要求（`data.wds_dir`）
实现**直接读取** `data.wds_dir` 下未压缩的 `*.tar` 分片，
不会将源分片解包成扁平目录。

注意事项：

* `wds_pack.cli.scan_manifest` 从 `data.template_text_key` 指定的字段读取消息列表；它也接受常用键 `messages` 和 `texts`。
* 如果 JSON 文件来自 `tools/vlm_data_preprocess/convert_to_webdataset.py`（多场景默认写入 `texts`），通常需要将 `data.template_text_key` 设置为 `texts`。
* `packed_multi_mix_qa`：JSON 必须声明 `media`/`media_type`（`text`、`image` 或 `video`）。图像/视频样本应提供 `name`/`media_files`；如缺失，则按扩展名从 WDS 成员推断媒体文件。
* V1 不支持 `.tgz` 输入，因为高效的字节区间读取需要未压缩的 tar。

## 3. 快速开始
```bash
cd tools/vlm_data_preprocess/offline_packing

# 1) 编辑 config.yaml（或复制 packed_vqa_demo.yaml）
# 2) 运行 4 步流水线（默认读取 config.yaml）
bash scripts/pack_wds.sh
```

切换到其他配置：

* 方式 1：将其覆盖/复制到 `config.yaml`
* 方式 2：使用 `--config your.yaml` 手动运行每个步骤（见下一节）

## 4. 流水线详情（对应 `pack_wds.sh`）

### 步骤 1：扫描 WDS manifest 并计算每个样本的 Token 长度（`wds_pack.cli.scan_manifest`）
* 输入：`data.wds_dir` 下的 `*.tar` 分片
* 处理：直接从 tar 读取 WDS 样本，根据 `sample.sample_type` + `model.model_type` 选择模板（`wds_pack.core.constants.TEMPLATES`），使用 `AutoProcessor` 或 `AutoTokenizer` 对文本+视觉输入进行分词，并记录 tar 字节定位信息
* 输出：`{data.work_dir}/sample_manifest.sqlite`（权威 manifest）、`token_len/` 下各媒体的 Token 报告，以及 `skipped_overlong.jsonl`（`token_len > max_token_len` 被跳过的样本，始终写出）；`sample_manifest.jsonl`、合并报告 `sample_len_report.txt` 和 `skipped_samples.jsonl` 仅在 `artifacts.debug_artifacts: true` 时保留

手动运行：
```bash
python -m wds_pack.cli.scan_manifest --config config.yaml
```

### 步骤 2：按媒体类型进行长度分桶与打包分组（`wds_pack.cli.pack_bins`）
* 输入：`sample_manifest.sqlite`（各媒体的 Token 长度）
* 处理：在 `sample.max_token_len` 约束下，为 text/image/video 分别将样本打包入"箱子"。生产算法为 `packing.algorithm: best_fit_decreasing`（BFD）；`hashbucket` 仍可用于旧的精确填充行为
* 输出：`{data.work_dir}/bins/bins_plan_{text,image,video}.jsonl`（中间文件 `bins/bins_boxs_{text,image,video}.pkl` 仅在 `artifacts.keep_intermediate: true` 时保留）

手动运行：
```bash
python -m wds_pack.cli.pack_bins --config config.yaml
```

### 步骤 3：生成打包计划（`wds_pack.cli.build_plan`）
* 输入：各媒体的 `bins/bins_plan_*.jsonl` + `sample_manifest.sqlite`
* 处理：将各媒体的 bins 转换为稳定的打包样本计划
* 输出：`{data.work_dir}/pack_plan.jsonl`（诊断文件 `unpacked_samples.jsonl` 仅在 `artifacts.debug_artifacts: true` 时保留）

手动运行：
```bash
python -m wds_pack.cli.build_plan --config config.yaml
```

### 步骤 4：将打包样本写回 WebDataset（`wds_pack.cli.write_wds`）
* 输入：`pack_plan.jsonl` + `sample_manifest.sqlite`；媒体字节从源 tar 字节偏移读取
* 输出：`data.packed_wds_dir/pretrain-*.tar` 加 Energon 元数据（`.nv-meta/dataset.yaml` + tar 索引）

手动运行：
```bash
python -m wds_pack.cli.write_wds --config config.yaml
```

## 5. 配置（`config.yaml`）
关键字段：

* `data.input_format` — WDS-native 打包设为 `wds`
* `data.wds_dir` — 输入 WebDataset 目录，包含未压缩的 `*.tar` 分片
* `data.template_text_key` — JSON 中的消息字段名（`messages` 或 `texts`）
* `data.work_dir` — manifest、token 报告、bins 与 pack plan 的工作目录
* `data.packed_wds_dir` — 最终打包 WDS 输出目录
* `sample.max_token_len` — 目标打包长度（如 8192 / 16384）
* `sample.sample_type` — V1 支持 `packed_multi_mix_qa` 或 `packed_chat_mix`
* `model.model_type` — 用于选择模板的模型标识符
* `model.processor_loader` — VLM 处理器用 `auto_processor`，纯文本冒烟测试用 `auto_tokenizer`
* `model.processor_kwargs.*` — 传递给 `transformers.AutoProcessor.from_pretrained` 的 HF 处理器参数
* `packed_wds.maxcount` / `maxsize` — tar 分片拆分策略

示例（摘录，完整字段见 `config.yaml`）：

```yaml
data:
  input_format: "wds"
  wds_dir: "/workspace/.../wds/"
  template_text_key: "texts"
  work_dir: "/workspace/.../packing_work/"
  packed_wds_dir: "/workspace/.../packed_wds/"

sample:
  max_token_len: 8192
  sample_type: packed_multi_mix_qa
```

## 6. 切换模型 / 调整图像处理
步骤 1 的 Token 计数取决于实际的 `AutoProcessor` 逻辑，因此可以通过配置更换模型或图像预处理参数：

* 更换模型：将 `model.processor_kwargs.pretrained_model_name_or_path` 设置为所需的 HF 模型/处理器；相应更新 `model.model_type`。
* 调整图像 Token 预算 / 分辨率：在 `model.processor_kwargs` 下添加处理器支持的参数（例如 Qwen-VL 的 `min_pixels`/`max_pixels`）。
* 模板对齐：如果添加了新的 `model.model_type`，确保 `tools/vlm_data_preprocess/offline_packing/wds_pack/core/constants.py` 中的 `TEMPLATES[sample_type][model_type]` 包含对应条目；否则步骤 1 将报错"No template for sample_type=..., model_type=..."。
* 媒体预处理：在 `media_preprocess` 下可以为每种模态指定预处理函数名（实现在 `tools/vlm_data_preprocess/offline_packing/wds_pack/media/preprocess.py`），以控制缩放/裁剪/帧读取行为。
