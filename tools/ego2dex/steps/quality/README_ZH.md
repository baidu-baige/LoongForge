# Quality 数据质检

`quality` 对 `retarget` 的结果执行三级质检，并输出 episode 保留标记和逐帧有效掩码。该步骤只生成质检产物，不修改源视频和 IK 文件。

> 当前 `quality` 未接入 `run-all`，`lerobot_writer` 也不会自动读取质检结果。需要在 `retarget` 后单独运行，并由下游根据 `manifest.parquet` 和 `*_frame_mask.npz` 应用筛选。

## 输入

`--retarget_dir` 中每个 episode 必须同时存在：

- `<episode>_ik.npz`：状态、双臂 IK 状态和位置误差。
- `<episode>_quality.npz`：手部检测、渲染、机器人 mask 和碰撞信息。

自动执行 L3 时还需要 `<episode>_robot_on_bg.mp4`；仅运行 L1/L2 或导入已有 L3 结果时不会读取视频。

可选的 `--state_dir` 指向 `align` 输出，用于读取任务文本；缺失时任务描述使用 `manipulation`。

## 判定流程

### L1：逐帧基础质量

一帧必须同时满足以下条件：

- 检测到手部。
- 左右臂 IK 位置误差均为有限值且小于 `0.05 m`；完整姿态 IK
  的旋转误差不作为 L1 门禁。
- 机器人成功渲染，像素数大于 0。
- 无自碰撞，双臂接触数不超过 1。
- 机器人 mask 占比为有限值且不超过 `0.70`。

被无效区间夹住、长度小于 `--min_valid_run_frames` 的短有效片段也会被剔除，默认阈值为 9 帧。质量字段缺失时按不通过处理。

### L2：数据集级动作质量

程序按 `robot_type + action 维度` 分组，从 IK 状态差分得到 action，并基于 L1 有效帧检测：

- 动作离群：超出 `Q1/Q99` 扩展区间，默认扩展倍数为 `3.0`。
- 突变：动作残差、加速度或 jerk 超出组内绝对值分位数，默认分位数为 `0.999`。

L2 帧有效条件为 `L1 有效 && 非离群 && 非突变`。episode 的无效帧比例不超过 `--max_invalid_ratio`（默认 `0.60`）且至少有一帧有效，才通过 L2。

### L3：视频与任务一致性

L3 以 4 FPS 从完整视频时间线上均匀采样，默认最多 32 帧，判断机器人操作是否与任务文本一致。返回字段为：

```json
{
  "is_consistent": true,
  "confidence": 0.92,
  "reasoning": "..."
}
```

自动模式通过 OpenAI 兼容接口调用 SGLang，并缓存已完成结果；重跑同一命令可继续失败或中断的任务。

## 运行方式

### 完整质检（默认）

先启动 SGLang 服务，并确保 `/v1/models` 可访问：

```bash
export EGO2ROBOT_VLM_MODEL=<SGLang 中的模型 ID>
python cli.py quality \
  --retarget_dir out/06_retarget \
  --state_dir out/02_align \
  --output_dir out/quality_curation
```

也可用 `--vlm_model` 指定模型，或用 `--server_url` 修改服务地址。自动模式要求每个 episode 都得到 L3 结果，否则命令报错；重跑时默认复用 `vlm_results.json` 缓存，`--overwrite_vlm` 可强制重新审核。

### 仅运行 L1/L2

```bash
python cli.py quality \
  --retarget_dir out/06_retarget \
  --state_dir out/02_align \
  --output_dir out/quality_curation \
  --skip_vlm
```

此时 L3 状态为 `pending`，最终 `keep_episode` 只取决于 L2。若同时添加 `--require_vlm`，缺少 L3 结果的 episode 会被拒绝。

### 使用已有 L3 结果

```bash
python cli.py quality \
  --retarget_dir out/06_retarget \
  --state_dir out/02_align \
  --output_dir out/quality_curation \
  --vlm_results /path/to/vlm_results.json \
  --require_vlm
```

`--vlm_results` 必须是以 episode 名称为 key 的 JSON 对象。未加 `--require_vlm` 时，没有 L3 结果的 episode 按 L2 结果决定；已有但判定不一致的结果仍会拒绝该 episode。

本地 Qwen3.5 checkpoint 也可通过 Transformers 单独生成结果，再按上述方式导入：

```bash
python -m steps.quality.transformers_runner \
  --model_dir /path/to/Qwen3.5 \
  --requests out/quality_curation/vlm_requests.jsonl \
  --output out/quality_curation/vlm_results.json
```

## 输出

| 文件 | 说明 |
|---|---|
| `manifest.parquet` | 每个 episode 的 L1/L2 统计、L3 结论、`keep_episode` 和 `drop_reason` |
| `<episode>_frame_mask.npz` | `l1_valid`、离群/突变标记及最终逐帧 `valid` 掩码 |
| `<episode>_vlm.json` | 单 episode 的 L3 结果；未审核时为 `pending` |
| `vlm_requests.jsonl` | 可交给 SGLang 或 Transformers runner 的审核请求 |
| `vlm_results.json` | 自动 L3 的缓存结果（仅自动模式） |
| `vlm_results.errors.json` | 自动 L3 尚未解决的错误（仅自动模式） |
| `summary.json` | episode、保留/丢弃、待审核和有效帧数量汇总 |

最终 episode 保留条件为：

```text
L2 通过 &&（L3 不作为门禁，或 L3 判定一致）
```

常用调参可通过 `python cli.py quality --help` 查看。建议先保持默认阈值，并结合 `drop_reason`、逐帧掩码和原视频抽查后再调整。
