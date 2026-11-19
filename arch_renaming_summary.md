# arch 对象重命名及理由汇总

## 版本与范围

本文对应 [PR #345](https://github.com/baidu-baige/LoongForge/pull/345) 的提交 `8dc46b9bafff9388606c9482bbf046ce409e1edc`，以该提交相对父提交的实际差异为依据。

后续调整：按要求回滚了损失配置相关重命名，第 6 节记录当前恢复后的名称；其他重命名仍以该提交为准。

本次目标是通过名称与注释提高可读性，保留模型计算、张量操作、执行顺序、默认值和原有校验。改动包含公开字段、函数、关键字参数和配置键的重命名，因此调用方需要同步名称；计算逻辑保持不变不代表接口向后兼容。

本提交涉及 32 个文件，包括 arch 源码、配套 YAML 配置和单元测试。没有重命名类、源码文件或示例脚本。后续同步主分支时引入的 `vpp2 → arch_vpp2` 入口调整不属于本提交。

## 1. 模型声明和注意力分组

| 对象位置 | 原名称 | 新名称 | 重命名理由 |
| --- | --- | --- | --- |
| `WorldModelSpec` 字段 | `groups` | `attention_groups` | 组内的 stream 共享一次联合注意力计算。明确分组用途，避免与优化器参数组或其他集合混淆，也便于理解多组模型拓扑。 |
| `AttentionGroup` 字段 | `members` | `streams` | 成员实际是 `StreamSpec`。直接表达集合内容，遍历和校验时不必额外推断“成员”是什么。 |
| spec 构建函数 | `build_world_model` | `build_world_model_spec` | 函数返回经过验证的 `WorldModelSpec`，并不实例化可执行模型。后缀明确返回对象及配置解析与模型组装的边界。 |

主要位置：

- `loongforge/embodied/model/arch/spec/world.py`
- `loongforge/embodied/model/arch/spec/__init__.py`
- `loongforge/embodied/model/arch/streams/engine.py`
- `loongforge/embodied/model/arch/builder.py`

组内 stream 顺序、分组结构、层映射和校验规则保持原样。`WorldModelSpec.streams` 仍表示按分组顺序展开后的全部 stream；`AttentionGroup.streams` 仅表示该组的 stream。

## 2. 训练时间步采样和推理更新

| 对象位置 | 原名称 | 新名称 | 重命名理由 |
| --- | --- | --- | --- |
| `DiffusionSpec` 字段 | `gate` | `sampling_gate` | 表达推理采样过程中“哪个 stream 在当前步骤推进”的控制策略。增加限定词，避免与网络内部的 modulation gate 混淆。 |
| `DiffusionSpec` 字段 | `sampler` | `solver` | 对象负责每一步的数值更新，例如 Euler 更新。与抽取 timestep 的 `TimestepSampler` 区分，同时避免将数值求解器的用途限定为推理。 |
| `WorldModel.sample_timesteps()`、`add_noise()` 局部变量 | `sampler` | `timestep_sampler` | 明确变量抽样的对象是训练时间步，便于理解其 `broadcast` 和 `sample()` 调用。 |

主要位置为 `diffusion/spec.py`、`builder.py` 及三个 provider 的 `spec.py`。

`sampling_gate` 与 `solver` 必须同时提供或同时省略的校验保持不变。本提交没有实现新的推理采样循环；`WorldModel.sample()` 原有的未实现边界仍然保留。

## 3. 时间步标量与时间步嵌入

| 对象位置 | 原名称 | 新名称 | 重命名理由 |
| --- | --- | --- | --- |
| `HeadSpec` 字段 | `needs_timestep` | `needs_timestep_embedding` | head modulation 消费的是 `[B, residual_width]` 嵌入，不是 `[B]` 时间步标量。名称直接表达所需表示形式。 |
| `OutputHead.forward()` 协议及具体 head 的关键字参数 | `timestep` | `timestep_embedding` | 在调用点即可辨认传入张量的含义，减少把标量时间步误传给 head 的可能。 |
| `_HeadBase` 校验方法 | `_check_timestep` | `_check_timestep_embedding` | 方法检查的是嵌入是否提供及其形状，名称与实际契约保持一致。 |
| `StreamState.extras` 数据键 | `"t_embed"` | `"timestep_embedding"` | 使用完整名称，使 stream 生产、builder 读取和 head 消费之间的数据流更容易追踪。 |
| `WanDiTStream.prepare()` 局部变量 | `t_embed` | `timestep_embedding` | 与 `extras` 数据键及 head 参数统一，避免同一张量在相邻阶段使用不同缩写。 |
| `WorldModel._run_heads()`、`PatchFlowHead.forward()` 及相关测试中的局部变量 | `embed` | `timestep_embedding` | 补全 embedding 的具体对象，明确它不是文本、图像或其他条件嵌入。 |

主要位置：

- `heads/spec.py`
- `heads/head.py`
- `heads/flow.py`
- `streams/blocks/wan_streams.py`
- `builder.py`

stream 的 `needs_timestep` 字段及 `prepare(..., timestep=...)` 参数仍使用原名称，因为它们接收时间步标量。head 的名称变化不应机械扩展到 stream 接口。

注册子模块名 `time_embedding`、`time_projection` 及参数键保持原样；本次局部变量和数据键重命名不要求改写这些 checkpoint 键。

## 4. Token 标签

| 数据位置 | 原名称 | 新名称 | 重命名理由 |
| --- | --- | --- | --- |
| `TokenLayout.labels` 的 stream 标签键 | `"stream"` | `"stream_id"` | 标签存储的是数值型 stream 编号，而非 stream 对象或名称字符串。 |
| packed sample 标签键 | `"sample"` | `"sample_id"` | 标签表示每个 token 所属的 packed sample，区别于 tensor 的 batch 维度和 sample 数据对象。 |
| `GeometrySpec.build_labels()` 和联合 layout 的 padding 标签键 | `"pad"` | `"is_padding"` | 使用布尔谓词形式，明确该位置是否仅为静态长度对齐而补齐。 |

同步范围包括几何标签生产、联合 layout 构建、`query_in_stream()`、`same_stream()`、`stream_edge()`、`same_sample()` 及对应测试夹具。

标签值、mask 表达式和缺失标签时的回退规则保持不变。这些变化主要针对标签字典的键；并非将所有同名局部变量或对象字段一并改名。

## 5. 后端能力协商

| 对象位置 | 原名称 | 新名称 | 重命名理由 |
| --- | --- | --- | --- |
| `CompiledMask` 静态方法 | `supports` | `negotiate_capability` | 返回值是 `EXACT`、`DEGRADED` 或 `UNSUPPORTED` 能力等级，而非布尔值。新名称提示调用方需要处理具体等级，并表达其结合可见性与运行场景判断能力的用途。 |

位置为 `visibility/mask.py`，相关说明及测试调用同步更新。判断逻辑与分类结果保持原样；改名本身不表示已将完整协商流程接入所有模型构造路径。

## 6. 已回滚：损失配置与配置覆盖辅助函数

| 对象位置 | PR #345 中的改名 | 当前恢复名称 |
| --- | --- | --- |
| `WorldModelConfig` 字段及 YAML 配置节 | `losses` / `losses:` | `heads` / `heads:` |
| 配置覆盖常量 | `_LOSS_KNOBS` | `_HEAD_KNOBS` |
| 配置覆盖辅助函数 | `_apply_loss_overrides` | `_apply_heads` |
| 上述辅助函数参数 | `loss_overrides` | `heads` |

原改名是为了强调这些配置实际覆盖 `LossSpec`。现按要求撤回这组名称调整，保留原有 `heads` 配置接口。

主要位置为 `model_configuration_arch.py`、`spec/world.py` 及 VPP2、InternW0-Delta 的 arch YAML。

配置节恢复为 `heads`，其条目仍按**输出 head 名称**匹配消费该 head 的 loss。例如：

```yaml
model:
  heads:
    video_flow:
      weight: 1.0
    action_flow:
      weight: 1.0
```

`WorldModelSpec.heads`、`WorldModel.heads` 和 `HeadSpec` 保留原名称，因为它们确实描述或保存输出 head。损失配置回滚后，进一步恢复了此前遗漏的 `HeadSpec.weight` 字段及负值校验，与重命名前的接口一致；有效监督权重继续从 `LossSpec` 读取，运行时损失计算未修改。

## 7. VPP2 视频前缀参数

| 对象位置 | 原名称 | 新名称 | 重命名理由 |
| --- | --- | --- | --- |
| `Vpp2Config` 字段、provider 参数、可见性辅助函数参数和 YAML 键 | `visible_video_frames` | `video_prefix_frames` | 表达配置意图中的“前缀帧数量”，避免泛化为任意可见帧集合。 |

同步位置包括 `model_configuration_arch.py`、`spec/providers/vpp2/spec.py`、`configs/models/embodied/arch/vpp2.yaml`。

需要区分名称意图和实现验证：本提交仅改名，原有 `noise_gated_history(window=...)` 调用及可见性表达式未修改。不能据此认定当前实现已具备严格前缀语义，也不能据此认定其与独立 VPP2 数值等价；这些需要单独验证。

## 8. 保留的名称与本次未包含的调整

以下名称在本提交中没有变化：

| 保留名称 | 说明 |
| --- | --- |
| `WorldModel`、`ArchWorldModel`、`WorldModelSpec`、`AttentionGroup` 等类名 | 本次主要调整字段和接口，没有重命名类。 |
| `model_id` | 配置和已有消费者继续使用该名称，没有保留不完整的 `encoder_model_id` 改名。 |
| `ArchWorldModel.from_pretrained()` | 保留现有构造入口，没有改为 `from_config()`。 |
| `DiffusionSpec.sampler_for()` | 保留方法名；其调用方仅把局部变量改成 `timestep_sampler`。 |
| provider 的 `action_width` 参数 | 仍与 `GeometrySpec.action_dim` 区分，未机械统一参数名。 |
| `HeadSpec.weight` | 恢复原有默认值 `1.0` 及负值校验；运行时继续拒绝非默认 head 权重，监督权重由 `LossSpec` 管理。 |
| `vpp2_to_arch` | 表示 legacy VPP2 checkpoint 的转换来源，保持原名称。 |
| `test_arch_registry.py` 等测试文件名 | 文件名保留，相关测试仅同步新的字段或标签名称。 |
| `vpp2` 模型/provider 入口 | PR #345 时仍叫 `vpp2`，后续 `arch_vpp2` 入口隔离不在本次汇总范围内。 |

## 9. 注释优化和验证边界

注释调整主要解释 spec 构造与运行时组装的职责、标量时间步与嵌入的区别、能力协商和推理尚未接入的部分，以及 InternW0-Delta 现有表示限制。它们没有实现新的模型能力。

损失配置名称回滚后的首次定向测试发现 `HeadSpec.weight` 定义遗漏。恢复重命名前已有的字段和负值校验后，在 `zhangqiu_pro6000` 中重新运行完整 arch 单元测试：

```bash
pytest -q tests/embodied/unit/test_arch*.py
```

当前结果为 **363 passed、4 skipped、0 failed**。4 个跳过项来自仅适用于 InternW0-Delta 的中间层 head 和 track_align 场景，其他模型按既有测试判断跳过。本次没有修改测试或增加跳过项。

`heads` 的 weight、warmup、enabled 配置覆盖，以及默认 head 权重和负值拒绝直接验证通过，`git diff --check` 通过。这些检查不能替代真实 VPP2 的同权重前向、梯度及端到端训练等价性验证。
