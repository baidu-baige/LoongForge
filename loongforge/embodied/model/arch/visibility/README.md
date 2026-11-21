# Visibility 与 attention mask

本组件以 `VisibilityTopology` 声明 token 可见性，由共享编译器生成
`AttentionPlan`，供 dense、SDPA、FlexAttention 使用。模型侧声明 `intra + reads`，
不自行拼接 bool attention 矩阵。底层 mask 统一使用 `True = 允许 attention`；
矩阵的行是 query，列是被读取的 source key。

## 主接口：reader-centric topology

```python
from loongforge.embodied.model.arch.visibility import VisibilityTopology

# 假设 layout 恰好包含 video、action、text 三个 streams。
topology = VisibilityTopology({
    "video": {"intra": "bidirectional", "reads": {}},
    "action": {
        "intra": "bidirectional",
        "reads": {"video": {"mode": "prefix_frames", "frames": 2}},
    },
    "text": {"intra": "bidirectional", "reads": {}},
})
template = topology.compile(layout)
batch_plan = template.bind(key_valid=token_valid)  # bool [B,L]
dense = batch_plan.materialize("dense")           # bool [B,1,L,L]
bias = batch_plan.materialize("sdpa")             # float32：允许为 0，否则 -inf
```

每个 leaf reader 必须显式声明 `intra`，控制自身 scope 内的 attention；
`reads` 将 source 名称映射到读取策略，省略等价于空映射。上例中 video 只读自身，
action 读自身和 video 的前两个 source frames，text 只读自身；
没有 video→action 或任何 text 跨流边。

非空 topology 是 **closed-world**：

- 必须精确声明 `layout.stream_names` 中的所有 stream；缺失、多余或未知 read
  source 都会报错，不能通过遗漏 stream 获得默认可见性。
- 未声明的 read edge 默认不可见。`reads` 不是对默认全可见矩阵追加限制。
- `VisibilityTopology({})` 是明确的“不限制 attention”，不是“所有边关闭”；
  `as_visibility()` 返回 `None`。
- 显式声明所有 stream 的 `intra` 和所有跨流 reads 为无 gate 的
  `bidirectional`，也可编译为 maskless。空 topology 或这种全双向 topology
  在没有其他通道约束时，对 dense、SDPA、Flex 的 `materialize()` 返回 `None`。
  声明中存在 runtime gate，或后续绑定 `key_valid` 时，仍可能需要 mask。

声明接受字符串、mapping 或已构造的 `ReaderSpec` / `ReadPolicy`；
主入口是 `topology.compile(layout)`，`as_visibility()` 用于兼容表达式接口。

## Policy modes 与静态坐标

| mode | 参数 | 在声明的 query/source rectangle 内的语义 |
| --- | --- | --- |
| `bidirectional` | 无 | 全部可见 |
| `prefix_frames` | 必需 `frames` | 仅读取 source scope 内按 frame 值排序的前 N 个不同 frame |
| `frame_causal` | 可选 `window` | `q.frame >= k.frame`；带窗口时还要求 `q.frame - k.frame < window` |
| `first_frame_causal` | 无 | source 首 frame 的 query 只读首 frame，其余 query 保持双向 |
| `chunk_causal` | 可选 `window` | `q.chunk >= k.chunk`；带窗口时还要求 `q.chunk - k.chunk < window` |
| `group_causal` | 必需 `group_size` | `floor(q.frame / group_size) >= floor(k.frame / group_size)` |
| `group_diagonal` | 必需 `group_size` | 仅相同 frame group 可见 |

`frames`、`window`、`group_size` 必须是正整数，不接受 bool、零、负数或不适用
的参数组合。除 `bidirectional` 外，策略需要静态 `frame` 或 `chunk` 标签；
缺标签、参与 scope 内的负坐标会显式报错。`prefix_frames` 只检查 source 的
frame 坐标，不要求 reader 拥有有效 frame。

**`prefix_frames` 是 source-relative，不是 joint sequence 的 token 前缀。**
例如 source 位于 action tokens 之后，frame 值为 `[17, 10, 42, 10, 42]`：
`frames=1` 选择 frame 10，`frames=2` 选择 frame 10 和 17，数量超过实际 frame
数时选择整个 source。source 可以是 stream，也可以是命名 region。

`first_frame_causal` 不等于普通 `frame_causal`：用于 video intra 时，只限制
首 frame queries，后续 frames 可读取所有 video frames。VPP2 provider 的
`three_frame`、`per_frame_causal` 输入别名分别映射到 `bidirectional`、
`frame_causal`，不是额外的通用 policy modes。

## Regions：token membership 与 query partition

`layout.with_regions(**memberships)` 返回新 layout；每个 membership 必须是
联合 token 顺序上的 **bool `[L]`**，不带 batch 维。regions 不改变 streams、
segments 或静态 labels，也不代表新增参数流或 expert。名称不得与 stream
冲突，不允许覆盖已有 region。

```python
# memberships 由 layout/recipe adapter 提供，例如 history_members: bool [L]。
layout = layout.with_regions(
    current=current_members,
    future=future_members,
    history=history_members,
)
topology = VisibilityTopology({
    "video": {"regions": {
        "current": {
            "intra": "bidirectional",
            "reads": {"history": "bidirectional"},
        },
        "future": {"intra": "bidirectional", "reads": {}},
    }},
    "action": {"intra": "bidirectional", "reads": {}},
})
template = topology.compile(layout)
```

这里假设 layout 只有 video/action；current 和 future 必须 partition video，
history 是 read-only source，不能与 current 重叠。region container 只能声明
`regions`，不能同时带 `intra` 或 `reads`，也不能嵌套 regions；
各 leaf 分别声明完整策略。

验证分为两层：

- layout 检查 membership 的名称、dtype、形状；read-only memberships 可以重叠。
- topology 编译器检查 query regions 必须存在、不能越出 owner stream，
  且无重叠、无遗漏地 partition 所属 stream。
- 对同一非空 query scope，`intra` 和每条 read 的 key scopes 必须互不重叠。
  同时读取重叠 regions、读取 stream 及其子region，或 reads 再次读取自身
  都可能触发 `Overlapping read rules`；不同 query scopes 读取重叠 source
  不构成冲突。检查依据 membership rectangles，而非 gate 的运行时结果。

全 False 的空 memberships 合法并保留。`restrict_to_streams()` 裁剪 layout
时仍保留所有 region 名称，包括裁剪后为空的 memberships；结构和 pairwise
同时裁剪 Q/K 两轴，`key_valid` 裁剪 key 轴，并保持原 concat 顺序和静态 labels。
涉及实际裁剪时，runtime gates 必须先 bind。

### InternW0 six-region 表达示例

下面展示 A/R/C/F/D/X 六个 regions 的可见性表达能力，**不是已接入的真实
InternW0-Δ provider layout**。A/R/C/F/D partition video，X partition action；
memberships 由调用者提供，仍只有 video/action 两个 streams。

```python
layout = layout.with_regions(**memberships)  # A/R/C/F/D/X：各自 bool [L]
sources = {
    "A": (),
    "R": ("A",),
    "C": ("A", "R"),
    "F": ("A", "R", "C"),
    "D": ("R", "C"),
    "X": ("A", "R", "C", "D"),
}
readers = {
    name: {
        "intra": "bidirectional",
        "reads": {source: "bidirectional" for source in keys},
    }
    for name, keys in sources.items()
}
topology = VisibilityTopology({
    "video": {"regions": {name: readers[name] for name in "ARCFD"}},
    "action": {"regions": {"X": readers["X"]}},
})
template = topology.compile(layout)
```

真实 delta tokens、delta-local reads 与 action-to-delta reads 尚未接入 provider。
当前 provider 保留近似两流策略：video frame-causal、action 双向、
action 读取固定 video frame prefix，无反向边；prefix 不意味着这些 frames
已被固定为 clean。six-region 示例不能作为完整 delta recipe 等价性的证据。

## ReadGate：显式 runtime labels

`ReadGate(label, side, op, threshold)` 对一条声明的 intra/read rectangle 应用
运行时比较：

- `label`：非空 label 名称，语义由训练 recipe 定义。
- `side="query" | "key"`：比较 query 或 source key 的坐标。
- `op="ge" | "gt" | "le" | "lt" | "eq"`：分别表示 ≥、>、≤、<、=。
- `threshold`：数值阈值，与 label 使用相同单位。

```python
from loongforge.embodied.model.arch.visibility import ReadGate, ReadPolicy

topology = VisibilityTopology({
    "video": {"intra": "bidirectional"},
    "action": {
        "intra": "bidirectional",
        "reads": {"video": ReadPolicy(
            "prefix_frames",
            frames=2,
            gate=ReadGate(label="level", side="query", op="ge", threshold=1),
        )},
    },
})
template = topology.compile(layout)  # 此例 layout 只有 video/action
batch_plan = template.bind(
    labels={"level": token_levels},  # 显式 [B,L]，按联合 token 顺序
    key_valid=token_valid,           # bool [B,L]
)
```

mapping 形式也可在 policy 中使用
`"gate": {"label": "level", "side": "query", "op": "ge", "threshold": 1}`。
上例仅允许 level ≥ 1 的 action queries 读取 video prefix，不影响 video
自身 attention 或 action intra。改成 `side="key"` 则按 video key 的 level
限制该边。将 gate 塞进全局 `key_valid` 会限制其他 reader，不能等价替代。

runtime labels 必须显式提供 `[B,L]`，不能用 batch 共享的静态 `[L]` 替代；
离散 bucket、连续 sigma/timestep 或 clean/noisy 标记的意义和单位由 recipe
决定。含 gate 的 template 保留 ungated structural，延迟 gate 到 `bind()`，
随后生成 pairwise；缺必要 label 的 bind、未绑定的 `materialize()` /
`dense_bool()` 都会报错。每次 bind 返回新 plan，不修改可跨 batch 复用的模板。

## 数据契约

| 对象/字段 | 形状 | 含义 |
| --- | --- | --- |
| `TokenLayout.labels` | 每个标签 `[L]` | batch 共享的 stream、frame、chunk 等静态信息 |
| `TokenLayout.regions` | 每个 membership bool `[L]` | 命名 token scopes |
| `AttentionPlan.structural` | bool `[L,L]` 或 `None` | 静态结构；始终不带 batch 维 |
| `AttentionPlan.key_valid` | bool `[B,L]` 或 `None` | 全局 key 有效性；例如 token padding |
| `AttentionPlan.pairwise` | bool `[B,L,L]` 或 `None` | 已绑定的运行时边限制 |
| `AttentionPlan.visibility` | 待绑定表达式或 `None` | 缺少 runtime label 时延迟求值 |

最终 mask 是广播后的 `structural & pairwise & key_valid`。不存在的通道表示不加限制。
`key_valid` 只屏蔽 key 列，不隐式屏蔽 query、不负责 residual 清零或 loss reduction；
训练侧仍需定义 query/output/loss validity。静态 `pad` label 本身也不自动成为
key mask。`plan.to(device)` 同步移动三个 mask 通道，不改变静态 layout。

## 与 WorldModel 的连接

WorldModel 在构造时通过 topology 编译 template。默认 `add_noise()` 将 batch 的
`attention_labels` 和 `attention_key_valid` 传入 `DiffusionRuntime`；
`run_group()` 对需要 runtime labels 的 plan 绑定显式映射，并绑定 key validity。
自定义训练 recipe 覆盖这些扩展点时也需要负责透传。

**builder 已移除自动 timestep → noise label 的推测**：不会把 stream timestep
展开成默认 `noise_level`，也不会给非 diffusion stream 猜测零噪声坐标。
diffusion timestep 不隐含 key noise label、clean-cache 决策或 KV routing。
batch adapter 负责把 labels/validity 对齐到联合 latent/patch token 顺序；
本组件不负责 RGB frame padding 到 VAE latent/patch 的转换。

## 后端和边界

- dense 返回 bool `[B或1,1,L,L]`；SDPA 返回同形状 additive float32 bias，
  允许为 0、禁止为 `-inf`。`dense_bool()` 即使 maskless 也会显式构造全 True
  mask；需要保留 maskless 路径时使用 `materialize()`。
- flex 返回 BlockMask，支持结构 mask、padding-only mask、动态边 mask；
  有约束时必须提供与协商配置一致的 `block_size`。
- flex 的 `mask_mod` 对 rounded-up tail block 做范围检查和安全索引。
- 没有任何约束时 `materialize()` 返回 `None`，保留 maskless 路径；
  backend 名称仍会校验。
- `CompiledMask.as_segments()` 仅允许严格按 segment block diagonal 的静态
  可见性；它不意味着 engine 已支持 varlen kernel。`AttentionPlan.materialize`
  显式拒绝 varlen。
- backend capability 只描述 mask 表达能力，不证明完整训练或 cache routing
  等价。`LayerAttentionOverride` 是接口声明，不代表已有内部 KV truncation
  consumer。

当前静态 mask 空间为 `O(L²)`，动态 pairwise 显式展开 `[B,L,L]`，空间为
`O(BL²)`；SDPA bias 还有浮点存储开销。Flex 后端并不消除当前编译器的 dense
lookup/pairwise 存储。这里优先保证正确性，尚未提供不展开动态边的稀疏编译，
也不声称动态编译已适配 CUDA Graph capture 或可变长度 KV cache。

## Legacy 兼容

底层 `Visibility` 和 rules 仍可通过
`AttentionPlan.from_visibility(visibility, layout)` 使用；表达式支持 `&`、`|`、
`~`、`restrict(where=...)`，runtime dependency 随组合传播。
`stream_edge(src, dst)` 表示 dst queries 读取 src keys；`restrict` 在 scope
外保持可见，不等于直接 `& scope`。旧 stream mapping 的未声明 query rows
保持可见，不是新 topology 的 closed-world 语义。

迁移时不要混淆以下接口：

- legacy `prefix_only(n)` 是 joint sequence 前 n 个 **tokens**，不等于
  source-relative `prefix_frames`。
- `frame_causal() | prefix_only(n)` 的旧组合不等于新声明式
  `first_frame_causal` mode 的“仅首 frame queries 受限”语义。
- legacy `noise_gate(threshold)` 使用 key-side `noise_level <= threshold`，
  仍需显式 runtime labels。旧规则直接 `evaluate(layout)` 的缺标签放开行为
  不应当作为训练兜底；生产 attention 使用 `AttentionPlan`，静态
  `Visibility.compile()` 会拒绝缺失 runtime dependency。

旧 preset 注册表已删除；复用声明直接构造 `VisibilityTopology`，不再通过字符串
查找规则组合。保留的 Python 文件各自有独立职责：

- `declarative.py`：模型侧 `intra + reads`、regions 与 gate 声明及编译。
- `combine.py`：统一表达式对象、runtime 标签求值与组合基础；声明式编译仍依赖它。
- `rules.py`：自定义表达式原语，包括尚未被声明式 policy 替代的 runtime history、
  packed sample 隔离及 block coupling，不只是重复的模型 preset。
- `mask.py`：mask 数据契约、batch 绑定、dense/SDPA/Flex 后端编译。
- `__init__.py`：包公共入口。

## 验证

相关单测（路径相对仓库根目录）：

```text
tests/embodied/unit/test_arch_visibility_declarative.py
tests/embodied/unit/test_arch_visibility_runtime.py
tests/embodied/unit/test_arch_visibility_vpp2_parity.py
tests/embodied/unit/test_arch_visibility_training_contract.py
tests/embodied/unit/test_arch_mask.py
tests/embodied/unit/test_arch_rules.py
tests/embodied/unit/test_arch_combine.py
tests/embodied/unit/test_arch_engine.py
tests/embodied/unit/test_arch_world_model.py
```

- declarative：closed-world、maskless、policy 参数和坐标验证、source-relative
  prefix、region partition/read overlap、six-region 表达、gate 边作用范围、
  template/bind 与裁剪后空 membership 保留。
- runtime：逐样本 gate 与组合代数、padding、stream 裁剪、dtype/shape、
  dense/SDPA/Flex predicate 一致性、tail 安全索引和全屏蔽 key 行的数值行为；
  CUDA 可用时另比较 FlexAttention 输出与梯度。
- vpp2_parity：通过 AST 提取并执行上游 mask 方法，以 SHA256 固定参考源码；
  对 native modes、联合 action/video prefix、source 顺序与 frame offset、
  partial-frame 边界、独立 key validity 和 Flex tail predicate 做对照，
  相应数值 parity 用例另比较 SDPA 输出及 Q/K/V 梯度。
  已验证范围还包括 query 侧 `ge`/`lt` gate 的阈值边界与边启闭；
  gate 用例仅验证边启闭，不包含 attention 输出或梯度对照，
  不能将全部 parity 用例描述为均包含梯度验证。
  group policies 是额外扩展，通过粗粒度 frame causality 对照，
  不属于上游 native modes。

parity 测试可通过 `LOONGFORGE_VPP2_REFERENCE_ROOT` 指定参考 checkout；
参考源码缺失时默认 skip，设置 `LOONGFORGE_REQUIRE_VPP2_REFERENCE=1` 则失败。
源码指纹改变会失败，不能把 skip 当作验证通过。2026-11-20 整合验证为
830 passed、4 个不适用模型的 head 用例 skipped；其中 VPP2 对照 348 项全部
通过，新增训练契约测试也通过。环境阻断、源码指纹和复现方式见
[VALIDATION.md](VALIDATION.md)。这些检查的范围是可见性与局部 attention 数值，
**不代表完整 VPP2 训练 recipe 等价，也不代表 KV routing 等价**。
