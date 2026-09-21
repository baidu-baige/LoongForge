# LoongForge —— 具身模型训练

本目录提供具身模型离线评测。训练通过 `loongforge/train.py` 启动，Torch 运行时位于 `loongforge/engines/torch/`，模型与数据实现位于 `loongforge/models/embodied/` 和 `loongforge/data/embodied/`。

- **Benchmark 适配器**：覆盖 LIBERO、CALVIN、SimplerEnv、RoboTwin 和 ManiSkill。
- **模型策略服务**：通过统一的 `predict_action` RPC 协议连接各模型工厂。
- **动作语义处理**：payload builder、action decoder、归一化与指标均保留在评测域。

---

## 评测范围

该评测栈只属于具身领域；当前没有全框架公共 `eval.py` 分发器。

---

## 快速开始

完整的框架使用说明请参阅 [用户手册](../../../docs/source_zh/embodied_tutorial/overview.md)，各模型的快速入门：

- [π0.5 (pi05)](../../../docs/source_zh/embodied_tutorial/quick_start_pi05.md)
- [GR00T-N1.6](../../../docs/source_zh/embodied_tutorial/quick_start_groot_n1_6.md)
- [GR00T-N1.7](../../../docs/source_zh/embodied_tutorial/quick_start_groot_n1_7.md)
- [FastWAM](../../../docs/source_zh/embodied_tutorial/quick_start_fastwam.md)
- [DreamZero](../../../docs/source_zh/embodied_tutorial/quick_start_dreamzero.md)
- [Cosmos3](../../../docs/source_zh/embodied_tutorial/quick_start_cosmos3.md)
- [xVLA](../../../docs/source_zh/embodied_tutorial/quick_start_xvla.md)
- [Lingbot-VA](../../../docs/source_zh/embodied_tutorial/quick_start_lingbot_va.md)

---

## 性能

相较主流开源 baseline 的训练加速比（性能仍在积极优化中，这些数字后续还会持续提升）：

| 模型 | 类型 | Baseline | 加速比 |
|---|---|---|---|
| DreamZero (DROID Wan2.2-5B Full) | WAM | DreamZero | **2.67×** |
| GR00T-N1.6 | VLA | LeRobot | **2.31×** |
| π0.5 | VLA | OpenPI | **2.23×** |
| Lingbot-VA | WAM | LingBot-VA | **1.80×** |
| xVLA | VLA | X-VLA | **1.69×** |

数据反映测量时刻的 baseline 与 LoongForge 版本，可能随实现演进而变化。跨所有模型族的完整基准图表见 [根 README](../../../README_zh.md#-性能表现)。

---

## 目录结构

```
loongforge/
├── train.py
├── models/
│   ├── catalog.py
│   └── embodied/
│       ├── registry.py
│       └── <model>/
├── data/embodied/
│   ├── dataloader.py
│   └── datasets/
├── engines/
│   ├── mcore/
│   └── torch/
│       ├── entrypoint.py
│       ├── parser.py
│       ├── training_args.py
│       ├── trainers/
│       ├── distributed/
│       └── optimizer/
└── evaluation/embodied/
```

---

## 评测架构

- `orchestrator/` 根据 YAML 运行 benchmark 并管理策略服务。
- `adapters/` 将 benchmark 的观测与结果转换为统一协议。
- `payload_builders/` 与 `action_decoders/` 将模型动作映射为环境动作。
- `servers/` 加载模型工厂，并通过 RPC 暴露 `predict_action`。
- `transport/` 负责 WebSocket/msgpack-numpy 客户端与服务端。
- `metrics/` 写入单次运行和套件报告。

新增评测模型时，在 `factories/` 添加工厂；只有动作语义不同才新增 payload builder 或 decoder，并在 `examples/embodied/*/eval/` 下添加示例 YAML。

---

详见 [评测用户指南](../../../docs/source_zh/embodied_tutorial/eval_user_guide.md)。
