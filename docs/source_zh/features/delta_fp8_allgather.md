# Delta-FP8 AllGather

Delta-FP8 AllGather 是 LoongForge embodied 训练栈中一项可选的 FSDP2 通信优化。它通过传输按 block 量化的 FP8 参数差值，代替完整 BF16 参数，从而减少参数 AllGather 通信量。该功能只改变通信精度，forward 和 backward 计算仍使用 FSDP 混合精度策略配置的 dtype。

Delta-FP8 AllGather 与 LoongForge 的端到端 [FP8 训练](fp8_training.md) 是两项独立功能。启用本功能不会将模型权重、激活或 GEMM 计算转为 FP8。

## 1. 使用条件

Delta-FP8 需同时满足以下条件：

- 通过 `--distributed-strategy fsdp` 使用 FSDP2
- 使用 CUDA 设备和 NCCL 分布式 backend
- NVIDIA GPU 的 compute capability 不低于 8.9
- PyTorch 支持 FP8 E4M3
- CUDA Triton backend 提供 `tl.float8e4nv`
- FSDP 参数通信使用 BF16 和默认 AllGather 实现

FSDP group 注册时会执行运行时能力检查。不支持的设备或 backend 会在启动阶段报错，并在错误中输出检测到的设备和 backend。非 BF16 参数组或自定义 AllGather 实现保留原生 FSDP 行为。

## 2. 使用方法

### 2.1 DreamZero Wan2.2-5B

DreamZero Wan2.2-5B Full FSDP recipe 已默认启用 Delta-FP8 及验证过的参数，无需额外传入 Delta-FP8 参数：

```bash
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh
```

如需使用原生 BF16 AllGather 进行 A/B 对比：

```bash
bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh \
    --no-fsdp-delta-fp8-allgather
```

### 2.2 其他 Embodied 模型

框架级默认为关闭。在 FSDP 启动脚本后追加以下参数，即可为符合条件的参数组启用 Delta-FP8：

```bash
bash path/to/fsdp_launcher.sh \
    --fsdp-delta-fp8-allgather
```

通信路径能够运行不代表已证明该功能适合所有模型的精度和性能要求。将它用于其他 recipe 前，需与同一模型的 BF16 FSDP baseline 对比 loss、step time 和峰值显存。

## 3. 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--fsdp-delta-fp8-allgather` | `False` | 为当前模型符合条件的 FSDP group 启用 Delta-FP8 |
| `--fsdp-delta-fp8-block` | `256` | 共享一个 FP32 scale 的元素数；必须是不大于 1,048,576 的正整数 2 次幂 |
| `--fsdp-delta-fp8-prime-steps` | `1` | 转入差值通信前，用于初始化每个 FSDP unit 的完整 BF16 AllGather 次数 |
| `--fsdp-delta-fp8-reprime-interval` | `0` | 每 N 次 unshard 执行一次完整 BF16 AllGather；`0` 表示关闭周期性重新初始化 |

除非同配置的 loss 和性能验证支持修改，建议保持默认 block size 和 prime 设置。更小的 block 能提供更精细的量化 scale，但会传输更多 scale 数据。周期性重新初始化可在参数非连续变化后重新锚定 reference，但也会增加完整 BF16 collective。

## 4. 工作原理

对于每个符合条件的 FSDP unit：

1. 通过完整 BF16 AllGather 初始化持久 BF16 reference。
2. 将本地参数 shard 与 reference 中对应的 shard 进行比较。
3. 将差值按 block 量化为 FP8 E4M3 payload，并为每个 block 生成 FP32 scale。
4. 在 rank 之间 gather FP8 payload 和 scale，并保留调用方的异步 collective 语义。
5. 反量化 gathered delta，并将其累加到 FSDP 使用的 BF16 reference。

将重建后的差值累加回 reference，可以使量化残差进入后续更新，而不是每次都相对固定初始值量化。reference 直接复用 FSDP unsharded 参数存储，量化 scratch buffer 在 FSDP unit 间共享，以避免为每个 unit 持久分配 staging buffer。

## 5. 验证建议

Delta-FP8 使用有损通信。对新模型、优化器、block size 或重新初始化策略，应使用相同的权重、数据顺序、batch 配置、精度、硬件和测量区间，与 canonical 原生 BF16 FSDP 运行进行对比。至少检查：

- 代表性训练区间内的逐步 loss 和 gradient norm
- 排除初始化和编译 warmup 后的稳定 step time
- 每个 rank 的 GPU 峰值显存
- NaN、Inf、通信和分布式运行时错误

不应仅根据一次短测成功就认定该模型已完成支持。只有在该 workload 的数值和性能行为完成验证后，才应将该配置纳入默认 recipe。

## 6. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 启动时报设备、backend 或 Triton FP8 类型不支持 | 使用原生 BF16 AllGather，或切换到支持的 CUDA/NCCL 环境 |
| 性能没有提升 | 确认通信使用 BF16 默认 FSDP AllGather，并确认 AllGather 是当前主要瓶颈 |
| loss 偏离 BF16 reference | 恢复默认 block 和 prime 设置；若差异仍不可接受，对该 workload 关闭 Delta-FP8 |
| 参数非连续更新后行为改变 | 验证非零 `--fsdp-delta-fp8-reprime-interval`，或重启训练以重新初始化 reference |
