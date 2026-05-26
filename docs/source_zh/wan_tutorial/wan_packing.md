# Wan2.2 Packing 训练

Wan2.2 packing 将多个变长视频样本拼接为一条 packed 训练序列。通过 THD `PackedSeqParams` 维护每个样本的注意力与 loss 边界，使 packed 样本之间互不注意力、padding token 不计入 loss。

## 适用场景

当 Wan2.2 训练集中包含长度差异较大的视频或文本提示时，建议开启 packing。Packing 可减少因 padding 带来的计算浪费，并兼容上下文并行训练。

支持的上下文并行模式：
- 无 CP：`CP_SIZE=1`
- Ring CP：`CP_SIZE>1`，`CP_ULYSSES_DEGREE=1`
- Ulysses CP：`CP_SIZE=CP_ULYSSES_DEGREE`
- Ring + Ulysses 混合：`CP_SIZE>CP_ULYSSES_DEGREE>1`

## 数据要求

使用与普通训练相同的预处理 Wan 数据集格式。每个样本应提供：
- `input_latents`：视频 latent 张量
- `y`：可选的图像条件 latent
- `context`：文本嵌入
- `seed`：用于确定性噪声和时间步生成的样本种子
- `grid_sizes`：可选的 latent patch 网格；缺失时 LoongForge 会从 `input_latents` 推导

Packing 支持变长样本。在 CP 训练中，packed bin 中的每个样本会先填充至逐样本 CP 切分边界，然后再拼接。

## 启用方式

在 Wan 预训练脚本中添加 packing 参数：

```bash
--packing-sft-data
--packing-buffer-size 512
```

示例启动命令：

```bash
cd examples/wan
CUDA_VISIBLE_DEVICES=0,1,2,3 \
CP_SIZE=4 \
CP_ULYSSES_DEGREE=2 \
bash pretrain_wan2.2_i2v_a14b.sh
```

`--packing-buffer-size` 控制在组成 packed bin 前缓冲的样本数量。更大的缓冲区可提高打包密度，但会占用更多主机内存。

## 注意事项与限制

- Packing 当前使用 `micro_batch_size=1`；开启 packing 时验证器会强制此设置。
- Packed 注意力路径使用 THD 元数据，与非 packed 稠密注意力可能不是逐位一致，但 loss 在数值上应保持接近。
- 确保 `seq_length` 足够容纳一个 packed bin。在 CP 模式下，LoongForge 会将有效序列长度对齐到所需的 CP 切分边界。
- 进行精度验证时，请在相同数据顺序下对比前几个训练迭代与关闭 packing 的运行结果，且两次运行的 `train-iters` 不应改变。
