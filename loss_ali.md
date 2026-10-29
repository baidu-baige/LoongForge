已按建议修改，并在修改后**重新实际跑了 GPU 验证，没有沿用旧结论**。

**结论：本轮已测试范围内 loss 对齐；但完整原配方 `14B / 8×32 / ZeRO-2` 仍未完成验收，不能宣称最终精度全部对齐。**

### 1. 公共修改已收敛

- **LingBot 训练器已恢复到 HEAD，当前无 diff。**
- 撤回公共 `batch_weight/chunk` 接口、递归 unwrap、公共 optimizer 强制 `foreach=False`、日志归约优化、LR fallback、全局 DeepSpeed 版本升级。
- `foreach=False` 移到 **VPP2 自己的初始化 hook**。
- 公共运行时仅保留原生 DeepSpeed 所需的 6 个文件，涉及引擎初始化、backward/step、保存恢复和参数校验。

逐文件必要性说明：
`/workspace/vpp2/alignment/refactor_20261027_1945/FILES.md`

### 2. 修改后的新实跑结果

| 验证项 | 结果 |
|---|---|
| 完整 14B 模型诊断，两侧各 100 步 | loss MAE/max 差均为 **0**；梯度范数、LR 一致；第 1/100 步梯度及最终全部可训参数精确一致；冻结参数未变 |
| 原生 Trainer、双卡 BF16 ZeRO-2、真实 LIBERO 数据、小型 VPP2，两侧各 3 次×100 步 | loss、LR、梯度范数、输入指纹及最终参数全部精确一致 |
| step50 断点恢复至 step100 | 与连续训练精确一致 |
| 公共 DDP/FSDP Trainer 小模型，修改后对比 HEAD | 各 5 步，loss、梯度范数、最终参数一致 |
| CPU 回归 | **206 passed** |

**边界说明：**
- 完整 14B 诊断采用 **B1、模型并行、CPU FP32-master AdamW**，不经过公共 Trainer，不能替代原生 ZeRO-2 验收。
- VPP2 自身的 FSDP 测试首步报 Tensor/DTensor 混用；换回 HEAD 公共框架仍能复现，**未通过**。
- 原配方显存不足：当前每卡空闲约 15–20GB，仅冻结 video 的 BF16 权重就需约 **31GB/卡**。没有停止其他任务。

### 3. 脚本在哪里

| 用途 | 官方仓库 | LoongForge |
|---|---|---|
| 完整原生配方入口，当前资源阻塞 | `video-prediction-policy-2/scripts/repro/run_action_zero2.sh` | `LoongForge/examples/embodied/vpp2/run_action_zero2_repro.sh` |
| 已实跑的原生 Trainer 小模型入口 | `video-prediction-policy-2/scripts/repro/run_native_trainer_probe.sh` | `LoongForge/examples/embodied/vpp2/run_native_trainer_probe.sh` |
| 完整 14B 诊断入口 | `video-prediction-policy-2/scripts/repro/run_gpu_probe.sh` | `LoongForge/examples/embodied/vpp2/run_gpu_probe.sh` |

以上路径均相对 `/workspace/vpp2/`，两个目录都保存了 `REPRODUCE.md`。

### 4. 结果在哪里

总报告：
**`/workspace/vpp2/alignment/refactor_20261027_1945/README.md`**

该目录下重点文件：
- `SUMMARY.json`：结论汇总
- `full100/comparison.json`：新跑完整 14B、100 步比较结果
- `native_final/comparison1.json`、`comparison2.json`：最终复现入口的端到端结果
- `native_final/resume_comparison.json`：恢复训练比较
- `common_mlp_final/comparison.json`：公共 DDP/FSDP 非回归
- `pytest_final2.log`：206 项测试日志

必要新增文件已使用 **`git add -N`** 放到 **Changes not staged for commit**；实际暂存内容为空，未提交。无关的 `docs/workflows/`、`logs/`、`loong.sh` 等保持未动。官方已有 tracked 源码/config 的 diff 指纹也保持不变。
