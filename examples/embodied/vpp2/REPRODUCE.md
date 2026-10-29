# VPP2 受控复现与验证入口

## 当前结论与边界（2026-10-27，公共修改收敛后重新实跑）

本轮证据总目录：两仓父目录下的
alignment/refactor_20261027_1945/，**最终结论以其中 README.md 和比较 JSON 为准**。
历史 alignment/gpu_20261027_fresh/ 不作为本次代码修改后的通过证据。

- 已重新跑通：真实 LIBERO 管线、双 GPU BF16 ZeRO-2、tiny 2-layer MoT，
  两侧继承原生 Trainer 完整生命周期，各 100 步；loss/LR/gradnorm/最终参数精确相同。
- 已重新跑通：tiny ZeRO-2 从 step50 恢复，51–100 与连续训练精确一致。
- 公共 DDP/FSDP Trainer 的独立小 MLP 回归：当前代码与 HEAD 公共框架各 5 步，
  loss/LR/gradnorm/最终参数精确一致。不是 VPP2 FSDP 验收。
- VPP2 tiny FSDP 首步有 Tensor/DTensor 混用错误，换回 HEAD 公共文件仍可复现，
  未为此修改公共框架，也不宣称 VPP2 FSDP 可用。
- 完整 14B 新诊断已完成：两侧各100步，loss MAE=0、gradnorm/LR差0、首尾梯度及最终全部可训参数精确相同，冻结未变。结果见本轮 full100/comparison.json。诊断使用真实全部
  40 层/真实 VAE、CLIP、数据，但 B1/模型并行/CPU FP32-master AdamW，
  **不经过公共 Trainer，不是原生 ZeRO-2 或论文精度验收**。
- **原配方完整 14B / 8×32 / ZeRO-2 验收仍 BLOCKED**：繁忙卡可用显存不足；
  仅 frozen video BF16 权重即约 31,271 MiB/卡，ZeRO-2 不切分这部分权重。
  未停止其他任务；未以 tiny 或模型并行诊断冒充原配方通过。

## 共享独立环境和资产

以下命令在两仓父目录执行。官方 live 工作区原有改动保持不变，只新增 scripts/repro/。

    python -m venv --system-site-packages .venv-vpp2-repro
    .venv-vpp2-repro/bin/python -m pip install --no-deps deepspeed==0.18.9
    export PYTHON_BIN="$PWD/.venv-vpp2-repro/bin/python"
    export VPP2_ASSETS_ROOT=/ssd1/vpp2_assets
    export VPP2_BASE_ROOT="$PWD/video-prediction-policy-2"
    export VPP2_LOONG_ROOT="$PWD/LoongForge"
    export VPP2_ALIGNMENT_INIT="$PWD/alignment/20261027/shared_init_seed42.pt"

本次实际环境为 .venv-vpp2-gpu/，DeepSpeed==0.18.9。没有升级系统环境或 pyproject。
若缺少共享初态，先生成一次（不要覆盖已经用作证据的文件）：

    CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" video-prediction-policy-2/scripts/repro/prepare_alignment_init.py \
      --output /path/to/shared_init_seed42.pt --seed 42
    export VPP2_ALIGNMENT_INIT=/path/to/shared_init_seed42.pt

只共享上游 action_init 未保存的四个 action 投影张量与两个 proprio 张量；
1010 个 backbone 张量严格加载真实 action_init.pt，不以随机权重代替。
这是明确标注的受控初态，不是某个历史官方 run 的初态。
默认真实本地数据只有 4 episodes / 579 帧，不是完整论文数据。

## 1. 完整原生配方（需足够空闲 8 卡，当前未验收）

先做无 GPU 分配预检：

    bash video-prediction-policy-2/scripts/repro/run_action_zero2.sh --dry-run
    bash LoongForge/examples/embodied/vpp2/run_action_zero2_repro.sh --dry-run

资源满足后顺序跑，使用**新目录**：

    OUTPUT_DIR=/tmp/new_vpp2_base TRAIN_ITERS=100 \
      bash video-prediction-policy-2/scripts/repro/run_action_zero2.sh
    OUTPUT_DIR=/tmp/new_vpp2_loong TRAIN_ITERS=100 \
      bash LoongForge/examples/embodied/vpp2/run_action_zero2_repro.sh
    "$PYTHON_BIN" LoongForge/examples/embodied/vpp2/compare_repro.py \
      --base /tmp/new_vpp2_base --transplant /tmp/new_vpp2_loong --steps 100

默认 8×32、GA1、workers0、完整 batch，**不支持内部 chunk**；
VPP2_ACTION_TRAIN_MICROBATCH_SIZE 非空直接拒绝。
两侧同 BF16 直接存参/无额外 autocast、seed42+rank、AdamW(0.9,0.95)、
foreach=False、wd0.01、lr1e-4、warmup500、schedule30000。
TRAIN_ITERS 只改变停止步数，不缩短 LR horizon；SAVE_INTERVAL 默认0。
设 VPP2_DETERMINISTIC=1 时两侧必须一起设。

官方 ComparisonTrainer 仅覆盖 scheduler horizon、私有 loader RNG 和 JSON 观测，
直接继承 LiberoTrainer.train()；LoongForge 使用原生 FinetuneTrainer，
只在显式 deepspeed strategy 分支进入新增后端。
自动保存 metrics、effective DS config、初态指纹、manifest、依赖、Git diff 和资产 hash。
比较器拒绝不完整 step、NaN、跳步、不同初态/资产/依赖/配方。
100步 MAE<=0.001 不是完整计划验收：≥3次重复、200步、2000步和 eval 仍需独立完成。

## 2. 实际公共 Trainer 端到端小模型回归（资源受限诊断）

这是 tiny 2层、mock VAE/CLIP、context 前16维，但两侧独立走真实数据管线，
不使用合成 batch，不手写训练 loop；BASE 和 TRANS 均继承各自训练器。
默认双卡、每卡B2、GA1。必须安装两个 checkout，探针辅助类从 LoongForge 导入。

    CUDA_VISIBLE_DEVICES=6,7 OUTPUT_DIR=/tmp/new_native_tiny_base \
      bash video-prediction-policy-2/scripts/repro/run_native_trainer_probe.sh
    CUDA_VISIBLE_DEVICES=6,7 OUTPUT_DIR=/tmp/new_native_tiny_loong SAVE_INTERVAL=50 \
      bash LoongForge/examples/embodied/vpp2/run_native_trainer_probe.sh
    "$PYTHON_BIN" LoongForge/examples/embodied/vpp2/compare_native_probe.py \
      --base /tmp/new_native_tiny_base --transplant /tmp/new_native_tiny_loong \
      --steps 100 --output /tmp/new_native_tiny_comparison.json

断点恢复（只复制指定 step50 checkpoint 到新输出，然后由公共后端恢复）：

    CUDA_VISIBLE_DEVICES=6,7 OUTPUT_DIR=/tmp/new_native_tiny_resume \
      RESUME_FROM=/tmp/new_native_tiny_loong/checkpoints/steps_50 \
      bash LoongForge/examples/embodied/vpp2/run_native_trainer_probe.sh
    "$PYTHON_BIN" LoongForge/examples/embodied/vpp2/compare_native_probe.py \
      --base /tmp/new_native_tiny_loong --transplant /tmp/new_native_tiny_resume \
      --steps 100 --start 51 --output /tmp/new_native_tiny_resume_comparison.json

本轮原始命令：run_native_tiny.sh、run_final_native_pair.sh、run_native_regressions.sh，
均在本轮证据目录；保留了失败日志，不覆盖已有输出。

## 3. 完整 14B 模型计算诊断（不是原生后端验收）

    export VPP2_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
    bash video-prediction-policy-2/scripts/repro/run_gpu_probe.sh \
      --output /tmp/new_full14b_base --steps 100 --repeats 1 --snapshot
    bash LoongForge/examples/embodied/vpp2/run_gpu_probe.sh \
      --output /tmp/new_full14b_loong --steps 100 --repeats 1 --snapshot
    "$PYTHON_BIN" LoongForge/examples/embodied/vpp2/compare_gpu_probe.py \
      --base /tmp/new_full14b_base --transplant /tmp/new_full14b_loong \
      --output /tmp/new_full14b_comparison.json --allow-single-run

默认 CUDA0=action、1–4=frozen video、5=VAE/CLIP。
保留真实权重/真实输入和全部层，手动模型并行及 CPU FP32-master AdamW。
单次100步必须显式 --allow-single-run；不产生计划要求的 run-to-run band。

## 4. CPU 检查与公共修改范围

    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
      "$PYTHON_BIN" LoongForge/examples/embodied/vpp2/check_data_parity.py
    cd LoongForge
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=. \
      "$PYTHON_BIN" -m pytest -q tests/embodied/unit/test_vpp2*.py tests/embodied/unit/test_deepspeed_backend.py

LingBot、公共 optimizer、递归 unwrap、公共日志优化和 LR fallback、pyproject、
公共 batch_weight/chunk API 均已撤回。VPP2 foreach 放在本模型 hook，
DS config 放本目录 zero2_bf16.json，独立依赖见 requirements-repro.txt。
公共运行时只保留显式 DS engine 生命周期/保存恢复及其 CLI 校验接缝；
不在模型 forward 中调用 optimizer/backward，也不为此修改其他模型。
