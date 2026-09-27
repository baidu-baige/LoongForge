<p align="right"><sub><a href="./README.md">English</a> | <b>简体中文</b></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="500">
  </picture>
</p>

<h3 align="center">更快地训练 LLM、VLM、Diffusion 与具身模型</h3>

<p align="center">
  <a href="https://loongforge.readthedocs.io/zh-cn/latest/index.html"><b>中文文档</b></a>
  &nbsp;·&nbsp;
  <a href="https://baidu-baige.github.io/LoongForge/blog/"><b>项目博客</b></a>
  &nbsp;·&nbsp;
  <a href="#quickstart"><b>快速开始</b></a>
  &nbsp;·&nbsp;
  <a href="#performance"><b>性能表现</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>支持模型</b></a>
</p>

<p align="center">
  <a href="https://github.com/baidu-baige/LoongForge/stargazers"><img height="22" src="https://img.shields.io/github/stars/baidu-baige/LoongForge?style=flat-square&color=FFD700&logo=github&logoColor=white&label=Stars" alt="GitHub stars"></a>
  <a href="./LICENSE"><img height="22" src="https://img.shields.io/badge/License-Apache_2.0-8250DF?style=flat-square" alt="Apache 2.0 开源协议"></a>
  <a href="https://hub.docker.com/u/loongforge"><img height="22" src="https://img.shields.io/badge/Docker_Image-loongforge-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker Hub 镜像"></a>
  <a href="./CONTRIBUTING.md"><img height="22" src="https://img.shields.io/badge/Pull_Requests-welcome-brightgreen?style=flat-square&logo=github&logoColor=white" alt="欢迎 PR"></a>
</p>

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/"><img src="https://img.shields.io/badge/🌐_Visit_Website-7C3AED?style=for-the-badge" alt="访问官网"></a>
  <a href="https://discord.gg/RnY39D6CM"><img src="https://img.shields.io/badge/Join_Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="加入 Discord 社区"></a>
  <a href="https://github.com/baidu-baige/LoongForge/issues/80"><img src="https://img.shields.io/badge/Join_WeChat_Group-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="加入微信群"></a>
  <a href="https://github.com/baidu-baige/LoongForge/issues/80"><img src="https://img.shields.io/badge/Join_RedNote-FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white" alt="小红书"></a>
  <a href="https://x.com/baidu_baige_lf"><img src="https://img.shields.io/badge/Follow_Us-000000?style=for-the-badge&logo=x&logoColor=white" alt="在 X 上关注我们"></a>
</p>

<p align="center"><i>⭐ 点个 Star，帮助更多人发现 LoongForge，也让社区不断壮大。</i></p>

</div>

## 🐉 LoongForge

**LoongForge** 是百度智能云 [百舸团队](https://cloud.baidu.com/product/aihc.html) 打造的开源训练框架，面向主流 **LLM、VLM、Diffusion 与具身模型**，提供[更快的训练速度](#performance)。

- **易用** —— 为每个支持的模型提供[开箱即用的配置](./configs/models/)与[启动示例](./examples)，覆盖**预训练**、**持续预训练**、**SFT** 与 **LoRA** 等训练范式。
- **高性能** —— 基于多后端架构（Megatron-LM 与 torch-native），在并行策略、显存占用、通信隐藏、算子效率等维度做**深度优化**，同时保证**训练 loss 曲线与基线对齐**。
- **源自生产** —— 由 [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr) 开源而来，既服务于企业客户的闭源模型，也支撑了[开源模型的训练与发布](#powered-by-loongforge)，生产规模最大达 **5,000+ XPU**。

---

<h6 align="left">性能对比举例：具身模型 DreamZero 吞吐达基线的 4.38 倍，loss 曲线保持对齐</h6>

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/assets/video/dreamzero-comparison.mp4">
    <picture>
      <source media="(prefers-reduced-motion: reduce)" srcset="./docs/assets/images/demo/dreamzero-poster.jpg">
      <img alt="DreamZero 训练左右对照：LoongForge 吞吐达到基线的 4.38 倍，训练 loss 曲线保持对齐" src="./docs/assets/images/demo/dreamzero-loop.webp" width="100%" />
    </picture>
  </a>
</p>

## 🏗️ 架构

由于不同类别、不同规模的模型，最优训练策略不同，LoongForge 采用多后端架构。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/architecture/loongforge-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/architecture/loongforge-architecture.svg">
    <img alt="LoongForge 架构：面向 LLM / VLM / 扩散模型的 Megatron 栈，与面向具身模型的 torch-native 栈" src="./docs/assets/images/architecture/loongforge-architecture.svg" width="100%">
  </picture>
</p>

- **Megatron 栈** —— 面向 LLM、VLM 与 Diffusion 模型。基于 [patch 过的 Megatron-LM](https://github.com/baidu-baige/Loong-Megatron) 构建，并扩展了 MoE 并行、组件级异构并行、长序列优化等能力。
- **Torch-Native 栈** —— 面向具身模型（VLA 与 WAM）。独立的 [torch-native 子系统](./loongforge/embodied)，支持 **DDP / ZeRO-1 / FSDP / HSDP**，并针对典型模型做了深度性能优化，涵盖 I/O、通信策略、kernel 效率等。

## 🔥 最新动态

- **[2026/09]** ✨ 新增 **[GLM-5.3-flash](./examples/glm5_next/)** 训练支持。
- **[2026/09]** ✨ 新增 **[Kimi-K3](./examples/kimi_k3/)** 的 LLM 与 VLM BF16 训练支持。
- **[2026/09]** ⚡ 新增优化后的 **[DreamZero Wan2.2-5B FSDP recipe](./examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh)**，集成 cache-aware 数据加载、attention block 编译、冻结模块处理与 FSDP2 Delta-FP8 Param AllGather。
- **[2026/08]** 🤖 新增 **[Wall-OSS-0.5](./examples/embodied/wall_oss_0_5/)** VLA 训练支持，并通过自定义融合算子提升训练吞吐。
- **[2026/08]** 📄 发布 **[TAOT 论文](https://arxiv.org/abs/2608.03676)** —— 通过拓扑感知的动态专家副本放置，优化 **MoE** 训练中的专家并行（**EP**）负载不均衡，相较业界方案开销最大可降低 **74%**，案例实测 **1.43× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-08-taot-topology-aware-expert-placement.html)]
- **[2026/08]** ✨ 新增 **GLM-5.2** 训练支持，并提供 **[GLM-5.2 + MoonViT](./configs/models/glm5.2_vit/)** 自定义组合[示例](./examples/glm5.2_vit/)，可用于为 GLM 扩展多模态能力。
- **[2026/08]** ✨ 新增 **MiniCPM-V-4.6** 与 **Qwen3.8-27B** 训练支持。
- **[2026/08]** 🧪 Embodied 栈新增统一[**评测模块**](./loongforge/embodied/eval/)，当前已覆盖 **Pi0.5 / xVLA / GR00T**，持续扩展中。
- **[2026/07]** 🐳 统一**预构建 Docker 镜像** —— LLM / VLM / VLA / Diffusion 全部模型家族共用同一镜像。
- **[2026/07]** 🤖 发布 **[LoongForge-Embodied](./loongforge/embodied)** —— 面向具身模型（Pi0.5、GR00T-N1.6/N1.7、xVLA、LingBot-VA、FastWAM、DreamZero、Cosmos3）的 torch-native DDP/FSDP 训练子系统，实测最高 **4.38× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-07-announcing-loongforge-embodied.html)]
- **[2026/07]** ✨ 新增 **DeepSeek-V4-Flash / DeepSeek-V4-Pro** 训练支持。

<details>
<summary><b>📅 更多</b></summary>

- **[2026/07]** ✨ 新增 **Qwen-Image-Edit-2511** 训练支持。
- **[2026/06]** 🤖 扩展 VLA 模型覆盖，新增 **GR00T N1.6**；GR00T 训练实现 **2.3× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html)]
- **[2026/05]** ⚡ **Wan 2.2** 训练 **加速 116%**，并新增 CP（上下文并行）与数据 packing 策略支持。
- **[2026/05]** ✨ 新增 **Kimi K2.5 / K2.6** 训练支持，并支持 **INT4 / NVFP4** PTQ 量化能力。
- **[2026/05]** 🎉 **v0.1.0** —— LoongForge 首个正式版本发布。
- **[2026/05]** 🌟 支持 **LLaVA-OneVision-2.0** 模型训练并协助其公开发布。
- **[2026/04]** 🧩 新增 **MiniMax-M2.7** 在 NVIDIA GPU 与昆仑芯 XPU 上的训练支持。
- **[2026/04]** 🚀 LoongForge 源码在 GitHub 上正式公开。[[blog](https://zhuanlan.zhihu.com/p/2031006068797600446)]
- **[2025/10]** 🌟 基于 AIAK-Training-LLM（LoongForge 前身）支持 **LLaVA-OneVision-1.5** 模型训练并协助其公开发布。[[blog](https://mp.weixin.qq.com/s/1y7Br15pBpUZ-90j5OGncA)]

</details>

## ✨ 核心特性

**🚀 基座模型**

* **MoE EP 通信优化** —— All2All / 激活卸载 / 计算全链路重叠，相对上游 Megatron-LM 实现**进一步显存降低**。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/moe_all2all_overlap.html)]
* **MoE 专家负载均衡** —— 基于拓扑感知算法动态复制热点专家，均衡专家并行（EP）负载，开销相较业界方案最大可降低 **74%**。[[TAOT 论文](https://arxiv.org/pdf/2608.03676)]
* **自适应 FP8 训练** —— 面向 LLM 和 VLM 的端到端 FP8，支持标准 **blockwise FP8**；可选**自适应**模式根据 GEMM 形状与效率逐算子选择最佳精度。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/adaptive_fp8.html)]
* **自定义融合算子** —— 为 DSA 类模型设计的 **FusedDSA** 等融合 kernel —— TileLang 版本已开源，高性能 CUDA 版本在百度百舸平台提供。
* **长序列训练** —— 将 LLM 训练扩展至长序列场景，依托**上下文并行（CP）**与**分块流水线调度**。

**🧩 多模态模型**

* **灵活组合** —— 通过配置即可将可互换的 ViT 与 LLM 组件自由组装为 VLM（如 **GLM-5.2 + MoonViT**），无需编写模型代码。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/model_combination.html)]
* **异构并行** —— 针对模型不同组件（如 ViT vs LLM）独立配置 TP / DP / 重计算 / 冻结策略，获得最优吞吐与显存占用。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)] [[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/heterogeneous_parallel.html)]
* **Encoder-Decoder 解耦训练** —— 消除 Encoder 引入的流水线气泡，将 ViT 与 LLM 拆分为独立任务。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/heterogeneous_parallel.html#full-heterogeneous-dp-parallel)]
* **DP 负载均衡** —— 显著提升多节点扩展效率，基于负载感知的数据重分发缓解序列打包不均衡问题。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)] [[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/data_parallel_balancing.html)]
* **灵活的数据流水线** —— 多模态数据基于 Energon **WebDataset** 输入，支持**在线**与**离线**两种序列 packing。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/dataset_conversion.html)]

**🤖 具身模型**

* **VLA 与 WAM 训练** —— 面向 **VLA 与世界-动作模型（WAM）** 的独立 **torch 原生 DDP/FSDP** 子系统，与 Megatron 核心解耦，支持 **DDP / ZeRO-1 / FSDP / HSDP** 多种分布式策略。[[README](./loongforge/embodied)]
* **逐模型深度定制优化** —— 实测相对官方基线 **1.79×–4.38× 加速**（见[性能表现](#performance)），针对每个模型在 I/O、通信策略、算子效率等维度深度优化训练代码。
* **FP8 通信优化** —— 在支持的 NVIDIA GPU 上压缩跨卡通信量，覆盖两种并行策略：**FSDP2** 场景对参数 AllGather 做按 block 的 FP8 delta 压缩，**DDP** 场景做 FP8 梯度 all-reduce。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/fp8_communication.html)]
* **统一评测** —— 在 **LIBERO / CALVIN / SimplerEnv / RoboTwin** 上评测训练出的策略，覆盖度持续完善。[[README](./loongforge/embodied/eval)]
* **Ego2Robot 数据转换** —— 将第一人称人类操作视频转换为覆盖 **16 种双臂机器人形态**的 **LeRobot v3.0** 训练数据。[[README](./loongforge/embodied/tools/ego2robot)]

**🔌 兼容性**

* **Mcore Bridge** —— 同时支持**离线** **Megatron ↔ HuggingFace** 双向转换与**在线**原生 HF 加载/保存。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/mcore_bridge.html)]
* **异构硬件** —— 通过轻侵入式插件设计，原生支持 **NVIDIA GPU** 与**昆仑芯 XPU**。

> 📖 深入阅读：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/features_index.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/features_index.html) · [具身模型](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/overview.html)

<a id="performance"></a>
## 📊 性能表现

各模型相对主流开源基线的训练吞吐加速——每个模型与其基线的对比，均在相同机型与相同训练超参数下测得：

<p align="center">
  <img alt="LoongForge 相对开源基线的训练吞吐加速——从 Qwen3-VL 的 1.45 倍到 DeepSeek-V3.2 Lite 的 5.04 倍" src="./docs/assets/images/benchmark_speedup.png" width="860" />
</p>

> DeepSeek-V3.2 Lite 为 DSA 算子级优化的结果，受测试环境规模限制，在减层配置下验证。<br>
> 数据为特定时间点的测试结果，双方实现持续演进，数值可能随之变化。

<a id="quickstart"></a>
## ⚡ 快速开始

### 1. 安装

使用[最新 NVIDIA GPU 预构建镜像](https://hub.docker.com/u/loongforge)（需安装 NVIDIA Container Toolkit）：

```bash
docker pull loongforge/loongforge:latest
mkdir -p workspace
docker run --gpus all --ipc=host -it --rm \
  -v "$(pwd)/workspace:/workspace/data" \
  -w /workspace/LoongForge \
  loongforge/loongforge:latest bash
```

[源码安装](https://loongforge.readthedocs.io/zh-cn/latest/get_started/installation.html) · [昆仑芯 XPU 安装](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/install_p800.html)。

### 2. 按硬件与模态选择教程

- **NVIDIA GPU**：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/quick_start_llm_pretrain.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/quick_start_vlm_pretrain.html) · [VLA & WAM](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_index.html) · [Diffusion](https://loongforge.readthedocs.io/zh-cn/latest/wan_tutorial/quick_start_wan_training.html)
- **昆仑芯 XPU**：[昆仑芯 XPU 教程](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/README.html)

### 3. 找到模型的启动脚本

NVIDIA GPU 启动脚本见 [`examples/`](./examples/)，昆仑芯 XPU 启动脚本见 [`examples_xpu/`](./examples_xpu/)，配置见 [`configs/models/`](./configs/models/)。

#### 示例：DreamZero LoRA 微调

以 **DreamZero Wan2.2-5B LoRA（单机 8 GPU、FSDP）** 为例。按[教程](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_dreamzero.html)准备权重（含 Wan2.1 CLIP）和 DROID（LeRobot v2）数据，随后在容器内执行：

```bash
cd /workspace/LoongForge
export WAN22_CKPT_DIR=/workspace/data/dreamzero/checkpoints/Wan2.2-TI2V-5B
export WAN21_CKPT_DIR=/workspace/data/dreamzero/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_PATH="$WAN22_CKPT_DIR/google/umt5-xxl"
export DATA_PATH=/workspace/data/dreamzero/data/droid_lerobot

EMBODIMENT_TAG=oxe_droid \
  bash examples/embodied/dreamzero/prepare_dreamzero_dataset.sh

GPUS_PER_NODE=8 TRAIN_ITERS=20 SAVE_INTERVAL=20 \
OUTPUT_DIR=/workspace/data/dreamzero/outputs/lora \
  bash examples/embodied/dreamzero/run_dreamzero_wan22_5b_lora_fsdp_finetune.sh
```

示例运行 20 步，训练产物保存在 `OUTPUT_DIR` 下。

<a id="models"></a>
## 🏛️ 支持的模型

点击任意模型查看训练示例；完整使用说明见[用户手册](https://loongforge.readthedocs.io/zh-cn/latest/index.html)，全部变体见[模型支持矩阵](https://loongforge.readthedocs.io/zh-cn/latest/get_started/support_model.html)。

<table width="100%">
<colgroup>
<col width="25%">
<col width="25%">
<col width="25%">
<col width="25%">
</colgroup>
<thead align="center" valign="bottom">
<tr><th width="25%">LLM</th><th width="25%">VLM</th><th width="25%">Diffusion</th><th width="25%">Embodied</th></tr>
</thead>
<tbody valign="top">
<tr>
<td valign="top">
<ul>
<li><a href="examples/deepseek_v2/">DeepSeek-V2</a> ✅</li>
<li><a href="examples/deepseek_v3/">DeepSeek-V3/V3.2</a> ✅</li>
<li><a href="examples/deepseek_v4/">DeepSeek-V4</a> ✅</li>
<li><a href="examples/llama2/">LLaMA2</a> ✅</li>
<li><a href="examples/llama3/">LLaMA3</a> ✅</li>
<li><a href="examples/llama3.1/">LLaMA3.1</a> ✅</li>
<li><a href="examples/qwen/">Qwen</a> ✅</li>
<li><a href="examples/qwen1.5/">Qwen1.5</a> ✅</li>
<li><a href="examples/qwen2/">Qwen2</a> ✅</li>
<li><a href="examples/qwen2.5/">Qwen2.5</a> ✅</li>
<li><a href="examples/qwen3/">Qwen3</a> ✅</li>
<li><a href="examples/qwen3_next/">Qwen3-Next</a> ✅</li>
<li><a href="examples/minimax/">MiniMax-M2.1/2.5/2.7</a> ✅</li>
<li><a href="examples/mimo/">MIMO</a> ✅</li>
<li><a href="examples/glm5/">GLM-5</a> ✅</li>
<li><a href="examples/glm5.2/">GLM-5.2</a> ✅</li>
<li><a href="examples/kimi_k3/">Kimi-K3</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/qwen2.5_vl/">Qwen2.5-VL</a> ✅</li>
<li><a href="examples/qwen3_vl/">Qwen3-VL</a> ✅</li>
<li><a href="examples/qwen3.5/">Qwen3.5</a> ✅</li>
<li><a href="examples/qwen3.6/">Qwen3.6</a> ✅</li>
<li><a href="examples/qwen3.8/">Qwen3.8</a> ✅</li>
<li><a href="examples/kimi_k2.x/kimi_k2.5/">Kimi-K2.5/2.6</a> ✅</li>
<li><a href="examples/kimi_k3/">Kimi-K3</a> ✅</li>
<li><a href="examples/minicpm_v_4_6/">MiniCPM-V-4.6</a> ✅</li>
<li><a href="examples/glm5.2_vit/">GLM-5.2 + MoonViT</a> ✅</li>
<li><a href="examples/glm5_next/">GLM-5.3-flash</a> ✅</li>
<li><a href="examples/ernie4.5/">ERNIE4.5-VL</a> ✅</li>
<li><a href="examples/llava_onevision_1.5/">LLaVA-OneVision-1.5</a> ✅</li>
<li><a href="examples/internvl2.5/">InternVL2.5</a> ✅</li>
<li><a href="examples/internvl3.5/">InternVL3.5</a> ✅</li>
<li><a href="examples/custom/">CustomCombinedModel Example</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/wan/">Wan2.1</a> ✅</li>
<li><a href="examples/wan/">Wan2.2</a> ✅</li>
<li><a href="examples/qwen_image/">Qwen-Image-Edit-2511</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/embodied/pi05/">Pi0.5</a> ✅</li>
<li><a href="examples/embodied/groot_n1_6/">GR00T-N1.6</a> ✅</li>
<li><a href="examples/embodied/groot_n1_7/">GR00T-N1.7</a> ✅</li>
<li><a href="examples/embodied/xvla/">xVLA</a> ✅</li>
<li><a href="examples/embodied/wall_oss_0_5/">Wall-OSS-0.5</a> ✅</li>
<li><a href="examples/embodied/fastwam/">FastWAM</a> ✅</li>
<li><a href="examples/embodied/lingbot_va/">LingBot-VA</a> ✅</li>
<li><a href="examples/embodied/cosmos3/">Cosmos3</a> ✅</li>
<li><a href="examples/embodied/dreamzero/">DreamZero</a> ✅</li>
</ul>
</td>
</tr>
</tbody>
</table>

<a id="powered-by-loongforge"></a>
## 🌟 基于 LoongForge 训练

基于 **LoongForge** 或其前身 **AIAK-Training-LLM** 训练的开源模型：

| 模型 | 亮点 |
|------|------|
| [**LLaVA-OneVision-2.0**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) | 新一代多模态模型，配套全新的 VideoCaption 与 Spatial 数据集 |
| [**Innovator-VL**](https://github.com/InnovatorLM/Innovator-VL/tree/main) | 面向高级推理的科学多模态大模型 |
| [**LLaVA-OneVision-1.5**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/1.5) | 面向多模态训练民主化的全开源框架 |
| [**Qianfan-VL**](https://github.com/baidubce/Qianfan-VL) | 面向企业的领域增强视觉-语言模型，参数量覆盖 3B ~ 70B |

## 📂 代码结构

<details>
<summary><b>📁 目录树</b></summary>

```
LoongForge/
├── loongforge/                   # 核心训练框架
│   ├── train/                    # 训练入口与训练器
│   │   ├── pretrain/             #   预训练（LLM、VLM）
│   │   ├── sft/                  #   SFT（LLM、VLM、InternVL、ERNIE）
│   │   └── diffusion/            #   Diffusion（WAN、Qwen-Image）
│   ├── models/                   # 统一的模型抽象层
│   │   ├── foundation/           #   LLM 主干（LLaMA、Qwen、DeepSeek、...）
│   │   ├── encoder/              #   视觉编码器（ViT、Qwen-VL、InternVL、...）
│   │   ├── omni_models/          #   多模态组合
│   │   ├── diffusion/            #   Diffusion 模型（WAN、Qwen-Image）
│   │   └── common/               #   公共 Layer 与工具
│   ├── embodied/                 # LoongForge-Embodied：独立的 torch-native（DDP/FSDP）具身
│   │                             #   （VLA + 世界-动作）训练子系统，详见 loongforge/embodied/README_zh.md
│   ├── data/                     # 数据流水线（多模态、视频、DP 负载均衡）
│   ├── tokenizer/                # Tokenizer
│   └── utils/                    # 配置映射、常量等
├── third_party/Loong-Megatron/   # Patched Megatron-LM（git submodule）
├── configs/                      # Hydra YAML 配置（模型、数据）
├── examples/                     # GPU 启动脚本
├── examples_xpu/                 # 昆仑芯 XPU 启动脚本
├── tools/                        # Checkpoint 转换、数据预处理
├── ops/                          # 自定义融合算子（含开源的 TileLang 版本）
├── patches/                      # TransformerEngine 补丁
├── docker/                       # Dockerfile（GPU & XPU）
├── tests/                        # 端到端测试（YAML 驱动）
└── docs/                         # 文档
```

</details>

## 📝 引用

如果您觉得 LoongForge 对您的工作有帮助，请引用本项目：

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A high-performance framework for training LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

如果您在 LoongForge 中使用 TAOT 进行 MoE 训练，可以引用我们的论文：

```bibtex
@article{zhang2026taot,
  title   = {{TAOT}: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in {MoE} Training},
  author  = {Zhang, Lingyun and Zhang, Henghua and Gu, Shilei and Mo, Kai and Han, Shuai and Li, Shiyong and Wang, Yanpeng and Shen, Dou},
  journal = {arXiv preprint arXiv:2608.03676},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03676}
}
```

## 🤝 参与贡献

我们非常欢迎社区贡献 —— 无论是 Bug 报告、功能提案还是 PR。在提交前请阅读 [贡献指南](./CONTRIBUTING.md)。

非常感谢 LoongForge 的所有贡献者：

<a href="https://github.com/baidu-baige/LoongForge/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=baidu-baige/LoongForge&v=2026-09-04" alt="LoongForge contributors" />
</a>

## 🙏 致谢

LoongForge 的成长离不开开源社区。其 Megatron 栈以 NVIDIA 的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 为基础，项目也从 [HuggingFace Transformers](https://github.com/huggingface/transformers)、[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)、[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)、[LeRobot](https://github.com/huggingface/lerobot) 以及所支持模型的官方实现（如 [OpenPI](https://github.com/Physical-Intelligence/openpi)、[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)）中汲取了经验。同时也特别感谢 [LINUX DO](https://linux.do/) 社区，为技术交流提供了友善的空间，并对开源分享给予支持。

<a id="contact"></a>
## 💬 联系我们

| 渠道 | 适用场景 |
|------|----------|
| [**GitHub Issue**](https://github.com/baidu-baige/LoongForge/issues/new/choose) | 问题反馈、使用疑问与功能建议 |
| [**开发者社区**](https://github.com/baidu-baige/LoongForge/issues/80) | 微信群、小红书等 |
| [**邮件**](mailto:loongforge@baidu.com) | 企业落地、大规模部署、商务合作，以及任何其他话题 |

## 📄 开源协议

LoongForge 基于 [Apache License 2.0](./LICENSE) 发布；部分文件改编自第三方项目，详见各文件头部。
