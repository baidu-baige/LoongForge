<p align="right"><sub><a href="./README.md">English</a> | <b>简体中文</b></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="520">
  </picture>
</p>

<h3 align="center">更快地训练 LLM、VLM、Diffusion 与具身模型</h3>

<p align="center">
  <a href="https://loongforge.readthedocs.io/zh-cn/latest/index.html"><b>文档</b></a>
  &nbsp;·&nbsp;
  <a href="https://baidu-baige.github.io/LoongForge/blog/"><b>博客</b></a>
  &nbsp;·&nbsp;
  <a href="#performance"><b>性能表现</b></a>
  &nbsp;·&nbsp;
  <a href="#quickstart"><b>快速开始</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>支持模型</b></a>
  &nbsp;·&nbsp;
  <a href="#powered-by-loongforge"><b>项目案例</b></a>
</p>

<p align="center">
  <a href="https://hellogithub.com/repository/baidu-baige/LoongForge" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=2eafb075947e41a4be4d157a6f0382ae&claim_uid=PjNLOfBMoxlaTEq&theme=neutral" alt="Featured｜HelloGitHub" width="250" height="54" /></a>
  <a href="https://atomgit.com/baidu-baige/LoongForge" target="_blank"><img src="https://atomgit.com/baidu-baige/LoongForge/star/new_badge.svg" alt="AtomGit 星标" width="250" height="54" /></a>
</p>

<p align="center">
  <a href="https://github.com/baidu-baige/LoongForge/stargazers"><img src="https://img.shields.io/github/stars/baidu-baige/LoongForge?style=flat-square&color=FFD700&logo=github&logoColor=white&label=Stars" alt="GitHub stars"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-8250DF?style=flat-square" alt="Apache 2.0 开源协议"></a>
  <a href="https://hub.docker.com/u/loongforge"><img src="https://img.shields.io/badge/Docker_Image-loongforge-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker Hub 镜像"></a>
  <a href="./CONTRIBUTING.md"><img src="https://img.shields.io/badge/Pull_Requests-welcome-brightgreen?style=flat-square&logo=github&logoColor=white" alt="欢迎 PR"></a>
</p>

<p align="center">
  <a href="https://discord.gg/RnY39D6CM"><img src="https://img.shields.io/badge/Join_Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="加入 Discord 社区"></a>
  <a href="https://github.com/baidu-baige/LoongForge/issues/80"><img src="https://img.shields.io/badge/WeChat_Group-07C160?style=for-the-badge&logo=wechat&logoColor=white" alt="加入微信群"></a>
  <a href="https://github.com/baidu-baige/LoongForge/issues/80"><img src="https://img.shields.io/badge/RedNote_Group-FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white" alt="小红书"></a>
  <a href="https://www.zhihu.com/people/nbqbn3"><img src="https://img.shields.io/badge/Follow_on_Zhihu-0084FF?style=for-the-badge&logo=zhihu&logoColor=white" alt="在知乎上关注我们"></a>
  <a href="https://x.com/baidu_baige_lf"><img src="https://img.shields.io/badge/Follow_on_X-000000?style=for-the-badge&logo=x&logoColor=white" alt="在 X 上关注我们"></a>
</p>

</div>

**LoongForge** 是百度智能云 [百舸团队](https://cloud.baidu.com/product/aihc.html) 打造的开源训练框架，旨在加速主流 **LLM、VLM、Diffusion 与具身模型**的训练。

- **高性能** —— 基于 Megatron-LM 与 torch-native 后端，在[深度优化训练速度](#performance)的同时，保证**训练 loss 与基线对齐**。
- **易用性** —— 支持 40+ 个开源[模型](#models)，并附带开箱即用的配置与启动示例，另有权重转换、数据预处理等配套工具；覆盖**预训练**、**中期训练**、**SFT** 与 **LoRA**。
- **源自生产** —— 前身为 [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr)，既落地于企业客户的闭源模型训练，也支撑了[多个开源项目](#powered-by-loongforge)；在实际生产中，最大训练规模达 **5,000+ XPU**。

---

**示例**：在 LoongForge 上训练具身模型 DreamZero，吞吐达开源基线的 4.38 倍，loss 曲线保持对齐

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/assets/video/dreamzero-comparison.mp4">
    <picture>
      <source media="(prefers-reduced-motion: reduce)" srcset="./docs/assets/images/demo/dreamzero-poster.jpg">
      <img alt="DreamZero 训练左右对照：LoongForge 吞吐达到开源基线的 4.38 倍，训练 loss 曲线保持对齐" src="./docs/assets/images/demo/dreamzero-loop.webp" width="100%" />
    </picture>
  </a>
</p>

<p align="center">⭐ 点个 <b>Star</b>，帮助更多人发现 LoongForge，也让社区不断壮大。</p>

<a id="architecture"></a>
## 🏗️ 架构

不同类别、不同规模的模型，适合的训练策略各不相同，因此 LoongForge 采用多后端架构。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/architecture/loongforge-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/architecture/loongforge-architecture.svg">
    <img alt="LoongForge 架构：面向 LLM / VLM / Diffusion 的 Megatron 后端，与面向具身模型的 torch-native 后端" src="./docs/assets/images/architecture/loongforge-architecture.svg" width="100%">
  </picture>
</p>

- **Megatron 后端** —— 面向 [LLM、VLM 与 Diffusion 模型](#llm-vlm)，并在 MoE 并行、组件级异构并行、长序列优化等方面做了扩展。
- **torch-native 后端** —— 面向[具身模型](#embodied)（VLA 与 WAM），基于原生 PyTorch，支持 **DDP / ZeRO-1 / FSDP / HSDP**，并针对典型模型做了深度性能优化，涵盖 I/O、通信策略、kernel 效率等。

## 🔥 最新动态

- **[2026/10]** 🤖 推出 **[LoongForge-VLDA](./cookbook/loongforge-vlda/)**，基于 LoongForge 训练的 Vision-Language-Delta-Action 模型：LIBERO **98.5%**，LIBERO-Plus 零样本 **84.5%**，真机实验较 π0.5 提升 **17–18 个百分点**。[[项目主页](https://baidu-baige.github.io/LoongForge/projects/vlda.html)] [[论文](https://arxiv.org/abs/2610.25874)]
- **[2026/10]** 🦾 开源 **[ego2robot](./tools/ego2robot/)** —— 实现第一人称人类操作视频到双臂机器人训练数据的转换管线，输出 **LeRobot v3.0** 格式，覆盖 **16 种机械臂本体**。
- **[2026/09]** ✨ 新增 **[GLM-5.3-flash](./examples/glm5_next/)** 训练支持。
- **[2026/09]** ✨ 新增 **[Kimi-K3](./examples/kimi_k3/)** 的 LLM 与 VLM BF16 训练支持。
- **[2026/09]** ⚡ 新增经过优化的 **[DreamZero Wan2.2-5B FSDP recipe](./examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh)**，集成 cache-aware 数据加载、attention block 编译、冻结模块处理与 FSDP2 Delta-FP8 Param AllGather。
- **[2026/08]** 🤖 新增 **[Wall-OSS-0.5](./examples/embodied/wall_oss_0_5/)** VLA 训练支持，并通过自定义融合算子提升训练吞吐。
- **[2026/08]** 📄 发布 **[TAOT 论文](https://arxiv.org/abs/2608.03676)** —— 通过拓扑感知的动态专家副本放置，优化 **MoE** 训练中的专家并行（**EP**）负载不均衡，相较业界方案开销最大可降低 **74%**，案例实测 **1.43× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-08-taot-topology-aware-expert-placement.html)]
- **[2026/08]** ✨ 新增 **GLM-5.2** 训练支持，并提供 **[GLM-5.2 + MoonViT](./configs/models/glm5.2_vit/)** 自定义组合[示例](./examples/glm5.2_vit/)，可用于为 GLM 扩展多模态能力。
- **[2026/08]** ✨ 新增 **MiniCPM-V-4.6** 与 **Qwen3.8-27B** 训练支持。
- **[2026/08]** 🧪 新增具身模型统一[**评测模块**](./loongforge/embodied/eval/)，当前已覆盖 **Pi0.5 / xVLA / GR00T**，持续扩展中。

<details>
<summary><b>📅 更多</b></summary>

- **[2026/07]** 🐳 统一**预构建 Docker 镜像** —— LLM / VLM / 具身 / Diffusion 全部模型家族共用同一镜像。
- **[2026/07]** 🤖 发布 **[LoongForge-Embodied](./loongforge/embodied)** —— 面向具身模型（Pi0.5、GR00T-N1.6/N1.7、xVLA、LingBot-VA、FastWAM、DreamZero、Cosmos3）的 torch-native DDP/FSDP 训练子系统，实测最高 **4.38× 加速**。
- **[2026/07]** ✨ 新增 **DeepSeek-V4-Flash / DeepSeek-V4-Pro** 训练支持。
- **[2026/07]** ✨ 新增 **Qwen-Image-Edit-2511** 训练支持。
- **[2026/06]** 🤖 扩展 VLA 模型覆盖，新增 **GR00T N1.6**；GR00T 训练实现 **2.3× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html)]
- **[2026/05]** ⚡ **Wan 2.2** 训练 **加速 116%**，并新增 CP（上下文并行）与数据 packing 策略支持。
- **[2026/05]** ✨ 新增 **Kimi K2.5 / K2.6** 训练支持，并支持 **INT4 / NVFP4** PTQ 量化能力。
- **[2026/05]** 🎉 **v0.1.0** —— LoongForge 首个正式版本发布。
- **[2026/05]** 🌟 支持 **LLaVA-OneVision-2.0** 模型训练并协助其公开发布。
- **[2026/04]** 🧩 新增 **MiniMax-M2.7** 在 NVIDIA GPU 与昆仑芯 XPU 上的训练支持。
- **[2026/04]** 🚀 LoongForge 源码在 GitHub 上正式公开。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-04-announcing-loongforge.html)]
- **[2025/10]** 🌟 基于 AIAK-Training-LLM（LoongForge 前身）支持 **LLaVA-OneVision-1.5** 模型训练并协助其公开发布。[[blog](https://baidu-baige.github.io/LoongForge/blog/2025-10-llava-onevision-case-study.html)]

</details>

<a id="embodied"></a>
## 🤖 具身模型

支持视觉-语言-动作（VLA）与世界-动作（WAM）模型的**数据处理、训练与评测**，在已评测模型上训练吞吐相对开源基线提升 **1.79×–4.38×**。[[文档](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_index.html)]（论文即将发布）

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/architecture/loongforge-embodied-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/architecture/loongforge-embodied-architecture.svg">
    <img alt="LoongForge-Embodied 架构：一体化数据管线、加速模型训练（可扩展的模型接入、加速工具集、分布式训练）与一体化评测" src="./docs/assets/images/architecture/loongforge-embodied-architecture.svg" width="100%">
  </picture>
</p>

### 数据

* **统一数据加载** —— 提供统一的数据加载接口，支持 LeRobot（v2.0 / v2.1 / v3.0）等典型格式；各模型的数据处理流程与其开源实现对齐。
* **特征缓存预计算（可选）** —— 支持为 DreamZero（VAE latent、文本 embedding）和 FastWAM（文本 embedding）预先计算冻结编码器的输出，训练时直接读取，避免每步重复编码。[[DreamZero](./tools/dreamzero_data_preprocess)] [[FastWAM](./examples/embodied/fastwam/precompute_text_embeds.sh)]
* **Ego2Robot** —— 将第一人称人类操作视频转换为 **LeRobot v3.0** 训练数据，覆盖 **16 种机械臂本体**，适配各自的关节布局与夹爪。[[README](./tools/ego2robot/README_ZH.md)]

<table width="100%">
  <tr>
    <th align="center" width="30%">人类演示（输入）</th>
    <th align="center" width="70%">Ego2Robot 输出示例（多种机械臂本体）</th>
  </tr>
  <tr>
    <td><img align="top" alt="第一人称视角下的人类操作视频" src="./docs/assets/images/demo/ego2robot-human.webp" width="100%"></td>
    <td><img align="top" alt="同一动作被转换到多种机械臂本体上的合成视频" src="./docs/assets/images/demo/ego2robot-robots.webp" width="100%"></td>
  </tr>
</table>

### 训练

* **典型模型深度优化** —— 针对各模型的瓶颈做定制优化（I/O、通信策略、`torch.compile`、CUDA Graph、自定义融合算子等），相对开源基线实测 **1.79×–4.38×** 吞吐提升（见[性能表现](#performance)）。
* **精度对齐** —— 训练 loss 可与开源基线对齐，已提供多个模型的对比曲线，并随支持模型的扩展持续补充。[[精度对比](./docs/assets/images/precision/)]
* **多种分布式策略** —— 支持 **DDP / DDP + ZeRO-1 / FSDP / HSDP**，可按模型规模与集群拓扑选择。[[教程](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/overview.html)]
* **FP8 通信优化（可选）** —— 在支持的 NVIDIA GPU 上，把跨卡通信数据压缩为 FP8，降低通信量：FSDP2 对参数 AllGather 按 block 压缩 FP8 差值，DDP 对梯度 all-reduce 做 FP8 压缩。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/fp8_communication.html)]

### 评测

* **统一评测** —— 在 LIBERO / CALVIN / SimplerEnv / RoboTwin / ManiSkill 仿真环境中对训练出的策略做闭环评测；覆盖范围持续扩展中。[[README](./loongforge/embodied/eval)]

> [!NOTE]
> 敬请期待：基于 LoongForge 训练的模型评测结果即将陆续公布。

<a id="llm-vlm"></a>
## 🚀 LLM · VLM · Diffusion

基于 [Megatron 后端](https://github.com/baidu-baige/Loong-Megatron)，支持 LLM、VLM 的大规模预训练与 SFT，以及 Diffusion 模型训练。各场景的关键优化特性如下：

### LLM

* **MoE EP 通信优化** —— All2All / 激活卸载 / 计算全链路重叠，相对上游 Megatron-LM 实现**进一步降低显存占用**。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/moe_all2all_overlap.html)]
* **MoE 专家负载均衡** —— 基于拓扑感知算法动态复制热点专家，均衡专家并行（EP）负载，开销相较业界方案最大可降低 **74%**。[[TAOT 论文](https://arxiv.org/pdf/2608.03676)]
* **自适应 FP8 训练** —— 面向 LLM 和 VLM 的端到端 FP8，支持标准 **blockwise FP8**；可选**自适应**模式根据 GEMM 形状与效率逐算子选择最佳精度。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/adaptive_fp8.html)]
* **自定义融合算子** —— 面向 DSA 类模型的 **FusedDSA** 等融合 kernel：TileLang 版本已开源，高性能 CUDA 版本在百度百舸平台提供。
* **长序列训练** —— 将 LLM 训练扩展至长序列场景，依托**上下文并行**（CP）与**分块流水线调度**。

### VLM

* **灵活组合** —— 通过配置即可将可互换的 ViT 与 LLM 组件自由组装为 VLM（如 **GLM-5.2 + MoonViT**），无需编写模型代码。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/model_combination.html)]
* **异构并行** —— 针对模型不同组件（如 ViT vs LLM）独立配置 TP / DP / 重计算 / 冻结策略，获得更优的吞吐与显存占用。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)] [[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/heterogeneous_parallel.html)]
* **Encoder-Decoder 解耦训练** —— 将 ViT 与 LLM 拆分为独立任务，消除 Encoder 引入的流水线气泡。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/heterogeneous_parallel.html#full-heterogeneous-dp-parallel)]
* **DP 负载均衡** —— 基于负载感知的数据重分发，缓解序列 packing 不均衡问题，显著提升多节点扩展效率。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)] [[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/data_parallel_balancing.html)]
* **灵活的数据流水线** —— 多模态数据基于 Energon **WebDataset** 输入，支持**在线**与**离线**两种序列 packing。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/dataset_conversion.html)]

### Diffusion

* **上下文并行** —— 支持 Ring、Ulysses 及二者混合的上下文并行（CP），将长视频序列切分到多卡训练。
* **变长样本 packing** —— 将多个变长视频样本拼接为一条序列，减少 padding 浪费，并兼容上下文并行。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/wan_tutorial/wan_packing.html)]

### 通用能力

* **Mcore Bridge** —— 同时支持**离线** **Megatron ↔ HuggingFace** 双向转换与**在线**原生 HF 加载/保存。[[使用方法](https://loongforge.readthedocs.io/zh-cn/latest/features/mcore_bridge.html)]
* **异构硬件** —— 通过轻侵入式插件设计，原生支持 **NVIDIA GPU** 与**昆仑芯 XPU**。

> 📖 更多特性：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/features_index.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/features_index.html)

<a id="performance"></a>
## 📊 性能表现

下图为各模型相对主流开源基线的训练吞吐加速。每个模型与其基线均在相同机型、相同训练超参数下测得：

<p align="center">
  <img alt="LoongForge 相对开源基线的训练吞吐加速——从 Qwen3-VL 的 1.45 倍到 DeepSeek-V3.2 Lite 的 5.04 倍" src="./docs/assets/images/benchmark_speedup.png" width="860" />
</p>

> DeepSeek-V3.2 Lite 为 DSA 算子级优化的结果，受测试环境规模限制，在减层配置下验证。<br>
> 数据为特定时间点的测试结果，双方实现持续演进，数值可能随之变化。

<a id="quickstart"></a>
## ⚡ 快速开始

### 1. 安装

以下以 NVIDIA GPU 为例，拉取[预构建镜像](https://hub.docker.com/u/loongforge)并启动容器（需安装 NVIDIA Container Toolkit）：

```bash
docker pull loongforge/loongforge:latest
mkdir -p workspace
docker run --gpus all --ipc=host -it --rm \
  -v "$(pwd)/workspace:/workspace/data" \
  -w /workspace/LoongForge \
  loongforge/loongforge:latest bash
```

> 上述示例将 `./workspace` 挂载到容器内 `/workspace/data`，用于存放权重、数据集与训练输出，退出容器后仍会保留。

源码构建等更多安装方式详见：[GPU 完整安装指南](https://loongforge.readthedocs.io/zh-cn/latest/get_started/installation.html) · [昆仑芯 XPU 安装指南](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/install_p800.html)

### 2. 按硬件与模态选择教程

- **NVIDIA GPU**：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/quick_start_llm_pretrain.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/quick_start_vlm_pretrain.html) · [VLA & WAM（具身）](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_index.html) · [Diffusion](https://loongforge.readthedocs.io/zh-cn/latest/wan_tutorial/quick_start_wan_training.html)
- **昆仑芯 XPU**：[昆仑芯 XPU 教程](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/README.html)

### 3. 任务启动示例

当前仓库已内置了各模型的启动示例脚本：NVIDIA GPU 脚本位于 [`examples/`](./examples/)，昆仑芯 XPU 脚本位于 [`examples_xpu/`](./examples_xpu/)，具体可查看[支持的模型](#models)。

以下以 **Pi0.5 在 LIBERO 上的 DDP 微调（单机 8 GPU）** 为例说明使用方法，以下命令均在上述容器内执行。

**① 准备训练所需的权重、tokenizer 与数据集**（`hf` 为 Hugging Face CLI）：

```bash
hf download lerobot/pi05_base --local-dir /workspace/data/pi05_base
hf download google/paligemma-3b-pt-224 --local-dir /workspace/data/paligemma-3b-pt-224
hf download lerobot/libero_10 --repo-type dataset --local-dir /workspace/data/libero_10
```

**② 启动训练**（通过环境变量指定路径，默认跑 20 步用于验证流程）：

```bash
export CHECKPOINT_PATH=/workspace/data/pi05_base
export TOKENIZER_PATH=/workspace/data/paligemma-3b-pt-224
export DATA_PATH=/workspace/data/libero_10
export OUTPUT_DIR=/workspace/data/outputs/pi05

bash examples/embodied/pi05/run_pi05_ddp_finetune.sh
```

日志与 TensorBoard 数据保存在 `OUTPUT_DIR` 下。FSDP / ZeRO-1 启动方式与 LIBERO 评测见 [Pi0.5 教程](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_pi05.html)。

<a id="models"></a>
## 🏛️ 支持的模型

点击任意模型查看训练示例，对应的模型参数配置详见 [`configs/models/`](./configs/models/)；全部变体见[模型支持矩阵](https://loongforge.readthedocs.io/zh-cn/latest/get_started/support_model.html)。

<table width="100%">
<colgroup>
<col width="25%">
<col width="25%">
<col width="25%">
<col width="25%">
</colgroup>
<thead align="center" valign="bottom">
<tr><th width="25%">Embodied</th><th width="25%">LLM</th><th width="25%">VLM</th><th width="25%">Diffusion</th></tr>
</thead>
<tbody valign="top">
<tr>
<td valign="top">
<ul>
<li><a href="examples/embodied/pi05/">Pi0.5</a></li>
<li><a href="examples/embodied/groot_n1_6/">GR00T-N1.6</a></li>
<li><a href="examples/embodied/groot_n1_7/">GR00T-N1.7</a></li>
<li><a href="examples/embodied/xvla/">xVLA</a></li>
<li><a href="examples/embodied/wall_oss_0_5/">Wall-OSS-0.5</a></li>
<li><a href="examples/embodied/fastwam/">FastWAM</a></li>
<li><a href="examples/embodied/lingbot_va/">LingBot-VA</a></li>
<li><a href="examples/embodied/cosmos3/">Cosmos3</a></li>
<li><a href="examples/embodied/dreamzero/">DreamZero</a></li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/deepseek_v2/">DeepSeek-V2</a></li>
<li><a href="examples/deepseek_v3/">DeepSeek-V3/V3.2</a></li>
<li><a href="examples/deepseek_v4/">DeepSeek-V4</a></li>
<li><a href="examples/llama2/">LLaMA2</a></li>
<li><a href="examples/llama3/">LLaMA3</a></li>
<li><a href="examples/llama3.1/">LLaMA3.1</a></li>
<li><a href="examples/qwen/">Qwen</a></li>
<li><a href="examples/qwen1.5/">Qwen1.5</a></li>
<li><a href="examples/qwen2/">Qwen2</a></li>
<li><a href="examples/qwen2.5/">Qwen2.5</a></li>
<li><a href="examples/qwen3/">Qwen3</a></li>
<li><a href="examples/qwen3_next/">Qwen3-Next</a></li>
<li><a href="examples/minimax/">MiniMax-M2.1/2.5/2.7</a></li>
<li><a href="examples/mimo/">MIMO</a></li>
<li><a href="examples/glm5/">GLM-5</a></li>
<li><a href="examples/glm5.2/">GLM-5.2</a></li>
<li><a href="examples/kimi_k3/">Kimi-K3</a></li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/qwen2.5_vl/">Qwen2.5-VL</a></li>
<li><a href="examples/qwen3_vl/">Qwen3-VL</a></li>
<li><a href="examples/qwen3.5/">Qwen3.5</a></li>
<li><a href="examples/qwen3.6/">Qwen3.6</a></li>
<li><a href="examples/qwen3.8/">Qwen3.8</a></li>
<li><a href="examples/kimi_k2.x/kimi_k2.5/">Kimi-K2.5/2.6</a></li>
<li><a href="examples/kimi_k3/">Kimi-K3</a></li>
<li><a href="examples/minicpm_v_4_6/">MiniCPM-V-4.6</a></li>
<li><a href="examples/glm5.2_vit/">GLM-5.2 + MoonViT</a></li>
<li><a href="examples/glm5_next/">GLM-5.3-flash</a></li>
<li><a href="examples/ernie4.5/">ERNIE4.5-VL</a></li>
<li><a href="examples/llava_onevision_1.5/">LLaVA-OneVision-1.5</a></li>
<li><a href="examples/internvl2.5/">InternVL2.5</a></li>
<li><a href="examples/internvl3.5/">InternVL3.5</a></li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/wan/">Wan2.1</a></li>
<li><a href="examples/wan/">Wan2.2</a></li>
<li><a href="examples/qwen_image/">Qwen-Image-Edit-2511</a></li>
</ul>
</td>
</tr>
</tbody>
</table>

> [!TIP]
> VLM 还可参考[自定义组合示例](./examples/custom/)，用配置将 ViT 与 LLM 自由组装为新模型。

<a id="powered-by-loongforge"></a>
## 🌟 基于 LoongForge 的优秀项目案例

更多项目见[项目展示页](https://baidu-baige.github.io/LoongForge/projects.html)。如果您也基于 LoongForge（或其前身 AIAK-Training-LLM）做了项目，欢迎[告诉我们](https://github.com/baidu-baige/LoongForge/issues/new)，我们很乐意收录。

### 社区项目

- [**LLaVA-OneVision-2.0**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) —— 新一代多模态模型，配套全新的 VideoCaption 与 Spatial 数据集
- [**Innovator-VL**](https://github.com/InnovatorLM/Innovator-VL/tree/main) —— 面向高级推理的科学多模态大模型
- [**LLaVA-OneVision-1.5**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/1.5) —— 旨在降低多模态训练门槛的全开源框架
- [**Qianfan-VL**](https://github.com/baidubce/Qianfan-VL) —— 面向企业的领域增强视觉-语言模型，参数量覆盖 3B ~ 70B

### LoongForge 团队项目

- [**LoongForge-VLDA**](./cookbook/loongforge-vlda/) —— 面向机器人操作的 Vision-Language-Delta-Action 模型：LIBERO **98.5%**，零样本 LIBERO-Plus **84.5%**，真机较 π₀.₅ 提升 **17–18 个百分点**。[[项目主页](https://baidu-baige.github.io/LoongForge/projects/vlda.html)] [[论文](https://arxiv.org/abs/2610.25874)]

https://github.com/user-attachments/assets/8563afca-f79a-4a45-bb4f-36e06c0ccbcb

<a id="blogs"></a>
## 📝 社区相关博客

- [VLA 训练如何一步步提速：以 Pi0.5 为例的 3 倍加速实践](https://zhuanlan.zhihu.com/p/2095555587304117777)
- [百度百舸 LoongForge 团队提出 TAOT：拓扑感知传输破解大规模 MoE 训练端到端加速 1.43 倍](https://mp.weixin.qq.com/s/Op-KpFe5G1Q2IxzIUgm2KA)
- [加速 GR00T N1.6 具身 VLA 模型训练！LoongForge 开源框架实现三层全链路优化，吞吐提升 2.3 倍、训练周期缩短 56.6%](https://mp.weixin.qq.com/s/lIMsQ0Q5G9KaDKGtrBt-Nw)
- [多模态模型终于不用「抽帧看世界」？LLaVA-OneVision-2.0 全开源全帧率技术解读](https://mp.weixin.qq.com/s/dechFptcXCvF30mfmC3dWA)
- [LoongForge 多模态异构并行训练加速，从问题到方案的全面解析](https://mp.weixin.qq.com/s/ZFWSxNZvDZJ38fyyb8t7vg)
- [百度百舸开源全模态训练框架 LoongForge：一套代码跑通 GPU 与昆仑芯，多模态训练提速 45%](https://zhuanlan.zhihu.com/p/2031006068797600446)
- [刷新记录！百度百舸助力 LLaVA-OneVision-1.5 高效训练](https://mp.weixin.qq.com/s/mdH3vUStOB13ME4aHx3O8w)

## 📂 代码结构

<details>
<summary><b>📁 目录树</b></summary>

```
LoongForge/
├── loongforge/                   # 核心训练框架
│   ├── train/                    # 训练入口与训练器
│   │   ├── pretrain/             #   预训练
│   │   ├── sft/                  #   监督微调（SFT）
│   │   └── diffusion/            #   Diffusion 训练
│   ├── models/                   # 统一的模型抽象层
│   │   ├── foundation/           #   LLM 主干
│   │   ├── encoder/              #   视觉编码器
│   │   ├── omni_models/          #   多模态组合（编码器 + 投影层 + 解码器）
│   │   ├── diffusion/            #   Diffusion 模型
│   │   └── common/               #   公共基类配置、Layer 与工具
│   ├── embodied/                 # 具身（VLA + 世界-动作）训练：模型、数据、训练器、分布式与评测
│   ├── data/                     # 数据加载与处理流水线
│   ├── tokenizer/                # Tokenizer
│   └── utils/                    # 配置映射、常量与公共工具
├── third_party/                  # 第三方依赖与补丁
│   ├── Loong-Megatron/            # Patched Megatron-LM（git submodule）
│   └── patches/                  # TransformerEngine 补丁
├── configs/                      # Hydra YAML 配置（模型、数据）
├── examples/                     # NVIDIA GPU 启动脚本
├── examples_xpu/                 # 昆仑芯 XPU 启动脚本
├── cookbook/                     # 基于 LoongForge 的端到端项目复现方案
├── tools/                        # 独立开发工具：Checkpoint 转换、数据预处理等（详见 tools/README.md）
├── ops/                          # 自定义融合算子（含开源的 TileLang 版本）
├── docker/                       # Dockerfile（GPU & XPU）
├── tests/                        # 测试套件（端到端、回归、单元）
└── docs/                         # 文档
```

</details>

## 📝 引用

如果您觉得 LoongForge 对您的工作有帮助，欢迎引用本项目：

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A high-performance framework for training LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

如果您觉得 TAOT 对您的工作有帮助，欢迎引用我们的论文：

```bibtex
@article{zhang2026taot,
  title   = {{TAOT}: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in {MoE} Training},
  author  = {Zhang, Lingyun and Zhang, Henghua and Gu, Shilei and Mo, Kai and Han, Shuai and Li, Shiyong and Wang, Yanpeng and Shen, Dou},
  journal = {arXiv preprint arXiv:2608.03676},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03676}
}
```

如果您觉得 LoongForge-VLDA 对您的工作有帮助，欢迎引用我们的论文：

```bibtex
@article{zhang2026vlda,
  title   = {{Vision-Language-Delta-Action}: Learning Compact Deltas as Actionable Transition Interfaces for Robotic Manipulation},
  author  = {Zhang, Lingsen and Li, Wei and He, Jie and Zhang, Henghua and Mo, Kai and Li, Shiyong and Wang, Yanpeng and Shao, Rui},
  journal = {arXiv preprint arXiv:2610.25874},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.25874}
}
```

## 🤝 参与贡献

我们非常欢迎社区贡献 —— 无论是 Bug 报告、功能提案还是 PR。在提交前请阅读 [贡献指南](./CONTRIBUTING.md)。

非常感谢 LoongForge 的所有贡献者：

<a href="https://github.com/baidu-baige/LoongForge/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=baidu-baige/LoongForge&v=2026-09-04" alt="LoongForge contributors" />
</a>

<a id="contact"></a>
## 💬 联系我们

| 渠道 | 适用场景 |
|------|----------|
| [**GitHub Issue**](https://github.com/baidu-baige/LoongForge/issues/new/choose) | 问题反馈、使用疑问与功能建议 |
| [**开发者社区**](https://github.com/baidu-baige/LoongForge/issues/80) | 微信群、小红书等 |
| [**邮件**](mailto:loongforge@baidu.com) | 企业落地、大规模部署、商务合作，以及任何其他话题 |

## 🙏 致谢

LoongForge 的成长离不开开源社区。其 Megatron 后端以 NVIDIA 的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 为基础，项目也从 [HuggingFace Transformers](https://github.com/huggingface/transformers)、[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)、[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)、[LeRobot](https://github.com/huggingface/lerobot) 以及所支持模型的官方实现（如 [OpenPI](https://github.com/Physical-Intelligence/openpi)、[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)）中汲取了经验。

同时也特别感谢 [LINUX DO](https://linux.do/) 社区为技术交流提供友善的空间，以及 [AtomGit](https://atomgit.com/baidu-baige/LoongForge) 为 LoongForge 提供官方镜像托管。

## 📄 开源协议

LoongForge 基于 [Apache License 2.0](./LICENSE) 发布；部分文件改编自第三方项目，详见各文件头部。
