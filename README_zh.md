<p align="right"><sub><a href="./README.md">English</a> | <b>简体中文</b></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="520">
  </picture>
</p>

<h4>面向 LLM、VLM、Diffusion 与 Embodied 模型的模块化、可扩展、高性能训练框架。</h4>

<p align="center">
  
[![Home](https://img.shields.io/badge/LoongForge-主页-8A2CE3?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0IiByb2xlPSJpbWciIGFyaWEtbGFiZWw9Ikxvb25nRm9yZ2UgbG9nbyI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImciIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzYzNjZGMSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjYwJSIgc3RvcC1jb2xvcj0iIzhCNUNGNiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiNGNTlFMEIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHg9IjIiIHk9IjIiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcng9IjE0IiBmaWxsPSJ1cmwoI2cpIi8+CiAgPHBhdGggZD0iTTE4IDQwIEMgMjIgMzAsIDI4IDI4LCAzMiAzMiBDIDM2IDM2LCA0MiAzNCwgNDYgMjQiCiAgICAgICAgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjMuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBmaWxsPSJub25lIi8+CiAgPGNpcmNsZSBjeD0iNDYiIGN5PSIyNCIgcj0iMy4yIiBmaWxsPSIjZmZmIi8+CiAgPGNpcmNsZSBjeD0iMTgiIGN5PSI0MCIgcj0iMi4yIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPHBhdGggZD0iTTI0IDQ2IEw0MCA0NiIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iMC43Ii8+Cjwvc3ZnPgo=)](https://baidu-baige.github.io/LoongForge/)
[![Docs](https://img.shields.io/badge/文档-Latest-00A3FF?logo=readthedocs)](https://loongforge.readthedocs.io/zh-cn/latest/index.html)
[![Blog](https://img.shields.io/badge/博客-View-FF6B35.svg?logo=github)](https://baidu-baige.github.io/LoongForge/blog/)
[![Release](https://img.shields.io/github/v/release/baidu-baige/LoongForge?include_prereleases&label=release&color=blue)](https://github.com/baidu-baige/LoongForge/releases)
[![License](https://img.shields.io/github/license/baidu-baige/LoongForge.svg?logo=github)](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-加入-4A154B.svg?logo=slack)](https://join.slack.com/t/baiduloongforge/shared_invite/zt-3ys3kaq2p-cmdw0nDoaHGOcKibgys5Yw)
[![WeChat](https://img.shields.io/badge/WeChat-Join-07C160.svg?logo=wechat)](https://github.com/baidu-baige/LoongForge/issues/80#issue-4594463290)

</p>

<p align="center">
  <b>🚀 训练加速最高可达 5.04×</b> &nbsp;·&nbsp;
  <b>🌐 原生支持 NVIDIA GPU 与昆仑芯 XPU</b>
</p>

<p align="center">
  <a href="#quickstart"><b>📖 快速开始</b></a>
  &nbsp;·&nbsp;
  <a href="#benchmark"><b>📊 性能数据</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>🤖 支持模型</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/baidu-baige/LoongForge/issues/74"><b>🚀 路线图</b></a>
</p>

</div>

## 💡 为什么选择 LoongForge？

> 🐉 LoongForge 是百度百舸 **Loong** 开源系列的一员 —— 名字源于中国传统 **龙舟**，象征协同发力与破浪前行。

<img width="2834" height="1160" alt="loongforge-overview-notitle1" src="https://github.com/user-attachments/assets/4bc9d3f2-914c-4c01-91dc-10d22c08e02f" />

**LoongForge** 是面向 **LLM、VLM、Diffusion 与 Embodied 模型** 的统一训练框架，覆盖 **预训练（Pre-training）**、**持续预训练（Continued Pre-training）** 和 **SFT**。基于 Megatron-LM 在 **模型覆盖度**、**训练性能** 和 **硬件支持** 三个维度做了深度系统性增强，相对主流开源训练方案有**显著的性能提升**。

在开源之前，LoongForge 的前身是 **AIAK-Training-LLM** —— 百度百舸的训练加速栈，已在 **教育**、**计算机视觉** 和 **Embodied AI** 等多家企业客户的生产训练中落地，相对客户原有方案通常带来 **30%~50% 加速**，最大规模的生产训练任务达到 **5,000+ XPU**。

## 🔥 最新动态

- **[2026/05]** ⚡ **Wan 2.2** 训练 **加速 116%**，并新增 CP（上下文并行）与数据 packing 策略支持。
- **[2026/05]** ✨ 新增 **Kimi K2.5 / K2.6** 训练支持，并支持 **INT4 / NVFP4** PTQ 量化能力。
- **[2026/05]** 🎉 **v0.1.0** —— LoongForge 首个正式版本发布。
- **[2026/05]** 🌟 支持 **LLaVA-OneVision-2.0** 模型训练并协助其公开发布。
- **[2026/05]** 🤖 扩展 VLA 模型覆盖，新增 **GR00T N1.6**；Pi0.5 与 GR00T 训练实现 **60%+ 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html)]
- **[2026/04]** 🧩 新增 **MiniMax-M2.7** 在 NVIDIA GPU 与昆仑芯 XPU 上的训练支持。
- **[2026/04]** 🚀 LoongForge 源码在 GitHub 上正式公开。[[blog]](https://zhuanlan.zhihu.com/p/2031006068797600446)
- **[2025/10]** 🌟 基于AIAK-Training-LLM（LoongForge 前身）支持 **LLaVA-OneVision-1.5** 模型训练并协助其公开发布。[[blog]](https://mp.weixin.qq.com/s/1y7Br15pBpUZ-90j5OGncA)

<a id="quickstart"></a>
## ⚡ 快速开始

完整的安装、教程与进阶使用请查阅文档 —— [English](https://loongforge.readthedocs.io/en/latest/index.html) · [中文](https://loongforge.readthedocs.io/zh-cn/latest/index.html)。

**1. 安装** —— 可使用 **预构建镜像** 或 **源码构建**：
- **NVIDIA GPU**：[安装指南](https://loongforge.readthedocs.io/zh-cn/latest/get_started/installation.html)
- **昆仑芯 XPU**：[安装指南](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/install_p800.html)

**2. 启动你的第一个训练任务** —— 根据目标硬件与模态选择教程：
- **NVIDIA GPU**：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/quick_start_llm_pretrain.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/quick_start_vlm_pretrain.html) · [VLA](https://loongforge.readthedocs.io/zh-cn/latest/vla_tutorial/quick_start_pi05_training.html) · [Diffusion (WAN)](https://loongforge.readthedocs.io/zh-cn/latest/wan_tutorial/quick_start_wan_training.html)
- **昆仑芯 XPU**：[昆仑芯 XPU 教程](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/README.html)

**3. 深入探索** —— 浏览 [`configs/models/`](./configs/models) 和 [`examples/`](./examples) / [`examples_xpu/`](./examples_xpu) 下的现成启动脚本。

## ✨ 核心特性

* **🧩 灵活的多模态组合** —— 通过配置驱动的方式，将可互换的 ViT 与 LLM 组件自由组装为 VLM。
* **⚡ 异构并行** —— 针对模型不同组件（如 ViT vs LLM）独立配置 TP / DP / 重计算策略，获得最优吞吐与显存占用。 [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)]
* **🔀 Encoder-Decoder 解耦训练** —— 将 ViT 与 LLM 拆分为独立任务，消除 Encoder 带来的流水线气泡。
* **⚖️ DP 负载均衡** —— 基于负载感知的数据重分发，缓解序列打包（sequence packing）不均衡问题，显著提升多节点扩展效率。 [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)]
* **🚀 MoE 原生优化** —— All2All / 激活卸载 / 计算全链路重叠，在 DeepSeek-V3、Qwen3-MoE 等模型上相对上游 Megatron-LM 实现**进一步显存降低**。
* **🔬 自适应 FP8 训练** —— 面向 LLM 和 VLM 的端到端 FP8，支持标准 **blockwise FP8**；可选 **自适应** 模式根据 GEMM 形状与效率逐算子选择最佳精度。
* **🔧 自定义融合算子** —— 为 DSA 类模型设计的 **FusedDSA** 等融合 Kernel —— TileLang 版本已开源，高性能 CUDA 版本在百度百舸平台提供。
* **🔁 灵活的 Checkpoint 机制** —— 支持离线 **Megatron ↔ HuggingFace** 双向转换，以及在线原生 HF 加载/保存，全流程无格式壁垒。
* **🧰 丰富的流水线与数据工具** —— 开箱即用的 **Pretrain / MidTrain / SFT / LoRA** 流水线，内置数据集格式转换与序列打包能力。
* **🌐 异构硬件** —— 通过轻侵入式插件设计，原生支持 **NVIDIA GPU** 与 **昆仑芯 XPU**。

> 📖 深入阅读：[LLM 特性](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/features_index.html) · [VLM 特性](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/features_index.html)

<a id="benchmark"></a>
## 📊 性能 Benchmark

在 **v0.1.1** 版本上针对 LLM、VLM、VLA、DIT 四类工作负载，与主流开源训练方案的对比结果：
<img alt="LoongForge Benchmark Speedup" src="docs/assets/images/benchmark_speedup.png" />

<details>
<summary><b>📋 细节描述</b></summary>

<br>

| 模型 | 类型 | 对比基线 | 配置 | 加速比 |
|---|---|---|---|---|
| Qwen3-30B-A3B | MoE | Megatron-LM<sup>†</sup> | 32 × A800<sup>‡</sup> · GBS 1024 · 32K | **1.16×** |
| DeepSeek-V3.2 Lite <sup>§</sup> | MoE + DSA | Megatron-LM<sup>†</sup> | 减层配置 · GBS 128 · 8K 序列 | **5.04×** |
| Qwen3-VL-30B-A3B | VLM | VeOmni<sup>†</sup> | 32 × A800<sup>‡</sup> · GBS 128 · 32K | **1.45×** |
| GR00T N1.6 | VLA | LeRobot<sup>†</sup> | 8 × A800<sup>‡</sup> · GBS 128 · 224×224 | **2.31×** |
| Pi0.5 | VLA | OpenPI<sup>†</sup> | 8 × A800<sup>‡</sup> · GBS 112 · 224×224 | **1.65×** |

> <sup>§</sup> 受测试台规模限制，**DeepSeek-V3.2** 在减层配置下单独验证 —— LoongForge 的 **DSA CUDA Kernel 优化** 相对 Megatron-LM 仍带来 **~5× 加速**，并可支持 **64K 序列长度**（基线在 8K 以上即 OOM）。<br>
> <sup>†</sup> 数据反映测量时对应基线的实现，后续可能随实现演进而变化。<br>
> <sup>‡</sup> 更多硬件平台的验证将在后续版本中陆续推出。<br>
</details>

## 🌟 基于 LoongForge 训练

- [LLaVA-OneVision-2.0](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) —— 新一代多模态模型，配套全新的 VideoCaption 和 Spatial 数据集。
- [LLaVA-OneVision-1.5](https://arxiv.org/abs/2509.23661) —— 面向多模态训练民主化的全开源框架。
- [Qianfan-VL](https://github.com/baidubce/Qianfan-VL) —— 面向企业的领域增强视觉-语言模型，参数量覆盖 3B ~ 70B。

<a id="models"></a>
## 🏛️ 支持的模型

LoongForge 已支持 LLM、VLM、Diffusion 与 VLA 等多模态的[广泛的 SOTA 模型](https://loongforge.readthedocs.io/zh-cn/latest/get_started/support_model.html)。

| **模态** | **架构** | **模型** |
|---------------|------------------|------------|
| **LLM** | DeepSeek-V2 | deepseek-v2-lite, deepseek-v2 |
| | DeepSeek-V3 | deepseek-v3, deepseek-v32 |
| | LLaMA2 | llama2-7b, llama2-13b, llama2-70b |
| | LLaMA3 | llama3-8b, llama3-70b |
| | LLaMA3.1 | llama3.1-8b, llama3.1-70b, llama3.1-405b |
| | Qwen | qwen-1.8b → qwen-72b |
| | Qwen1.5 | qwen1.5-0.5b → qwen1.5-72b |
| | Qwen2 | qwen2-0.5b → qwen2-72b |
| | Qwen2.5 | qwen2.5-0.5b → qwen2.5-72b |
| | Qwen3 | qwen3-0.6b → qwen3-480b-a35b, qwen3-coder-30b-a3b |
| | Qwen3-Next | qwen3-next-80b-a3b |
| | MiniMax | minimax-m2.1, minimax-m2.5, minimax-m2.7 |
| | MIMO | mimo-7b |
| | GLM | glm5 |
| **VLM** | Qwen2.5-VL | qwen2.5-vl-3b → qwen2.5-vl-72b |
| | Qwen3-VL | qwen3-vl-30b-a3b, qwen3-vl-235b-a22b |
| | Qwen3.5 | qwen3.5-0.8b → qwen3.5-397b-a17b |
| | Qwen3.6 | qwen3.6-27b, qwen3.6-35b-a3b |
| | Kimi-K2.5 | kimi-k2.5, kimi-k2.6 |
| | ERNIE4.5-VL | ernie4.5vl-28b-a3b |
| | LLaVA-OneVision-1.5 | llava-onevision-1.5-4b |
| | InternVL2.5 | internvl2.5-8b → internvl2.5-78b |
| | InternVL3.5 | internvl3.5-8b → internvl3.5-241b-a28b |
| | CustomCombinedModel | ViT + LLM backbone 灵活组合（[示例](https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml)） |
| **Diffusion** | WAN2.2 | wan2.2_i2v_a14b |
| **VLA** | Pi | pi0.5 |
| | GR00T | groot-n1.6 |

## 🏗️ 代码结构

<details>
<summary><b>📁 目录树</b></summary>

```
LoongForge/
├── loongforge/                   # 核心训练框架
│   ├── train/                    # 训练入口与训练器
│   │   ├── pretrain/             #   预训练（LLM、VLM）
│   │   ├── sft/                  #   SFT（LLM、VLM、InternVL、ERNIE）
│   │   ├── diffusion/            #   Diffusion（WAN）
│   │   └── embodied/             #   Embodied AI（Pi0.5、GR00T）
│   ├── models/                   # 统一的模型抽象层
│   │   ├── foundation/           #   LLM 主干（LLaMA、Qwen、DeepSeek、...）
│   │   ├── encoder/              #   视觉编码器（ViT、Qwen-VL、InternVL、...）
│   │   ├── omni_models/          #   多模态组合
│   │   ├── diffusion/            #   Diffusion 模型（WAN）
│   │   ├── embodied/             #   Embodied 模型（Pi0.5、GR00T）
│   │   └── common/               #   公共 Layer 与工具
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

## 🤝 参与贡献

我们非常欢迎社区贡献 —— 无论是 Bug 报告、功能提案还是 PR。在提交前请阅读 [贡献指南](https://github.com/baidu-baige/LoongForge/blob/master/CONTRIBUTING.md)。

## 📄 开源协议

LoongForge 基于 [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE) 发布。部分源文件改编自第三方开源项目，请以各文件头部标注的版权与署名信息为准。

## 📝 引用

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A modular, scalable, high-performance training framework for LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

## 🙏 致谢

LoongForge 构建于 NVIDIA 的 Megatron-LM 之上，同时也从 HuggingFace Transformers、LLaMA-Factory、Megatron-Bridge 等更多优秀开源项目中汲取了灵感。衷心感谢这些社区所做的杰出贡献。

## 💬 联系我们
<a id="contact"></a>

欢迎通过 GitHub Issue 提交问题、反馈或功能建议，也可以加入我们的开发者社区：

- **微信群** — [扫码加入](https://github.com/baidu-baige/LoongForge/issues/80#issue-4594463290)
- **Slack** — [点击加入](https://join.slack.com/t/baiduloongforge/shared_invite/zt-3ys3kaq2p-cmdw0nDoaHGOcKibgys5Yw)
