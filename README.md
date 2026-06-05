<p align="right"><sub><b>English</b> | <a href="./README_zh.md">简体中文</a></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="520">
  </picture>
</p>

<h4>A modular, scalable, high-performance training framework for LLMs, VLMs, diffusion, and embodied models.</h4>

<p align="center">

[![Home](https://img.shields.io/badge/LoongForge-Home-8A2CE3?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0IiByb2xlPSJpbWciIGFyaWEtbGFiZWw9Ikxvb25nRm9yZ2UgbG9nbyI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImciIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzYzNjZGMSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjYwJSIgc3RvcC1jb2xvcj0iIzhCNUNGNiIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiNGNTlFMEIiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgPC9kZWZzPgogIDxyZWN0IHg9IjIiIHk9IjIiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcng9IjE0IiBmaWxsPSJ1cmwoI2cpIi8+CiAgPHBhdGggZD0iTTE4IDQwIEMgMjIgMzAsIDI4IDI4LCAzMiAzMiBDIDM2IDM2LCA0MiAzNCwgNDYgMjQiCiAgICAgICAgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjMuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBmaWxsPSJub25lIi8+CiAgPGNpcmNsZSBjeD0iNDYiIGN5PSIyNCIgcj0iMy4yIiBmaWxsPSIjZmZmIi8+CiAgPGNpcmNsZSBjeD0iMTgiIGN5PSI0MCIgcj0iMi4yIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPHBhdGggZD0iTTI0IDQ2IEw0MCA0NiIgc3Ryb2tlPSIjZmZmIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgb3BhY2l0eT0iMC43Ii8+Cjwvc3ZnPgo=)](https://baidu-baige.github.io/LoongForge/)
[![Docs](https://img.shields.io/badge/Docs-Latest-00A3FF?logo=readthedocs)](https://loongforge.readthedocs.io/en/latest/index.html)
[![Blog](https://img.shields.io/badge/Blog-View-FF6B35.svg?logo=github)](https://baidu-baige.github.io/LoongForge/blog/)
[![Release](https://img.shields.io/github/v/release/baidu-baige/LoongForge?include_prereleases&label=release&color=blue)](https://github.com/baidu-baige/LoongForge/releases)
[![License](https://img.shields.io/github/license/baidu-baige/LoongForge.svg?logo=github)](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE)
[![Slack](https://img.shields.io/badge/Slack-Join-4A154B.svg?logo=slack)](https://join.slack.com/t/baiduloongforge/shared_invite/zt-3ys3kaq2p-cmdw0nDoaHGOcKibgys5Yw)
[![WeChat](https://img.shields.io/badge/WeChat-Join-07C160.svg?logo=wechat)](#contact)

</p>

<p align="center">
  <b>🚀 Up to 5.04× training speedup</b> &nbsp;·&nbsp;
  <b>🌐 Native NVIDIA GPU & Kunlun XPU support</b>
</p>

<p align="center">
  <a href="#quickstart"><b>📖 Quick Start</b></a>
  &nbsp;·&nbsp;
  <a href="#benchmark"><b>📊 Benchmark</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>🤖 Supported Models</b></a>
  &nbsp;·&nbsp;
  <a href="https://github.com/baidu-baige/LoongForge/issues/74"><b>🚀 Roadmap</b></a>
</p>

</div>

## 💡 Why LoongForge?

> 🐉 LoongForge is part of Baidu Baige's **Loong** open-source series — named after the traditional Chinese **loong boat (龙舟)**, a symbol of coordinated power and forward momentum.

**LoongForge** is a unified training framework for **LLMs, VLMs, diffusion, and embodied models**, covering **pre-training**, **continued pre-training**, and **SFT**. Built upon Megatron-LM with deep systemic enhancements across **model coverage**, **training performance**, and **hardware support**, it delivers **significant speedups over mainstream open-source baselines**.

Before going open-source, LoongForge was developed as **AIAK-Training-LLM**, Baidu Baige's training acceleration stack. It has supported production training for enterprise customers across **Education**, **Computer Vision**, and **Embodied AI**, typically delivering **30%~50% speedup over customer baselines**, with the largest production runs reaching **5,000+ XPUs**.

## 🔥 Latest News

- **[2026/05]** ⚡ Accelerated **Wan 2.2** training by **116%**, and added CP and data packing support.
- **[2026/05]** ✨ Added training support for **Kimi K2.5 / K2.6**, and introduced **INT4 / NVFP4** PTQ.
- **[2026/05]** 🎉 **v0.1.0** — first official tagged release of LoongForge.
- **[2026/05]** 🌟 Powered the training and public release of **LLaVA-OneVision-2.0**.
- **[2026/05]** 🤖 Expanded VLA coverage with **GR00T N1.6**; **60%+ speedup** on Pi0.5 and GR00T training.
- **[2026/04]** 🧩 Added training support for **MiniMax-M2.7** on both NVIDIA GPU and Kunlun XPU.
- **[2026/04]** 🚀 LoongForge source code publicly available on GitHub. [[blog]](https://baidu-baige.github.io/LoongForge/blog/2026-04-announcing-loongforge.html)
- **[2025/10]** 🌟 Powered the training and public release of **LLaVA-OneVision-1.5** under **AIAK-Training-LLM**, the predecessor of LoongForge. [[blog]](https://baidu-baige.github.io/LoongForge/blog/2025-10-llava-onevision-case-study.html)

<a id="quickstart"></a>
## ⚡ Quick Start

See the full documentation for installation, tutorials, and advanced usage — [English](https://loongforge.readthedocs.io/en/latest/index.html) · [中文](https://loongforge.readthedocs.io/zh-cn/latest/index.html).

**1. Install** — via [**Docker**](./docker) (*prebuilt images coming soon*) or **source build**:
- **NVIDIA GPU**: [Installation Guide](https://loongforge.readthedocs.io/en/latest/get_started/installation.html)
- **Kunlun XPU**: [Installation Guide](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/install_p800.html)

**2. Launch your first training run** — follow a tutorial for your target hardware and modality:
- **NVIDIA GPU**: [LLM](https://loongforge.readthedocs.io/en/latest/llm_tutorial/quick_start_llm_pretrain.html) · [VLM](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html) · [VLA](https://loongforge.readthedocs.io/en/latest/vla_tutorial/quick_start_pi05_training.html) · [Diffusion (WAN)](https://loongforge.readthedocs.io/en/latest/wan_tutorial/quick_start_wan_training.html)
- **Kunlun XPU**: [Kunlun XPU Tutorials](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/README.html)

**3. Explore** — browse [`configs/models/`](./configs/models) and [`examples/`](./examples) / [`examples_xpu/`](./examples_xpu) for ready-to-run scripts.

## ✨ Key Features

* **🧩 Flexible Multi-Modal Composition** — Configuration-driven assembly of VLMs from interchangeable ViT and LLM components.
* **⚡ Heterogeneous Parallelism** — Independent TP / DP / recompute per model component (e.g., ViT vs. LLM) for optimal throughput and memory. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)]
* **🔀 Decoupled Encoder-Decoder Training** — Separates ViT and LLM into independent tasks, eliminating encoder-induced pipeline bubbles.
* **⚖️ DP Load Balancing** — Load-aware data redistribution mitigates sequence-packing imbalance, improving multi-node scaling efficiency. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)]
* **🚀 MoE-Native Optimization** — Overlapped All2All / activation offload / compute, with **further memory reduction** beyond upstream Megatron-LM on DeepSeek-V3, Qwen3-MoE, etc.
* **🔬 Adaptive FP8 Training** — End-to-end FP8 for LLMs and VLMs with standard **blockwise FP8**; optional **adaptive** mode picks per-operator precision by GEMM shape and efficiency.
* **🔧 Custom Fused Operators** — Fused kernels like **FusedDSA** for DSA-style models — TileLang version open-sourced, high-performance CUDA version available on Baidu Baige platform.
* **🔁 Flexible Checkpointing** — Offline bidirectional **Megatron ↔ HuggingFace** conversion plus native online HF load/save — no format barriers across your workflow.
* **🧰 Versatile Pipelines & Data Tools** — Out-of-the-box **Pretrain / MidTrain / SFT / LoRA**, with built-in dataset format conversion and sequence packing.
* **🌐 Heterogeneous Hardware** — Native support for **NVIDIA GPUs** and **Kunlun XPUs** via a minimally-intrusive plugin design.

> 📖 Deep-dive: [LLM features](https://loongforge.readthedocs.io/en/latest/llm_tutorial/features_index.html) · [VLM features](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/features_index.html)

<a id="benchmark"></a>
## 📊 Benchmark

Measured on **v0.1.1** across LLM, VLM, VLA and DIT workloads against mainstream open-source training baselines:

<img alt="LoongForge Benchmark Speedup" src="docs/assets/images/benchmark_speedup.png" />

<details>
<summary><b>📋 Detailed configurations & footnotes</b></summary>

<br>

| Model | Type | Baseline | Configuration | Speedup |
|---|---|---|---|---|
| Qwen3-30B-A3B | MoE | Megatron-LM<sup>†</sup> | 32 × A800<sup>‡</sup> · GBS 1024 · 32K | **1.16×** |
| DeepSeek-V3.2 Lite <sup>§</sup> | MoE + DSA | Megatron-LM<sup>†</sup> | Reduced-layer · GBS 128 · 8K | **5.04×** |
| Qwen3-VL-30B-A3B | VLM | VeOmni<sup>†</sup> | 32 × A800<sup>‡</sup> · GBS 128 · 32K | **1.45×** |
| GR00T N1.6 | VLA | LeRobot<sup>†</sup> | 8 × A800<sup>‡</sup> · GBS 128 · 224×224 | **2.31×** |
| Pi0.5 | VLA | OpenPI<sup>†</sup> | 8 × A800<sup>‡</sup> · GBS 112 · 224×224 | **1.65×** |

> <sup>§</sup> Due to test-bed scale limits, **DeepSeek-V3.2** was validated separately on a reduced-layer configuration — LoongForge's **DSA CUDA kernel optimizations** still deliver **~5× speedup** over Megatron-LM and reach **64K sequence** (baseline OOMs beyond 8K).<br>
> <sup>†</sup> Numbers reflect baseline and LoongForge versions at the time of measurement, and may evolve as implementations change.<br>
> <sup>‡</sup> Validation on additional hardware is rolling out in upcoming releases.<br>

</details>

## 🌟 Powered by LoongForge

- [LLaVA-OneVision-2.0](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) — Next-generation multimodal model, with new VideoCaption and Spatial datasets.
- [LLaVA-OneVision-1.5](https://arxiv.org/abs/2509.23661) — Fully open framework for democratized multimodal training.
- [Qianfan-VL](https://github.com/baidubce/Qianfan-VL) — Domain-Enhanced Vision-Language Models for Enterprise, 3B to 70B parameters.

<a id="models"></a>
## 🏛️ Supported Models

LoongForge supports a broad range of [state-of-the-art models](https://loongforge.readthedocs.io/en/latest/get_started/support_model.html) across LLM, VLM, diffusion, and VLA.

| **Modality** | **Architectures** | **Models** |
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
| | CustomCombinedModel | Flexible ViT + LLM backbone configuration ([example](https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml)) |
| **Diffusion** | WAN2.2 | wan2.2_i2v_a14b |
| **VLA** | Pi | pi0.5 |
| | GR00T | groot-n1.6 |

## 🏗️ Repository Layout

<details>
<summary><b>📁 Directory tree</b></summary>

```
LoongForge/
├── loongforge/                   # Core training framework
│   ├── train/                    # Training entry points & trainers
│   │   ├── pretrain/             #   Pretrain (LLM, VLM)
│   │   ├── sft/                  #   SFT (LLM, VLM, InternVL, ERNIE)
│   │   ├── diffusion/            #   Diffusion (WAN)
│   │   └── embodied/             #   Embodied AI (Pi0.5, GR00T)
│   ├── models/                   # Unified model abstractions
│   │   ├── foundation/           #   LLM backbones (LLaMA, Qwen, DeepSeek, ...)
│   │   ├── encoder/              #   Vision encoders (ViT, Qwen-VL, InternVL, ...)
│   │   ├── omni_models/          #   Multi-modal composition
│   │   ├── diffusion/            #   Diffusion models (WAN)
│   │   ├── embodied/             #   Embodied models (Pi0.5, GR00T)
│   │   └── common/               #   Shared layers and utilities
│   ├── data/                     # Data pipelines (multi-modal, video, DP balance)
│   ├── tokenizer/                # Tokenizers
│   └── utils/                    # Config map, constants, etc.
├── third_party/Loong-Megatron/   # Patched Megatron-LM (git submodule)
├── configs/                      # Hydra YAML configs (models, data)
├── examples/                     # GPU launch scripts
├── examples_xpu/                 # Kunlun XPU launch scripts
├── tools/                        # Checkpoint conversion, data preprocessing
├── ops/                          # Custom fused operators (incl. open-sourced TileLang)
├── patches/                      # TransformerEngine patches
├── docker/                       # Dockerfiles (GPU & XPU)
├── tests/                        # E2E test suite (YAML-driven)
└── docs/                         # Documentation
```

</details>

## 🤝 Contributing

We warmly welcome community contributions — bug reports, feature proposals, and PRs alike. Please read our [Contributing Guidelines](https://github.com/baidu-baige/LoongForge/blob/master/CONTRIBUTING.md) before submitting.

## 📄 License

LoongForge is released under the [Apache License 2.0](https://github.com/baidu-baige/LoongForge/blob/master/LICENSE). Some files are derived from third-party open-source projects; please refer to the specific file headers for their respective copyright and attribution.

## 📝 Citation

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A modular, scalable, high-performance training framework for LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

## 🙏 Acknowledgments

LoongForge is built upon NVIDIA's Megatron-LM. We also drew inspiration from several excellent open-source projects, including but not limited to HuggingFace Transformers, LLaMA-Factory, and Megatron-Bridge. We sincerely thank these communities for their outstanding contributions.

## 💬 Contact
<a id="contact"></a>

Open a GitHub issue for questions, feedback, or feature requests. You can also [join our Slack community](https://join.slack.com/t/baiduloongforge/shared_invite/zt-3ys3kaq2p-cmdw0nDoaHGOcKibgys5Yw) or scan the WeChat QR code below to join our developer community.

<img width="377" alt="LoongForge WeChat Community"  src="https://github.com/user-attachments/assets/7516fcea-5f30-49ba-989f-a33e105bc32b" />




