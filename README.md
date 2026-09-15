<p align="right"><sub><b>English</b> | <a href="./README_zh.md">简体中文</a></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="682">
  </picture>
</p>

<h3 align="center">Train LLMs, VLMs, diffusion and embodied models, faster.</h3>

<p align="center">
  <a href="https://github.com/baidu-baige/LoongForge/stargazers"><img src="https://img.shields.io/github/stars/baidu-baige/LoongForge?style=flat&logo=github&color=4F46E5" alt="GitHub stars"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green?logo=apache&logoColor=white" alt="License: Apache-2.0"></a>
  <a href="https://hub.docker.com/u/loongforge"><img src="https://img.shields.io/badge/Docker-loongforge-2496ED?logo=docker&logoColor=white" alt="Docker images on Docker Hub"></a>
  <a href="./CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?logo=github&logoColor=white" alt="PRs welcome"></a>
</p>

<p align="center">
  <a href="#performance"><img src="https://img.shields.io/badge/⚡_Speedup-up_to_5.04x-3B4FD8" alt="Training throughput speedup up to 5.04x over open-source baselines"></a>
  <a href="#models"><img src="https://img.shields.io/badge/📦_Models-40%2B_ready_to_run-7C3AED" alt="40+ ready-to-run model examples"></a>
  <img src="https://img.shields.io/badge/🖥_Hardware-NVIDIA%2BKunlun-EC4899" alt="Runs on NVIDIA GPUs and Kunlun XPUs">
  <img src="https://img.shields.io/badge/🏭_Production-5000%2B_XPUs-DB2777" alt="Proven in production with runs up to 5,000+ XPUs">
</p>

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/"><b>🌐 Website</b></a>
  &nbsp;·&nbsp;
  <a href="https://loongforge.readthedocs.io/en/latest/index.html"><b>📖 Docs</b></a>
  &nbsp;·&nbsp;
  <a href="https://baidu-baige.github.io/LoongForge/blog/"><b>✍️ Blog</b></a>
  &nbsp;·&nbsp;
  <a href="#quickstart"><b>⚡ Quick Start</b></a>
  &nbsp;·&nbsp;
  <a href="#performance"><b>📊 Performance</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>🏛️ Supported Models</b></a>
  &nbsp;·&nbsp;
  <a href="#contact"><b>💬 Contact Us</b></a>
</p>

---

<h6 align="left">Example: embodied-model training on LoongForge — DreamZero at 4.38× baseline throughput, loss curves aligned</h6>

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/assets/video/dreamzero-comparison.mp4">
    <picture>
      <source media="(prefers-reduced-motion: reduce)" srcset="./docs/assets/images/demo/dreamzero-poster.jpg">
      <img alt="DreamZero training run compared side by side: LoongForge reaches 4.38x the baseline throughput while the training loss curves stay aligned" src="./docs/assets/images/demo/dreamzero-loop.webp" width="100%" />
    </picture>
  </a>
</p>

</div>

## 💡 Why LoongForge?

**LoongForge** is an open-source training framework developed by the [Baidu AI Cloud Baige team](https://cloud.baidu.com/product/aihc.html), built to deliver [faster training](#performance) for mainstream LLMs, VLMs, diffusion, and embodied models, thereby significantly reducing costs.

- **Easy to Use** — [Ready-to-run configs](./configs/models/) and [launch examples](./examples) for supported models, covering pre-training, continued pre-training, SFT, and LoRA.
- **High Performance** — Built on multiple training backends (Megatron-LM and torch-native), with **deep optimizations** for each model family across parallelism strategy, memory footprint, communication overlap, and kernel efficiency, while keeping **training loss curves aligned with the baseline**.
- **Proven at Scale** — Open-sourced from [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr), a training acceleration suite serving enterprise customers in Education, Computer Vision, and Embodied AI, **with the largest production runs reaching 5,000+ XPUs**.

> 🐉 LoongForge is named after the traditional Chinese **loong boat (龙舟)**, a symbol of coordinated power and forward momentum.

## 🏗️ Architecture

Since optimal training strategies differ across model families and scales, LoongForge adopts a multi-backend architecture.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/architecture/loongforge-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/architecture/loongforge-architecture.svg">
    <img alt="LoongForge architecture: a patched-Megatron stack for LLMs, VLMs and diffusion models alongside a torch-native stack for embodied models" src="./docs/assets/images/architecture/loongforge-architecture.svg" width="100%">
  </picture>
</p>

- **Megatron Stack** — For LLMs, VLMs, and diffusion models. Powered by a [patched Megatron-LM](https://github.com/baidu-baige/Loong-Megatron) and extended with MoE parallelism, per-component heterogeneous parallelism, long-sequence optimizations, etc.
- **Torch-Native Stack** — For embodied models (VLA and WAM). A standalone [torch-native subsystem](./loongforge/embodied) featuring **DDP / ZeRO-1 / FSDP / HSDP**, with deep optimizations for representative models across I/O, communication strategy, kernel efficiency, etc.

## 🔥 Latest News

- **[2026/09]** ✨ Added training support for **[GLM-5.3-flash](./examples/glm5_next/)**.
- **[2026/09]** ✨ Added **[Kimi-K3](./examples/kimi_k3/)** BF16 training support for both LLMs and VLMs.
- **[2026/09]** ⚡ Added an optimized **[DreamZero Wan2.2-5B FSDP recipe](./examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh)** with cache-aware data loading, compiled attention blocks, frozen-module handling, and Delta-FP8 AllGather.
- **[2026/08]** 🤖 Added VLA training support for **[Wall-OSS-0.5](./examples/embodied/wall_oss_0_5/)**, with custom fused operators for higher training throughput.
- **[2026/08]** 📄 Released the **[TAOT paper](https://arxiv.org/abs/2608.03676)** — topology-aware dynamic expert replica placement that tackles expert-parallel (**EP**) load imbalance in **MoE** training, cutting overhead by up to **74%** over industry solutions, with **1.43× speedup** measured on a real training case. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-08-taot-topology-aware-expert-placement.html)]
- **[2026/08]** ✨ Added training support for **GLM-5.2**, along with a **[GLM-5.2 + MoonViT](./configs/models/glm5.2_vit/)** custom-composition [example](./examples/glm5.2_vit/) for extending GLM with multimodal capabilities.
- **[2026/08]** ✨ Added training support for **MiniCPM-V-4.6** and **Qwen3.8-27B**.
- **[2026/08]** 🧪 Introduced a unified [**evaluation module**](./loongforge/embodied/eval/) for the embodied stack, currently covering **Pi0.5 / xVLA / GR00T**, with more models on the way.
- **[2026/07]** 🐳 Unified the **prebuilt Docker images** — all model families (LLM / VLM / VLA / Diffusion) now share a single image.
- **[2026/07]** 🤖 Released **[LoongForge-Embodied](./loongforge/embodied)**, a torch-native DDP/FSDP training subsystem for embodied models (Pi0.5, GR00T-N1.6/N1.7, xVLA, LingBot-VA, FastWAM, DreamZero, and Cosmos3), with up to **4.38× speedup**. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-07-announcing-loongforge-embodied.html)]
- **[2026/07]** ✨ Added training support for **Qwen-Image-Edit-2511**.
- **[2026/07]** ✨ Added training support for **DeepSeek-V4-Flash / DeepSeek-V4-Pro**.

<details>
<summary><b>📅 More</b></summary>

- **[2026/06]** 🤖 Expanded VLA coverage with **GR00T N1.6**; **2.3× speedup** on GR00T training. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html)]
- **[2026/05]** ⚡ Accelerated **Wan 2.2** training by **116%**, and added CP and data packing support.
- **[2026/05]** ✨ Added training support for **Kimi K2.5 / K2.6**, and introduced **INT4 / NVFP4** PTQ.
- **[2026/05]** 🎉 **v0.1.0** — first official tagged release of LoongForge.
- **[2026/05]** 🌟 Powered the training and public release of **LLaVA-OneVision-2.0**.
- **[2026/04]** 🧩 Added training support for **MiniMax-M2.7** on both NVIDIA GPU and Kunlun XPU.
- **[2026/04]** 🚀 LoongForge source code publicly available on GitHub. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-04-announcing-loongforge.html)]
- **[2025/10]** 🌟 Powered the training and public release of **LLaVA-OneVision-1.5** under **AIAK-Training-LLM**, the predecessor of LoongForge. [[blog](https://baidu-baige.github.io/LoongForge/blog/2025-10-llava-onevision-case-study.html)]

</details>

## ✨ Key Features

**🚀 Foundation Models**

* **MoE EP Communication Optimization** — Overlapped All2All / activation offload / compute, with **further memory reduction** beyond upstream Megatron-LM on DeepSeek-V3, Qwen3-MoE, etc.
* **MoE Expert Load Balancing** — Topology-aware dynamic replication of hot experts to balance EP workloads, with up to **74%** lower overhead than industry solutions. [[TAOT Paper](https://arxiv.org/pdf/2608.03676)]
* **Adaptive FP8 Training** — End-to-end FP8 for LLMs and VLMs with standard **blockwise FP8**; optional **adaptive** mode picks per-operator precision by GEMM shape and efficiency.
* **Custom Fused Operators** — Fused kernels like **FusedDSA** for DSA-style models — TileLang version open-sourced, high-performance CUDA version available on Baidu Baige platform.
* **Long-Sequence Training** — **Context Parallel (CP)** with **chunked-pipeline scheduling** scales LLM training to long sequence lengths.

**🧩 Multi-Modal Models**

* **Flexible Multi-Modal Composition** — Assemble VLMs from interchangeable ViT and LLM components (e.g. **GLM-5.2 + MoonViT**) straight from config — no custom model code.
* **Heterogeneous Parallelism** — Independent TP / DP / recompute / freeze per model component (e.g., ViT vs. LLM) for optimal throughput and memory. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)]
* **Decoupled Encoder-Decoder Training** — Separates ViT and LLM into independent tasks, eliminating encoder-induced pipeline bubbles.
* **DP Load Balancing** — Load-aware data redistribution mitigates sequence-packing imbalance, improving multi-node scaling efficiency. [[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)]

**🤖 Embodied Models**

* **VLA & WAM Training** — A dedicated **torch-native DDP/FSDP** subsystem for **VLA and world-action (WAM)** models, decoupled from the Megatron core, with flexible **DDP / ZeRO-1 / FSDP / HSDP** strategies. [[README](./loongforge/embodied)]
* **Delta-FP8 FSDP Communication** — Optionally compresses BF16 FSDP2 AllGather deltas into blockwise FP8 on supported NVIDIA GPUs while keeping model computation in BF16. [[Usage](./docs/source/features/delta_fp8_allgather.md)]
* **Per-Model Deep Optimization** — Training code deeply customized for each supported model across I/O, communication strategy, and kernel efficiency — **1.79×–4.38×** over official baselines in our [benchmarks](#performance).
* **Unified Evaluation** — Evaluate trained policies on **LIBERO / CALVIN / SimplerEnv / RoboTwin**, with coverage expanding continuously.

**🧰 Workflow & Compatibility**

* **Versatile Pipelines & Data Tools** — Out-of-the-box **Pretrain / MidTrain / SFT / LoRA**, with built-in dataset format conversion and sequence packing.
* **Flexible Checkpointing** — Offline bidirectional **Megatron ↔ HuggingFace** conversion plus native online HF load/save — no format barriers across your workflow.
* **Heterogeneous Hardware** — Native support for **NVIDIA GPUs** and **Kunlun XPUs** via a minimally-intrusive plugin design.

> 📖 Deep-dive: [LLM](https://loongforge.readthedocs.io/en/latest/llm_tutorial/features_index.html) · [VLM](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/features_index.html) · [Embodied Model](https://loongforge.readthedocs.io/en/latest/embodied_tutorial/overview.html)

<a id="performance"></a>
## 📊 Performance

Training throughput speedups over mainstream open-source baselines — each model and its baseline were benchmarked on the same machine type with the same training hyperparameters:

<p align="center">
  <img alt="LoongForge benchmark speedups over open-source baselines — from 1.45x on Qwen3-VL up to 5.04x on DeepSeek-V3.2 Lite" src="./docs/assets/images/benchmark_speedup.png" width="860" />
</p>

> DeepSeek-V3.2 Lite reflects DSA operator-level optimizations and was validated on a reduced-layer configuration due to test-bed scale limits.<br>
> Numbers were measured at a point in time and may evolve as implementations change on both sides.

<a id="quickstart"></a>
## ⚡ Quick Start

**1. Install** — use the [**unified prebuilt Docker image**](https://hub.docker.com/u/loongforge) (one image for all model families) or build from source:
- **NVIDIA GPU**: [Installation Guide](https://loongforge.readthedocs.io/en/latest/get_started/installation.html)
- **Kunlun XPU**: [Installation Guide](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/install_p800.html)

**2. Pick a tutorial** — by hardware and modality:
- **NVIDIA GPU**: [LLM](https://loongforge.readthedocs.io/en/latest/llm_tutorial/quick_start_llm_pretrain.html) · [VLM](https://loongforge.readthedocs.io/en/latest/vlm_tutorial/quick_start_vlm_pretrain.html) · [VLA & WAM](https://loongforge.readthedocs.io/en/latest/embodied_tutorial/overview.html) · [Diffusion](https://loongforge.readthedocs.io/en/latest/wan_tutorial/quick_start_wan_training.html)
- **Kunlun XPU**: [Kunlun XPU Tutorials](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/README.html)

**3. Find your model's scripts** — ready-to-run launch scripts are available under [`examples/`](./examples) / [`examples_xpu/`](./examples_xpu), with configs in [`configs/models/`](./configs/models).

<a id="models"></a>
## 🏛️ Supported Models

LoongForge supports a broad range of model families across LLM, VLM, diffusion, and embodied. Select a model below to open its training examples. For complete usage instructions, see the [User Guide](https://loongforge.readthedocs.io/en/latest/index.html) and the full [model support matrix](https://loongforge.readthedocs.io/en/latest/get_started/support_model.html).

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

## 🌟 Powered by LoongForge

Open-source models trained with LoongForge or its predecessor AIAK-Training-LLM:

| Model | Highlights |
|-------|-------------|
| [**LLaVA-OneVision-2.0**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) | Next-generation multimodal model, with new VideoCaption and Spatial datasets |
| [**Innovator-VL**](https://github.com/InnovatorLM/Innovator-VL/tree/main) | Scientific multimodal LLM for advanced reasoning |
| [**LLaVA-OneVision-1.5**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/1.5) | Fully open framework for democratized multimodal training |
| [**Qianfan-VL**](https://github.com/baidubce/Qianfan-VL) | Domain-enhanced vision-language models for enterprise, 3B–70B parameters |

## 📂 Repository Layout

<details>
<summary><b>📁 Directory tree</b></summary>

```
LoongForge/
├── loongforge/                   # Core training framework
│   ├── train/                    # Training entry points & trainers
│   │   ├── pretrain/             #   Pretrain (LLM, VLM)
│   │   ├── sft/                  #   SFT (LLM, VLM, InternVL, ERNIE)
│   │   └── diffusion/            #   Diffusion (WAN, Qwen-Image)
│   ├── models/                   # Unified model abstractions
│   │   ├── foundation/           #   LLM backbones (LLaMA, Qwen, DeepSeek, ...)
│   │   ├── encoder/              #   Vision encoders (ViT, Qwen-VL, InternVL, ...)
│   │   ├── omni_models/          #   Multi-modal composition
│   │   ├── diffusion/            #   Diffusion models (WAN, Qwen-Image)
│   │   └── common/               #   Shared layers and utilities
│   ├── embodied/                 # LoongForge-Embodied: standalone torch-native (DDP/FSDP)
│   │                             #   embodied (VLA + world-action) subsystem — see loongforge/embodied/README.md
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

## 📝 Citation

If you find LoongForge helpful, please cite this project:

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A high-performance framework for training LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

If you use TAOT for MoE training in LoongForge, you can cite our paper:

```bibtex
@article{zhang2026taot,
  title   = {{TAOT}: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in {MoE} Training},
  author  = {Zhang, Lingyun and Zhang, Henghua and Gu, Shilei and Mo, Kai and Han, Shuai and Li, Shiyong and Wang, Yanpeng and Shen, Dou},
  journal = {arXiv preprint arXiv:2608.03676},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03676}
}
```

## 🤝 Contributing

We warmly welcome community contributions — bug reports, feature proposals, and PRs alike. Please read our [Contributing Guidelines](./CONTRIBUTING.md) before submitting.

Thanks to all our contributors:

<a href="https://github.com/baidu-baige/LoongForge/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=baidu-baige/LoongForge&v=2026-09-04" alt="LoongForge contributors" />
</a>

## 🙏 Acknowledgments

LoongForge builds on NVIDIA's [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and draws inspiration from many excellent open-source projects, including [HuggingFace Transformers](https://github.com/huggingface/transformers), [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), and [LeRobot](https://github.com/huggingface/lerobot), as well as the official implementations of the models it supports (e.g. [OpenPI](https://github.com/Physical-Intelligence/openpi), [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)). We sincerely thank these communities for their outstanding contributions, and would also like to extend our gratitude to the [LINUX DO](https://linux.do/) community for its welcoming space for technical discussion and support for open-source sharing.

<a id="contact"></a>
## 💬 Contact Us

- **GitHub Issues** — Bug reports, usage questions, and feature requests. [Open an issue](https://github.com/baidu-baige/LoongForge/issues/new/choose).
- **Developer Communities** — **WeChat group**, **Xiaohongshu**, and more. [Join here](https://github.com/baidu-baige/LoongForge/issues/80).
- **Email** — Enterprise adoption, large-scale deployment, partnership, or any other topic. [loongforge@baidu.com](mailto:loongforge@baidu.com).

## 📄 License

LoongForge is released under the [Apache License 2.0](./LICENSE). Some files are derived from third-party open-source projects; please refer to the specific file headers for their respective copyright and attribution.
