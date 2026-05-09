# 昆仑芯 P800 说明

LoongForge 支持在昆仑芯 P800 XPU 上进行训练，涵盖 LLM、VLM、VLA 等多种模型类型。

## 快速开始

### 安装

参考 [昆仑芯 P800 安装](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/install_p800.html)

### 快速开始：VLM 模型训练

参考 [快速开始：昆仑芯 P800 上 VLM 模型 SFT（监督微调）训练](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_vlm_p800.html)

### 快速开始：LLM 模型训练

**预训练**：参考 [快速开始：昆仑芯 P800 上 LLM 模型预训练](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_llm_pretrain_p800.html)

**SFT（监督微调）**：参考 [快速开始：昆仑芯 P800 上 LLM 模型 SFT（监督微调）训练](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_llm_sft_p800.html)

### 快速开始：VLA 模型训练

**SFT（监督微调）**：[快速开始：昆仑芯 P800 上 VLA 模型 SFT（监督微调）训练](https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/quick_start_vla_p800.html)

## 支持的模型

| **模型类型** | **模型系列** | **模型** | **预训练** | **SFT（监督微调）** |
|:---|:---|:---|:---:|:---:|
| LLM | DeepSeek-V3.1 | deepseek_v3_group_bf16 | | ✅ (示例) |
| | MiniMax | minimax_m2.5_230b | | ✅ (示例) |
| | | minimax_m2.7_230b | | ✅ (示例) |
| | Qwen2.5 | qwen2.5_0.5b | | |
| | | qwen2.5_1.5b | | |
| | | qwen2.5_3b | | |
| | | qwen2.5_7b | | |
| | | qwen2.5_14b | | |
| | | qwen2.5_32b | | ✅ (示例) |
| | | qwen2.5_72b | | |
| | Qwen3 | qwen3_8b | | ✅ (示例) |
| | | qwen3_14b | | ✅ (示例) |
| | | qwen3_32b | | ✅ (示例) |
| | | qwen3_30b_a3b | ✅ (示例) | ✅ (示例) |
| | | qwen3_235b_a22b | | |
| | | qwen3_480b_a35b | | |
| VLM | Qwen3-VL | qwen3_vl_30b_a3b | ✅ (示例) | ✅ (示例) |
| | | qwen3_vl_235b_a22b | ✅ (示例) | ✅ (示例) |
| | InternVL-3.5 | internvl3.5_8b | | ✅ (示例) |
| | | internvl3.5_14b | | ✅ (示例) |
| | | internvl3.5_38b | | ✅ (示例) |
| | | internvl3.5_30b_a3b | | ✅ (示例) |
| | | internvl3.5_241b_a28b | | ✅ (示例) |
| | Qwen-3.5 | qwen3.5_35b_a3b | ✅ (示例) | ✅ (示例) |
| VLA | PI 0.5 | | | ✅ (示例) |
