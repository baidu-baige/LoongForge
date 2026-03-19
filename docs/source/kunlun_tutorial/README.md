# Kunlunxin P800 README

OmniTraining supports training on Kunlunxin P800, covering mainstream LLM, VLM, VLA, and other model types.

## Quick Start

### Installation

Refer to [Installation on Kunlunxin P800](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/install_p800.md)

### Quick Start: VLM Model Training

Refer to [Quick Start: VLM Model SFT Training on Kunlunxin P800](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_vlm_p800.md)

### Quick Start: LLM Model Training

**Pretrain**: Refer to [Quick Start: LLM Model Pretrain Training on Kunlunxin P800](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_llm_pretrain_p800.md)

**SFT**: Refer to [Quick Start: LLM Model SFT Training on Kunlunxin P800](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_llm_sft_p800.md)

### Quick Start: VLA Model Training

**SFT**: [Quick Start: VLA Model SFT Training on Kunlunxin P800](https://github.com/baidu-baige/BaigeOmni/blob/master/docs/source/kunlun_tutorial/quick_start_vla_p800.md)

## Supported Models

| **Model Type** | **Model Category** | **Model** | **Pretrain** | **SFT** |
|:---|:---|:---|:---:|:---:|
| LLM | DeepSeek-V3.1 | deepseek_v3_group_bf16 | | ✅ (example) |
| | Qwen2.5 | qwen2.5_0.5b | | |
| | | qwen2.5_1.5b | | |
| | | qwen2.5_3b | | |
| | | qwen2.5_7b | | |
| | | qwen2.5_14b | | |
| | | qwen2.5_32b | | ✅ (example) |
| | | qwen2.5_72b | | |
| | Qwen3 | qwen3_8b | | ✅ (example) |
| | | qwen3_14b | | ✅ (example) |
| | | qwen3_32b | | ✅ (example) |
| | | qwen3_30b_a3b | ✅ (example) | ✅ (example) |
| | | qwen3_235b_a22b | | |
| | | qwen3_480b_a35b | | |
| VLM | Qwen3-VL | qwen3_vl_30b_a3b | ✅ (example) | ✅ (example) |
| | | qwen3_vl_235b_a22b | ✅ (example) | ✅ (example) |
| | InternVL-3.5 | internvl3.5_8b | | ✅ (example) |
| | | internvl3.5_14b | | ✅ (example) |
| | | internvl3.5_38b | | ✅ (example) |
| | | internvl3.5_30b_a3b | | ✅ (example) |
| | | internvl3.5_241b_a28b | | ✅ (example) |
| VLA | PI 0.5 | | ✅(exampe) | ✅(exampe) |