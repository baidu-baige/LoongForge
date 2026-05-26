.. LoongForge 中文文档主文件

LoongForge 中文文档
=====================

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://loongforge.readthedocs.io/en/latest/
.. image:: https://img.shields.io/github/license/baidu-baige/LoongForge.svg
   :target: https://github.com/baidu-baige/LoongForge/blob/master/LICENSE
.. image:: https://img.shields.io/github/stars/baidu-baige/LoongForge.svg?style=social
   :target: https://github.com/baidu-baige/LoongForge

面向语言、多模态与具身模型的模块化、可扩展、高效训练框架。基于 Megatron-LM 并进行了大量增强。

---

.. toctree::
   :maxdepth: 1
   :caption: 快速开始

   get_started/installation
   get_started/support_model
   get_started/optimization_guide

.. toctree::
   :maxdepth: 2
   :caption: LLM 训练

   llm_tutorial/quick_start_llm_pretrain
   llm_tutorial/quick_start_llm_sft
   llm_tutorial/llm_ckpt_convert
   高级特性 <llm_tutorial/features_index>

.. toctree::
   :maxdepth: 2
   :caption: VLM 训练

   vlm_tutorial/quick_start_vlm_pretrain
   vlm_tutorial/quick_start_vlm_sft
   vlm_tutorial/dataset_conversion
   vlm_tutorial/vlm_ckpt_convert
   高级特性 <vlm_tutorial/features_index>

.. toctree::
   :maxdepth: 1
   :caption: VLA 训练

   vla_tutorial/quick_start_pi05_training

.. toctree::
   :maxdepth: 1
   :caption: Diffusion 训练

   wan_tutorial/quick_start_wan_training
   wan_tutorial/wan_packing

.. toctree::
   :maxdepth: 1
   :caption: 昆仑训练

   kunlun_tutorial/README
   kunlun_tutorial/install_p800
   kunlun_tutorial/quick_start_llm_pretrain_p800
   kunlun_tutorial/quick_start_llm_sft_p800
   kunlun_tutorial/quick_start_vlm_p800
   kunlun_tutorial/quick_start_vla_p800

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   advance/support_new_model

.. toctree::
   :maxdepth: 1
   :caption: 更多

   HEADER_GUIDELINES
   faqs
