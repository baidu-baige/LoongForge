LoongForge
==========================

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://loongforge.readthedocs.io/en/latest/
.. image:: https://img.shields.io/github/license/baidu-baige/LoongForge.svg
   :target: https://github.com/baidu-baige/LoongForge/blob/master/LICENSE
.. image:: https://img.shields.io/github/stars/baidu-baige/LoongForge.svg?style=social
   :target: https://github.com/baidu-baige/LoongForge

A modular, scalable, and highly efficient training framework for language, multimodal, and embodied models. Built upon Megatron-LM with significant enhancements.

---

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation
   get_started/support_model
   get_started/optimization_guide

.. toctree::
   :maxdepth: 2
   :caption: LLM Training

   llm_tutorial/quick_start_llm_pretrain
   llm_tutorial/quick_start_llm_sft
   llm_tutorial/llm_ckpt_convert
   Advanced Features <llm_tutorial/features_index>

.. toctree::
   :maxdepth: 2
   :caption: VLM Training

   vlm_tutorial/quick_start_vlm_pretrain
   vlm_tutorial/quick_start_vlm_sft
   vlm_tutorial/dataset_conversion
   vlm_tutorial/vlm_ckpt_convert
   Advanced Features <vlm_tutorial/features_index>

.. toctree::
   :maxdepth: 1
   :caption: Embodied Training

   embodied_tutorial/overview
   features/delta_fp8_allgather
   embodied_tutorial/quick_start_pi05
   embodied_tutorial/quick_start_groot_n1_6
   embodied_tutorial/quick_start_groot_n1_7
   embodied_tutorial/quick_start_lingbot_va
   embodied_tutorial/quick_start_xvla
   embodied_tutorial/quick_start_dreamzero
   embodied_tutorial/quick_start_fastwam
   embodied_tutorial/quick_start_cosmos3
   embodied_tutorial/eval_overview

.. toctree::
   :maxdepth: 1
   :caption: Diffusion Training

   wan_tutorial/quick_start_wan_training
   wan_tutorial/wan_packing
   qwen_image_edit_tutorial/quick_start_qwen_image_edit_2511_training

.. toctree::
   :maxdepth: 1
   :caption: KunLun Training

   kunlun_tutorial/README
   kunlun_tutorial/install_p800
   kunlun_tutorial/quick_start_llm_pretrain_p800
   kunlun_tutorial/quick_start_llm_sft_p800
   kunlun_tutorial/quick_start_vlm_p800

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   advance/support_new_model

.. toctree::
   :maxdepth: 1
   :caption: More

   CONTRIBUTING
   HEADER_GUIDELINES
   faqs
