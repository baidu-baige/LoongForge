// LoongForge site — shared interactions (with i18n + GitHub icon + theme)
(function () {
  'use strict';

  // ===== i18n =====
  const I18N = {
    en: {
      'nav.home': 'Home', 'nav.docs': 'Docs', 'nav.blog': 'Blog', 'nav.about': 'About', 'nav.contact': 'Contact',
      'footer.tagline': 'Modular, scalable training framework for LLM / VLM / VLA / Diffusion.',
      'footer.project': 'Project', 'footer.resources': 'Resources', 'footer.loong': 'Baige Loong Series',
      'footer.copyright': '© 2026 LoongForge Authors · Apache License 2.0 · Built with ♥ by the Baidu Baige Team',
      'footer.quickstart': 'Quick Start', 'footer.modelconfigs': 'Model Configs',
      'footer.examples': 'Examples', 'footer.contributing': 'Contributing',
      'footer.training': 'Training framework', 'footer.workflow': 'Agent framework',

      'hero.badge': '🐉 Part of the Baidu-Baige Loong open-source series',
      'hero.subtitle_html': '<b>Ready-to-run</b> configs for <b>35+</b> model families — built on Megatron-LM and <span class="whitespace-nowrap">torch-native</span> backends.',
      'hero.cta.start': '🚀 Quick Start',
      'hero.cta.github': '⭐ View on GitHub',
      'hero.cta.docs': '📚 Read the Docs',
      'hero.stat.speedup': 'Max training speedup',
      'hero.demo.label': '🎬 4.38× faster on this run — loss curves stay aligned with the baseline',
      'hero.demo.play': 'Play the demo — 83 seconds, has background music',
      'hero.slogan': 'Train LLMs, VLMs, Diffusion & Embodied models — faster.',
      'sb.models': 'Model families supported',
      'sb.chips': 'NVIDIA & Kunlun backends',
      'sb.license': 'Open source license',

      'hero.vp.1.k': 'Easy',
      'hero.vp.1.t': 'One framework, broad coverage',
      'hero.vp.1.d': 'Full coverage of mainstream open-source LLMs, VLMs, MoE, diffusion, and VLA models. Ready-to-run configs and launch scripts included.',
      'hero.vp.2.k': 'Efficient',
      'hero.vp.2.t': 'Up to ~5× training speedup',
      'hero.vp.2.d': 'Deep performance optimizations — fused kernels, adaptive FP8, MoE A2A overlap, and multimodal pipeline scheduling.',
      'hero.vp.3.k': 'Multi-chip',
      'hero.vp.3.t': 'NVIDIA GPU & Kunlun XPU',
      'hero.vp.3.d': 'Native heterogeneous hardware support — one framework, minimal migration between GPU and XPU.',

      'news.title': '🔥 Latest News', 'news.all': 'All posts →',

      'arch.title': '🏗️ Architecture',
      'arch.subtitle': 'One unified stack — from model composition down to GPU / XPU silicon.',

      'feat.title': '✨ Key Features',
      'feat.subtitle_html': 'A quick tour of what sets LoongForge apart',

      'feat.cat.1.title': 'MoE',
      'feat.cat.1.item.1.t': 'Tri-Stream Overlap',
      'feat.cat.1.item.1.d': 'MoE EP comm × compute × offload in parallel — higher throughput than upstream.',

      'feat.cat.2.title': 'Multimodal',
      'feat.cat.2.item.2.t': 'Heterogeneous & Disaggregated',
      'feat.cat.2.item.2.d': 'Independent TP / PP / DP per component + decoupled ViT / LLM scheduling that kills pipeline bubbles.',
      'feat.cat.2.item.4.t': 'DP Load Balancing',
      'feat.cat.2.item.4.d': 'Fixes packing-induced imbalance at cluster scale.',

      'feat.cat.3.title': 'Performance',
      'feat.cat.3.item.1.t': 'Adaptive FP8',
      'feat.cat.3.item.1.d': 'Per-operator FP8 decisions by GEMM shape.',
      'feat.cat.3.item.2.t': 'Fused Operators',
      'feat.cat.3.item.2.d': 'FusedDSA / Sparse MLA kernels for end-to-end speedup.',

      'feat.cat.4.item.1.t': 'ChunkPipe',
      'feat.cat.4.item.1.d': 'Chunked long-sequence pipelining toward million-length contexts.',

      'feat.cat.5.title': 'Usability',
      'feat.cat.5.item.1.t': 'HF ↔ Megatron',
      'feat.cat.5.item.1.d': 'Bidirectional checkpoint conversion + online HF load/save.',

      'feat.cat.6.title': 'Training',
      'feat.cat.6.item.1.t': 'Pretrain + SFT + LoRA',
      'feat.cat.6.item.1.d': 'One codebase covers key training stages.',

      'feat.cat.7.title': 'Embodied',
      'feat.cat.7.item.1.t': 'Embodied Training',
      'feat.cat.7.item.1.d': 'Dedicated torch-native DDP/FSDP subsystem for VLA & WAM models, with DDP / ZeRO-1 / FSDP / HSDP.',

      'models.title': '🏛️ Supported Models',
      'models.subtitle': 'From compact SLMs to large-scale MoE giants — all batteries-included',
      'models.custom.t': 'CustomCombinedModel',
      'models.custom.d_html': 'Compose any ViT + any LLM backbone via a YAML file. <a href="https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml" target="_blank" class="text-indigo-500 underline">Example →</a>',

      'qs.title': '🚀 Quick Start',
      'qs.subtitle': 'From install to launch — jump straight to the tutorial for your model type',
      'qs2.install.t': 'Install',
      'qs2.install.d': 'Recommended: one Docker image bundles the CUDA/XPU toolchains, patched Megatron, and TransformerEngine — so every node trains from the same environment. Source build is also supported.',
      'qs2.docker': 'Show Docker build commands',
      'qs2.guide': 'Installation guide ↗',
      'qs2.path.t': 'Pick your path',
      'qs2.path.d': 'Choose your model type — each card opens a runnable, step-by-step tutorial.',
      'qs2.open': 'Open tutorial ↗',
      'qs2.cat.llm.d': 'Dense & MoE LLMs — pretrain, SFT & LoRA.',
      'qs2.cat.vlm.d': 'Vision-language models — composable ViT × LLM.',
      'qs2.cat.diff.d': 'Video & image diffusion — WAN & Qwen-Image.',
      'qs2.cat.vla.d': 'VLA & world-action models — DDP / FSDP.',
      'qs2.explore.t': 'Explore & launch',
      'qs2.explore.d': 'Browse ready-to-run configs and example scripts, or expand a common launch command.',

      'powered.title': '🌟 Powered by LoongForge',
      'powered.subtitle': 'Open-source models trained with LoongForge or its predecessor AIAK-Training-LLM',
      'powered.new': 'NEW',
      'powered.1.t': 'LLaVA-OneVision-2.0',
      'powered.1.d': 'Next-generation fully open multimodal model — improved data, training recipe, and scaling.',
      'powered.2.t': 'Innovator-VL',
      'powered.2.d': 'Scientific multimodal large language model for advanced reasoning.',
      'powered.3.t': 'LLaVA-OneVision-1.5',
      'powered.3.d': 'Fully open framework for democratized multimodal training.',
      'powered.4.t': 'Qianfan-VL',
      'powered.4.d': 'Domain-enhanced universal vision-language models.',

      'blog.hero.title': 'Engineering Blog',
      'blog.hero.desc': 'Releases, performance deep-dives, and stories from the LoongForge team',

      'about.hero.title': 'About LoongForge',
      'about.hero.desc': 'A training framework born from real-world, large-scale production workloads — and shared back with the community',

      'about.story.title': '🐉 Our Story',
      'about.story.p1_html': '<b>LoongForge</b> is an open-source training framework developed by the <a class="text-indigo-600 underline" href="https://cloud.baidu.com/product/aihc.html" target="_blank" rel="noopener">Baidu AI Cloud Baige team</a>, built to deliver faster training for mainstream <b>LLMs</b>, <b>VLMs</b>, <b>diffusion</b>, and <b>embodied</b> models — and thereby significantly reduce cost.',
      'about.story.p2_html': 'LoongForge was open-sourced from <a class="text-indigo-600 underline" href="https://cloud.baidu.com/doc/AIHC/s/Alyo476jr" target="_blank" rel="noopener">AIAK-Training-LLM</a>, a training acceleration suite delivered to enterprise customers on Baidu AI Cloud, after years of hardening under real production workloads:',
      'about.story.li.1_html': 'Serving customers across <b>Education</b>, <b>Computer Vision</b>, and <b>Embodied AI</b>, typically delivering a <b>30%~50% speedup</b> over their baselines',
      'about.story.li.2_html': 'Largest production runs reaching <b>5,000+ XPUs</b>',
      'about.story.p3_html': 'It now joins the Baige <b>Loong</b> open-source series — named after the traditional Chinese <b>loong boat (龙舟)</b>, a symbol of coordinated power and forward momentum.',
      'about.story.p4_html': 'Want to see what else we are building? Explore our other open-source projects on the <a class="text-indigo-600 underline" href="https://github.com/baidu-baige" target="_blank" rel="noopener">baidu-baige GitHub organization</a>.',

      'about.license.title': '📄 License & Citation',
      'about.license.heading': 'License',
      'about.license.body_html': 'LoongForge is released under the <a href="https://github.com/baidu-baige/LoongForge/blob/master/LICENSE" target="_blank" class="text-indigo-600 underline">Apache License 2.0</a>. Some files are derived from third-party open-source projects — please consult file headers for their specific notices.',
      'about.cite.heading': 'Citation',
      'about.ack.title': '🙏 Acknowledgments',
      'about.ack.body': 'LoongForge is built upon NVIDIA\'s Megatron-LM. We also referenced and drew inspiration from excellent open-source projects including Transformers, LLaMA-Factory, and Megatron-Bridge. We sincerely thank these communities for their outstanding contributions.',

      'about.contact.title': '✉️ Contact Us',
      'about.contact.lead': 'We\'d love to hear from you — reach us through any of these channels',
      'about.contact.wechat.d': 'Join the developer group — scan the QR code posted in our GitHub issue',
      'about.contact.rednote.d': 'Follow us on RedNote (小红书) for release notes and practice sharing',
      'about.contact.email.d': 'loongforge@baidu.com — for collaboration, adoption, and any other enquiries',
      'about.contact.slack.d': 'Chat with the team and other users in our Slack workspace',

      'bench.title': '📊 Benchmark',
      'bench.subtitle': 'Training throughput speedup over mainstream open-source baselines — each model benchmarked on the same machine type with the same training hyperparameters',
      'bench.baseline': '1.0× baseline',
      'bench.note': 'Numbers were measured at a point in time and may evolve as implementations change on both sides.',
      'bench.ds.desc': 'DeepSeek-V3.2 Lite reflects DSA operator-level optimizations and was validated on a reduced-layer configuration due to test-bed scale limits.',
      'bench.issue.ask': "Need a model LoongForge doesn't cover yet?",
      'bench.issue.cta': 'Open an issue',
      'hw.title': '💎 Hardware Compatibility',
      'hw.subtitle': 'One codebase, two silicon stacks — production-ready on NVIDIA GPU and Baidu Kunlun XPU',
      'hw.nv.t': 'NVIDIA GPU',
      'hw.nv.d': 'Built on the community Megatron + TransformerEngine ecosystem, with LoongForge optimizations layered on top.',
      'hw.xpu.t': 'Kunlun XPU',
      'hw.xpu.d': 'XPU Plugin mechanism shields the upper stack from adaptation differences, while integrating an XPU-specific optimization toolchain.',

      'comm.title': '🤝 Community',
      'comm.subtitle': 'Built in the open — join discussions, report issues, and contribute',
      'comm.1.t': 'GitHub Issues', 'comm.1.d': 'File bug reports and feature requests.',
      'comm.2.t': 'Discussions', 'comm.2.d': 'Ask questions and share experiences.',
      'comm.3.t': 'Contributing', 'comm.3.d': 'Read the guide and send your first PR.',
      'comm.contrib.title': 'Contributors',
      'comm.contrib.sub': 'LoongForge is built in the open by these developers — your name could be next.',
      'comm.contrib.cta': '🛠️ Become a contributor',

    },
    zh: {
      'nav.home': '首页', 'nav.docs': '文档', 'nav.blog': '博客', 'nav.about': '关于', 'nav.contact': '联系',
      'footer.tagline': '面向 LLM / VLM / VLA / Diffusion 的模块化、可扩展训练框架。',
      'footer.project': '项目', 'footer.resources': '资源', 'footer.loong': '百舸 Loong 系列',
      'footer.copyright': '© 2026 LoongForge Authors · Apache License 2.0 · 由百度百舸团队用 ♥ 构建',
      'footer.quickstart': '快速开始', 'footer.modelconfigs': '模型配置',
      'footer.examples': '示例', 'footer.contributing': '贡献指南',
      'footer.training': '训练框架', 'footer.workflow': '智能体框架',

      'hero.badge': '🐉 百度百舸 Loong 开源家族成员',
      'hero.subtitle_html': '<b>35+</b> 模型家族的训练配置<b>开箱可用</b> —— 基于 Megatron-LM 与 <span class="whitespace-nowrap">torch-native</span> 双后端。',
      'hero.cta.start': '🚀 快速上手',
      'hero.cta.github': '⭐ 访问 GitHub',
      'hero.cta.docs': '📚 阅读文档',
      'hero.stat.speedup': '最大训练加速',
      'hero.demo.label': '🎬 实测举例：训练吞吐加速 4.38×，loss 曲线与基线对齐',
      'hero.demo.play': '播放演示 —— 83 秒，含背景音乐',
      'hero.slogan': '更快地训练 LLM、VLM、Diffusion 与具身智能模型。',
      'sb.models': '支持模型家族',
      'sb.chips': 'NVIDIA 与昆仑芯后端',
      'sb.license': '开源许可协议',

      'hero.vp.1.k': '易用',
      'hero.vp.1.t': '一套框架，广泛覆盖',
      'hero.vp.1.d': '全面覆盖主流开源 LLM、VLM、MoE、扩散与 VLA 模型。内置开箱即用的模型配置与启动脚本。',
      'hero.vp.2.k': '高效',
      'hero.vp.2.t': '训练加速最高 ~5×',
      'hero.vp.2.d': '深度性能优化 —— 融合算子、自适应 FP8、MoE A2A 通信计算重叠、多模态流水调度。',
      'hero.vp.3.k': '多芯',
      'hero.vp.3.t': 'NVIDIA GPU 与昆仑芯 XPU',
      'hero.vp.3.d': '原生异构硬件支持 —— 同一套框架，GPU 与 XPU 之间迁移成本极低。',

      'news.title': '🔥 最新动态', 'news.all': '查看全部 →',

      'arch.title': '🏗️ 架构',
      'arch.subtitle': '一套统一架构 —— 从模型组装贯通到 GPU / XPU 芯片。',

      'feat.title': '✨ 关键特性',
      'feat.subtitle_html': 'LoongForge 差异化能力速览',

      'feat.cat.1.title': 'MoE',
      'feat.cat.1.item.1.t': '通信/计算/Offload 三流并行',
      'feat.cat.1.item.1.d': 'MoE EP 通信 × 计算 × Offload 三流并行，吞吐优于上游。',

      'feat.cat.2.title': '多模态',
      'feat.cat.2.item.2.t': '异构并行 & 分离训练',
      'feat.cat.2.item.2.d': '不同组件独立设置 TP / PP / DP，并通过 ViT / LLM 解耦调度消除流水气泡。',
      'feat.cat.2.item.4.t': 'DP 负载均衡',
      'feat.cat.2.item.4.d': '修复 packing 带来的 DP 倾斜。',

      'feat.cat.3.title': '性能',
      'feat.cat.3.item.1.t': '自适应 FP8',
      'feat.cat.3.item.1.d': '按 GEMM 形状逐算子决策是否启用 FP8。',
      'feat.cat.3.item.2.t': '融合算子',
      'feat.cat.3.item.2.d': 'FusedDSA / 稀疏 MLA 融合算子，端到端加速。',

      'feat.cat.4.item.1.t': 'ChunkPipe',
      'feat.cat.4.item.1.d': '长序列分块流水，面向百万级上下文。',

      'feat.cat.5.title': '易用性',
      'feat.cat.5.item.1.t': 'HF ↔ Megatron',
      'feat.cat.5.item.1.d': '双向检查点转换 + 在线 HF 加载/保存。',

      'feat.cat.6.title': '训练范式',
      'feat.cat.6.item.1.t': 'Pretrain + SFT + LoRA',
      'feat.cat.6.item.1.d': '同一套代码覆盖关键训练阶段。',

      'feat.cat.7.title': '具身智能',
      'feat.cat.7.item.1.t': '具身模型训练',
      'feat.cat.7.item.1.d': '面向 VLA 与 WAM 模型的独立 torch-native DDP/FSDP 子系统，支持 DDP / ZeRO-1 / FSDP / HSDP。',

      'models.title': '🏛️ 支持的模型',
      'models.subtitle': '从紧凑的小模型到大规模 MoE 巨兽 —— 开箱即用',
      'models.custom.t': '自定义组合模型',
      'models.custom.d_html': '通过 YAML 自由组合任意 ViT + 任意 LLM 主干。<a href="https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml" target="_blank" class="text-indigo-500 underline">示例 →</a>',

      'qs.title': '🚀 快速开始',
      'qs.subtitle': '从安装到启动 —— 按你的模型类型直达对应教程',
      'qs2.install.t': '安装',
      'qs2.install.d': '推荐使用 Docker：单一镜像打包了 CUDA/XPU 工具链、打过补丁的 Megatron 与 TransformerEngine，保证每个节点环境一致；同时也完整支持源码安装。',
      'qs2.docker': '展开 Docker 构建命令',
      'qs2.guide': '安装指南 ↗',
      'qs2.path.t': '选择你的路径',
      'qs2.path.d': '选择你的模型类型 —— 每张卡片直达可运行的分步教程。',
      'qs2.open': '查看教程 ↗',
      'qs2.cat.llm.d': '稠密与 MoE 大语言模型 —— 预训练、SFT、LoRA。',
      'qs2.cat.vlm.d': '视觉-语言模型 —— 可组合 ViT × LLM。',
      'qs2.cat.diff.d': '视频与图像扩散 —— WAN、Qwen-Image。',
      'qs2.cat.vla.d': '具身 VLA 与世界-动作模型 —— DDP / FSDP。',
      'qs2.explore.t': '探索与启动',
      'qs2.explore.d': '浏览开箱即用的配置与示例脚本，或展开查看通用启动命令。',

      'powered.title': '🌟 由 LoongForge 驱动',
      'powered.subtitle': '基于 LoongForge 或其前身 AIAK-Training-LLM 训练的开源模型',
      'powered.new': 'NEW',
      'powered.1.t': 'LLaVA-OneVision-2.0',
      'powered.1.d': '新一代完全开放的多模态模型 —— 在数据、训练配方与 Scale 上全面升级。',
      'powered.2.t': 'Innovator-VL',
      'powered.2.d': '面向高阶推理的科学多模态大语言模型。',
      'powered.3.t': 'LLaVA-OneVision-1.5',
      'powered.3.d': '面向多模态训练民主化的完全开放框架。',
      'powered.4.t': 'Qianfan-VL',
      'powered.4.d': '领域增强的通用视觉-语言模型。',

      'blog.hero.title': '工程博客',
      'blog.hero.desc': '来自 LoongForge 团队的版本发布、性能深挖与案例分享',

      'about.hero.title': '关于 LoongForge',
      'about.hero.desc': '一个诞生于真实大规模生产负载、回馈开源社区的训练框架',

      'about.story.title': '🐉 我们的故事',
      'about.story.p1_html': '<b>LoongForge</b> 是一款开源训练框架，由百度智能云 <a class="text-indigo-600 underline" href="https://cloud.baidu.com/product/aihc.html" target="_blank" rel="noopener">百舸团队</a> 开发，旨在为主流 <b>LLM</b>、<b>VLM</b>、<b>Diffusion</b> 与 <b>具身模型</b> 提供更快的训练速度，从而显著降低成本。',
      'about.story.p2_html': 'LoongForge 由 <a class="text-indigo-600 underline" href="https://cloud.baidu.com/doc/AIHC/s/Alyo476jr" target="_blank" rel="noopener">AIAK-Training-LLM</a> 开源而来 —— 一套在百度智能云上交付给企业客户的训练加速套件，在真实生产负载下沉淀多年：',
      'about.story.li.1_html': '服务 <b>教育</b>、<b>计算机视觉</b>、<b>具身智能</b> 等领域客户，相较其基线通常实现 <b>30%~50% 加速</b>',
      'about.story.li.2_html': '最大生产规模达 <b>5,000+ XPU</b>',
      'about.story.p3_html': '它如今加入百舸 <b>Loong</b> 开源系列 —— 取名自中国传统 <b>龙舟</b>，象征协同之力与前行之势。',
      'about.story.p4_html': '想了解我们还在做什么？欢迎访问 <a class="text-indigo-600 underline" href="https://github.com/baidu-baige" target="_blank" rel="noopener">baidu-baige GitHub 组织</a>，查看团队的其他开源项目。',

      'about.license.title': '📄 开源协议与引用',
      'about.license.heading': '开源协议',
      'about.license.body_html': 'LoongForge 采用 <a href="https://github.com/baidu-baige/LoongForge/blob/master/LICENSE" target="_blank" class="text-indigo-600 underline">Apache License 2.0</a> 协议发布。部分文件衍生自第三方开源项目 —— 其具体协议请参见对应文件头。',
      'about.cite.heading': '引用',
      'about.ack.title': '🙏 致谢',
      'about.ack.body': 'LoongForge 构建于 NVIDIA 的 Megatron-LM 之上，同时借鉴并参考了 Transformers、LLaMA-Factory、Megatron-Bridge 等优秀开源项目。真诚感谢这些社区的卓越贡献。',

      'about.contact.title': '✉️ 联系我们',
      'about.contact.lead': '欢迎随时联系我们 —— 通过以下任一渠道都可以',
      'about.contact.wechat.d': '加入开发者交流群 —— 扫描 GitHub issue 中的群二维码',
      'about.contact.rednote.d': '关注我们的小红书账号，获取版本动态与实践分享',
      'about.contact.email.d': 'loongforge@baidu.com —— 合作、落地及其他任何事宜',
      'about.contact.slack.d': '在 Slack workspace 中与团队和其他使用者交流',

      'bench.title': '📊 性能基准',
      'bench.subtitle': '相较主流开源基线的训练吞吐加速 —— 每个模型均在同机型、同训练超参下实测对比',
      'bench.baseline': '1.0× 基线',
      'bench.note': '数据为某一时间点的实测结果，随双方实现演进可能发生变化。',
      'bench.ds.desc': 'DeepSeek-V3.2 Lite 体现的是 DSA 算子级优化，且受测试集群规模限制，基于减层模型配置验证。',
      'bench.issue.ask': '需要的模型还没被覆盖？',
      'bench.issue.cta': '到 Issues 提出',

      'hw.title': '💎 硬件兼容性',
      'hw.subtitle': '同一套代码，两套芯片栈 —— NVIDIA GPU 与百度昆仑芯 XPU 均已生产化落地',
      'hw.nv.t': 'NVIDIA GPU',
      'hw.nv.d': '基于社区 Megatron + TransformerEngine 生态，在此之上构建并扩充 LoongForge 自研优化。',
      'hw.xpu.t': '昆仑芯 XPU',
      'hw.xpu.d': '采用 XPU Plugin 机制，向上屏蔽 XPU 适配差异，同时集成 XPU 专属优化技术栈。',

      'comm.title': '🤝 开源社区',
      'comm.subtitle': '开放共建 —— 欢迎参与讨论、反馈问题、贡献代码',
      'comm.1.t': 'GitHub Issues', 'comm.1.d': '提交 Bug 报告与功能请求。',
      'comm.2.t': '讨论区', 'comm.2.d': '提问交流与经验分享。',
      'comm.3.t': '贡献指南', 'comm.3.d': '阅读指南，发起你的第一个 PR。',
      'comm.contrib.title': '贡献者',
      'comm.contrib.sub': 'LoongForge 由这些开发者共同构建 —— 下一个名字可以是你。',
      'comm.contrib.cta': '🛠️ 成为贡献者',

    }
  };

  const LANG_KEY = 'lf-lang';
  function getLang() {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === 'zh' || saved === 'en') return saved;
    const nav = (navigator.language || '').toLowerCase();
    return nav.startsWith('zh') ? 'zh' : 'en';
  }
  function applyI18n(lang) {
    const dict = I18N[lang] || I18N.en;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.dataset.i18n;
      if (dict[key] != null) el.textContent = dict[key];
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.dataset.i18nHtml;
      if (dict[key] != null) el.innerHTML = dict[key];
    });
    // Update toggle button labels
    document.querySelectorAll('[data-lang-btn]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.langBtn === lang);
    });
    // Language-aware ReadTheDocs links
    applyDocsLinks(lang);
    // Language-aware blog links (Latest News cards, etc.)
    applyBlogLinks(lang);
    // Notify pages (blog list, etc.) that language changed
    try { window.dispatchEvent(new Event('lf:langchange')); } catch (e) { }
  }

  function applyBlogLinks(lang) {
    document.querySelectorAll('[data-href-en][data-href-zh]').forEach(a => {
      a.href = lang === 'zh' ? a.dataset.hrefZh : a.dataset.hrefEn;
    });
  }

  function applyDocsLinks(lang) {
    const base = lang === 'zh'
      ? 'https://loongforge.readthedocs.io/zh-cn/latest/index.html'
      : 'https://loongforge.readthedocs.io/en/latest/index.html';
    document.querySelectorAll('[data-docs-link]').forEach(a => { a.href = base; });
  }
  function setLang(lang) {
    localStorage.setItem(LANG_KEY, lang);
    applyI18n(lang);
  }

  // ===== Shared site header (single source of truth) =====
  // Usage on any page:
  //   <header data-site-header data-active="home|blog|about" [data-base="../"] [data-lang-mode="post"]></header>
  // Any change to navbar structure only needs to be made here.
  function renderSiteHeader() {
    const hosts = document.querySelectorAll('header[data-site-header]');
    if (!hosts.length) return;
    hosts.forEach(host => {
      const active = host.dataset.active || '';
      const base = host.dataset.base || '';
      const langMode = host.dataset.langMode || 'site'; // 'site' | 'post'
      const act = name => active === name
        ? 'text-indigo-600 dark:text-indigo-300 font-semibold'
        : 'hover:text-indigo-600 dark:hover:text-indigo-300';
      const actMob = name => active === name
        ? 'block py-1 text-indigo-600 dark:text-indigo-300 font-semibold'
        : 'block py-1 hover:text-indigo-600 dark:hover:text-indigo-300';
      const langBtns = langMode === 'post'
        ? `<button type="button" data-post-lang="en">EN</button><button type="button" data-post-lang="zh">中文</button>`
        : `<button type="button" data-lang-btn="en">EN</button><button type="button" data-lang-btn="zh">中文</button>`;
      const ghSvg = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.79 8.21 11.39.6.11.82-.26.82-.58 0-.29-.01-1.05-.02-2.06-3.34.72-4.04-1.61-4.04-1.61-.55-1.39-1.33-1.76-1.33-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.83 1.24 1.83 1.24 1.07 1.83 2.8 1.3 3.49.99.11-.77.42-1.3.76-1.6-2.67-.3-5.47-1.34-5.47-5.96 0-1.32.47-2.39 1.24-3.23-.12-.3-.54-1.52.12-3.16 0 0 1.01-.32 3.3 1.23a11.48 11.48 0 0 1 6 0c2.29-1.55 3.3-1.23 3.3-1.23.66 1.64.24 2.86.12 3.16.77.84 1.24 1.91 1.24 3.23 0 4.63-2.8 5.65-5.48 5.95.43.37.81 1.1.81 2.22 0 1.6-.01 2.89-.01 3.28 0 .32.22.7.83.58A12.01 12.01 0 0 0 24 12.5C24 5.87 18.63.5 12 .5z"/></svg>`;
      host.className = 'sticky top-0 z-40 backdrop-blur bg-white/80 dark:bg-gray-950/75 border-b border-gray-200 dark:border-gray-800';
      host.innerHTML = `
    <div class="max-w-7xl 2xl:max-w-[1600px] mx-auto px-5 sm:px-8 lg:px-12 xl:px-20 2xl:px-28 py-4 flex items-center justify-between">
      <a href="${base}index.html" class="flex items-center gap-2.5 font-bold text-xl">
        <img src="${base}assets/img/logo.svg" class="w-9 h-9" alt="logo" />
        <span>LoongForge</span>
      </a>
      <nav class="hidden md:flex items-center gap-8 text-[15px] font-medium">
        <a href="${base}index.html" class="${act('home')}" data-i18n="nav.home">Home</a>
        <a href="https://loongforge.readthedocs.io/en/latest/index.html" target="_blank" rel="noopener" data-docs-link class="ext ${act('docs')}" data-i18n="nav.docs">Docs</a>
        <a href="${base}blog.html" class="${act('blog')}" data-i18n="nav.blog">Blog</a>
        <a href="${base}about.html" class="${act('about')}" data-i18n="nav.about">About</a>
        <a href="${base}about.html#contact" class="${act('contact')}" data-i18n="nav.contact">Contact</a>
      </nav>
      <div class="flex items-center gap-3">
        <div class="lang-switch hidden sm:inline-flex" role="group" aria-label="Language">${langBtns}</div>
        <a href="https://github.com/baidu-baige/LoongForge" target="_blank" rel="noopener"
          class="gh-pill hidden sm:inline-flex" aria-label="Star LoongForge on GitHub" title="Star LoongForge on GitHub ★">
          ${ghSvg}
          <span>GitHub</span>
          <span class="text-gray-300 dark:text-gray-600">|</span>
          <span class="star">★</span>
          <span data-gh-stars>Star</span>
        </a>
        <button data-theme-toggle class="p-2.5 rounded-lg text-lg hover:bg-gray-100 dark:hover:bg-gray-800" aria-label="Toggle theme">
          <span data-theme-icon>🌙</span>
        </button>
        <button data-mobile-toggle class="md:hidden p-2.5 rounded-lg text-xl leading-none hover:bg-gray-100 dark:hover:bg-gray-800" aria-label="Menu">☰</button>
      </div>
    </div>
    <div id="mobile-menu" class="md:hidden hidden border-t border-gray-200 dark:border-gray-800 px-5 sm:px-8 py-4 space-y-2 text-base">
      <a href="${base}index.html" class="${actMob('home')}" data-i18n="nav.home">Home</a>
      <a href="https://loongforge.readthedocs.io/en/latest/index.html" target="_blank" rel="noopener" data-docs-link class="block py-1 ext" data-i18n="nav.docs">Docs</a>
      <a href="${base}blog.html" class="${actMob('blog')}" data-i18n="nav.blog">Blog</a>
      <a href="${base}about.html" class="${actMob('about')}" data-i18n="nav.about">About</a>
      <a href="${base}about.html#contact" class="${actMob('contact')}" data-i18n="nav.contact">Contact</a>
      <div class="lang-switch mt-2">${langBtns}</div>
    </div>`;
    });
  }

  // ===== Dark mode =====
  const THEME_KEY = 'lf-theme';
  const root = document.documentElement;
  const savedTheme = localStorage.getItem(THEME_KEY);
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (savedTheme === 'dark' || (!savedTheme && prefersDark)) root.classList.add('dark');

  function toggleTheme() {
    root.classList.toggle('dark');
    localStorage.setItem(THEME_KEY, root.classList.contains('dark') ? 'dark' : 'light');
    updateThemeIcon();
  }
  function updateThemeIcon() {
    document.querySelectorAll('[data-theme-icon]').forEach(el => {
      el.textContent = root.classList.contains('dark') ? '☀️' : '🌙';
    });
  }

  // ===== Mobile menu =====
  function toggleMobileMenu() {
    const m = document.getElementById('mobile-menu');
    if (m) m.classList.toggle('hidden');
  }

  // ===== Copy to clipboard =====
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for file:// or insecure contexts
    return new Promise((resolve, reject) => {
      try {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand('copy');
        document.body.removeChild(ta);
        ok ? resolve() : reject(new Error('execCommand copy failed'));
      } catch (e) { reject(e); }
    });
  }
  function attachCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
      if (pre.dataset.copyAttached) return;
      pre.dataset.copyAttached = '1';
      pre.style.position = 'relative';
      const btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.type = 'button';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {
        const code = pre.querySelector('code') || pre;
        try {
          await copyText(code.innerText);
          btn.textContent = 'Copied!';
          setTimeout(() => (btn.textContent = 'Copy'), 1400);
        } catch (e) {
          btn.textContent = 'Failed';
          setTimeout(() => (btn.textContent = 'Copy'), 1400);
        }
      });
      pre.appendChild(btn);
    });
  }

  // ===== Tabs =====
  function initTabs() {
    document.querySelectorAll('[data-tabs]').forEach(group => {
      const btns = group.querySelectorAll('[data-tab]');
      const panels = group.querySelectorAll('[data-panel]');
      btns.forEach(btn => {
        btn.addEventListener('click', () => {
          const key = btn.dataset.tab;
          btns.forEach(b => b.classList.toggle('active', b.dataset.tab === key));
          panels.forEach(p => p.classList.toggle('hidden', p.dataset.panel !== key));
        });
      });
    });
  }

  // ===== Reveal =====
  function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('visible')); return; }
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) { en.target.classList.add('visible'); io.unobserve(en.target); }
      });
    }, { threshold: 0.08 });
    els.forEach(e => io.observe(e));
  }

  // ===== TOC spy =====
  function initTocSpy() {
    const links = document.querySelectorAll('.toc-link[href^="#"]');
    if (!links.length) return;
    const targets = [...links].map(l => document.querySelector(l.getAttribute('href'))).filter(Boolean);
    if (!targets.length) return;
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          const id = '#' + en.target.id;
          links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === id));
        }
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    targets.forEach(t => io.observe(t));
  }

  // ===== Scroll progress bar =====
  function initScrollProgress() {
    const bar = document.getElementById('scroll-progress');
    if (!bar) return;
    const upd = () => {
      const h = document.documentElement;
      const scrolled = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      bar.style.width = (scrolled * 100).toFixed(2) + '%';
    };
    window.addEventListener('scroll', upd, { passive: true });
    window.addEventListener('resize', upd);
    upd();
  }

  // ===== Quick Start step tabs =====
  function initQuickStart() {
    const root = document.getElementById('qs-interactive');
    if (!root) return;
    const tabs = root.querySelectorAll('.qs-tab');
    const panels = root.querySelectorAll('.qs-panel');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const step = tab.dataset.qsStep;
        tabs.forEach(t => t.classList.toggle('active', t === tab));
        panels.forEach(p => p.classList.toggle('active', p.dataset.qsPanel === step));
      });
    });
  }

  // ===== Animated stat counters =====
  function initCounters() {
    const els = document.querySelectorAll('[data-count-to]');
    if (!els.length) return;
    const run = el => {
      const to = parseFloat(el.dataset.countTo);
      const pre = el.dataset.prefix || '';
      const suf = el.dataset.suffix || '';
      const dur = 1100;
      const start = performance.now();
      const tick = now => {
        const p = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - p, 3);
        el.textContent = pre + Math.round(to * eased) + suf;
        if (p < 1) requestAnimationFrame(tick);
        else el.textContent = pre + to + suf;
      };
      requestAnimationFrame(tick);
    };
    if (!('IntersectionObserver' in window)) { els.forEach(run); return; }
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) { run(en.target); io.unobserve(en.target); }
      });
    }, { threshold: 0.4 });
    els.forEach(e => io.observe(e));
  }

  // ===== Init =====
  document.addEventListener('DOMContentLoaded', () => {
    renderSiteHeader();
    applyI18n(getLang());
    updateThemeIcon();
    attachCopyButtons();
    initTabs();
    initReveal();
    initTocSpy();
    initScrollProgress();
    initQuickStart();
    initCounters();
    window.LF = { toggleTheme, toggleMobileMenu, setLang };

    document.querySelectorAll('[data-theme-toggle]').forEach(el => el.addEventListener('click', toggleTheme));
    document.querySelectorAll('[data-mobile-toggle]').forEach(el => el.addEventListener('click', toggleMobileMenu));
    document.querySelectorAll('[data-lang-btn]').forEach(btn =>
      btn.addEventListener('click', () => setLang(btn.dataset.langBtn))
    );

    // Blog post language switch (separate from site i18n): navigate between
    // foo.html (EN) and foo.zh.html (ZH). Mark the current language active.
    (function initPostLang() {
      const btns = document.querySelectorAll('[data-post-lang]');
      if (!btns.length) return;
      const path = location.pathname;
      const isZh = /\.zh\.html$/.test(path);
      const curLang = isZh ? 'zh' : 'en';
      btns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.postLang === curLang);
        btn.addEventListener('click', () => {
          const target = btn.dataset.postLang;
          if (target === curLang) return;
          const next = target === 'zh'
            ? path.replace(/\.html$/, '.zh.html')
            : path.replace(/\.zh\.html$/, '.html');
          if (next !== path) location.href = next + location.hash;
        });
      });
    })();

    // GitHub stars (public REST API; no auth; client-side cache)
    // The count stays hidden below STARS_MIN (avoids showing a small number that
    // would read as negative social proof; self-heals once we cross the threshold).
    const STARS_MIN = 500;
    const starEls = document.querySelectorAll('#gh-stars, [data-gh-stars]');
    const fmt = n => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
    const showStars = n => {
      if (typeof n !== 'number' || n < STARS_MIN) return;
      starEls.forEach(el => el.textContent = fmt(n));
    };
    const CACHE_KEY = 'lf-gh-stats-v1';
    const CACHE_TTL = 60 * 60 * 1000; // 1 hour
    let cached = null;
    try { cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null'); } catch (e) { }
    if (cached && Date.now() - cached.ts < CACHE_TTL) {
      showStars(cached.stars);
    }
    const next = { ts: Date.now(), stars: cached?.stars };
    if (starEls.length) {
      fetch('https://api.github.com/repos/baidu-baige/LoongForge')
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (d && typeof d.stargazers_count === 'number') {
            next.stars = d.stargazers_count;
            showStars(d.stargazers_count);
            try { localStorage.setItem(CACHE_KEY, JSON.stringify(next)); } catch (e) { }
          }
        })
        .catch(() => { });
    }

    // Contributors wall — driven entirely by the GitHub API; no manual list to
    // maintain. New contributors appear automatically as the repo changes.
    initContributors();
    initDemoVideo();
  });

  // Demo video is click-to-load: the preview is a real link to the mp4, so with
  // no JS a click just opens the file in the browser's own player. Nothing of
  // the video is fetched until the visitor asks for it, which keeps the shared
  // GitHub Pages bandwidth spent on actual plays rather than page views.
  function initDemoVideo() {
    document.querySelectorAll('[data-demo-play]').forEach(link => {
      link.addEventListener('click', e => {
        // Let the browser handle "open in new tab/window" itself.
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
        e.preventDefault();
        const video = document.createElement('video');
        video.className = 'lf-demo-video';
        video.controls = true;
        video.autoplay = true;
        video.playsInline = true;
        video.setAttribute('playsinline', '');
        if (link.dataset.w) video.setAttribute('width', link.dataset.w);
        if (link.dataset.h) video.setAttribute('height', link.dataset.h);
        video.src = link.getAttribute('href');
        // If the file cannot be fetched, fall back to the preview rather than
        // leaving an empty player behind.
        video.addEventListener('error', () => {
          video.remove();
          link.hidden = false;
        });
        link.hidden = true;
        link.parentNode.insertBefore(video, link.nextSibling);
        video.focus();
      });
    });
  }

  function initContributors() {
    const countEl = document.getElementById('gh-contrib');
    const grid = document.querySelector('[data-contrib-grid]');
    if (!countEl && !grid) return;
    const KEY = 'lf-gh-contribs-v1';
    const TTL = 6 * 60 * 60 * 1000; // 6 hours
    const MAX = 30;
    const API = 'https://api.github.com/repos/baidu-baige/LoongForge/contributors?per_page=100';
    const GRAPH_URL = 'https://github.com/baidu-baige/LoongForge/graphs/contributors';
    const isBot = u => u.type === 'Bot' || /\[bot\]$/.test(u.login || '');

    const render = (list, count) => {
      if (countEl && count) countEl.textContent = count;
      if (!grid || !Array.isArray(list) || !list.length) return;
      const frag = document.createDocumentFragment();
      list.slice(0, MAX).forEach(u => {
        const a = document.createElement('a');
        a.className = 'lf-avatar';
        a.href = u.html_url;
        a.target = '_blank';
        a.rel = 'noopener';
        a.title = u.login;
        a.setAttribute('aria-label', u.login);
        const img = document.createElement('img');
        img.src = u.avatar_url + (u.avatar_url.indexOf('?') > -1 ? '&' : '?') + 's=128';
        img.alt = '';
        img.width = 56;
        img.height = 56;
        img.loading = 'lazy';
        img.referrerPolicy = 'no-referrer';
        img.addEventListener('error', () => a.remove());
        a.appendChild(img);
        frag.appendChild(a);
      });
      if (list.length > MAX) {
        const more = document.createElement('a');
        more.className = 'lf-avatar-more';
        more.href = GRAPH_URL;
        more.target = '_blank';
        more.rel = 'noopener';
        more.textContent = '+' + (list.length - MAX);
        frag.appendChild(more);
      }
      grid.innerHTML = '';
      grid.appendChild(frag);
      grid.hidden = false;
    };
    // Render last known data first (survives an offline / rate-limited load),
    // then refresh in the background when the cache is stale.
    let cache = null;
    try { cache = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) { }
    if (cache && cache.list) render(cache.list, cache.count);
    if (cache && Date.now() - cache.ts < TTL) return;

    fetch(API)
      .then(r => r.ok ? r.json() : null)
      .then(arr => {
        if (!Array.isArray(arr) || !arr.length) return;
        const list = arr.filter(u => !isBot(u)).map(u => ({
          login: u.login, avatar_url: u.avatar_url, html_url: u.html_url
        }));
        if (!list.length) return;
        // per_page=100 returns both the list and an exact count below 100.
        const count = arr.length >= 100 ? '100+' : String(list.length);
        render(list, count);
        try { localStorage.setItem(KEY, JSON.stringify({ ts: Date.now(), count, list })); } catch (e) { }
      })
      .catch(() => { });
  }
})();
