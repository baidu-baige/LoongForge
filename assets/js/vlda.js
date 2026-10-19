// LoongForge-VLDA project page — page-local i18n and figure lightbox.
// English lives in the HTML (elements carry data-t="key"); this file supplies the
// Chinese overrides. The first run caches each element's original markup so the
// language can be switched back and forth. Language comes from the site-wide
// localStorage key `lf-lang`; main.js fires `lf:langchange` when it changes.
(function () {
  'use strict';

  const ZH = {
    'hero.title': '<b>Vision-Language-Delta-Action：</b>学习紧凑的 Delta，作为机器人操作的可执行状态转移接口。',
    'hero.aff1': '哈尔滨工业大学（深圳）',
    'hero.aff3': '深圳河套学院',
    'hero.equal': '* 共同第一作者',
    'hero.corr': '† 通讯作者',
    'hero.code': '代码 ↗',
    'hero.paper':'论文 · arXiv 即将上线',
    'd.h': 'Delta 在看什么',
    'd.p': 'Delta 集中在运动与交互的区域，而不是静态场景。',
    's.libero': 'LIBERO 平均',
    's.plus': 'LIBERO-Plus（零样本）',
    's.rt': 'RoboTwin 2.0（10 个困难任务）',
    's.r1': '真机 · R1 Lite',
    's.ag': '真机 · AgileX',
    's.gpu': '单卡即可完成真机推理',

    'i.eyebrow': '核心思路',
    'i.h': '不必预测完整的未来，<em>只需抓住“变化”。</em>',
    'fig1.cap': '<b>图 1.</b> VLA、WAM 与 VLDA 的对比。VLDA 保留转移信号，舍弃冗余的未来像素。',
    'open': '查看大图 ↗',
    'close': '关闭 ×',
    'i.p1': '<b>一个动作究竟改变了什么？</b>机器人操作本质上是一连串状态转移，而每一帧画面的大部分内容都保持不变。VLA 模型需要隐式地推断这种变化；世界-动作模型（WAM）则预测完整的未来画面，把容量浪费在不会移动的像素上。',
    'i.p2': 'VLDA 只把<b>真正发生变化的部分</b>提炼成紧凑的 Delta，并作为显式接口交给动作头：先理解场景，再表达转移，最后执行动作。',
    'flow.u': '理解',
    'flow.us': '视觉与语言被编码为任务相关的表征。',
    'flow.t': '转移 · Δ',
    'flow.ts': '紧凑的 Delta 刻画任务所需的局部与全局变化。',
    'flow.e': '执行',
    'flow.es': 'Delta 对动作头进行精炼，推理时无需未来画面。',
    'i.c2': '状态转移建模',
    'i.c3': '流匹配动作生成',

    'm.eyebrow': '方法',
    'm.h': '从 Delta 到动作，<em>只需三步。</em>',
    'm.p': 'V-JEPA 2 的 EMA 教师提供目标转移；两个滤波模块将其提炼为 Delta；一个精炼模块再把 Delta 送回动作头。',
    'm.ltf': '选出转移幅度最大的 top-64 个区域，并用 Sinkhorn 最优传输与预测的 Delta 做匹配，避免 Delta 被静态背景稀释。',
    'm.gta': '把局部 patch 锚定到转移的全局摘要上，使 Delta 与整体任务保持一致。',
    'm.ddar': '<b>DAMG</b> 通过 InfoNCE 损失把 Delta 落到动作流形上；<b>DSAF</b> 用基于 Delta 的噪声偏置与前缀 token 为动作流提供起点。',
    'fig2.cap': '<b>图 2.</b> VLDA 整体架构：教师构建目标转移，LT-Filter 与 GT-Anchor 提炼 Delta，DDAR 将其送入动作生成。',
    'tr.st1': '阶段一',
    'tr.st2': '阶段二',
    'sc.vs': '对比',
    'r.dlt': '较 π0.5 提升（百分点）',
    'tr.s1': '学习状态转移',
    'tr.s1p': '无需动作标注，从教师处预训练 Delta。',
    'tr.s2': '学习动作',
    'tr.s2p': '以 Delta 为接口训练动作头，并保留阶段一损失作为正则。',
    'bl.p': '<b>基于 LoongForge 构建。</b>VLDA 使用 LoongForge 具身训练栈：DDP、ZeRO-1、FSDP、HSDP，按组件冻结与检查点管理。训练使用 8 张 GPU。推理不需要未来观测，训练完成后即可丢弃 EMA 教师。',

    'r.eyebrow': '实验结果',
    'r.h': '更强，而且<em>始终如一。</em>',
    'r.p': '三个仿真基准与一组消融实验，我们的结果已高亮。',
    'r.tab.abl': '消融',
    'r.lb.l': 'LIBERO 平均',
    'r.lb.h': '平均成功率最高，长程任务提升最大。',
    'r.lb.p': 'VLDA 平均领先 π0.5 1.6 个点，也超过其余所有基线，在 Long 套件上达到 95.2%。',
    'r.lb.m': 'Spatial · Object · Goal · Long',
    'r.lb.c': 'LIBERO 成功率（%）',
    'r.pl.l': '个百分点，领先 π0.5（零样本）',
    'r.pl.h': '世界被扰动时依然稳健。',
    'r.pl.p': '在七个扰动维度上均领先 π0.5，并在其中六个维度上领先最强基线 GuidedVLA（平均 +9.1）。在 Long 套件上差距进一步拉大：领先 π0.5 13.4 个点，领先 GuidedVLA 12.1 个点。',
    'r.pl.m': '7 种扰动维度 · 10,030 次 rollout',
    'r.pl.c': 'LIBERO-Plus 零样本成功率（%）',
    'r.rb.l': '个百分点，领先 π0.5（10 个困难任务）',
    'r.rb.h': '在复杂的状态转移任务上提升最大。',
    'r.rb.p': '在所选的每个任务上均为最优或并列最优。Put Bottles Dustbin 提升 +46，Stacking Three +30，Place Dual Shoes +27，Open Microwave +27。全部 50 个任务上，VLDA 为 50.86%，π0.5 为 43.84%。',
    'r.rb.m': '10 个困难任务 · 50 任务平均 50.86% 对 43.84%',
    'r.rb.c': 'RoboTwin 2.0 成功率（%）',
    'r.ab.l': '个百分点，篮子搬运任务提升',
    'r.ab.h': '每个模块都有其价值。',
    'r.ab.p': '去掉任何一个部分都会损失精度。LT-Filter 与 GT-Anchor 单独使用都有提升，两者结合提升最大，在真机上尤为明显。',
    'r.ab.m': 'R1 Lite 篮子搬运：两者皆无 48% → 两者皆有 72%',
    'r.ab.c': '消融实验（除特别说明外为 LIBERO 平均）',
    'a.dsaf': '去掉 DSAF',
    'a.dsaf.d': '移除 Delta 种子动作流',
    'a.pre': '去掉转移预训练',
    'a.pre.d': '跳过阶段一',
    'a.full': '预测完整未来',
    'a.full.d': '预测整帧画面而非 Delta',
    'a.damg': '去掉 DAMG（LIBERO-Plus）',
    'a.damg.d': '加入 DAMG 后光照 +6.3，背景 +3.7',
    'a.t.h': 'LT-Filter 与 GT-Anchor',
    'a.t.n': '仿真列为 LIBERO 成功率（%）；真机列依次为叠圈、清扫桌面、传送带分拣、篮子搬运（R1 Lite）。单独使用都有提升，两者结合提升最大。',

    'w.eyebrow': '真机实验',
    'w.h': '真机上，<em>依然领先。</em>',
    'w.p': '在 GALAXEA R1 Lite 上完成四个长程任务、每个任务 25 次试验，并在第二个平台 AgileX Cobot Magic 上验证。数值为最终阶段成功率（%）。',
    'r.t1': '叠圈',
    'r.t2': '清扫桌面',
    'r.t3': '传送带分拣',
    'r.t4': '篮子搬运',
    'r.plat': '平台（平均）',
    'w.big': '个百分点，领先 π0.5（R1 Lite）',
    'w.lead': '四个任务平均，VLDA 达到 73.0%，π0.5 为 56.0%；在第二个机器人上优势依然保持。篮子搬运任务的垫子位置在训练分布之外，VLDA 的最终放置成功率仍达 72%。',
    'fig3.cap': '<b>图 3.</b> 操作过程中学到的 Delta 可视化。',

    'c.eyebrow': '动手使用',
    'c.h': '代码与<em>引用。</em>',
    'c.gh.h': 'LoongForge-VLDA Cookbook',
    'c.gh': 'VLDA 的训练代码、配置与示例都在 LoongForge 中。',
    'c.bib': '引用',
    'c.note': 'arXiv 条目将在预印本公开后补充。',
    'f.tag': 'LoongForge 项目',
    'f.top': '回到顶部 ↑'
  };

  const nodes = Array.prototype.slice.call(document.querySelectorAll('[data-t]'));
  // Cache the English markup once, before any override is applied.
  nodes.forEach(function (el) { el.setAttribute('data-en', el.innerHTML); });

  function getLang() {
    try {
      const saved = localStorage.getItem('lf-lang');
      if (saved === 'zh' || saved === 'en') return saved;
    } catch (e) { }
    return (navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  }

  function apply() {
    const lang = getLang();
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    nodes.forEach(function (el) {
      const key = el.getAttribute('data-t');
      el.innerHTML = (lang === 'zh' && Object.prototype.hasOwnProperty.call(ZH, key))
        ? ZH[key]
        : el.getAttribute('data-en');
    });
  }

  apply();
  window.addEventListener('lf:langchange', apply);

  // Figure lightbox: "View full figure" opens the original image full screen.
  const box = document.getElementById('vl-lightbox');
  if (box) {
    const img = box.querySelector('img');
    const closeBtn = document.getElementById('vl-lightbox-close');
    const triggers = Array.prototype.slice.call(document.querySelectorAll('[data-open]'));
    let opener = null;

    function open(src, alt, from) {
      opener = from;
      img.src = src;
      img.alt = alt || '';
      box.classList.add('open');
      document.body.style.overflow = 'hidden';
      if (closeBtn) closeBtn.focus();
    }
    function close() {
      box.classList.remove('open');
      img.removeAttribute('src');
      document.body.style.overflow = '';
      if (opener) opener.focus();
    }

    triggers.forEach(function (b) {
      b.addEventListener('click', function () {
        const fig = b.closest('figure');
        const thumb = fig && fig.querySelector('img');
        open(b.getAttribute('data-open'), thumb && thumb.alt, b);
      });
    });
    if (closeBtn) closeBtn.addEventListener('click', close);
    box.addEventListener('click', function (e) { if (e.target === box) close(); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && box.classList.contains('open')) close();
    });
  }
})();
