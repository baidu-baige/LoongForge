# LoongForge GitHub Pages Site

这是 [LoongForge](https://github.com/baidu-baige/LoongForge) 项目的官方 GitHub Pages 站点源码。使用纯静态 HTML + Tailwind CSS（CDN）+ 原生 JS 构建，**无需任何构建工具**。

## 📁 目录结构

```
site/
├── index.html              # 首页（Hero / Value Props / Models / Quick Start / Latest News / …）
├── docs.html               # 文档跳转（重定向到 ReadTheDocs）
├── blog.html               # 博客列表（从 posts.json 动态渲染，支持 EN/ZH）
├── about.html              # 项目介绍 / Roadmap / License / Citation
├── 404.html                # 自定义 404 页
├── blog/                   # 单篇博客文章
│   ├── index.html                                     # 重定向到 blog.html（防止目录列出）
│   ├── 2025-10-llava-onevision-case-study.html        # LLaVA-OneVision-1.5 案例（EN）
│   ├── 2025-10-llava-onevision-case-study.zh.html     # LLaVA-OneVision-1.5 案例（ZH）
│   ├── 2026-04-announcing-loongforge.html             # Announcing LoongForge（EN）
│   ├── 2026-04-announcing-loongforge.zh.html          # Announcing LoongForge（ZH）
│   ├── 2026-05-loongforge-v0.1.0-release.html         # v0.1.0 Release Notes（EN）
│   ├── 2026-05-loongforge-v0.1.0-release.zh.html      # v0.1.0 Release Notes（ZH）
│   ├── 2026-05-loongforge-dp-load-balancing.html      # DP Load Balancing（EN）
│   ├── 2026-05-loongforge-dp-load-balancing.zh.html   # DP Load Balancing（ZH）
│   ├── 2026-05-loongforge-heterogeneous-parallel-training.html      # Heterogeneous Parallel Training（EN）
│   └── 2026-05-loongforge-heterogeneous-parallel-training.zh.html   # Heterogeneous Parallel Training（ZH）
├── assets/
│   ├── css/style.css
│   ├── js/main.js          # 导航/暗黑/复制/Tab/scroll-spy/i18n
│   ├── js/blog.js          # 博客列表页（blog.html）从 posts.json 动态渲染
│   ├── js/home-news.js     # 首页 Latest News 从 posts.json 动态渲染（取最新 3 条）
│   ├── data/posts.json     # ⭐ 博客"单一数据源"，新增博客只需改这一个文件
│   └── img/logo.svg
├── robots.txt
├── sitemap.xml
├── .nojekyll               # 禁用 Jekyll，让 GH Pages 直接发布静态文件
└── README.md               # 本文件
```

## 🖥️ 本地预览

由于使用了 `fetch('assets/data/posts.json')`，必须通过 HTTP 服务器访问（不能用 `file://` 直接打开 blog.html，否则 CORS 报错）：

```bash
cd site
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000
```

## 🚀 上线方案（推荐 GitHub Actions）

在仓库根目录新增 `.github/workflows/gh-pages.yml`：

```yaml
name: Deploy GitHub Pages
on:
  push:
    branches: [master]
    paths: ['site/**', '.github/workflows/gh-pages.yml']
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./site
      - id: deployment
        uses: actions/deploy-pages@v4
```

GitHub → **Settings → Pages** 将 **Source** 设为 `GitHub Actions`。每次 `master` 分支 `site/` 目录有变更，自动构建并发布到 `https://baidu-baige.github.io/LoongForge/`。

## 🌐 自定义域名（可选）

1. 在 `site/` 下新建 `CNAME`，内容为目标域名（例如 `loongforge.baidu.com`）
2. 在 DNS 提供商设置 `CNAME` 记录指向 `baidu-baige.github.io`
3. GitHub Pages Settings 勾选 `Enforce HTTPS`

## 🔄 后续维护

### 新增一篇博客（3 步，无需改首页）

> 首页 Latest News 和 `blog.html` 列表**都**由 `site/assets/data/posts.json` 驱动，**只改一个文件就会自动同步**。

**Step 1 — 复制模板，得到两篇正文文件（EN + ZH）**
```bash
cp site/blog/2026-04-announcing-loongforge.html    site/blog/YYYY-MM-slug.html
cp site/blog/2026-04-announcing-loongforge.zh.html site/blog/YYYY-MM-slug.zh.html
```
然后编辑这两个文件：改 `<title>` / `<h1>` / 正文；把底部"上一篇 / 下一篇"链接、`POST_EN` / `POST_ZH` 常量换成新文章 slug；确认底部"查看原文"指向公开 URL。

**Step 2 — 在 `site/assets/data/posts.json` 数组开头加一条**
```json
{
  "slug": "YYYY-MM-slug",
  "date": "YYYY-MM-DD",
  "featured": false,
  "tags_en": ["Tag"],
  "tags_zh": ["标签"],
  "en": { "title": "…", "author": "The LoongForge Team", "summary": "…", "url": "blog/YYYY-MM-slug.html" },
  "zh": { "title": "…", "author": "LoongForge 团队", "summary": "…", "url": "blog/YYYY-MM-slug.zh.html" }
}
```
（`featured: true` 的条目会被置于 `blog.html` 的大封面位，整个数组里建议只有一个 featured。）

**Step 3 — 在 `site/sitemap.xml` 里加两行 URL（EN + ZH）**，提交 PR → 合并 → GH Actions 自动发布。

✅ **不再需要**：改 `index.html` 的 Latest News 卡片、改 `main.js` 里的 `news.*` i18n 键 —— 首页会根据 `posts.json` 按日期倒序自动取最新 3 条渲染。

### 更新全站样式

- 组件粒度：直接编辑对应 `*.html`
- 全局样式：`assets/css/style.css`
- 主题色：`:root` 中的 `--lf-primary` / `--lf-accent`
- 文案（导航 / Hero / Features / News）：`assets/js/main.js` 里的 `I18N_EN` / `I18N_ZH`

### 新增一个 Key Feature

首页 Key Features 分为 **5 大类**（MoE 基础优化 / 多模态并行 / 计算效率 / 长序列 / 生态与易用性）。新增特性时**归类再落条目**，不要再加平级新类别：

**Step 1 — 在 `site/assets/js/main.js` 两套 i18n 里各加一条**
```js
// EN
'feat.cat.4.item.4.t': 'Ring Attention',
'feat.cat.4.item.4.d': 'Context-parallel attention for million-length sequences.',
// ZH
'feat.cat.4.item.4.t': 'Ring Attention',
'feat.cat.4.item.4.d': '面向百万长度序列的 context-parallel 注意力实现。',
```

**Step 2 — 在 `site/index.html` 对应分类块的子特性网格里加一条 `<div class="feat-item">`**
```html
<div class="feat-item">
  <div class="font-semibold text-sm mb-1" data-i18n="feat.cat.4.item.4.t">Ring Attention</div>
  <div class="text-xs text-gray-600 dark:text-gray-400" data-i18n="feat.cat.4.item.4.d">Context-parallel attention for million-length sequences.</div>
</div>
```

**Step 3 — 如果归类明显不合适，再考虑是否需要新增大类**（通常不需要；若必须，同步更新 `main.js` 的 `feat.cat.X.title/subtitle` + `index.html` 的一整个分类块）。

## 📝 贡献

欢迎通过 PR 提交新博客、性能数据、模型支持矩阵更新、错字 / 链接修复。

---

© 2026 LoongForge Authors · Apache License 2.0
