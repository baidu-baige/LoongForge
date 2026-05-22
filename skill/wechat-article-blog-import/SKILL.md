---
name: wechat-article-blog-import
description: Use when importing WeChat mp.weixin articles into a static blog or GitHub Pages site, especially with paired Chinese/English posts, source-fidelity requirements, removed WeChat media, syntax-highlighted snippets, metadata indexes, or local preview checks.
---

# WeChat Article Blog Import

## Overview

Import WeChat articles as static blog posts with source fidelity first, then site integration. Core principle: preserve Chinese source text, explicitly document allowed transformations, and verify the rendered site against freshly fetched source before claiming completion.

## When to Use

Use for:
- mp.weixin / WeChat articles becoming static HTML/Markdown blog posts.
- Paired Chinese/English pages where Chinese must stay faithful and English is translated.
- Static sites with post indexes, sitemap, README, or homepage cards.
- Articles containing WeChat images, code snippets, ASCII diagrams, or command blocks.

Do not use for generic blog writing where there is no external source to preserve.

## Default User Experience

When the user provides a WeChat/blog link and asks to add it to the current GitHub Pages or static blog site, proceed end-to-end using existing site conventions. Do not ask for extra requirements when the target site, source link, and desired import are clear.

Ask only when:
- the source article cannot be extracted after approved fallback attempts;
- multiple target sites or branches are plausible;
- the project has no existing blog conventions to reuse;
- a policy choice is genuinely ambiguous, such as whether to keep images or author names.

Default outcome: import the article, update indexes/site references, run local preview and checks, then give the user a concise completion summary with preview URL and verification evidence.

## Workflow

1. **Inspect existing blog patterns**
   - Read one recent Chinese post, one English post, post index data, sitemap, and shared JS/CSS.
   - Reuse existing slug, language-switch, author/date, Prism, and navigation conventions.

2. **Extract source text with fallback**
   - Try direct fetch only to confirm accessibility; WeChat often returns verification pages.
   - If direct fetch fails, use the approved local extractor/tool or user-provided saved HTML/Markdown.
   - Save or keep refetchable source artifacts until final verification.
   - Record title, date, author policy, original URL, headings, body, code blocks, and images.

3. **Convert content deliberately**
   - Chinese page: preserve wording and order; only convert structure (headings/lists/code) and approved cleanup.
   - English page: translate faithfully with the same structure.
   - Remove WeChat-hosted images unless explicitly requested otherwise.
   - Wrap code/commands/ASCII in `<pre><code class="language-*">`; use `language-python` for Python-like pseudocode, `language-bash` for shell, `language-yaml` for YAML, and `language-text` for diagrams/logs.
   - Avoid long box-drawing diagrams when CJK/full-width text causes alignment issues; prefer stable arrow/indent diagrams.

4. **Update site integration**
   - Update post data/index files with correct dates and descending sort.
   - Update sitemap and README if the site tracks explicit post files.
   - Update homepage cards/benchmark text only when requested or directly relevant.
   - If shared JS text changes and the page references an unversioned JS file, add a cache-busting query string following site convention.

5. **Load syntax dependencies**
   - Adding `language-*` classes is not enough. Ensure Prism/Highlight.js language components are loaded on every page that uses them.
   - Keep text diagrams as text blocks; do not force them into Python highlighting.

6. **Local preview and source comparison**
   - Serve locally with the project’s static preview command.
   - Check homepage, blog index, every new post, language switches, JS/CSS assets, and post data over HTTP.
   - Refetch or reread source artifacts and compare Chinese pages section-by-section.
   - Allowed source differences must be explicit: removed WeChat images, HTML escaping, approved diagram reformatting, and metadata wrappers.

## Quick Reference Checks

| Area | Required check |
| --- | --- |
| Source fidelity | Fresh source vs Chinese page heading/paragraph coverage |
| WeChat artifacts | No `环境异常`, `去验证`, `mmbiz.qpic.cn`, `wx_fmt`, or stray `<img>` |
| Code blocks | No bare `<pre><code>`; all blocks have useful `language-*` classes |
| Highlighting | Language component scripts exist for used languages |
| Metadata | Post index JSON valid, sorted, and points to existing files |
| Navigation | Previous/next/blog index/language switch links resolve |
| Cache | Updated shared JS has version bump when browser caching can hide changes |
| Preview | Changed pages and assets return HTTP 200 locally |

## Final Verification Pattern

Use a small script or commands that prove:
- post index parses and all referenced files exist;
- sitemap/README include new URLs when applicable;
- changed pages have no WeChat or conversion artifacts;
- representative source headings exist in the Chinese page;
- changed local URLs return 200;
- `git diff --stat` and `git status --short` match expected scope.

Only report completion with the actual checks run.

## Final Summary Contract

Keep the final response concise. Include:
- imported article title, slug, and created Chinese/English page paths;
- metadata/index/site files updated;
- local preview URL(s) checked;
- verification commands or checks that passed, with actual results;
- source-fidelity result for the Chinese page and any allowed differences, such as removed WeChat images or approved diagram reformatting;
- git diff/status scope and anything left for the user to decide.

Do not claim completion if local preview or source comparison was skipped. Say exactly what was not verified and why.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Trusting WeChat fetch output without checking for verification pages | Search for verification text and refetch/ask for local source if blocked |
| Changing Chinese wording during cleanup | Preserve text; only change markup unless user approves edits |
| Removing images but leaving broken WeChat URLs | Grep for `mmbiz.qpic.cn`, `wx_fmt`, and `<img>` |
| Adding `language-python` but no Prism Python component | Add the matching component script and hard-refresh/cache-bust |
| Box-drawing diagrams misalign in CJK text | Use simpler arrow/indent text diagrams |
| Updating JS i18n but user sees old copy | Add or update a cache-busting query on the script reference |
| Declaring done after local HTML checks only | Also compare against freshly fetched/source artifacts and inspect diff scope |
