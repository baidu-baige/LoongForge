// Papers renderer — bilingual. Powers papers.html from assets/data/papers.json.
// Adding a paper? Prepend an entry to papers.json (newest first); nothing else changes.
// When a preprint goes live: set links.paper, status to "preprint", add venue, update bibtex.
(function () {
  'use strict';
  const listEl = document.getElementById('paper-list');
  if (!listEl) return;

  const TEXT = {
    en: {
      paper: 'Paper', blog: 'Blog', project: 'Project', code: 'Code', bibtex: 'BibTeX',
      soon: 'Paper coming soon', preprint: 'Preprint', empty: 'No papers yet.',
      loadFail: 'Failed to load papers: '
    },
    zh: {
      paper: '论文', blog: '博客', project: '项目', code: '代码', bibtex: 'BibTeX',
      soon: '论文即将发布', preprint: '预印本', empty: '暂无论文。',
      loadFail: '论文加载失败：'
    }
  };

  let papers = [];
  let jumped = false;

  function getLang() {
    try {
      const saved = localStorage.getItem('lf-lang');
      if (saved === 'zh' || saved === 'en') return saved;
    } catch (e) { }
    return (navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
  }

  // Only http(s) and site-relative paths are allowed as links.
  function safeUrl(u) {
    const s = String(u || '').trim();
    if (/^https?:\/\//i.test(s)) return s;
    if (/^[a-z][a-z0-9+.-]*:/i.test(s) || s.indexOf('//') === 0) return '#';
    return s || '#';
  }

  function linkHtml(label, url) {
    const ext = /^https?:\/\//i.test(url);
    return `<a class="paper-link" href="${esc(url)}"${ext ? ' target="_blank" rel="noopener"' : ''}>${esc(label)}${ext ? ' ↗' : ''}</a>`;
  }

  function entryHtml(p, lang) {
    const t = TEXT[lang];
    const loc = p[lang] || p.en || {};
    const l = p.links || {};
    const blog = lang === 'zh' && l.blog_zh ? l.blog_zh : l.blog;

    const links = [];
    if (l.paper) links.push(linkHtml(t.paper, safeUrl(l.paper)));
    if (blog) links.push(linkHtml(t.blog, safeUrl(blog)));
    if (l.project) links.push(linkHtml(t.project, safeUrl(l.project)));
    if (l.code) links.push(linkHtml(t.code, safeUrl(l.code)));

    const status = p.status === 'soon'
      ? `<span class="chip">${esc(t.soon)}</span>`
      : `<span class="chip accent">${esc(p.venue || t.preprint)}</span>`;
    const title = l.paper
      ? `<a href="${esc(safeUrl(l.paper))}" target="_blank" rel="noopener" class="hover:text-indigo-600 dark:hover:text-indigo-300">${esc(loc.title)}</a>`
      : esc(loc.title);

    return `
      <article id="${esc(p.id)}" class="paper-row">
        <div class="paper-date">${esc(p.date)}</div>
        <div class="paper-body">
          <h2 class="paper-title">${title}</h2>
          <p class="paper-authors">${esc(p.authors || '')}</p>
          <p class="paper-summary">${esc(loc.summary || '')}</p>
          <div class="paper-meta">${status}${links.join('')}</div>
          <details class="paper-bib">
            <summary>${esc(t.bibtex)}</summary>
            <pre class="cite-pre"><code class="language-bash">${esc(p.bibtex || '')}</code></pre>
          </details>
        </div>
      </article>`;
  }

  function render() {
    const lang = getLang();
    listEl.innerHTML = papers.length
      ? papers.map(p => entryHtml(p, lang)).join('')
      : `<p class="text-gray-500">${esc(TEXT[lang].empty)}</p>`;
    if (window.LF && window.LF.attachCopyButtons) window.LF.attachCopyButtons();
    // Honour a deep link such as papers.html#taot once, after the first render.
    if (!jumped && location.hash) {
      jumped = true;
      const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
      if (target) target.scrollIntoView();
    }
  }

  fetch('assets/data/papers.json?v=20261019a', { cache: 'no-cache' })
    .then(r => r.json())
    .then(list => { papers = Array.isArray(list) ? list : []; render(); })
    .catch(err => {
      listEl.innerHTML = `<p class="text-red-500">${esc(TEXT[getLang()].loadFail)}${esc(err)}</p>`;
    });

  window.addEventListener('lf:langchange', () => { if (papers.length) render(); });
})();
