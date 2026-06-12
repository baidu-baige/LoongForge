// Blog list renderer — bilingual
(function () {
  'use strict';
  const listEl = document.getElementById('post-list');
  const featuredEl = document.getElementById('featured-post');
  const tagBar = document.getElementById('tag-bar');
  if (!listEl) return;

  let allPosts = [];
  let activeTag = 'All';

  function getLang() {
    try {
      const saved = localStorage.getItem('lf-lang');
      if (saved === 'zh' || saved === 'en') return saved;
    } catch (e) { }
    return (navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[c]);
  }

  function viewFor(p, lang) {
    const loc = p[lang] || p.en;
    const tags = (lang === 'zh' ? p.tags_zh : p.tags_en) || p.tags_en || [];
    return {
      title: loc.title,
      url: loc.url,
      author: loc.author,
      summary: loc.summary,
      date: p.date,
      featured: !!p.featured,
      tags: tags
    };
  }

  function featuredChipLabel(lang) { return lang === 'zh' ? '精选' : 'Featured'; }
  function allTagLabel(lang) { return lang === 'zh' ? '全部' : 'All'; }

  function renderFeatured(v, lang) {
    if (!v || !featuredEl) return;
    featuredEl.innerHTML = `
      <a href="${v.url}" class="group block rounded-3xl overflow-hidden border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 card-hover">
        <div class="grid md:grid-cols-2 gap-0">
          <div class="p-8 md:p-10 flex flex-col justify-center">
            <div class="flex gap-2 mb-4 flex-wrap">
              <span class="chip">${featuredChipLabel(lang)}</span>
              ${v.tags.map(t => `<span class="chip">${escapeHtml(t)}</span>`).join('')}
            </div>
            <h2 class="text-2xl md:text-3xl font-bold mb-3 group-hover:text-indigo-600 dark:group-hover:text-indigo-300">${escapeHtml(v.title)}</h2>
            <p class="text-gray-600 dark:text-gray-300 mb-5">${escapeHtml(v.summary)}</p>
            <div class="text-sm text-gray-500">${escapeHtml(v.date)} · ${escapeHtml(v.author)}</div>
          </div>
          <div class="hidden md:block bg-gradient-to-br from-indigo-500 via-violet-500 to-amber-400 relative">
            <div class="absolute inset-0 opacity-20 hero-grid"></div>
            <div class="absolute inset-0 flex items-center justify-center text-white text-7xl">🐉</div>
          </div>
        </div>
      </a>`;
  }

  function renderList(views) {
    listEl.innerHTML = views.map(v => `
      <a href="${v.url}" class="group rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 card-hover block">
        <div class="flex gap-2 mb-3 flex-wrap">
          ${v.tags.map(t => `<span class="chip">${escapeHtml(t)}</span>`).join('')}
        </div>
        <h3 class="text-xl font-semibold mb-2 group-hover:text-indigo-600 dark:group-hover:text-indigo-300">${escapeHtml(v.title)}</h3>
        <p class="text-gray-600 dark:text-gray-400 text-sm mb-4 line-clamp-3">${escapeHtml(v.summary)}</p>
        <div class="text-xs text-gray-500">${escapeHtml(v.date)} · ${escapeHtml(v.author)}</div>
      </a>`).join('');
  }

  function renderTags(tags, lang) {
    if (!tagBar) return;
    const uniq = [allTagLabel(lang), ...new Set(tags)];
    tagBar.innerHTML = uniq.map(t =>
      `<button class="tab-btn ${t === activeTag ? 'active' : ''}" data-tag="${escapeHtml(t)}">${escapeHtml(t)}</button>`
    ).join('');
    tagBar.querySelectorAll('button[data-tag]').forEach(btn => {
      btn.addEventListener('click', () => {
        activeTag = btn.dataset.tag;
        refresh();
      });
    });
  }

  function refresh() {
    const lang = getLang();
    const allLabel = allTagLabel(lang);
    // Normalize activeTag when switching languages
    if (activeTag !== allLabel && activeTag !== 'All' && activeTag !== '全部') {
      // keep tag-specific value; will be re-resolved vs new language tag list
    }
    const views = allPosts.map(p => viewFor(p, lang));
    const filtered = (activeTag === allLabel || activeTag === 'All' || activeTag === '全部')
      ? views
      : views.filter(v => v.tags.includes(activeTag));
    // Always feature the most recent post (allPosts is already date-desc sorted).
    const featured = views[0];
    renderFeatured(featured, lang);
    renderList(filtered.filter(v => v.url !== (featured && featured.url)));
    renderTags(views.flatMap(v => v.tags), lang);
  }

  fetch('assets/data/posts.json?v=20260510', { cache: 'no-cache' })
    .then(r => r.json())
    .then(posts => {
      allPosts = posts.sort((a, b) => (a.date < b.date ? 1 : -1));
      activeTag = allTagLabel(getLang());
      refresh();
    })
    .catch(err => {
      listEl.innerHTML = `<p class="text-red-500">Failed to load posts: ${err}</p>`;
    });

  // Re-render when language switches
  window.addEventListener('lf:langchange', () => {
    activeTag = allTagLabel(getLang());
    refresh();
  });
})();
