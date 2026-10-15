// Projects renderer — bilingual. Powers projects.html and the home-page
// "Powered by LoongForge" strip from one data source: assets/data/projects.json.
// Adding a project? Append an entry to projects.json; both places update.
(function () {
  'use strict';
  const pageEl = document.getElementById('project-groups');
  const tabsEl = document.getElementById('project-tabs');
  const homeSection = document.querySelector('[data-home-projects-section]');
  const homeEl = document.querySelector('[data-home-projects]');
  if (!pageEl && !homeEl) return;

  const GROUPS = ['loongforge', 'community'];
  const TEXT = {
    en: {
      all: 'All',
      newBadge: 'NEW',
      featured: 'Featured',
      loongforge: { title: 'LoongForge Projects', desc: 'Projects built and maintained by the LoongForge team.', empty: 'More LoongForge projects are on the way.' },
      community: { title: 'Built with LoongForge', desc: 'Open-source work from the community, trained with LoongForge or its predecessor AIAK-Training-LLM.', empty: 'No community projects yet.' },
      loadFail: 'Failed to load projects: '
    },
    zh: {
      all: '全部',
      newBadge: 'NEW',
      featured: '精选',
      loongforge: { title: 'LoongForge 项目', desc: '由 LoongForge 团队构建和维护的项目。', empty: '更多 LoongForge 项目即将上线。' },
      community: { title: '社区项目', desc: '社区基于 LoongForge 或其前身 AIAK-Training-LLM 训练的开源工作。', empty: '暂无社区项目。' },
      loadFail: '项目加载失败：'
    }
  };

  let allProjects = [];
  let activeFilter = (function () {
    const h = (location.hash || '').replace('#', '');
    return GROUPS.indexOf(h) >= 0 ? h : 'all';
  })();

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

  // Only http(s) and site-relative paths are allowed as project links.
  function safeUrl(u) {
    const s = String(u || '').trim();
    if (/^https?:\/\//i.test(s)) return s;
    if (/^[a-z][a-z0-9+.-]*:/i.test(s) || s.indexOf('//') === 0) return '#';
    return s || '#';
  }
  function isExternal(u) { return /^https?:\/\//i.test(u); }

  function viewFor(p, lang) {
    const loc = p[lang] || p.en || {};
    return {
      category: p.category,
      icon: p.icon || '📦',
      badge: p.badge,
      featured: !!p.featured,
      home: !!p.home,
      org: p.org || '',
      url: safeUrl(loc.url || p.url),
      title: loc.title || '',
      summary: loc.summary || '',
      tags: (lang === 'zh' ? p.tags_zh : p.tags_en) || p.tags_en || []
    };
  }

  function cardHtml(v, lang, compact) {
    const t = TEXT[lang];
    const ext = isExternal(v.url);
    const attrs = ext ? ' target="_blank" rel="noopener"' : '';
    const span = (v.featured && !compact) ? ' lg:col-span-2' : '';
    const badge = v.badge === 'new'
      ? `<span class="text-[10px] font-bold tracking-wider px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300">${t.newBadge}</span>`
      : (v.featured && !compact ? `<span class="chip accent">${t.featured}</span>` : '');
    return `
      <a href="${escapeHtml(v.url)}"${attrs} class="group flex flex-col rounded-[1.25rem] p-7 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 card-hover${span}">
        <div class="flex items-start justify-between mb-3">
          <div class="flex items-center gap-2">
            <span class="text-2xl" aria-hidden="true">${escapeHtml(v.icon)}</span>${badge}
          </div>
          <span class="text-gray-400 group-hover:text-indigo-500 transition-colors" aria-hidden="true">${ext ? '↗' : '→'}</span>
        </div>
        <h3 class="font-semibold mb-1 group-hover:text-indigo-600 dark:group-hover:text-indigo-300">${escapeHtml(v.title)}</h3>
        <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">${escapeHtml(v.summary)}</p>
        <div class="mt-auto pt-2 flex flex-wrap items-center justify-between gap-2">
          <div class="flex flex-wrap gap-1.5">${v.tags.map(x => `<span class="chip">${escapeHtml(x)}</span>`).join('')}</div>
          <span class="text-xs text-gray-500">${escapeHtml(v.org)}</span>
        </div>
      </a>`;
  }

  function groupHtml(group, items, lang) {
    const g = TEXT[lang][group];
    const sorted = items.slice().sort((a, b) => (b.featured ? 1 : 0) - (a.featured ? 1 : 0));
    const body = sorted.length
      ? `<div class="grid md:grid-cols-2 lg:grid-cols-3 gap-6">${sorted.map(v => cardHtml(v, lang, false)).join('')}</div>`
      : `<div class="rounded-[1.25rem] border border-dashed border-gray-300 dark:border-gray-700 p-10 text-center text-gray-500 dark:text-gray-400">
           <div class="text-3xl mb-2" aria-hidden="true">🐉</div>${escapeHtml(g.empty)}
         </div>`;
    return `
      <section class="mb-14 last:mb-0">
        <h2 class="text-2xl md:text-3xl font-bold mb-2">${escapeHtml(g.title)}</h2>
        <p class="text-gray-600 dark:text-gray-400 mb-6">${escapeHtml(g.desc)}</p>
        ${body}
      </section>`;
  }

  function renderTabs(lang) {
    if (!tabsEl) return;
    const t = TEXT[lang];
    const tabs = [['all', t.all]].concat(GROUPS.map(g => [g, t[g].title]));
    tabsEl.innerHTML = tabs.map(x =>
      `<button type="button" class="tab-btn ${x[0] === activeFilter ? 'active' : ''}" data-filter="${x[0]}">${escapeHtml(x[1])}</button>`
    ).join('');
    tabsEl.querySelectorAll('button[data-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        activeFilter = btn.dataset.filter;
        try {
          history.replaceState(null, '', activeFilter === 'all' ? location.pathname + location.search : '#' + activeFilter);
        } catch (e) { }
        render();
      });
    });
  }

  function renderPage(views, lang) {
    if (!pageEl) return;
    renderTabs(lang);
    const groups = activeFilter === 'all' ? GROUPS : [activeFilter];
    pageEl.innerHTML = groups.map(g => groupHtml(g, views.filter(v => v.category === g), lang)).join('');
  }

  function renderHome(views, lang) {
    if (!homeEl) return;
    const items = views.filter(v => v.home);
    if (!items.length) return;
    homeEl.innerHTML = items.map(v => cardHtml(v, lang, true)).join('');
    if (homeSection) homeSection.hidden = false;
  }

  function render() {
    if (!allProjects.length) return;
    const lang = getLang();
    const views = allProjects.map(p => viewFor(p, lang));
    renderPage(views, lang);
    renderHome(views, lang);
  }

  fetch('assets/data/projects.json?v=20261015b', { cache: 'no-cache' })
    .then(r => r.json())
    .then(list => {
      allProjects = Array.isArray(list) ? list : [];
      render();
    })
    .catch(err => {
      // The home strip is decorative: stay hidden. The projects page says so.
      if (pageEl) pageEl.innerHTML = `<p class="text-red-500">${TEXT[getLang()].loadFail}${escapeHtml(err)}</p>`;
    });

  window.addEventListener('lf:langchange', render);
})();
