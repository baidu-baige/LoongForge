// Home page "Latest News" timeline — rendered from assets/data/news.json.
// Adding a news item? Prepend/append an entry to news.json; the timeline updates.
// (Long-form posts live in posts.json and are shown on blog.html, not here.)
(function () {
  'use strict';
  const section = document.querySelector('[data-home-news-section]');
  const listEl = document.querySelector('[data-home-news]');
  const moreBtn = document.querySelector('[data-home-news-more]');
  if (!listEl) return;

  const VISIBLE = 3; // items shown initially
  const STEP = 5;    // items revealed per "show earlier" click
  const MONTHS_EN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const TEXT = {
    en: { read: 'Read →', view: 'Details ↗', more: n => 'Show earlier (' + n + ')', less: 'Show less' },
    zh: { read: '阅读 →', view: '详情 ↗', more: n => '展开更早动态（' + n + '）', less: '收起' }
  };

  let items = [];
  let shown = VISIBLE;

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

  // Only http(s) and site-relative paths are allowed as news links.
  function safeUrl(u) {
    const s = String(u || '').trim();
    if (!s) return '';
    if (/^https?:\/\//i.test(s)) return s;
    if (/^[a-z][a-z0-9+.-]*:/i.test(s) || s.indexOf('//') === 0) return '';
    return s;
  }

  // Escape first, then turn **x** into <b>x</b>: the only markup allowed in data.
  function richText(s) {
    return escapeHtml(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  }

  function monthLabel(ym, lang) {
    const p = String(ym).split('-');
    const y = p[0], m = parseInt(p[1], 10);
    if (!m) return escapeHtml(ym);
    return lang === 'zh' ? y + ' 年 ' + m + ' 月' : MONTHS_EN[m - 1] + ' ' + y;
  }

  function itemHtml(it, lang, extra) {
    const t = TEXT[lang];
    const loc = it[lang] || it.en || {};
    const url = safeUrl(loc.url || it.url);
    const ext = /^https?:\/\//i.test(url);
    const link = url
      ? ` <a class="lf-tl-link" href="${escapeHtml(url)}"${ext ? ' target="_blank" rel="noopener"' : ''}>${ext ? t.view : t.read}</a>`
      : '';
    return `<li class="lf-tl-item"${extra ? ' data-extra hidden' : ''}>
        <span class="lf-tl-ico" aria-hidden="true">${escapeHtml(it.icon || '•')}</span>
        <p class="lf-tl-text">${richText(loc.text || '')}${link}</p>
      </li>`;
  }

  function render() {
    if (!items.length) return;
    const lang = getLang();
    const t = TEXT[lang];
    let html = '';
    let lastMonth = '';
    let monthHtml = '';
    let monthHasVisible = false;

    function flush() {
      if (!monthHtml) return;
      html += monthHtml.replace('<!--hidden-->', monthHasVisible ? '' : ' data-extra hidden');
      monthHtml = '';
    }

    items.forEach((it, i) => {
      const extra = i >= shown;
      if (it.date !== lastMonth) {
        flush();
        lastMonth = it.date;
        monthHasVisible = false;
        monthHtml = `<li class="lf-tl-month"<!--hidden-->><time datetime="${escapeHtml(it.date)}">${monthLabel(it.date, lang)}</time></li>`;
      }
      if (!extra) monthHasVisible = true;
      monthHtml += itemHtml(it, lang, extra);
    });
    flush();
    listEl.innerHTML = html;

    if (moreBtn) {
      const all = shown >= items.length;
      moreBtn.hidden = items.length <= VISIBLE;
      moreBtn.textContent = all ? t.less : t.more(items.length - shown);
      moreBtn.setAttribute('aria-expanded', all ? 'true' : 'false');
    }
    if (section) section.hidden = false;
  }

  if (moreBtn) {
    // Reveal STEP more per click; once everything is shown the button folds back.
    moreBtn.addEventListener('click', () => {
      shown = shown >= items.length ? VISIBLE : shown + STEP;
      render();
    });
  }

  fetch('assets/data/news.json?v=20261015c', { cache: 'no-cache' })
    .then(r => r.json())
    .then(list => {
      if (!Array.isArray(list)) return;
      // Newest month first; array order breaks ties within a month.
      items = list
        .map((it, i) => ({ it: it, i: i }))
        .sort((a, b) => (a.it.date === b.it.date ? a.i - b.i : (a.it.date < b.it.date ? 1 : -1)))
        .map(x => x.it);
      render();
    })
    // Decorative section: if the feed cannot be read, stay hidden.
    .catch(() => { });

  window.addEventListener('lf:langchange', render);
})();
