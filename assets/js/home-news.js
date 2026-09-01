// Home page "Latest post" strip — auto-populated from assets/data/posts.json.
// Adding a new blog? Just append an entry to posts.json; this strip + blog.html
// update automatically.
(function () {
  'use strict';
  const bar = document.querySelector('[data-news-bar]');
  if (!bar) return;

  const titleEl = bar.querySelector('[data-news-bar-title]');
  const dateEl = bar.querySelector('[data-news-bar-date]');
  const linkEl = bar.querySelector('[data-news-bar-link]');
  if (!titleEl || !dateEl || !linkEl) return;

  let latest = null;

  function getLang() {
    try {
      const saved = localStorage.getItem('lf-lang');
      if (saved === 'zh' || saved === 'en') return saved;
    } catch (e) { }
    return (navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  }

  function render() {
    if (!latest) return;
    const loc = latest[getLang()] || latest.en;
    if (!loc || !loc.title) return;
    titleEl.textContent = loc.title;
    dateEl.textContent = latest.date;
    dateEl.setAttribute('datetime', latest.date);
    if (loc.url) linkEl.setAttribute('href', loc.url);
    bar.hidden = false;
  }

  fetch('assets/data/posts.json?v=20260510', { cache: 'no-cache' })
    .then(r => r.json())
    .then(posts => {
      if (!Array.isArray(posts) || !posts.length) return;
      latest = posts.slice().sort((a, b) => (a.date < b.date ? 1 : -1))[0];
      render();
    })
    // A decorative strip is not worth a red error line under the hero: if the
    // feed cannot be read, stay hidden and let the nav's Blog link carry it.
    .catch(() => { });

  window.addEventListener('lf:langchange', render);
})();
