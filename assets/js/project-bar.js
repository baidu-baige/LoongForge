// Shared "Back to Projects" link for every project sub-page (projects/vlda.html, ...).
// Same pattern as the blog posts' "Back to Blog". It returns to the Projects
// page on its "All" tab. Markup lives in the page (English fallback); this script
// only translates it. To add a project page, copy the <nav data-project-bar>
// block and load this script. Language: localStorage `lf-lang`.
(function () {
  'use strict';
  const back = document.querySelector('[data-project-bar] [data-pb="back"]');
  if (!back) return;

  function getLang() {
    try {
      const saved = localStorage.getItem('lf-lang');
      if (saved === 'zh' || saved === 'en') return saved;
    } catch (e) { }
    return (navigator.language || '').toLowerCase().indexOf('zh') === 0 ? 'zh' : 'en';
  }

  function apply() {
    back.textContent = getLang() === 'zh' ? '返回项目' : 'Back to Projects';
  }

  apply();
  window.addEventListener('lf:langchange', apply);
})();
