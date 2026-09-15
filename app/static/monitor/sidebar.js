/* Sidebar view switching. Each nav item shows one view; the others stay hidden. */
(() => {
  'use strict';
  const sidebar = document.querySelector('.sidebar');
  const toggle = document.querySelector('.navigation-toggle');
  const links = [...sidebar.querySelectorAll('a.nav-item[data-view]')];
  const views = new Map([...document.querySelectorAll('.view[data-view]')].map(view => [view.dataset.view, view]));
  const chat = document.getElementById('analyst');
  const crumb = document.getElementById('section-crumb');
  const fallback = links[0].dataset.view;

  const closeMenu = () => { sidebar.dataset.open = 'false'; toggle.setAttribute('aria-expanded', 'false'); };

  function show(name, updateHistory = true) {
    const target = views.has(name) ? name : fallback;
    views.forEach((view, key) => { view.hidden = key !== target; });
    links.forEach(link => {
      const selected = link.dataset.view === target;
      link.classList.toggle('active', selected);
      if (selected) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    const label = links.find(link => link.dataset.view === target);
    if (label) crumb.textContent = label.querySelector('span').textContent;
    closeMenu();
    scrollTo({top: 0, behavior: 'instant'});
    if (updateHistory && location.hash !== '#' + target) history.pushState(null, '', '#' + target);
    try { sessionStorage.setItem('cryptolens-view', target); } catch { /* private mode */ }
    if (target === 'analyst') document.getElementById('chat-input').focus({preventScroll: true});
    window.dispatchEvent(new CustomEvent('cryptolens:view', {detail: target}));
    sizeChat();
  }

  /* The analyst has a view to itself, so the conversation runs to the bottom of
     the viewport rather than the fixed cap it needed as one panel among many. */
  function sizeChat() {
    if (views.get('analyst').hidden || innerWidth < 900) { chat.style.removeProperty('--chat-height'); return; }
    const top = Math.max(16, chat.getBoundingClientRect().top);
    chat.style.setProperty('--chat-height', Math.max(420, Math.min(1400, innerHeight - top - 24)) + 'px');
  }

  toggle.addEventListener('click', () => {
    const open = sidebar.dataset.open !== 'true';
    sidebar.dataset.open = String(open);
    toggle.setAttribute('aria-expanded', String(open));
  });
  sidebar.addEventListener('keydown', event => { if (event.key === 'Escape') { closeMenu(); toggle.focus(); } });
  links.forEach(link => link.addEventListener('click', event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    show(link.dataset.view);
  }));
  document.querySelector('.sidebar-brand').addEventListener('click', event => { event.preventDefault(); show(fallback); });
  addEventListener('popstate', () => show(location.hash.slice(1), false));
  addEventListener('resize', sizeChat);

  let stored = null;
  try { stored = sessionStorage.getItem('cryptolens-view'); } catch { /* private mode */ }
  show(location.hash.slice(1) || stored || fallback, false);
})();
