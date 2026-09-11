(() => {
  const links = [...document.querySelectorAll('.sidebar a.nav-item[href^="#"]')];
  if (!links.length) return;

  const activate = link => {
    links.forEach(item => {
      const selected = item === link;
      item.classList.toggle('active', selected);
      if (selected) item.setAttribute('aria-current', 'page');
      else item.removeAttribute('aria-current');
    });
  };

  links.forEach(link => link.addEventListener('click', () => activate(link)));

  const sections = links
    .map(link => ({link, section: document.querySelector(link.getAttribute('href'))}))
    .filter(item => item.section);

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries
        .filter(entry => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      const match = sections.find(item => item.section === visible.target);
      if (match) activate(match.link);
    }, {rootMargin: '-16% 0px -68% 0px', threshold: [0, .1, .35]});
    sections.forEach(item => observer.observe(item.section));
  }
})();
