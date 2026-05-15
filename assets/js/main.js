// Zen Minimal — just enough interaction, nothing more
document.addEventListener('DOMContentLoaded', () => {
  // Respect reduced motion
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // Gentle fade-in for post items
  const items = document.querySelectorAll('.post-item, .topic-item');
  items.forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transition = 'opacity 0.5s ease';
    el.style.transitionDelay = `${i * 0.06}s`;
    requestAnimationFrame(() => { el.style.opacity = '1'; });
  });
});
