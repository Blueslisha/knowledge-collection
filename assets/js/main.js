document.addEventListener('DOMContentLoaded', () => {
  // Parallax grain overlay offset (subtle)
  const grain = document.querySelector('.grain');
  if (grain) {
    document.addEventListener('mousemove', (e) => {
      const x = (e.clientX / window.innerWidth) * 4;
      const y = (e.clientY / window.innerHeight) * 4;
      grain.style.backgroundPosition = `${x}px ${y}px`;
    });
  }

  // Staggered card reveal (respects reduced motion)
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    document.querySelectorAll('.post-card, .topic-card').forEach(el => {
      el.style.animation = 'none';
      el.style.opacity = '1';
      el.style.transform = 'none';
    });
  }
});
