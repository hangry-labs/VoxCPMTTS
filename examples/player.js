function renderExamples() {
  const grid = document.querySelector("[data-example-grid]");
  if (!grid || !Array.isArray(window.VOICE_EXAMPLES)) {
    return;
  }

  grid.innerHTML = window.VOICE_EXAMPLES.map((item) => `
    <article class="brand-card p-5">
      <div class="mb-4 flex items-start justify-between gap-3 border-b border-orange-500/20 pb-3">
        <div>
          <h2 class="text-lg font-bold text-orange-400">${item.name}</h2>
          <p class="mt-1 text-sm text-[#ffb076]/80">${item.language}</p>
        </div>
        <span class="rounded-full border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-gray-300">${item.mode}</span>
      </div>
      <p class="mb-4 text-sm leading-6 text-[#ffb076]">${item.text}</p>
      <pre class="overflow-auto rounded-lg border border-orange-500/20 bg-black/50 p-3 text-xs text-orange-100"><code>${item.api}</code></pre>
    </article>
  `).join("");
}

renderExamples();
