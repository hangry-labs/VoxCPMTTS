const state = {
  cardAudios: [],
  currentCard: null,
  introAudio: new Audio(),
  manifest: null,
  selected: null,
  volume: 0.85,
  lastVolume: 0.85,
};

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function assetUrl(file) {
  return `assets/${file}`;
}

function formatTime(value) {
  if (!Number.isFinite(value)) {
    return "0:00";
  }

  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function metaFor(language) {
  return window.LANGUAGE_META?.[language.slug] || {
    nativeName: language.language,
    icon: language.language.slice(0, 2).toUpperCase(),
  };
}

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

function sampleBadge(sample) {
  const parts = [];
  if (sample.voice_profile) {
    parts.push(sample.voice_profile.replace(/_/g, " "));
  }
  if (!sample.voice_profile && sample.control) {
    parts.push(sample.control);
  }
  if (sample.nonverbal_tag) {
    parts.push(sample.nonverbal_tag);
  }
  return parts.length ? parts.join(" / ") : "voice sample";
}

function setNowPlaying(label) {
  const target = document.querySelector("[data-now-playing]");
  if (target) {
    target.textContent = label || "Ready";
  }
}

function pauseCards(exceptAudio = null) {
  state.cardAudios.forEach((audio) => {
    if (audio !== exceptAudio) {
      audio.pause();
    }
  });
}

function clearCurrentCard() {
  if (state.currentCard) {
    state.currentCard.classList.remove("is-playing");
    state.currentCard.removeAttribute("aria-current");
  }
  state.currentCard = null;
}

function applyAudioVolume(audio) {
  const isMuted = state.volume <= 0.001;
  audio.volume = state.volume;
  audio.muted = isMuted;
}

function allAudios() {
  return [state.introAudio, ...state.cardAudios];
}

function updateVolumeControl() {
  const volumeButton = document.querySelector(".volume-button");
  const volumeSlider = document.querySelector(".volume-slider");
  const volumeIconOn = document.querySelector(".volume-icon-on");
  const volumeIconMuted = document.querySelector(".volume-icon-muted");
  const isMuted = state.volume <= 0.001;

  allAudios().forEach(applyAudioVolume);

  if (volumeSlider) {
    volumeSlider.value = state.volume.toString();
  }

  if (volumeButton) {
    volumeButton.dataset.muted = isMuted.toString();
    volumeButton.setAttribute("aria-label", isMuted ? "Unmute audio" : "Mute audio");
  }

  if (volumeIconOn && volumeIconMuted) {
    volumeIconOn.hidden = isMuted;
    volumeIconMuted.hidden = !isMuted;
    volumeIconOn.style.display = isMuted ? "none" : "block";
    volumeIconMuted.style.display = isMuted ? "block" : "none";
  }
}

function initVolumeControl() {
  const volumeButton = document.querySelector(".volume-button");
  const volumeSlider = document.querySelector(".volume-slider");
  if (volumeSlider) {
    state.volume = Number.parseFloat(volumeSlider.value || "0.85");
    state.lastVolume = state.volume > 0 ? state.volume : 0.85;
    volumeSlider.addEventListener("input", () => {
      state.volume = Number.parseFloat(volumeSlider.value);
      if (state.volume > 0) {
        state.lastVolume = state.volume;
      }
      updateVolumeControl();
    });
  }

  if (volumeButton) {
    volumeButton.addEventListener("click", () => {
      if (state.volume > 0) {
        state.lastVolume = state.volume;
        state.volume = 0;
      } else {
        state.volume = state.lastVolume || 0.85;
      }
      updateVolumeControl();
    });
  }

  updateVolumeControl();
}

function playIntro(file, label) {
  pauseCards();
  clearCurrentCard();
  state.introAudio.pause();
  state.introAudio.src = assetUrl(file);
  state.introAudio.currentTime = 0;
  applyAudioVolume(state.introAudio);
  setNowPlaying(label);
  warmAudio(state.introAudio);
  state.introAudio.play().catch(() => {
    setNowPlaying("Press play again if your browser blocked autoplay.");
  });
}

function renderLanguageButtons() {
  const target = document.querySelector("[data-language-list]");
  if (!target) {
    return;
  }

  target.innerHTML = state.manifest.languages
    .map((language) => {
      const meta = metaFor(language);
      return `
        <button class="language-button" type="button" data-language-button="${escapeHtml(language.slug)}" aria-pressed="false">
          <span class="language-icon" aria-hidden="true">${escapeHtml(meta.icon)}</span>
          <span>${escapeHtml(meta.nativeName)}</span>
        </button>
      `;
    })
    .join("");

  target.querySelectorAll("[data-language-button]").forEach((button) => {
    button.addEventListener("click", () => chooseLanguage(button.dataset.languageButton, true));
  });
}

function renderAudioCard({ classes = "", eyebrow, title, description, file, label, badge = "" }) {
  return `
    <article class="brand-card p-5 ${classes}" data-audio-card data-audio-label="${escapeHtml(label)}">
      <div class="mb-4 flex items-start justify-between gap-3 border-b border-orange-500/20 pb-3">
        <div class="min-w-0">
          <p class="text-xs font-extrabold uppercase tracking-[0.2em] text-orange-300">${escapeHtml(eyebrow)}</p>
          <h3 class="mt-2 text-lg font-bold text-orange-100">${escapeHtml(title)}</h3>
        </div>
        ${badge ? `<span class="rounded-full border border-orange-500/30 bg-orange-500/10 px-2 py-1 text-xs font-semibold uppercase text-gray-300">${escapeHtml(badge)}</span>` : ""}
      </div>
      <p class="mb-4 text-sm leading-6 text-[#ffd0a3]">${escapeHtml(description)}</p>
      <audio preload="metadata" src="${escapeHtml(assetUrl(file))}"></audio>
    </article>
  `;
}

function renderSelectedLanguage() {
  const language = state.selected;
  const target = document.querySelector("[data-language-detail]");
  if (!target || !language) {
    return;
  }

  state.introAudio.pause();
  pauseCards();
  clearCurrentCard();

  const meta = metaFor(language);
  const clone = language.clone[0];
  const randomSamples = language.random
    .map((sample, index) =>
      renderAudioCard({
        eyebrow: `Sample ${String(index + 1).padStart(2, "0")}`,
        title: sampleBadge(sample),
        description: sample.text,
        file: sample.file,
        label: `${meta.nativeName} sample ${index + 1}`,
        badge: "Play",
      }),
    )
    .join("");

  target.innerHTML = `
    <section>
      <div class="brand-card mb-4 cursor-default p-5">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div class="flex min-w-0 items-center gap-4">
            <span class="language-icon large" aria-hidden="true">${escapeHtml(meta.icon)}</span>
            <div>
              <h2 class="text-3xl font-extrabold text-orange-300">${escapeHtml(meta.nativeName)}</h2>
              <p class="mt-1 text-sm font-semibold text-[#ffb076]/75">${escapeHtml(language.language)}</p>
            </div>
          </div>
          <button class="filter-button is-active" type="button" data-random-intro="${escapeHtml(language.slug)}">Play random intro</button>
        </div>
      </div>

      <div class="mb-4 grid grid-cols-1 gap-4">
        ${renderAudioCard({
          classes: "clone-card",
          eyebrow: "Cross-language clone demo",
          title: "Same English reference voice, different language",
          description:
            "Cloned from examples/original_clone.mp3. The reference is English, and this sample shows how the voice identity carries into the selected language.",
          file: clone.file,
          label: `${meta.nativeName} cloned voice demo`,
          badge: "Clone",
        })}
      </div>

      <div class="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        ${randomSamples}
      </div>
    </section>
  `;

  target.querySelector("[data-random-intro]")?.addEventListener("click", () => {
    const intro = randomItem(language.intro);
    playIntro(intro.file, `${meta.nativeName} intro`);
  });

  enhanceAudioCards(target);
}

function chooseLanguage(slug, autoplayIntro = true) {
  const language = state.manifest.languages.find((item) => item.slug === slug);
  if (!language) {
    return;
  }

  state.selected = language;
  renderSelectedLanguage();

  document.querySelectorAll("[data-language-button]").forEach((button) => {
    const active = button.dataset.languageButton === slug;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });

  if (autoplayIntro) {
    const intro = randomItem(language.intro);
    playIntro(intro.file, `${metaFor(language).nativeName} intro`);
  } else {
    setNowPlaying("Ready");
  }
}

function warmAudio(audio) {
  if (!audio || audio.readyState > 0) {
    return;
  }
  audio.load();
}

function enhanceAudioCards(container) {
  state.cardAudios = Array.from(container.querySelectorAll("[data-audio-card] audio"));

  state.cardAudios.forEach((audio) => {
    const card = audio.closest("[data-audio-card]");
    const label = card?.dataset.audioLabel || "voice sample";
    applyAudioVolume(audio);
    audio.removeAttribute("controls");

    if (!card || audio.nextElementSibling?.classList.contains("player")) {
      return;
    }

    const controls = document.createElement("div");
    controls.className = "player";
    controls.innerHTML = `
      <button class="progress-button" type="button" aria-label="Seek sample">
        <span class="progress-track" aria-hidden="true">
          <span class="progress-fill"></span>
          <span class="progress-knob"></span>
        </span>
      </button>
      <span class="duration">0:00</span>
    `;
    audio.insertAdjacentElement("afterend", controls);

    const progressButton = controls.querySelector(".progress-button");
    const progressFill = controls.querySelector(".progress-fill");
    const progressKnob = controls.querySelector(".progress-knob");
    const duration = controls.querySelector(".duration");

    function setProgress(value) {
      const progress = Math.max(0, Math.min(100, value));
      progressFill.style.width = `${progress}%`;
      progressKnob.style.left = `${progress}%`;
    }

    function seekToPosition(event) {
      if (!Number.isFinite(audio.duration) || audio.duration <= 0) {
        return;
      }
      const rect = progressButton.getBoundingClientRect();
      const position = (event.clientX - rect.left) / rect.width;
      const progress = Math.max(0, Math.min(1, position));
      audio.currentTime = progress * audio.duration;
      setProgress(progress * 100);
    }

    function togglePlayback() {
      if (audio.paused) {
        state.introAudio.pause();
        pauseCards(audio);
        warmAudio(audio);
        audio.play();
      } else {
        audio.pause();
      }
    }

    card.setAttribute("role", "button");
    card.setAttribute("tabindex", "0");
    card.setAttribute("aria-label", `Play ${label}`);

    card.addEventListener("click", togglePlayback);
    card.addEventListener("pointerenter", () => warmAudio(audio));
    card.addEventListener("pointerdown", () => warmAudio(audio), { passive: true });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        togglePlayback();
      }
    });

    progressButton.addEventListener("click", (event) => {
      event.stopPropagation();
      seekToPosition(event);
    });

    progressButton.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      progressButton.setPointerCapture(event.pointerId);
      seekToPosition(event);
    });

    progressButton.addEventListener("pointermove", (event) => {
      if (progressButton.hasPointerCapture(event.pointerId)) {
        event.preventDefault();
        event.stopPropagation();
        seekToPosition(event);
      }
    });

    progressButton.addEventListener("pointerup", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (progressButton.hasPointerCapture(event.pointerId)) {
        progressButton.releasePointerCapture(event.pointerId);
      }
    });

    progressButton.addEventListener("pointercancel", (event) => {
      if (progressButton.hasPointerCapture(event.pointerId)) {
        progressButton.releasePointerCapture(event.pointerId);
      }
    });

    audio.addEventListener("loadedmetadata", () => {
      duration.textContent = `0:00 / ${formatTime(audio.duration)}`;
    });

    audio.addEventListener("timeupdate", () => {
      if (Number.isFinite(audio.duration) && audio.duration > 0) {
        setProgress((audio.currentTime / audio.duration) * 100);
        duration.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
      }
    });

    audio.addEventListener("play", () => {
      clearCurrentCard();
      state.currentCard = card;
      card.classList.add("is-playing");
      card.setAttribute("aria-current", "true");
      card.setAttribute("aria-label", `Pause ${label}`);
      setNowPlaying(label);
    });

    audio.addEventListener("pause", () => {
      card.classList.remove("is-playing");
      card.removeAttribute("aria-current");
      card.setAttribute("aria-label", `Play ${label}`);
      if (state.currentCard === card) {
        state.currentCard = null;
      }
    });

    audio.addEventListener("ended", () => {
      setProgress(0);
      card.classList.remove("is-playing");
      card.removeAttribute("aria-current");
      card.setAttribute("aria-label", `Play ${label}`);
      setNowPlaying("Ready");
    });
  });

  updateVolumeControl();
}

function initExamples() {
  const status = document.querySelector("[data-load-status]");
  try {
    if (!window.EXAMPLE_MANIFEST) {
      throw new Error("embedded example data is missing");
    }
    state.manifest = window.EXAMPLE_MANIFEST;
    if (status) {
      status.textContent = `${state.manifest.languages.length} languages loaded`;
    }
    initVolumeControl();
    renderLanguageButtons();
    chooseLanguage("english", false);
  } catch (error) {
    if (status) {
      status.textContent = `Examples unavailable: ${error.message}`;
    }
  }
}

state.introAudio.addEventListener("ended", () => setNowPlaying("Ready"));
initExamples();
