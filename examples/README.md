---
layout: default
title: Examples guide
permalink: /examples/guide/
---
# VoxCPMTTS Examples

This directory contains the static project page and browser-side examples for Hangry Labs VoxCPMTTS.

## Static Project Page

- `index.md` integrates the gallery into the shared GitHub Pages website.
- `index.html` remains the standalone gallery, with local styles in `gallery.css`.
- `voices.js` embeds the generated sample manifest used by the page.
- `player.js` renders the language picker and audio sample player.
- `background.js` supports the page background.

## Generated Audio

- `original_clone.mp3` is the project seed voice, generated from a randomized KokoroTTS voice and used consistently for clone examples.
- `assets/manifest.json` indexes the generated samples.
- `assets/<language>/random/` contains 10 native-language voice-design samples.
- `assets/<language>/intro/` contains 3 translated project intro samples.
- `assets/<language>/clone/` contains 1 translated clone sample using `original_clone.mp3`.
- `assets/featured/` contains curated product demonstrations that pair generated audio with the UI used to create it.

## SSML-H Documents

- `ssml-h/polish-launch-dialogue.ssml` is an extended fixed-seed Polish launch sequence covering the countdown, emergency response, recovery, and launch.
- `ssml-h/english-launch-dialogue.ssml` and `ssml-h/german-launch-dialogue.ssml` provide matching three-speaker launch scenarios for multilingual testing.

Regenerate and validate samples with the project maintenance scripts.
After regeneration, rebuild `voices.js` from `assets/manifest.json` so the static page uses the latest sample metadata.

Preview the complete website with `task pages` and open
`http://127.0.0.1:9005/VoxCPMTTS/examples/`. The overview is built from the
root README, with a shared header and six interface languages.

For the standalone gallery, preview locally from the repository root:

```powershell
python -m http.server 9000
```

Then open `http://localhost:9000/examples/`.
