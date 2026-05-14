# VoxCPMTTS Examples

This directory contains the static project page and browser-side examples for Hangry Labs VoxCPMTTS.

## Static Project Page

- `index.html` is the GitHub Pages landing page for Hangry Labs VoxCPMTTS.
- `voices.js` embeds the generated sample manifest used by the page.
- `player.js` renders the language picker and audio sample player.
- `background.js` supports the page background.

## Generated Audio

- `original_clone.mp3` is the local reference voice used for clone examples.
- `assets/manifest.json` indexes the generated samples.
- `assets/<language>/random/` contains 10 native-language voice-design samples.
- `assets/<language>/intro/` contains 3 translated project intro samples.
- `assets/<language>/clone/` contains 1 translated clone sample using `original_clone.mp3`.

Regenerate and validate samples with the project maintenance scripts.
After regeneration, rebuild `voices.js` from `assets/manifest.json` so the static page uses the latest sample metadata.

Preview locally from the repository root:

```powershell
python -m http.server 9000
```

Then open `http://localhost:9000/examples/`.
