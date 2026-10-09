<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/examples/">
    <img src="assets/voxcpmtts_logo_horizontal.webp" alt="Hangry Labs VoxCPMTTS-logo" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <strong>Norsk bokmål</strong> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs VoxCPMTTS

En bruksklar Docker-pakke for flerspråklig tekst til tale med Nano-vLLM-inferens, lokalt nettlesergrensesnitt og HTTP-API.

Denne Hangry Labs-versjonen er laget for enkel lokal bruk. Start én container, åpne grensesnittet eller bruk API-et uten å konfigurere Python, modeller og lydverktøy manuelt.

## Dette får du

- Nettlesergrensesnitt for generering, strømming, stemmedesign, stemmekloning, lokal transkripsjon og valgfri ikke-destruktiv stemmebehandling
- Vedvarende stemmeprofiler med designoppskrifter, portretter, tagger og gjenbruk på tvers av arbeidsflyter
- SSML og SSML-H for fortellinger, dialoger med flere talere og skuespill
- Native HTTP-API for applikasjoner og lokale integrasjoner
- Generering på 30 språk støttet av VoxCPM2
- WAV-, MP3-, FLAC- og OGG-utgang
- Nano-vLLM med CUDA-grafakselerasjon
- Et komplett Docker-bilde med modellressursene som kreves for frakoblet inferens

> [!IMPORTANT]
> Prosjektets egen og oppstrømsavledede kildekode bruker Apache-2.0. Docker-bildene er samlede distribusjoner som også inneholder modellressurser, NVIDIA CUDA-biblioteker, FFmpeg og andre pakker med egne vilkår. Les [merknadene om tredjepartskomponenter](THIRD_PARTY_NOTICES.md) før distribusjon eller videredistribusjon.

## Hurtigstart

Start det komplette bildet på en NVIDIA-GPU:

```bash
docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:latest
```

Åpne deretter:

- Grensesnitt: [http://localhost:8808/nb](http://localhost:8808/nb)
- API-dokumentasjon: [http://localhost:8808/tts/docs](http://localhost:8808/tts/docs)
- [Eksempler på språk og stemmer](https://hangry-labs.github.io/VoxCPMTTS/examples/)

Volumet `voxcpmtts_data` beholder modellbuffer, stemmeprofiler, referanselyd og fremtidige innstillinger når containeren erstattes. Det komplette bildet inneholder de låste modellressursene og kan kjøre normal inferens uten nettverk etter nedlasting.

Publiserte GitHub-utgivelser og tilhørende Git-tagger er uforanderlige. Versjonerte Docker-tagger er også uforanderlige, mens `latest` og `latest_tiny` med hensikt følger det nyeste øyeblikksbildet.

## Mer informasjon

Les [den norske produktsiden og installasjonsveiledningen](https://hangrylabs.app/nb/software/voxcpmtts). Full teknisk dokumentasjon vedlikeholdes i den [engelske README-filen](README.md). Feil og forslag kan rapporteres i [GitHub Issues](https://github.com/Hangry-Labs/VoxCPMTTS/issues).
