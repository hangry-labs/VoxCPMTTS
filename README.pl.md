<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/examples/">
    <img src="assets/voxcpmtts_logo_horizontal.webp" alt="Logo Hangry Labs VoxCPMTTS" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <strong>Polski</strong> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <a href="README.es.md">Español</a>
</p>

# Hangry Labs VoxCPMTTS

Gotowy do użycia w Dockerze, wielojęzyczny system zamiany tekstu na mowę z inferencją Nano-vLLM, lokalnym interfejsem przeglądarkowym i API HTTP.

Ta wersja Hangry Labs została przygotowana z myślą o prostej lokalnej pracy. Uruchom jeden kontener, otwórz interfejs albo użyj API bez ręcznej konfiguracji Pythona, modeli i narzędzi audio.

## Co oferuje projekt

- Interfejs przeglądarkowy do generowania, strumieniowania, projektowania i klonowania głosu oraz lokalnej transkrypcji
- Trwałe profile głosowe z recepturami projektu, portretami, tagami i ponownym użyciem w różnych trybach
- SSML i SSML-H do narracji, dialogów wielu postaci i słuchowisk
- Natywne API HTTP dla aplikacji i lokalnych integracji
- Generowanie w 30 językach obsługiwanych przez VoxCPM2
- Format WAV, MP3, FLAC i OGG
- Nano-vLLM z akceleracją grafów CUDA
- Pełny obraz Docker zawierający zasoby modelu potrzebne do pracy offline

> [!IMPORTANT]
> Kod własny projektu i kod pochodzący z projektu nadrzędnego korzystają z licencji Apache-2.0. Obrazy Docker są jednak dystrybucjami zbiorczymi zawierającymi również zasoby modeli, biblioteki NVIDIA CUDA, FFmpeg i inne pakiety na ich własnych warunkach. Przed wdrożeniem lub redystrybucją przeczytaj [informacje o komponentach zewnętrznych](THIRD_PARTY_NOTICES.md).

## Szybki start

Uruchom pełny obraz na karcie NVIDIA:

```bash
docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:latest
```

Następnie otwórz:

- Interfejs: [http://localhost:8808/pl](http://localhost:8808/pl)
- Dokumentację API: [http://localhost:8808/tts/docs](http://localhost:8808/tts/docs)
- [Przykłady języków i głosów](https://hangry-labs.github.io/VoxCPMTTS/examples/)

Wolumin `voxcpmtts_data` zachowuje pamięć podręczną modeli, profile głosowe, dźwięk referencyjny i przyszłe ustawienia po wymianie kontenera. Pełny obraz zawiera przypięte zasoby modeli i po pobraniu może wykonywać zwykłą inferencję bez sieci.

Opublikowane wydania GitHub i powiązane tagi Git są niezmienne. Wersjonowane tagi obrazów Docker również są niezmienne, natomiast `latest` i `latest_tiny` celowo wskazują najnowszą wersję rozwojową.

## Więcej informacji

Przeczytaj [pełny polski opis produktu i instrukcję instalacji](https://hangrylabs.app/pl/software/voxcpmtts). Pełna dokumentacja techniczna jest utrzymywana w [angielskim pliku README](README.md). Błędy i propozycje można zgłaszać w [GitHub Issues](https://github.com/Hangry-Labs/VoxCPMTTS/issues).
