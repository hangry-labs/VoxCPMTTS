<p align="center">
  <a href="https://hangry-labs.github.io/VoxCPMTTS/examples/">
    <img src="assets/voxcpmtts_logo_horizontal.webp" alt="Logotipo de Hangry Labs VoxCPMTTS" width="900">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.nb.md">Norsk bokmål</a> ·
  <a href="README.pl.md">Polski</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.zh.md">简体中文</a> ·
  <strong>Español</strong>
</p>

# Hangry Labs VoxCPMTTS

Paquete Docker de texto a voz multilingüe listo para usar, con inferencia Nano-vLLM, interfaz local en el navegador y API HTTP.

Esta versión de Hangry Labs está pensada para un uso local sencillo. Inicia un contenedor y genera voz desde la interfaz o la API sin configurar manualmente Python, modelos ni herramientas de audio.

## Qué ofrece el proyecto

- Interfaz web para generación, transmisión, diseño y clonación de voces, además de transcripción local
- Perfiles de voz persistentes con recetas de diseño, retratos, etiquetas y reutilización entre flujos de trabajo
- SSML y SSML-H para narraciones, diálogos con varios hablantes y obras de audio
- API HTTP nativa para aplicaciones e integraciones locales
- Generación en los 30 idiomas compatibles con VoxCPM2
- Salida WAV, MP3, FLAC y OGG
- Inferencia Nano-vLLM con aceleración mediante grafos CUDA
- Imagen Docker completa con los recursos de modelo necesarios para inferencia sin conexión

> [!IMPORTANT]
> El código propio del proyecto y el derivado del proyecto original usan Apache-2.0. Sin embargo, las imágenes Docker son distribuciones agregadas que también incluyen recursos de modelos, bibliotecas NVIDIA CUDA, FFmpeg y otros paquetes bajo sus respectivas condiciones. Lee los [avisos de terceros](THIRD_PARTY_NOTICES.md) antes de implementar o redistribuir.

## Inicio rápido

Ejecuta la imagen completa en una GPU NVIDIA:

```bash
docker run --name voxcpmtts --restart unless-stopped -p 8808:8808 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v voxcpmtts_data:/app/persistent hangrylabs/voxcpmtts:latest
```

Después abre:

- Interfaz: [http://localhost:8808/es](http://localhost:8808/es)
- Documentación de la API: [http://localhost:8808/tts/docs](http://localhost:8808/tts/docs)
- [Ejemplos de idiomas y voces](https://hangry-labs.github.io/VoxCPMTTS/examples/)

El volumen `voxcpmtts_data` conserva la caché de modelos, los perfiles de voz, el audio de referencia y futuros ajustes al reemplazar el contenedor. La imagen completa incluye los recursos fijados del modelo y, una vez descargada, puede realizar la inferencia normal sin red.

Las versiones publicadas en GitHub y sus etiquetas Git asociadas son inmutables. Las etiquetas versionadas de Docker también son inmutables, mientras que `latest` y `latest_tiny` apuntan deliberadamente a la instantánea más reciente.

## Más información

Consulta la [página del producto y la guía de instalación en español](https://hangrylabs.app/es/software/voxcpmtts). La documentación técnica completa se mantiene en el [README en inglés](README.md). Puedes informar de errores y sugerencias en [GitHub Issues](https://github.com/Hangry-Labs/VoxCPMTTS/issues).
