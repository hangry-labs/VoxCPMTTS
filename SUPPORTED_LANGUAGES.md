# Supported Languages

VoxCPMTTS uses `openbmb/VoxCPM2`. The model officially supports 30 languages and does not require a language tag for synthesis: provide text in a supported language directly. The `language` field in the VoxCPMTTS UI and API is a compatibility hint used by related features such as transcription and timestamp alignment; VoxCPM2 detects the synthesis language from the input text.

Source: [official VoxCPM2 model card](https://huggingface.co/openbmb/VoxCPM2)

## Official Language List

All 30 official languages are listed individually below. The benchmark manifest
contains prompts for every language. The published speech-quality run measured
24 languages because its Qwen3-ASR judge did not advertise the other six.

| # | Language | Common code | Published benchmark |
|---:|---|---|---|
| 1 | Arabic | `ar` | Measured |
| 2 | Burmese | `my` | Not judged by the configured ASR |
| 3 | Chinese | `zh` | Measured |
| 4 | Danish | `da` | Measured |
| 5 | Dutch | `nl` | Measured |
| 6 | English | `en` | Measured |
| 7 | Finnish | `fi` | Measured |
| 8 | French | `fr` | Measured |
| 9 | German | `de` | Measured |
| 10 | Greek | `el` | Measured |
| 11 | Hebrew | `he` | Not judged by the configured ASR |
| 12 | Hindi | `hi` | Measured |
| 13 | Indonesian | `id` | Measured |
| 14 | Italian | `it` | Measured |
| 15 | Japanese | `ja` | Measured |
| 16 | Khmer | `km` | Not judged by the configured ASR |
| 17 | Korean | `ko` | Measured |
| 18 | Lao | `lo` | Not judged by the configured ASR |
| 19 | Malay | `ms` | Measured |
| 20 | Norwegian | `no`, `nb`, `nn` | Not judged by the configured ASR |
| 21 | Polish | `pl` | Measured |
| 22 | Portuguese | `pt` | Measured |
| 23 | Russian | `ru` | Measured |
| 24 | Spanish | `es` | Measured |
| 25 | Swahili | `sw` | Not judged by the configured ASR |
| 26 | Swedish | `sv` | Measured |
| 27 | Tagalog / Filipino | `tl`, `fil` | Measured as Filipino |
| 28 | Thai | `th` | Measured |
| 29 | Turkish | `tr` | Measured |
| 30 | Vietnamese | `vi` | Measured |

## Chinese Dialects

The model card additionally documents these Chinese dialect groups:

- Sichuanese (`四川话`)
- Cantonese (`粤语`)
- Wu Chinese (`吴语`)
- Northeastern Mandarin (`东北话`)
- Henan dialect (`河南话`)
- Shaanxi dialect (`陕西话`)
- Shandong dialect (`山东话`)
- Tianjin dialect (`天津话`)
- Southern Min (`闽南话`)

Dialect generation is driven by the input text and optional voice direction. VoxCPMTTS does not expose these dialects as separate model checkpoints.

## Practical Notes

- Official support means the language was included in the model's documented coverage. Quality can still vary with the text, voice design, reference recording, punctuation, and generation settings.
- A reference recording should be clean and should match its transcript exactly when transcript-guided cloning is used.
- Long-form protection is enabled by default in VoxCPMTTS. It divides long passages at natural boundaries to reduce accumulated ringing or background artifacts; users can disable it in Generation settings or with `"protect_long_audio": false` in the native API.
- The reproducible speech-quality benchmark covers the intersection of VoxCPM2 and the configured Qwen3-ASR judge. See [`benchmarks/speech-quality/DETAILS.md`](benchmarks/speech-quality/DETAILS.md) for measured results rather than treating official support as a quality ranking.
