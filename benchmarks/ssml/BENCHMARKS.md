# SSML Dialogue Reliability

This paired benchmark measures the same ten-turn English dialogue through standard SSML and SSML-H. Standard SSML uses one generated voice with explicit 120 ms turn boundaries; SSML-H uses the dynamic Host and Guest definitions shown in the product example. Each mode receives one unmeasured warmup followed by five fixed-seed rounds by default.

All WAV files are generated and staged before the separate Qwen3-ASR phase. A measured call passes only when transcript similarity is at least 95%, all ten turn-ending phrases occur in order, and the transcript reaches `within a single structure`. This directly catches progressive truncation while retaining timing, real-time factor, audio duration, transcript, and audio hash evidence for future comparisons.

ASR is a consistent semantic judge, not a subjective voice-quality score. Failed rows require listening to the retained or independently reproduced WAV before assigning the failure to synthesis rather than recognition.

Run a full result with `task benchmark-ssml`. Validate wiring without changing history with `task benchmark-ssml-smoke`. The TTS and ASR URLs can be overridden through Task variables or direct runner arguments.

| Run | Mode | Rounds | Passed | Pass rate | Tail complete | Turn endings | Mean similarity | Minimum similarity | Mean TTS seconds | Median RTF | ASR model |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 08.10.2026 14:57:14 - 1.0-snapshot | Standard SSML | 5 | 5 | 100.00% | 100.00% | 100.00% | 99.92% | 99.66% | 14.055 | 0.2965 | Qwen/Qwen3-ASR-0.6B-hf |
| 08.10.2026 14:57:14 - 1.0-snapshot | SSML-H | 5 | 5 | 100.00% | 100.00% | 100.00% | 99.83% | 99.13% | 16.717 | 0.3505 | Qwen/Qwen3-ASR-0.6B-hf |
