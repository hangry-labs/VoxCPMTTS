# Inference Speed Details

Run `20261022-085849`; 10 inference steps; concurrency 1; warmed requests.
RTF is wall-clock generation time divided by generated audio duration. Lower is better.
Each call uses a stable seed derived from its call ID so compatible runtimes can be compared exactly.

## nano

| Mode | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| design | 240 | 0.278 | 0.336 | 3.59x |
| clone | 72 | 0.247 | 0.296 | 3.96x |

### Languages

| Language | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| Arabic | 13 | 0.278 | 0.325 | 3.82x |
| Chinese | 13 | 0.297 | 0.327 | 3.37x |
| Danish | 13 | 0.278 | 0.314 | 3.83x |
| Dutch | 13 | 0.268 | 0.310 | 3.85x |
| English | 13 | 0.263 | 0.301 | 3.80x |
| Finnish | 13 | 0.278 | 0.318 | 3.63x |
| French | 13 | 0.284 | 0.319 | 3.61x |
| German | 13 | 0.271 | 0.322 | 3.70x |
| Greek | 13 | 0.276 | 0.316 | 3.58x |
| Hindi | 13 | 0.264 | 0.289 | 3.95x |
| Indonesian | 13 | 0.253 | 0.271 | 3.97x |
| Italian | 13 | 0.266 | 0.309 | 3.81x |
| Japanese | 13 | 0.234 | 0.288 | 4.17x |
| Korean | 13 | 0.275 | 0.309 | 3.64x |
| Malay | 13 | 0.281 | 0.332 | 3.71x |
| Polish | 13 | 0.279 | 0.358 | 3.64x |
| Portuguese | 13 | 0.270 | 0.292 | 3.84x |
| Russian | 13 | 0.250 | 0.310 | 3.93x |
| Spanish | 13 | 0.280 | 0.334 | 3.51x |
| Swedish | 13 | 0.290 | 0.351 | 3.44x |
| Tagalog | 13 | 0.309 | 0.365 | 3.54x |
| Thai | 13 | 0.262 | 0.301 | 3.89x |
| Turkish | 13 | 0.256 | 0.304 | 3.89x |
| Vietnamese | 13 | 0.277 | 0.314 | 3.71x |
