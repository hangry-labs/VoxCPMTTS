# Inference Speed Details

Run `20261022-054748`; 10 inference steps; concurrency 1; warmed requests.
RTF is wall-clock generation time divided by generated audio duration. Lower is better.
Each call uses a stable seed derived from its call ID so compatible runtimes can be compared exactly.

## nano

| Mode | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| design | 240 | 0.277 | 0.330 | 3.63x |
| clone | 72 | 0.251 | 0.297 | 3.92x |

### Languages

| Language | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| Arabic | 13 | 0.289 | 0.327 | 3.66x |
| Chinese | 13 | 0.289 | 0.347 | 3.30x |
| Danish | 13 | 0.278 | 0.331 | 3.64x |
| Dutch | 13 | 0.283 | 0.325 | 3.71x |
| English | 13 | 0.272 | 0.331 | 3.66x |
| Finnish | 13 | 0.282 | 0.322 | 3.67x |
| French | 13 | 0.295 | 0.301 | 3.65x |
| German | 13 | 0.264 | 0.317 | 3.82x |
| Greek | 13 | 0.263 | 0.295 | 3.74x |
| Hindi | 13 | 0.254 | 0.310 | 3.99x |
| Indonesian | 13 | 0.259 | 0.305 | 3.81x |
| Italian | 13 | 0.274 | 0.309 | 3.77x |
| Japanese | 13 | 0.226 | 0.286 | 4.18x |
| Korean | 13 | 0.294 | 0.309 | 3.53x |
| Malay | 13 | 0.277 | 0.366 | 3.58x |
| Polish | 13 | 0.271 | 0.320 | 3.69x |
| Portuguese | 13 | 0.268 | 0.318 | 3.77x |
| Russian | 13 | 0.237 | 0.293 | 4.07x |
| Spanish | 13 | 0.268 | 0.314 | 3.81x |
| Swedish | 13 | 0.244 | 0.271 | 4.13x |
| Tagalog | 13 | 0.266 | 0.318 | 3.85x |
| Thai | 13 | 0.254 | 0.283 | 4.02x |
| Turkish | 13 | 0.271 | 0.302 | 3.87x |
| Vietnamese | 13 | 0.310 | 0.361 | 3.23x |
