# Inference Speed Details

Run `20261029-070100`; 10 inference steps; concurrency 1; warmed requests.
RTF is wall-clock generation time divided by generated audio duration. Lower is better.
Each call uses a stable seed derived from its call ID so compatible runtimes can be compared exactly.

## nano

| Mode | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| design | 240 | 0.271 | 0.320 | 3.71x |
| clone | 72 | 0.242 | 0.306 | 4.05x |

### Languages

| Language | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| Arabic | 13 | 0.267 | 0.305 | 3.98x |
| Chinese | 13 | 0.273 | 0.304 | 3.62x |
| Danish | 13 | 0.272 | 0.336 | 3.81x |
| Dutch | 13 | 0.267 | 0.304 | 3.74x |
| English | 13 | 0.269 | 0.314 | 3.81x |
| Finnish | 13 | 0.307 | 0.400 | 3.20x |
| French | 13 | 0.264 | 0.319 | 3.79x |
| German | 13 | 0.256 | 0.307 | 3.95x |
| Greek | 13 | 0.286 | 0.338 | 3.37x |
| Hindi | 13 | 0.293 | 0.341 | 3.70x |
| Indonesian | 13 | 0.248 | 0.264 | 4.04x |
| Italian | 13 | 0.280 | 0.302 | 3.80x |
| Japanese | 13 | 0.224 | 0.283 | 4.26x |
| Korean | 13 | 0.265 | 0.294 | 3.78x |
| Malay | 13 | 0.259 | 0.319 | 3.84x |
| Polish | 13 | 0.254 | 0.297 | 3.94x |
| Portuguese | 13 | 0.254 | 0.287 | 3.95x |
| Russian | 13 | 0.242 | 0.295 | 4.06x |
| Spanish | 13 | 0.266 | 0.307 | 3.90x |
| Swedish | 13 | 0.232 | 0.268 | 4.24x |
| Tagalog | 13 | 0.264 | 0.305 | 3.97x |
| Thai | 13 | 0.254 | 0.281 | 4.10x |
| Turkish | 13 | 0.257 | 0.281 | 4.09x |
| Vietnamese | 13 | 0.274 | 0.294 | 3.83x |
