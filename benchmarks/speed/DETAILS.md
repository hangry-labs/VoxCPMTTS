# Inference Speed Details

Run `20261007-184048`; 10 inference steps; concurrency 1; warmed requests.
RTF is wall-clock generation time divided by generated audio duration. Lower is better.
Generation is unseeded because the VoxCPM API does not expose a shared deterministic seed.

## native

| Mode | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| design | 240 | 1.143 | 1.280 | 0.86x |
| clone | 72 | 1.115 | 1.232 | 0.89x |

### Languages

| Language | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| Arabic | 13 | 1.213 | 1.270 | 0.85x |
| Chinese | 13 | 1.137 | 1.446 | 0.85x |
| Danish | 13 | 1.150 | 2.852 | 0.72x |
| Dutch | 13 | 1.175 | 1.302 | 0.87x |
| English | 13 | 1.201 | 1.326 | 0.84x |
| Finnish | 13 | 1.160 | 1.331 | 0.84x |
| French | 13 | 1.124 | 1.232 | 0.88x |
| German | 13 | 1.130 | 1.203 | 0.89x |
| Greek | 13 | 1.100 | 1.228 | 0.89x |
| Hindi | 13 | 1.098 | 1.190 | 0.90x |
| Indonesian | 13 | 1.131 | 1.157 | 0.90x |
| Italian | 13 | 1.104 | 1.317 | 0.89x |
| Japanese | 13 | 1.083 | 1.166 | 0.91x |
| Korean | 13 | 1.134 | 1.242 | 0.89x |
| Malay | 13 | 1.099 | 1.224 | 0.91x |
| Polish | 13 | 1.082 | 1.118 | 0.91x |
| Portuguese | 13 | 1.154 | 1.237 | 0.85x |
| Russian | 13 | 1.165 | 1.234 | 0.86x |
| Spanish | 13 | 1.162 | 1.250 | 0.85x |
| Swedish | 13 | 1.131 | 1.163 | 0.89x |
| Tagalog | 13 | 1.165 | 1.339 | 0.84x |
| Thai | 13 | 1.134 | 1.258 | 0.88x |
| Turkish | 13 | 1.099 | 1.209 | 0.91x |
| Vietnamese | 13 | 1.134 | 1.258 | 0.88x |

## nano

| Mode | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| design | 240 | 0.275 | 0.316 | 3.63x |
| clone | 72 | 0.258 | 0.307 | 3.80x |

### Languages

| Language | Calls | Median RTF | p95 RTF | Speed |
|---|---:|---:|---:|---:|
| Arabic | 13 | 0.272 | 0.299 | 3.83x |
| Chinese | 13 | 0.284 | 0.323 | 3.42x |
| Danish | 13 | 0.269 | 0.294 | 3.71x |
| Dutch | 13 | 0.269 | 0.309 | 3.68x |
| English | 13 | 0.271 | 0.292 | 3.73x |
| Finnish | 13 | 0.278 | 0.315 | 3.65x |
| French | 13 | 0.270 | 0.280 | 3.82x |
| German | 13 | 0.277 | 0.301 | 3.74x |
| Greek | 13 | 0.267 | 0.291 | 3.68x |
| Hindi | 13 | 0.269 | 0.285 | 3.74x |
| Indonesian | 13 | 0.262 | 0.313 | 3.73x |
| Italian | 13 | 0.266 | 0.282 | 3.91x |
| Japanese | 13 | 0.253 | 0.294 | 3.86x |
| Korean | 13 | 0.278 | 0.322 | 3.41x |
| Malay | 13 | 0.297 | 0.311 | 3.46x |
| Polish | 13 | 0.277 | 0.312 | 3.66x |
| Portuguese | 13 | 0.290 | 0.328 | 3.53x |
| Russian | 13 | 0.276 | 0.316 | 3.66x |
| Spanish | 13 | 0.283 | 0.304 | 3.63x |
| Swedish | 13 | 0.261 | 0.289 | 3.77x |
| Tagalog | 13 | 0.265 | 0.304 | 3.85x |
| Thai | 13 | 0.264 | 0.291 | 3.92x |
| Turkish | 13 | 0.254 | 0.282 | 3.95x |
| Vietnamese | 13 | 0.276 | 0.326 | 3.57x |
