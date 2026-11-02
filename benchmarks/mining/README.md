# Mining CPU/GPU benchmark

This benchmark checks image-level mining correctness and latency when annotation arrays use NumPy CPU or CuPy CUDA storage. Image decoding, pHash, HOG, histograms, quality metrics, and OpenCV K-Means intentionally remain CPU operations; CUDA storage is preserved for selected `Detections` outputs.

```powershell
.venv\Scripts\python benchmarks/mining/benchmark.py
```

Results are written to `artifacts/benchmark/mining_cpu_gpu.json` and `artifacts/benchmark/mining_cpu_gpu.md`. The timed dataset contains 64 images and 512 boxes. Each measurement uses one warm-up followed by the median of five runs.

## Quadro M1200 result

Measured on a 4 GB Quadro M1200 with CuPy 14.2.0:

| Operation | CPU (ms) | GPU-backed (ms) | GPU / CPU |
| --- | ---: | ---: | ---: |
| `stats_annotations` | 1.250 | 1.258 | 1.006x |
| `score_annotations` | 1.170 | 1.163 | 0.994x |
| `stats_quality` | 244.046 | 245.087 | 1.004x |
| `find_duplicates` | 168.462 | 170.181 | 1.010x |
| `cluster` | 104.487 | 104.560 | 1.001x |
| `select_weighted` | 2.272 | 57.729 | 25.405x |
| `balance` | 18.072 | 123.807 | 6.851x |

Statistics, scoring, quality, duplicate search, and clustering have effectively equal latency because their image algorithms are CPU-native. Selection and balancing preserve independent GPU annotation arrays, so many small device-to-device copies dominate this deliberately small benchmark. They are supported and correct, but GPU storage is useful for downstream GPU postprocessing rather than accelerating mining itself.
