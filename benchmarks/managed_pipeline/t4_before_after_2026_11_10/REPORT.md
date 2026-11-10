# Klygo managed-input pipeline: T4 before/after benchmark

Date: 2026-11-10 (Asia/Bangkok)  
Branch: `perf/managed-input-pipeline`  
Before: `5ed34c82fa6db1f700d0e0a591e69e9259538640`  
After: `237efc7537f65f6ff1016aac60acd9b67c6f2ad4`

## Environment and protocol

- Google Colab, NVIDIA Tesla T4 (15,360 MiB), GPU UUID `GPU-33698f6d-0be2-670f-5e3f-355cc78707fa`
- NVIDIA driver `580.82.07`, power limit 70 W; PyTorch `2.14.1+cu130`
- 16 configurations: 2 revisions × 2 model families × 2 sources × 2 batch sizes
- 88 supported cases, 7 repeats per supported case, 616 measured passes
- Cross-revision output parity: **PASS** for every comparable case
- Values below are medians in milliseconds; lower is better.

## Managed-path results

| Model | Source | Batch | Before non-stream | After non-stream | Delta | Before stream | After stream | Delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yolo | images | 1 | 71.498 | 70.220 | -1.79% | 66.947 | 66.927 | -0.03% |
| yolo | images | 4 | 118.536 | 104.202 | -12.09% | 103.359 | 104.606 | +1.21% |
| yolo | video | 1 | 77.713 | 81.260 | +4.56% | 79.248 | 81.051 | +2.28% |
| yolo | video | 4 | 138.032 | 141.104 | +2.23% | 132.342 | 136.120 | +2.85% |
| yolo26 | images | 1 | 55.704 | 54.258 | -2.60% | 53.161 | 52.232 | -1.75% |
| yolo26 | images | 4 | 106.900 | 118.518 | +10.87% | 97.409 | 101.318 | +4.01% |
| yolo26 | video | 1 | 63.127 | 46.304 | -26.65% | 50.851 | 64.058 | +25.97% |
| yolo26 | video | 4 | 120.846 | 126.496 | +4.68% | 121.872 | 121.688 | -0.15% |

Across the 16 managed comparisons there are 9 improvements and 7 regressions. The simple mean delta is +0.85% and the median delta is +1.72%; positive means slower. This is therefore **not a universal performance win**. It is workload-dependent.

The largest regression is yolo26 video batch-1 managed stream (+25.97%); the next is yolo26 images batch-4 managed non-stream (+10.87%). The same yolo26 video batch-1 configuration improves 26.65% in non-stream mode, which is a strong sign that the effect is path-specific. Re-run these outliers with locked clocks/power where possible and preserve per-repeat samples externally before changing code again.

## Evidence-retention note

The Colab run completed successfully and created `t4_before_after_evidence.zip` (3,088,125 bytes, SHA-256 `9c9aa82fe05cec863f365c4e6486ba8df480cc81da7af8e0642fffec86994e30`). The runtime recycled before the archive could be downloaded, so the raw per-repeat arrays are **not retained**. This directory intentionally contains only the aggregate medians, run metadata, driver source, and the checksum of the lost archive. Do not treat it as raw-run evidence.

Colab copy used for the run: <https://colab.research.google.com/drive/1nz4gMe2xglLKrfxynlK-YjUE1hMIbc59>

