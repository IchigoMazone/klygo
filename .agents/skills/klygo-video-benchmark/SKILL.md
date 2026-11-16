---
name: klygo-video-benchmark
description: Run end-to-end H.264 video streaming benchmarks for Klygo (`benchmarks/gpu_suite/run.py --profile video`) across upstream-stream models (`yolo`, `yolo11`, `yolo26`, `yolo-world`) and 6 execution modes, measuring decoded FPS, TTFF (`first_frame_ms`), P50/P95/P99 delivery latency, parity, and VRAM/RSS. Saves reports under `benchmarks/results/video/<gpu>_<YYYY_MM_DD>/`.
---

# Klygo Video Streaming Benchmark Skill

Use this skill when measuring **end-to-end video decoding + streaming inference** (separate from predecoded `images` packs in `klygo-release-benchmark`).

## 1. Scope & 3-Tier Video Matrix
- **Models**: `yolo`, `yolo11`, `yolo26`, `yolo-world` across `float32` and `float16`.
- **3 Real Video Tiers (Intel IoT DevKit CC-BY-4.0)**:
  1. `people_120f_432p` (`people-detection.mp4`, `768×432`, `120 frames`, `repeats=7`)
  2. `cars_240f_720p` (`car-detection.mp4`, `1280×720` HD, `240 frames`, `repeats=5`)
  3. `aisle_480f_1080p` (`store-aisle-detection.mp4`, `1920×1080` Full HD, `480 frames`, `repeats=3`)
- **Modes**: `raw-non-stream`, `raw-stream`, `native-non-stream`, `native-stream`, `managed-non-stream`, `managed-stream`.
- **Metrics**: End-to-end FPS (including H.264 decode), Time-To-First-Frame (`ttff_ms`), delivery latency (`p50_ms`, `p95_ms`, `p99_ms`), and numeric parity (`parity_pass`).

## 2. Standard Command
```bash
PYTHONPATH=. python benchmarks/video/run_video_benchmark.py \
  --tier all \
  --output artifacts/benchmarks/video \
  --zip artifacts/benchmarks/colab_l4_video_results.zip
```

## 3. Package `.zip`, Upload GitHub Release & Push Reports
1. **Compress raw output** (never commit `.zip` or `artifacts/`):
   ```python
   from klygo import archive
   archive.compress("artifacts/benchmarks/video", "artifacts/benchmarks/video_<gpu>_results.zip", include_root=True, overwrite=True)
   ```
2. **Upload `.zip` to GitHub Release** (`gh release create "benchmark-video-<gpu>-$(date +%Y%m%d)-$(git rev-parse --short HEAD)" artifacts/benchmarks/video_<gpu>_results.zip --prerelease`, or browser fallback via `releases/new`).
3. **Commit & push** `REPORT.md`, `charts/*.png`, executed `.ipynb`, and lightweight JSON summaries under `benchmarks/results/video/<gpu>_<YYYY_MM_DD>/`:
   ```bash
   git add benchmarks/results/video/
   git add -f benchmarks/results/video/**/charts/*.png
   git commit -m "Add video streaming benchmark report (<gpu>, <YYYY-MM-DD>)"
   git push origin HEAD
   ```
