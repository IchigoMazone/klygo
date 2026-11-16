---
name: klygo-video-benchmark
description: Run end-to-end H.264 video streaming benchmarks for Klygo (`benchmarks/gpu_suite/run.py --profile video`) across upstream-stream models (`yolo`, `yolo11`, `yolo26`, `yolo-world`) and 6 execution modes, measuring decoded FPS, TTFF (`first_frame_ms`), P50/P95/P99 delivery latency, parity, and VRAM/RSS. Saves reports under `benchmarks/results/video/<gpu>_<YYYY_MM_DD>/`.
---

# Klygo Video Streaming Benchmark Skill

Use this skill when measuring **end-to-end video decoding + streaming inference** (separate from predecoded `images` packs in `klygo-release-benchmark`).

## 1. Scope & Metrics
- **Models**: `yolo`, `yolo11`, `yolo26`, `yolo-world` (the 4 models with native upstream video stream engines).
- **Modes**: `raw-non-stream`, `raw-stream`, `native-non-stream`, `native-stream`, `managed-non-stream`, `managed-stream` (at `batch=1`; larger batches mark `native-*` as `N/A`).
- **Metrics**: End-to-end FPS (including H.264 decode), Time-To-First-Frame (`first_frame_ms`), delivery latency (`P50`, `P95`, `P99` ms), numeric parity vs `raw-non-stream`, and VRAM/RSS/NVML telemetry.

## 2. Standard Command
```bash
for MODEL in yolo yolo11 yolo26 yolo-world; do
  PYTHONPATH=. python benchmarks/gpu_suite/run.py \
    --model "$MODEL" \
    --profile video \
    --video-frames 120 \
    --batch 1 \
    --repeats 7 \
    --device cuda:0 \
    --output "artifacts/benchmarks/video/${MODEL}"
done
```
*(Pass `--source-video /path/to/video.mp4` to benchmark a real external video instead of the generated 1280x720 H.264 bus clip.)*

## 3. Saving Results
Generate `REPORT.md` + charts and store lightweight JSONs under:
```text
benchmarks/results/video/<gpu>_<YYYY_MM_DD>/
```
Upload full raw `.zip` bundles to GitHub Releases and commit only `REPORT.md`, `charts/*.png`, `.ipynb`, and summary JSONs.
