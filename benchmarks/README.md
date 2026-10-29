# Klygo Official Benchmark Suite

This directory contains the official multi-backend benchmark harness, telemetry datasets, and analytical evaluation reports for the **Klygo** framework across multi-core CPUs and NVIDIA CUDA GPUs.

---

## 1. Directory Structure

```text
benchmarks/
├── README.md                          # Official Benchmark Suite documentation & guide
├── runner.py                          # Multi-backend intra-model benchmark CLI runner
├── metrics/                           # Quantitative telemetry and tabular benchmarks
│   ├── benchmark_metrics.json         # Complete raw JSON measurements
│   └── benchmark_tables.md            # Formatted tables of FPS, Latency, TTFF, RAM, VRAM
└── evaluations/                       # Dedicated analytical reports and architectural insights
    ├── summary_evaluation.md          # Multi-backend cross-tier evaluation & recommendations
    ├── cpu_evaluation.md              # In-depth CPU latency, memory footprint, and bottlenecks
    └── gpu_evaluation.md              # In-depth GPU acceleration, VRAM containment, and throughput
```

---

## 2. Execution Modes Evaluated

Every integrated detection model is evaluated across up to 6 execution modes:

```text
                         ┌── Raw Non-stream (Upstream predict(stream=False))
        ┌─ 1. Raw ───────┤
        │                └── Raw Stream (Upstream predict(stream=True))
        │
        │                ┌── Klygo Native Non-stream (native=True, stream=False)
Modes ──┼─ 2. Native ────┤
        │                └── Klygo Native Stream (native=True, model_stream=True)
        │
        │                ┌── Klygo Managed Non-stream (native=False, stream=False)
        └─ 3. Managed ───┤
                         └── Klygo Managed Stream (native=False, stream=True)
```

1. **Raw Non-stream**: Direct upstream library call (`ultralytics.YOLO.predict(stream=False)`, etc.). Accumulates all results in RAM before returning.
2. **Raw Stream**: Upstream library streaming generator (`ultralytics.YOLO.predict(stream=True)`). Yields results frame-by-frame. (Unavailable in Hugging Face Grounding DINO).
3. **Klygo Native Non-stream**: `detector.predict(..., native=True)` bypasses Klygo's media and preprocessing pipeline while returning standardized `outputs.Detections`.
4. **Klygo Native Stream**: `detector.predict(..., native=True, model_stream=True)` delegates stream generation directly to the upstream model engine.
5. **Klygo Managed Non-stream**: Full Klygo pipeline (`media.MediaFrames` + `preprocessing.compose`), tracking affine transformation traces for automatic bounding box restoration.
6. **Klygo Managed Stream**: Generator yielding standardized `outputs.Detection` frame-by-frame with bounded memory footprint and automated coordinate restoration.

---

## 3. Quickstart & Running Benchmarks

### Basic Execution (CPU & CUDA auto-detection)
```bash
python benchmarks/runner.py
```

### Options
```bash
# Test specific device
python benchmarks/runner.py --device cpu
python benchmarks/runner.py --device cuda:0

# Test specific models
python benchmarks/runner.py --models yolov8,yolo-world
```

The runner automatically generates a 4-frame synthetic clip if no source video is provided, ensuring deterministic, reproducible metrics without requiring gigabytes of media downloads.

---

## 4. Key Findings Summary

- **First Frame Responsiveness (TTFF)**: Streaming modes yield frame #1 in **285 ms – 399 ms** on CPU/GPU for YOLO models, compared to **1.8 s – 17.2 s** when using blocking non-streaming modes.
- **Filling Upstream Gaps**: Hugging Face Grounding DINO does not support lazy streaming natively. `Klygo Managed Stream` closes this gap, reducing initial response latency from 5.2 s to **2.0 s** on GPU.
- **Memory Predictability**: Accumulating results without streaming consumes hundreds of megabytes on the Python heap (up to 360 MB for YOLO-World). In contrast, Klygo streaming modes cap Heap memory at a flat **~2.2 MB – 2.5 MB**.
- **GPU VRAM Containment**: In Klygo Managed mode, YOLOv8 requires only **27.8 MB VRAM** and YOLO-World uses **114.8 MB VRAM**, avoiding CUDA cache bloat.

For detailed numbers and architectural insights:
- [Summary Evaluation Report](evaluations/summary_evaluation.md)
- [CPU Evaluation & Bottleneck Analysis](evaluations/cpu_evaluation.md)
- [GPU Acceleration & VRAM Analysis](evaluations/gpu_evaluation.md)
- [Full Performance Tables](metrics/benchmark_tables.md)
