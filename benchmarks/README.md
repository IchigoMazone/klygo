# Klygo Official Benchmark Suite

This directory contains the official multi-backend benchmark harness, telemetry datasets, and analytical evaluation reports for the **Klygo** framework across multi-core CPUs and NVIDIA CUDA GPUs.

---

## 1. Directory Structure & Organization

```text
benchmarks/
├── README.md                          # Master Benchmark Suite documentation & navigation
├── runner.py                          # Master multi-backend benchmark CLI orchestrator
├── data/                              # Shared benchmark datasets (images, video clips, annotations)
│   └── .gitkeep
├── outputs_cpu_gpu/                   # NumPy/CuPy Detections microbenchmarks
│   ├── README.md                      # Methodology, tables, and conclusions
│   └── metrics/                       # Raw NMS and storage-operation measurements
├── round_1/                           # Round 1: Pipeline Smoke Test (Historical Baseline)
│   ├── README.md                      # Round 1 summary & methodology retrospective
│   ├── metrics/
│   │   ├── benchmark_metrics.json     # Raw JSON measurements (Round 1)
│   │   └── benchmark_tables.md        # Summary tables of FPS, Latency, TTFF, RAM, VRAM
│   └── evaluations/
│       ├── summary_evaluation.md      # Multi-backend cross-tier evaluation
│       ├── cpu_evaluation.md          # CPU latency & memory observations
│       └── gpu_evaluation.md          # GPU acceleration & VRAM observations
└── round_2/                           # Round 2: Rigorous Multi-Process Benchmark Harness
    ├── README.md                      # Round 2 scientific standards & execution plan
    ├── metrics/                       # Target directory for Round 2 raw JSON & tabular metrics
    └── evaluations/                   # Target directory for Round 2 analytical reports & scorecards
```

---

## 2. Benchmark Iterations

| Iteration | Status | Primary Purpose & Methodology | Reference |
| :--- | :--- | :--- | :--- |
| **Round 1** | Completed | Initial API pipeline smoke test verifying end-to-end execution of all 6 modes across backends. Identified 9 core methodology limitations (model load in raw timer, lack of warmup, small 4-frame dataset, memory tracking asymmetry). | [`benchmarks/round_1/README.md`](round_1/README.md) |
| **Round 2** | Architecture Ready | Rigorous multi-process benchmark with subprocess isolation, symmetric warmup, memory delta tracking, 30–60 real frames, 5 repeats, detection parity verification, and full 5-model coverage. | [`benchmarks/round_2/README.md`](round_2/README.md) |
| **Outputs CPU/GPU** | Completed | Focused NumPy/CuPy benchmark for NMS, device transfers, precision conversion, and threshold operations on `Detections`. | [`benchmarks/outputs_cpu_gpu/README.md`](outputs_cpu_gpu/README.md) |

---

## 3. Execution Modes Evaluated

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
2. **Raw Stream**: Upstream library streaming generator (`ultralytics.YOLO.predict(stream=True)`). Yields results frame-by-frame.
3. **Klygo Native Non-stream**: `detector.predict(..., native=True)` bypasses Klygo's media and preprocessing pipeline while returning standardized `outputs.Detections`.
4. **Klygo Native Stream**: `detector.predict(..., native=True, model_stream=True)` delegates stream generation directly to the upstream model engine.
5. **Klygo Managed Non-stream**: Full Klygo pipeline (`media.MediaFrames` + `preprocessing.compose`), tracking affine transformation traces for automatic bounding box restoration.
6. **Klygo Managed Stream**: Generator yielding standardized `outputs.Detection` frame-by-frame with bounded memory footprint and automated coordinate restoration.

---

## 4. Running Benchmarks

```bash
# Run standard benchmark suite
python benchmarks/runner.py

# Run on specific hardware target
python benchmarks/runner.py --device cpu
python benchmarks/runner.py --device cuda:0

# Run specific model backends
python benchmarks/runner.py --models yolov8,yolo-world,grounding-dino
```
