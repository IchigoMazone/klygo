---
name: klygo-multi-gpu-benchmark
description: Run standardized multi-backend benchmarks across multiple GPUs and accelerators (CUDA, Apple Silicon MPS, CPU) for all integrated Klygo detector architectures (YOLOv8, YOLO-World v2, Grounding DINO, Faster R-CNN, RT-DETR / D-FINE) across all 6 execution modes (Raw Stream, Raw Non-stream, Native Stream, Native Non-stream, Managed Stream, Managed Non-stream).
---

# Klygo Multi-GPU & Multi-Backend Benchmark Skill

This skill provides an authoritative, reproducible benchmarking workflow to evaluate object detection models across multiple GPUs and hardware accelerators using **Klygo**.

Read [references/multi-gpu-guide.md](references/multi-gpu-guide.md) for detailed hardware architecture profiles (RTX 4090, A100/H100, Colab T4/V100, Quadro, Apple Silicon MPS, and CPU).

---

## 1. Capabilities & Core Architecture

The benchmark evaluates:
- **5 Detector Model Families**:
  1. `yolo`: Ultralytics YOLOv8 / YOLOv11 (CNN 1-Stage)
  2. `yolo-world`: Ultralytics YOLO-World v2 (Open-Vocabulary Text Prompts)
  3. `grounding-dino`: Hugging Face Grounding DINO tiny (Deformable Cross-Attention Transformer)
  4. `faster-rcnn`: Torchvision Faster R-CNN MobileNetV3 FPN (Two-Stage RPN + RoIAlign)
  5. `d-fine`: KerasHub / Keras 3 RT-DETR / D-FINE Nano (End-to-End DETR Transformer)

- **6 Execution Modes**:
  1. `raw-non-stream`: Pure upstream framework batched/non-streaming pipeline (e.g. `model.predict(source, stream=False)`)
  2. `raw-stream`: Pure upstream framework streaming pipeline (e.g. `model.predict(source, stream=True)`; only supported if upstream framework has a built-in video stream engine)
  3. `native-non-stream`: Klygo native backend (`native=True`) receiving direct inputs without `media`/`preprocessing` transforms, locking Klygo decoder parameters (supported across all detector subclasses)
  4. `native-stream`: Klygo native backend with upstream stream engine enabled (`native=True, model_stream=True`; only supported on models whose backend has an upstream stream engine like Ultralytics)
  5. `managed-non-stream`: Klygo managed pipeline batched (`native=False, stream=False`) with `media` decoding & coordinate trace
  6. `managed-stream`: Klygo managed pipeline generator (`native=False, stream=True`) with `media` lazy decoding & coordinate trace

- **Safe Resource Bounding & Anti-Freeze Guidelines**:
  - **Downsized Sample Clips**: When benchmarking on CPU or memory-constrained accelerators, use short clips (8–10 frames, 320×240 or 320×320 resolution) to prevent system freezing or OOM errors.
  - **Explicit Memory Reclamation**: Run `gc.collect()` and `torch.cuda.empty_cache()` between each model and mode transition.
  - **Structured Run Folders**: Organize every benchmark execution inside an isolated timestamped folder under `artifacts/benchmarks/run_<YYYYMMDD_HHMMSS>/` containing both raw JSON metrics (`benchmark_metrics.json`) and a formatted Markdown scorecard (`benchmark_report.md`).

- **Hardware Targeting**:
  - Multi-GPU enumeration: Automatically tests and compares each GPU card (`--device all-gpus`).
  - Specific GPU selection: `--device cuda:0`, `--device cuda:1`, etc.
  - Apple Silicon Metal: `--device mps`.
  - CPU Fallback: `--device cpu`.

- **Comprehensive Multi-Criteria Metrics**:
  - Throughput: Median, Mean, and Standard Deviation (Jitter) of FPS across repeat runs.
  - Latency: Time to First Detection (`first_frame_ms`) and sustained per-frame latency (`ms_per_frame`).
  - Memory Profile: Peak Python heap memory (`tracemalloc`) and OS process memory delta (`psutil` RSS).
  - Detection Parity: Total detected bounding boxes across runs ensuring zero loss of output integrity.

---

## 2. Benchmark Script & Entry Points

The benchmark runner is provided as a single unified script at:
- `examples/integration/multi_backend_benchmark.py` (Repository integration example)
- `.agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py` (Skill local executable)

Both entry points are strictly dogfooded with Klygo public APIs (`klygo.files`, `klygo.media`, `klygo.models`, `klygo.outputs`, `klygo.utils`).

---

## 3. Standard CLI Recipes

### Recipe 1: Benchmark All Models on the Primary GPU
```bash
python examples/integration/multi_backend_benchmark.py \
    --device auto \
    --models all \
    --modes all \
    --repeats 3 \
    --max-frames 10 \
    --output artifacts/benchmark/report.json \
    --markdown artifacts/benchmark/report.md
```

### Recipe 2: Benchmark Across All Available GPUs in a Multi-GPU Node
```bash
python examples/integration/multi_backend_benchmark.py \
    --device all-gpus \
    --models yolo,yolo-world,faster-rcnn \
    --modes native-stream,managed-stream \
    --repeats 3 \
    --output artifacts/benchmark/multi_gpu_matrix.json \
    --markdown artifacts/benchmark/multi_gpu_matrix.md
```

### Recipe 3: Quick Smoke Test on CPU or Colab T4
```bash
python examples/integration/multi_backend_benchmark.py \
    --device auto \
    --models yolo \
    --repeats 1 \
    --warmup 1 \
    --max-frames 2 \
    --output artifacts/benchmark/smoke_report.json
```

### Recipe 4: High-Resolution (640px) Real-Time Video Evaluation
```bash
python examples/integration/multi_backend_benchmark.py \
    --device cuda:0 \
    --models yolo,d-fine \
    --source artifacts/benchmark-4-modes/stream_120.mp4 \
    --imgsz 640 \
    --max-frames 60 \
    --repeats 2
```

---

## 4. Output Artifacts & Reporting

Every benchmark execution saves its results into an isolated run folder (`artifacts/benchmarks/run_<timestamp>/`) with segregated subdirectories:
```text
artifacts/benchmarks/run_<timestamp>/
├── README.md                          # Quick index and navigation guide
├── metrics/                           # Numerical telemetry and raw tabular data
│   ├── benchmark_metrics.json         # Complete raw JSON measurements
│   └── benchmark_tables.md            # Markdown performance tables (FPS, Latency, RAM, VRAM)
└── evaluations/                       # Dedicated analytical reports and insights
    ├── summary_evaluation.md          # Multi-backend cross-tier evaluation & architectural insights
    ├── cpu_evaluation.md              # In-depth CPU latency, memory footprint, and bottlenecks
    └── gpu_evaluation.md              # In-depth GPU acceleration, VRAM containment, and throughput
```

---

## 5. Verification & Quality Gates

When creating or modifying benchmarks under this skill:
1. **Rule 1 (Facade Imports)**: Only import from `klygo` top-level facades.
2. **Rule 2 (Dogfooding)**: Use `klygo.files` for file resolution, directories, and data serialization. Never import raw `pathlib.Path` or `json.dump`.
3. **Rule 3 (English Only)**: All comments, docstrings, CLI help messages, and report text must be written in English.
4. **Rule 5 (Artifact Isolation)**: Always save benchmark outputs inside `artifacts/benchmark/`, never inside package source directories.
