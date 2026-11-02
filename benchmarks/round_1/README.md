# Benchmark Round 1: Pipeline Smoke Test & Historical Baseline

This directory contains the telemetry datasets and analytical reports from the **Round 1 (Smoke Test)** evaluation run.

---

## 1. Directory Contents

```text
benchmarks/round_1/
├── README.md                      # Round 1 documentation & methodology retrospective
├── metrics/
│   ├── benchmark_metrics.json     # Raw JSON execution metrics (Round 1)
│   └── benchmark_tables.md        # Initial summary tables across CPU & GPU
└── evaluations/
    ├── summary_evaluation.md      # Multi-backend cross-tier evaluation
    ├── cpu_evaluation.md          # CPU latency & memory observations
    └── gpu_evaluation.md          # GPU acceleration & VRAM observations
```

---

## 2. Round 1 Methodology Retrospective & Limitations

While Round 1 successfully validated that the 6 execution modes (Raw Stream, Raw Non-stream, Native Stream, Native Non-stream, Managed Stream, Managed Non-stream) execute without errors across model backends, a rigorous scientific review identified **9 critical methodological limitations** that prevent these figures from being used as definitive quantitative speedup claims:

1. **Cold-start vs Steady-State Asymmetry**: Raw mode included model instantiation (`YOLO(model_id)`) inside the timer, while Klygo preloaded models outside the measurement block.
2. **Fixed Sequential Execution & Lack of Symmetric Warmup**: Managed modes always executed last, benefiting from hot CUDA kernel caches, pre-warmed OS page caches, and PyTorch allocator pools.
3. **Small Sample Size (4 Synthetic Frames, 1 Repeat)**: A single OS scheduling jitter or garbage collection cycle heavily distorted FPS figures; lacks repeats, median, and variance metrics.
4. **Asymmetric Memory Tracking**: Raw model weights were instantiated inside `tracemalloc`, whereas Klygo models were allocated prior to tracking; RSS was measured as absolute values rather than baseline deltas.
5. **Memory Bleed Between Frameworks**: Earlier models remained allocated in memory/VRAM when subsequent models executed.
6. **Inconsistent Batch Sizing**: Raw Grounding DINO processed 4 frames in a single batch, while Klygo used a batch size of 2.
7. **Non-Stream TTFF Misnomer**: Non-streaming execution times were labeled TTFF, which conflates total batch completion with generator streaming responsiveness.
8. **Absence of Detection Parity Verification**: Bounding box outputs, confidence scores, and coordinate hashes were not verified against ground truth to confirm output integrity.
9. **Incomplete Architecture Coverage**: Lacked Torchvision Faster R-CNN, KerasHub D-FINE, and Locate Anything in native modes.

These results are preserved here strictly as a historical baseline. All future authoritative benchmarks are conducted under **Round 2** specifications.
