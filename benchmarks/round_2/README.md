# Benchmark Round 2: Rigorous Multi-Process Benchmark Harness

This directory contains the specifications, test artifacts, telemetry metrics, and analytical reports for the **Round 2** benchmark evaluation.

---

## 1. Directory Structure

```text
benchmarks/round_2/
├── README.md                      # Round 2 methodology specification & execution plan
├── metrics/                       # Target directory for Round 2 raw JSON & tabular metrics
│   └── .gitkeep
└── evaluations/                   # Target directory for Round 2 analytical reports & hardware scorecards
    └── .gitkeep
```

---

## 2. Round 2 Scientific Methodology Standards

Round 2 addresses every methodological limitation identified in Round 1 through the following rigorous evaluation contract:

1. **Subprocess Isolation**: Every single execution mode runs inside an independent subprocess (`multiprocessing` / Python subprocess), guaranteeing zero state leakage, zero CUDA allocator caching bias, and 100% memory reclamation upon process termination.
2. **Symmetric Warmup Phase**: Exactly 3–5 warmup frames are processed and discarded for every mode prior to timer and memory instrumentation.
3. **Strict Model Loading Isolation**: Models, weights, and graph compilation occur strictly outside the timing and memory measurement window across all pipelines (Raw, Native, and Managed).
4. **Memory Delta Instrumentation**: Tracks true delta memory:
   $$\Delta \text{Memory} = \text{Peak Memory} - \text{Baseline Memory}$$
   measured directly before and during the inference loop for both Python Heap (`tracemalloc`) and OS Process RSS / GPU VRAM (`psutil` / `torch.cuda`).
5. **Real Dataset & Multi-Repeat Variance**: Evaluated on 30–60 real validation frames (640×640) with 5 independent repeats, reporting **Median FPS**, **Mean FPS**, **Standard Deviation (Jitter)**, and **p95 Latency**.
6. **Input & Batch Parity**: Standardized batch sizing (`batch=1` default) with identical image formats across all competing modes.
7. **Strict Latency Metric Separation**:
   - **TTFF (Time to First Frame)**: Exclusively measured on lazy generators (`next(stream)`).
   - **Total Batch Latency (ms)**: Measured on blocking batch completions.
8. **Automated Detection Parity Check**: Validates that Raw, Native, and Managed pipelines produce identical bounding box counts, class labels, and mean IoU.
9. **Full Multi-Backend Matrix**: Covers all 5 integrated architectures:
   - Ultralytics YOLOv8 / YOLOv11
   - Ultralytics YOLO-World v2
   - Hugging Face Grounding DINO
   - Torchvision Faster R-CNN
   - KerasHub / Keras 3 D-FINE
