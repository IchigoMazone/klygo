# Model and Streaming Benchmark Evaluation Criteria

This document defines how Klygo performance must be measured, which execution
paths must be compared, and what constitutes an acceptable result. It separates
raw backend speed from the cost of Klygo features so that benchmark conclusions
remain technically fair.

## 1. Execution Paths Under Evaluation

### A. Raw backend

The backend is called directly and returns its native result type.

```python
raw = detector.model.predict(source=video, stream=True, verbose=False)
```

For Ultralytics, this returns `ultralytics.engine.results.Results`. It does not
provide Klygo `Detection` objects, stable UIDs, source identity, shared label
registries, preprocessing trace restoration, or standardized metadata.

### B. Klygo native

The backend owns input decoding and inference, while Klygo normalizes every
result into its public output containers.

```python
results = detector.predict(video, stream=True, native=True, verbose=False)
```

This path measures the cost of Klygo output conversion and lifecycle management
without measuring the Klygo media pipeline.

### C. Klygo managed

Klygo owns media loading, batching, inference dispatch, result normalization,
metadata, source identity, and coordinate restoration.

```python
results = detector.predict(video, stream=True, verbose=False)
```

This is the primary public pipeline and the most important production result.

### D. Klygo managed, tuned

This is the managed pipeline with workload-specific scheduling parameters.

```python
results = detector.predict(
    video,
    stream=True,
    batch=2,
    workers=4,
    prefetch=2,
    verbose=False,
)
```

This path determines whether scheduling hides decode overhead without violating
the latency or memory requirements of the workload.

## 2. Required Comparisons

| ID | Compare | Purpose | Main conclusion |
| :--- | :--- | :--- | :--- |
| C1 | Raw backend vs Klygo native | Isolate output normalization, locking, state management, and `Detection` construction | Cost of the Klygo output layer |
| C2 | Klygo native vs Klygo managed | Isolate Klygo media decoding, batching, and preprocessing dispatch | Cost or benefit of the Klygo execution pipeline |
| C3 | Raw backend vs Klygo managed | Measure the complete user-facing abstraction cost | Real end-to-end Klygo overhead |
| C4 | Managed default vs managed tuned | Evaluate `batch`, `workers`, and `prefetch` | Whether scheduling improves throughput or harms latency |
| C5 | Non-stream vs stream | Compare materialized and lazy execution | Streaming latency and memory benefit |
| C6 | Plain input vs traced preprocessing input | Measure coordinate restoration | Cost and correctness of trace handling |
| C7 | Standard vs packed output | Evaluate offline result storage | RAM reduction and conversion cost |
| C8 | CPU vs CUDA; float32 vs float16/bfloat16 | Evaluate hardware and precision scaling | Device utilization and numerical stability |
| C9 | One stream vs concurrent streams | Test contention and backpressure | Multi-camera scalability |
| C10 | Base commit vs candidate commit | Validate an optimization | Measured gain without correctness regression |

Raw backend and Klygo managed do not return feature-equivalent objects. C3 must
therefore be reported as the price of the complete Klygo contract, not as an
isolated inference regression. C1 and C2 identify where that price is paid.

## 3. Benchmark Categories

### 3.1 Latency

Measure:

- Model load time.
- Time to first frame.
- Per-frame latency at P50, P95, and P99.
- End-to-end latency from decode start to completed output object.
- Frame-time jitter.

Average latency alone is insufficient for realtime evaluation. P95 and P99
identify visible stalls that average FPS can hide.

### 3.2 Throughput

Measure:

- Frames per second after warmup.
- Images per second for indexed folders.
- Total processing time for a fixed frame count.
- Throughput scaling as batch size increases.

Report both the median of complete runs and paired differences between execution
paths from the same repeat.

### 3.3 Resource use

Measure:

- Peak and steady-state RAM.
- Peak and steady-state VRAM.
- CPU utilization.
- GPU utilization.
- GPU power and clock stability when available.
- Memory growth during a 10- to 30-minute stream.

A lazy stream must not retain all decoded frames. Long-run memory should reach a
stable plateau after initialization and cache warmup.

### 3.4 Correctness and parity

Compare:

- Frame count and frame order.
- Box count per frame.
- Labels and class IDs.
- Scores.
- Bounding-box coordinates.
- Source path, source ID, and frame index.
- UID uniqueness and stability.
- Restored coordinates after preprocessing.

Recommended tolerances for equivalent precision are:

```text
maximum box-coordinate delta: 0.01 pixel
maximum score delta:          0.0001
frame and label parity:       100%
```

Different precision modes must declare wider tolerances before the run.

### 3.5 Realtime behavior

Measure:

- Dropped frames.
- Queue growth and backpressure.
- Capture-to-result latency.
- Recovery after decode failure or source interruption.
- Shutdown time and resource release.

Realtime frame budgets are:

| Source rate | Frame budget |
| :--- | ---: |
| 15 FPS | 66.67 ms |
| 30 FPS | 33.33 ms |
| 60 FPS | 16.67 ms |

Passing average FPS is not enough. P95 latency must remain within the required
frame budget, or the report must explicitly describe buffering and dropped-frame
behavior.

### 3.6 Stability

Run representative streams for at least 10 minutes and production candidates
for 30 minutes. Record exceptions, corrupted outputs, memory growth, FPS decay,
and final resource cleanup. A performance gain is invalid if the optimized path
changes output order, loses frames, or leaks memory.

## 4. Workload Matrix

### 4.1 Inputs

- One image.
- A folder containing 100, 1,000, and 10,000 images.
- A short deterministic MP4 for repeatable profiling.
- A varied-content production video.
- A 10- to 30-minute video.
- Webcam input.
- RTSP input when available.
- `media.load(..., stream=True)` input.
- Indexed `MediaFrames` input.
- In-memory PIL and NumPy frames.
- Preprocessed frames carrying coordinate traces.

### 4.2 Video properties

| Dimension | Required values |
| :--- | :--- |
| Resolution | 640x480, 1280x720, 1920x1080; 4K when relevant |
| Source rate | 15, 30, and 60 FPS |
| Codec | H.264, H.265, and MJPEG when supported |
| Scene | Static, fast motion, no detections, sparse boxes, dense boxes |

A repeated-frame video is useful for isolating framework overhead. It must be
paired with varied real footage before making production claims.

### 4.3 Scheduling parameters

```text
batch:     1, 2, 4, 8, 16
workers:   1, 2, 4, 8
prefetch:  1, 2, 4
imgsz:     320, 640, 1280
precision: float32, float16, bfloat16 where supported
```

Realtime runs should prioritize `batch=1` or `batch=2`. Offline throughput runs
may use larger batches. A configuration must not be described as faster without
also reporting its latency and memory cost.

### 4.4 Models and backends

- YOLO nano: exposes fixed Python and output-layer overhead.
- YOLO small or medium: tests a more GPU-bound workload.
- YOLO-World: tests prompt and class-state handling.
- Grounding DINO: tests Hugging Face input casting and optional `device_map`.
- Faster R-CNN: tests Torchvision integration.
- D-FINE or RT-DETR: tests KerasHub integration.

Only a backend with a genuinely separate native pipeline should claim a native
speed comparison. When `native=True` follows the same Klygo pipeline, differences
must be treated as measurement noise.

### 4.5 Hardware

At minimum, record:

- CPU model and logical core count.
- GPU model and VRAM.
- Operating system.
- Python, CUDA, Torch, backend, and Klygo versions.
- Power mode when known.

Recommended GPU coverage includes an older local GPU, NVIDIA T4, a recent RTX
GPU, and NVIDIA A100 when available. Results must not be extrapolated from one GPU
class to another without measurement.

## 5. Fairness Requirements

Every compared path must use the same:

- Model weights.
- Input file and decoded frame selection.
- Device and precision.
- Image size.
- Batch size where both paths support it.
- Confidence threshold, IoU threshold, NMS settings, and class filters.
- Video stride and maximum frame count.
- Output consumption: lazy iterators must be fully consumed.

The benchmark procedure must:

1. Warm up every path using the complete representative workload.
2. Synchronize CUDA immediately before and after timed sections.
3. Run paths in alternating order.
4. Use at least seven measured repeats for final reports.
5. Report the median, P95 where applicable, and every raw sample.
6. Verify output parity after timing.
7. Avoid unrelated CPU, disk, or GPU workloads.
8. Store generated inputs and reports under `artifacts/`.

Model loading should be excluded from inference timing and reported separately.

## 6. Required Result Breakdown

Where instrumentation permits, report these stages separately:

```text
decode
preprocessing
CPU-to-GPU transfer
inference
NMS and backend postprocessing
GPU-to-CPU transfer
Detection construction
UID, source, label, and metadata attachment
coordinate restoration
locking and prediction-state management
```

The optimization target must be selected from measured stage costs. Total time
alone is not sufficient evidence for changing an API or removing a feature.

## 7. Calculations

```text
FPS = processed_frames / total_seconds

raw speedup over managed (%) =
    (managed_seconds / raw_seconds - 1) * 100

managed FPS loss relative to raw (%) =
    (raw_fps - managed_fps) / raw_fps * 100

memory growth (%) =
    (final_steady_memory - initial_steady_memory) / initial_steady_memory * 100
```

The report must state which formula it uses. Time speedup and FPS loss are related
but produce different percentages and must not be presented as the same metric.

## 8. Acceptance Gates

These are initial engineering targets and may be tightened for a specific
deployment:

| Criterion | Target |
| :--- | :--- |
| Klygo native overhead relative to raw backend | At most 5-10% |
| Klygo managed overhead relative to raw backend | At most 10-15% |
| Frame and label parity | 100% |
| Dropped frames for file input | 0 |
| Long-stream RAM growth after warmup | At most 5% |
| P95 latency | Within the deployment frame budget |
| Stream exceptions during stability run | 0 |
| Resource cleanup after close or unload | Complete |

For a fast nano model at batch size one, fixed output-construction costs may be
more visible. Any exception to the target must include a stage-level profile and
an explanation of the user-facing functionality that accounts for the cost.

## 9. Standard Report Table

Every final benchmark should provide one row per execution path and configuration:

| Path | Device | Precision | Batch | Workers | Prefetch | First frame | P50 | P95 | FPS | RAM peak | VRAM peak | Frames | Boxes | Parity |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Raw backend | | | | | | | | | | | | | | |
| Klygo native | | | | | | | | | | | | | | |
| Klygo managed | | | | | | | | | | | | | | |
| Klygo managed tuned | | | | | | | | | | | | | | |

The accompanying report must include raw timing samples, environment details,
the exact command or script, the tested commit, and generated artifact paths.

## 10. Current Local Reference Result

The local reference run on 2026-10-28 used YOLOv8n, CUDA 12.6, a Quadro M1200,
`imgsz=320`, and a 120-frame 640x480 MP4 after full-video warmup.

| Path | Median FPS | Median first frame | Median total time |
| :--- | ---: | ---: | ---: |
| Raw Ultralytics | 49.53 | 36.0 ms | 2.423 s |
| Klygo managed | 37.34 | 61.1 ms | 3.214 s |

All 120 frames had matching box counts. Raw Ultralytics was 32.6% faster by
total-time ratio, while Klygo delivered 24.6% fewer FPS. Both exceeded the source
rate of 30 FPS. This run used repeated image content and therefore serves as an
overhead baseline, not a complete production-video conclusion.

