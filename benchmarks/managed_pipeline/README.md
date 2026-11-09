# Managed input pipeline benchmark

This benchmark measures ordered video preparation separately from real YOLO
prediction. It uses a selected checkout through `PYTHONPATH`, a warmup run, and
the median of three repetitions. Reports and input clips belong under
`artifacts/managed-pipeline/`.

## Run

From the repository root:

```bash
PYTHONPATH=. python benchmarks/managed_pipeline/benchmark.py clip.mp4 --scope decode --batch 4 --output artifacts/managed-pipeline/decode.json
PYTHONPATH=. python benchmarks/managed_pipeline/benchmark.py clip.mp4 --scope yolo --weights artifacts/weights/yolov8n.pt --batch 4 --device cpu --output artifacts/managed-pipeline/yolo.json
```

Use `--device cuda:0` for CUDA measurements. CUDA work is synchronized around
prediction, not on each frame. The decode benchmark includes a full pixel hash
for parity checking and excludes inference. Prediction measurements include
source resolution, inference, conversion, and output consumption. They exclude
model loading and warmup. Every repeated run checks its output signature.

`--native` additionally measures Klygo native prediction at batch 1, since the
current native API reserves the top-level batching controls. Compare native and
managed rates directly only when `--batch 1` is used. This flag is optional so
older checkouts with native loading-option forwarding errors can still be
benchmarked. No public prediction parameters were added or changed.

## Local comparison against fb0f42f

Environment: Python 3.12, OpenCV 5.0.0, PyTorch 2.15.0+cpu,
Ultralytics 8.4.208, CPU only. OpenCV image processing used one thread;
Ultralytics selected eight PyTorch CPU threads. Timings are illustrative local
measurements, not a prediction for a different GPU, codec, or source.

Input preparation used a 120-frame, 1280x720 H.264 test-pattern clip with GOP 60,
B-frames, and batch 4. The source and decoded pixel signatures matched in every
configuration.

| Preparation configuration | Before, frames/s | After, frames/s |
| --- | ---: | ---: |
| Every frame, workers=1, prefetch=1 | 303.03 | 371.18 |
| Every frame, workers=4, prefetch=2 | 32.02 | 378.89 |
| Every third frame, workers=1, prefetch=1 | 19.62 | 246.80 |
| Every third frame, workers=4, prefetch=2 | 24.42 | 271.74 |

The large improvements above concern preparation only: ordered reads eliminate
worker-induced backward seeks, and short forward strides use decoder advancement
instead of repeated keyframe seeks. Model inference is not included.

A separate 48-frame H.264 clip made from Ultralytics' bus sample exercised real
YOLOv8n detections at imgsz=320, batch 4. Rounded coordinates, scores, and class
IDs matched between checkouts, including non-empty detections.

| Full prediction configuration | Before, FPS | After, FPS |
| --- | ---: | ---: |
| Raw YOLO control | 80.35 | 77.77 |
| Managed, workers=1, prefetch=1 | 77.39 | 70.32 |
| Managed, workers=4, prefetch=2 | 77.02 | 82.39 |

The default-worker CPU measurement regressed in this run, while the four-worker
configuration improved. A confirmation run using five repetitions measured
76.57 -> 86.15 FPS for one worker and 71.92 -> 82.13 FPS for four workers. However,
the raw control also varied substantially (58.53 -> 83.13 FPS in that run).
These short runs do not establish a universal end-to-end speedup or isolate a
precise model-level gain; repeat measurements on the actual source and hardware
before choosing workers and prefetch. In particular, the preparation speedup
must not be presented as a YOLO FPS speedup. CUDA throughput has not been measured
in this environment.

## Correctness and compatibility

The public API signatures and float64 output defaults are unchanged. Regression
tests cover ordered video reads, output pixel parity, lazy preprocessing trace
restoration, retained user caches, single-batch parallelism, bounded source
consumption, early close, source/decode errors, and native array ownership.
Independent image files still decode in worker threads. Video decode happens in
one source producer, followed by parallel preprocessing where needed. Already
prepared BGR video frames bypass a redundant worker hand-off.

Validation: 517 tests completed successfully, with 21 conditional skips. Model
and media API audits passed. The full documentation audit reports six broken
links also present at fb0f42f. The four-backend integration entry point could not
run because Keras was unavailable; real YOLO CPU inference was tested separately.
