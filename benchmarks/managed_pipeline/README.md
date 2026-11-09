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
must not be presented as a YOLO FPS speedup. CUDA results are recorded separately below.

## Correctness and compatibility

The public API signatures and float64 output defaults are unchanged. Regression
tests cover ordered video reads, output pixel parity, lazy preprocessing trace
restoration, retained user caches, single-batch parallelism, bounded source
consumption, early close, source/decode errors, and native array ownership.
Independent image files still decode in worker threads. Video decode happens in
source order, followed by parallel preprocessing where needed. The default single-frame, single-worker path uses one executor; larger or transformed pipelines use a separate source producer. Already
prepared BGR video frames bypass a redundant worker hand-off.

Validation: 519 tests completed successfully, with 21 conditional skips. Model
and media API audits passed. The six pre-existing case-sensitive documentation links found at fb0f42f were corrected. The four-backend integration entry point could not
run because Keras was unavailable; real YOLO CPU inference was tested separately.

## Recorded Tesla T4 comparison

Tesla T4; PyTorch 2.14.1+cu130; Ultralytics 8.4.208; OpenCV 4.14.0.
YOLOv8n, imgsz=640, FP32; 240 frames, 1280x720; full-clip warmup and three timed repetitions.

| Configuration | Raw FPS | Baseline FPS | Initial optimization FPS |
| --- | ---: | ---: | ---: |
| Native, batch 1 | 85.86 | 84.69 | 78.09 |
| Managed, batch 1, workers 1 | 85.86 | 75.13 | 64.05 |
| Managed, batch 1, workers 4 | 85.86 | 76.14 | 64.57 |
| Managed, batch 4, workers 4 | 159.68 | 27.48 | 122.30 |
| Managed, batch 4, workers 4, stride 3 | 97.34 | 7.91 | 84.64 |

Batch-1 confirmation runs varied, including reversed comparisons; all timings are
retained in the JSON report rather than selecting only favorable runs. Strict hash
differences were followed by numerical comparisons: per-frame counts and class IDs
matched; the largest measured coordinate error was 0.00048828125 pixel and score
error was 0.0000013709068298339844.

### Final single-worker follow-up

Interleaved order: follow-up A, previous A, previous B, follow-up B; three timings per
process. Pooled over six timings per version:

- Previous optimization: 75.54568461090905 FPS.
- Follow-up: 77.34311744093024 FPS (about 2.4% higher in this run).
- Follow-up batch 4, workers 4: 128.66798054151445 FPS.
- Source SHA256: `5ac4d0bc083821a384d9517a6e2438d8f9ed4a7bee97c634905e29954589241e`.

The small batch-1 difference is not evidence of a universal gain. The stronger result
is removal of reader-induced backward seeks in multi-worker video batches. The
controlled clip is not the original video behind the earlier 90/87/30 FPS observation.
Raw stride-3 can begin at a different frame; compare the two Klygo revisions for parity.

## Reproduce the GPU measurements

Open [T4_Before_After.ipynb](T4_Before_After.ipynb) on Colab with a T4 runtime and
run in order. The notebook preserves the exact measured code, including full-clip
warmup, isolated per-revision processes, CUDA timing boundaries, and numerical
output checks. It pins the two initial source revisions and applies the recorded
follow-up patch in a separate worktree. The runtime's CUDA/PyTorch packages are
not reinstalled; their actual versions are recorded.

[metrics/t4.json](metrics/t4.json) contains the actual individual timings for all
13 main cases, seven confirmation cases, and five follow-up cases, including
unfavorable and variable runs. The exact tested follow-up is identified by its
base commit, source hash, and [patch](metrics/t4_followup.diff). New runs write to
artifacts/; the recorded report is a deliberately versioned benchmark fixture.
No model weights, videos, browser state, or account identifiers are committed.

The CLI helper at the top of this guide includes signature verification inside
its timing interval. Its historical CPU results and the CUDA notebook results
therefore have different measurement scopes; do not compare those FPS values as
if they were the same protocol. Use the notebook for this recorded GPU comparison.
