# Six-mode Colab GPU suite

## Recorded Tesla T4 results

The [2026-11-10 Colab import](colab_t4_2026_11_10/README.md) records 25 selected
configurations, 124 supported cases and seven repeats per case, with all printed
parity flags passing. It includes the executed notebook, rounded FPS summaries,
a separate routing control check and complete 601.77-second stability telemetry.
The large matrix bundle was hidden in the saved Colab output; individual matrix
timings and latency/telemetry records are unavailable in this import. See the
report's evidence section before interpreting the results.

The suite measures registered YOLOv8n, YOLO11n, YOLO26n, YOLO-World v2 small,
Grounding DINO tiny/base, Faster R-CNN MobileNetV3 FPN and LocateAnything-3B,
plus the existing KerasHub D-FINE Nano integration. Nano/small representatives
cover the registry families; this is not every possible YOLO checkpoint size.

Six rows are always reported: raw non-stream/stream, native non-stream/stream,
managed non-stream/stream. A framework without an upstream source-stream engine
gets N/A for raw/native stream; Python loops are not relabeled as native streams.
At the tested revision, Klygo native-stream consumes the upstream generator into
a materialized result. Its time to first caller-visible result includes that
buffering. The public API locks native video batch to one; larger video batches
therefore have N/A for native rows.

## Protocol

- Isolated model subprocesses; CUDA required unless CPU is explicitly requested.
- FP32 for detector families; LocateAnything uses FP16, one `person` query,
  `generation_mode=fast`, `max_new_tokens=512`, deterministic sampling disabled.
- Same weights, prompts, thresholds and effective inference batch across paths.
- One complete warmup per mode and seven measured repeats in alternating order.
- CUDA synchronization around each complete timed pass. Output conversion and
  parity checks occur after timing. First-frame and interval timestamps in those
  passes are host delivery observations, not isolated kernel timings.
- A separate latency diagnostic synchronizes at each result and reports P50,
  P95/P99 delivery intervals and pack-call latency. Non-stream/batched delivery
  has bursts; zero-length intervals do not imply zero inference latency.
- Throughput passes retain yielded results for untimed comparison. Their memory
  peaks include result retention; bounded streaming memory is checked separately
  by the non-retaining ten-minute stability probe.
- Sampled RSS/CPU and optional NVML utilization, power, clocks and temperature;
  Torch allocated/reserved VRAM peaks. Sampling cannot capture arbitrarily short
  peaks. NVML utilization is a device-window statistic, not exact per-call work.
- Numeric parity checks compare per-frame labels/counts and coordinates/scores
  with tolerances 0.01 pixel and 0.0001. Mismatches and errors remain in reports.
- Package versions, exact commit, runner hash, load time and input digest saved.

## Workloads

`images` uses predecoded 640x480 PIL packs with equal effective batch (one or
four images) in every supported path. Each timed pass repeats the pack eight
times. Decode is excluded in all paths. These short API-pack measurements expose
input handling and output overhead; they are not long-video throughput results.

`video` uses a controlled 120-frame 1280x720 moving bus clip, encoded as H.264
through Klygo media helpers. It includes decoding and source loading, and applies
to the Ultralytics models that own a real source decoder. It is not varied
production footage, webcam capture or a long stability qualification.

```bash
PYTHONPATH=. python benchmarks/gpu_suite/run.py --model yolo --batch 1 \
  --repeats 7 --output artifacts/benchmarks/six-mode/images-b1
PYTHONPATH=. python benchmarks/gpu_suite/run.py --model yolo --profile video \
  --video-frames 120 --batch 1 --repeats 7 \
  --output artifacts/benchmarks/six-mode/video-b1
```

LocateAnything requires the explicit `--allow-remote-code` flag and the pinned
official NVIDIA checkpoint. It is measured separately at batch one with one
pack per repeat due to its sequential autoregressive generation. A failed load,
truncated output or OOM is recorded as an error rather than fabricated FPS.

Runtime outputs belong under `artifacts/benchmarks/`. Published measured reports
are deliberately versioned only when requested; weights and generated inputs
are never committed. See [the benchmark criteria](../../docs/models/BENCHMARK_CRITERIA.md).
