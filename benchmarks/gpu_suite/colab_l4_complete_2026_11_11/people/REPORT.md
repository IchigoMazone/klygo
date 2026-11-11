# Colab GPU benchmark: six execution modes

All values below are actual measurements. Unsupported modes are N/A; failed jobs remain errors.

## Scope and interpretation

- Image-pack results exclude decoding uniformly and include a separate public API call per pack. They are not video throughput.
- Video results include source loading/decoding. Compare paths within a model, profile and batch, not across different workloads.
- Seven alternating measured repeats follow complete warmup. Conversion, hashing and parity checks are outside throughput timing.
- Native-stream currently buffers upstream results into a materialized Klygo result. TTFF therefore includes that buffering.
- Native video batch is locked to one by the tested API; native rows at video batch four are N/A.
- Most detectors use FP32. LocateAnything uses FP16, one query, fast generation and a 512-token limit; it is not directly comparable to FP32 detector precision.
- First-frame figures in throughput runs are host-visible delivery observations. The separate ready-latency pass synchronizes each result and is not included in FPS.
- Batched/non-stream output arrives in bursts. Delivery intervals are not isolated per-frame inference time; ready pack latency includes the entire call.
- Image diagnostic sample counts depend on the recorded iteration and batch settings. Their interpolated P95/P99 are descriptive samples, not tail-latency SLA evidence.
- Throughput passes retain yielded results for untimed parity; peak memory includes retention. The separate stability probe consumes without retaining frames.
- RSS is sampled at 50 ms; short peaks can be missed. CPU percentages are approximate, expressed relative to one core and include probe setup. NVML uses device-window statistics.
- Reserved VRAM includes allocator caches from preceding modes. Allocated peaks describe live tensor allocation; neither is an isolated minimum-memory requirement.
- Factory load time excludes process/hardware initialization and includes checkpoint lookup/download when needed; cache states differ between jobs.
- These recorded workloads do not qualify production accuracy or live-camera behavior. A separate stability result, when present, covers one managed pipeline, not all models or 30-minute production use.

## Environment

GPU: **NVIDIA L4**, VRAM 22563.12 MiB. CUDA 13.0.
Python: `3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]`. OS: `Linux-6.6.159+-x86_64-with-glibc2.39`.
CPU logical cores: 12; Torch threads: 2.

| Package | Version |
| --- | --- |
| klygo | 2.1.0 |
| torch | 2.14.1+cu130 |
| torchvision | 0.29.1+cu130 |
| ultralytics | 8.4.208 |
| transformers | 4.57.1 |
| keras | 3.13.2 |
| keras-hub | 0.26.0 |
| opencv-python | 5.0.0.93 |

Tested commit(s): `8b872985e9f7a9910e0736b79bed6c0effe0b518`

## FPS matrix — video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 113.50 | 113.05 | 108.15 | 106.13 | 104.11 | 103.56 |
| yolo11 | 90.06 | 91.62 | 88.16 | 87.18 | 85.81 | 85.14 |
| yolo26 | 82.70 | 83.61 | 80.59 | 79.53 | 78.82 | 78.92 |
| yolo-world | 82.10 | 82.22 | 80.60 | 77.83 | 78.64 | 78.49 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 113.50 ± 1.61 | 1057.18 | 1069.69/1069.69/1069.69 | 1926.74 | 34.28/62.00 | 120/60 | PASS |
| raw-stream | 113.05 ± 1.62 | 23.08 | 1086.45/1086.45/1086.45 | 1933.03 | 34.28/62.00 | 120/60 | PASS |
| native-non-stream | 108.15 ± 1.44 | 1109.49 | 1118.10/1118.10/1118.10 | 1933.04 | 34.28/62.00 | 120/60 | PASS |
| native-stream | 106.13 ± 1.25 | 1130.65 | 1131.91/1131.91/1131.91 | 1926.73 | 34.25/62.00 | 120/60 | PASS |
| managed-non-stream | 104.11 ± 1.52 | 1152.56 | 1141.15/1141.15/1141.15 | 1825.30 | 24.66/62.00 | 120/60 | PASS |
| managed-stream | 103.56 ± 0.92 | 32.91 | 1183.29/1183.29/1183.29 | 1825.30 | 24.66/62.00 | 120/60 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.069990 | 112.15 | 1069.90 | 1923.53 | 34.28 | 129.27 | 13.64 | 38.76 | 2040.00 | 68.82 |
| 2 | 1.041169 | 115.26 | 1041.09 | 1925.18 | 34.28 | 126.62 | 13.76 | 39.63 | 2040.00 | 70.81 |
| 3 | 1.036204 | 115.81 | 1036.12 | 1925.71 | 34.28 | 126.87 | 13.86 | 39.77 | 2040.00 | 70.81 |
| 4 | 1.056152 | 113.62 | 1056.07 | 1925.74 | 34.28 | 125.43 | 15.38 | 39.49 | 2040.00 | 72.00 |
| 5 | 1.057267 | 113.50 | 1057.18 | 1926.27 | 34.28 | 126.54 | 14.62 | 39.45 | 2040.00 | 72.05 |
| 6 | 1.077104 | 111.41 | 1077.02 | 1926.62 | 34.28 | 125.27 | 14.18 | 40.70 | 2040.00 | 73.77 |
| 7 | 1.066503 | 112.52 | 1066.42 | 1926.74 | 34.28 | 126.91 | 14.55 | 39.82 | 2040.00 | 73.82 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.07 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3760524583252238, 'inference': 6.496674824984439, 'postprocess': 0.8423283333362027}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.078121 | 111.30 | 23.72 | 1923.49 | 34.28 | 127.04 | 13.36 | 39.22 | 2040.00 | 68.82 |
| 2 | 1.090879 | 110.00 | 24.52 | 1932.53 | 34.28 | 125.99 | 13.09 | 39.33 | 2040.00 | 70.59 |
| 3 | 1.044847 | 114.85 | 23.58 | 1925.71 | 34.28 | 126.10 | 14.57 | 39.79 | 2040.00 | 70.90 |
| 4 | 1.060394 | 113.17 | 22.66 | 1933.03 | 34.28 | 126.48 | 14.91 | 39.35 | 2040.00 | 71.91 |
| 5 | 1.067062 | 112.46 | 22.83 | 1926.30 | 34.28 | 125.41 | 14.82 | 40.24 | 2040.00 | 72.27 |
| 6 | 1.061487 | 113.05 | 23.08 | 1927.24 | 34.28 | 126.09 | 13.91 | 39.99 | 2040.00 | 73.14 |
| 7 | 1.053082 | 113.95 | 22.65 | 1926.71 | 34.28 | 126.62 | 15.29 | 40.19 | 2040.00 | 73.81 |

Ready result delivery-interval P50/P95/P99 ms: 8.96/9.83/11.25.
Separate latency-pass duration: 1.09 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4029875333487023, 'inference': 6.500840366660062, 'postprocess': 0.8714656416259459}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.70% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.109572 | 108.15 | 1109.49 | 1923.89 | 34.28 | 124.56 | 13.86 | 40.07 | 2040.00 | 68.91 |
| 2 | 1.126396 | 106.53 | 1126.30 | 1932.45 | 34.28 | 126.83 | 14.43 | 39.31 | 2040.00 | 70.13 |
| 3 | 1.108023 | 108.30 | 1107.90 | 1925.74 | 34.28 | 125.08 | 15.27 | 39.82 | 2040.00 | 70.91 |
| 4 | 1.088977 | 110.20 | 1088.88 | 1933.04 | 34.28 | 127.09 | 15.64 | 39.41 | 2040.00 | 71.91 |
| 5 | 1.093135 | 109.78 | 1093.05 | 1926.28 | 34.28 | 125.04 | 14.86 | 40.39 | 2040.00 | 72.77 |
| 6 | 1.111415 | 107.97 | 1111.32 | 1927.27 | 34.28 | 127.11 | 13.83 | 39.88 | 2040.00 | 73.04 |
| 7 | 1.127582 | 106.42 | 1127.49 | 1926.73 | 34.28 | 125.73 | 15.09 | 40.24 | 2040.00 | 73.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.12 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### native-stream

6.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.137924 | 105.46 | 1137.84 | 1924.02 | 34.25 | 123.20 | 12.87 | 39.25 | 2040.00 | 69.04 |
| 2 | 1.158797 | 103.56 | 1158.72 | 1822.49 | 34.25 | 122.92 | 12.57 | 38.83 | 2040.00 | 70.00 |
| 3 | 1.130718 | 106.13 | 1130.65 | 1925.79 | 34.25 | 123.40 | 13.43 | 39.48 | 2040.00 | 71.00 |
| 4 | 1.122640 | 106.89 | 1122.57 | 1823.23 | 34.25 | 122.91 | 13.04 | 38.95 | 2040.00 | 71.83 |
| 5 | 1.124824 | 106.68 | 1124.75 | 1926.27 | 34.25 | 122.86 | 13.26 | 39.54 | 2040.00 | 72.78 |
| 6 | 1.122976 | 106.86 | 1122.90 | 1818.37 | 34.25 | 122.83 | 14.00 | 39.93 | 2040.00 | 72.91 |
| 7 | 1.145008 | 104.80 | 1144.94 | 1926.73 | 34.25 | 123.34 | 13.09 | 39.87 | 2040.00 | 73.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-non-stream

8.61% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.146489 | 104.67 | 1146.39 | 1816.02 | 24.66 | 121.98 | 13.04 | 39.41 | 2040.00 | 69.52 |
| 2 | 1.152664 | 104.11 | 1152.56 | 1817.45 | 24.66 | 121.96 | 13.61 | 38.79 | 2040.00 | 69.91 |
| 3 | 1.158766 | 103.56 | 1158.67 | 1817.78 | 24.66 | 122.26 | 12.52 | 39.23 | 2040.00 | 71.00 |
| 4 | 1.147133 | 104.61 | 1147.02 | 1818.26 | 24.66 | 121.86 | 13.83 | 38.90 | 2040.00 | 71.83 |
| 5 | 1.170989 | 102.48 | 1170.89 | 1818.15 | 24.66 | 122.00 | 13.42 | 39.81 | 2040.00 | 72.88 |
| 6 | 1.156167 | 103.79 | 1156.07 | 1825.30 | 24.66 | 121.88 | 12.87 | 39.61 | 2040.00 | 72.91 |
| 7 | 1.117431 | 107.39 | 1117.34 | 1818.66 | 24.66 | 122.23 | 14.78 | 39.67 | 2040.00 | 73.91 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.14 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-stream

9.02% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.157920 | 103.63 | 32.91 | 1823.82 | 24.66 | 121.87 | 13.04 | 39.19 | 2040.00 | 69.78 |
| 2 | 1.158931 | 103.54 | 32.03 | 1824.12 | 24.66 | 121.86 | 12.74 | 38.73 | 2040.00 | 69.87 |
| 3 | 1.158725 | 103.56 | 33.24 | 1817.96 | 24.66 | 121.75 | 13.83 | 39.36 | 2040.00 | 71.30 |
| 4 | 1.151465 | 104.22 | 32.29 | 1818.27 | 24.66 | 122.23 | 12.52 | 38.66 | 2040.00 | 71.57 |
| 5 | 1.182740 | 101.46 | 33.15 | 1825.17 | 24.66 | 122.49 | 12.54 | 38.97 | 2040.00 | 72.92 |
| 6 | 1.159679 | 103.48 | 32.38 | 1825.30 | 24.66 | 122.19 | 13.22 | 39.04 | 2040.00 | 72.83 |
| 7 | 1.152003 | 104.17 | 33.02 | 1818.48 | 24.66 | 121.70 | 13.65 | 40.04 | 2040.00 | 73.91 |

Ready result delivery-interval P50/P95/P99 ms: 9.71/10.74/12.46.
Separate latency-pass duration: 1.18 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.

## Detail — yolo11, video, batch 1

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 90.06 ± 0.80 | 1332.41 | 1321.06/1321.06/1321.06 | 2007.48 | 64.67/90.00 | 120/64 | PASS |
| raw-stream | 91.62 ± 0.98 | 25.28 | 1334.93/1334.93/1334.93 | 2008.00 | 64.67/90.00 | 120/64 | PASS |
| native-non-stream | 88.16 ± 0.41 | 1361.10 | 1372.64/1372.64/1372.64 | 2009.98 | 64.67/90.00 | 120/64 | PASS |
| native-stream | 87.18 ± 0.95 | 1376.46 | 1369.48/1369.48/1369.48 | 2008.00 | 64.67/90.00 | 120/64 | PASS |
| managed-non-stream | 85.81 ± 0.81 | 1398.31 | 1398.80/1398.80/1398.80 | 1899.72 | 56.12/90.00 | 120/64 | PASS |
| managed-stream | 85.14 ± 0.64 | 34.81 | 1422.87/1422.87/1422.87 | 1904.51 | 56.12/90.00 | 120/64 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.330168 | 90.21 | 1330.08 | 2003.11 | 64.67 | 122.66 | 13.78 | 39.80 | 2040.00 | 75.19 |
| 2 | 1.352365 | 88.73 | 1352.29 | 2005.42 | 64.67 | 120.85 | 14.11 | 40.05 | 2040.00 | 76.00 |
| 3 | 1.326083 | 90.49 | 1326.00 | 2005.45 | 64.67 | 121.81 | 12.93 | 40.17 | 2040.00 | 76.00 |
| 4 | 1.322611 | 90.73 | 1322.53 | 2005.92 | 64.67 | 122.01 | 13.89 | 40.62 | 2040.00 | 77.00 |
| 5 | 1.332504 | 90.06 | 1332.41 | 2005.88 | 64.67 | 121.35 | 13.11 | 40.59 | 2040.00 | 77.07 |
| 6 | 1.345573 | 89.18 | 1345.50 | 2006.10 | 64.67 | 123.51 | 13.48 | 40.64 | 2040.00 | 77.67 |
| 7 | 1.349481 | 88.92 | 1349.39 | 2007.48 | 64.67 | 121.33 | 12.81 | 40.41 | 2040.00 | 77.26 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3316358999342508, 'inference': 8.688080349998017, 'postprocess': 0.8616135332507232}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.64% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.305228 | 91.94 | 25.15 | 2002.96 | 64.67 | 120.95 | 13.54 | 39.90 | 2040.00 | 75.77 |
| 2 | 1.338684 | 89.64 | 24.75 | 2005.08 | 64.67 | 121.19 | 12.78 | 40.14 | 2040.00 | 76.00 |
| 3 | 1.314103 | 91.32 | 25.79 | 2005.46 | 64.67 | 121.37 | 14.15 | 40.13 | 2040.00 | 76.30 |
| 4 | 1.309766 | 91.62 | 25.06 | 2005.89 | 64.67 | 121.07 | 12.78 | 40.61 | 2040.00 | 77.00 |
| 5 | 1.301059 | 92.23 | 25.51 | 2005.89 | 64.67 | 121.45 | 13.27 | 41.07 | 2040.00 | 77.58 |
| 6 | 1.323557 | 90.66 | 25.32 | 2006.13 | 64.67 | 121.13 | 12.96 | 40.60 | 2040.00 | 77.81 |
| 7 | 1.298434 | 92.42 | 25.28 | 2008.00 | 64.67 | 120.99 | 14.42 | 40.55 | 2040.00 | 77.12 |

Ready result delivery-interval P50/P95/P99 ms: 10.96/12.40/13.53.
Separate latency-pass duration: 1.33 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3143542333485432, 'inference': 8.504489533379456, 'postprocess': 0.8536688583186939}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.364281 | 87.96 | 1364.19 | 2003.44 | 64.67 | 120.57 | 12.89 | 40.52 | 2040.00 | 75.82 |
| 2 | 1.357412 | 88.40 | 1357.33 | 2004.98 | 64.67 | 122.26 | 14.48 | 40.00 | 2040.00 | 76.00 |
| 3 | 1.363589 | 88.00 | 1363.50 | 2005.50 | 64.67 | 120.19 | 13.86 | 40.83 | 2040.00 | 76.82 |
| 4 | 1.353697 | 88.65 | 1353.61 | 2009.98 | 64.67 | 122.39 | 14.15 | 40.60 | 2040.00 | 77.00 |
| 5 | 1.367550 | 87.75 | 1367.42 | 2005.92 | 64.67 | 120.57 | 14.11 | 40.74 | 2040.00 | 77.86 |
| 6 | 1.361189 | 88.16 | 1361.10 | 2005.78 | 64.67 | 122.22 | 14.11 | 40.41 | 2040.00 | 77.85 |
| 7 | 1.349914 | 88.89 | 1349.83 | 2008.02 | 64.67 | 120.91 | 14.07 | 41.04 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.37 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### native-stream

3.89% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.373212 | 87.39 | 1373.13 | 2003.50 | 64.67 | 118.87 | 12.89 | 39.71 | 2040.00 | 75.89 |
| 2 | 1.373370 | 87.38 | 1373.30 | 1896.64 | 64.67 | 119.10 | 12.86 | 39.66 | 2040.00 | 75.93 |
| 3 | 1.396396 | 85.94 | 1396.33 | 2005.50 | 64.67 | 118.62 | 12.82 | 40.00 | 2040.00 | 76.82 |
| 4 | 1.374020 | 87.34 | 1373.95 | 1902.52 | 64.67 | 119.35 | 12.82 | 40.28 | 2040.00 | 77.00 |
| 5 | 1.388892 | 86.40 | 1388.82 | 2005.88 | 64.67 | 119.09 | 12.57 | 40.48 | 2040.00 | 77.86 |
| 6 | 1.376533 | 87.18 | 1376.46 | 1897.47 | 64.67 | 118.80 | 12.86 | 40.42 | 2040.00 | 77.82 |
| 7 | 1.413395 | 84.90 | 1413.22 | 2008.00 | 64.67 | 118.92 | 11.83 | 40.35 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.37 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-non-stream

5.45% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.396424 | 85.93 | 1396.33 | 1895.59 | 56.12 | 118.99 | 12.86 | 39.95 | 2040.00 | 75.93 |
| 2 | 1.393737 | 86.10 | 1393.63 | 1896.91 | 56.12 | 118.43 | 14.71 | 39.89 | 2040.00 | 75.93 |
| 3 | 1.398414 | 85.81 | 1398.31 | 1897.49 | 56.12 | 118.62 | 12.29 | 39.70 | 2040.00 | 76.86 |
| 4 | 1.400164 | 85.70 | 1400.07 | 1897.62 | 56.12 | 117.93 | 12.86 | 40.19 | 2040.00 | 76.96 |
| 5 | 1.416042 | 84.74 | 1415.94 | 1897.77 | 56.12 | 118.75 | 12.79 | 40.34 | 2040.00 | 77.79 |
| 6 | 1.395135 | 86.01 | 1395.05 | 1897.83 | 56.12 | 118.06 | 12.14 | 40.60 | 2040.00 | 77.86 |
| 7 | 1.429631 | 83.94 | 1429.53 | 1899.72 | 56.12 | 118.06 | 12.24 | 40.03 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.40 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-stream

5.54% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.414572 | 84.83 | 35.24 | 1896.48 | 56.12 | 118.62 | 12.24 | 39.39 | 2040.00 | 75.86 |
| 2 | 1.427268 | 84.08 | 34.81 | 1896.74 | 56.12 | 117.90 | 12.38 | 39.40 | 2040.00 | 75.93 |
| 3 | 1.393875 | 86.09 | 34.68 | 1904.33 | 56.12 | 118.36 | 13.14 | 40.23 | 2040.00 | 76.86 |
| 4 | 1.403889 | 85.48 | 34.27 | 1904.51 | 56.12 | 117.86 | 13.29 | 39.88 | 2040.00 | 76.86 |
| 5 | 1.409376 | 85.14 | 35.60 | 1897.63 | 56.12 | 118.53 | 13.36 | 40.07 | 2040.00 | 77.86 |
| 6 | 1.408009 | 85.23 | 34.12 | 1897.76 | 56.12 | 117.81 | 12.75 | 40.00 | 2040.00 | 77.82 |
| 7 | 1.417014 | 84.69 | 36.34 | 1899.39 | 56.12 | 118.32 | 12.31 | 39.83 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.80/12.92/14.98.
Separate latency-pass duration: 1.42 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.

## Detail — yolo26, video, batch 1

Status: **complete**. Load 0.12 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.70 ± 0.91 | 1450.87 | 1425.25/1425.25/1425.25 | 1941.13 | 62.83/90.00 | 120/61 | PASS |
| raw-stream | 83.61 ± 0.35 | 26.46 | 1446.67/1446.67/1446.67 | 1943.52 | 62.83/90.00 | 120/61 | PASS |
| native-non-stream | 80.59 ± 1.38 | 1488.91 | 1489.06/1489.06/1489.06 | 1943.44 | 62.83/90.00 | 120/61 | PASS |
| native-stream | 79.53 ± 0.75 | 1508.71 | 1492.89/1492.89/1492.89 | 1941.13 | 62.80/90.00 | 120/61 | PASS |
| managed-non-stream | 78.82 ± 0.86 | 1522.38 | 1540.19/1540.19/1540.19 | 1834.45 | 55.34/90.00 | 120/61 | PASS |
| managed-stream | 78.92 ± 0.72 | 36.13 | 1559.18/1559.18/1559.18 | 1840.04 | 55.34/90.00 | 120/61 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.450953 | 82.70 | 1450.87 | 1932.86 | 62.83 | 121.37 | 12.93 | 39.31 | 2040.00 | 75.93 |
| 2 | 1.433879 | 83.69 | 1433.80 | 1937.05 | 62.83 | 119.73 | 12.93 | 40.21 | 2040.00 | 76.83 |
| 3 | 1.483219 | 80.91 | 1483.14 | 1936.82 | 62.83 | 118.69 | 12.63 | 39.78 | 2040.00 | 76.83 |
| 4 | 1.438368 | 83.43 | 1438.29 | 1931.73 | 62.83 | 120.09 | 13.03 | 39.92 | 2040.00 | 76.93 |
| 5 | 1.442291 | 83.20 | 1442.20 | 1932.00 | 62.83 | 119.18 | 13.00 | 39.94 | 2040.00 | 76.93 |
| 6 | 1.452575 | 82.61 | 1452.50 | 1941.13 | 62.83 | 120.25 | 13.83 | 39.92 | 2040.00 | 76.86 |
| 7 | 1.452452 | 82.62 | 1452.36 | 1941.12 | 62.83 | 119.83 | 13.10 | 39.80 | 2040.00 | 76.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.43 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3659080500095417, 'inference': 9.650747699978032, 'postprocess': 0.8648955166942566}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.55% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.445370 | 83.02 | 26.85 | 1934.52 | 62.83 | 119.71 | 12.93 | 39.48 | 2040.00 | 76.00 |
| 2 | 1.429992 | 83.92 | 26.50 | 1943.52 | 62.83 | 119.57 | 12.93 | 39.71 | 2040.00 | 76.83 |
| 3 | 1.437444 | 83.48 | 26.46 | 1936.70 | 62.83 | 119.74 | 13.48 | 40.05 | 2040.00 | 76.86 |
| 4 | 1.435162 | 83.61 | 26.25 | 1939.51 | 62.83 | 119.29 | 13.07 | 39.98 | 2040.00 | 76.86 |
| 5 | 1.432499 | 83.77 | 26.33 | 1932.03 | 62.83 | 119.58 | 14.10 | 40.08 | 2040.00 | 77.00 |
| 6 | 1.432598 | 83.76 | 25.53 | 1941.17 | 62.83 | 119.70 | 13.10 | 39.93 | 2040.00 | 76.86 |
| 7 | 1.444473 | 83.08 | 27.00 | 1941.13 | 62.83 | 119.27 | 13.21 | 39.95 | 2040.00 | 76.83 |

Ready result delivery-interval P50/P95/P99 ms: 12.00/12.76/13.09.
Separate latency-pass duration: 1.45 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3848552417130122, 'inference': 9.586948375014495, 'postprocess': 0.8694882583692257}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.62% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.488992 | 80.59 | 1488.91 | 1935.12 | 62.83 | 118.95 | 13.00 | 39.70 | 2040.00 | 76.00 |
| 2 | 1.509942 | 79.47 | 1509.85 | 1943.44 | 62.83 | 120.42 | 12.87 | 39.64 | 2040.00 | 76.40 |
| 3 | 1.544733 | 77.68 | 1544.64 | 1936.72 | 62.83 | 120.87 | 13.06 | 39.99 | 2040.00 | 76.90 |
| 4 | 1.491917 | 80.43 | 1491.80 | 1939.51 | 62.83 | 120.53 | 13.13 | 39.95 | 2040.00 | 76.97 |
| 5 | 1.475807 | 81.31 | 1475.72 | 1932.03 | 62.83 | 118.96 | 14.33 | 40.03 | 2040.00 | 77.00 |
| 6 | 1.473942 | 81.41 | 1473.81 | 1940.11 | 62.83 | 121.22 | 13.93 | 40.07 | 2040.00 | 76.87 |
| 7 | 1.471526 | 81.55 | 1471.40 | 1941.17 | 62.83 | 119.14 | 13.37 | 40.08 | 2040.00 | 76.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.49 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### native-stream

4.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.493129 | 80.37 | 1493.06 | 1935.28 | 62.80 | 117.45 | 12.73 | 39.63 | 2040.00 | 76.00 |
| 2 | 1.527309 | 78.57 | 1527.24 | 1834.57 | 62.80 | 116.67 | 13.06 | 39.80 | 2040.00 | 76.00 |
| 3 | 1.533318 | 78.26 | 1533.24 | 1931.36 | 62.80 | 117.41 | 12.23 | 39.56 | 2040.00 | 76.84 |
| 4 | 1.503435 | 79.82 | 1503.36 | 1830.76 | 62.80 | 117.12 | 13.40 | 39.93 | 2040.00 | 76.93 |
| 5 | 1.508779 | 79.53 | 1508.71 | 1932.01 | 62.80 | 117.70 | 13.27 | 39.85 | 2040.00 | 77.00 |
| 6 | 1.501247 | 79.93 | 1501.17 | 1830.49 | 62.80 | 117.53 | 13.40 | 39.50 | 2040.00 | 76.93 |
| 7 | 1.510578 | 79.44 | 1510.49 | 1941.13 | 62.80 | 117.82 | 12.80 | 39.55 | 2040.00 | 76.63 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.49 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-non-stream

4.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.506263 | 79.67 | 1506.16 | 1826.16 | 55.34 | 116.67 | 12.73 | 39.68 | 2040.00 | 76.00 |
| 2 | 1.527191 | 78.58 | 1527.10 | 1834.45 | 55.34 | 117.41 | 12.84 | 39.46 | 2040.00 | 76.00 |
| 3 | 1.520569 | 78.92 | 1520.48 | 1823.70 | 55.34 | 117.01 | 12.32 | 39.65 | 2040.00 | 76.87 |
| 4 | 1.533458 | 78.25 | 1533.34 | 1830.73 | 55.34 | 117.24 | 12.90 | 39.58 | 2040.00 | 76.94 |
| 5 | 1.504279 | 79.77 | 1504.18 | 1823.97 | 55.34 | 117.07 | 12.73 | 39.76 | 2040.00 | 77.00 |
| 6 | 1.553608 | 77.24 | 1553.52 | 1830.86 | 55.34 | 116.98 | 12.10 | 39.45 | 2040.00 | 76.94 |
| 7 | 1.522485 | 78.82 | 1522.38 | 1833.03 | 55.34 | 116.92 | 12.55 | 39.72 | 2040.00 | 76.55 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.54 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-stream

4.80% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.520541 | 78.92 | 36.09 | 1833.96 | 55.34 | 117.07 | 12.84 | 39.29 | 2040.00 | 76.00 |
| 2 | 1.536787 | 78.08 | 35.83 | 1834.21 | 55.34 | 116.86 | 12.42 | 39.26 | 2040.00 | 76.00 |
| 3 | 1.529256 | 78.47 | 39.17 | 1830.55 | 55.34 | 116.59 | 12.68 | 39.41 | 2040.00 | 76.87 |
| 4 | 1.532944 | 78.28 | 35.98 | 1830.72 | 55.34 | 117.01 | 12.52 | 39.40 | 2040.00 | 76.87 |
| 5 | 1.518605 | 79.02 | 36.30 | 1830.75 | 55.34 | 117.44 | 12.68 | 39.63 | 2040.00 | 77.00 |
| 6 | 1.517802 | 79.06 | 36.74 | 1830.85 | 55.34 | 116.65 | 13.00 | 39.57 | 2040.00 | 76.94 |
| 7 | 1.495359 | 80.25 | 36.13 | 1840.04 | 55.34 | 117.24 | 13.67 | 39.54 | 2040.00 | 76.80 |

Ready result delivery-interval P50/P95/P99 ms: 12.74/15.87/16.47.
Separate latency-pass duration: 1.56 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.

## Detail — yolo-world, video, batch 1

Status: **complete**. Load 5.80 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.10 ± 0.95 | 1461.49 | 1442.91/1442.91/1442.91 | 2191.18 | 115.88/134.00 | 120/61 | PASS |
| raw-stream | 82.22 ± 1.23 | 26.26 | 1440.70/1440.70/1440.70 | 2191.18 | 115.88/134.00 | 120/61 | PASS |
| native-non-stream | 80.60 ± 0.74 | 1488.80 | 1483.94/1483.94/1483.94 | 2191.19 | 115.88/134.00 | 120/61 | PASS |
| native-stream | 77.83 ± 0.53 | 1541.84 | 1524.37/1524.37/1524.37 | 2191.18 | 115.88/134.00 | 120/61 | PASS |
| managed-non-stream | 78.64 ± 0.95 | 1525.87 | 1517.00/1517.00/1517.00 | 2083.07 | 107.97/134.00 | 120/61 | PASS |
| managed-stream | 78.49 ± 0.99 | 35.53 | 1559.52/1559.52/1559.52 | 2090.27 | 107.97/134.00 | 120/61 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.465981 | 81.86 | 1465.89 | 2181.85 | 115.88 | 120.68 | 19.00 | 46.61 | 2040.00 | 77.80 |
| 2 | 1.481482 | 81.00 | 1481.40 | 2190.27 | 115.88 | 120.04 | 20.23 | 46.67 | 2040.00 | 77.93 |
| 3 | 1.464594 | 81.93 | 1464.51 | 2190.29 | 115.88 | 119.19 | 19.93 | 46.48 | 2040.00 | 77.87 |
| 4 | 1.446926 | 82.93 | 1446.84 | 2190.72 | 115.88 | 119.44 | 20.62 | 46.38 | 2040.00 | 75.93 |
| 5 | 1.431496 | 83.83 | 1431.41 | 2190.76 | 115.88 | 119.87 | 20.31 | 46.18 | 2040.00 | 75.86 |
| 6 | 1.444065 | 83.10 | 1443.98 | 2191.18 | 115.88 | 119.54 | 20.28 | 45.99 | 2040.00 | 75.86 |
| 7 | 1.461578 | 82.10 | 1461.49 | 2191.18 | 115.88 | 119.13 | 19.59 | 45.83 | 2040.00 | 75.86 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.44 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3657911750139344, 'inference': 9.750612616691493, 'postprocess': 0.8889672666403688}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.445637 | 83.01 | 26.39 | 2181.94 | 115.88 | 119.28 | 19.34 | 47.38 | 2040.00 | 77.86 |
| 2 | 1.496632 | 80.18 | 26.24 | 2190.24 | 115.88 | 118.73 | 18.27 | 46.65 | 2040.00 | 78.20 |
| 3 | 1.459512 | 82.22 | 26.09 | 2190.25 | 115.88 | 119.08 | 20.03 | 47.36 | 2040.00 | 77.86 |
| 4 | 1.427646 | 84.05 | 26.11 | 2190.76 | 115.88 | 119.37 | 20.31 | 46.06 | 2040.00 | 75.93 |
| 5 | 1.475508 | 81.33 | 26.26 | 2190.76 | 115.88 | 119.37 | 20.70 | 46.25 | 2040.00 | 75.87 |
| 6 | 1.459842 | 82.20 | 26.34 | 2190.88 | 115.88 | 119.61 | 19.62 | 46.35 | 2040.00 | 75.83 |
| 7 | 1.454486 | 82.50 | 26.94 | 2191.18 | 115.88 | 119.27 | 19.93 | 46.35 | 2040.00 | 75.86 |

Ready result delivery-interval P50/P95/P99 ms: 12.05/12.66/12.92.
Separate latency-pass duration: 1.44 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3369957583790892, 'inference': 9.612335558313134, 'postprocess': 0.8897608333503134}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

1.52% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.473685 | 81.43 | 1473.61 | 2182.54 | 115.88 | 118.55 | 19.27 | 47.04 | 2040.00 | 77.87 |
| 2 | 1.504052 | 79.78 | 1503.97 | 2190.16 | 115.88 | 120.64 | 19.23 | 46.42 | 2040.00 | 78.87 |
| 3 | 1.482719 | 80.93 | 1482.63 | 2190.27 | 115.88 | 118.88 | 20.00 | 47.00 | 2040.00 | 77.20 |
| 4 | 1.488886 | 80.60 | 1488.80 | 2190.72 | 115.88 | 120.19 | 20.27 | 45.90 | 2040.00 | 76.10 |
| 5 | 1.506839 | 79.64 | 1506.75 | 2190.76 | 115.88 | 118.91 | 20.33 | 46.71 | 2040.00 | 75.87 |
| 6 | 1.507500 | 79.60 | 1507.41 | 2190.92 | 115.88 | 120.64 | 18.47 | 46.39 | 2040.00 | 75.80 |
| 7 | 1.481668 | 80.99 | 1481.57 | 2191.19 | 115.88 | 118.95 | 20.00 | 46.87 | 2040.00 | 75.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.48 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### native-stream

5.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.540256 | 77.91 | 1540.18 | 2182.61 | 115.88 | 117.39 | 18.29 | 46.64 | 2040.00 | 78.06 |
| 2 | 1.541906 | 77.83 | 1541.84 | 2081.14 | 115.88 | 117.05 | 19.77 | 46.17 | 2040.00 | 78.87 |
| 3 | 1.544235 | 77.71 | 1544.17 | 2190.29 | 115.88 | 116.74 | 18.23 | 45.79 | 2040.00 | 76.94 |
| 4 | 1.525335 | 78.67 | 1525.26 | 2082.78 | 115.88 | 116.48 | 18.39 | 46.21 | 2040.00 | 76.06 |
| 5 | 1.539835 | 77.93 | 1539.76 | 2190.73 | 115.88 | 117.19 | 18.42 | 45.62 | 2040.00 | 75.84 |
| 6 | 1.555927 | 77.12 | 1555.84 | 2083.02 | 115.88 | 117.67 | 19.23 | 46.00 | 2040.00 | 75.81 |
| 7 | 1.555456 | 77.15 | 1555.39 | 2191.18 | 115.88 | 117.62 | 18.45 | 45.53 | 2040.00 | 75.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.52 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-non-stream

5.03% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.542205 | 77.81 | 1542.09 | 2073.39 | 107.97 | 117.01 | 19.03 | 46.60 | 2040.00 | 78.48 |
| 2 | 1.547401 | 77.55 | 1547.29 | 2081.09 | 107.97 | 116.96 | 18.03 | 46.12 | 2040.00 | 78.87 |
| 3 | 1.525962 | 78.64 | 1525.87 | 2082.54 | 107.97 | 116.78 | 19.52 | 45.49 | 2040.00 | 76.94 |
| 4 | 1.519649 | 78.97 | 1519.56 | 2082.79 | 107.97 | 116.94 | 18.65 | 45.75 | 2040.00 | 76.58 |
| 5 | 1.501545 | 79.92 | 1501.45 | 2082.91 | 107.97 | 116.95 | 19.37 | 45.35 | 2040.00 | 75.83 |
| 6 | 1.522797 | 78.80 | 1522.70 | 2083.02 | 107.97 | 116.97 | 19.19 | 45.47 | 2040.00 | 75.87 |
| 7 | 1.555177 | 77.16 | 1555.08 | 2083.07 | 107.97 | 117.91 | 18.71 | 45.19 | 2040.00 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.52 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-stream

5.40% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.524976 | 78.69 | 36.03 | 2080.65 | 107.97 | 116.53 | 18.52 | 46.25 | 2040.00 | 78.87 |
| 2 | 1.560141 | 76.92 | 35.53 | 2080.90 | 107.97 | 116.67 | 18.58 | 46.24 | 2040.00 | 78.87 |
| 3 | 1.558724 | 76.99 | 35.59 | 2082.61 | 107.97 | 117.08 | 19.00 | 46.04 | 2040.00 | 76.87 |
| 4 | 1.545903 | 77.62 | 34.98 | 2082.73 | 107.97 | 117.39 | 18.87 | 45.19 | 2040.00 | 76.77 |
| 5 | 1.508756 | 79.54 | 35.19 | 2082.92 | 107.97 | 116.59 | 19.53 | 45.37 | 2040.00 | 75.87 |
| 6 | 1.528836 | 78.49 | 34.40 | 2083.02 | 107.97 | 116.68 | 18.39 | 45.12 | 2040.00 | 75.84 |
| 7 | 1.523388 | 78.77 | 36.16 | 2090.27 | 107.97 | 116.75 | 18.39 | 45.44 | 2040.00 | 75.90 |

Ready result delivery-interval P50/P95/P99 ms: 12.77/14.23/16.43.
Separate latency-pass duration: 1.56 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
