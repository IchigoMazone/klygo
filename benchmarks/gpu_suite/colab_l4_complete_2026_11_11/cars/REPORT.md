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
| yolo | 116.05 | 115.64 | 111.91 | 109.24 | 107.48 | 107.34 |
| yolo11 | 93.52 | 93.49 | 91.04 | 89.36 | 87.56 | 88.35 |
| yolo26 | 82.95 | 82.77 | 81.16 | 78.84 | 77.62 | 77.72 |
| yolo-world | 82.41 | 82.52 | 80.92 | 78.39 | 78.86 | 78.54 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 116.05 ± 1.77 | 1033.96 | 1037.62/1037.62/1037.62 | 1933.85 | 34.27/62.00 | 120/50 | PASS |
| raw-stream | 115.64 ± 2.02 | 22.07 | 1028.20/1028.20/1028.20 | 1933.82 | 34.27/62.00 | 120/50 | PASS |
| native-non-stream | 111.91 ± 1.48 | 1072.20 | 1071.34/1071.34/1071.34 | 1933.95 | 34.27/62.00 | 120/50 | PASS |
| native-stream | 109.24 ± 1.79 | 1098.44 | 1086.30/1086.30/1086.30 | 1933.79 | 34.25/62.00 | 120/50 | PASS |
| managed-non-stream | 107.48 ± 1.22 | 1116.37 | 1122.22/1122.22/1122.22 | 1825.00 | 24.66/62.00 | 120/50 | PASS |
| managed-stream | 107.34 ± 1.22 | 29.88 | 1130.44/1130.44/1130.44 | 1825.03 | 24.66/62.00 | 120/50 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.034045 | 116.05 | 1033.96 | 1923.89 | 34.27 | 129.57 | 15.10 | 40.79 | 2040.00 | 74.90 |
| 2 | 1.036602 | 115.76 | 1036.52 | 1932.96 | 34.27 | 128.06 | 13.76 | 41.37 | 2040.00 | 76.52 |
| 3 | 1.062541 | 112.94 | 1062.44 | 1933.00 | 34.27 | 127.70 | 15.00 | 41.19 | 2040.00 | 76.82 |
| 4 | 1.042667 | 115.09 | 1042.59 | 1933.61 | 34.27 | 127.21 | 14.57 | 42.25 | 2040.00 | 77.67 |
| 5 | 1.025273 | 117.04 | 1025.17 | 1933.85 | 34.27 | 127.89 | 14.90 | 42.22 | 2040.00 | 77.76 |
| 6 | 1.014039 | 118.34 | 1013.96 | 1933.82 | 34.27 | 127.47 | 14.00 | 41.97 | 2040.00 | 77.81 |
| 7 | 1.021621 | 117.46 | 1021.53 | 1933.82 | 34.27 | 128.00 | 13.95 | 41.97 | 2040.00 | 77.81 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.04 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2935838750308903, 'inference': 6.376110500036702, 'postprocess': 0.7530205666474407}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.55% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.044145 | 114.93 | 21.88 | 1925.42 | 34.27 | 127.57 | 15.10 | 41.05 | 2040.00 | 75.00 |
| 2 | 1.041718 | 115.19 | 22.14 | 1932.95 | 34.27 | 127.59 | 13.76 | 41.20 | 2040.00 | 76.19 |
| 3 | 1.075113 | 111.62 | 22.07 | 1933.14 | 34.27 | 128.27 | 15.00 | 41.24 | 2040.00 | 76.82 |
| 4 | 1.037675 | 115.64 | 22.73 | 1933.61 | 34.27 | 128.56 | 14.57 | 41.47 | 2040.00 | 77.19 |
| 5 | 1.034222 | 116.03 | 22.24 | 1933.61 | 34.27 | 127.60 | 14.14 | 42.51 | 2040.00 | 77.71 |
| 6 | 1.019581 | 117.70 | 21.35 | 1933.82 | 34.27 | 128.22 | 13.76 | 41.78 | 2040.00 | 77.81 |
| 7 | 1.021510 | 117.47 | 22.00 | 1933.82 | 34.27 | 127.81 | 14.14 | 42.14 | 2040.00 | 77.90 |

Ready result delivery-interval P50/P95/P99 ms: 8.14/10.06/11.06.
Separate latency-pass duration: 1.03 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.323344416634124, 'inference': 6.399469766635472, 'postprocess': 0.7812750416708999}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

4.56% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.106202 | 108.48 | 1106.12 | 1925.47 | 34.27 | 126.31 | 14.91 | 41.12 | 2040.00 | 75.27 |
| 2 | 1.091738 | 109.92 | 1091.65 | 1932.86 | 34.27 | 128.06 | 13.73 | 41.13 | 2040.00 | 76.00 |
| 3 | 1.072286 | 111.91 | 1072.20 | 1932.98 | 34.27 | 126.12 | 15.68 | 42.36 | 2040.00 | 76.82 |
| 4 | 1.071129 | 112.03 | 1071.03 | 1933.62 | 34.27 | 129.53 | 15.00 | 42.21 | 2040.00 | 77.00 |
| 5 | 1.068431 | 112.31 | 1068.34 | 1933.84 | 34.27 | 127.02 | 13.91 | 42.24 | 2040.00 | 77.82 |
| 6 | 1.085648 | 110.53 | 1085.55 | 1933.80 | 34.27 | 128.68 | 13.91 | 41.80 | 2040.00 | 77.82 |
| 7 | 1.068191 | 112.34 | 1068.10 | 1933.95 | 34.27 | 126.73 | 15.18 | 42.66 | 2040.00 | 77.82 |

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

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### native-stream

5.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.129311 | 106.26 | 1129.22 | 1925.60 | 34.25 | 125.45 | 12.78 | 40.80 | 2040.00 | 75.70 |
| 2 | 1.098508 | 109.24 | 1098.44 | 1823.96 | 34.25 | 123.32 | 14.91 | 41.46 | 2040.00 | 75.91 |
| 3 | 1.082930 | 110.81 | 1082.85 | 1933.03 | 34.25 | 124.27 | 14.09 | 41.54 | 2040.00 | 76.86 |
| 4 | 1.082446 | 110.86 | 1082.38 | 1824.74 | 34.25 | 123.97 | 13.45 | 41.28 | 2040.00 | 77.00 |
| 5 | 1.111179 | 107.99 | 1111.10 | 1933.62 | 34.25 | 123.52 | 13.57 | 41.46 | 2040.00 | 77.78 |
| 6 | 1.102202 | 108.87 | 1102.13 | 1824.98 | 34.25 | 124.32 | 13.64 | 41.72 | 2040.00 | 77.86 |
| 7 | 1.079974 | 111.11 | 1079.90 | 1933.79 | 34.25 | 123.73 | 13.82 | 41.58 | 2040.00 | 77.73 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.09 s; not used for throughput.

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

7.59% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.138466 | 105.41 | 1138.37 | 1816.19 | 24.66 | 123.34 | 13.00 | 40.74 | 2040.00 | 75.83 |
| 2 | 1.115275 | 107.60 | 1115.17 | 1823.87 | 24.66 | 123.76 | 14.13 | 40.54 | 2040.00 | 75.91 |
| 3 | 1.116462 | 107.48 | 1116.37 | 1824.40 | 24.66 | 123.37 | 13.09 | 41.05 | 2040.00 | 76.91 |
| 4 | 1.119677 | 107.17 | 1119.56 | 1824.75 | 24.66 | 123.59 | 13.43 | 41.44 | 2040.00 | 76.91 |
| 5 | 1.112360 | 107.88 | 1112.26 | 1824.82 | 24.66 | 123.53 | 14.70 | 41.38 | 2040.00 | 77.83 |
| 6 | 1.130641 | 106.13 | 1130.55 | 1824.98 | 24.66 | 123.79 | 14.52 | 41.15 | 2040.00 | 77.83 |
| 7 | 1.098997 | 109.19 | 1098.90 | 1825.00 | 24.66 | 123.35 | 13.50 | 41.32 | 2040.00 | 77.55 |

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


### managed-stream

9.40% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.131285 | 106.07 | 32.08 | 1823.45 | 24.66 | 123.05 | 13.96 | 40.61 | 2040.00 | 75.83 |
| 2 | 1.140455 | 105.22 | 30.56 | 1823.74 | 24.66 | 123.10 | 13.00 | 40.31 | 2040.00 | 75.91 |
| 3 | 1.115359 | 107.59 | 29.55 | 1824.48 | 24.66 | 123.59 | 13.09 | 41.36 | 2040.00 | 76.91 |
| 4 | 1.109261 | 108.18 | 29.66 | 1824.58 | 24.66 | 123.12 | 14.09 | 40.69 | 2040.00 | 76.82 |
| 5 | 1.102780 | 108.82 | 29.27 | 1824.88 | 24.66 | 123.73 | 13.45 | 41.25 | 2040.00 | 77.82 |
| 6 | 1.117948 | 107.34 | 29.88 | 1824.97 | 24.66 | 122.63 | 12.91 | 41.13 | 2040.00 | 77.83 |
| 7 | 1.122083 | 106.94 | 31.28 | 1825.03 | 24.66 | 123.22 | 13.22 | 41.24 | 2040.00 | 77.09 |

Ready result delivery-interval P50/P95/P99 ms: 9.04/10.52/11.22.
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

## Detail — yolo11, video, batch 1

Status: **complete**. Load 0.12 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 93.52 ± 0.76 | 1283.11 | 1280.37/1280.37/1280.37 | 2011.84 | 64.67/90.00 | 120/36 | PASS |
| raw-stream | 93.49 ± 0.44 | 24.19 | 1295.96/1295.96/1295.96 | 2012.01 | 64.67/90.00 | 120/36 | PASS |
| native-non-stream | 91.04 ± 1.13 | 1318.04 | 1327.94/1327.94/1327.94 | 2011.82 | 64.67/90.00 | 120/36 | PASS |
| native-stream | 89.36 ± 1.16 | 1342.76 | 1340.48/1340.48/1340.48 | 2011.87 | 64.67/90.00 | 120/36 | PASS |
| managed-non-stream | 87.56 ± 1.66 | 1370.41 | 1364.04/1364.04/1364.04 | 1902.35 | 56.12/90.00 | 120/36 | PASS |
| managed-stream | 88.35 ± 0.90 | 32.61 | 1392.96/1392.96/1392.96 | 1903.66 | 56.12/90.00 | 120/36 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.284168 | 93.45 | 1284.08 | 2002.66 | 64.67 | 124.92 | 13.54 | 39.98 | 2040.00 | 75.92 |
| 2 | 1.278182 | 93.88 | 1278.10 | 2011.83 | 64.67 | 123.09 | 14.46 | 41.28 | 2040.00 | 76.88 |
| 3 | 1.271988 | 94.34 | 1271.91 | 2011.84 | 64.67 | 122.98 | 13.69 | 40.68 | 2040.00 | 76.81 |
| 4 | 1.279757 | 93.77 | 1279.68 | 2009.71 | 64.67 | 122.60 | 14.15 | 40.87 | 2040.00 | 76.27 |
| 5 | 1.301805 | 92.18 | 1301.72 | 2009.66 | 64.67 | 123.02 | 13.38 | 40.56 | 2040.00 | 76.54 |
| 6 | 1.296125 | 92.58 | 1296.03 | 2009.85 | 64.67 | 122.38 | 14.31 | 40.82 | 2040.00 | 77.00 |
| 7 | 1.283189 | 93.52 | 1283.11 | 2009.95 | 64.67 | 122.84 | 14.27 | 40.84 | 2040.00 | 76.96 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.28 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3852129749996795, 'inference': 8.400112291663694, 'postprocess': 0.7168268583579144}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.16% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.288768 | 93.11 | 24.31 | 2003.24 | 64.67 | 122.57 | 13.54 | 40.33 | 2040.00 | 76.00 |
| 2 | 1.280199 | 93.74 | 24.08 | 2012.01 | 64.67 | 122.84 | 13.69 | 40.58 | 2040.00 | 76.85 |
| 3 | 1.288663 | 93.12 | 24.19 | 2011.80 | 64.67 | 122.44 | 13.54 | 40.47 | 2040.00 | 76.85 |
| 4 | 1.283576 | 93.49 | 24.66 | 2009.66 | 64.67 | 122.53 | 13.54 | 40.38 | 2040.00 | 76.00 |
| 5 | 1.290448 | 92.99 | 23.89 | 2009.66 | 64.67 | 122.67 | 13.69 | 40.90 | 2040.00 | 76.65 |
| 6 | 1.281546 | 93.64 | 23.74 | 2009.85 | 64.67 | 122.88 | 13.54 | 40.61 | 2040.00 | 76.92 |
| 7 | 1.273524 | 94.23 | 28.17 | 2009.88 | 64.67 | 124.98 | 13.54 | 40.67 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 10.53/12.38/13.76.
Separate latency-pass duration: 1.30 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3306660250085163, 'inference': 8.458312533358972, 'postprocess': 0.7502087749874894}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.98% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.326363 | 90.47 | 1326.25 | 2003.28 | 64.67 | 122.39 | 13.70 | 40.96 | 2040.00 | 75.93 |
| 2 | 1.318131 | 91.04 | 1318.04 | 2011.35 | 64.67 | 123.27 | 13.74 | 40.97 | 2040.00 | 76.81 |
| 3 | 1.292398 | 92.85 | 1292.30 | 2011.82 | 64.67 | 121.71 | 13.58 | 41.14 | 2040.00 | 76.88 |
| 4 | 1.322178 | 90.76 | 1322.08 | 2009.69 | 64.67 | 123.61 | 13.78 | 40.65 | 2040.00 | 76.00 |
| 5 | 1.293729 | 92.76 | 1293.64 | 2009.68 | 64.67 | 122.08 | 14.31 | 40.60 | 2040.00 | 76.85 |
| 6 | 1.334759 | 89.90 | 1334.65 | 2009.86 | 64.67 | 124.02 | 14.04 | 40.83 | 2040.00 | 76.93 |
| 7 | 1.309615 | 91.63 | 1309.53 | 2004.70 | 64.67 | 122.18 | 13.54 | 40.70 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.33 s; not used for throughput.

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

5.10% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.333098 | 90.02 | 1333.02 | 2003.41 | 64.67 | 120.00 | 13.37 | 40.27 | 2040.00 | 76.00 |
| 2 | 1.343399 | 89.33 | 1343.33 | 1902.46 | 64.67 | 120.16 | 13.19 | 40.28 | 2040.00 | 76.74 |
| 3 | 1.337899 | 89.69 | 1337.83 | 2011.87 | 64.67 | 119.84 | 13.15 | 40.69 | 2040.00 | 76.81 |
| 4 | 1.342833 | 89.36 | 1342.76 | 1901.67 | 64.67 | 120.21 | 13.52 | 40.48 | 2040.00 | 76.00 |
| 5 | 1.339842 | 89.56 | 1339.76 | 2009.67 | 64.67 | 120.02 | 14.41 | 40.51 | 2040.00 | 76.85 |
| 6 | 1.383025 | 86.77 | 1382.94 | 1901.86 | 64.67 | 119.91 | 12.71 | 40.61 | 2040.00 | 76.93 |
| 7 | 1.364241 | 87.96 | 1364.13 | 2004.67 | 64.67 | 120.33 | 13.43 | 40.47 | 2040.00 | 76.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.34 s; not used for throughput.

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

7.22% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.335862 | 89.83 | 1335.74 | 1894.05 | 56.12 | 119.54 | 13.00 | 40.11 | 2040.00 | 76.00 |
| 2 | 1.370512 | 87.56 | 1370.41 | 1902.35 | 56.12 | 120.03 | 13.14 | 40.51 | 2040.00 | 76.29 |
| 3 | 1.384699 | 86.66 | 1384.61 | 1901.47 | 56.12 | 120.49 | 13.39 | 40.74 | 2040.00 | 76.75 |
| 4 | 1.352067 | 88.75 | 1351.97 | 1901.67 | 56.12 | 119.14 | 13.67 | 40.00 | 2040.00 | 76.00 |
| 5 | 1.365168 | 87.90 | 1365.07 | 1901.70 | 56.12 | 119.68 | 13.57 | 40.41 | 2040.00 | 76.86 |
| 6 | 1.391226 | 86.25 | 1391.13 | 1901.86 | 56.12 | 119.07 | 13.86 | 40.22 | 2040.00 | 76.93 |
| 7 | 1.414880 | 84.81 | 1414.78 | 1896.49 | 56.12 | 120.08 | 12.55 | 40.36 | 2040.00 | 76.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.36 s; not used for throughput.

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

6.08% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.362631 | 88.06 | 33.47 | 1900.86 | 56.12 | 119.67 | 13.43 | 40.17 | 2040.00 | 76.00 |
| 2 | 1.336354 | 89.80 | 31.49 | 1902.16 | 56.12 | 119.83 | 12.85 | 40.02 | 2040.00 | 76.04 |
| 3 | 1.355836 | 88.51 | 32.29 | 1901.57 | 56.12 | 119.62 | 13.37 | 39.98 | 2040.00 | 76.15 |
| 4 | 1.346247 | 89.14 | 32.61 | 1901.67 | 56.12 | 119.19 | 13.33 | 40.06 | 2040.00 | 76.00 |
| 5 | 1.358239 | 88.35 | 32.74 | 1901.76 | 56.12 | 119.30 | 13.19 | 40.42 | 2040.00 | 76.89 |
| 6 | 1.374982 | 87.27 | 32.23 | 1901.85 | 56.12 | 119.64 | 12.75 | 40.06 | 2040.00 | 76.93 |
| 7 | 1.372451 | 87.43 | 34.07 | 1903.66 | 56.12 | 119.18 | 13.57 | 40.17 | 2040.00 | 76.86 |

Ready result delivery-interval P50/P95/P99 ms: 11.15/12.64/13.34.
Separate latency-pass duration: 1.39 s; not used for throughput.

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

Status: **complete**. Load 0.13 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.95 ± 1.02 | 1446.58 | 1437.41/1437.41/1437.41 | 1946.11 | 62.82/90.00 | 120/35 | PASS |
| raw-stream | 82.77 ± 1.19 | 25.89 | 1428.48/1428.48/1428.48 | 1945.88 | 62.82/90.00 | 120/35 | PASS |
| native-non-stream | 81.16 ± 0.64 | 1478.51 | 1475.91/1475.91/1475.91 | 1945.89 | 62.82/90.00 | 120/35 | PASS |
| native-stream | 78.84 ± 0.72 | 1521.99 | 1508.71/1508.71/1508.71 | 1945.88 | 62.80/90.00 | 120/35 | PASS |
| managed-non-stream | 77.62 ± 1.24 | 1545.81 | 1560.56/1560.56/1560.56 | 1837.10 | 55.34/90.00 | 120/35 | PASS |
| managed-stream | 77.72 ± 0.86 | 33.98 | 1608.71/1608.71/1608.71 | 1844.27 | 55.34/90.00 | 120/35 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.448268 | 82.86 | 1448.17 | 1933.83 | 62.82 | 122.20 | 13.59 | 39.82 | 2040.00 | 76.31 |
| 2 | 1.488234 | 80.63 | 1488.13 | 1943.66 | 62.82 | 120.56 | 12.67 | 40.20 | 2040.00 | 76.93 |
| 3 | 1.443960 | 83.10 | 1443.87 | 1943.67 | 62.82 | 120.40 | 12.79 | 39.89 | 2040.00 | 76.90 |
| 4 | 1.446661 | 82.95 | 1446.58 | 1944.43 | 62.82 | 120.22 | 13.48 | 40.24 | 2040.00 | 77.00 |
| 5 | 1.459113 | 82.24 | 1459.03 | 1944.43 | 62.82 | 120.58 | 13.07 | 40.01 | 2040.00 | 76.93 |
| 6 | 1.434541 | 83.65 | 1434.46 | 1946.11 | 62.82 | 120.13 | 13.52 | 40.16 | 2040.00 | 76.00 |
| 7 | 1.436988 | 83.51 | 1436.90 | 1945.88 | 62.82 | 120.43 | 13.14 | 39.82 | 2040.00 | 76.00 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.360871983350383, 'inference': 9.780492858324882, 'postprocess': 0.7273246083741469}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.23% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.427254 | 84.08 | 25.60 | 1935.41 | 62.82 | 120.79 | 13.69 | 40.21 | 2040.00 | 76.79 |
| 2 | 1.491692 | 80.45 | 25.98 | 1943.66 | 62.82 | 120.49 | 12.60 | 39.77 | 2040.00 | 76.90 |
| 3 | 1.434764 | 83.64 | 27.04 | 1943.71 | 62.82 | 120.23 | 12.93 | 39.95 | 2040.00 | 76.97 |
| 4 | 1.456513 | 82.39 | 25.39 | 1944.43 | 62.82 | 120.72 | 12.93 | 40.07 | 2040.00 | 77.00 |
| 5 | 1.449740 | 82.77 | 26.95 | 1944.43 | 62.82 | 121.32 | 12.62 | 39.94 | 2040.00 | 76.93 |
| 6 | 1.458198 | 82.29 | 25.32 | 1945.16 | 62.82 | 119.93 | 12.86 | 39.70 | 2040.00 | 76.00 |
| 7 | 1.440905 | 83.28 | 25.89 | 1945.88 | 62.82 | 120.35 | 13.48 | 40.13 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.59/12.74/14.19.
Separate latency-pass duration: 1.43 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3314988333225604, 'inference': 9.638121608357627, 'postprocess': 0.7277476499893965}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.504946 | 79.74 | 1504.86 | 1935.39 | 62.82 | 119.70 | 13.80 | 39.78 | 2040.00 | 76.80 |
| 2 | 1.490499 | 80.51 | 1490.41 | 1943.57 | 62.82 | 121.11 | 13.53 | 39.91 | 2040.00 | 76.93 |
| 3 | 1.477967 | 81.19 | 1477.87 | 1943.70 | 62.82 | 119.99 | 13.00 | 40.07 | 2040.00 | 76.97 |
| 4 | 1.470568 | 81.60 | 1470.48 | 1944.46 | 62.82 | 121.38 | 13.80 | 40.11 | 2040.00 | 77.00 |
| 5 | 1.479217 | 81.12 | 1479.13 | 1944.44 | 62.82 | 120.26 | 12.90 | 40.00 | 2040.00 | 76.93 |
| 6 | 1.473297 | 81.45 | 1473.21 | 1945.18 | 62.82 | 122.25 | 13.70 | 39.72 | 2040.00 | 76.00 |
| 7 | 1.478611 | 81.16 | 1478.51 | 1945.89 | 62.82 | 119.41 | 13.73 | 39.96 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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

5.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.527707 | 78.55 | 1527.63 | 1935.54 | 62.80 | 118.01 | 12.32 | 39.66 | 2040.00 | 76.74 |
| 2 | 1.495573 | 80.24 | 1495.50 | 1834.68 | 62.80 | 118.02 | 12.23 | 39.78 | 2040.00 | 76.87 |
| 3 | 1.517978 | 79.05 | 1517.91 | 1943.75 | 62.80 | 118.27 | 12.35 | 39.93 | 2040.00 | 77.00 |
| 4 | 1.517147 | 79.10 | 1517.08 | 1835.32 | 62.80 | 118.02 | 12.71 | 40.02 | 2040.00 | 77.00 |
| 5 | 1.522061 | 78.84 | 1521.99 | 1944.43 | 62.80 | 117.69 | 12.10 | 39.93 | 2040.00 | 76.90 |
| 6 | 1.539071 | 77.97 | 1539.00 | 1835.79 | 62.80 | 118.80 | 12.13 | 39.47 | 2040.00 | 76.00 |
| 7 | 1.530340 | 78.41 | 1530.27 | 1945.88 | 62.80 | 118.04 | 12.19 | 39.56 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.51 s; not used for throughput.

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

7.12% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.551386 | 77.35 | 1551.28 | 1826.20 | 55.34 | 117.72 | 11.97 | 39.83 | 2040.00 | 76.81 |
| 2 | 1.532773 | 78.29 | 1532.67 | 1834.57 | 55.34 | 117.96 | 12.58 | 39.71 | 2040.00 | 76.87 |
| 3 | 1.563804 | 76.74 | 1563.71 | 1835.07 | 55.34 | 117.62 | 12.74 | 39.52 | 2040.00 | 77.00 |
| 4 | 1.493015 | 80.37 | 1492.91 | 1835.29 | 55.34 | 117.70 | 12.53 | 39.79 | 2040.00 | 77.00 |
| 5 | 1.518048 | 79.05 | 1517.94 | 1835.64 | 55.34 | 117.42 | 12.97 | 39.55 | 2040.00 | 76.87 |
| 6 | 1.545930 | 77.62 | 1545.81 | 1835.79 | 55.34 | 117.31 | 12.90 | 39.65 | 2040.00 | 76.00 |
| 7 | 1.550899 | 77.37 | 1550.80 | 1837.10 | 55.34 | 117.99 | 12.06 | 39.44 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
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


### managed-stream

6.87% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.543472 | 77.75 | 35.79 | 1833.95 | 55.34 | 117.54 | 12.39 | 39.58 | 2040.00 | 76.74 |
| 2 | 1.543978 | 77.72 | 33.91 | 1834.42 | 55.34 | 117.87 | 12.19 | 39.75 | 2040.00 | 76.81 |
| 3 | 1.562568 | 76.80 | 34.17 | 1835.22 | 55.34 | 117.46 | 12.45 | 39.41 | 2040.00 | 77.00 |
| 4 | 1.553370 | 77.25 | 33.64 | 1835.29 | 55.34 | 118.21 | 12.26 | 39.48 | 2040.00 | 77.00 |
| 5 | 1.529847 | 78.44 | 33.98 | 1835.72 | 55.34 | 117.27 | 12.71 | 39.83 | 2040.00 | 76.58 |
| 6 | 1.533095 | 78.27 | 33.21 | 1835.79 | 55.34 | 117.84 | 12.32 | 39.43 | 2040.00 | 76.00 |
| 7 | 1.579109 | 75.99 | 34.85 | 1844.27 | 55.34 | 117.55 | 12.16 | 39.38 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.68/15.97/18.00.
Separate latency-pass duration: 1.61 s; not used for throughput.

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

Status: **complete**. Load 5.83 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.41 ± 0.79 | 1455.97 | 1463.35/1463.35/1463.35 | 2190.01 | 115.88/132.00 | 120/41 | PASS |
| raw-stream | 82.52 ± 0.98 | 26.06 | 1465.43/1465.43/1465.43 | 2190.39 | 115.88/132.00 | 120/41 | PASS |
| native-non-stream | 80.92 ± 0.95 | 1482.95 | 1469.85/1469.85/1469.85 | 2190.30 | 115.88/132.00 | 120/41 | PASS |
| native-stream | 78.39 ± 0.85 | 1530.71 | 1530.09/1530.09/1530.09 | 2185.35 | 115.88/132.00 | 120/41 | PASS |
| managed-non-stream | 78.86 ± 0.77 | 1521.51 | 1514.09/1514.09/1514.09 | 2081.45 | 107.97/132.00 | 120/41 | PASS |
| managed-stream | 78.54 ± 1.10 | 33.49 | 1545.49/1545.49/1545.49 | 2082.82 | 107.97/132.00 | 120/41 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.488140 | 80.64 | 1488.05 | 2180.20 | 115.88 | 121.47 | 18.10 | 46.43 | 2040.00 | 77.80 |
| 2 | 1.460079 | 82.19 | 1460.00 | 2189.99 | 115.88 | 119.95 | 19.97 | 47.46 | 2040.00 | 78.90 |
| 3 | 1.442503 | 83.19 | 1442.42 | 2190.01 | 115.88 | 119.97 | 19.14 | 47.57 | 2040.00 | 78.86 |
| 4 | 1.452003 | 82.64 | 1451.92 | 2184.00 | 115.88 | 121.97 | 19.93 | 46.44 | 2040.00 | 76.00 |
| 5 | 1.453337 | 82.57 | 1453.25 | 2184.01 | 115.88 | 121.91 | 19.34 | 45.89 | 2040.00 | 75.97 |
| 6 | 1.456583 | 82.38 | 1456.48 | 2185.35 | 115.88 | 120.41 | 18.86 | 46.95 | 2040.00 | 75.83 |
| 7 | 1.456050 | 82.41 | 1455.97 | 2185.35 | 115.88 | 121.49 | 18.66 | 45.67 | 2040.00 | 75.79 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.46 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3410253082838608, 'inference': 10.084494608334657, 'postprocess': 0.773125233316326}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.454192 | 82.52 | 26.06 | 2180.78 | 115.88 | 120.18 | 18.55 | 46.94 | 2040.00 | 77.86 |
| 2 | 1.488013 | 80.64 | 25.87 | 2190.39 | 115.88 | 120.31 | 19.13 | 47.03 | 2040.00 | 78.90 |
| 3 | 1.453877 | 82.54 | 25.90 | 2190.01 | 115.88 | 119.80 | 19.34 | 47.83 | 2040.00 | 78.41 |
| 4 | 1.454998 | 82.47 | 29.76 | 2184.00 | 115.88 | 121.93 | 18.55 | 46.24 | 2040.00 | 76.52 |
| 5 | 1.457173 | 82.35 | 29.43 | 2184.01 | 115.88 | 121.63 | 19.86 | 46.09 | 2040.00 | 75.97 |
| 6 | 1.432377 | 83.78 | 25.43 | 2185.38 | 115.88 | 120.40 | 18.72 | 46.20 | 2040.00 | 75.79 |
| 7 | 1.440744 | 83.29 | 26.42 | 2185.35 | 115.88 | 119.75 | 19.79 | 46.02 | 2040.00 | 75.86 |

Ready result delivery-interval P50/P95/P99 ms: 11.81/13.75/15.70.
Separate latency-pass duration: 1.47 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3445309500184521, 'inference': 9.800966824999099, 'postprocess': 0.7744233416360657}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

1.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.512706 | 79.33 | 1512.61 | 2181.40 | 115.88 | 119.95 | 18.60 | 47.46 | 2040.00 | 77.87 |
| 2 | 1.475543 | 81.33 | 1475.46 | 2190.30 | 115.88 | 121.76 | 19.67 | 47.10 | 2040.00 | 78.93 |
| 3 | 1.503779 | 79.80 | 1503.68 | 2190.03 | 115.88 | 121.21 | 19.33 | 46.94 | 2040.00 | 77.93 |
| 4 | 1.476214 | 81.29 | 1476.10 | 2184.00 | 115.88 | 121.70 | 19.20 | 46.91 | 2040.00 | 76.87 |
| 5 | 1.483034 | 80.92 | 1482.95 | 2184.02 | 115.88 | 121.26 | 19.67 | 46.06 | 2040.00 | 75.90 |
| 6 | 1.514649 | 79.23 | 1514.55 | 2184.66 | 115.88 | 121.05 | 18.55 | 46.02 | 2040.00 | 75.71 |
| 7 | 1.478652 | 81.16 | 1478.56 | 2185.36 | 115.88 | 119.81 | 19.60 | 46.53 | 2040.00 | 75.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.47 s; not used for throughput.

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

4.01% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.534649 | 78.19 | 1534.58 | 2181.42 | 115.88 | 117.80 | 18.77 | 46.42 | 2040.00 | 78.00 |
| 2 | 1.503521 | 79.81 | 1503.45 | 2081.50 | 115.88 | 117.80 | 18.30 | 46.85 | 2040.00 | 78.90 |
| 3 | 1.545949 | 77.62 | 1545.88 | 2183.68 | 115.88 | 117.58 | 17.61 | 46.19 | 2040.00 | 77.90 |
| 4 | 1.510117 | 79.46 | 1510.05 | 2075.19 | 115.88 | 117.75 | 17.93 | 45.77 | 2040.00 | 76.90 |
| 5 | 1.511647 | 79.38 | 1511.58 | 2184.01 | 115.88 | 117.39 | 18.70 | 45.91 | 2040.00 | 75.90 |
| 6 | 1.539028 | 77.97 | 1538.96 | 2075.38 | 115.88 | 117.60 | 18.77 | 45.30 | 2040.00 | 75.32 |
| 7 | 1.530786 | 78.39 | 1530.71 | 2185.35 | 115.88 | 117.81 | 18.06 | 45.74 | 2040.00 | 75.81 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.53 s; not used for throughput.

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

4.50% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.504948 | 79.74 | 1504.83 | 2073.21 | 107.97 | 117.48 | 17.93 | 46.60 | 2040.00 | 78.53 |
| 2 | 1.532787 | 78.29 | 1532.69 | 2081.45 | 107.97 | 117.58 | 18.29 | 47.03 | 2040.00 | 78.87 |
| 3 | 1.524388 | 78.72 | 1524.30 | 2075.03 | 107.97 | 117.70 | 19.10 | 46.09 | 2040.00 | 77.77 |
| 4 | 1.502358 | 79.87 | 1502.27 | 2075.20 | 107.97 | 117.41 | 18.60 | 46.52 | 2040.00 | 76.93 |
| 5 | 1.497406 | 80.14 | 1497.31 | 2075.24 | 107.97 | 118.06 | 18.53 | 45.65 | 2040.00 | 75.87 |
| 6 | 1.532585 | 78.30 | 1532.49 | 2075.38 | 107.97 | 117.70 | 17.94 | 46.07 | 2040.00 | 75.65 |
| 7 | 1.521596 | 78.86 | 1521.51 | 2076.59 | 107.97 | 117.67 | 18.10 | 45.25 | 2040.00 | 75.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.06.
Separate latency-pass duration: 1.51 s; not used for throughput.

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

5.23% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.503134 | 79.83 | 35.46 | 2080.93 | 107.97 | 117.21 | 18.20 | 46.96 | 2040.00 | 78.73 |
| 2 | 1.522610 | 78.81 | 33.08 | 2081.33 | 107.97 | 117.41 | 19.10 | 46.63 | 2040.00 | 78.87 |
| 3 | 1.550464 | 77.40 | 32.94 | 2075.09 | 107.97 | 117.61 | 18.26 | 46.27 | 2040.00 | 76.94 |
| 4 | 1.527929 | 78.54 | 33.49 | 2075.19 | 107.97 | 117.69 | 17.90 | 45.67 | 2040.00 | 76.94 |
| 5 | 1.563184 | 76.77 | 35.31 | 2075.27 | 107.97 | 117.84 | 17.69 | 45.68 | 2040.00 | 75.84 |
| 6 | 1.511594 | 79.39 | 33.33 | 2075.36 | 107.97 | 117.51 | 17.93 | 44.91 | 2040.00 | 75.80 |
| 7 | 1.542594 | 77.79 | 35.61 | 2082.82 | 107.97 | 117.84 | 17.97 | 46.25 | 2040.00 | 75.84 |

Ready result delivery-interval P50/P95/P99 ms: 12.38/14.31/15.19.
Separate latency-pass duration: 1.55 s; not used for throughput.

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
