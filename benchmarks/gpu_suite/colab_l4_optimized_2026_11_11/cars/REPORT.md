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

Tested commit(s): `7729cb719cc493fc45fc2c885c848b909d770ecd`

## FPS matrix — video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 111.34 | 110.81 | 107.74 | 104.82 | 104.05 | 106.52 |
| yolo11 | 92.01 | 92.26 | 89.67 | 87.60 | 88.78 | 88.31 |
| yolo26 | 82.88 | 83.37 | 80.16 | 80.07 | 80.22 | 79.58 |
| yolo-world | 82.32 | 82.98 | 81.36 | 79.91 | 78.94 | 79.74 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 111.34 ± 2.45 | 1077.68 | 1125.45/1125.45/1125.45 | 1926.45 | 34.27/62.00 | 120/50 | PASS |
| raw-stream | 110.81 ± 3.89 | 23.76 | 1067.00/1067.00/1067.00 | 1926.48 | 34.27/62.00 | 120/50 | PASS |
| native-non-stream | 107.74 ± 3.30 | 1113.65 | 1111.54/1111.54/1111.54 | 1930.76 | 34.27/62.00 | 120/50 | PASS |
| native-stream | 104.82 ± 3.12 | 23.58 | 1137.89/1137.89/1137.89 | 1926.45 | 34.25/62.00 | 120/50 | PASS |
| managed-non-stream | 104.05 ± 2.31 | 1153.21 | 1113.89/1113.89/1113.89 | 1824.30 | 24.66/62.00 | 120/50 | PASS |
| managed-stream | 106.52 ± 2.79 | 30.13 | 1123.51/1123.51/1123.51 | 1824.29 | 24.66/62.00 | 120/50 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.108739 | 108.23 | 1108.63 | 1920.86 | 34.27 | 129.10 | 13.27 | 40.84 | 2040.00 | 75.00 |
| 2 | 1.077771 | 111.34 | 1077.68 | 1926.26 | 34.27 | 126.53 | 13.50 | 41.04 | 2040.00 | 75.50 |
| 3 | 1.073006 | 111.84 | 1072.92 | 1925.39 | 34.27 | 127.29 | 13.64 | 40.92 | 2040.00 | 75.64 |
| 4 | 1.106098 | 108.49 | 1105.99 | 1926.15 | 34.27 | 127.42 | 12.82 | 40.92 | 2040.00 | 75.73 |
| 5 | 1.060719 | 113.13 | 1060.64 | 1926.19 | 34.27 | 127.12 | 14.18 | 40.85 | 2040.00 | 75.73 |
| 6 | 1.093557 | 109.73 | 1093.46 | 1926.45 | 34.27 | 126.13 | 13.55 | 41.33 | 2040.00 | 75.82 |
| 7 | 1.044457 | 114.89 | 1044.37 | 1926.45 | 34.27 | 127.29 | 14.05 | 41.25 | 2040.00 | 75.86 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4912475835293055, 'inference': 6.719205549946612, 'postprocess': 0.81815910836364}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.11% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.070185 | 112.13 | 22.07 | 1924.33 | 34.27 | 127.61 | 13.41 | 41.17 | 2040.00 | 75.09 |
| 2 | 1.090537 | 110.04 | 24.04 | 1925.30 | 34.27 | 126.96 | 13.27 | 40.61 | 2040.00 | 75.36 |
| 3 | 1.082886 | 110.81 | 23.22 | 1925.36 | 34.27 | 127.06 | 14.55 | 40.97 | 2040.00 | 75.64 |
| 4 | 1.092073 | 109.88 | 23.76 | 1926.15 | 34.27 | 126.66 | 13.09 | 40.75 | 2040.00 | 75.59 |
| 5 | 1.061899 | 113.01 | 24.32 | 1926.20 | 34.27 | 127.55 | 14.82 | 41.11 | 2040.00 | 75.82 |
| 6 | 1.057321 | 113.49 | 22.22 | 1926.45 | 34.27 | 127.27 | 13.57 | 41.11 | 2040.00 | 75.81 |
| 7 | 1.176765 | 101.97 | 24.16 | 1926.48 | 34.27 | 127.12 | 13.92 | 41.04 | 2040.00 | 75.92 |

Ready result delivery-interval P50/P95/P99 ms: 8.71/9.41/10.63.
Separate latency-pass duration: 1.07 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3658942749467922, 'inference': 6.536354233276143, 'postprocess': 0.8150894916525431}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.66% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.149299 | 104.41 | 1149.21 | 1924.31 | 34.27 | 125.46 | 13.74 | 41.50 | 2040.00 | 75.17 |
| 2 | 1.133111 | 105.90 | 1133.00 | 1930.76 | 34.27 | 128.04 | 14.43 | 40.78 | 2040.00 | 75.48 |
| 3 | 1.095099 | 109.58 | 1095.01 | 1925.37 | 34.27 | 126.19 | 14.86 | 41.69 | 2040.00 | 75.77 |
| 4 | 1.113757 | 107.74 | 1113.65 | 1926.18 | 34.27 | 128.98 | 14.91 | 40.85 | 2040.00 | 75.83 |
| 5 | 1.102409 | 108.85 | 1102.32 | 1926.20 | 34.27 | 126.45 | 15.00 | 41.16 | 2040.00 | 75.82 |
| 6 | 1.112166 | 107.90 | 1112.06 | 1926.48 | 34.27 | 128.45 | 14.26 | 41.52 | 2040.00 | 75.87 |
| 7 | 1.200121 | 99.99 | 1200.02 | 1926.46 | 34.27 | 126.71 | 12.79 | 40.58 | 2040.00 | 75.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.04.
Separate latency-pass duration: 1.11 s; not used for throughput.

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

5.30% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.156090 | 103.80 | 24.76 | 1924.48 | 34.25 | 123.29 | 12.91 | 40.51 | 2040.00 | 75.00 |
| 2 | 1.144870 | 104.82 | 21.16 | 1823.39 | 34.25 | 123.33 | 12.61 | 40.32 | 2040.00 | 75.00 |
| 3 | 1.125819 | 106.59 | 24.79 | 1925.41 | 34.25 | 123.82 | 13.22 | 41.14 | 2040.00 | 75.65 |
| 4 | 1.132195 | 105.99 | 21.61 | 1823.77 | 34.25 | 123.64 | 13.17 | 40.53 | 2040.00 | 75.48 |
| 5 | 1.141568 | 105.12 | 23.58 | 1926.16 | 34.25 | 123.33 | 13.17 | 40.90 | 2040.00 | 75.78 |
| 6 | 1.151508 | 104.21 | 21.91 | 1823.93 | 34.25 | 123.11 | 13.17 | 41.12 | 2040.00 | 75.78 |
| 7 | 1.233997 | 97.24 | 25.05 | 1926.45 | 34.25 | 123.25 | 12.20 | 40.47 | 2040.00 | 75.84 |

Ready result delivery-interval P50/P95/P99 ms: 9.20/10.30/11.88.
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


### managed-non-stream

7.01% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.126411 | 106.53 | 1126.32 | 1815.09 | 24.66 | 123.37 | 13.43 | 41.01 | 2040.00 | 75.17 |
| 2 | 1.153316 | 104.05 | 1153.21 | 1823.31 | 24.66 | 123.33 | 13.96 | 40.41 | 2040.00 | 75.04 |
| 3 | 1.134762 | 105.75 | 1134.67 | 1816.76 | 24.66 | 123.23 | 13.30 | 41.32 | 2040.00 | 75.52 |
| 4 | 1.165205 | 102.99 | 1165.11 | 1824.14 | 24.66 | 123.24 | 13.67 | 40.27 | 2040.00 | 75.42 |
| 5 | 1.147297 | 104.59 | 1147.20 | 1817.38 | 24.66 | 124.39 | 12.78 | 41.06 | 2040.00 | 75.78 |
| 6 | 1.190210 | 100.82 | 1190.11 | 1824.30 | 24.66 | 123.06 | 12.83 | 40.46 | 2040.00 | 75.75 |
| 7 | 1.193995 | 100.50 | 1193.89 | 1817.66 | 24.66 | 123.45 | 12.25 | 40.38 | 2040.00 | 75.92 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.11 s; not used for throughput.

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

4.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.126542 | 106.52 | 31.68 | 1822.83 | 24.66 | 122.66 | 14.00 | 40.72 | 2040.00 | 75.17 |
| 2 | 1.125090 | 106.66 | 29.54 | 1823.12 | 24.66 | 123.21 | 12.96 | 40.53 | 2040.00 | 75.17 |
| 3 | 1.144572 | 104.84 | 30.69 | 1822.87 | 24.66 | 123.36 | 13.48 | 40.75 | 2040.00 | 75.61 |
| 4 | 1.116689 | 107.46 | 30.08 | 1824.10 | 24.66 | 124.34 | 13.04 | 40.56 | 2040.00 | 75.52 |
| 5 | 1.134403 | 105.78 | 30.13 | 1824.16 | 24.66 | 123.19 | 13.74 | 40.57 | 2040.00 | 75.74 |
| 6 | 1.125129 | 106.65 | 28.52 | 1824.29 | 24.66 | 123.30 | 13.13 | 40.53 | 2040.00 | 75.74 |
| 7 | 1.208975 | 99.26 | 30.32 | 1823.36 | 24.66 | 123.22 | 12.54 | 40.51 | 2040.00 | 75.88 |

Ready result delivery-interval P50/P95/P99 ms: 9.13/10.16/10.46.
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

## Detail — yolo11, video, batch 1

Status: **complete**. Load 0.12 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 92.01 ± 2.26 | 1304.14 | 1288.65/1288.65/1288.65 | 2012.24 | 64.67/90.00 | 120/36 | PASS |
| raw-stream | 92.26 ± 0.80 | 24.34 | 1318.75/1318.75/1318.75 | 2012.21 | 64.67/90.00 | 120/36 | PASS |
| native-non-stream | 89.67 ± 1.22 | 1338.08 | 1360.77/1360.77/1360.77 | 2012.25 | 64.67/90.00 | 120/36 | PASS |
| native-stream | 87.60 ± 1.45 | 25.44 | 1353.11/1353.11/1353.11 | 2012.27 | 64.67/90.00 | 120/36 | PASS |
| managed-non-stream | 88.78 ± 1.31 | 1351.60 | 1362.90/1362.90/1362.90 | 1903.78 | 56.12/90.00 | 120/36 | PASS |
| managed-stream | 88.31 ± 1.05 | 31.78 | 1384.72/1384.72/1384.72 | 1905.21 | 56.12/90.00 | 120/36 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.303328 | 92.07 | 1303.24 | 2002.32 | 64.67 | 124.23 | 12.69 | 40.42 | 2040.00 | 76.85 |
| 2 | 1.322922 | 90.71 | 1322.84 | 2012.24 | 64.67 | 122.25 | 12.81 | 40.85 | 2040.00 | 77.78 |
| 3 | 1.304221 | 92.01 | 1304.14 | 2012.21 | 64.67 | 123.12 | 13.46 | 40.69 | 2040.00 | 77.77 |
| 4 | 1.293755 | 92.75 | 1293.68 | 2010.09 | 64.67 | 122.31 | 14.27 | 40.57 | 2040.00 | 76.92 |
| 5 | 1.300411 | 92.28 | 1300.30 | 2010.09 | 64.67 | 122.46 | 13.77 | 40.91 | 2040.00 | 76.85 |
| 6 | 1.392912 | 86.15 | 1392.83 | 2010.24 | 64.67 | 121.49 | 12.71 | 40.49 | 2040.00 | 76.00 |
| 7 | 1.315873 | 91.19 | 1315.78 | 2010.29 | 64.67 | 122.36 | 13.41 | 40.23 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.29 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3523104167385707, 'inference': 8.593969433271317, 'postprocess': 0.7152292417382947}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.312113 | 91.46 | 24.24 | 2003.89 | 64.67 | 122.78 | 13.22 | 41.00 | 2040.00 | 76.93 |
| 2 | 1.300605 | 92.26 | 24.07 | 2012.20 | 64.67 | 122.73 | 13.19 | 40.77 | 2040.00 | 77.69 |
| 3 | 1.285843 | 93.32 | 24.34 | 2012.21 | 64.67 | 122.60 | 13.77 | 41.13 | 2040.00 | 77.77 |
| 4 | 1.293388 | 92.78 | 23.85 | 2010.09 | 64.67 | 122.49 | 13.00 | 40.50 | 2040.00 | 76.85 |
| 5 | 1.304197 | 92.01 | 24.72 | 2010.09 | 64.67 | 122.54 | 12.69 | 40.44 | 2040.00 | 76.85 |
| 6 | 1.319606 | 90.94 | 24.73 | 2010.28 | 64.67 | 122.10 | 13.22 | 40.36 | 2040.00 | 76.00 |
| 7 | 1.299301 | 92.36 | 28.35 | 2010.27 | 64.67 | 124.13 | 12.88 | 40.35 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 10.79/11.96/13.31.
Separate latency-pass duration: 1.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4175726500222179, 'inference': 8.57917510002153, 'postprocess': 0.7369392917553341}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

1.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.318544 | 91.01 | 1318.46 | 2003.91 | 64.67 | 121.89 | 14.59 | 40.63 | 2040.00 | 76.93 |
| 2 | 1.338169 | 89.67 | 1338.08 | 2012.10 | 64.67 | 123.30 | 14.15 | 40.40 | 2040.00 | 77.78 |
| 3 | 1.316986 | 91.12 | 1316.89 | 2012.25 | 64.67 | 122.25 | 13.44 | 40.84 | 2040.00 | 77.78 |
| 4 | 1.345361 | 89.20 | 1345.25 | 2010.10 | 64.67 | 122.93 | 13.07 | 40.69 | 2040.00 | 76.93 |
| 5 | 1.366244 | 87.83 | 1366.16 | 2010.13 | 64.67 | 121.55 | 13.29 | 40.83 | 2040.00 | 76.86 |
| 6 | 1.333756 | 89.97 | 1333.66 | 2010.25 | 64.67 | 123.59 | 13.30 | 40.57 | 2040.00 | 76.00 |
| 7 | 1.355756 | 88.51 | 1355.68 | 2006.82 | 64.67 | 122.89 | 13.22 | 40.56 | 2040.00 | 76.00 |

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


### native-stream

5.52% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.369880 | 87.60 | 25.85 | 2004.02 | 64.67 | 120.24 | 12.75 | 40.68 | 2040.00 | 76.93 |
| 2 | 1.397438 | 85.87 | 23.07 | 1903.89 | 64.67 | 119.91 | 13.00 | 40.76 | 2040.00 | 77.07 |
| 3 | 1.393062 | 86.14 | 25.81 | 2012.27 | 64.67 | 119.29 | 12.14 | 40.52 | 2040.00 | 77.00 |
| 4 | 1.365293 | 87.89 | 23.75 | 1902.11 | 64.67 | 119.61 | 12.71 | 40.59 | 2040.00 | 76.93 |
| 5 | 1.372132 | 87.46 | 25.44 | 2010.10 | 64.67 | 119.66 | 12.86 | 40.25 | 2040.00 | 76.79 |
| 6 | 1.331050 | 90.15 | 23.64 | 1902.26 | 64.67 | 119.71 | 13.44 | 40.67 | 2040.00 | 76.00 |
| 7 | 1.355431 | 88.53 | 29.01 | 2006.79 | 64.67 | 119.26 | 12.78 | 40.14 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 10.86/11.98/12.43.
Separate latency-pass duration: 1.35 s; not used for throughput.

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

3.73% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.393381 | 86.12 | 1393.26 | 1895.38 | 56.12 | 119.24 | 13.36 | 40.19 | 2040.00 | 77.00 |
| 2 | 1.372248 | 87.45 | 1372.14 | 1903.78 | 56.12 | 119.52 | 12.43 | 40.57 | 2040.00 | 77.00 |
| 3 | 1.350973 | 88.82 | 1350.88 | 1901.88 | 56.12 | 119.69 | 12.52 | 40.23 | 2040.00 | 77.00 |
| 4 | 1.388643 | 86.42 | 1388.55 | 1902.11 | 56.12 | 120.12 | 13.14 | 40.22 | 2040.00 | 77.00 |
| 5 | 1.351702 | 88.78 | 1351.60 | 1902.13 | 56.12 | 119.65 | 13.81 | 40.01 | 2040.00 | 76.78 |
| 6 | 1.347020 | 89.09 | 1346.92 | 1902.27 | 56.12 | 119.20 | 12.78 | 40.23 | 2040.00 | 76.00 |
| 7 | 1.345077 | 89.21 | 1344.97 | 1898.02 | 56.12 | 119.30 | 13.33 | 40.40 | 2040.00 | 76.00 |

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

3.64% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.350706 | 88.84 | 32.08 | 1902.16 | 56.12 | 119.09 | 12.63 | 40.28 | 2040.00 | 77.00 |
| 2 | 1.364445 | 87.95 | 31.09 | 1903.61 | 56.12 | 119.52 | 13.29 | 40.27 | 2040.00 | 77.00 |
| 3 | 1.363257 | 88.02 | 31.78 | 1902.00 | 56.12 | 119.82 | 13.21 | 40.17 | 2040.00 | 77.00 |
| 4 | 1.358836 | 88.31 | 31.09 | 1902.09 | 56.12 | 120.41 | 12.52 | 40.11 | 2040.00 | 77.00 |
| 5 | 1.357211 | 88.42 | 32.10 | 1902.18 | 56.12 | 119.22 | 12.48 | 40.03 | 2040.00 | 76.63 |
| 6 | 1.399750 | 85.73 | 31.03 | 1902.25 | 56.12 | 118.34 | 12.75 | 39.90 | 2040.00 | 76.00 |
| 7 | 1.353714 | 88.65 | 33.31 | 1905.21 | 56.12 | 119.19 | 13.41 | 40.10 | 2040.00 | 75.93 |

Ready result delivery-interval P50/P95/P99 ms: 10.98/13.54/14.66.
Separate latency-pass duration: 1.38 s; not used for throughput.

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
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.88 ± 0.64 | 1447.75 | 1444.43/1444.43/1444.43 | 1947.03 | 62.82/90.00 | 120/35 | PASS |
| raw-stream | 83.37 ± 1.13 | 25.56 | 1461.52/1461.52/1461.52 | 1947.07 | 62.82/90.00 | 120/35 | PASS |
| native-non-stream | 80.16 ± 0.84 | 1496.92 | 1475.96/1475.96/1475.96 | 1947.05 | 62.82/90.00 | 120/35 | PASS |
| native-stream | 80.07 ± 0.53 | 27.21 | 1484.82/1484.82/1484.82 | 1947.04 | 62.80/90.00 | 120/35 | PASS |
| managed-non-stream | 80.22 ± 0.73 | 1495.79 | 1486.05/1486.05/1486.05 | 1838.85 | 55.34/90.00 | 120/35 | PASS |
| managed-stream | 79.58 ± 0.72 | 33.42 | 1513.29/1513.29/1513.29 | 1845.10 | 55.34/90.00 | 120/35 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.457842 | 82.31 | 1457.76 | 1933.50 | 62.82 | 122.22 | 12.83 | 40.13 | 2040.00 | 76.28 |
| 2 | 1.437532 | 83.48 | 1437.45 | 1944.61 | 62.82 | 120.01 | 12.66 | 40.44 | 2040.00 | 77.66 |
| 3 | 1.434745 | 83.64 | 1434.66 | 1944.63 | 62.82 | 120.29 | 12.93 | 40.52 | 2040.00 | 77.79 |
| 4 | 1.447834 | 82.88 | 1447.75 | 1945.04 | 62.82 | 119.89 | 13.48 | 40.75 | 2040.00 | 76.93 |
| 5 | 1.463977 | 81.97 | 1463.88 | 1945.07 | 62.82 | 120.49 | 13.00 | 40.29 | 2040.00 | 76.93 |
| 6 | 1.450588 | 82.73 | 1450.51 | 1947.03 | 62.82 | 120.01 | 12.66 | 40.28 | 2040.00 | 76.00 |
| 7 | 1.437700 | 83.47 | 1437.62 | 1947.03 | 62.82 | 120.45 | 12.79 | 39.95 | 2040.00 | 76.00 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.378179133340988, 'inference': 9.841395074909087, 'postprocess': 0.7245844583091335}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.43% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.438463 | 83.42 | 25.44 | 1935.05 | 62.82 | 119.98 | 13.72 | 40.31 | 2040.00 | 76.76 |
| 2 | 1.410192 | 85.09 | 25.92 | 1944.62 | 62.82 | 120.11 | 12.93 | 40.52 | 2040.00 | 77.07 |
| 3 | 1.451154 | 82.69 | 25.19 | 1944.64 | 62.82 | 120.47 | 12.66 | 40.48 | 2040.00 | 77.86 |
| 4 | 1.468717 | 81.70 | 25.51 | 1945.41 | 62.82 | 120.14 | 13.00 | 40.23 | 2040.00 | 76.93 |
| 5 | 1.460966 | 82.14 | 25.86 | 1945.04 | 62.82 | 120.21 | 12.79 | 40.19 | 2040.00 | 76.93 |
| 6 | 1.439423 | 83.37 | 25.56 | 1946.45 | 62.82 | 120.41 | 12.79 | 40.04 | 2040.00 | 76.00 |
| 7 | 1.431472 | 83.83 | 26.05 | 1947.07 | 62.82 | 120.39 | 12.93 | 40.02 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.98/13.12/14.20.
Separate latency-pass duration: 1.46 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3347586749508384, 'inference': 9.726797758473063, 'postprocess': 0.7261713083456319}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.20% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.514154 | 79.25 | 1514.05 | 1935.77 | 62.82 | 119.80 | 13.48 | 40.57 | 2040.00 | 76.84 |
| 2 | 1.481260 | 81.01 | 1481.17 | 1943.93 | 62.82 | 121.21 | 12.87 | 40.52 | 2040.00 | 77.00 |
| 3 | 1.516973 | 79.10 | 1516.88 | 1944.68 | 62.82 | 119.84 | 12.68 | 40.33 | 2040.00 | 77.87 |
| 4 | 1.474975 | 81.36 | 1474.89 | 1945.66 | 62.82 | 122.09 | 14.10 | 40.44 | 2040.00 | 77.00 |
| 5 | 1.492827 | 80.38 | 1492.72 | 1945.05 | 62.82 | 119.48 | 12.87 | 40.14 | 2040.00 | 76.87 |
| 6 | 1.497022 | 80.16 | 1496.92 | 1946.29 | 62.82 | 121.35 | 13.53 | 40.40 | 2040.00 | 76.00 |
| 7 | 1.501033 | 79.94 | 1500.94 | 1947.05 | 62.82 | 119.44 | 12.80 | 40.07 | 2040.00 | 76.00 |

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

2.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.489411 | 80.57 | 28.21 | 1935.80 | 62.80 | 118.27 | 13.07 | 39.98 | 2040.00 | 76.87 |
| 2 | 1.511885 | 79.37 | 24.19 | 1836.02 | 62.80 | 117.61 | 12.43 | 40.17 | 2040.00 | 77.00 |
| 3 | 1.513727 | 79.27 | 28.65 | 1944.70 | 62.80 | 118.13 | 12.52 | 40.09 | 2040.00 | 77.77 |
| 4 | 1.490637 | 80.50 | 24.03 | 1836.70 | 62.80 | 117.73 | 13.23 | 40.31 | 2040.00 | 77.00 |
| 5 | 1.498701 | 80.07 | 27.21 | 1945.04 | 62.80 | 118.00 | 12.40 | 39.95 | 2040.00 | 76.67 |
| 6 | 1.493116 | 80.37 | 23.89 | 1836.89 | 62.80 | 118.32 | 13.23 | 39.85 | 2040.00 | 76.00 |
| 7 | 1.505150 | 79.73 | 27.21 | 1947.04 | 62.80 | 118.07 | 12.27 | 39.85 | 2040.00 | 75.93 |

Ready result delivery-interval P50/P95/P99 ms: 12.01/13.18/13.74.
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


### managed-non-stream

4.05% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.496821 | 80.17 | 1496.72 | 1827.58 | 55.34 | 117.54 | 13.57 | 40.34 | 2040.00 | 76.90 |
| 2 | 1.501364 | 79.93 | 1501.26 | 1835.92 | 55.34 | 118.21 | 13.07 | 39.97 | 2040.00 | 76.93 |
| 3 | 1.494764 | 80.28 | 1494.66 | 1836.47 | 55.34 | 117.75 | 13.13 | 40.22 | 2040.00 | 77.80 |
| 4 | 1.477604 | 81.21 | 1477.49 | 1836.70 | 55.34 | 117.26 | 12.90 | 40.45 | 2040.00 | 77.00 |
| 5 | 1.524103 | 78.73 | 1523.88 | 1836.74 | 55.34 | 118.13 | 13.00 | 39.88 | 2040.00 | 76.03 |
| 6 | 1.493743 | 80.34 | 1493.65 | 1836.89 | 55.34 | 117.93 | 13.37 | 39.86 | 2040.00 | 76.00 |
| 7 | 1.495893 | 80.22 | 1495.79 | 1838.85 | 55.34 | 117.98 | 13.23 | 39.70 | 2040.00 | 75.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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


### managed-stream

4.18% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.523452 | 78.77 | 34.07 | 1834.38 | 55.34 | 117.30 | 12.13 | 39.88 | 2040.00 | 76.94 |
| 2 | 1.508976 | 79.52 | 33.18 | 1835.76 | 55.34 | 118.39 | 12.67 | 39.84 | 2040.00 | 76.93 |
| 3 | 1.494685 | 80.28 | 32.81 | 1836.60 | 55.34 | 117.51 | 12.30 | 39.97 | 2040.00 | 77.63 |
| 4 | 1.483105 | 80.91 | 32.68 | 1836.68 | 55.34 | 117.94 | 12.67 | 40.07 | 2040.00 | 77.07 |
| 5 | 1.507948 | 79.58 | 33.42 | 1836.81 | 55.34 | 118.08 | 13.00 | 40.09 | 2040.00 | 76.00 |
| 6 | 1.504801 | 79.74 | 33.74 | 1836.88 | 55.34 | 118.13 | 12.27 | 39.62 | 2040.00 | 76.00 |
| 7 | 1.517556 | 79.07 | 34.00 | 1845.10 | 55.34 | 117.74 | 13.10 | 39.59 | 2040.00 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 12.22/13.49/14.64.
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

## Detail — yolo-world, video, batch 1

Status: **complete**. Load 5.84 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.32 ± 0.80 | 1457.63 | 1457.65/1457.65/1457.65 | 2192.99 | 115.88/134.00 | 120/41 | PASS |
| raw-stream | 82.98 ± 1.46 | 25.73 | 1452.02/1452.02/1452.02 | 2192.96 | 115.88/134.00 | 120/41 | PASS |
| native-non-stream | 81.36 ± 1.01 | 1474.92 | 1477.84/1477.84/1477.84 | 2192.97 | 115.88/134.00 | 120/41 | PASS |
| native-stream | 79.91 ± 0.86 | 26.88 | 1564.88/1564.88/1564.88 | 2192.96 | 115.88/134.00 | 120/41 | PASS |
| managed-non-stream | 78.94 ± 1.18 | 1520.00 | 1467.25/1467.25/1467.25 | 2084.65 | 107.97/134.00 | 120/41 | PASS |
| managed-stream | 79.74 ± 1.05 | 32.43 | 1520.19/1520.19/1520.19 | 2091.82 | 107.97/134.00 | 120/41 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.462465 | 82.05 | 1462.37 | 2181.16 | 115.88 | 121.94 | 19.27 | 46.28 | 2040.00 | 77.07 |
| 2 | 1.436670 | 83.53 | 1436.57 | 2190.39 | 115.88 | 121.16 | 20.10 | 46.56 | 2040.00 | 78.83 |
| 3 | 1.462895 | 82.03 | 1462.79 | 2190.48 | 115.88 | 120.50 | 19.27 | 46.53 | 2040.00 | 78.17 |
| 4 | 1.457729 | 82.32 | 1457.63 | 2190.84 | 115.88 | 120.73 | 19.00 | 45.92 | 2040.00 | 76.00 |
| 5 | 1.435768 | 83.58 | 1435.69 | 2190.84 | 115.88 | 120.15 | 19.07 | 45.94 | 2040.00 | 75.93 |
| 6 | 1.443789 | 83.11 | 1443.71 | 2192.96 | 115.88 | 120.35 | 18.86 | 45.87 | 2040.00 | 75.72 |
| 7 | 1.470718 | 81.59 | 1470.63 | 2192.99 | 115.88 | 120.29 | 18.93 | 45.72 | 2040.00 | 75.67 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4150251998216845, 'inference': 9.799086100110799, 'postprocess': 0.7605581583144764}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.66% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.473477 | 81.44 | 26.20 | 2181.76 | 115.88 | 120.04 | 18.67 | 45.72 | 2040.00 | 77.80 |
| 2 | 1.436009 | 83.56 | 25.73 | 2189.81 | 115.88 | 121.52 | 19.62 | 46.80 | 2040.00 | 78.69 |
| 3 | 1.449770 | 82.77 | 25.60 | 2190.45 | 115.88 | 120.54 | 19.72 | 47.33 | 2040.00 | 78.14 |
| 4 | 1.446192 | 82.98 | 25.49 | 2191.21 | 115.88 | 120.36 | 19.10 | 46.47 | 2040.00 | 76.72 |
| 5 | 1.505373 | 79.71 | 26.53 | 2190.84 | 115.88 | 120.50 | 18.13 | 45.35 | 2040.00 | 75.93 |
| 6 | 1.434288 | 83.67 | 25.96 | 2192.66 | 115.88 | 120.44 | 19.10 | 45.44 | 2040.00 | 75.55 |
| 7 | 1.436660 | 83.53 | 25.50 | 2192.96 | 115.88 | 120.24 | 19.10 | 45.76 | 2040.00 | 75.69 |

Ready result delivery-interval P50/P95/P99 ms: 11.96/12.87/13.42.
Separate latency-pass duration: 1.45 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.379107583276588, 'inference': 9.887851183384555, 'postprocess': 0.8051231166973594}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

1.19% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.476673 | 81.26 | 1476.58 | 2181.74 | 115.88 | 119.60 | 18.93 | 46.14 | 2040.00 | 77.90 |
| 2 | 1.471140 | 81.57 | 1471.06 | 2189.75 | 115.88 | 122.06 | 20.20 | 46.34 | 2040.00 | 78.57 |
| 3 | 1.471511 | 81.55 | 1471.43 | 2190.47 | 115.88 | 119.80 | 19.73 | 47.35 | 2040.00 | 77.93 |
| 4 | 1.475004 | 81.36 | 1474.92 | 2191.22 | 115.88 | 121.84 | 19.90 | 46.84 | 2040.00 | 76.90 |
| 5 | 1.520011 | 78.95 | 1519.90 | 2190.86 | 115.88 | 120.23 | 18.81 | 46.07 | 2040.00 | 75.94 |
| 6 | 1.494859 | 80.28 | 1494.75 | 2191.70 | 115.88 | 122.05 | 19.57 | 45.33 | 2040.00 | 75.63 |
| 7 | 1.467887 | 81.75 | 1467.80 | 2192.97 | 115.88 | 119.87 | 19.00 | 45.58 | 2040.00 | 75.80 |

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

3.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.501644 | 79.91 | 27.49 | 2181.78 | 115.88 | 118.77 | 18.13 | 46.07 | 2040.00 | 77.90 |
| 2 | 1.493685 | 80.34 | 23.83 | 2081.85 | 115.88 | 118.10 | 18.60 | 46.61 | 2040.00 | 78.00 |
| 3 | 1.521673 | 78.86 | 27.12 | 2190.50 | 115.88 | 118.21 | 18.48 | 46.75 | 2040.00 | 77.94 |
| 4 | 1.498778 | 80.07 | 24.28 | 2081.55 | 115.88 | 118.16 | 19.37 | 45.74 | 2040.00 | 76.87 |
| 5 | 1.521542 | 78.87 | 26.88 | 2190.85 | 115.88 | 118.45 | 18.42 | 45.20 | 2040.00 | 75.90 |
| 6 | 1.495481 | 80.24 | 23.62 | 2082.69 | 115.88 | 118.46 | 19.27 | 45.14 | 2040.00 | 75.60 |
| 7 | 1.535690 | 78.14 | 27.56 | 2192.96 | 115.88 | 118.48 | 18.06 | 45.12 | 2040.00 | 75.84 |

Ready result delivery-interval P50/P95/P99 ms: 12.52/15.22/17.08.
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


### managed-non-stream

3.73% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.521106 | 78.89 | 1521.01 | 2073.58 | 107.97 | 118.63 | 18.84 | 45.91 | 2040.00 | 77.94 |
| 2 | 1.490259 | 80.52 | 1490.16 | 2081.78 | 107.97 | 117.88 | 18.50 | 46.30 | 2040.00 | 78.03 |
| 3 | 1.486934 | 80.70 | 1486.84 | 2081.37 | 107.97 | 118.45 | 18.50 | 46.18 | 2040.00 | 77.90 |
| 4 | 1.549873 | 77.43 | 1549.78 | 2081.56 | 107.97 | 117.92 | 17.68 | 45.24 | 2040.00 | 76.94 |
| 5 | 1.523372 | 78.77 | 1523.28 | 2082.55 | 107.97 | 117.87 | 18.13 | 44.78 | 2040.00 | 75.87 |
| 6 | 1.495974 | 80.22 | 1495.86 | 2082.70 | 107.97 | 117.94 | 18.37 | 45.95 | 2040.00 | 75.83 |
| 7 | 1.520092 | 78.94 | 1520.00 | 2084.65 | 107.97 | 118.54 | 18.71 | 45.13 | 2040.00 | 75.74 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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


### managed-stream

3.59% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.470528 | 81.60 | 33.10 | 2081.29 | 107.97 | 117.76 | 18.93 | 46.63 | 2040.00 | 77.93 |
| 2 | 1.488250 | 80.63 | 33.35 | 2081.57 | 107.97 | 117.95 | 18.53 | 46.01 | 2040.00 | 77.93 |
| 3 | 1.485268 | 80.79 | 31.94 | 2081.45 | 107.97 | 118.24 | 19.27 | 46.06 | 2040.00 | 77.87 |
| 4 | 1.524896 | 78.69 | 32.22 | 2081.55 | 107.97 | 118.76 | 18.55 | 45.80 | 2040.00 | 77.00 |
| 5 | 1.510707 | 79.43 | 32.39 | 2082.60 | 107.97 | 117.50 | 19.00 | 44.87 | 2040.00 | 75.87 |
| 6 | 1.517778 | 79.06 | 32.43 | 2082.68 | 107.97 | 118.01 | 18.00 | 45.06 | 2040.00 | 75.87 |
| 7 | 1.504917 | 79.74 | 33.95 | 2091.82 | 107.97 | 117.96 | 18.20 | 44.95 | 2040.00 | 75.87 |

Ready result delivery-interval P50/P95/P99 ms: 12.38/13.73/15.49.
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

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
