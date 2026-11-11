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
| yolo | 109.24 | 109.67 | 105.02 | 102.58 | 104.70 | 102.43 |
| yolo11 | 90.09 | 91.10 | 88.14 | 87.32 | 87.21 | 86.85 |
| yolo26 | 80.46 | 80.27 | 77.99 | 77.73 | 77.83 | 76.96 |
| yolo-world | 79.84 | 79.60 | 77.01 | 76.26 | 76.46 | 75.68 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 109.24 ± 3.18 | 1098.46 | 1081.55/1081.55/1081.55 | 1927.40 | 34.28/62.00 | 120/60 | PASS |
| raw-stream | 109.67 ± 1.40 | 23.55 | 1156.01/1156.01/1156.01 | 1933.99 | 34.28/62.00 | 120/60 | PASS |
| native-non-stream | 105.02 ± 1.35 | 1142.57 | 1141.70/1141.70/1141.70 | 1934.00 | 34.28/62.00 | 120/60 | PASS |
| native-stream | 102.58 ± 2.25 | 24.73 | 1161.17/1161.17/1161.17 | 1927.40 | 34.25/62.00 | 120/60 | PASS |
| managed-non-stream | 104.70 ± 3.24 | 1146.04 | 1145.62/1145.62/1145.62 | 1826.18 | 24.66/62.00 | 120/60 | PASS |
| managed-stream | 102.43 ± 3.55 | 32.58 | 1164.59/1164.59/1164.59 | 1826.34 | 24.66/62.00 | 120/60 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.093145 | 109.77 | 1093.06 | 1922.85 | 34.28 | 128.08 | 13.64 | 35.67 | 2040.00 | 53.82 |
| 2 | 1.102176 | 108.88 | 1102.09 | 1926.14 | 34.28 | 126.22 | 13.36 | 36.34 | 2040.00 | 55.91 |
| 3 | 1.188577 | 100.96 | 1188.47 | 1926.68 | 34.28 | 125.62 | 13.33 | 35.79 | 2040.00 | 56.17 |
| 4 | 1.098539 | 109.24 | 1098.46 | 1926.71 | 34.28 | 126.10 | 13.14 | 36.31 | 2040.00 | 58.00 |
| 5 | 1.118937 | 107.24 | 1118.83 | 1927.27 | 34.28 | 124.78 | 13.35 | 36.37 | 2040.00 | 58.48 |
| 6 | 1.092027 | 109.89 | 1091.95 | 1927.36 | 34.28 | 126.01 | 13.55 | 36.83 | 2040.00 | 60.00 |
| 7 | 1.098549 | 109.24 | 1098.46 | 1927.40 | 34.28 | 125.51 | 13.45 | 36.71 | 2040.00 | 60.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.08 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3773809583653929, 'inference': 6.676775950033213, 'postprocess': 0.8486413418571223}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.092540 | 109.84 | 23.09 | 1924.39 | 34.28 | 125.74 | 13.50 | 35.66 | 2040.00 | 53.82 |
| 2 | 1.102904 | 108.80 | 23.55 | 1933.50 | 34.28 | 125.22 | 13.18 | 35.96 | 2040.00 | 55.91 |
| 3 | 1.128344 | 106.35 | 24.08 | 1926.68 | 34.28 | 125.34 | 13.13 | 35.78 | 2040.00 | 56.83 |
| 4 | 1.094206 | 109.67 | 23.35 | 1933.99 | 34.28 | 125.76 | 13.45 | 36.06 | 2040.00 | 57.91 |
| 5 | 1.094999 | 109.59 | 24.39 | 1927.23 | 34.28 | 126.38 | 13.91 | 36.76 | 2040.00 | 58.82 |
| 6 | 1.093462 | 109.74 | 23.85 | 1927.01 | 34.28 | 125.76 | 13.73 | 36.73 | 2040.00 | 59.91 |
| 7 | 1.083680 | 110.73 | 23.12 | 1927.40 | 34.28 | 125.65 | 14.73 | 37.26 | 2040.00 | 60.23 |

Ready result delivery-interval P50/P95/P99 ms: 9.39/11.39/12.15.
Separate latency-pass duration: 1.16 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3445292000142217, 'inference': 6.690232866715935, 'postprocess': 0.8661661000587628}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

4.31% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.146046 | 104.71 | 1145.96 | 1924.76 | 34.28 | 124.93 | 14.04 | 36.06 | 2040.00 | 54.00 |
| 2 | 1.149731 | 104.37 | 1149.64 | 1933.42 | 34.28 | 126.55 | 14.52 | 35.76 | 2040.00 | 55.83 |
| 3 | 1.131810 | 106.02 | 1131.72 | 1926.71 | 34.28 | 124.45 | 13.65 | 36.04 | 2040.00 | 56.87 |
| 4 | 1.176397 | 102.01 | 1176.28 | 1934.00 | 34.28 | 127.12 | 13.38 | 36.03 | 2040.00 | 57.88 |
| 5 | 1.137664 | 105.48 | 1137.56 | 1927.25 | 34.28 | 124.81 | 13.78 | 36.60 | 2040.00 | 58.87 |
| 6 | 1.142660 | 105.02 | 1142.57 | 1931.63 | 34.28 | 127.23 | 14.43 | 36.41 | 2040.00 | 59.91 |
| 7 | 1.134219 | 105.80 | 1134.12 | 1927.41 | 34.28 | 124.60 | 14.39 | 37.15 | 2040.00 | 60.74 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
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


### native-stream

4.84% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.173532 | 102.26 | 25.51 | 1924.92 | 34.25 | 122.69 | 13.00 | 35.49 | 2040.00 | 54.25 |
| 2 | 1.169824 | 102.58 | 22.61 | 1823.45 | 34.25 | 121.70 | 12.88 | 35.48 | 2040.00 | 55.67 |
| 3 | 1.191923 | 100.68 | 24.73 | 1926.76 | 34.25 | 122.70 | 13.12 | 35.72 | 2040.00 | 56.96 |
| 4 | 1.208398 | 99.31 | 23.93 | 1824.20 | 34.25 | 122.92 | 12.33 | 35.80 | 2040.00 | 57.92 |
| 5 | 1.144061 | 104.89 | 25.53 | 1927.23 | 34.25 | 122.61 | 12.70 | 36.42 | 2040.00 | 58.91 |
| 6 | 1.138163 | 105.43 | 22.63 | 1824.32 | 34.25 | 122.01 | 13.04 | 36.06 | 2040.00 | 59.87 |
| 7 | 1.151702 | 104.19 | 24.79 | 1927.40 | 34.25 | 123.58 | 12.74 | 36.55 | 2040.00 | 60.78 |

Ready result delivery-interval P50/P95/P99 ms: 9.61/10.58/10.90.
Separate latency-pass duration: 1.16 s; not used for throughput.

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

4.33% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.162344 | 103.24 | 1162.25 | 1816.96 | 24.66 | 121.66 | 13.33 | 35.77 | 2040.00 | 54.75 |
| 2 | 1.145581 | 104.75 | 1145.48 | 1818.41 | 24.66 | 122.26 | 14.22 | 35.15 | 2040.00 | 55.09 |
| 3 | 1.222797 | 98.14 | 1222.69 | 1818.82 | 24.66 | 122.23 | 12.12 | 35.86 | 2040.00 | 56.92 |
| 4 | 1.226622 | 97.83 | 1226.50 | 1826.18 | 24.66 | 122.01 | 12.60 | 35.32 | 2040.00 | 57.76 |
| 5 | 1.145033 | 104.80 | 1144.93 | 1819.12 | 24.66 | 121.67 | 13.22 | 36.26 | 2040.00 | 58.91 |
| 6 | 1.142413 | 105.04 | 1142.32 | 1819.35 | 24.66 | 122.02 | 14.13 | 36.13 | 2040.00 | 59.52 |
| 7 | 1.146135 | 104.70 | 1146.04 | 1819.28 | 24.66 | 122.46 | 13.91 | 36.95 | 2040.00 | 60.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.15 s; not used for throughput.

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

7.12% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.171508 | 102.43 | 32.74 | 1817.83 | 24.66 | 120.98 | 12.67 | 35.28 | 2040.00 | 54.92 |
| 2 | 1.244780 | 96.40 | 31.56 | 1818.18 | 24.66 | 121.27 | 12.60 | 35.08 | 2040.00 | 54.92 |
| 3 | 1.245419 | 96.35 | 33.08 | 1826.04 | 24.66 | 122.95 | 12.12 | 35.31 | 2040.00 | 57.00 |
| 4 | 1.243419 | 96.51 | 32.43 | 1826.15 | 24.66 | 121.54 | 12.12 | 35.09 | 2040.00 | 57.20 |
| 5 | 1.164100 | 103.08 | 32.58 | 1819.22 | 24.66 | 122.64 | 13.17 | 36.28 | 2040.00 | 59.00 |
| 6 | 1.169798 | 102.58 | 31.39 | 1819.33 | 24.66 | 122.49 | 12.83 | 36.01 | 2040.00 | 59.04 |
| 7 | 1.154981 | 103.90 | 33.27 | 1826.34 | 24.66 | 122.68 | 12.70 | 36.36 | 2040.00 | 60.87 |

Ready result delivery-interval P50/P95/P99 ms: 9.72/10.25/11.11.
Separate latency-pass duration: 1.16 s; not used for throughput.

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
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 90.09 ± 1.71 | 1331.93 | 1295.88/1295.88/1295.88 | 2005.83 | 64.67/90.00 | 120/64 | PASS |
| raw-stream | 91.10 ± 1.79 | 25.38 | 1307.15/1307.15/1307.15 | 2005.83 | 64.67/90.00 | 120/64 | PASS |
| native-non-stream | 88.14 ± 1.66 | 1361.29 | 1325.93/1325.93/1325.93 | 2009.16 | 64.67/90.00 | 120/64 | PASS |
| native-stream | 87.32 ± 0.98 | 26.46 | 1329.66/1329.66/1329.66 | 2005.80 | 64.67/90.00 | 120/64 | PASS |
| managed-non-stream | 87.21 ± 0.96 | 1375.96 | 1324.18/1324.18/1324.18 | 1904.57 | 56.12/90.00 | 120/64 | PASS |
| managed-stream | 86.85 ± 2.20 | 33.50 | 1361.65/1361.65/1361.65 | 1904.64 | 56.12/90.00 | 120/64 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.310965 | 91.54 | 1310.87 | 2002.73 | 64.67 | 123.45 | 13.67 | 36.59 | 2040.00 | 62.85 |
| 2 | 1.339759 | 89.57 | 1339.68 | 2005.08 | 64.67 | 121.22 | 13.19 | 37.51 | 2040.00 | 64.00 |
| 3 | 1.342170 | 89.41 | 1342.09 | 2005.09 | 64.67 | 121.98 | 12.63 | 37.29 | 2040.00 | 64.44 |
| 4 | 1.308352 | 91.72 | 1308.25 | 2005.58 | 64.67 | 121.83 | 12.85 | 37.30 | 2040.00 | 65.92 |
| 5 | 1.298679 | 92.40 | 1298.60 | 2005.58 | 64.67 | 121.54 | 13.77 | 37.36 | 2040.00 | 65.92 |
| 6 | 1.332008 | 90.09 | 1331.93 | 2005.78 | 64.67 | 122.15 | 13.37 | 37.83 | 2040.00 | 67.00 |
| 7 | 1.372735 | 87.42 | 1372.63 | 2005.83 | 64.67 | 121.81 | 12.00 | 37.51 | 2040.00 | 67.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.30 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.358857808357546, 'inference': 8.495801808354978, 'postprocess': 0.858659483340792}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.90% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.317262 | 91.10 | 25.28 | 2003.50 | 64.67 | 121.30 | 13.37 | 37.16 | 2040.00 | 62.93 |
| 2 | 1.310429 | 91.57 | 26.30 | 2005.11 | 64.67 | 121.34 | 12.59 | 36.82 | 2040.00 | 64.00 |
| 3 | 1.325761 | 90.51 | 25.84 | 2005.12 | 64.67 | 122.00 | 13.22 | 37.30 | 2040.00 | 64.81 |
| 4 | 1.312889 | 91.40 | 25.66 | 2005.62 | 64.67 | 121.26 | 12.78 | 37.38 | 2040.00 | 65.81 |
| 5 | 1.286338 | 93.29 | 24.81 | 2005.58 | 64.67 | 121.75 | 14.54 | 37.80 | 2040.00 | 65.92 |
| 6 | 1.320047 | 90.91 | 25.07 | 2005.79 | 64.67 | 121.53 | 12.78 | 37.19 | 2040.00 | 67.00 |
| 7 | 1.373861 | 87.35 | 25.38 | 2005.83 | 64.67 | 121.63 | 12.43 | 37.76 | 2040.00 | 67.50 |

Ready result delivery-interval P50/P95/P99 ms: 10.78/12.43/12.67.
Separate latency-pass duration: 1.31 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3200679083790117, 'inference': 8.56202306670942, 'postprocess': 0.88299942502393}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.72% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.361474 | 88.14 | 1361.38 | 2003.57 | 64.67 | 120.78 | 12.78 | 36.96 | 2040.00 | 63.00 |
| 2 | 1.361413 | 88.14 | 1361.29 | 2005.00 | 64.67 | 122.68 | 13.59 | 37.06 | 2040.00 | 63.93 |
| 3 | 1.319921 | 90.91 | 1319.81 | 2005.13 | 64.67 | 121.54 | 13.48 | 37.49 | 2040.00 | 64.89 |
| 4 | 1.399446 | 85.75 | 1399.34 | 2005.53 | 64.67 | 122.71 | 12.29 | 37.08 | 2040.00 | 65.86 |
| 5 | 1.334063 | 89.95 | 1333.96 | 2005.59 | 64.67 | 121.02 | 13.78 | 37.58 | 2040.00 | 66.00 |
| 6 | 1.370544 | 87.56 | 1370.44 | 2009.16 | 64.67 | 122.69 | 13.43 | 37.91 | 2040.00 | 66.93 |
| 7 | 1.356749 | 88.45 | 1356.62 | 2005.81 | 64.67 | 120.56 | 13.19 | 37.55 | 2040.00 | 67.70 |

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

3.78% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.365156 | 87.90 | 26.58 | 2003.69 | 64.67 | 119.37 | 12.39 | 36.66 | 2040.00 | 63.00 |
| 2 | 1.390336 | 86.31 | 24.42 | 1902.55 | 64.67 | 117.99 | 12.46 | 36.95 | 2040.00 | 63.89 |
| 3 | 1.341665 | 89.44 | 26.64 | 2005.15 | 64.67 | 119.45 | 12.33 | 36.95 | 2040.00 | 64.85 |
| 4 | 1.375694 | 87.23 | 24.66 | 1903.98 | 64.67 | 119.44 | 12.57 | 37.36 | 2040.00 | 65.25 |
| 5 | 1.377253 | 87.13 | 27.05 | 2005.58 | 64.67 | 118.72 | 12.18 | 37.34 | 2040.00 | 66.00 |
| 6 | 1.364005 | 87.98 | 24.55 | 1904.56 | 64.67 | 119.17 | 13.18 | 37.32 | 2040.00 | 66.89 |
| 7 | 1.374308 | 87.32 | 26.46 | 2005.80 | 64.67 | 119.38 | 12.29 | 37.50 | 2040.00 | 67.82 |

Ready result delivery-interval P50/P95/P99 ms: 11.14/11.82/12.16.
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


### managed-non-stream

3.66% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.376060 | 87.21 | 1375.96 | 1894.52 | 56.12 | 118.60 | 13.04 | 36.87 | 2040.00 | 63.00 |
| 2 | 1.377023 | 87.14 | 1376.92 | 1902.84 | 56.12 | 118.41 | 12.75 | 36.39 | 2040.00 | 63.86 |
| 3 | 1.346735 | 89.10 | 1346.63 | 1897.36 | 56.12 | 119.80 | 12.70 | 36.87 | 2040.00 | 64.93 |
| 4 | 1.360052 | 88.23 | 1359.96 | 1904.35 | 56.12 | 118.97 | 13.07 | 37.28 | 2040.00 | 65.00 |
| 5 | 1.386297 | 86.56 | 1386.19 | 1897.67 | 56.12 | 119.56 | 12.86 | 37.18 | 2040.00 | 66.04 |
| 6 | 1.380745 | 86.91 | 1380.65 | 1904.57 | 56.12 | 118.79 | 13.14 | 36.53 | 2040.00 | 66.86 |
| 7 | 1.354765 | 88.58 | 1354.67 | 1897.88 | 56.12 | 119.17 | 12.48 | 37.48 | 2040.00 | 67.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.32 s; not used for throughput.

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
| 1 | 1.381736 | 86.85 | 36.15 | 1902.31 | 56.12 | 119.38 | 12.29 | 36.47 | 2040.00 | 63.50 |
| 2 | 1.428438 | 84.01 | 33.44 | 1902.59 | 56.12 | 118.76 | 12.69 | 36.42 | 2040.00 | 63.83 |
| 3 | 1.371555 | 87.49 | 34.04 | 1904.20 | 56.12 | 118.84 | 13.00 | 36.79 | 2040.00 | 65.00 |
| 4 | 1.375755 | 87.22 | 33.33 | 1904.32 | 56.12 | 118.86 | 12.29 | 36.85 | 2040.00 | 65.00 |
| 5 | 1.457456 | 82.34 | 34.93 | 1904.42 | 56.12 | 118.90 | 12.38 | 36.30 | 2040.00 | 66.38 |
| 6 | 1.413007 | 84.93 | 32.97 | 1904.55 | 56.12 | 118.96 | 12.10 | 36.67 | 2040.00 | 66.83 |
| 7 | 1.356376 | 88.47 | 33.50 | 1904.64 | 56.12 | 119.40 | 12.63 | 37.09 | 2040.00 | 67.93 |

Ready result delivery-interval P50/P95/P99 ms: 11.31/11.96/12.21.
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

## Detail — yolo26, video, batch 1

Status: **complete**. Load 0.12 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 80.46 ± 1.39 | 1491.36 | 1564.37/1564.37/1564.37 | 1937.64 | 62.83/90.00 | 120/61 | PASS |
| raw-stream | 80.27 ± 0.92 | 26.92 | 1571.07/1571.07/1571.07 | 1944.16 | 62.83/90.00 | 120/61 | PASS |
| native-non-stream | 77.99 ± 1.51 | 1538.46 | 1559.78/1559.78/1559.78 | 1944.20 | 62.83/90.00 | 120/61 | PASS |
| native-stream | 77.73 ± 1.04 | 27.88 | 1591.45/1591.45/1591.45 | 1937.61 | 62.80/90.00 | 120/61 | PASS |
| managed-non-stream | 77.83 ± 1.03 | 1541.72 | 1575.49/1575.49/1575.49 | 1836.27 | 55.34/90.00 | 120/61 | PASS |
| managed-stream | 76.96 ± 1.48 | 35.80 | 1606.85/1606.85/1606.85 | 1836.28 | 55.34/90.00 | 120/61 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.495772 | 80.23 | 1495.69 | 1933.25 | 62.83 | 120.87 | 12.40 | 37.60 | 2040.00 | 69.00 |
| 2 | 1.491439 | 80.46 | 1491.36 | 1936.87 | 62.83 | 118.84 | 12.57 | 38.42 | 2040.00 | 70.77 |
| 3 | 1.472749 | 81.48 | 1472.67 | 1936.92 | 62.83 | 119.21 | 13.13 | 38.08 | 2040.00 | 70.87 |
| 4 | 1.481075 | 81.02 | 1481.00 | 1937.34 | 62.83 | 118.91 | 13.13 | 38.58 | 2040.00 | 71.93 |
| 5 | 1.476907 | 81.25 | 1476.82 | 1937.34 | 62.83 | 119.04 | 12.87 | 38.47 | 2040.00 | 71.87 |
| 6 | 1.536779 | 78.09 | 1536.70 | 1937.61 | 62.83 | 119.29 | 12.19 | 38.69 | 2040.00 | 72.94 |
| 7 | 1.532679 | 78.29 | 1532.59 | 1937.64 | 62.83 | 119.63 | 12.26 | 38.68 | 2040.00 | 73.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.56 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4052791833213025, 'inference': 9.967724074992171, 'postprocess': 0.8765791167055189}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.485424 | 80.79 | 26.92 | 1934.50 | 62.83 | 119.72 | 12.80 | 37.83 | 2040.00 | 69.23 |
| 2 | 1.499801 | 80.01 | 26.79 | 1936.53 | 62.83 | 119.09 | 12.27 | 37.86 | 2040.00 | 70.33 |
| 3 | 1.482677 | 80.93 | 27.33 | 1936.93 | 62.83 | 118.84 | 13.33 | 37.99 | 2040.00 | 70.87 |
| 4 | 1.507283 | 79.61 | 30.40 | 1943.84 | 62.83 | 119.25 | 12.93 | 38.27 | 2040.00 | 71.87 |
| 5 | 1.471600 | 81.54 | 26.87 | 1937.34 | 62.83 | 119.40 | 13.03 | 38.88 | 2040.00 | 72.00 |
| 6 | 1.495003 | 80.27 | 26.70 | 1944.16 | 62.83 | 119.42 | 12.37 | 38.43 | 2040.00 | 72.93 |
| 7 | 1.523175 | 78.78 | 27.18 | 1937.64 | 62.83 | 119.36 | 12.19 | 38.93 | 2040.00 | 73.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.88/15.50/16.45.
Separate latency-pass duration: 1.57 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.400990700130933, 'inference': 9.872885191665167, 'postprocess': 0.8923807498983175}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.08% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.529167 | 78.47 | 1529.08 | 1934.18 | 62.83 | 118.27 | 13.35 | 37.69 | 2040.00 | 69.74 |
| 2 | 1.538562 | 77.99 | 1538.46 | 1942.75 | 62.83 | 119.79 | 13.45 | 37.96 | 2040.00 | 70.00 |
| 3 | 1.601074 | 74.95 | 1600.99 | 1936.90 | 62.83 | 118.03 | 12.62 | 38.27 | 2040.00 | 70.94 |
| 4 | 1.578073 | 76.04 | 1577.99 | 1943.87 | 62.83 | 119.78 | 12.47 | 38.25 | 2040.00 | 71.88 |
| 5 | 1.522447 | 78.82 | 1522.35 | 1937.35 | 62.83 | 118.60 | 13.94 | 38.86 | 2040.00 | 72.00 |
| 6 | 1.521058 | 78.89 | 1520.94 | 1944.20 | 62.83 | 120.02 | 13.81 | 38.61 | 2040.00 | 72.94 |
| 7 | 1.539300 | 77.96 | 1539.18 | 1937.62 | 62.83 | 118.68 | 13.35 | 38.89 | 2040.00 | 73.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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


### native-stream

3.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.553176 | 77.26 | 27.88 | 1934.34 | 62.80 | 116.92 | 12.10 | 37.60 | 2040.00 | 69.84 |
| 2 | 1.538128 | 78.02 | 26.10 | 1833.33 | 62.80 | 116.98 | 12.06 | 37.75 | 2040.00 | 70.00 |
| 3 | 1.578490 | 76.02 | 29.86 | 1936.94 | 62.80 | 117.38 | 11.50 | 37.92 | 2040.00 | 70.94 |
| 4 | 1.529041 | 78.48 | 25.97 | 1833.95 | 62.80 | 116.92 | 13.23 | 38.17 | 2040.00 | 71.74 |
| 5 | 1.543813 | 77.73 | 27.99 | 1937.31 | 62.80 | 117.24 | 12.65 | 38.41 | 2040.00 | 72.00 |
| 6 | 1.536535 | 78.10 | 26.11 | 1834.31 | 62.80 | 117.43 | 12.16 | 38.56 | 2040.00 | 72.87 |
| 7 | 1.582646 | 75.82 | 28.44 | 1937.61 | 62.80 | 117.28 | 11.88 | 38.63 | 2040.00 | 73.25 |

Ready result delivery-interval P50/P95/P99 ms: 13.33/14.31/15.61.
Separate latency-pass duration: 1.59 s; not used for throughput.

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

3.29% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.539198 | 77.96 | 1539.10 | 1826.41 | 55.34 | 116.87 | 12.71 | 37.72 | 2040.00 | 69.94 |
| 2 | 1.542617 | 77.79 | 1542.52 | 1834.77 | 55.34 | 116.91 | 12.58 | 37.80 | 2040.00 | 70.00 |
| 3 | 1.541821 | 77.83 | 1541.72 | 1828.95 | 55.34 | 116.85 | 12.19 | 37.95 | 2040.00 | 71.00 |
| 4 | 1.521590 | 78.86 | 1521.50 | 1835.59 | 55.34 | 116.94 | 13.10 | 38.25 | 2040.00 | 71.19 |
| 5 | 1.562484 | 76.80 | 1562.31 | 1829.20 | 55.34 | 116.95 | 12.06 | 38.36 | 2040.00 | 72.00 |
| 6 | 1.530478 | 78.41 | 1530.38 | 1836.27 | 55.34 | 116.98 | 12.71 | 38.74 | 2040.00 | 72.87 |
| 7 | 1.583136 | 75.80 | 1583.03 | 1829.50 | 55.34 | 117.15 | 12.25 | 38.67 | 2040.00 | 73.72 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.58 s; not used for throughput.

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

5.28% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.533726 | 78.24 | 35.25 | 1834.21 | 55.34 | 116.80 | 12.81 | 37.81 | 2040.00 | 69.94 |
| 2 | 1.564951 | 76.68 | 35.80 | 1834.53 | 55.34 | 116.79 | 12.19 | 37.62 | 2040.00 | 69.94 |
| 3 | 1.556819 | 77.08 | 36.09 | 1835.79 | 55.34 | 116.94 | 12.06 | 37.90 | 2040.00 | 71.00 |
| 4 | 1.559336 | 76.96 | 34.87 | 1835.94 | 55.34 | 117.46 | 12.58 | 37.89 | 2040.00 | 71.00 |
| 5 | 1.579436 | 75.98 | 36.60 | 1835.97 | 55.34 | 116.70 | 12.81 | 38.21 | 2040.00 | 72.25 |
| 6 | 1.527189 | 78.58 | 34.98 | 1836.27 | 55.34 | 116.64 | 12.84 | 38.32 | 2040.00 | 72.71 |
| 7 | 1.618647 | 74.14 | 35.81 | 1836.28 | 55.34 | 117.17 | 12.39 | 38.61 | 2040.00 | 73.88 |

Ready result delivery-interval P50/P95/P99 ms: 13.18/14.50/17.00.
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
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 79.84 ± 1.70 | 1502.86 | 1484.33/1484.33/1484.33 | 2193.12 | 115.88/134.00 | 120/61 | PASS |
| raw-stream | 79.60 ± 0.82 | 27.13 | 1473.63/1473.63/1473.63 | 2193.12 | 115.88/134.00 | 120/61 | PASS |
| native-non-stream | 77.01 ± 0.75 | 1558.18 | 1537.82/1537.82/1537.82 | 2193.10 | 115.88/134.00 | 120/61 | PASS |
| native-stream | 76.26 ± 1.87 | 27.66 | 1512.41/1512.41/1512.41 | 2193.09 | 115.88/134.00 | 120/61 | PASS |
| managed-non-stream | 76.46 ± 1.93 | 1569.32 | 1510.58/1510.58/1510.58 | 2083.80 | 107.97/134.00 | 120/61 | PASS |
| managed-stream | 75.68 ± 2.28 | 35.10 | 1513.83/1513.83/1513.83 | 2083.84 | 107.97/134.00 | 120/61 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.525718 | 78.65 | 1525.63 | 2180.85 | 115.88 | 121.73 | 19.35 | 45.17 | 2040.00 | 74.87 |
| 2 | 1.453825 | 82.54 | 1453.74 | 2189.77 | 115.88 | 119.58 | 19.14 | 46.20 | 2040.00 | 74.93 |
| 3 | 1.458005 | 82.30 | 1457.91 | 2189.78 | 115.88 | 119.60 | 18.72 | 45.71 | 2040.00 | 74.93 |
| 4 | 1.516153 | 79.15 | 1516.03 | 2192.82 | 115.88 | 119.83 | 18.55 | 46.14 | 2040.00 | 75.81 |
| 5 | 1.502956 | 79.84 | 1502.86 | 2192.79 | 115.88 | 119.76 | 18.47 | 45.50 | 2040.00 | 75.80 |
| 6 | 1.523660 | 78.76 | 1523.57 | 2193.12 | 115.88 | 119.01 | 18.03 | 45.77 | 2040.00 | 76.84 |
| 7 | 1.469708 | 81.65 | 1469.62 | 2193.12 | 115.88 | 119.89 | 18.77 | 46.04 | 2040.00 | 76.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.48 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4851857167438236, 'inference': 10.08794441675794, 'postprocess': 0.9199353999671681}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.69% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.516467 | 79.13 | 27.04 | 2181.61 | 115.88 | 119.84 | 18.81 | 45.91 | 2040.00 | 74.90 |
| 2 | 1.498889 | 80.06 | 27.13 | 2189.77 | 115.88 | 119.21 | 19.13 | 45.22 | 2040.00 | 74.87 |
| 3 | 1.487987 | 80.65 | 31.05 | 2189.80 | 115.88 | 121.64 | 18.53 | 45.90 | 2040.00 | 74.93 |
| 4 | 1.512991 | 79.31 | 26.56 | 2192.80 | 115.88 | 119.37 | 18.65 | 45.55 | 2040.00 | 75.84 |
| 5 | 1.507534 | 79.60 | 26.43 | 2192.78 | 115.88 | 119.62 | 18.10 | 45.88 | 2040.00 | 75.87 |
| 6 | 1.534189 | 78.22 | 27.17 | 2193.12 | 115.88 | 119.38 | 18.42 | 46.07 | 2040.00 | 76.84 |
| 7 | 1.494630 | 80.29 | 27.22 | 2193.09 | 115.88 | 120.32 | 19.27 | 46.41 | 2040.00 | 76.87 |

Ready result delivery-interval P50/P95/P99 ms: 12.20/12.99/14.70.
Separate latency-pass duration: 1.47 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4374506916586445, 'inference': 10.034587100062708, 'postprocess': 0.9435987749687532}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

5.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.547127 | 77.56 | 1547.04 | 2181.61 | 115.88 | 119.37 | 18.16 | 45.95 | 2040.00 | 74.87 |
| 2 | 1.568857 | 76.49 | 1568.75 | 2189.69 | 115.88 | 120.55 | 18.16 | 45.41 | 2040.00 | 74.94 |
| 3 | 1.557961 | 77.02 | 1557.85 | 2184.96 | 115.88 | 118.91 | 18.29 | 46.00 | 2040.00 | 74.87 |
| 4 | 1.558283 | 77.01 | 1558.18 | 2192.10 | 115.88 | 120.34 | 18.42 | 46.27 | 2040.00 | 75.42 |
| 5 | 1.590425 | 75.45 | 1590.32 | 2192.79 | 115.88 | 119.09 | 18.88 | 45.53 | 2040.00 | 75.84 |
| 6 | 1.546886 | 77.58 | 1546.75 | 2192.39 | 115.88 | 120.42 | 18.03 | 46.11 | 2040.00 | 76.71 |
| 7 | 1.570546 | 76.41 | 1570.46 | 2193.10 | 115.88 | 119.05 | 18.16 | 46.16 | 2040.00 | 76.88 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
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


### native-stream

4.01% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.541682 | 77.84 | 29.21 | 2181.62 | 115.88 | 118.14 | 18.45 | 45.19 | 2040.00 | 74.87 |
| 2 | 1.512177 | 79.36 | 25.16 | 2080.79 | 115.88 | 117.62 | 19.47 | 45.49 | 2040.00 | 74.87 |
| 3 | 1.577223 | 76.08 | 27.66 | 2184.98 | 115.88 | 117.90 | 17.94 | 45.12 | 2040.00 | 74.88 |
| 4 | 1.558521 | 77.00 | 26.31 | 2083.43 | 115.88 | 117.01 | 18.42 | 45.24 | 2040.00 | 75.10 |
| 5 | 1.612019 | 74.44 | 29.00 | 2192.78 | 115.88 | 117.57 | 17.25 | 45.21 | 2040.00 | 75.88 |
| 6 | 1.573517 | 76.26 | 25.39 | 2083.80 | 115.88 | 117.51 | 17.50 | 46.09 | 2040.00 | 76.00 |
| 7 | 1.622182 | 73.97 | 29.61 | 2193.09 | 115.88 | 116.83 | 17.79 | 45.56 | 2040.00 | 77.09 |

Ready result delivery-interval P50/P95/P99 ms: 12.65/13.42/14.76.
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

4.28% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.547565 | 77.54 | 1547.45 | 2072.44 | 107.97 | 117.40 | 17.90 | 45.10 | 2040.00 | 74.87 |
| 2 | 1.491205 | 80.47 | 1491.11 | 2080.73 | 107.97 | 117.13 | 19.67 | 45.15 | 2040.00 | 74.87 |
| 3 | 1.607916 | 74.63 | 1607.82 | 2075.86 | 107.97 | 117.43 | 18.41 | 44.93 | 2040.00 | 74.94 |
| 4 | 1.581308 | 75.89 | 1581.21 | 2083.32 | 107.97 | 117.20 | 17.53 | 44.86 | 2040.00 | 74.94 |
| 5 | 1.569417 | 76.46 | 1569.32 | 2083.48 | 107.97 | 116.94 | 17.44 | 45.22 | 2040.00 | 75.88 |
| 6 | 1.579885 | 75.95 | 1579.79 | 2083.75 | 107.97 | 117.30 | 18.16 | 45.13 | 2040.00 | 75.94 |
| 7 | 1.532552 | 78.30 | 1532.45 | 2083.80 | 107.97 | 117.41 | 19.29 | 45.67 | 2040.00 | 77.65 |

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


### managed-stream

5.45% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.501838 | 79.90 | 35.10 | 2080.20 | 107.97 | 117.53 | 19.20 | 45.77 | 2040.00 | 74.87 |
| 2 | 1.530542 | 78.40 | 34.73 | 2080.55 | 107.97 | 117.00 | 18.87 | 45.31 | 2040.00 | 74.90 |
| 3 | 1.615577 | 74.28 | 36.43 | 2083.13 | 107.97 | 117.09 | 17.18 | 45.24 | 2040.00 | 74.94 |
| 4 | 1.585675 | 75.68 | 34.67 | 2083.30 | 107.97 | 116.95 | 17.41 | 44.45 | 2040.00 | 74.94 |
| 5 | 1.619044 | 74.12 | 35.05 | 2083.51 | 107.97 | 116.72 | 17.76 | 45.15 | 2040.00 | 75.88 |
| 6 | 1.610877 | 74.49 | 36.66 | 2083.76 | 107.97 | 117.30 | 17.44 | 45.04 | 2040.00 | 75.88 |
| 7 | 1.549801 | 77.43 | 35.20 | 2083.84 | 107.97 | 117.68 | 17.97 | 45.85 | 2040.00 | 77.71 |

Ready result delivery-interval P50/P95/P99 ms: 12.60/13.27/13.72.
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

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
