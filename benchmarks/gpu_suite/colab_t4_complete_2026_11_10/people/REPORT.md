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
- Image diagnostics contain only 8 or 32 deliveries. Their interpolated P95/P99 are descriptive samples, not tail-latency SLA evidence.
- Throughput passes retain yielded results for untimed parity; peak memory includes retention. The separate stability probe consumes without retaining frames.
- RSS is sampled at 50 ms; short peaks can be missed. CPU percentages are approximate, expressed relative to one core and include probe setup. NVML uses device-window statistics.
- Reserved VRAM includes allocator caches from preceding modes. Allocated peaks describe live tensor allocation; neither is an isolated minimum-memory requirement.
- Factory load time excludes process/hardware initialization and includes checkpoint lookup/download when needed; cache states differ between jobs.
- This controlled public bus workload does not qualify production accuracy or live-camera behavior. A separate stability result, when present, covers one managed pipeline, not all models or 30-minute production use.

## Environment

GPU: **Tesla T4**, VRAM 14912.69 MiB. CUDA 13.0.
Python: `3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]`. OS: `Linux-6.6.159+-x86_64-with-glibc2.39`.
CPU logical cores: 2; Torch threads: 2.

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

Tested commit(s): `f83352df3c45ac19bf2579383985301f80b45bc3`

## Attempt audit

The original model groups are frozen: native overrides are transient and are not remembered by managed prediction. Real Detector API tests confirm that the original measurements used independent mode settings.
Rejected control probes: 0. These attempted an explicit managed-time override of the frozen model group, were rejected before final measurements, and are excluded from tables.
Setup/execution failures are retained separately from the selected final attempts:

- None: None — None

## FPS matrix — video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 97.21 | 99.41 | 92.00 | 88.69 | 89.44 | 86.34 |
| yolo11 | 81.88 | 80.10 | 77.53 | 72.72 | 73.87 | 72.78 |
| yolo26 | 58.15 | 75.21 | 73.54 | 68.69 | 68.25 | 53.49 |
| yolo-world | 76.48 | 77.43 | 70.81 | 61.24 | 71.48 | 70.91 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 97.21 ± 4.13 | 1234.31 | 1334.13/1334.13/1334.13 | 1892.94 | 34.28/62.00 | 120/60 | PASS |
| raw-stream | 99.41 ± 14.49 | 25.09 | 1202.63/1202.63/1202.63 | 1892.94 | 34.28/62.00 | 120/60 | PASS |
| native-non-stream | 92.00 ± 14.59 | 1304.28 | 1299.01/1299.01/1299.01 | 1892.97 | 34.28/62.00 | 120/60 | PASS |
| native-stream | 88.69 ± 7.22 | 1352.93 | 1322.47/1322.47/1322.47 | 1892.94 | 34.25/62.00 | 120/60 | PASS |
| managed-non-stream | 89.44 ± 6.94 | 1341.58 | 2028.05/2028.05/2028.05 | 1782.93 | 24.66/62.00 | 120/60 | PASS |
| managed-stream | 86.34 ± 13.68 | 32.95 | 1626.22/1626.22/1626.22 | 1724.30 | 24.66/62.00 | 120/60 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.186362 | 101.15 | 1186.25 | 1830.06 | 34.28 | 134.99 | 32.90 | 35.51 | 995.00 | 50.00 |
| 2 | 1.245991 | 96.31 | 1245.91 | 1833.86 | 34.28 | 132.80 | 31.95 | 35.69 | 962.05 | 53.00 |
| 3 | 1.194603 | 100.45 | 1194.48 | 1833.82 | 34.28 | 131.92 | 33.90 | 36.02 | 989.29 | 53.52 |
| 4 | 1.259486 | 95.28 | 1259.39 | 1834.97 | 34.28 | 128.05 | 34.27 | 35.99 | 878.86 | 56.00 |
| 5 | 1.190883 | 100.77 | 1190.76 | 1834.94 | 34.28 | 131.58 | 31.48 | 37.46 | 1065.00 | 56.00 |
| 6 | 1.234405 | 97.21 | 1234.31 | 1889.20 | 34.28 | 136.80 | 33.48 | 36.53 | 950.71 | 58.05 |
| 7 | 1.340366 | 89.53 | 1340.21 | 1892.94 | 34.28 | 133.18 | 27.61 | 36.35 | 1020.00 | 58.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
Separate latency-pass duration: 1.33 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8816287833563667, 'inference': 6.865948583291963, 'postprocess': 0.8652026999773929}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.52% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.204569 | 99.62 | 25.08 | 1830.04 | 34.28 | 134.94 | 32.70 | 35.81 | 1050.00 | 50.70 |
| 2 | 1.243440 | 96.51 | 24.78 | 1833.83 | 34.28 | 132.90 | 32.55 | 35.70 | 990.00 | 53.00 |
| 3 | 1.200755 | 99.94 | 24.04 | 1833.82 | 34.28 | 132.92 | 34.05 | 36.31 | 1003.57 | 53.95 |
| 4 | 1.957020 | 61.32 | 39.93 | 1834.97 | 34.28 | 97.78 | 25.73 | 33.32 | 750.50 | 55.93 |
| 5 | 1.160463 | 103.41 | 25.40 | 1834.94 | 34.28 | 133.66 | 32.55 | 36.91 | 1017.00 | 56.40 |
| 6 | 1.260306 | 95.21 | 25.09 | 1835.11 | 34.28 | 132.30 | 30.82 | 36.94 | 1005.00 | 58.00 |
| 7 | 1.207154 | 99.41 | 25.50 | 1892.94 | 34.28 | 133.20 | 35.00 | 36.76 | 977.86 | 58.95 |

Ready result delivery-interval P50/P95/P99 ms: 9.56/12.01/16.79.
Separate latency-pass duration: 1.20 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8922957750040343, 'inference': 6.892632466633586, 'postprocess': 0.9044142083136345}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

7.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.302926 | 92.10 | 1302.84 | 1830.05 | 34.28 | 131.83 | 33.00 | 34.89 | 996.82 | 50.95 |
| 2 | 1.273728 | 94.21 | 1273.63 | 1833.71 | 34.28 | 134.66 | 36.14 | 35.46 | 953.86 | 52.95 |
| 3 | 1.921509 | 62.45 | 1921.36 | 1833.85 | 34.28 | 99.33 | 27.10 | 33.34 | 820.00 | 54.00 |
| 4 | 1.349768 | 88.90 | 1349.62 | 1834.96 | 34.28 | 131.73 | 33.14 | 35.49 | 1005.00 | 55.86 |
| 5 | 1.262279 | 95.07 | 1262.18 | 1834.97 | 34.28 | 132.24 | 34.09 | 36.82 | 1005.00 | 56.91 |
| 6 | 1.304378 | 92.00 | 1304.28 | 1835.12 | 34.28 | 134.30 | 35.17 | 36.45 | 928.04 | 58.00 |
| 7 | 1.898030 | 63.22 | 1897.85 | 1892.97 | 34.28 | 102.70 | 26.93 | 34.25 | 859.50 | 59.00 |

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

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### native-stream

13.61% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.374250 | 87.32 | 1374.18 | 1830.20 | 34.25 | 128.33 | 28.71 | 34.83 | 1016.88 | 51.00 |
| 2 | 1.316970 | 91.12 | 1316.89 | 1723.65 | 34.25 | 129.21 | 33.91 | 34.95 | 910.91 | 52.27 |
| 3 | 1.462300 | 82.06 | 1462.23 | 1833.87 | 34.25 | 125.42 | 34.64 | 34.08 | 741.60 | 54.00 |
| 4 | 1.296337 | 92.57 | 1296.26 | 1724.95 | 34.25 | 129.98 | 36.05 | 35.56 | 858.41 | 55.00 |
| 5 | 1.353014 | 88.69 | 1352.93 | 1834.95 | 34.25 | 131.57 | 32.74 | 35.44 | 910.43 | 56.96 |
| 6 | 1.345161 | 89.21 | 1345.04 | 1724.24 | 34.25 | 130.50 | 31.43 | 35.46 | 915.00 | 57.96 |
| 7 | 1.677299 | 71.54 | 1677.20 | 1892.94 | 34.25 | 115.65 | 32.30 | 34.61 | 701.11 | 59.00 |

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

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-non-stream

10.57% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.295892 | 92.60 | 1295.78 | 1720.39 | 24.66 | 129.67 | 33.74 | 34.80 | 950.87 | 51.30 |
| 2 | 1.341723 | 89.44 | 1341.58 | 1723.54 | 24.66 | 131.11 | 38.41 | 34.19 | 785.45 | 52.00 |
| 3 | 1.325534 | 90.53 | 1325.44 | 1724.05 | 24.66 | 129.52 | 32.22 | 35.22 | 945.00 | 54.17 |
| 4 | 1.392657 | 86.17 | 1392.54 | 1724.30 | 24.66 | 130.43 | 31.42 | 34.72 | 895.62 | 55.00 |
| 5 | 1.321338 | 90.82 | 1321.19 | 1724.12 | 24.66 | 133.75 | 34.77 | 35.26 | 846.14 | 57.00 |
| 6 | 1.663617 | 72.13 | 1663.51 | 1724.23 | 24.66 | 116.27 | 33.04 | 34.01 | 676.67 | 57.41 |
| 7 | 1.351741 | 88.77 | 1351.63 | 1782.93 | 24.66 | 131.30 | 32.04 | 35.73 | 900.00 | 59.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 2.03 s; not used for throughput.

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

13.04% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.563864 | 76.73 | 33.66 | 1722.24 | 24.66 | 119.18 | 28.15 | 34.02 | 975.00 | 51.92 |
| 2 | 2.038968 | 58.85 | 45.64 | 1723.43 | 24.66 | 100.98 | 27.94 | 32.36 | 673.59 | 51.97 |
| 3 | 1.327255 | 90.41 | 31.19 | 1724.17 | 24.66 | 130.43 | 34.13 | 34.90 | 908.48 | 54.91 |
| 4 | 1.389906 | 86.34 | 34.25 | 1724.30 | 24.66 | 128.89 | 28.21 | 35.23 | 975.00 | 54.96 |
| 5 | 1.346136 | 89.14 | 32.25 | 1724.16 | 24.66 | 129.46 | 33.91 | 35.21 | 905.22 | 57.00 |
| 6 | 1.995004 | 60.15 | 32.95 | 1724.23 | 24.66 | 101.25 | 24.34 | 34.09 | 862.50 | 57.00 |
| 7 | 1.371704 | 87.48 | 31.41 | 1721.38 | 24.66 | 129.63 | 33.48 | 35.54 | 870.00 | 59.48 |

Ready result delivery-interval P50/P95/P99 ms: 11.62/23.30/27.32.
Separate latency-pass duration: 1.63 s; not used for throughput.

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

Status: **complete**. Load 0.61 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 81.88 ± 11.80 | 1465.43 | 1490.05/1490.05/1490.05 | 1881.50 | 64.96/90.00 | 120/64 | PASS |
| raw-stream | 80.10 ± 6.48 | 27.83 | 2208.67/2208.67/2208.67 | 1873.08 | 64.96/90.00 | 120/64 | PASS |
| native-non-stream | 77.53 ± 13.26 | 1547.70 | 1712.16/1712.16/1712.16 | 1872.93 | 64.96/90.00 | 120/64 | PASS |
| native-stream | 72.72 ± 2.47 | 1649.98 | 1658.39/1658.39/1658.39 | 1868.18 | 64.94/90.00 | 120/64 | PASS |
| managed-non-stream | 73.87 ± 10.41 | 1624.47 | 1653.70/1653.70/1653.70 | 1762.11 | 56.35/90.00 | 120/64 | PASS |
| managed-stream | 72.78 ± 8.78 | 35.49 | 1642.09/1642.09/1642.09 | 1762.00 | 56.35/90.00 | 120/64 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.452828 | 82.60 | 1452.72 | 1868.00 | 64.96 | 133.42 | 35.44 | 35.54 | 841.20 | 62.00 |
| 2 | 1.465501 | 81.88 | 1465.43 | 1878.70 | 64.96 | 129.81 | 33.32 | 36.37 | 869.40 | 64.00 |
| 3 | 1.468303 | 81.73 | 1468.22 | 1881.50 | 64.96 | 130.52 | 34.64 | 36.42 | 836.40 | 64.00 |
| 4 | 2.112537 | 56.80 | 2112.41 | 1820.34 | 64.96 | 103.70 | 28.30 | 35.16 | 658.18 | 66.00 |
| 5 | 1.455880 | 82.42 | 1455.80 | 1820.34 | 64.96 | 128.95 | 34.40 | 36.45 | 823.20 | 66.00 |
| 6 | 1.433691 | 83.70 | 1433.60 | 1811.66 | 64.96 | 131.08 | 35.36 | 37.36 | 865.20 | 68.00 |
| 7 | 2.000865 | 59.97 | 2000.73 | 1811.68 | 64.96 | 107.44 | 27.41 | 35.85 | 797.81 | 68.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.49 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.7795511416655547, 'inference': 9.107017133427084, 'postprocess': 0.9058779417273399}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.06% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.498163 | 80.10 | 28.57 | 1867.97 | 64.96 | 126.92 | 31.77 | 35.79 | 881.54 | 62.00 |
| 2 | 1.449929 | 82.76 | 27.52 | 1873.08 | 64.96 | 129.62 | 35.60 | 35.94 | 782.40 | 64.00 |
| 3 | 1.436685 | 83.53 | 27.83 | 1818.93 | 64.96 | 130.15 | 35.12 | 36.41 | 881.40 | 64.24 |
| 4 | 1.647698 | 72.83 | 26.06 | 1819.39 | 64.96 | 121.38 | 31.56 | 36.45 | 848.89 | 66.00 |
| 5 | 1.478039 | 81.19 | 26.24 | 1820.37 | 64.96 | 129.45 | 34.04 | 36.99 | 866.54 | 66.00 |
| 6 | 1.542336 | 77.80 | 29.97 | 1811.63 | 64.96 | 128.00 | 30.85 | 36.79 | 870.00 | 67.96 |
| 7 | 1.834051 | 65.43 | 42.48 | 1811.63 | 64.96 | 115.19 | 33.34 | 36.34 | 671.90 | 68.00 |

Ready result delivery-interval P50/P95/P99 ms: 17.17/24.47/28.52.
Separate latency-pass duration: 2.21 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8939038166839357, 'inference': 9.306095841642067, 'postprocess': 0.9163437083392031}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.269099 | 52.88 | 2268.97 | 1868.03 | 64.96 | 99.76 | 28.11 | 34.23 | 697.29 | 62.40 |
| 2 | 2.414688 | 49.70 | 2414.62 | 1872.93 | 64.96 | 101.25 | 28.38 | 34.04 | 616.62 | 63.97 |
| 3 | 1.508483 | 79.55 | 1508.38 | 1820.16 | 64.96 | 129.07 | 34.42 | 36.29 | 886.73 | 64.85 |
| 4 | 1.534667 | 78.19 | 1534.58 | 1819.40 | 64.96 | 130.75 | 33.73 | 36.42 | 855.00 | 65.96 |
| 5 | 1.504587 | 79.76 | 1504.51 | 1820.37 | 64.96 | 129.45 | 34.62 | 37.04 | 873.46 | 66.54 |
| 6 | 1.569770 | 76.44 | 1569.63 | 1811.64 | 64.96 | 130.36 | 34.70 | 36.89 | 800.56 | 67.78 |
| 7 | 1.547817 | 77.53 | 1547.70 | 1811.66 | 64.96 | 129.89 | 33.00 | 37.74 | 885.00 | 68.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.03.
Separate latency-pass duration: 1.71 s; not used for throughput.

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

9.95% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.650052 | 72.72 | 1649.98 | 1868.18 | 64.94 | 123.16 | 33.89 | 34.88 | 709.29 | 62.86 |
| 2 | 1.726960 | 69.49 | 1726.87 | 1762.90 | 64.94 | 122.09 | 34.83 | 35.39 | 780.00 | 63.76 |
| 3 | 1.614360 | 74.33 | 1614.29 | 1820.20 | 64.94 | 127.04 | 30.32 | 36.26 | 908.57 | 64.96 |
| 4 | 1.566673 | 76.60 | 1566.60 | 1759.99 | 64.94 | 127.72 | 34.11 | 36.31 | 782.78 | 65.37 |
| 5 | 1.577761 | 76.06 | 1577.69 | 1820.36 | 64.94 | 126.79 | 31.52 | 36.90 | 897.22 | 66.89 |
| 6 | 1.660023 | 72.29 | 1659.88 | 1759.92 | 64.94 | 128.01 | 34.52 | 36.24 | 750.52 | 67.07 |
| 7 | 1.661958 | 72.20 | 1661.87 | 1811.63 | 64.94 | 128.03 | 32.32 | 37.04 | 761.79 | 68.04 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.03.
Separate latency-pass duration: 1.66 s; not used for throughput.

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

10.22% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.576646 | 76.11 | 1576.54 | 1758.35 | 56.35 | 128.13 | 35.07 | 35.85 | 795.00 | 63.00 |
| 2 | 1.624575 | 73.87 | 1624.47 | 1762.11 | 56.35 | 126.04 | 33.82 | 35.71 | 748.39 | 63.04 |
| 3 | 2.581894 | 46.48 | 2581.72 | 1759.77 | 56.35 | 98.77 | 25.31 | 34.09 | 629.23 | 65.00 |
| 4 | 1.621715 | 74.00 | 1621.62 | 1760.00 | 56.35 | 127.98 | 35.43 | 35.90 | 758.57 | 65.07 |
| 5 | 1.604716 | 74.78 | 1604.54 | 1759.79 | 56.35 | 127.01 | 34.68 | 36.45 | 766.61 | 66.96 |
| 6 | 1.679898 | 71.43 | 1679.80 | 1759.90 | 56.35 | 128.03 | 32.79 | 36.31 | 776.79 | 67.00 |
| 7 | 1.665257 | 72.06 | 1665.10 | 1760.56 | 56.35 | 126.91 | 34.48 | 36.70 | 751.55 | 68.72 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.05.
Separate latency-pass duration: 1.65 s; not used for throughput.

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

10.22% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.577822 | 76.05 | 32.61 | 1760.86 | 56.35 | 127.35 | 35.44 | 35.50 | 747.78 | 63.00 |
| 2 | 1.616089 | 74.25 | 36.59 | 1762.00 | 56.35 | 127.56 | 31.21 | 35.52 | 855.00 | 63.00 |
| 3 | 1.618355 | 74.15 | 35.23 | 1759.88 | 56.35 | 127.84 | 36.43 | 35.80 | 699.64 | 65.00 |
| 4 | 1.707751 | 70.27 | 33.72 | 1759.99 | 56.35 | 124.99 | 31.59 | 35.84 | 778.45 | 65.00 |
| 5 | 1.922516 | 62.42 | 36.16 | 1759.81 | 56.35 | 117.53 | 29.56 | 35.74 | 742.97 | 67.00 |
| 6 | 2.322085 | 51.68 | 48.15 | 1759.90 | 56.35 | 104.37 | 27.50 | 34.57 | 602.92 | 67.00 |
| 7 | 1.648823 | 72.78 | 35.49 | 1760.59 | 56.35 | 127.50 | 32.50 | 36.88 | 789.64 | 68.96 |

Ready result delivery-interval P50/P95/P99 ms: 13.32/16.41/26.14.
Separate latency-pass duration: 1.64 s; not used for throughput.

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

Status: **complete**. Load 0.70 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 58.15 ± 11.50 | 2063.62 | 1865.13/1865.13/1865.13 | 1818.71 | 62.83/90.00 | 120/61 | PASS |
| raw-stream | 75.21 ± 6.04 | 28.21 | 2278.74/2278.74/2278.74 | 1818.71 | 62.83/90.00 | 120/61 | PASS |
| native-non-stream | 73.54 ± 1.91 | 1631.75 | 1666.84/1666.84/1666.84 | 1818.70 | 62.83/90.00 | 120/61 | PASS |
| native-stream | 68.69 ± 1.10 | 1746.94 | 1712.77/1712.77/1712.77 | 1816.73 | 62.80/90.00 | 120/61 | PASS |
| managed-non-stream | 68.25 ± 2.99 | 1758.21 | 1663.40/1663.40/1663.40 | 1764.91 | 55.57/90.00 | 120/61 | PASS |
| managed-stream | 53.49 ± 10.84 | 36.51 | 1701.28/1701.28/1701.28 | 1764.90 | 55.57/90.00 | 120/61 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.647596 | 72.83 | 1647.51 | 1813.01 | 62.83 | 132.23 | 34.59 | 37.30 | 783.33 | 70.96 |
| 2 | 2.494670 | 48.10 | 2494.57 | 1816.70 | 62.83 | 99.22 | 25.92 | 36.00 | 706.15 | 72.00 |
| 3 | 1.584139 | 75.75 | 1584.06 | 1816.68 | 62.83 | 129.09 | 36.41 | 37.78 | 771.67 | 72.00 |
| 4 | 2.063733 | 58.15 | 2063.62 | 1818.71 | 62.83 | 116.40 | 26.91 | 37.19 | 851.47 | 73.91 |
| 5 | 2.297029 | 52.24 | 2296.91 | 1818.69 | 62.83 | 106.38 | 30.14 | 36.54 | 598.33 | 73.97 |
| 6 | 1.610098 | 74.53 | 1610.00 | 1766.75 | 62.83 | 126.88 | 33.57 | 38.57 | 867.86 | 75.00 |
| 7 | 2.121465 | 56.56 | 2121.34 | 1766.76 | 62.83 | 108.14 | 27.41 | 37.71 | 827.65 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.03.
Separate latency-pass duration: 1.87 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.9331117333573882, 'inference': 10.555736191652917, 'postprocess': 0.9354090250023243}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-4.80% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.583465 | 75.78 | 27.16 | 1812.98 | 62.83 | 128.49 | 33.70 | 37.63 | 834.44 | 71.00 |
| 2 | 1.594578 | 75.26 | 28.21 | 1816.62 | 62.83 | 134.19 | 33.15 | 38.07 | 845.56 | 72.00 |
| 3 | 1.609453 | 74.56 | 31.70 | 1816.68 | 62.83 | 128.59 | 33.81 | 37.96 | 853.33 | 72.26 |
| 4 | 1.563612 | 76.75 | 27.06 | 1818.68 | 62.83 | 129.02 | 33.85 | 38.11 | 857.78 | 73.85 |
| 5 | 1.629052 | 73.66 | 32.70 | 1818.71 | 62.83 | 128.52 | 35.57 | 38.00 | 817.50 | 74.00 |
| 6 | 1.595517 | 75.21 | 26.79 | 1766.73 | 62.83 | 127.06 | 33.37 | 38.45 | 855.56 | 75.00 |
| 7 | 2.019584 | 59.42 | 44.15 | 1766.73 | 62.83 | 115.66 | 33.28 | 37.31 | 652.97 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 17.85/29.35/35.42.
Separate latency-pass duration: 2.28 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.9151721250106373, 'inference': 10.074391633353722, 'postprocess': 0.9038748083109264}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

-18.75% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.612105 | 74.44 | 1612.03 | 1813.01 | 62.83 | 128.58 | 34.00 | 37.70 | 859.29 | 71.00 |
| 2 | 1.664218 | 72.11 | 1664.13 | 1816.52 | 62.83 | 130.10 | 35.50 | 37.66 | 817.50 | 72.00 |
| 3 | 1.631850 | 73.54 | 1631.75 | 1816.70 | 62.83 | 127.78 | 34.11 | 38.04 | 855.54 | 72.89 |
| 4 | 1.605097 | 74.76 | 1605.02 | 1818.70 | 62.83 | 127.52 | 35.30 | 38.08 | 858.89 | 73.26 |
| 5 | 1.596965 | 75.14 | 1596.88 | 1818.70 | 62.83 | 128.81 | 34.59 | 38.19 | 860.00 | 74.00 |
| 6 | 1.637854 | 73.27 | 1637.77 | 1766.74 | 62.83 | 127.07 | 35.25 | 38.65 | 841.07 | 75.00 |
| 7 | 1.723794 | 69.61 | 1723.69 | 1766.76 | 62.83 | 126.55 | 33.93 | 38.34 | 806.90 | 75.10 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.67 s; not used for throughput.

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

-15.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.743515 | 68.83 | 1743.44 | 1813.16 | 62.80 | 125.58 | 31.27 | 37.47 | 833.00 | 71.00 |
| 2 | 1.767586 | 67.89 | 1767.51 | 1706.79 | 62.80 | 124.94 | 32.27 | 37.29 | 822.50 | 72.00 |
| 3 | 1.765421 | 67.97 | 1765.35 | 1816.73 | 62.80 | 125.74 | 29.10 | 37.63 | 861.29 | 72.97 |
| 4 | 1.747035 | 68.69 | 1746.94 | 1765.53 | 62.80 | 126.65 | 34.37 | 37.27 | 745.00 | 73.00 |
| 5 | 1.771308 | 67.75 | 1771.24 | 1765.59 | 62.80 | 125.20 | 31.40 | 37.93 | 838.50 | 74.00 |
| 6 | 1.691372 | 70.95 | 1691.29 | 1705.20 | 62.80 | 128.50 | 35.38 | 38.25 | 741.21 | 74.97 |
| 7 | 1.737259 | 69.07 | 1737.19 | 1766.73 | 62.80 | 128.34 | 32.52 | 38.54 | 782.07 | 75.79 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.71 s; not used for throughput.

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

-11.52% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.972023 | 60.85 | 1971.91 | 1702.98 | 55.57 | 118.99 | 32.12 | 36.70 | 748.18 | 71.00 |
| 2 | 1.758327 | 68.25 | 1758.21 | 1706.66 | 55.57 | 126.59 | 33.17 | 37.24 | 757.50 | 71.93 |
| 3 | 1.726706 | 69.50 | 1726.58 | 1764.69 | 55.57 | 126.49 | 34.50 | 37.50 | 773.00 | 73.00 |
| 4 | 1.825924 | 65.72 | 1825.83 | 1764.91 | 55.57 | 126.29 | 34.13 | 37.31 | 720.97 | 73.00 |
| 5 | 1.753242 | 68.44 | 1753.14 | 1705.07 | 55.57 | 125.36 | 35.37 | 37.78 | 722.50 | 74.00 |
| 6 | 1.800189 | 66.66 | 1800.08 | 1705.21 | 55.57 | 127.41 | 36.42 | 37.70 | 676.94 | 74.68 |
| 7 | 1.737179 | 69.08 | 1737.06 | 1705.24 | 55.57 | 126.86 | 35.07 | 38.26 | 728.28 | 75.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.06.
Separate latency-pass duration: 1.66 s; not used for throughput.

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

8.70% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.556241 | 46.94 | 52.06 | 1704.83 | 55.57 | 103.22 | 28.10 | 35.64 | 599.25 | 71.00 |
| 2 | 1.786811 | 67.16 | 35.92 | 1705.89 | 55.57 | 126.56 | 34.23 | 37.28 | 722.50 | 71.43 |
| 3 | 2.248394 | 53.37 | 34.25 | 1764.80 | 55.57 | 108.58 | 27.22 | 36.59 | 760.42 | 73.00 |
| 4 | 2.243206 | 53.49 | 53.77 | 1764.90 | 55.57 | 114.92 | 30.81 | 36.53 | 618.75 | 73.00 |
| 5 | 1.783107 | 67.30 | 36.12 | 1705.12 | 55.57 | 125.45 | 34.35 | 37.67 | 758.71 | 74.00 |
| 6 | 2.823702 | 42.50 | 57.43 | 1705.21 | 55.57 | 97.50 | 24.80 | 36.62 | 663.41 | 74.00 |
| 7 | 1.723421 | 69.63 | 36.51 | 1705.45 | 55.57 | 128.30 | 32.86 | 38.56 | 836.90 | 75.97 |

Ready result delivery-interval P50/P95/P99 ms: 13.97/16.47/25.86.
Separate latency-pass duration: 1.70 s; not used for throughput.

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

Status: **complete**. Load 8.83 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `56c176337add102635b2d994ba56503ffb8885e402543144e2de2dfb0cc0e940`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 76.48 ± 1.43 | 1569.05 | 1591.23/1591.23/1591.23 | 2038.86 | 115.16/140.00 | 120/61 | PASS |
| raw-stream | 77.43 ± 13.53 | 27.14 | 1534.54/1534.54/1534.54 | 1986.65 | 115.16/140.00 | 120/61 | PASS |
| native-non-stream | 70.81 ± 9.96 | 1694.55 | 1587.81/1587.81/1587.81 | 1986.54 | 115.16/140.00 | 120/61 | PASS |
| native-stream | 61.24 ± 8.44 | 1959.50 | 1700.51/1700.51/1700.51 | 1983.24 | 115.13/140.00 | 120/61 | PASS |
| managed-non-stream | 71.48 ± 9.26 | 1678.72 | 1679.29/1679.29/1679.29 | 1933.24 | 107.97/140.00 | 120/61 | PASS |
| managed-stream | 70.91 ± 5.83 | 34.40 | 2604.56/2604.56/2604.56 | 1932.18 | 107.97/140.00 | 120/61 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.606990 | 74.67 | 1606.88 | 1979.57 | 115.16 | 129.75 | 43.65 | 67.14 | 1384.04 | 81.96 |
| 2 | 1.569127 | 76.48 | 1569.05 | 2038.86 | 115.16 | 131.26 | 40.85 | 65.80 | 1427.78 | 78.89 |
| 3 | 1.533999 | 78.23 | 1533.92 | 1981.25 | 115.16 | 130.73 | 42.85 | 63.89 | 1415.19 | 78.00 |
| 4 | 1.535448 | 78.15 | 1535.37 | 1983.02 | 115.16 | 130.52 | 42.08 | 64.32 | 1460.77 | 74.96 |
| 5 | 1.561331 | 76.86 | 1561.25 | 1983.00 | 115.16 | 126.81 | 42.00 | 66.57 | 1462.78 | 74.96 |
| 6 | 1.598050 | 75.09 | 1597.95 | 1983.26 | 115.16 | 130.83 | 41.89 | 63.78 | 1411.67 | 75.59 |
| 7 | 1.591558 | 75.40 | 1591.47 | 1983.23 | 115.16 | 128.37 | 39.85 | 64.68 | 1471.11 | 75.96 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.59 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8842490333099704, 'inference': 9.954931849999108, 'postprocess': 0.9074904416290034}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.33% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.316934 | 51.79 | 41.24 | 1981.56 | 115.16 | 99.74 | 29.14 | 60.34 | 1407.57 | 82.43 |
| 2 | 1.549797 | 77.43 | 27.14 | 1986.65 | 115.16 | 130.08 | 42.00 | 64.29 | 1403.08 | 78.96 |
| 3 | 1.578952 | 76.00 | 27.00 | 1982.14 | 115.16 | 129.56 | 40.93 | 63.46 | 1432.78 | 77.96 |
| 4 | 1.530314 | 78.42 | 26.71 | 1982.91 | 115.16 | 131.16 | 40.65 | 63.53 | 1457.88 | 74.96 |
| 5 | 1.518071 | 79.05 | 26.79 | 1983.00 | 115.16 | 129.51 | 42.42 | 66.14 | 1456.73 | 74.96 |
| 6 | 2.476509 | 48.46 | 43.66 | 1984.18 | 115.16 | 103.91 | 27.00 | 56.32 | 1376.92 | 75.38 |
| 7 | 1.543842 | 77.73 | 27.15 | 1983.23 | 115.16 | 130.27 | 40.12 | 67.73 | 1447.50 | 76.04 |

Ready result delivery-interval P50/P95/P99 ms: 12.31/15.13/26.46.
Separate latency-pass duration: 1.53 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 2.73611849994874, 'inference': 13.96794791667162, 'postprocess': 1.270183274914416}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

8.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.607260 | 74.66 | 1607.17 | 1981.55 | 115.16 | 129.42 | 42.96 | 62.18 | 1327.78 | 82.74 |
| 2 | 1.694622 | 70.81 | 1694.55 | 1986.54 | 115.16 | 129.26 | 40.39 | 65.07 | 1431.96 | 79.00 |
| 3 | 2.315630 | 51.82 | 2315.52 | 1982.23 | 115.16 | 102.07 | 30.32 | 60.47 | 1469.19 | 77.43 |
| 4 | 2.128561 | 56.38 | 2128.46 | 1982.90 | 115.16 | 111.87 | 35.74 | 53.33 | 1230.86 | 74.26 |
| 5 | 1.533968 | 78.23 | 1533.89 | 1983.00 | 115.16 | 130.58 | 44.42 | 64.68 | 1446.35 | 74.96 |
| 6 | 1.761393 | 68.13 | 1761.28 | 1984.19 | 115.16 | 123.51 | 40.31 | 66.13 | 1483.97 | 75.69 |
| 7 | 1.611108 | 74.48 | 1611.03 | 1983.25 | 115.16 | 129.01 | 41.56 | 65.84 | 1447.22 | 76.56 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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


### native-stream

27.74% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.668680 | 71.91 | 1668.61 | 1981.61 | 115.13 | 127.97 | 39.64 | 64.64 | 1416.43 | 82.96 |
| 2 | 2.191174 | 54.77 | 2191.01 | 1933.31 | 115.13 | 112.26 | 34.72 | 54.55 | 1231.67 | 79.19 |
| 3 | 1.959567 | 61.24 | 1959.50 | 1982.24 | 115.13 | 122.35 | 34.50 | 57.30 | 1350.94 | 76.16 |
| 4 | 2.128304 | 56.38 | 2128.20 | 1873.39 | 115.13 | 111.79 | 31.91 | 62.07 | 1517.57 | 75.34 |
| 5 | 2.298658 | 52.20 | 2298.57 | 1982.98 | 115.13 | 104.97 | 28.82 | 63.19 | 1560.79 | 74.97 |
| 6 | 1.669470 | 71.88 | 1669.35 | 1873.24 | 115.13 | 128.04 | 39.66 | 66.05 | 1490.17 | 75.03 |
| 7 | 1.731745 | 69.29 | 1731.67 | 1983.24 | 115.13 | 127.75 | 36.10 | 66.82 | 1488.62 | 76.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.70 s; not used for throughput.

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

8.81% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.678831 | 71.48 | 1678.72 | 1929.54 | 107.97 | 127.91 | 41.31 | 63.54 | 1400.17 | 82.55 |
| 2 | 2.143760 | 55.98 | 2143.64 | 1933.24 | 107.97 | 109.74 | 31.91 | 65.09 | 1500.43 | 81.00 |
| 3 | 1.669099 | 71.90 | 1669.00 | 1873.20 | 107.97 | 129.15 | 38.39 | 65.58 | 1471.61 | 76.14 |
| 4 | 1.665095 | 72.07 | 1665.00 | 1873.39 | 107.97 | 129.13 | 38.86 | 64.87 | 1484.46 | 75.96 |
| 5 | 2.111890 | 56.82 | 2111.79 | 1873.10 | 107.97 | 114.38 | 34.77 | 52.82 | 1244.14 | 73.80 |
| 6 | 1.639911 | 73.17 | 1639.81 | 1873.25 | 107.97 | 128.98 | 39.64 | 65.37 | 1483.93 | 74.96 |
| 7 | 2.291551 | 52.37 | 2291.43 | 1874.16 | 107.97 | 107.04 | 28.39 | 63.68 | 1539.08 | 77.03 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
Separate latency-pass duration: 1.68 s; not used for throughput.

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

7.66% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.709890 | 70.18 | 34.40 | 1932.02 | 107.97 | 126.86 | 39.76 | 62.62 | 1405.34 | 81.97 |
| 2 | 1.655717 | 72.48 | 33.89 | 1932.18 | 107.97 | 129.12 | 38.82 | 65.05 | 1409.46 | 81.43 |
| 3 | 1.696197 | 70.75 | 35.11 | 1873.31 | 107.97 | 128.87 | 37.90 | 64.41 | 1480.34 | 75.97 |
| 4 | 1.692342 | 70.91 | 35.71 | 1873.39 | 107.97 | 128.24 | 35.82 | 64.64 | 1495.71 | 75.96 |
| 5 | 1.680961 | 71.39 | 34.12 | 1873.16 | 107.97 | 128.16 | 38.00 | 66.29 | 1498.97 | 74.86 |
| 6 | 1.645544 | 72.92 | 33.95 | 1873.25 | 107.97 | 128.51 | 36.89 | 65.07 | 1487.14 | 74.93 |
| 7 | 2.134300 | 56.22 | 46.29 | 1874.37 | 107.97 | 113.44 | 34.40 | 52.78 | 1227.43 | 76.37 |

Ready result delivery-interval P50/P95/P99 ms: 20.56/28.07/30.24.
Separate latency-pass duration: 2.60 s; not used for throughput.

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
