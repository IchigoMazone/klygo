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
| yolo | 96.40 | 101.82 | 97.23 | 92.83 | 92.76 | 89.86 |
| yolo11 | 78.38 | 79.63 | 79.96 | 74.62 | 74.91 | 74.67 |
| yolo26 | 72.72 | 75.20 | 71.82 | 68.39 | 67.31 | 57.35 |
| yolo-world | 78.09 | 77.44 | 75.99 | 72.81 | 72.80 | 71.14 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 96.40 ± 14.58 | 1244.73 | 1170.08/1170.08/1170.08 | 1722.37 | 34.27/62.00 | 120/50 | PASS |
| raw-stream | 101.82 ± 13.69 | 24.00 | 1201.61/1201.61/1201.61 | 1722.31 | 34.27/62.00 | 120/50 | PASS |
| native-non-stream | 97.23 ± 3.37 | 1234.11 | 1299.33/1299.33/1299.33 | 1722.32 | 34.27/62.00 | 120/50 | PASS |
| native-stream | 92.83 ± 14.79 | 1292.55 | 1964.88/1964.88/1964.88 | 1722.16 | 34.25/62.00 | 120/50 | PASS |
| managed-non-stream | 92.76 ± 12.43 | 1293.58 | 1481.09/1481.09/1481.09 | 1612.34 | 24.66/62.00 | 120/50 | PASS |
| managed-stream | 89.86 ± 9.03 | 31.62 | 1330.14/1330.14/1330.14 | 1612.34 | 24.66/62.00 | 120/50 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.168290 | 102.71 | 1168.16 | 1713.90 | 34.27 | 137.20 | 33.10 | 41.94 | 1044.00 | 74.00 |
| 2 | 1.468311 | 81.73 | 1468.17 | 1721.72 | 34.27 | 119.60 | 28.83 | 41.79 | 1038.12 | 75.67 |
| 3 | 1.900667 | 63.14 | 1900.56 | 1721.73 | 34.27 | 101.16 | 27.24 | 38.16 | 767.59 | 75.10 |
| 4 | 1.244809 | 96.40 | 1244.73 | 1722.16 | 34.27 | 131.92 | 32.71 | 42.77 | 1040.00 | 77.00 |
| 5 | 1.173380 | 102.27 | 1173.29 | 1722.16 | 34.27 | 134.52 | 31.80 | 44.07 | 1065.00 | 77.00 |
| 6 | 1.228920 | 97.65 | 1228.83 | 1722.37 | 34.27 | 137.60 | 32.95 | 42.81 | 1002.14 | 78.00 |
| 7 | 1.477052 | 81.24 | 1476.95 | 1719.75 | 34.27 | 120.31 | 29.48 | 41.63 | 988.80 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.17 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8915401416279565, 'inference': 6.807513649975287, 'postprocess': 0.778020008351632}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-2.24% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.152341 | 104.14 | 24.00 | 1716.70 | 34.27 | 134.58 | 32.75 | 43.17 | 1062.00 | 74.00 |
| 2 | 1.155043 | 103.89 | 24.40 | 1721.69 | 34.27 | 134.56 | 33.40 | 42.50 | 1038.75 | 75.50 |
| 3 | 1.197087 | 100.24 | 23.22 | 1721.73 | 34.27 | 134.93 | 37.60 | 41.77 | 910.50 | 75.95 |
| 4 | 1.178583 | 101.82 | 23.46 | 1722.13 | 34.27 | 135.20 | 32.55 | 43.07 | 1012.50 | 76.95 |
| 5 | 1.167836 | 102.75 | 23.14 | 1722.16 | 34.27 | 133.74 | 30.80 | 44.14 | 1077.00 | 77.30 |
| 6 | 1.201453 | 99.88 | 24.26 | 1722.31 | 34.27 | 134.23 | 32.30 | 42.96 | 996.75 | 78.00 |
| 7 | 1.813652 | 66.16 | 34.66 | 1719.73 | 34.27 | 101.37 | 26.54 | 39.62 | 802.50 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 9.77/12.18/14.85.
Separate latency-pass duration: 1.20 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8233839000079872, 'inference': 6.6945495332826495, 'postprocess': 0.7970826667209016}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

1.61% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.232891 | 97.33 | 1232.81 | 1716.76 | 34.27 | 133.57 | 34.33 | 42.62 | 1037.86 | 74.00 |
| 2 | 1.214537 | 98.80 | 1214.44 | 1721.59 | 34.27 | 135.66 | 32.86 | 42.86 | 1047.14 | 75.05 |
| 3 | 1.234205 | 97.23 | 1234.11 | 1721.76 | 34.27 | 136.68 | 33.10 | 42.17 | 1035.00 | 76.00 |
| 4 | 1.322804 | 90.72 | 1322.72 | 1722.14 | 34.27 | 131.55 | 35.73 | 40.62 | 867.27 | 76.86 |
| 5 | 1.202798 | 99.77 | 1202.73 | 1722.18 | 34.27 | 134.30 | 34.86 | 43.29 | 1030.71 | 77.62 |
| 6 | 1.248739 | 96.10 | 1248.65 | 1722.32 | 34.27 | 135.56 | 35.10 | 42.03 | 970.00 | 78.00 |
| 7 | 1.302498 | 92.13 | 1302.40 | 1719.76 | 34.27 | 131.73 | 37.50 | 42.22 | 927.95 | 78.00 |

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

8.38% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.756414 | 68.32 | 1756.28 | 1716.90 | 34.25 | 108.41 | 23.96 | 39.28 | 965.89 | 74.00 |
| 2 | 1.275603 | 94.07 | 1275.52 | 1611.56 | 34.25 | 130.92 | 32.50 | 41.51 | 1005.00 | 75.00 |
| 3 | 1.292624 | 92.83 | 1292.55 | 1721.79 | 34.25 | 131.21 | 31.36 | 41.52 | 989.32 | 76.00 |
| 4 | 2.042087 | 58.76 | 2041.98 | 1612.20 | 34.25 | 98.20 | 25.90 | 38.40 | 748.55 | 76.03 |
| 5 | 1.285222 | 93.37 | 1285.09 | 1722.16 | 34.25 | 130.68 | 27.91 | 43.22 | 1050.68 | 77.86 |
| 6 | 1.331915 | 90.10 | 1331.83 | 1612.38 | 34.25 | 132.68 | 31.52 | 41.63 | 967.17 | 78.00 |
| 7 | 1.258391 | 95.36 | 1258.26 | 1719.74 | 34.25 | 132.90 | 31.14 | 42.60 | 1008.41 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.96 s; not used for throughput.

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
| 1 | 1.708889 | 70.22 | 1708.78 | 1607.07 | 24.66 | 112.90 | 30.67 | 38.37 | 725.56 | 74.04 |
| 2 | 1.248531 | 96.11 | 1248.39 | 1610.79 | 24.66 | 131.66 | 31.62 | 41.84 | 1026.43 | 75.00 |
| 3 | 1.293728 | 92.76 | 1293.58 | 1611.92 | 24.66 | 130.69 | 31.18 | 41.42 | 991.36 | 76.00 |
| 4 | 1.374329 | 87.32 | 1374.20 | 1612.15 | 24.66 | 129.13 | 30.57 | 41.48 | 979.57 | 76.39 |
| 5 | 1.822806 | 65.83 | 1822.68 | 1612.23 | 24.66 | 108.37 | 24.66 | 40.16 | 963.62 | 77.79 |
| 6 | 1.278637 | 93.85 | 1278.52 | 1612.34 | 24.66 | 131.82 | 33.18 | 41.77 | 970.91 | 77.95 |
| 7 | 1.279038 | 93.82 | 1278.94 | 1609.88 | 24.66 | 132.58 | 33.18 | 42.06 | 964.77 | 78.00 |

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

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Sequential video frame indices=True.


### managed-stream

7.27% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.308708 | 91.69 | 31.95 | 1609.56 | 24.66 | 130.96 | 32.50 | 40.50 | 957.27 | 74.86 |
| 2 | 1.278215 | 93.88 | 29.98 | 1610.69 | 24.66 | 130.83 | 30.18 | 41.96 | 1014.55 | 74.95 |
| 3 | 1.368507 | 87.69 | 30.98 | 1612.04 | 24.66 | 128.73 | 32.91 | 40.54 | 904.57 | 76.00 |
| 4 | 1.335354 | 89.86 | 30.69 | 1612.15 | 24.66 | 129.46 | 31.43 | 41.60 | 969.78 | 76.00 |
| 5 | 1.765792 | 67.96 | 39.88 | 1612.27 | 24.66 | 109.94 | 33.25 | 39.00 | 662.14 | 77.61 |
| 6 | 1.346018 | 89.15 | 31.62 | 1612.34 | 24.66 | 131.35 | 32.35 | 40.54 | 934.57 | 77.96 |
| 7 | 1.277728 | 93.92 | 32.59 | 1609.88 | 24.66 | 130.98 | 32.68 | 42.44 | 995.45 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 10.54/13.66/23.84.
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

## Detail — yolo11, video, batch 1

Status: **complete**. Load 0.12 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 78.38 ± 9.74 | 1530.99 | 1489.45/1489.45/1489.45 | 1748.62 | 64.94/90.00 | 120/36 | PASS |
| raw-stream | 79.63 ± 12.38 | 26.97 | 2316.65/2316.65/2316.65 | 1748.62 | 64.94/90.00 | 120/36 | PASS |
| native-non-stream | 79.96 ± 5.56 | 1500.68 | 1563.35/1563.35/1563.35 | 1748.64 | 64.94/90.00 | 120/36 | PASS |
| native-stream | 74.62 ± 12.60 | 1608.11 | 1570.35/1570.35/1570.35 | 1748.62 | 64.94/90.00 | 120/36 | PASS |
| managed-non-stream | 74.91 ± 1.57 | 1601.77 | 1573.59/1573.59/1573.59 | 1696.46 | 56.35/90.00 | 120/36 | PASS |
| managed-stream | 74.67 ± 10.45 | 32.61 | 1588.77/1588.77/1588.77 | 1696.49 | 56.35/90.00 | 120/36 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.531079 | 78.38 | 1530.99 | 1742.00 | 64.94 | 134.81 | 35.73 | 39.55 | 751.15 | 77.00 |
| 2 | 1.554261 | 77.21 | 1554.19 | 1748.05 | 64.94 | 130.89 | 34.38 | 39.55 | 796.73 | 76.00 |
| 3 | 1.549027 | 77.47 | 1548.94 | 1748.04 | 64.94 | 129.76 | 33.31 | 39.61 | 806.54 | 76.00 |
| 4 | 1.489800 | 80.55 | 1489.72 | 1748.45 | 64.94 | 129.71 | 35.24 | 39.71 | 789.60 | 77.00 |
| 5 | 1.461928 | 82.08 | 1461.83 | 1748.48 | 64.94 | 130.99 | 33.36 | 39.78 | 873.60 | 77.00 |
| 6 | 1.438502 | 83.42 | 1438.42 | 1748.62 | 64.94 | 131.11 | 34.12 | 40.07 | 891.25 | 77.00 |
| 7 | 2.187697 | 54.85 | 2187.58 | 1748.62 | 64.94 | 100.51 | 25.82 | 38.35 | 760.00 | 77.00 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.9551719999905497, 'inference': 9.685654766637223, 'postprocess': 0.8103924667087389}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.50% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.025392 | 59.25 | 27.38 | 1742.90 | 64.94 | 107.65 | 27.13 | 38.35 | 794.52 | 77.00 |
| 2 | 1.546486 | 77.60 | 27.01 | 1747.99 | 64.94 | 132.04 | 30.96 | 39.92 | 834.23 | 76.00 |
| 3 | 1.507055 | 79.63 | 26.51 | 1748.07 | 64.94 | 130.89 | 35.23 | 39.51 | 804.23 | 76.04 |
| 4 | 2.203063 | 54.47 | 37.83 | 1748.42 | 64.94 | 100.06 | 27.42 | 38.32 | 703.18 | 77.00 |
| 5 | 1.476606 | 81.27 | 26.97 | 1748.45 | 64.94 | 130.15 | 33.68 | 40.01 | 871.20 | 77.00 |
| 6 | 1.416937 | 84.69 | 24.64 | 1748.59 | 64.94 | 130.21 | 32.79 | 40.67 | 885.00 | 77.00 |
| 7 | 1.420233 | 84.49 | 25.07 | 1748.62 | 64.94 | 130.77 | 37.67 | 40.06 | 773.12 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 17.59/26.53/34.94.
Separate latency-pass duration: 2.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 2.4036115249903864, 'inference': 12.402533608330183, 'postprocess': 0.9871418332447016}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.19% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.802022 | 66.59 | 1801.94 | 1742.96 | 64.94 | 117.87 | 34.66 | 38.57 | 682.76 | 77.00 |
| 2 | 1.649422 | 72.75 | 1649.28 | 1747.93 | 64.94 | 127.91 | 33.50 | 39.31 | 762.32 | 76.00 |
| 3 | 1.512832 | 79.32 | 1512.73 | 1748.06 | 64.94 | 130.31 | 33.42 | 39.34 | 842.88 | 76.27 |
| 4 | 1.476938 | 81.25 | 1476.86 | 1748.43 | 64.94 | 132.72 | 36.16 | 39.77 | 831.60 | 77.00 |
| 5 | 1.493915 | 80.33 | 1493.82 | 1748.48 | 64.94 | 130.42 | 34.36 | 39.94 | 843.60 | 77.00 |
| 6 | 1.500808 | 79.96 | 1500.68 | 1748.60 | 64.94 | 131.71 | 35.27 | 39.82 | 868.85 | 77.00 |
| 7 | 1.480448 | 81.06 | 1480.35 | 1748.64 | 64.94 | 129.75 | 34.28 | 40.33 | 882.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
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

7.95% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.588503 | 75.54 | 1588.42 | 1743.09 | 64.94 | 128.25 | 31.85 | 39.05 | 836.67 | 76.96 |
| 2 | 2.413521 | 49.72 | 2413.42 | 1637.88 | 64.94 | 100.54 | 25.62 | 37.50 | 676.62 | 76.08 |
| 3 | 2.381453 | 50.39 | 2381.35 | 1748.09 | 64.94 | 99.64 | 25.58 | 37.46 | 683.75 | 76.00 |
| 4 | 1.608190 | 74.62 | 1608.11 | 1638.49 | 64.94 | 129.57 | 32.52 | 39.57 | 842.78 | 77.00 |
| 5 | 1.575620 | 76.16 | 1575.43 | 1748.45 | 64.94 | 130.53 | 32.22 | 39.73 | 845.56 | 77.00 |
| 6 | 1.617670 | 74.18 | 1617.60 | 1638.67 | 64.94 | 125.84 | 31.67 | 39.46 | 833.33 | 77.00 |
| 7 | 1.535220 | 78.16 | 1535.14 | 1748.62 | 64.94 | 128.20 | 32.23 | 39.98 | 852.12 | 77.15 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
Separate latency-pass duration: 1.57 s; not used for throughput.

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

4.18% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.570120 | 76.43 | 1570.02 | 1633.25 | 56.35 | 129.30 | 34.62 | 39.53 | 792.69 | 77.00 |
| 2 | 1.619165 | 74.11 | 1619.05 | 1637.10 | 56.35 | 128.90 | 33.54 | 39.43 | 791.25 | 76.89 |
| 3 | 1.626366 | 73.78 | 1626.26 | 1638.26 | 56.35 | 125.70 | 33.25 | 39.03 | 770.36 | 76.68 |
| 4 | 1.551452 | 77.35 | 1551.31 | 1638.44 | 56.35 | 129.15 | 32.00 | 39.80 | 858.46 | 77.00 |
| 5 | 1.601876 | 74.91 | 1601.77 | 1638.50 | 56.35 | 129.56 | 33.74 | 39.40 | 788.33 | 77.00 |
| 6 | 1.585926 | 75.67 | 1585.83 | 1638.61 | 56.35 | 129.92 | 34.15 | 39.27 | 791.67 | 77.00 |
| 7 | 1.646613 | 72.88 | 1646.47 | 1696.46 | 56.35 | 127.85 | 31.54 | 39.36 | 821.79 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
Separate latency-pass duration: 1.57 s; not used for throughput.

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

2.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.566590 | 76.60 | 32.61 | 1635.75 | 56.35 | 129.11 | 33.42 | 39.42 | 837.69 | 77.00 |
| 2 | 1.607099 | 74.67 | 32.34 | 1636.05 | 56.35 | 128.45 | 32.15 | 39.36 | 820.56 | 76.96 |
| 3 | 1.563131 | 76.77 | 33.39 | 1638.36 | 56.35 | 128.61 | 32.37 | 39.42 | 833.89 | 76.93 |
| 4 | 1.526167 | 78.63 | 31.83 | 1638.44 | 56.35 | 129.41 | 32.62 | 39.67 | 826.15 | 76.96 |
| 5 | 2.477551 | 48.43 | 33.65 | 1638.53 | 56.35 | 99.93 | 27.49 | 37.46 | 629.23 | 77.00 |
| 6 | 1.733550 | 69.22 | 58.25 | 1638.61 | 56.35 | 124.95 | 33.83 | 38.72 | 699.83 | 77.00 |
| 7 | 1.633136 | 73.48 | 32.42 | 1696.49 | 56.35 | 127.72 | 34.07 | 39.20 | 781.07 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.61/15.85/32.06.
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

## Detail — yolo26, video, batch 1

Status: **complete**. Load 0.22 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 72.72 ± 8.97 | 1650.10 | 1689.49/1689.49/1689.49 | 1798.97 | 62.91/90.00 | 120/35 | PASS |
| raw-stream | 75.20 ± 10.25 | 28.15 | 2158.41/2158.41/2158.41 | 1798.05 | 62.91/90.00 | 120/35 | PASS |
| native-non-stream | 71.82 ± 2.25 | 1670.84 | 1999.58/1999.58/1999.58 | 1798.04 | 62.91/90.00 | 120/35 | PASS |
| native-stream | 68.39 ± 1.21 | 1754.57 | 1847.05/1847.05/1847.05 | 1798.02 | 62.89/90.00 | 120/35 | PASS |
| managed-non-stream | 67.31 ± 6.40 | 1782.68 | 1739.54/1739.54/1739.54 | 1688.62 | 55.57/90.00 | 120/35 | PASS |
| managed-stream | 57.35 ± 9.35 | 36.26 | 1781.78/1781.78/1781.78 | 1688.69 | 55.57/90.00 | 120/35 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.610798 | 74.50 | 1610.71 | 1788.56 | 62.91 | 130.87 | 38.78 | 39.29 | 739.44 | 77.00 |
| 2 | 1.909253 | 62.85 | 1909.15 | 1794.29 | 62.91 | 117.01 | 29.97 | 39.02 | 845.81 | 77.42 |
| 3 | 2.177064 | 55.12 | 2176.94 | 1794.39 | 62.91 | 107.37 | 30.66 | 38.24 | 640.71 | 77.11 |
| 4 | 1.624478 | 73.87 | 1624.39 | 1795.45 | 62.91 | 128.51 | 34.14 | 39.36 | 832.50 | 77.14 |
| 5 | 2.154554 | 55.70 | 2154.41 | 1795.42 | 62.91 | 109.76 | 27.68 | 38.44 | 771.18 | 77.12 |
| 6 | 1.604392 | 74.79 | 1604.31 | 1798.97 | 62.91 | 130.04 | 35.26 | 39.05 | 819.44 | 77.00 |
| 7 | 1.650190 | 72.72 | 1650.10 | 1798.97 | 62.91 | 129.66 | 33.82 | 39.05 | 827.14 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.04.
Separate latency-pass duration: 1.69 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.929971425003411, 'inference': 10.46093627502008, 'postprocess': 0.743129041628284}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.77% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.582680 | 75.82 | 27.08 | 1790.41 | 62.91 | 129.95 | 34.44 | 39.30 | 830.56 | 77.00 |
| 2 | 1.574368 | 76.22 | 27.24 | 1794.29 | 62.91 | 129.88 | 33.41 | 39.56 | 856.11 | 77.19 |
| 3 | 1.611823 | 74.45 | 28.15 | 1794.37 | 62.91 | 129.51 | 35.22 | 39.69 | 795.00 | 77.67 |
| 4 | 1.595760 | 75.20 | 28.15 | 1796.37 | 62.91 | 130.22 | 34.52 | 39.35 | 800.56 | 77.00 |
| 5 | 2.059576 | 58.26 | 43.07 | 1795.45 | 62.91 | 113.56 | 32.94 | 38.51 | 666.62 | 77.32 |
| 6 | 1.595748 | 75.20 | 26.34 | 1798.02 | 62.91 | 130.41 | 33.63 | 39.52 | 832.78 | 77.00 |
| 7 | 2.337816 | 51.33 | 29.34 | 1798.05 | 62.91 | 103.45 | 28.44 | 37.72 | 706.67 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 17.35/27.12/29.68.
Separate latency-pass duration: 2.16 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.9182182249854425, 'inference': 10.25316377498863, 'postprocess': 0.7570954583949666}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.20% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.666122 | 72.02 | 1666.02 | 1790.44 | 62.91 | 129.49 | 32.54 | 39.31 | 858.21 | 77.00 |
| 2 | 1.670952 | 71.82 | 1670.84 | 1794.17 | 62.91 | 129.52 | 36.21 | 39.59 | 795.00 | 77.00 |
| 3 | 1.642591 | 73.06 | 1642.52 | 1794.39 | 62.91 | 130.62 | 34.04 | 39.28 | 819.64 | 77.29 |
| 4 | 1.660280 | 72.28 | 1660.17 | 1796.37 | 62.91 | 130.46 | 34.54 | 39.73 | 816.96 | 77.00 |
| 5 | 1.721651 | 69.70 | 1721.56 | 1797.81 | 62.91 | 130.67 | 34.72 | 39.30 | 790.86 | 77.59 |
| 6 | 1.680411 | 71.41 | 1680.31 | 1798.04 | 62.91 | 130.49 | 33.75 | 39.23 | 822.86 | 77.00 |
| 7 | 1.806410 | 66.43 | 1806.32 | 1798.04 | 62.91 | 124.38 | 34.60 | 38.97 | 736.50 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 2.00 s; not used for throughput.

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

5.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.760854 | 68.15 | 1760.72 | 1790.56 | 62.89 | 126.07 | 30.77 | 38.87 | 817.00 | 77.00 |
| 2 | 1.808958 | 66.34 | 1808.87 | 1684.38 | 62.89 | 126.49 | 34.40 | 38.63 | 717.00 | 77.00 |
| 3 | 1.759331 | 68.21 | 1759.26 | 1794.41 | 62.89 | 126.01 | 30.50 | 38.70 | 803.00 | 77.10 |
| 4 | 1.711393 | 70.12 | 1711.32 | 1685.65 | 62.89 | 127.77 | 33.93 | 39.26 | 808.45 | 77.00 |
| 5 | 1.724218 | 69.60 | 1724.14 | 1797.79 | 62.89 | 127.33 | 33.00 | 39.06 | 748.97 | 77.83 |
| 6 | 1.750854 | 68.54 | 1750.78 | 1688.65 | 62.89 | 125.89 | 34.23 | 38.62 | 760.50 | 77.00 |
| 7 | 1.754660 | 68.39 | 1754.57 | 1798.02 | 62.89 | 127.86 | 31.90 | 38.81 | 819.50 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
Separate latency-pass duration: 1.85 s; not used for throughput.

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

8.93% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.754579 | 68.39 | 1754.48 | 1680.39 | 55.57 | 127.77 | 35.31 | 38.67 | 734.48 | 77.00 |
| 2 | 1.842901 | 65.11 | 1842.79 | 1684.28 | 55.57 | 127.55 | 33.35 | 38.64 | 752.90 | 77.00 |
| 3 | 1.763369 | 68.05 | 1763.27 | 1684.88 | 55.57 | 127.90 | 34.07 | 38.98 | 778.00 | 77.03 |
| 4 | 1.782787 | 67.31 | 1782.68 | 1685.08 | 55.57 | 126.49 | 34.60 | 38.80 | 727.00 | 77.00 |
| 5 | 1.736890 | 69.09 | 1736.78 | 1688.47 | 55.57 | 127.09 | 33.83 | 38.89 | 798.00 | 77.93 |
| 6 | 2.363765 | 50.77 | 2363.67 | 1688.62 | 55.57 | 106.45 | 29.35 | 37.59 | 612.57 | 77.00 |
| 7 | 1.836007 | 65.36 | 1835.84 | 1688.48 | 55.57 | 128.52 | 33.77 | 38.62 | 718.55 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.74 s; not used for throughput.

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

13.93% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.292742 | 52.34 | 34.87 | 1683.34 | 55.57 | 107.14 | 26.08 | 37.62 | 748.33 | 77.00 |
| 2 | 2.097537 | 57.21 | 48.71 | 1683.46 | 55.57 | 115.92 | 31.76 | 37.88 | 669.71 | 77.00 |
| 3 | 1.739379 | 68.99 | 35.43 | 1684.99 | 55.57 | 128.28 | 33.66 | 38.91 | 752.59 | 77.00 |
| 4 | 2.773452 | 43.27 | 64.71 | 1685.08 | 55.57 | 98.78 | 26.64 | 37.23 | 625.36 | 77.00 |
| 5 | 1.733865 | 69.21 | 33.98 | 1688.54 | 55.57 | 127.00 | 32.00 | 39.14 | 798.50 | 77.63 |
| 6 | 2.092253 | 57.35 | 36.26 | 1688.62 | 55.57 | 116.49 | 29.26 | 38.50 | 774.26 | 77.00 |
| 7 | 1.880077 | 63.83 | 40.33 | 1688.69 | 55.57 | 126.97 | 35.31 | 38.27 | 679.69 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 14.42/17.70/29.65.
Separate latency-pass duration: 1.78 s; not used for throughput.

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

Status: **complete**. Load 6.94 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: end-to-end caller-provided video; includes source decoding. Input SHA256: `2e80d8aab05b9fe4e0eb0a0f40b678ffbb2739ddaf8df93f87b7ac3247b22123`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 78.09 ± 11.01 | 1536.54 | 1757.60/1757.60/1757.60 | 2000.77 | 115.15/140.00 | 120/41 | PASS |
| raw-stream | 77.44 ± 5.12 | 26.36 | 2315.75/2315.75/2315.75 | 1953.67 | 115.15/140.00 | 120/41 | PASS |
| native-non-stream | 75.99 ± 10.14 | 1579.16 | 1582.23/1582.23/1582.23 | 1953.64 | 115.15/140.00 | 120/41 | PASS |
| native-stream | 72.81 ± 8.36 | 1648.16 | 1622.99/1622.99/1622.99 | 1953.65 | 115.13/140.00 | 120/41 | PASS |
| managed-non-stream | 72.80 ± 8.97 | 1648.15 | 1616.97/1616.97/1616.97 | 1891.96 | 107.97/140.00 | 120/41 | PASS |
| managed-stream | 71.14 ± 8.15 | 33.32 | 1626.11/1626.11/1626.11 | 1843.21 | 107.97/140.00 | 120/41 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.575960 | 76.14 | 1575.88 | 1944.05 | 115.15 | 132.35 | 42.35 | 63.39 | 1397.88 | 81.96 |
| 2 | 1.536630 | 78.09 | 1536.54 | 2000.77 | 115.15 | 132.42 | 42.77 | 64.16 | 1432.50 | 78.23 |
| 3 | 1.520394 | 78.93 | 1520.32 | 1944.15 | 115.15 | 130.38 | 43.46 | 66.27 | 1422.69 | 78.00 |
| 4 | 2.008178 | 59.76 | 2008.10 | 1948.02 | 115.15 | 109.52 | 32.03 | 59.48 | 1421.72 | 74.94 |
| 5 | 1.532810 | 78.29 | 1532.73 | 1948.03 | 115.15 | 131.42 | 41.50 | 64.57 | 1459.62 | 74.96 |
| 6 | 1.528315 | 78.52 | 1528.22 | 1953.64 | 115.15 | 129.76 | 43.08 | 65.86 | 1445.19 | 75.96 |
| 7 | 2.300055 | 52.17 | 2299.94 | 1953.67 | 115.15 | 98.84 | 28.03 | 60.25 | 1480.54 | 75.68 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/1.59.
Separate latency-pass duration: 1.76 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.970891116676891, 'inference': 10.014138750026783, 'postprocess': 0.7942895000269345}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.06% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.663180 | 72.15 | 50.11 | 1944.05 | 115.15 | 127.71 | 40.71 | 63.40 | 1402.50 | 81.96 |
| 2 | 1.706051 | 70.34 | 41.01 | 1946.69 | 115.15 | 128.32 | 38.79 | 60.65 | 1344.31 | 78.59 |
| 3 | 1.521288 | 78.88 | 25.52 | 1945.29 | 115.15 | 132.23 | 43.73 | 65.30 | 1425.58 | 77.96 |
| 4 | 1.799444 | 66.69 | 25.68 | 1948.05 | 115.15 | 115.10 | 36.13 | 65.13 | 1497.58 | 75.68 |
| 5 | 1.549569 | 77.44 | 38.46 | 1948.03 | 115.15 | 129.76 | 42.15 | 64.79 | 1461.35 | 74.96 |
| 6 | 1.501530 | 79.92 | 26.36 | 1953.61 | 115.15 | 131.97 | 42.32 | 67.05 | 1455.00 | 75.68 |
| 7 | 1.526435 | 78.61 | 26.12 | 1953.67 | 115.15 | 132.23 | 42.96 | 63.70 | 1385.19 | 75.73 |

Ready result delivery-interval P50/P95/P99 ms: 18.06/26.79/34.44.
Separate latency-pass duration: 2.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.9593704749543879, 'inference': 10.572778508283895, 'postprocess': 0.8504081583926867}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.33% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.875361 | 63.99 | 1875.22 | 1944.04 | 115.15 | 118.61 | 38.29 | 65.58 | 1440.48 | 82.10 |
| 2 | 2.316909 | 51.79 | 2316.78 | 1946.56 | 115.15 | 101.29 | 30.43 | 62.81 | 1446.89 | 78.97 |
| 3 | 1.582606 | 75.82 | 1582.52 | 1945.27 | 115.15 | 132.69 | 44.24 | 66.10 | 1428.60 | 77.72 |
| 4 | 1.551885 | 77.33 | 1551.81 | 1948.04 | 115.15 | 133.50 | 43.73 | 66.55 | 1445.77 | 75.96 |
| 5 | 1.540280 | 77.91 | 1540.13 | 1948.04 | 115.15 | 131.72 | 43.92 | 66.41 | 1456.73 | 74.96 |
| 6 | 1.579238 | 75.99 | 1579.16 | 1952.71 | 115.15 | 130.15 | 40.38 | 68.35 | 1461.35 | 75.12 |
| 7 | 1.515718 | 79.17 | 1515.63 | 1953.64 | 115.15 | 132.88 | 42.40 | 66.07 | 1444.80 | 76.00 |

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


### native-stream

7.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.362568 | 50.79 | 2362.50 | 1944.09 | 115.13 | 105.70 | 29.05 | 56.71 | 1332.24 | 81.42 |
| 2 | 1.648228 | 72.81 | 1648.16 | 1836.40 | 115.13 | 129.25 | 39.21 | 64.14 | 1445.89 | 79.93 |
| 3 | 1.757188 | 68.29 | 1757.08 | 1945.30 | 115.13 | 124.43 | 38.43 | 63.09 | 1475.50 | 77.00 |
| 4 | 1.614654 | 74.32 | 1614.59 | 1837.05 | 115.13 | 128.32 | 41.07 | 64.11 | 1458.89 | 75.93 |
| 5 | 1.621567 | 74.00 | 1621.47 | 1948.03 | 115.13 | 129.20 | 40.37 | 64.84 | 1470.00 | 74.96 |
| 6 | 1.707403 | 70.28 | 1707.26 | 1843.13 | 115.13 | 127.24 | 39.62 | 64.69 | 1487.59 | 75.10 |
| 7 | 1.638177 | 73.25 | 1638.06 | 1953.65 | 115.13 | 129.95 | 37.82 | 63.34 | 1476.43 | 76.50 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.03/0.04.
Separate latency-pass duration: 1.62 s; not used for throughput.

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

7.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.692884 | 70.88 | 1692.71 | 1891.96 | 107.97 | 128.30 | 39.97 | 63.85 | 1409.48 | 81.00 |
| 2 | 1.649523 | 72.75 | 1649.43 | 1836.31 | 107.97 | 129.17 | 39.68 | 66.13 | 1424.46 | 79.96 |
| 3 | 2.407695 | 49.84 | 2407.59 | 1834.48 | 107.97 | 102.51 | 27.92 | 57.34 | 1394.61 | 76.53 |
| 4 | 1.579001 | 76.00 | 1578.91 | 1837.06 | 107.97 | 127.61 | 40.48 | 63.93 | 1454.44 | 75.96 |
| 5 | 1.628377 | 73.69 | 1628.28 | 1839.07 | 107.97 | 127.32 | 39.71 | 67.54 | 1497.86 | 75.14 |
| 6 | 1.639278 | 73.20 | 1639.19 | 1843.13 | 107.97 | 129.13 | 37.64 | 65.73 | 1495.71 | 74.96 |
| 7 | 1.648297 | 72.80 | 1648.15 | 1843.18 | 107.97 | 127.74 | 39.79 | 66.60 | 1465.18 | 76.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.06.
Separate latency-pass duration: 1.62 s; not used for throughput.

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

9.10% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.686770 | 71.14 | 33.08 | 1836.04 | 107.97 | 129.98 | 38.96 | 65.27 | 1435.71 | 80.96 |
| 2 | 1.686948 | 71.13 | 33.32 | 1836.18 | 107.97 | 128.33 | 38.31 | 65.65 | 1419.31 | 80.10 |
| 3 | 1.658705 | 72.35 | 32.54 | 1836.97 | 107.97 | 128.67 | 40.93 | 63.32 | 1421.79 | 75.86 |
| 4 | 1.680046 | 71.43 | 32.51 | 1837.05 | 107.97 | 126.77 | 37.14 | 65.49 | 1485.00 | 75.97 |
| 5 | 2.248397 | 53.37 | 35.39 | 1842.09 | 107.97 | 108.07 | 29.54 | 61.55 | 1499.59 | 75.00 |
| 6 | 2.099022 | 57.17 | 52.06 | 1842.18 | 107.97 | 113.86 | 34.71 | 51.64 | 1249.71 | 74.29 |
| 7 | 1.648727 | 72.78 | 35.38 | 1843.21 | 107.97 | 126.70 | 40.36 | 66.12 | 1470.00 | 76.96 |

Ready result delivery-interval P50/P95/P99 ms: 13.23/17.19/20.58.
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

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
