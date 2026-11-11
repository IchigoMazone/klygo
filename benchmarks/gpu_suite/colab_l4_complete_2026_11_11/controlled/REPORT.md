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

## FPS matrix — images, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 100.37 | 102.37 | 79.32 | 79.23 | 73.02 | 70.34 |
| yolo11 | 86.08 | 85.93 | 67.86 | 68.12 | 62.17 | 61.31 |
| yolo26 | 77.29 | 78.35 | 63.49 | 64.37 | 59.91 | 58.45 |
| yolo-world | 77.23 | 77.59 | 64.59 | 63.45 | 59.59 | 58.08 |
| grounding-dino | 4.89 | N/A | 4.82 | N/A | 4.81 | 4.81 |
| grounding-dino-base | 3.72 | N/A | 3.68 | N/A | 3.66 | 3.66 |
| faster-rcnn | 43.97 | N/A | 39.93 | N/A | 39.21 | 38.77 |
| d-fine | 2.67 | N/A | 2.64 | N/A | 2.65 | 2.64 |
| locate-anything | 2.44 | N/A | 2.44 | N/A | 2.44 | 2.42 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — images, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 248.41 | 246.08 | 171.56 | 168.80 | 140.21 | 131.37 |
| yolo11 | 210.24 | 209.94 | 154.17 | 155.12 | 128.69 | 124.19 |
| yolo26 | 195.69 | 196.73 | 146.47 | 144.16 | 126.67 | 118.32 |
| yolo-world | 195.79 | 195.84 | 150.70 | 144.42 | 127.69 | 118.31 |
| grounding-dino | 5.06 | N/A | 5.04 | N/A | 5.01 | 5.03 |
| grounding-dino-base | 3.71 | N/A | 3.70 | N/A | 3.70 | 3.70 |
| faster-rcnn | 71.81 | N/A | 68.23 | N/A | 66.34 | 65.99 |
| d-fine | 10.37 | N/A | 10.25 | N/A | 10.09 | 10.22 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 107.72 | 107.71 | 100.70 | 98.94 | 95.65 | 94.55 |
| yolo11 | 87.28 | 86.52 | 82.84 | 82.84 | 80.43 | 79.00 |
| yolo26 | 80.93 | 80.15 | 77.18 | 76.06 | 75.45 | 74.19 |
| yolo-world | 76.87 | 77.74 | 75.55 | 72.32 | 73.53 | 72.74 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — video, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 226.75 | 222.78 | N/A | N/A | 199.09 | 192.75 |
| yolo11 | 198.00 | 196.86 | N/A | N/A | 179.61 | 173.85 |
| yolo26 | 190.11 | 189.97 | N/A | N/A | 172.81 | 166.35 |
| yolo-world | 187.12 | 186.63 | N/A | N/A | 171.96 | 164.54 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, images, batch 1

Status: **complete**. Load 0.33 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 100.37 ± 1.71 | 14.06 | 9.78/10.66/13.18 | 1874.73 | 27.84/60.00 | 64/320 | PASS |
| raw-stream | 102.37 ± 1.82 | 14.37 | 9.69/11.64/12.74 | 1893.91 | 27.84/60.00 | 64/320 | PASS |
| native-non-stream | 79.32 ± 1.01 | 17.94 | 12.42/13.18/16.69 | 1894.07 | 27.81/60.00 | 64/320 | PASS |
| native-stream | 79.23 ± 0.71 | 17.95 | 12.50/14.54/15.06 | 1895.22 | 27.81/60.00 | 64/320 | PASS |
| managed-non-stream | 73.02 ± 0.83 | 15.01 | 14.06/16.04/16.86 | 1895.27 | 27.81/60.00 | 64/320 | PASS |
| managed-stream | 70.34 ± 0.61 | 15.59 | 14.67/15.57/18.51 | 1895.30 | 27.81/60.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.637626 | 100.37 | 11.21 | 1870.37 | 27.84 | 100.49 | 10.85 | 33.39 | 2040.00 | 47.69 |
| 2 | 0.644230 | 99.34 | 13.53 | 1874.15 | 27.84 | 100.52 | 11.23 | 34.80 | 2040.00 | 49.54 |
| 3 | 0.638443 | 100.24 | 14.71 | 1874.16 | 27.84 | 100.48 | 11.15 | 34.77 | 2040.00 | 49.77 |
| 4 | 0.629087 | 101.73 | 14.06 | 1874.55 | 27.84 | 100.52 | 13.62 | 34.90 | 2040.00 | 51.00 |
| 5 | 0.653183 | 97.98 | 14.41 | 1874.55 | 27.84 | 100.40 | 10.38 | 34.69 | 2040.00 | 51.31 |
| 6 | 0.619226 | 103.35 | 14.06 | 1874.73 | 27.84 | 100.51 | 13.00 | 35.92 | 2040.00 | 52.62 |
| 7 | 0.637584 | 100.38 | 14.08 | 1874.73 | 27.84 | 100.45 | 11.23 | 35.36 | 2040.00 | 52.69 |

Ready result delivery-interval P50/P95/P99 ms: 9.78/10.65/13.18.
Separate latency-pass duration: 0.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6099755312511945, 'inference': 6.665473781264808, 'postprocess': 1.2708802499936667}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.71% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.617590 | 103.63 | 13.63 | 1873.00 | 27.84 | 100.52 | 11.92 | 34.32 | 2040.00 | 47.85 |
| 2 | 0.617097 | 103.71 | 14.49 | 1893.51 | 27.84 | 100.50 | 11.46 | 34.30 | 2040.00 | 49.31 |
| 3 | 0.633902 | 100.96 | 14.20 | 1874.16 | 27.84 | 100.46 | 11.38 | 35.27 | 2040.00 | 49.77 |
| 4 | 0.625187 | 102.37 | 14.80 | 1893.91 | 27.84 | 100.47 | 13.31 | 34.13 | 2040.00 | 51.00 |
| 5 | 0.638731 | 100.20 | 14.37 | 1874.55 | 27.84 | 100.49 | 12.46 | 35.42 | 2040.00 | 51.62 |
| 6 | 0.623071 | 102.72 | 11.50 | 1873.85 | 27.84 | 100.47 | 11.46 | 34.73 | 2040.00 | 52.38 |
| 7 | 0.647131 | 98.90 | 14.65 | 1874.73 | 27.84 | 100.45 | 11.08 | 34.84 | 2040.00 | 52.85 |

Ready result delivery-interval P50/P95/P99 ms: 9.70/11.64/12.71.
Separate latency-pass duration: 0.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.5973378281289854, 'inference': 6.441346203107656, 'postprocess': 1.2570334375077152}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

27.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.821362 | 77.92 | 17.00 | 1892.42 | 27.81 | 100.48 | 10.88 | 34.92 | 2040.00 | 47.88 |
| 2 | 0.802579 | 79.74 | 17.91 | 1893.38 | 27.81 | 100.47 | 10.75 | 33.88 | 2040.00 | 49.00 |
| 3 | 0.816034 | 78.43 | 17.74 | 1893.52 | 27.81 | 100.51 | 9.94 | 35.04 | 2040.00 | 49.82 |
| 4 | 0.805022 | 79.50 | 18.11 | 1893.91 | 27.81 | 100.48 | 10.75 | 34.05 | 2040.00 | 50.88 |
| 5 | 0.799132 | 80.09 | 18.57 | 1890.68 | 27.81 | 100.80 | 11.25 | 35.51 | 2040.00 | 51.44 |
| 6 | 0.806858 | 79.32 | 18.52 | 1894.07 | 27.81 | 100.46 | 10.56 | 34.30 | 2040.00 | 52.00 |
| 7 | 0.827024 | 77.39 | 17.94 | 1893.20 | 27.81 | 100.47 | 9.82 | 35.02 | 2040.00 | 52.88 |

Ready result delivery-interval P50/P95/P99 ms: 12.43/13.18/16.68.
Separate latency-pass duration: 0.81 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

26.27% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.805107 | 79.49 | 18.14 | 1892.57 | 27.81 | 100.46 | 10.88 | 34.34 | 2040.00 | 47.88 |
| 2 | 0.807899 | 79.22 | 17.80 | 1894.94 | 27.81 | 100.46 | 11.50 | 33.49 | 2040.00 | 48.81 |
| 3 | 0.807732 | 79.23 | 18.13 | 1893.57 | 27.81 | 100.49 | 9.75 | 34.25 | 2040.00 | 49.88 |
| 4 | 0.815763 | 78.45 | 17.95 | 1895.07 | 27.81 | 100.51 | 9.65 | 33.55 | 2040.00 | 50.88 |
| 5 | 0.802007 | 79.80 | 18.05 | 1893.91 | 27.81 | 100.46 | 11.50 | 34.10 | 2040.00 | 51.75 |
| 6 | 0.809249 | 79.09 | 17.83 | 1895.22 | 27.81 | 100.51 | 10.25 | 34.52 | 2040.00 | 52.00 |
| 7 | 0.792763 | 80.73 | 14.62 | 1891.68 | 27.81 | 100.41 | 10.25 | 34.70 | 2040.00 | 52.88 |

Ready result delivery-interval P50/P95/P99 ms: 12.50/14.55/15.05.
Separate latency-pass duration: 0.82 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

37.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.876527 | 73.02 | 14.91 | 1892.92 | 27.81 | 100.48 | 10.89 | 33.83 | 2040.00 | 48.00 |
| 2 | 0.883760 | 72.42 | 14.94 | 1893.29 | 27.81 | 100.46 | 10.00 | 33.01 | 2040.00 | 48.78 |
| 3 | 0.880659 | 72.67 | 14.88 | 1893.71 | 27.81 | 100.51 | 10.00 | 34.46 | 2040.00 | 50.00 |
| 4 | 0.881622 | 72.59 | 15.39 | 1895.07 | 27.81 | 100.48 | 9.44 | 33.92 | 2040.00 | 50.50 |
| 5 | 0.861254 | 74.31 | 15.28 | 1893.94 | 27.81 | 100.46 | 9.94 | 34.35 | 2040.00 | 51.71 |
| 6 | 0.862274 | 74.22 | 15.01 | 1895.20 | 27.81 | 100.49 | 9.12 | 34.11 | 2040.00 | 51.88 |
| 7 | 0.864658 | 74.02 | 15.32 | 1895.27 | 27.81 | 100.40 | 10.89 | 34.15 | 2040.00 | 53.00 |

Ready result delivery-interval P50/P95/P99 ms: 14.07/16.03/16.86.
Separate latency-pass duration: 0.91 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

41.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.903293 | 70.85 | 16.03 | 1894.52 | 27.81 | 100.40 | 10.39 | 33.24 | 2040.00 | 48.00 |
| 2 | 0.903828 | 70.81 | 15.78 | 1892.06 | 27.81 | 100.37 | 9.00 | 32.79 | 2040.00 | 48.61 |
| 3 | 0.895112 | 71.50 | 15.41 | 1895.02 | 27.81 | 100.39 | 9.11 | 33.07 | 2040.00 | 50.00 |
| 4 | 0.917953 | 69.72 | 15.58 | 1895.06 | 27.81 | 100.40 | 9.05 | 33.10 | 2040.00 | 50.11 |
| 5 | 0.910985 | 70.25 | 17.89 | 1895.17 | 27.81 | 100.41 | 9.58 | 34.35 | 2040.00 | 51.89 |
| 6 | 0.909914 | 70.34 | 15.58 | 1895.20 | 27.81 | 100.36 | 9.11 | 33.59 | 2040.00 | 51.89 |
| 7 | 0.915442 | 69.91 | 15.59 | 1895.30 | 27.81 | 100.39 | 10.37 | 33.86 | 2040.00 | 53.00 |

Ready result delivery-interval P50/P95/P99 ms: 14.68/15.55/18.48.
Separate latency-pass duration: 0.95 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo11, images, batch 1

Status: **complete**. Load 0.26 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 86.08 ± 1.29 | 15.90 | 11.54/13.75/14.41 | 1923.91 | 59.81/88.00 | 64/320 | PASS |
| raw-stream | 85.93 ± 0.76 | 15.63 | 11.70/12.57/14.04 | 1923.91 | 59.81/88.00 | 64/320 | PASS |
| native-non-stream | 67.86 ± 0.96 | 19.19 | 14.62/15.46/18.77 | 1943.40 | 59.77/88.00 | 64/320 | PASS |
| native-stream | 68.12 ± 1.10 | 16.84 | 14.56/15.15/15.94 | 1944.96 | 59.77/88.00 | 64/320 | PASS |
| managed-non-stream | 62.17 ± 1.23 | 17.84 | 15.74/16.19/17.08 | 1945.01 | 59.77/88.00 | 64/320 | PASS |
| managed-stream | 61.31 ± 0.65 | 20.97 | 16.18/16.95/18.74 | 1945.08 | 59.77/88.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.736905 | 86.85 | 13.80 | 1920.20 | 59.81 | 100.41 | 12.87 | 34.57 | 2040.00 | 54.87 |
| 2 | 0.735290 | 87.04 | 16.13 | 1923.32 | 59.81 | 100.42 | 13.07 | 36.01 | 2040.00 | 56.27 |
| 3 | 0.746678 | 85.71 | 15.90 | 1923.32 | 59.81 | 100.37 | 11.00 | 35.59 | 2040.00 | 56.60 |
| 4 | 0.743519 | 86.08 | 15.69 | 1923.73 | 59.81 | 100.42 | 13.40 | 35.51 | 2040.00 | 57.73 |
| 5 | 0.744493 | 85.96 | 15.90 | 1923.74 | 59.81 | 100.47 | 11.47 | 35.46 | 2040.00 | 57.67 |
| 6 | 0.769025 | 83.22 | 18.75 | 1923.91 | 59.81 | 100.49 | 12.94 | 36.12 | 2040.00 | 58.81 |
| 7 | 0.738348 | 86.68 | 16.05 | 1923.91 | 59.81 | 100.50 | 10.67 | 35.80 | 2040.00 | 58.73 |

Ready result delivery-interval P50/P95/P99 ms: 11.54/13.75/14.41.
Separate latency-pass duration: 0.75 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6017578750032726, 'inference': 8.32169924998638, 'postprocess': 1.2155811718841392}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.74% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.743233 | 86.11 | 15.63 | 1921.95 | 59.81 | 100.46 | 11.73 | 35.34 | 2040.00 | 55.00 |
| 2 | 0.737340 | 86.80 | 13.54 | 1921.56 | 59.81 | 100.47 | 12.27 | 35.13 | 2040.00 | 56.00 |
| 3 | 0.752206 | 85.08 | 17.23 | 1923.32 | 59.81 | 100.40 | 11.13 | 35.25 | 2040.00 | 56.73 |
| 4 | 0.744815 | 85.93 | 13.45 | 1921.09 | 59.81 | 100.45 | 10.73 | 35.02 | 2040.00 | 57.53 |
| 5 | 0.750035 | 85.33 | 15.68 | 1923.74 | 59.81 | 100.46 | 12.53 | 35.89 | 2040.00 | 57.80 |
| 6 | 0.736490 | 86.90 | 13.45 | 1922.15 | 59.81 | 100.45 | 11.47 | 35.44 | 2040.00 | 58.80 |
| 7 | 0.751934 | 85.11 | 15.95 | 1923.91 | 59.81 | 100.48 | 10.93 | 35.89 | 2040.00 | 58.87 |

Ready result delivery-interval P50/P95/P99 ms: 11.70/12.57/14.02.
Separate latency-pass duration: 0.76 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6030499062426031, 'inference': 8.374673734362403, 'postprocess': 1.2629346718746604}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

26.84% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.930525 | 68.78 | 19.28 | 1940.38 | 59.77 | 100.47 | 11.58 | 34.96 | 2040.00 | 55.00 |
| 2 | 0.946145 | 67.64 | 16.83 | 1940.36 | 59.77 | 100.47 | 9.79 | 34.82 | 2040.00 | 56.00 |
| 3 | 0.958794 | 66.75 | 19.39 | 1939.34 | 59.77 | 100.48 | 11.00 | 34.89 | 2040.00 | 56.84 |
| 4 | 0.943097 | 67.86 | 17.71 | 1940.90 | 59.77 | 100.80 | 11.05 | 34.47 | 2040.00 | 57.00 |
| 5 | 0.932243 | 68.65 | 19.19 | 1942.09 | 59.77 | 100.48 | 11.47 | 35.53 | 2040.00 | 57.89 |
| 6 | 0.919278 | 69.62 | 16.64 | 1943.40 | 59.77 | 100.46 | 10.42 | 35.14 | 2040.00 | 58.00 |
| 7 | 0.948205 | 67.50 | 19.34 | 1941.09 | 59.77 | 100.51 | 11.79 | 35.46 | 2040.00 | 59.00 |

Ready result delivery-interval P50/P95/P99 ms: 14.61/15.46/18.76.
Separate latency-pass duration: 0.95 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

25.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.921754 | 69.43 | 16.58 | 1941.78 | 59.77 | 100.46 | 9.68 | 33.86 | 2040.00 | 55.00 |
| 2 | 0.960559 | 66.63 | 20.78 | 1944.07 | 59.77 | 100.48 | 10.58 | 34.22 | 2040.00 | 55.95 |
| 3 | 0.939492 | 68.12 | 16.84 | 1941.75 | 59.77 | 100.50 | 9.47 | 34.67 | 2040.00 | 56.84 |
| 4 | 0.965565 | 66.28 | 20.62 | 1944.80 | 59.77 | 100.51 | 9.30 | 33.99 | 2040.00 | 57.00 |
| 5 | 0.936021 | 68.37 | 16.45 | 1942.09 | 59.77 | 100.51 | 10.74 | 35.27 | 2040.00 | 57.84 |
| 6 | 0.933948 | 68.53 | 20.58 | 1944.96 | 59.77 | 100.52 | 10.89 | 34.48 | 2040.00 | 58.00 |
| 7 | 0.941499 | 67.98 | 16.71 | 1941.09 | 59.77 | 100.49 | 9.47 | 34.90 | 2040.00 | 59.00 |

Ready result delivery-interval P50/P95/P99 ms: 14.56/15.15/15.94.
Separate latency-pass duration: 0.93 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

39.31% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.026589 | 62.34 | 17.71 | 1943.45 | 59.77 | 100.47 | 9.90 | 34.13 | 2040.00 | 55.00 |
| 2 | 1.029508 | 62.17 | 17.75 | 1943.02 | 59.77 | 100.48 | 9.86 | 33.99 | 2040.00 | 55.90 |
| 3 | 1.051663 | 60.86 | 17.93 | 1942.27 | 59.77 | 100.52 | 10.24 | 34.39 | 2040.00 | 56.86 |
| 4 | 1.064270 | 60.14 | 20.14 | 1944.80 | 59.77 | 100.51 | 8.82 | 33.72 | 2040.00 | 57.00 |
| 5 | 1.016413 | 62.97 | 17.84 | 1944.84 | 59.77 | 100.47 | 9.14 | 34.77 | 2040.00 | 57.86 |
| 6 | 1.061095 | 60.32 | 17.96 | 1941.14 | 59.77 | 100.48 | 9.14 | 34.35 | 2040.00 | 58.00 |
| 7 | 1.015632 | 63.01 | 17.74 | 1945.01 | 59.77 | 100.52 | 10.10 | 34.64 | 2040.00 | 59.00 |

Ready result delivery-interval P50/P95/P99 ms: 15.74/16.19/17.08.
Separate latency-pass duration: 1.01 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

40.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.037706 | 61.67 | 20.97 | 1943.60 | 59.77 | 100.37 | 8.95 | 34.26 | 2040.00 | 55.38 |
| 2 | 1.052017 | 60.84 | 17.78 | 1938.97 | 59.77 | 100.36 | 8.90 | 33.60 | 2040.00 | 55.81 |
| 3 | 1.052334 | 60.82 | 21.73 | 1944.77 | 59.77 | 100.35 | 9.76 | 34.11 | 2040.00 | 56.90 |
| 4 | 1.043071 | 61.36 | 17.68 | 1940.02 | 59.77 | 100.37 | 8.81 | 33.53 | 2040.00 | 57.00 |
| 5 | 1.037248 | 61.70 | 21.22 | 1944.92 | 59.77 | 100.27 | 8.90 | 34.06 | 2040.00 | 57.90 |
| 6 | 1.069320 | 59.85 | 17.93 | 1942.50 | 59.77 | 100.37 | 8.82 | 34.01 | 2040.00 | 58.00 |
| 7 | 1.043867 | 61.31 | 24.12 | 1945.08 | 59.77 | 100.38 | 9.90 | 34.83 | 2040.00 | 59.00 |

Ready result delivery-interval P50/P95/P99 ms: 16.18/16.95/18.67.
Separate latency-pass duration: 1.05 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo26, images, batch 1

Status: **complete**. Load 0.28 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 77.29 ± 1.76 | 17.18 | 12.82/15.71/15.82 | 1864.23 | 59.02/88.00 | 64/320 | PASS |
| raw-stream | 78.35 ± 0.79 | 16.87 | 12.51/15.29/16.23 | 1876.00 | 59.02/88.00 | 64/320 | PASS |
| native-non-stream | 63.49 ± 0.57 | 19.85 | 15.74/17.07/19.34 | 1874.39 | 58.99/88.00 | 64/320 | PASS |
| native-stream | 64.37 ± 0.70 | 20.73 | 15.86/16.88/18.29 | 1876.29 | 58.99/88.00 | 64/320 | PASS |
| managed-non-stream | 59.91 ± 0.66 | 18.32 | 16.91/19.55/19.94 | 1876.27 | 58.99/88.00 | 64/320 | PASS |
| managed-stream | 58.45 ± 0.72 | 18.62 | 17.30/20.17/21.53 | 1877.34 | 58.99/88.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.793693 | 80.64 | 14.35 | 1864.23 | 59.02 | 100.51 | 13.75 | 35.16 | 2040.00 | 60.62 |
| 2 | 0.828001 | 77.29 | 17.33 | 1855.28 | 59.02 | 100.51 | 11.47 | 35.87 | 2040.00 | 61.76 |
| 3 | 0.832505 | 76.88 | 17.12 | 1855.28 | 59.02 | 100.53 | 12.35 | 35.80 | 2040.00 | 61.71 |
| 4 | 0.832163 | 76.91 | 17.25 | 1855.70 | 59.02 | 100.50 | 11.47 | 35.90 | 2040.00 | 62.53 |
| 5 | 0.852214 | 75.10 | 18.75 | 1855.70 | 59.02 | 100.47 | 11.06 | 35.80 | 2040.00 | 62.65 |
| 6 | 0.815379 | 78.49 | 17.18 | 1856.38 | 59.02 | 100.49 | 13.65 | 36.91 | 2040.00 | 63.65 |
| 7 | 0.812353 | 78.78 | 17.12 | 1856.38 | 59.02 | 100.46 | 11.94 | 36.18 | 2040.00 | 63.59 |

Ready result delivery-interval P50/P95/P99 ms: 12.82/15.71/15.82.
Separate latency-pass duration: 0.85 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6086301093475299, 'inference': 9.520456921869425, 'postprocess': 1.2207311406271515}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.45% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.798836 | 80.12 | 13.89 | 1864.23 | 59.02 | 100.50 | 13.44 | 36.23 | 2040.00 | 60.88 |
| 2 | 0.824293 | 77.64 | 17.86 | 1874.53 | 59.02 | 100.50 | 11.71 | 35.86 | 2040.00 | 61.53 |
| 3 | 0.814241 | 78.60 | 16.87 | 1855.28 | 59.02 | 100.50 | 11.94 | 36.59 | 2040.00 | 61.76 |
| 4 | 0.819097 | 78.13 | 14.13 | 1855.70 | 59.02 | 100.50 | 11.47 | 35.74 | 2040.00 | 62.29 |
| 5 | 0.816005 | 78.43 | 16.79 | 1855.70 | 59.02 | 100.49 | 12.41 | 36.47 | 2040.00 | 62.88 |
| 6 | 0.816802 | 78.35 | 19.69 | 1876.00 | 59.02 | 100.51 | 11.82 | 36.03 | 2040.00 | 63.35 |
| 7 | 0.819756 | 78.07 | 16.91 | 1856.38 | 59.02 | 100.46 | 13.53 | 36.81 | 2040.00 | 63.76 |

Ready result delivery-interval P50/P95/P99 ms: 12.51/15.29/16.21.
Separate latency-pass duration: 0.82 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6088927187413162, 'inference': 9.578554624987135, 'postprocess': 1.2391599843830647}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

23.24% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.024427 | 62.47 | 18.10 | 1874.06 | 58.99 | 100.54 | 9.90 | 35.36 | 2040.00 | 60.81 |
| 2 | 1.020443 | 62.72 | 23.62 | 1874.39 | 58.99 | 100.47 | 10.76 | 35.18 | 2040.00 | 61.00 |
| 3 | 0.998703 | 64.08 | 20.09 | 1873.07 | 58.99 | 100.51 | 10.40 | 35.49 | 2040.00 | 61.80 |
| 4 | 1.011530 | 63.27 | 17.23 | 1871.44 | 58.99 | 100.49 | 11.25 | 35.50 | 2040.00 | 62.00 |
| 5 | 1.007540 | 63.52 | 19.85 | 1871.65 | 58.99 | 100.51 | 9.80 | 36.08 | 2040.00 | 62.80 |
| 6 | 1.008092 | 63.49 | 17.05 | 1872.77 | 58.99 | 100.49 | 10.00 | 35.32 | 2040.00 | 63.00 |
| 7 | 1.002813 | 63.82 | 20.15 | 1872.79 | 58.99 | 100.45 | 11.55 | 35.94 | 2040.00 | 63.70 |

Ready result delivery-interval P50/P95/P99 ms: 15.74/17.07/19.34.
Separate latency-pass duration: 1.02 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

21.49% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.019183 | 62.80 | 17.13 | 1874.30 | 58.99 | 100.49 | 9.95 | 34.74 | 2040.00 | 60.90 |
| 2 | 0.991643 | 64.54 | 20.04 | 1871.91 | 58.99 | 100.47 | 11.70 | 35.19 | 2040.00 | 61.00 |
| 3 | 1.008335 | 63.47 | 21.17 | 1875.45 | 58.99 | 100.48 | 10.60 | 35.47 | 2040.00 | 61.90 |
| 4 | 1.010985 | 63.30 | 21.19 | 1876.10 | 58.99 | 100.76 | 11.10 | 34.86 | 2040.00 | 62.00 |
| 5 | 0.994305 | 64.37 | 21.08 | 1874.66 | 58.99 | 100.50 | 10.35 | 35.30 | 2040.00 | 62.90 |
| 6 | 0.992735 | 64.47 | 20.54 | 1876.29 | 58.99 | 100.50 | 11.40 | 35.00 | 2040.00 | 63.00 |
| 7 | 0.993861 | 64.40 | 20.73 | 1876.00 | 58.99 | 100.48 | 10.80 | 35.68 | 2040.00 | 63.80 |

Ready result delivery-interval P50/P95/P99 ms: 15.87/16.88/18.29.
Separate latency-pass duration: 1.02 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

30.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.059606 | 60.40 | 18.26 | 1872.39 | 58.99 | 100.45 | 10.52 | 35.09 | 2040.00 | 60.90 |
| 2 | 1.077779 | 59.38 | 18.52 | 1874.16 | 58.99 | 100.51 | 11.09 | 34.71 | 2040.00 | 61.00 |
| 3 | 1.068266 | 59.91 | 18.32 | 1875.90 | 58.99 | 100.53 | 11.09 | 34.82 | 2040.00 | 61.91 |
| 4 | 1.073093 | 59.64 | 18.39 | 1876.10 | 58.99 | 100.48 | 10.68 | 34.39 | 2040.00 | 62.00 |
| 5 | 1.058287 | 60.48 | 18.26 | 1875.84 | 58.99 | 100.49 | 11.14 | 35.17 | 2040.00 | 63.00 |
| 6 | 1.068284 | 59.91 | 18.01 | 1876.27 | 58.99 | 100.50 | 9.86 | 34.88 | 2040.00 | 63.00 |
| 7 | 1.093239 | 58.54 | 19.03 | 1875.53 | 58.99 | 100.46 | 9.68 | 35.26 | 2040.00 | 63.82 |

Ready result delivery-interval P50/P95/P99 ms: 16.91/19.56/19.94.
Separate latency-pass duration: 1.10 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

33.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.094269 | 58.49 | 18.47 | 1871.52 | 58.99 | 100.41 | 9.77 | 34.40 | 2040.00 | 61.00 |
| 2 | 1.094881 | 58.45 | 18.91 | 1871.64 | 58.99 | 100.41 | 9.82 | 34.24 | 2040.00 | 61.00 |
| 3 | 1.088129 | 58.82 | 18.54 | 1876.06 | 58.99 | 100.37 | 10.82 | 34.78 | 2040.00 | 61.91 |
| 4 | 1.113114 | 57.50 | 18.75 | 1876.11 | 58.99 | 100.37 | 9.18 | 34.64 | 2040.00 | 61.91 |
| 5 | 1.083890 | 59.05 | 18.62 | 1876.25 | 58.99 | 100.40 | 10.73 | 34.90 | 2040.00 | 62.91 |
| 6 | 1.096209 | 58.38 | 18.57 | 1876.28 | 58.99 | 100.39 | 9.18 | 34.79 | 2040.00 | 63.00 |
| 7 | 1.122087 | 57.04 | 22.32 | 1877.34 | 58.99 | 100.40 | 9.26 | 35.38 | 2040.00 | 63.83 |

Ready result delivery-interval P50/P95/P99 ms: 17.30/20.18/21.47.
Separate latency-pass duration: 1.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo-world, images, batch 1

Status: **complete**. Load 7.66 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 77.23 ± 2.36 | 17.39 | 12.71/13.73/14.79 | 2108.68 | 115.04/132.00 | 64/384 | PASS |
| raw-stream | 77.59 ± 1.75 | 16.69 | 12.66/13.18/15.16 | 2108.68 | 115.04/132.00 | 64/384 | PASS |
| native-non-stream | 64.59 ± 0.73 | 20.03 | 15.66/17.95/19.90 | 2126.85 | 115.01/132.00 | 64/384 | PASS |
| native-stream | 63.45 ± 0.92 | 17.10 | 15.95/17.79/18.98 | 2126.85 | 115.01/132.00 | 64/384 | PASS |
| managed-non-stream | 59.59 ± 0.54 | 18.05 | 16.96/17.68/19.60 | 2128.76 | 115.01/132.00 | 64/384 | PASS |
| managed-stream | 58.08 ± 0.62 | 18.45 | 17.37/18.12/21.17 | 2127.22 | 115.01/132.00 | 64/384 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.897522 | 71.31 | 16.07 | 2104.88 | 115.04 | 100.50 | 16.83 | 41.80 | 2040.00 | 65.83 |
| 2 | 0.828648 | 77.23 | 17.39 | 2108.08 | 115.04 | 100.45 | 19.29 | 44.61 | 2040.00 | 67.71 |
| 3 | 0.831563 | 76.96 | 17.46 | 2108.09 | 115.04 | 100.50 | 17.65 | 43.80 | 2040.00 | 67.65 |
| 4 | 0.822132 | 77.85 | 17.22 | 2108.53 | 115.04 | 100.53 | 19.35 | 44.02 | 2040.00 | 68.76 |
| 5 | 0.823957 | 77.67 | 17.13 | 2108.54 | 115.04 | 100.44 | 16.88 | 42.85 | 2040.00 | 68.71 |
| 6 | 0.832735 | 76.86 | 17.39 | 2108.68 | 115.04 | 100.47 | 17.35 | 45.39 | 2040.00 | 69.76 |
| 7 | 0.819838 | 78.06 | 17.41 | 2108.68 | 115.04 | 100.48 | 17.06 | 44.24 | 2040.00 | 69.71 |

Ready result delivery-interval P50/P95/P99 ms: 12.71/13.73/14.80.
Separate latency-pass duration: 0.82 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6362555156265159, 'inference': 10.411101687523683, 'postprocess': 1.3888161406114818}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.811846 | 78.83 | 16.69 | 2106.64 | 115.04 | 100.50 | 17.53 | 43.32 | 2040.00 | 65.76 |
| 2 | 0.853536 | 74.98 | 14.72 | 2105.44 | 115.04 | 100.49 | 17.24 | 43.49 | 2040.00 | 67.18 |
| 3 | 0.828282 | 77.27 | 17.45 | 2108.09 | 115.04 | 100.46 | 20.12 | 43.75 | 2040.00 | 67.76 |
| 4 | 0.839984 | 76.19 | 14.05 | 2106.77 | 115.04 | 100.53 | 19.82 | 44.02 | 2040.00 | 68.76 |
| 5 | 0.796106 | 80.39 | 16.87 | 2108.54 | 115.04 | 100.44 | 18.75 | 45.04 | 2040.00 | 68.75 |
| 6 | 0.821467 | 77.91 | 14.34 | 2108.68 | 115.04 | 100.48 | 19.53 | 44.23 | 2040.00 | 69.82 |
| 7 | 0.824885 | 77.59 | 17.36 | 2108.68 | 115.04 | 100.47 | 20.12 | 45.27 | 2040.00 | 70.12 |

Ready result delivery-interval P50/P95/P99 ms: 12.67/13.18/15.14.
Separate latency-pass duration: 0.82 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.5938402656546771, 'inference': 9.480974671909337, 'postprocess': 1.2546358906213584}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

18.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000150 | 63.99 | 20.03 | 2123.64 | 115.01 | 100.46 | 15.00 | 42.38 | 2040.00 | 65.90 |
| 2 | 1.005286 | 63.66 | 16.86 | 2124.91 | 115.01 | 100.49 | 16.65 | 40.87 | 2040.00 | 66.90 |
| 3 | 0.989263 | 64.69 | 20.19 | 2126.26 | 115.01 | 100.47 | 17.10 | 43.40 | 2040.00 | 67.90 |
| 4 | 1.003207 | 63.80 | 16.64 | 2125.53 | 115.01 | 100.49 | 15.05 | 42.72 | 2040.00 | 68.25 |
| 5 | 0.976675 | 65.53 | 20.34 | 2126.71 | 115.01 | 100.45 | 17.80 | 43.63 | 2040.00 | 68.80 |
| 6 | 0.980339 | 65.28 | 16.52 | 2126.85 | 115.01 | 100.50 | 16.40 | 43.72 | 2040.00 | 69.50 |
| 7 | 0.990939 | 64.59 | 20.45 | 2126.85 | 115.01 | 100.48 | 16.00 | 43.97 | 2040.00 | 69.90 |

Ready result delivery-interval P50/P95/P99 ms: 15.66/17.95/19.89.
Separate latency-pass duration: 1.02 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

21.48% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.002085 | 63.87 | 17.09 | 2123.88 | 115.01 | 100.45 | 16.40 | 42.45 | 2040.00 | 65.90 |
| 2 | 1.033227 | 61.94 | 17.32 | 2125.93 | 115.01 | 100.48 | 16.29 | 41.36 | 2040.00 | 66.81 |
| 3 | 1.008745 | 63.45 | 17.43 | 2125.16 | 115.01 | 100.47 | 14.60 | 42.59 | 2040.00 | 67.80 |
| 4 | 1.009565 | 63.39 | 17.16 | 2124.36 | 115.01 | 100.72 | 15.00 | 41.06 | 2040.00 | 68.20 |
| 5 | 1.000946 | 63.94 | 17.02 | 2125.54 | 115.01 | 100.45 | 14.80 | 42.90 | 2040.00 | 68.80 |
| 6 | 1.001736 | 63.89 | 16.87 | 2125.67 | 115.01 | 100.49 | 14.70 | 42.75 | 2040.00 | 69.40 |
| 7 | 1.035322 | 61.82 | 17.10 | 2126.85 | 115.01 | 100.45 | 14.05 | 42.01 | 2040.00 | 70.10 |

Ready result delivery-interval P50/P95/P99 ms: 15.96/17.79/18.97.
Separate latency-pass duration: 1.03 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

29.65% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.070302 | 59.80 | 18.10 | 2127.31 | 115.01 | 100.55 | 15.77 | 40.82 | 2040.00 | 66.09 |
| 2 | 1.080401 | 59.24 | 17.88 | 2126.58 | 115.01 | 100.49 | 15.14 | 41.99 | 2040.00 | 66.64 |
| 3 | 1.074035 | 59.59 | 18.41 | 2128.53 | 115.01 | 100.47 | 13.95 | 41.76 | 2040.00 | 67.82 |
| 4 | 1.073648 | 59.61 | 17.90 | 2128.75 | 115.01 | 100.52 | 14.27 | 41.68 | 2040.00 | 67.91 |
| 5 | 1.068221 | 59.91 | 17.91 | 2128.76 | 115.01 | 100.48 | 14.32 | 42.44 | 2040.00 | 68.91 |
| 6 | 1.078129 | 59.36 | 18.05 | 2127.71 | 115.01 | 100.46 | 15.23 | 42.56 | 2040.00 | 68.91 |
| 7 | 1.097649 | 58.31 | 18.08 | 2126.55 | 115.01 | 100.49 | 13.95 | 41.95 | 2040.00 | 70.50 |

Ready result delivery-interval P50/P95/P99 ms: 16.96/17.68/19.59.
Separate latency-pass duration: 1.09 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

32.32% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.112958 | 57.50 | 18.41 | 2125.70 | 115.01 | 100.40 | 15.26 | 40.45 | 2040.00 | 66.26 |
| 2 | 1.086911 | 58.88 | 18.31 | 2124.67 | 115.01 | 100.33 | 13.73 | 40.24 | 2040.00 | 66.55 |
| 3 | 1.101852 | 58.08 | 18.45 | 2124.59 | 115.01 | 100.36 | 13.82 | 41.48 | 2040.00 | 67.91 |
| 4 | 1.093471 | 58.53 | 18.60 | 2124.79 | 115.01 | 100.38 | 14.27 | 41.02 | 2040.00 | 67.82 |
| 5 | 1.087266 | 58.86 | 18.55 | 2126.06 | 115.01 | 100.40 | 13.73 | 41.35 | 2040.00 | 68.91 |
| 6 | 1.101892 | 58.08 | 18.35 | 2124.78 | 115.01 | 100.41 | 14.14 | 40.89 | 2040.00 | 68.77 |
| 7 | 1.116241 | 57.34 | 18.57 | 2127.22 | 115.01 | 100.41 | 13.74 | 42.44 | 2040.00 | 70.48 |

Ready result delivery-interval P50/P95/P99 ms: 17.34/18.12/21.11.
Separate latency-pass duration: 1.12 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=64.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — grounding-dino, images, batch 1

Status: **complete**. Load 22.68 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 4.89 ± 0.02 | 203.85 | 204.12/209.43/210.15 | 2942.87 | 1366.93/1424.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 4.82 ± 0.02 | 206.52 | 208.41/211.83/212.17 | 2942.87 | 1366.92/1424.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 4.81 ± 0.02 | 206.90 | 207.45/212.10/213.07 | 2947.56 | 1366.92/1424.00 | 8/40 | PASS |
| managed-stream | 4.81 ± 0.02 | 204.84 | 207.73/212.93/214.54 | 2947.56 | 1366.92/1424.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.636760 | 4.89 | 200.04 | 2942.87 | 1366.93 | 100.47 | 71.76 | 55.36 | 1525.91 | 73.94 |
| 2 | 1.638883 | 4.88 | 203.88 | 2942.87 | 1366.93 | 100.45 | 51.27 | 57.79 | 1510.45 | 76.39 |
| 3 | 1.626824 | 4.92 | 204.59 | 2913.92 | 1366.93 | 100.46 | 41.58 | 67.35 | 1445.00 | 76.58 |
| 4 | 1.624596 | 4.92 | 203.72 | 2913.92 | 1366.93 | 100.40 | 83.39 | 69.15 | 1454.09 | 79.15 |
| 5 | 1.632205 | 4.90 | 205.64 | 2942.87 | 1366.93 | 100.44 | 82.61 | 65.02 | 1431.82 | 79.94 |
| 6 | 1.634787 | 4.89 | 203.85 | 2942.87 | 1366.93 | 100.49 | 60.48 | 51.74 | 1619.55 | 79.45 |
| 7 | 1.639607 | 4.88 | 202.54 | 2942.87 | 1366.93 | 100.47 | 81.70 | 64.03 | 1499.55 | 78.15 |

Ready result delivery-interval P50/P95/P99 ms: 204.12/209.44/210.16.
Separate latency-pass duration: 1.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

1.79% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.660198 | 4.82 | 205.80 | 2942.87 | 1366.92 | 100.48 | 81.82 | 64.65 | 1518.64 | 74.06 |
| 2 | 1.652492 | 4.84 | 202.82 | 2938.70 | 1366.92 | 100.45 | 77.39 | 54.94 | 1600.91 | 76.64 |
| 3 | 1.660121 | 4.82 | 207.55 | 2942.87 | 1366.92 | 100.45 | 42.39 | 62.29 | 1582.27 | 77.52 |
| 4 | 1.660142 | 4.82 | 204.89 | 2942.87 | 1366.92 | 100.45 | 73.24 | 58.33 | 1561.82 | 78.82 |
| 5 | 1.668878 | 4.79 | 210.86 | 2938.71 | 1366.92 | 100.46 | 65.94 | 51.96 | 1538.82 | 80.24 |
| 6 | 1.661979 | 4.81 | 206.52 | 2938.71 | 1366.92 | 100.45 | 40.73 | 63.50 | 1507.73 | 79.70 |
| 7 | 1.668955 | 4.79 | 209.49 | 2942.87 | 1366.92 | 100.45 | 71.06 | 57.40 | 1568.82 | 78.41 |

Ready result delivery-interval P50/P95/P99 ms: 208.41/211.83/212.17.
Separate latency-pass duration: 1.67 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

1.50% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.657308 | 4.83 | 206.01 | 2945.73 | 1366.92 | 100.50 | 69.12 | 51.08 | 1606.36 | 75.15 |
| 2 | 1.657447 | 4.83 | 204.27 | 2947.55 | 1366.92 | 100.48 | 81.82 | 64.11 | 1511.36 | 75.76 |
| 3 | 1.667039 | 4.80 | 209.56 | 2946.38 | 1366.92 | 100.50 | 71.06 | 55.19 | 1505.29 | 77.59 |
| 4 | 1.671657 | 4.79 | 206.90 | 2944.56 | 1366.92 | 100.50 | 49.00 | 59.70 | 1522.94 | 78.79 |
| 5 | 1.674246 | 4.78 | 207.29 | 2946.39 | 1366.92 | 100.48 | 40.91 | 63.57 | 1493.38 | 80.18 |
| 6 | 1.659370 | 4.82 | 204.54 | 2947.56 | 1366.92 | 100.44 | 53.85 | 55.25 | 1579.55 | 79.88 |
| 7 | 1.662936 | 4.81 | 207.34 | 2944.57 | 1366.92 | 100.44 | 44.59 | 60.89 | 1527.35 | 77.76 |

Ready result delivery-interval P50/P95/P99 ms: 207.46/212.09/213.07.
Separate latency-pass duration: 1.67 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

1.75% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.659503 | 4.82 | 205.61 | 2947.56 | 1366.92 | 100.45 | 42.42 | 62.83 | 1499.55 | 74.88 |
| 2 | 1.661426 | 4.82 | 204.27 | 2947.16 | 1366.92 | 100.42 | 54.73 | 52.74 | 1600.91 | 75.97 |
| 3 | 1.665048 | 4.80 | 204.11 | 2944.57 | 1366.92 | 100.44 | 78.53 | 61.96 | 1512.35 | 78.03 |
| 4 | 1.661160 | 4.82 | 203.37 | 2947.56 | 1366.92 | 100.44 | 46.55 | 59.02 | 1515.00 | 78.00 |
| 5 | 1.675438 | 4.77 | 205.45 | 2946.39 | 1366.92 | 100.42 | 68.24 | 54.54 | 1538.82 | 80.24 |
| 6 | 1.663389 | 4.81 | 204.84 | 2947.56 | 1366.92 | 100.38 | 77.00 | 61.34 | 1521.18 | 80.35 |
| 7 | 1.662053 | 4.81 | 205.75 | 2944.57 | 1366.92 | 100.40 | 42.91 | 60.60 | 1458.64 | 78.06 |

Ready result delivery-interval P50/P95/P99 ms: 207.73/212.94/214.56.
Separate latency-pass duration: 1.67 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — grounding-dino-base, images, batch 1

Status: **complete**. Load 20.92 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 3.72 ± 0.02 | 267.96 | 267.74/271.91/272.92 | 2910.48 | 1600.91/1662.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 3.68 ± 0.02 | 271.89 | 270.94/274.45/275.67 | 2910.48 | 1600.90/1662.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 3.66 ± 0.01 | 272.31 | 272.33/272.91/272.96 | 2919.86 | 1600.90/1662.00 | 8/40 | PASS |
| managed-stream | 3.66 ± 0.02 | 279.41 | 272.25/276.55/277.77 | 2919.86 | 1600.90/1662.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.134663 | 3.75 | 272.88 | 2910.48 | 1600.91 | 100.45 | 71.02 | 61.88 | 1388.72 | 77.51 |
| 2 | 2.156837 | 3.71 | 268.11 | 2910.48 | 1600.91 | 100.44 | 74.33 | 62.96 | 1381.05 | 80.56 |
| 3 | 2.152923 | 3.72 | 266.80 | 2910.48 | 1600.91 | 100.47 | 72.19 | 62.57 | 1354.88 | 80.58 |
| 4 | 2.152602 | 3.72 | 269.24 | 2910.48 | 1600.91 | 100.46 | 72.37 | 63.18 | 1395.00 | 77.58 |
| 5 | 2.147819 | 3.72 | 267.96 | 2910.48 | 1600.91 | 100.47 | 74.12 | 62.10 | 1360.12 | 77.23 |
| 6 | 2.139637 | 3.74 | 264.78 | 2910.48 | 1600.91 | 100.49 | 75.12 | 62.10 | 1368.14 | 75.40 |
| 7 | 2.130246 | 3.76 | 265.13 | 2910.48 | 1600.91 | 100.48 | 75.16 | 61.11 | 1360.81 | 75.56 |

Ready result delivery-interval P50/P95/P99 ms: 267.74/271.91/272.91.
Separate latency-pass duration: 2.14 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

1.42% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.165058 | 3.70 | 270.86 | 2910.48 | 1600.90 | 100.46 | 72.93 | 63.32 | 1367.79 | 78.23 |
| 2 | 2.197672 | 3.64 | 277.09 | 2910.48 | 1600.90 | 100.48 | 70.84 | 62.64 | 1382.39 | 80.34 |
| 3 | 2.177779 | 3.67 | 270.33 | 2910.48 | 1600.90 | 100.49 | 69.05 | 61.89 | 1378.98 | 80.16 |
| 4 | 2.183901 | 3.66 | 279.09 | 2910.48 | 1600.90 | 100.47 | 68.36 | 62.56 | 1383.07 | 77.93 |
| 5 | 2.174781 | 3.68 | 271.89 | 2910.48 | 1600.90 | 100.57 | 72.18 | 62.34 | 1413.75 | 76.91 |
| 6 | 2.174938 | 3.68 | 277.15 | 2910.48 | 1600.90 | 100.43 | 69.45 | 62.46 | 1380.00 | 75.66 |
| 7 | 2.156533 | 3.71 | 268.49 | 2910.48 | 1600.90 | 100.45 | 74.60 | 62.87 | 1412.44 | 75.33 |

Ready result delivery-interval P50/P95/P99 ms: 270.94/274.45/275.67.
Separate latency-pass duration: 2.17 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

1.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.183517 | 3.66 | 272.18 | 2919.85 | 1600.90 | 100.45 | 70.25 | 61.98 | 1376.59 | 78.66 |
| 2 | 2.199307 | 3.64 | 281.28 | 2919.86 | 1600.90 | 100.50 | 70.61 | 62.59 | 1395.34 | 80.23 |
| 3 | 2.179191 | 3.67 | 271.17 | 2919.86 | 1600.90 | 100.44 | 70.86 | 63.13 | 1384.09 | 79.82 |
| 4 | 2.186396 | 3.66 | 279.79 | 2919.86 | 1600.90 | 100.47 | 71.45 | 61.99 | 1403.18 | 78.36 |
| 5 | 2.181975 | 3.67 | 271.38 | 2919.86 | 1600.90 | 100.50 | 68.93 | 62.30 | 1400.11 | 76.55 |
| 6 | 2.185993 | 3.66 | 277.25 | 2919.86 | 1600.90 | 100.46 | 69.45 | 61.48 | 1387.16 | 75.66 |
| 7 | 2.174283 | 3.68 | 272.31 | 2919.86 | 1600.90 | 100.44 | 73.36 | 62.17 | 1376.59 | 75.18 |

Ready result delivery-interval P50/P95/P99 ms: 272.33/272.90/272.96.
Separate latency-pass duration: 2.18 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

1.94% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.208208 | 3.62 | 280.30 | 2919.86 | 1600.90 | 100.56 | 71.45 | 62.09 | 1372.50 | 79.32 |
| 2 | 2.188128 | 3.66 | 279.68 | 2919.86 | 1600.90 | 100.42 | 71.64 | 62.34 | 1424.32 | 79.64 |
| 3 | 2.191865 | 3.65 | 279.41 | 2919.86 | 1600.90 | 100.44 | 71.23 | 62.42 | 1377.95 | 79.18 |
| 4 | 2.194265 | 3.65 | 279.13 | 2919.86 | 1600.90 | 100.48 | 70.14 | 62.60 | 1390.23 | 78.70 |
| 5 | 2.184994 | 3.66 | 280.89 | 2919.86 | 1600.90 | 100.45 | 71.39 | 61.63 | 1379.32 | 76.32 |
| 6 | 2.182017 | 3.67 | 278.58 | 2919.86 | 1600.90 | 100.44 | 69.45 | 61.94 | 1413.41 | 76.00 |
| 7 | 2.182536 | 3.67 | 278.21 | 2919.86 | 1600.90 | 100.40 | 74.52 | 62.67 | 1411.02 | 75.18 |

Ready result delivery-interval P50/P95/P99 ms: 272.26/276.42/277.59.
Separate latency-pass duration: 2.18 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — faster-rcnn, images, batch 1

Status: **complete**. Load 2.05 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 43.97 ± 0.19 | 24.98 | 22.73/24.08/24.41 | 1916.14 | 202.88/324.00 | 32/256 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 39.93 ± 0.39 | 26.43 | 24.76/25.67/26.33 | 1951.29 | 202.85/324.00 | 32/256 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 39.21 ± 0.27 | 26.93 | 28.91/29.93/30.02 | 1951.27 | 202.85/324.00 | 32/256 | PASS |
| managed-stream | 38.77 ± 0.70 | 26.90 | 29.05/30.08/30.87 | 1953.66 | 202.85/324.00 | 32/256 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.732227 | 43.70 | 25.88 | 1915.14 | 202.88 | 124.25 | 32.27 | 53.03 | 2040.00 | 73.67 |
| 2 | 0.733797 | 43.61 | 25.14 | 1915.86 | 202.88 | 123.90 | 32.47 | 52.49 | 2040.00 | 75.13 |
| 3 | 0.729624 | 43.86 | 24.29 | 1915.86 | 202.88 | 123.82 | 28.93 | 54.69 | 2040.00 | 75.27 |
| 4 | 0.727800 | 43.97 | 24.98 | 1916.08 | 202.88 | 123.99 | 30.47 | 52.13 | 2040.00 | 76.73 |
| 5 | 0.725966 | 44.08 | 24.49 | 1916.08 | 202.88 | 123.91 | 33.67 | 53.59 | 2040.00 | 76.87 |
| 6 | 0.726769 | 44.03 | 25.06 | 1916.14 | 202.88 | 123.95 | 29.47 | 52.74 | 2040.00 | 78.00 |
| 7 | 0.725433 | 44.11 | 24.16 | 1916.14 | 202.88 | 124.00 | 33.53 | 54.39 | 2040.00 | 78.13 |

Ready result delivery-interval P50/P95/P99 ms: 22.73/24.07/24.41.
Separate latency-pass duration: 0.73 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

10.11% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.791092 | 40.45 | 26.31 | 1915.14 | 202.85 | 122.05 | 27.25 | 53.35 | 2040.00 | 73.75 |
| 2 | 0.808462 | 39.58 | 27.17 | 1950.91 | 202.85 | 121.71 | 26.65 | 53.06 | 2040.00 | 74.76 |
| 3 | 0.806703 | 39.67 | 26.55 | 1915.87 | 202.85 | 121.86 | 32.75 | 53.72 | 2040.00 | 75.25 |
| 4 | 0.801344 | 39.93 | 26.43 | 1951.21 | 202.85 | 121.75 | 32.94 | 52.11 | 2040.00 | 76.50 |
| 5 | 0.804294 | 39.79 | 26.57 | 1916.08 | 202.85 | 121.62 | 26.94 | 52.03 | 2040.00 | 76.75 |
| 6 | 0.798115 | 40.09 | 26.29 | 1951.29 | 202.85 | 121.78 | 28.25 | 54.07 | 2040.00 | 77.62 |
| 7 | 0.788095 | 40.60 | 26.30 | 1916.14 | 202.85 | 121.92 | 32.75 | 55.09 | 2040.00 | 78.38 |

Ready result delivery-interval P50/P95/P99 ms: 24.77/25.67/26.32.
Separate latency-pass duration: 0.80 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

11.99% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.816050 | 39.21 | 26.46 | 1948.04 | 202.85 | 123.02 | 30.50 | 51.46 | 2040.00 | 73.94 |
| 2 | 0.814502 | 39.29 | 27.13 | 1950.80 | 202.85 | 122.16 | 31.35 | 54.09 | 2040.00 | 74.88 |
| 3 | 0.817713 | 39.13 | 27.02 | 1951.05 | 202.85 | 122.95 | 31.71 | 54.01 | 2040.00 | 75.53 |
| 4 | 0.807680 | 39.62 | 26.78 | 1948.90 | 202.85 | 122.04 | 30.00 | 54.67 | 2040.00 | 76.38 |
| 5 | 0.825251 | 38.78 | 26.93 | 1951.24 | 202.85 | 122.73 | 31.24 | 51.72 | 2040.00 | 76.76 |
| 6 | 0.819427 | 39.05 | 27.02 | 1951.27 | 202.85 | 121.72 | 32.35 | 51.58 | 2040.00 | 77.53 |
| 7 | 0.812429 | 39.39 | 26.45 | 1948.96 | 202.85 | 123.12 | 30.06 | 53.53 | 2040.00 | 78.38 |

Ready result delivery-interval P50/P95/P99 ms: 28.91/29.94/30.02.
Separate latency-pass duration: 0.93 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

13.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.824554 | 38.81 | 26.50 | 1951.68 | 202.85 | 121.26 | 32.44 | 50.97 | 2040.00 | 74.00 |
| 2 | 0.842094 | 38.00 | 27.20 | 1951.84 | 202.85 | 120.80 | 30.59 | 51.73 | 2040.00 | 74.24 |
| 3 | 0.825451 | 38.77 | 26.90 | 1951.14 | 202.85 | 121.16 | 26.65 | 51.72 | 2040.00 | 75.76 |
| 4 | 0.815495 | 39.24 | 26.86 | 1953.53 | 202.85 | 121.44 | 27.12 | 51.35 | 2040.00 | 76.12 |
| 5 | 0.828195 | 38.64 | 26.78 | 1953.60 | 202.85 | 121.13 | 32.71 | 52.36 | 2040.00 | 77.29 |
| 6 | 0.857531 | 37.32 | 26.92 | 1948.91 | 202.85 | 121.75 | 28.94 | 52.22 | 2040.00 | 77.24 |
| 7 | 0.814524 | 39.29 | 26.93 | 1953.66 | 202.85 | 121.54 | 27.29 | 53.29 | 2040.00 | 78.47 |

Ready result delivery-interval P50/P95/P99 ms: 29.06/30.09/30.74.
Separate latency-pass duration: 0.94 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — d-fine, images, batch 1

Status: **complete**. Load 21.37 s. Device `cuda:0`, dtype `float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.67 ± 0.02 | 373.29 | 376.43/379.73/380.63 | 3192.07 | 146.90/238.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.64 ± 0.01 | 377.95 | 378.54/382.27/383.05 | 3192.09 | 152.71/238.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.65 ± 0.01 | 377.64 | 377.62/380.04/380.75 | 3201.10 | 154.29/238.00 | 8/40 | PASS |
| managed-stream | 2.64 ± 0.01 | 377.75 | 376.94/379.61/379.98 | 3201.22 | 142.60/238.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.041674 | 2.63 | 389.58 | 3191.51 | 145.51 | 101.84 | 2.25 | 35.18 | 2040.00 | 75.00 |
| 2 | 3.017369 | 2.65 | 375.74 | 3191.72 | 137.52 | 101.81 | 2.28 | 35.55 | 2040.00 | 76.00 |
| 3 | 2.996627 | 2.67 | 374.38 | 3191.72 | 137.35 | 101.81 | 2.27 | 35.44 | 2040.00 | 76.00 |
| 4 | 2.988323 | 2.68 | 372.12 | 3191.87 | 140.80 | 101.78 | 2.22 | 35.71 | 2040.00 | 77.00 |
| 5 | 2.993033 | 2.67 | 372.95 | 3191.89 | 140.80 | 101.77 | 2.36 | 35.80 | 2040.00 | 77.00 |
| 6 | 2.995936 | 2.67 | 372.56 | 3192.05 | 146.90 | 101.79 | 2.20 | 35.73 | 2040.00 | 77.00 |
| 7 | 2.993306 | 2.67 | 373.29 | 3192.07 | 139.12 | 101.77 | 2.24 | 35.79 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 376.43/379.75/380.64.
Separate latency-pass duration: 3.01 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.79% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.026384 | 2.64 | 382.53 | 3191.53 | 152.71 | 101.81 | 2.18 | 35.26 | 2040.00 | 75.07 |
| 2 | 3.022233 | 2.65 | 377.95 | 3191.40 | 148.59 | 101.77 | 2.32 | 35.47 | 2040.00 | 76.00 |
| 3 | 3.020269 | 2.65 | 375.80 | 3191.75 | 150.77 | 101.76 | 2.42 | 35.57 | 2040.00 | 76.00 |
| 4 | 3.028848 | 2.64 | 376.90 | 3191.57 | 141.82 | 101.83 | 2.37 | 35.77 | 2040.00 | 76.95 |
| 5 | 3.009558 | 2.66 | 373.35 | 3191.91 | 151.16 | 101.82 | 2.20 | 35.70 | 2040.00 | 77.00 |
| 6 | 3.035761 | 2.64 | 379.81 | 3191.74 | 148.59 | 101.78 | 2.30 | 35.73 | 2040.00 | 77.00 |
| 7 | 3.051552 | 2.62 | 379.48 | 3192.09 | 147.22 | 101.84 | 2.23 | 35.60 | 2040.00 | 76.85 |

Ready result delivery-interval P50/P95/P99 ms: 378.54/382.27/383.05.
Separate latency-pass duration: 3.03 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

0.92% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.014482 | 2.65 | 377.62 | 3200.64 | 148.59 | 101.90 | 2.28 | 35.35 | 2040.00 | 75.55 |
| 2 | 3.027547 | 2.64 | 380.20 | 3200.74 | 154.29 | 101.91 | 2.25 | 35.33 | 2040.00 | 76.00 |
| 3 | 3.023246 | 2.65 | 375.33 | 3200.87 | 151.16 | 101.87 | 2.33 | 35.54 | 2040.00 | 76.00 |
| 4 | 3.016438 | 2.65 | 379.13 | 3200.93 | 151.14 | 101.90 | 2.25 | 35.62 | 2040.00 | 76.73 |
| 5 | 3.026786 | 2.64 | 377.64 | 3201.02 | 147.22 | 101.92 | 2.18 | 35.87 | 2040.00 | 77.00 |
| 6 | 3.040222 | 2.63 | 379.61 | 3201.10 | 151.14 | 101.98 | 2.20 | 35.71 | 2040.00 | 77.00 |
| 7 | 3.020965 | 2.65 | 376.53 | 3200.67 | 147.22 | 101.91 | 2.20 | 35.61 | 2040.00 | 76.63 |

Ready result delivery-interval P50/P95/P99 ms: 377.62/380.04/380.75.
Separate latency-pass duration: 3.02 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

0.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.011924 | 2.66 | 374.23 | 3200.68 | 142.53 | 101.79 | 2.36 | 35.46 | 2040.00 | 75.93 |
| 2 | 3.040278 | 2.63 | 382.00 | 3200.70 | 142.60 | 101.92 | 2.20 | 35.32 | 2040.00 | 76.00 |
| 3 | 3.045021 | 2.63 | 385.00 | 3200.90 | 142.60 | 102.04 | 2.28 | 35.59 | 2040.00 | 76.00 |
| 4 | 3.034023 | 2.64 | 376.39 | 3200.91 | 142.53 | 101.87 | 2.28 | 35.59 | 2040.00 | 76.28 |
| 5 | 3.027959 | 2.64 | 374.67 | 3201.05 | 142.53 | 101.85 | 2.22 | 35.88 | 2040.00 | 77.00 |
| 6 | 3.015793 | 2.65 | 377.75 | 3201.08 | 142.21 | 101.92 | 2.22 | 35.82 | 2040.00 | 77.00 |
| 7 | 3.013149 | 2.66 | 380.69 | 3201.22 | 142.21 | 101.93 | 2.30 | 35.66 | 2040.00 | 76.90 |

Ready result delivery-interval P50/P95/P99 ms: 376.84/379.61/379.96.
Separate latency-pass duration: 3.02 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=8.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — locate-anything, images, batch 1

Status: **complete**. Load 55.95 s. Device `cuda:0`, dtype `torch.float16`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.44 ± 0.02 | 409.84 | 407.37/407.37/407.37 | 4172.50 | 7910.11/8364.00 | 1/4 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.44 ± 0.03 | 409.53 | 411.25/411.25/411.25 | 4172.50 | 7910.11/8364.00 | 1/4 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.44 ± 0.02 | 409.49 | 425.16/425.16/425.16 | 4172.50 | 7910.11/8364.00 | 1/4 | PASS |
| managed-stream | 2.42 ± 0.01 | 412.27 | 411.46/411.46/411.46 | 4172.51 | 7910.11/8364.00 | 1/4 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.413371 | 2.42 | 413.32 | 4172.50 | 7910.11 | 102.44 | 48.56 | 52.39 | 1681.67 | 79.00 |
| 2 | 0.410536 | 2.44 | 410.49 | 4172.50 | 7910.11 | 102.41 | 48.11 | 52.73 | 1715.00 | 80.22 |
| 3 | 0.415933 | 2.40 | 415.88 | 4172.50 | 7910.11 | 102.43 | 36.44 | 50.09 | 1773.33 | 80.56 |
| 4 | 0.407730 | 2.45 | 407.68 | 4172.50 | 7910.11 | 102.57 | 35.56 | 50.14 | 1800.00 | 80.78 |
| 5 | 0.409885 | 2.44 | 409.84 | 4172.50 | 7910.11 | 102.46 | 35.78 | 53.92 | 1688.33 | 80.11 |
| 6 | 0.407957 | 2.45 | 407.90 | 4172.50 | 7910.11 | 102.48 | 48.89 | 51.10 | 1786.67 | 80.56 |
| 7 | 0.409069 | 2.44 | 409.02 | 4172.50 | 7910.11 | 102.38 | 49.89 | 53.28 | 1725.00 | 79.78 |

Ready result delivery-interval P50/P95/P99 ms: 407.36/407.36/407.36.
Separate latency-pass duration: 0.41 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.33% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.409584 | 2.44 | 409.53 | 4172.50 | 7910.11 | 102.46 | 39.11 | 50.70 | 1753.33 | 79.56 |
| 2 | 0.422864 | 2.36 | 422.81 | 4172.50 | 7910.11 | 102.40 | 49.89 | 54.39 | 1705.00 | 79.78 |
| 3 | 0.411390 | 2.43 | 411.33 | 4172.50 | 7910.11 | 102.49 | 37.78 | 52.73 | 1810.00 | 80.11 |
| 4 | 0.409395 | 2.44 | 409.32 | 4172.50 | 7910.11 | 102.52 | 48.22 | 53.45 | 1771.67 | 80.33 |
| 5 | 0.409333 | 2.44 | 409.28 | 4172.50 | 7910.11 | 102.42 | 49.44 | 53.95 | 1778.33 | 80.78 |
| 6 | 0.409322 | 2.44 | 409.26 | 4172.50 | 7910.11 | 102.45 | 42.67 | 52.62 | 1725.00 | 80.11 |
| 7 | 0.414807 | 2.41 | 414.75 | 4172.50 | 7910.11 | 102.63 | 48.67 | 53.90 | 1725.00 | 79.56 |

Ready result delivery-interval P50/P95/P99 ms: 411.24/411.24/411.24.
Separate latency-pass duration: 0.41 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

0.05% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.405199 | 2.47 | 405.12 | 4172.50 | 7910.11 | 102.46 | 39.56 | 55.12 | 1796.67 | 79.56 |
| 2 | 0.410746 | 2.43 | 410.68 | 4172.50 | 7910.11 | 102.48 | 57.33 | 54.80 | 1793.33 | 79.56 |
| 3 | 0.413001 | 2.42 | 412.94 | 4172.50 | 7910.11 | 102.40 | 36.78 | 53.78 | 1691.67 | 80.00 |
| 4 | 0.408879 | 2.45 | 408.82 | 4172.50 | 7910.11 | 102.47 | 50.00 | 54.16 | 1701.67 | 80.33 |
| 5 | 0.408486 | 2.45 | 408.43 | 4172.50 | 7910.11 | 102.34 | 57.11 | 51.53 | 1806.67 | 80.56 |
| 6 | 0.409546 | 2.44 | 409.49 | 4172.50 | 7910.11 | 102.47 | 38.22 | 52.71 | 1796.67 | 80.11 |
| 7 | 0.412762 | 2.42 | 412.70 | 4172.50 | 7910.11 | 102.40 | 39.56 | 52.08 | 1773.33 | 79.56 |

Ready result delivery-interval P50/P95/P99 ms: 425.15/425.15/425.15.
Separate latency-pass duration: 0.43 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

0.16% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.413237 | 2.42 | 412.97 | 4172.51 | 7910.11 | 102.44 | 46.78 | 53.58 | 1688.33 | 79.11 |
| 2 | 0.410819 | 2.43 | 410.59 | 4172.51 | 7910.11 | 102.48 | 48.89 | 50.46 | 1770.00 | 79.78 |
| 3 | 0.412492 | 2.42 | 412.27 | 4172.51 | 7910.11 | 102.45 | 47.22 | 53.56 | 1761.67 | 80.11 |
| 4 | 0.409177 | 2.44 | 408.94 | 4172.51 | 7910.11 | 102.44 | 49.78 | 50.60 | 1790.00 | 80.78 |
| 5 | 0.410559 | 2.44 | 410.32 | 4172.51 | 7910.11 | 102.35 | 48.78 | 52.08 | 1735.00 | 80.11 |
| 6 | 0.413108 | 2.42 | 412.85 | 4172.51 | 7910.11 | 102.39 | 39.56 | 50.63 | 1766.67 | 80.56 |
| 7 | 0.416092 | 2.40 | 415.85 | 4172.51 | 7910.11 | 102.52 | 35.56 | 50.52 | 1793.33 | 79.33 |

Ready result delivery-interval P50/P95/P99 ms: 411.25/411.25/411.25.
Separate latency-pass duration: 0.41 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=1.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo, images, batch 4

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 248.41 ± 3.64 | 18.23 | 16.09/18.07/22.29 | 2189.70 | 76.00/186.00 | 256/1280 | PASS |
| raw-stream | 246.08 ± 2.54 | 18.19 | 16.46/17.92/20.48 | 2275.46 | 76.00/186.00 | 256/1280 | PASS |
| native-non-stream | 171.56 ± 3.12 | 24.26 | 22.73/25.40/26.58 | 2290.47 | 75.87/186.00 | 256/1280 | PASS |
| native-stream | 168.80 ± 2.21 | 28.86 | 23.78/25.14/25.85 | 2297.68 | 75.87/186.00 | 256/1280 | PASS |
| managed-non-stream | 140.21 ± 1.28 | 29.49 | 28.04/29.15/30.54 | 2305.23 | 75.87/186.00 | 256/1280 | PASS |
| managed-stream | 131.37 ± 1.56 | 30.52 | 30.78/32.79/34.01 | 2416.58 | 75.87/186.00 | 256/1280 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.054837 | 242.69 | 23.33 | 2179.29 | 76.00 | 100.49 | 19.67 | 44.73 | 2040.00 | 74.81 |
| 2 | 1.030572 | 248.41 | 17.89 | 2189.63 | 76.00 | 100.74 | 22.00 | 48.22 | 2040.00 | 75.90 |
| 3 | 1.019397 | 251.13 | 17.82 | 2189.70 | 76.00 | 100.48 | 19.19 | 45.31 | 2040.00 | 76.00 |
| 4 | 1.053275 | 243.05 | 18.17 | 2104.52 | 76.00 | 100.51 | 20.90 | 49.74 | 2040.00 | 77.62 |
| 5 | 1.028096 | 249.00 | 18.23 | 2104.52 | 76.00 | 100.49 | 18.48 | 45.69 | 2040.00 | 77.81 |
| 6 | 1.056584 | 242.29 | 18.51 | 2104.90 | 76.00 | 100.51 | 19.33 | 48.34 | 2040.00 | 77.81 |
| 7 | 1.030171 | 248.50 | 18.68 | 2104.90 | 76.00 | 100.50 | 19.43 | 46.30 | 2040.00 | 77.81 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/16.71/18.45.
Separate latency-pass duration: 1.06 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.5564384804692679, 'inference': 1.7915830468790972, 'postprocess': 1.036830675779754}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.46% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.048172 | 244.23 | 17.95 | 2062.73 | 76.00 | 100.45 | 20.90 | 47.78 | 2040.00 | 75.10 |
| 2 | 1.020447 | 250.87 | 22.24 | 2275.46 | 76.00 | 100.42 | 21.24 | 46.20 | 2040.00 | 75.81 |
| 3 | 1.046179 | 244.70 | 17.84 | 2189.70 | 76.00 | 100.45 | 21.05 | 48.21 | 2040.00 | 76.52 |
| 4 | 1.048404 | 244.18 | 17.73 | 2097.48 | 76.00 | 100.50 | 20.48 | 45.42 | 2040.00 | 77.71 |
| 5 | 1.031305 | 248.23 | 18.19 | 2104.52 | 76.00 | 100.49 | 19.38 | 48.56 | 2040.00 | 77.95 |
| 6 | 1.040309 | 246.08 | 22.74 | 2191.03 | 76.00 | 100.49 | 19.81 | 47.35 | 2040.00 | 77.90 |
| 7 | 1.031300 | 248.23 | 18.19 | 2104.90 | 76.00 | 100.49 | 20.10 | 49.61 | 2040.00 | 77.90 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/17.11/18.49.
Separate latency-pass duration: 1.07 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.5513019296934374, 'inference': 1.7969618867237358, 'postprocess': 1.0370997734323595}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

42.84% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.501854 | 170.46 | 23.69 | 2152.12 | 75.87 | 100.50 | 15.93 | 45.51 | 2040.00 | 74.93 |
| 2 | 1.465441 | 174.69 | 29.20 | 2290.47 | 75.87 | 100.41 | 16.34 | 44.46 | 2040.00 | 75.38 |
| 3 | 1.536531 | 166.61 | 23.82 | 2279.78 | 75.87 | 100.52 | 15.10 | 46.43 | 2040.00 | 76.06 |
| 4 | 1.519792 | 168.44 | 24.24 | 2197.09 | 75.87 | 100.48 | 14.39 | 44.98 | 2040.00 | 76.90 |
| 5 | 1.468555 | 174.32 | 24.26 | 2190.66 | 75.87 | 100.50 | 14.97 | 45.97 | 2040.00 | 77.77 |
| 6 | 1.472015 | 173.91 | 28.66 | 2191.03 | 75.87 | 100.47 | 16.13 | 45.74 | 2040.00 | 77.40 |
| 7 | 1.492181 | 171.56 | 24.52 | 2186.05 | 75.87 | 100.52 | 15.20 | 47.54 | 2040.00 | 77.80 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/24.60/25.56.
Separate latency-pass duration: 1.50 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

45.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.535879 | 166.68 | 28.86 | 2167.16 | 75.87 | 100.53 | 15.71 | 46.06 | 2040.00 | 74.87 |
| 2 | 1.516584 | 168.80 | 28.52 | 2291.09 | 75.87 | 100.49 | 14.40 | 44.92 | 2040.00 | 75.27 |
| 3 | 1.527784 | 167.56 | 29.27 | 2297.68 | 75.87 | 100.50 | 14.58 | 43.91 | 2040.00 | 76.13 |
| 4 | 1.518389 | 168.60 | 29.57 | 2198.86 | 75.87 | 100.50 | 15.16 | 44.17 | 2040.00 | 76.87 |
| 5 | 1.491409 | 171.65 | 28.43 | 2190.66 | 75.87 | 100.50 | 14.80 | 43.89 | 2040.00 | 77.77 |
| 6 | 1.483023 | 172.62 | 29.53 | 2199.82 | 75.87 | 100.45 | 15.50 | 43.84 | 2040.00 | 77.43 |
| 7 | 1.498232 | 170.87 | 28.70 | 2191.03 | 75.87 | 100.49 | 16.53 | 45.83 | 2040.00 | 77.63 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/24.48/25.20.
Separate latency-pass duration: 1.51 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

75.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.825892 | 140.21 | 29.54 | 2167.69 | 75.87 | 100.47 | 12.38 | 43.25 | 2040.00 | 74.78 |
| 2 | 1.804804 | 141.84 | 29.62 | 2290.47 | 75.87 | 100.47 | 13.22 | 42.08 | 2040.00 | 74.89 |
| 3 | 1.833469 | 139.63 | 29.38 | 2305.23 | 75.87 | 100.49 | 13.24 | 42.81 | 2040.00 | 76.14 |
| 4 | 1.835386 | 139.48 | 29.32 | 2193.87 | 75.87 | 100.48 | 12.00 | 42.55 | 2040.00 | 76.62 |
| 5 | 1.848413 | 138.50 | 29.32 | 2195.34 | 75.87 | 100.52 | 11.89 | 43.18 | 2040.00 | 77.46 |
| 6 | 1.805432 | 141.79 | 29.51 | 2195.71 | 75.87 | 100.49 | 12.33 | 43.41 | 2040.00 | 77.33 |
| 7 | 1.813091 | 141.20 | 29.49 | 2195.72 | 75.87 | 100.49 | 12.50 | 43.79 | 2040.00 | 77.11 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/28.47/29.20.
Separate latency-pass duration: 1.80 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

87.54% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.978214 | 129.41 | 30.46 | 2416.58 | 75.87 | 100.79 | 11.50 | 42.03 | 2040.00 | 74.80 |
| 2 | 1.983831 | 129.04 | 30.19 | 2415.07 | 75.87 | 100.74 | 10.90 | 40.67 | 2040.00 | 74.70 |
| 3 | 1.938586 | 132.06 | 40.23 | 2305.78 | 75.87 | 100.76 | 11.69 | 42.98 | 2040.00 | 76.31 |
| 4 | 1.946782 | 131.50 | 29.89 | 2308.73 | 75.87 | 100.84 | 11.38 | 41.04 | 2040.00 | 76.31 |
| 5 | 1.970061 | 129.95 | 30.52 | 2404.70 | 75.87 | 100.95 | 12.10 | 42.69 | 2040.00 | 77.44 |
| 6 | 1.948689 | 131.37 | 35.35 | 2399.35 | 75.87 | 100.74 | 11.69 | 41.44 | 2040.00 | 77.26 |
| 7 | 1.919497 | 133.37 | 30.55 | 2405.80 | 75.87 | 100.79 | 12.55 | 43.09 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/31.81/32.84.
Separate latency-pass duration: 1.96 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo11, images, batch 4

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 210.24 ± 5.00 | 25.67 | 19.89/20.92/28.64 | 2295.69 | 112.51/184.00 | 256/1344 | PASS |
| raw-stream | 209.94 ± 3.62 | 25.41 | 19.20/22.40/25.01 | 2273.06 | 112.51/184.00 | 256/1344 | PASS |
| native-non-stream | 154.17 ± 4.35 | 31.51 | 25.78/27.20/31.47 | 2272.10 | 112.39/184.00 | 256/1344 | PASS |
| native-stream | 155.12 ± 1.49 | 35.56 | 26.09/28.02/33.91 | 2288.88 | 112.39/184.00 | 256/1344 | PASS |
| managed-non-stream | 128.69 ± 2.74 | 31.36 | 30.11/31.73/33.09 | 2396.39 | 112.39/184.00 | 256/1344 | PASS |
| managed-stream | 124.19 ± 1.69 | 35.69 | 31.82/33.87/34.82 | 2465.71 | 112.39/184.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.280847 | 199.87 | 36.11 | 2295.69 | 112.51 | 100.49 | 19.27 | 45.37 | 2040.00 | 76.31 |
| 2 | 1.240046 | 206.44 | 25.37 | 2182.75 | 112.51 | 100.44 | 18.84 | 45.88 | 2040.00 | 77.00 |
| 3 | 1.200085 | 213.32 | 25.23 | 2182.88 | 112.51 | 100.45 | 18.08 | 44.55 | 2040.00 | 77.00 |
| 4 | 1.208306 | 211.87 | 25.54 | 2188.73 | 112.51 | 100.49 | 18.62 | 46.57 | 2040.00 | 76.96 |
| 5 | 1.217636 | 210.24 | 25.67 | 2188.73 | 112.51 | 100.50 | 18.60 | 44.94 | 2040.00 | 76.92 |
| 6 | 1.205544 | 212.35 | 25.94 | 2189.86 | 112.51 | 100.49 | 18.83 | 47.39 | 2040.00 | 76.83 |
| 7 | 1.254258 | 204.10 | 26.88 | 2189.86 | 112.51 | 100.45 | 17.92 | 43.94 | 2040.00 | 76.80 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/20.47/20.92.
Separate latency-pass duration: 1.26 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8003994687548754, 'inference': 2.2569766132800595, 'postprocess': 1.0489797812507362}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.70% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.192362 | 214.70 | 24.54 | 2049.24 | 112.51 | 100.47 | 19.67 | 47.09 | 2040.00 | 76.50 |
| 2 | 1.254632 | 204.04 | 29.61 | 2270.06 | 112.51 | 100.47 | 18.96 | 44.81 | 2040.00 | 77.08 |
| 3 | 1.208506 | 211.83 | 24.79 | 2182.88 | 112.51 | 100.45 | 18.71 | 46.91 | 2040.00 | 77.17 |
| 4 | 1.244966 | 205.63 | 19.70 | 2185.21 | 112.51 | 100.49 | 17.48 | 45.66 | 2040.00 | 76.92 |
| 5 | 1.219407 | 209.94 | 25.41 | 2188.73 | 112.51 | 100.55 | 18.08 | 48.08 | 2040.00 | 77.28 |
| 6 | 1.215397 | 210.63 | 29.35 | 2273.06 | 112.51 | 100.51 | 18.96 | 45.35 | 2040.00 | 76.84 |
| 7 | 1.224324 | 209.10 | 26.39 | 2189.86 | 112.51 | 100.42 | 19.88 | 45.67 | 2040.00 | 76.76 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/20.34/22.79.
Separate latency-pass duration: 1.25 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6985841796929293, 'inference': 2.236123898441633, 'postprocess': 1.031066402349623}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

37.74% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.625608 | 157.48 | 30.44 | 2145.67 | 112.39 | 100.50 | 15.45 | 45.00 | 2040.00 | 76.52 |
| 2 | 1.779966 | 143.82 | 34.68 | 2270.98 | 112.39 | 100.66 | 12.86 | 42.92 | 2040.00 | 76.94 |
| 3 | 1.672897 | 153.03 | 31.79 | 2272.10 | 112.39 | 100.45 | 14.82 | 45.34 | 2040.00 | 76.94 |
| 4 | 1.666524 | 153.61 | 26.36 | 2263.14 | 112.39 | 100.48 | 14.79 | 44.71 | 2040.00 | 76.94 |
| 5 | 1.650622 | 155.09 | 31.51 | 2263.14 | 112.39 | 100.65 | 15.00 | 45.48 | 2040.00 | 76.94 |
| 6 | 1.660512 | 154.17 | 25.69 | 2268.08 | 112.39 | 100.50 | 14.27 | 43.30 | 2040.00 | 76.15 |
| 7 | 1.653781 | 154.80 | 31.76 | 2268.08 | 112.39 | 100.51 | 14.91 | 45.06 | 2040.00 | 76.48 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/26.22/27.62.
Separate latency-pass duration: 1.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

35.49% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.663714 | 153.87 | 34.67 | 2146.63 | 112.39 | 100.47 | 15.03 | 44.83 | 2040.00 | 76.76 |
| 2 | 1.640131 | 156.09 | 34.83 | 2270.47 | 112.39 | 100.45 | 14.00 | 45.07 | 2040.00 | 76.88 |
| 3 | 1.632092 | 156.85 | 35.56 | 2272.28 | 112.39 | 100.45 | 14.30 | 44.95 | 2040.00 | 76.94 |
| 4 | 1.650289 | 155.12 | 35.99 | 2282.48 | 112.39 | 100.50 | 14.55 | 45.06 | 2040.00 | 76.94 |
| 5 | 1.649813 | 155.17 | 25.76 | 2264.90 | 112.39 | 100.44 | 13.82 | 45.39 | 2040.00 | 76.94 |
| 6 | 1.653325 | 154.84 | 37.06 | 2288.88 | 112.39 | 100.48 | 14.18 | 41.69 | 2040.00 | 76.24 |
| 7 | 1.681399 | 152.25 | 37.12 | 2273.06 | 112.39 | 100.49 | 14.29 | 43.16 | 2040.00 | 76.29 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/26.63/28.07.
Separate latency-pass duration: 1.67 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

63.37% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.950095 | 131.28 | 31.27 | 2143.96 | 112.39 | 100.50 | 12.74 | 43.44 | 2040.00 | 76.64 |
| 2 | 1.913152 | 133.81 | 37.38 | 2396.39 | 112.39 | 100.46 | 12.42 | 44.19 | 2040.00 | 76.89 |
| 3 | 1.993655 | 128.41 | 30.98 | 2276.43 | 112.39 | 100.46 | 12.10 | 43.05 | 2040.00 | 76.95 |
| 4 | 2.014689 | 127.07 | 38.02 | 2277.50 | 112.39 | 100.49 | 12.10 | 44.06 | 2040.00 | 76.95 |
| 5 | 1.989259 | 128.69 | 31.36 | 2270.58 | 112.39 | 100.47 | 12.20 | 43.62 | 2040.00 | 76.70 |
| 6 | 2.022755 | 126.56 | 38.10 | 2288.88 | 112.39 | 100.49 | 11.93 | 43.09 | 2040.00 | 76.05 |
| 7 | 1.936493 | 132.20 | 31.36 | 2276.98 | 112.39 | 100.49 | 12.15 | 43.44 | 2040.00 | 76.21 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/30.86/32.04.
Separate latency-pass duration: 1.94 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

67.98% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.104695 | 121.63 | 34.97 | 2465.71 | 112.39 | 100.71 | 11.33 | 41.94 | 2040.00 | 76.67 |
| 2 | 2.083014 | 122.90 | 36.53 | 2452.00 | 112.39 | 100.74 | 11.24 | 43.12 | 2040.00 | 76.93 |
| 3 | 2.079183 | 123.13 | 44.04 | 2281.99 | 112.39 | 100.78 | 11.81 | 41.96 | 2040.00 | 76.90 |
| 4 | 2.016055 | 126.98 | 35.47 | 2258.04 | 112.39 | 100.74 | 11.30 | 42.76 | 2040.00 | 76.92 |
| 5 | 2.061298 | 124.19 | 42.57 | 2273.40 | 112.39 | 100.82 | 11.76 | 41.94 | 2040.00 | 76.63 |
| 6 | 2.060193 | 124.26 | 35.69 | 2268.25 | 112.39 | 100.76 | 11.12 | 39.98 | 2040.00 | 76.24 |
| 7 | 2.052570 | 124.72 | 33.76 | 2459.86 | 112.39 | 100.76 | 12.07 | 43.06 | 2040.00 | 76.10 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/32.58/33.79.
Separate latency-pass duration: 2.05 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo26, images, batch 4

Status: **complete**. Load 0.13 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 195.69 ± 1.75 | 28.03 | 20.78/21.63/28.66 | 2239.93 | 111.72/184.00 | 256/1344 | PASS |
| raw-stream | 196.73 ± 3.44 | 28.58 | 20.32/24.58/26.25 | 2119.08 | 111.72/184.00 | 256/1344 | PASS |
| native-non-stream | 146.47 ± 5.32 | 35.27 | 27.23/29.42/32.89 | 2119.08 | 111.60/184.00 | 256/1344 | PASS |
| native-stream | 144.16 ± 3.85 | 38.60 | 25.75/27.36/29.06 | 2139.33 | 111.60/184.00 | 256/1344 | PASS |
| managed-non-stream | 126.67 ± 1.25 | 33.15 | 31.13/32.65/34.64 | 2235.09 | 111.60/184.00 | 256/1344 | PASS |
| managed-stream | 118.32 ± 2.19 | 37.04 | 33.67/35.03/36.00 | 2394.56 | 111.60/184.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.308201 | 195.69 | 39.45 | 2239.93 | 111.72 | 100.48 | 18.69 | 44.99 | 2040.00 | 76.81 |
| 2 | 1.317657 | 194.28 | 28.03 | 2018.82 | 111.72 | 100.48 | 18.67 | 46.24 | 2040.00 | 77.93 |
| 3 | 1.315856 | 194.55 | 27.98 | 2018.91 | 111.72 | 100.51 | 17.59 | 43.81 | 2040.00 | 77.85 |
| 4 | 1.288594 | 198.67 | 27.86 | 2025.66 | 111.72 | 100.47 | 17.88 | 45.92 | 2040.00 | 76.85 |
| 5 | 1.302726 | 196.51 | 27.83 | 2025.66 | 111.72 | 100.47 | 18.12 | 44.05 | 2040.00 | 76.81 |
| 6 | 1.316101 | 194.51 | 28.15 | 2025.91 | 111.72 | 100.49 | 17.67 | 45.53 | 2040.00 | 76.30 |
| 7 | 1.293462 | 197.92 | 28.06 | 2025.92 | 111.72 | 100.48 | 18.12 | 44.01 | 2040.00 | 76.27 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/21.13/21.62.
Separate latency-pass duration: 1.33 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.7322089843722068, 'inference': 2.578452121094088, 'postprocess': 1.0534998046836108}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.301299 | 196.73 | 27.26 | 1986.93 | 111.72 | 100.43 | 19.04 | 46.87 | 2040.00 | 76.85 |
| 2 | 1.339842 | 191.07 | 31.48 | 2101.72 | 111.72 | 100.45 | 17.96 | 45.29 | 2040.00 | 77.89 |
| 3 | 1.296063 | 197.52 | 28.04 | 2018.91 | 111.72 | 100.45 | 19.31 | 46.57 | 2040.00 | 77.85 |
| 4 | 1.282904 | 199.55 | 31.80 | 2110.03 | 111.72 | 100.45 | 18.69 | 46.26 | 2040.00 | 76.85 |
| 5 | 1.309906 | 195.43 | 28.18 | 2025.66 | 111.72 | 100.73 | 17.46 | 45.92 | 2040.00 | 76.92 |
| 6 | 1.340161 | 191.02 | 32.82 | 2119.08 | 111.72 | 100.47 | 17.52 | 46.04 | 2040.00 | 75.93 |
| 7 | 1.288819 | 198.63 | 28.58 | 2025.92 | 111.72 | 100.49 | 18.58 | 46.50 | 2040.00 | 76.46 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/20.59/24.57.
Separate latency-pass duration: 1.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.7329225234382619, 'inference': 2.5861584414119676, 'postprocess': 1.0505986054631933}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

33.43% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.743222 | 146.85 | 33.60 | 2064.04 | 111.60 | 100.47 | 13.51 | 44.72 | 2040.00 | 76.94 |
| 2 | 1.923775 | 133.07 | 39.33 | 2103.82 | 111.60 | 100.64 | 13.23 | 43.82 | 2040.00 | 77.51 |
| 3 | 1.755742 | 145.81 | 35.27 | 2108.41 | 111.60 | 100.49 | 13.83 | 45.14 | 2040.00 | 77.66 |
| 4 | 1.786846 | 143.27 | 38.71 | 2110.03 | 111.60 | 100.55 | 13.25 | 44.02 | 2040.00 | 76.89 |
| 5 | 1.719306 | 148.90 | 34.09 | 2118.82 | 111.60 | 100.47 | 14.86 | 44.04 | 2040.00 | 76.71 |
| 6 | 1.747823 | 146.47 | 38.52 | 2119.08 | 111.60 | 100.48 | 14.71 | 43.94 | 2040.00 | 75.94 |
| 7 | 1.739567 | 147.16 | 34.14 | 2110.30 | 111.60 | 100.49 | 13.74 | 44.39 | 2040.00 | 76.11 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/27.69/29.73.
Separate latency-pass duration: 1.75 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

35.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.861168 | 137.55 | 27.07 | 2065.36 | 111.60 | 100.50 | 13.97 | 44.17 | 2040.00 | 77.05 |
| 2 | 1.801872 | 142.07 | 38.67 | 2122.04 | 111.60 | 100.50 | 14.56 | 43.62 | 2040.00 | 77.56 |
| 3 | 1.778165 | 143.97 | 38.60 | 2113.71 | 111.60 | 100.50 | 14.78 | 43.77 | 2040.00 | 77.42 |
| 4 | 1.775755 | 144.16 | 39.23 | 2139.33 | 111.60 | 100.52 | 13.33 | 43.11 | 2040.00 | 76.94 |
| 5 | 1.736498 | 147.42 | 38.17 | 2118.82 | 111.60 | 100.46 | 14.83 | 43.20 | 2040.00 | 76.71 |
| 6 | 1.720313 | 148.81 | 38.78 | 2130.21 | 111.60 | 100.45 | 13.83 | 43.62 | 2040.00 | 75.94 |
| 7 | 1.738824 | 147.23 | 37.75 | 2110.30 | 111.60 | 100.50 | 14.89 | 44.61 | 2040.00 | 76.23 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/26.89/27.38.
Separate latency-pass duration: 1.66 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

55.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.018348 | 126.84 | 33.04 | 2084.36 | 111.60 | 100.50 | 12.82 | 42.39 | 2040.00 | 77.17 |
| 2 | 2.047447 | 125.03 | 39.23 | 2226.28 | 111.60 | 100.52 | 12.76 | 41.28 | 2040.00 | 77.46 |
| 3 | 2.048740 | 124.95 | 33.15 | 2133.01 | 111.60 | 100.51 | 12.10 | 44.09 | 2040.00 | 77.17 |
| 4 | 2.021020 | 126.67 | 39.60 | 2234.82 | 111.60 | 100.49 | 12.72 | 42.38 | 2040.00 | 76.90 |
| 5 | 2.053407 | 124.67 | 32.92 | 2125.27 | 111.60 | 100.50 | 12.05 | 43.67 | 2040.00 | 76.41 |
| 6 | 2.006268 | 127.60 | 38.87 | 2235.09 | 111.60 | 100.47 | 12.78 | 41.40 | 2040.00 | 75.95 |
| 7 | 2.008363 | 127.47 | 32.37 | 2134.32 | 111.60 | 100.60 | 12.50 | 43.05 | 2040.00 | 76.03 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/31.50/32.66.
Separate latency-pass duration: 2.00 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

64.99% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.163687 | 118.32 | 36.16 | 2394.31 | 111.60 | 100.71 | 12.23 | 41.29 | 2040.00 | 77.40 |
| 2 | 2.118281 | 120.85 | 37.04 | 2374.62 | 111.60 | 100.74 | 11.86 | 40.61 | 2040.00 | 77.40 |
| 3 | 2.171052 | 117.92 | 36.67 | 2386.76 | 111.60 | 100.68 | 11.70 | 44.49 | 2040.00 | 77.00 |
| 4 | 2.113631 | 121.12 | 37.36 | 2384.88 | 111.60 | 100.68 | 11.57 | 40.59 | 2040.00 | 76.95 |
| 5 | 2.177379 | 117.57 | 36.26 | 2394.56 | 111.60 | 100.71 | 11.36 | 42.94 | 2040.00 | 76.05 |
| 6 | 2.099939 | 121.91 | 37.10 | 2389.24 | 111.60 | 100.71 | 11.67 | 40.07 | 2040.00 | 75.90 |
| 7 | 2.205851 | 116.06 | 38.25 | 2387.50 | 111.60 | 100.73 | 11.09 | 42.16 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/34.11/35.04.
Separate latency-pass duration: 2.16 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo-world, images, batch 4

Status: **complete**. Load 5.78 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 195.79 ± 2.47 | 27.51 | 21.07/21.97/24.81 | 2624.57 | 214.50/304.00 | 256/1344 | PASS |
| raw-stream | 195.84 ± 2.47 | 27.33 | 20.70/21.81/25.33 | 2383.18 | 214.50/304.00 | 256/1344 | PASS |
| native-non-stream | 150.70 ± 2.41 | 32.94 | 27.05/28.58/31.74 | 2477.22 | 214.38/304.00 | 256/1344 | PASS |
| native-stream | 144.42 ± 1.09 | 38.24 | 27.90/29.56/33.19 | 2478.98 | 214.38/304.00 | 256/1344 | PASS |
| managed-non-stream | 127.69 ± 2.70 | 32.74 | 31.45/32.40/32.67 | 2590.60 | 214.38/304.00 | 256/1344 | PASS |
| managed-stream | 118.31 ± 2.60 | 37.86 | 33.28/34.64/36.21 | 2650.37 | 214.38/304.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.344240 | 190.44 | 38.84 | 2624.57 | 214.50 | 100.46 | 36.93 | 56.47 | 2018.33 | 80.11 |
| 2 | 1.305208 | 196.14 | 27.51 | 2255.66 | 214.50 | 100.47 | 36.77 | 60.70 | 1991.54 | 78.08 |
| 3 | 1.292636 | 198.04 | 27.39 | 2255.77 | 214.50 | 100.50 | 38.31 | 56.62 | 1992.69 | 77.77 |
| 4 | 1.317689 | 194.28 | 27.41 | 2263.86 | 214.50 | 100.49 | 37.56 | 59.15 | 2014.44 | 75.89 |
| 5 | 1.298872 | 197.09 | 27.34 | 2263.86 | 214.50 | 100.47 | 37.77 | 57.06 | 1998.46 | 75.92 |
| 6 | 1.307522 | 195.79 | 27.73 | 2383.18 | 214.50 | 100.50 | 38.08 | 60.35 | 2007.69 | 76.23 |
| 7 | 1.314100 | 194.81 | 27.92 | 2383.18 | 214.50 | 100.52 | 37.19 | 55.48 | 1984.44 | 76.37 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/21.65/21.95.
Separate latency-pass duration: 1.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.7468085195334595, 'inference': 2.6590340000041124, 'postprocess': 1.0829652890596009}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.52% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.333784 | 191.94 | 27.67 | 2374.82 | 214.50 | 100.50 | 40.19 | 62.38 | 1971.11 | 80.30 |
| 2 | 1.307271 | 195.83 | 21.29 | 2248.63 | 214.50 | 100.46 | 39.77 | 56.07 | 2021.54 | 78.00 |
| 3 | 1.296700 | 197.42 | 27.24 | 2255.77 | 214.50 | 100.47 | 37.62 | 59.54 | 2014.04 | 77.50 |
| 4 | 1.323745 | 193.39 | 31.21 | 2347.07 | 214.50 | 100.47 | 37.59 | 60.35 | 1998.89 | 75.89 |
| 5 | 1.288569 | 198.67 | 27.33 | 2263.86 | 214.50 | 100.66 | 40.88 | 60.12 | 2004.81 | 75.96 |
| 6 | 1.292583 | 198.05 | 21.41 | 2379.66 | 214.50 | 100.50 | 40.00 | 59.22 | 2015.77 | 76.00 |
| 7 | 1.307206 | 195.84 | 27.68 | 2383.18 | 214.50 | 100.50 | 37.27 | 57.09 | 2015.77 | 76.38 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/21.05/21.93.
Separate latency-pass duration: 1.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.7546814218848397, 'inference': 2.6399719453120696, 'postprocess': 1.0623775039082517}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

30.24% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.764983 | 145.04 | 33.36 | 2461.88 | 214.38 | 100.53 | 29.60 | 55.83 | 2025.43 | 80.11 |
| 2 | 1.699853 | 150.60 | 26.91 | 2326.62 | 214.38 | 100.63 | 30.15 | 58.70 | 2029.41 | 78.15 |
| 3 | 1.698830 | 150.69 | 32.75 | 2333.71 | 214.38 | 100.48 | 30.35 | 57.91 | 2029.41 | 77.35 |
| 4 | 1.698242 | 150.74 | 36.45 | 2347.07 | 214.38 | 100.49 | 31.53 | 57.28 | 2032.06 | 75.56 |
| 5 | 1.697173 | 150.84 | 33.13 | 2342.38 | 214.38 | 100.47 | 30.59 | 57.38 | 2027.65 | 75.65 |
| 6 | 1.698761 | 150.70 | 26.56 | 2470.78 | 214.38 | 100.50 | 29.00 | 53.11 | 2032.94 | 75.29 |
| 7 | 1.675094 | 152.83 | 32.94 | 2477.22 | 214.38 | 100.50 | 31.91 | 58.52 | 2032.94 | 76.41 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/27.48/28.80.
Separate latency-pass duration: 1.74 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### native-stream

35.20% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.777385 | 144.03 | 38.24 | 2468.15 | 214.38 | 100.49 | 30.89 | 55.23 | 2022.08 | 80.31 |
| 2 | 1.764604 | 145.08 | 38.41 | 2338.93 | 214.38 | 100.42 | 29.97 | 54.67 | 2029.29 | 78.26 |
| 3 | 1.789704 | 143.04 | 37.98 | 2338.70 | 214.38 | 100.48 | 30.11 | 53.63 | 2030.42 | 77.08 |
| 4 | 1.751630 | 146.15 | 38.35 | 2366.99 | 214.38 | 100.48 | 28.03 | 54.97 | 2030.14 | 75.51 |
| 5 | 1.780318 | 143.79 | 37.49 | 2347.36 | 214.38 | 100.46 | 30.17 | 55.51 | 2033.33 | 75.36 |
| 6 | 1.772568 | 144.42 | 38.29 | 2478.98 | 214.38 | 100.50 | 29.43 | 52.64 | 2037.43 | 75.31 |
| 7 | 1.757606 | 145.65 | 38.14 | 2477.22 | 214.38 | 100.48 | 31.11 | 53.52 | 2040.00 | 76.37 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/28.69/29.60.
Separate latency-pass duration: 1.80 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

53.46% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.062875 | 124.10 | 32.37 | 2456.13 | 214.38 | 100.47 | 26.95 | 54.33 | 2024.27 | 80.15 |
| 2 | 2.079244 | 123.12 | 39.36 | 2338.35 | 214.38 | 100.48 | 26.38 | 55.14 | 2038.57 | 78.38 |
| 3 | 2.004886 | 127.69 | 32.51 | 2354.75 | 214.38 | 100.53 | 27.20 | 52.33 | 2040.00 | 76.62 |
| 4 | 1.975319 | 129.60 | 36.07 | 2416.58 | 214.38 | 100.47 | 26.82 | 50.62 | 2040.00 | 75.60 |
| 5 | 1.992520 | 128.48 | 32.74 | 2367.58 | 214.38 | 100.48 | 26.35 | 53.23 | 2040.00 | 75.25 |
| 6 | 1.983633 | 129.06 | 38.13 | 2590.60 | 214.38 | 100.48 | 26.70 | 50.77 | 2038.50 | 75.00 |
| 7 | 2.061069 | 124.21 | 32.43 | 2476.34 | 214.38 | 100.48 | 24.88 | 51.80 | 2040.00 | 76.29 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/32.03/32.42.
Separate latency-pass duration: 2.01 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

61.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.163890 | 118.31 | 51.28 | 2461.51 | 214.38 | 100.72 | 25.53 | 48.94 | 2037.91 | 79.26 |
| 2 | 2.106530 | 121.53 | 37.86 | 2435.38 | 214.38 | 100.71 | 24.76 | 49.82 | 2036.07 | 79.48 |
| 3 | 2.174088 | 117.75 | 37.11 | 2613.16 | 214.38 | 100.74 | 23.86 | 55.45 | 2040.00 | 76.11 |
| 4 | 2.088556 | 122.57 | 32.89 | 2640.69 | 214.38 | 100.71 | 24.71 | 47.81 | 2040.00 | 75.76 |
| 5 | 2.193242 | 116.72 | 35.65 | 2650.37 | 214.38 | 100.70 | 25.34 | 51.47 | 2040.00 | 75.18 |
| 6 | 2.097725 | 122.04 | 38.73 | 2646.72 | 214.38 | 100.75 | 24.02 | 47.85 | 2040.00 | 74.43 |
| 7 | 2.196296 | 116.56 | 53.32 | 2481.33 | 214.38 | 100.85 | 23.45 | 48.46 | 2040.00 | 76.27 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/33.97/34.82.
Separate latency-pass duration: 2.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=256.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — grounding-dino, images, batch 4

Status: **complete**. Load 12.48 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 5.06 ± 0.01 | 789.94 | 790.06/792.78/793.10 | 3018.31 | 3486.48/4298.00 | 32/176 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 5.04 ± 0.00 | 792.48 | 795.15/796.86/797.35 | 3018.32 | 3486.46/4298.00 | 32/176 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 5.01 ± 0.01 | 796.68 | 798.12/799.76/800.10 | 3055.55 | 3486.46/4298.00 | 32/176 | PASS |
| managed-stream | 5.03 ± 0.01 | 793.75 | 795.79/799.04/799.45 | 3018.46 | 3486.46/4298.00 | 32/176 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.337919 | 5.05 | 792.49 | 3013.89 | 3486.48 | 100.48 | 73.11 | 64.82 | 1379.88 | 79.61 |
| 2 | 6.300410 | 5.08 | 785.22 | 3017.62 | 3486.48 | 100.45 | 74.83 | 63.84 | 1393.57 | 75.16 |
| 3 | 6.309332 | 5.07 | 789.70 | 3017.64 | 3486.48 | 100.43 | 76.29 | 64.09 | 1392.26 | 75.49 |
| 4 | 6.326024 | 5.06 | 790.52 | 3017.99 | 3486.48 | 100.48 | 72.63 | 63.75 | 1378.81 | 77.19 |
| 5 | 6.323604 | 5.06 | 791.36 | 3018.00 | 3486.48 | 100.43 | 79.24 | 64.18 | 1374.88 | 77.13 |
| 6 | 6.326899 | 5.06 | 789.94 | 3018.31 | 3486.48 | 100.45 | 81.04 | 62.83 | 1396.90 | 76.36 |
| 7 | 6.316056 | 5.07 | 788.36 | 3018.31 | 3486.48 | 100.46 | 79.27 | 63.88 | 1383.81 | 76.36 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/791.57/792.79.
Separate latency-pass duration: 6.32 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.53% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.354957 | 5.04 | 793.89 | 3016.41 | 3486.46 | 100.46 | 76.94 | 65.72 | 1371.02 | 78.39 |
| 2 | 6.345557 | 5.04 | 789.41 | 3017.59 | 3486.46 | 100.46 | 77.79 | 64.99 | 1401.31 | 74.90 |
| 3 | 6.349589 | 5.04 | 792.48 | 3017.66 | 3486.46 | 100.46 | 77.80 | 64.47 | 1389.80 | 75.71 |
| 4 | 6.353802 | 5.04 | 786.48 | 3017.95 | 3486.46 | 100.45 | 75.69 | 65.12 | 1401.14 | 77.07 |
| 5 | 6.358538 | 5.03 | 794.98 | 3018.00 | 3486.46 | 100.46 | 75.18 | 65.57 | 1387.20 | 76.76 |
| 6 | 6.349981 | 5.04 | 787.87 | 3018.28 | 3486.46 | 100.43 | 75.06 | 64.69 | 1380.35 | 76.17 |
| 7 | 6.349765 | 5.04 | 798.06 | 3018.32 | 3486.46 | 100.43 | 71.91 | 62.56 | 1398.31 | 76.61 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/795.51/796.90.
Separate latency-pass duration: 6.36 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

1.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.390595 | 5.01 | 796.68 | 3053.91 | 3486.46 | 100.44 | 81.28 | 61.50 | 1411.29 | 77.67 |
| 2 | 6.372793 | 5.02 | 794.89 | 3045.72 | 3486.46 | 100.47 | 70.79 | 61.52 | 1387.20 | 75.41 |
| 3 | 6.366644 | 5.03 | 795.79 | 3045.84 | 3486.46 | 100.42 | 70.82 | 62.15 | 1387.68 | 76.11 |
| 4 | 6.399649 | 5.00 | 802.42 | 3055.45 | 3486.46 | 100.47 | 79.59 | 61.22 | 1357.38 | 77.30 |
| 5 | 6.395061 | 5.00 | 797.68 | 3055.55 | 3486.46 | 100.45 | 75.45 | 61.28 | 1388.67 | 76.75 |
| 6 | 6.377751 | 5.02 | 797.43 | 3046.40 | 3486.46 | 100.46 | 71.95 | 64.56 | 1384.49 | 76.21 |
| 7 | 6.388776 | 5.01 | 796.00 | 3051.17 | 3486.46 | 100.48 | 78.46 | 65.68 | 1395.47 | 76.37 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/798.70/799.78.
Separate latency-pass duration: 6.38 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

0.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.364361 | 5.03 | 787.02 | 3016.42 | 3486.46 | 100.42 | 75.78 | 65.45 | 1370.43 | 76.64 |
| 2 | 6.376018 | 5.02 | 798.33 | 3016.43 | 3486.46 | 100.45 | 72.35 | 65.08 | 1394.41 | 75.76 |
| 3 | 6.356315 | 5.03 | 784.31 | 3017.83 | 3486.46 | 100.43 | 78.90 | 64.87 | 1407.99 | 76.40 |
| 4 | 6.374221 | 5.02 | 795.58 | 3017.95 | 3486.46 | 100.42 | 79.31 | 63.27 | 1382.36 | 77.02 |
| 5 | 6.374942 | 5.02 | 790.66 | 3018.16 | 3486.46 | 100.48 | 79.87 | 64.19 | 1396.65 | 76.45 |
| 6 | 6.367884 | 5.03 | 797.29 | 3018.27 | 3486.46 | 100.42 | 80.57 | 63.30 | 1407.05 | 76.37 |
| 7 | 6.367245 | 5.03 | 793.75 | 3018.46 | 3486.46 | 100.41 | 80.19 | 62.90 | 1403.50 | 76.57 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/797.79/799.04.
Separate latency-pass duration: 6.36 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — grounding-dino-base, images, batch 4

Status: **complete**. Load 12.64 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 3.71 ± 0.00 | 1080.62 | 1078.43/1085.19/1086.10 | 3024.92 | 3731.84/4524.00 | 32/168 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 3.70 ± 0.01 | 1081.16 | 1082.43/1087.81/1088.93 | 3024.74 | 3731.82/4524.00 | 32/168 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 3.70 ± 0.01 | 1084.51 | 1085.58/1087.67/1088.01 | 3062.24 | 3731.82/4524.00 | 32/168 | PASS |
| managed-stream | 3.70 ± 0.01 | 1073.58 | 1081.55/1088.29/1089.13 | 3024.98 | 3731.82/4524.00 | 32/168 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.655426 | 3.70 | 1076.94 | 3023.47 | 3731.84 | 100.46 | 83.02 | 66.21 | 1288.87 | 78.31 |
| 2 | 8.632174 | 3.71 | 1081.12 | 3024.73 | 3731.84 | 100.45 | 82.00 | 65.89 | 1293.40 | 76.47 |
| 3 | 8.625487 | 3.71 | 1082.07 | 3024.73 | 3731.84 | 100.44 | 82.21 | 66.39 | 1303.17 | 76.91 |
| 4 | 8.632416 | 3.71 | 1085.40 | 3024.75 | 3731.84 | 100.45 | 83.00 | 66.13 | 1290.70 | 76.12 |
| 5 | 8.623601 | 3.71 | 1077.02 | 3024.75 | 3731.84 | 100.42 | 82.58 | 65.51 | 1297.76 | 76.18 |
| 6 | 8.627555 | 3.71 | 1074.36 | 3024.91 | 3731.84 | 100.45 | 82.19 | 65.76 | 1292.35 | 76.83 |
| 7 | 8.630886 | 3.71 | 1080.62 | 3024.92 | 3731.84 | 100.44 | 82.07 | 66.12 | 1296.28 | 76.55 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1082.65/1085.30.
Separate latency-pass duration: 8.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.32% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.655260 | 3.70 | 1085.16 | 3022.04 | 3731.82 | 100.47 | 82.65 | 65.86 | 1294.80 | 76.77 |
| 2 | 8.634264 | 3.71 | 1071.98 | 3024.73 | 3731.82 | 100.50 | 81.79 | 65.79 | 1304.48 | 75.87 |
| 3 | 8.673531 | 3.69 | 1087.25 | 3024.73 | 3731.82 | 100.44 | 83.08 | 65.79 | 1295.72 | 77.46 |
| 4 | 8.642070 | 3.70 | 1076.00 | 3024.74 | 3731.82 | 100.48 | 82.30 | 65.70 | 1305.70 | 76.09 |
| 5 | 8.651444 | 3.70 | 1077.06 | 3024.30 | 3731.82 | 100.45 | 82.42 | 65.55 | 1291.22 | 76.19 |
| 6 | 8.661253 | 3.69 | 1081.16 | 3013.36 | 3731.82 | 100.44 | 81.95 | 66.38 | 1290.35 | 76.92 |
| 7 | 8.671326 | 3.69 | 1081.66 | 3011.41 | 3731.82 | 100.48 | 82.64 | 65.82 | 1303.61 | 76.18 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1084.67/1087.94.
Separate latency-pass duration: 8.67 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

0.32% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.650972 | 3.70 | 1085.60 | 3046.92 | 3731.82 | 100.43 | 82.08 | 65.14 | 1306.99 | 75.57 |
| 2 | 8.647491 | 3.70 | 1083.65 | 3037.15 | 3731.82 | 100.43 | 82.00 | 65.65 | 1315.99 | 75.28 |
| 3 | 8.693681 | 3.68 | 1087.95 | 3062.24 | 3731.82 | 100.46 | 82.36 | 66.13 | 1287.92 | 77.43 |
| 4 | 8.666649 | 3.69 | 1084.51 | 3057.55 | 3731.82 | 100.43 | 81.99 | 65.67 | 1305.09 | 76.41 |
| 5 | 8.650938 | 3.70 | 1084.23 | 3033.24 | 3731.82 | 100.43 | 82.25 | 66.06 | 1302.92 | 76.35 |
| 6 | 8.670199 | 3.69 | 1084.95 | 3057.68 | 3731.82 | 100.45 | 81.90 | 65.52 | 1296.07 | 76.68 |
| 7 | 8.656655 | 3.70 | 1080.23 | 3057.83 | 3731.82 | 100.43 | 83.19 | 65.71 | 1305.26 | 75.98 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1086.69/1087.69.
Separate latency-pass duration: 8.69 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

0.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.637790 | 3.70 | 1072.48 | 3023.50 | 3731.82 | 100.44 | 82.33 | 65.40 | 1304.13 | 74.91 |
| 2 | 8.628111 | 3.71 | 1073.55 | 3023.50 | 3731.82 | 100.43 | 81.51 | 65.76 | 1315.47 | 74.86 |
| 3 | 8.676318 | 3.69 | 1076.48 | 3024.75 | 3731.82 | 100.49 | 82.54 | 65.83 | 1299.54 | 77.26 |
| 4 | 8.669089 | 3.69 | 1080.68 | 3008.54 | 3731.82 | 100.49 | 82.37 | 65.58 | 1303.87 | 76.76 |
| 5 | 8.633744 | 3.71 | 1073.58 | 3024.77 | 3731.82 | 100.44 | 82.56 | 65.72 | 1298.20 | 76.50 |
| 6 | 8.657578 | 3.70 | 1085.58 | 3024.86 | 3731.82 | 100.49 | 82.16 | 65.88 | 1298.46 | 76.64 |
| 7 | 8.653000 | 3.70 | 1072.55 | 3024.98 | 3731.82 | 100.42 | 83.19 | 65.72 | 1300.84 | 76.17 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/1085.18/1088.30.
Separate latency-pass duration: 8.66 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — faster-rcnn, images, batch 4

Status: **complete**. Load 1.66 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 71.81 ± 0.21 | 55.72 | 55.57/56.44/56.57 | 2113.17 | 561.57/1006.00 | 128/1120 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 68.23 ± 0.35 | 59.19 | 58.76/60.23/61.00 | 2187.94 | 561.45/1006.00 | 128/1120 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 66.34 ± 0.67 | 59.63 | 60.28/69.40/69.77 | 2192.93 | 561.45/1006.00 | 128/1120 | PASS |
| managed-stream | 65.99 ± 0.73 | 61.21 | 60.24/62.58/63.21 | 2193.38 | 561.45/1006.00 | 128/1120 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.781098 | 71.87 | 56.20 | 2050.95 | 561.57 | 120.13 | 63.03 | 65.68 | 1648.75 | 76.92 |
| 2 | 1.785154 | 71.70 | 55.25 | 2113.17 | 561.57 | 119.39 | 67.42 | 67.30 | 1631.67 | 79.36 |
| 3 | 1.782411 | 71.81 | 55.66 | 2113.17 | 561.57 | 119.24 | 62.89 | 61.89 | 1620.83 | 79.44 |
| 4 | 1.783037 | 71.79 | 56.56 | 2078.85 | 561.57 | 119.42 | 65.11 | 65.60 | 1652.50 | 77.50 |
| 5 | 1.771348 | 72.26 | 55.15 | 2078.85 | 561.57 | 119.17 | 64.36 | 65.74 | 1665.00 | 77.33 |
| 6 | 1.774769 | 72.12 | 55.72 | 2090.80 | 561.57 | 119.28 | 68.22 | 67.92 | 1669.17 | 76.19 |
| 7 | 1.783292 | 71.78 | 55.75 | 2090.80 | 561.57 | 119.32 | 62.53 | 64.43 | 1674.58 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/55.89/56.47.
Separate latency-pass duration: 1.78 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

5.56% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.889559 | 67.74 | 59.19 | 2050.95 | 561.45 | 115.74 | 64.42 | 67.47 | 1650.39 | 77.39 |
| 2 | 1.884351 | 67.93 | 62.11 | 2182.96 | 561.45 | 115.36 | 63.34 | 65.00 | 1680.79 | 79.42 |
| 3 | 1.893444 | 67.60 | 58.63 | 2113.18 | 561.45 | 115.77 | 64.47 | 65.54 | 1661.45 | 79.13 |
| 4 | 1.873816 | 68.31 | 62.44 | 2167.68 | 561.45 | 115.45 | 60.58 | 64.18 | 1719.47 | 77.68 |
| 5 | 1.865732 | 68.61 | 58.46 | 2078.85 | 561.45 | 115.34 | 62.95 | 66.17 | 1698.16 | 77.18 |
| 6 | 1.875996 | 68.23 | 59.24 | 2187.94 | 561.45 | 115.48 | 64.37 | 65.69 | 1674.47 | 76.37 |
| 7 | 1.875110 | 68.26 | 58.65 | 2090.80 | 561.45 | 115.51 | 64.05 | 64.54 | 1699.74 | 75.63 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/59.46/60.52.
Separate latency-pass duration: 1.89 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=128.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

8.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.922060 | 66.60 | 59.63 | 2192.93 | 561.45 | 116.36 | 61.46 | 65.40 | 1844.62 | 78.13 |
| 2 | 1.972479 | 64.89 | 61.34 | 2182.51 | 561.45 | 118.54 | 57.55 | 64.40 | 1746.75 | 79.15 |
| 3 | 1.929524 | 66.34 | 59.72 | 2185.77 | 561.45 | 116.01 | 61.21 | 65.53 | 1732.31 | 78.59 |
| 4 | 1.917186 | 66.76 | 59.38 | 2167.68 | 561.45 | 116.20 | 62.10 | 63.44 | 1763.46 | 77.69 |
| 5 | 1.920933 | 66.63 | 59.21 | 2168.96 | 561.45 | 116.12 | 58.85 | 63.91 | 1729.62 | 77.13 |
| 6 | 1.942328 | 65.90 | 60.16 | 2187.94 | 561.45 | 116.32 | 58.95 | 66.07 | 1782.31 | 76.23 |
| 7 | 1.948593 | 65.69 | 59.43 | 2190.28 | 561.45 | 116.33 | 62.36 | 67.97 | 1717.69 | 75.72 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/67.35/69.40.
Separate latency-pass duration: 1.99 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=128.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

9.28% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.958013 | 65.37 | 66.45 | 2193.38 | 561.45 | 114.92 | 57.13 | 66.86 | 1783.46 | 78.28 |
| 2 | 1.995176 | 64.15 | 59.50 | 2053.53 | 561.45 | 116.78 | 56.58 | 64.45 | 1778.62 | 78.85 |
| 3 | 1.931459 | 66.27 | 59.55 | 2181.59 | 561.45 | 114.59 | 59.28 | 66.30 | 1693.46 | 78.18 |
| 4 | 1.936551 | 66.10 | 62.38 | 2156.11 | 561.45 | 114.76 | 59.31 | 63.48 | 1672.69 | 77.85 |
| 5 | 1.935814 | 66.12 | 64.21 | 2168.97 | 561.45 | 115.04 | 61.46 | 65.81 | 1856.92 | 76.82 |
| 6 | 1.939709 | 65.99 | 60.02 | 2102.54 | 561.45 | 115.14 | 59.13 | 64.29 | 1804.23 | 76.26 |
| 7 | 1.946617 | 65.76 | 61.21 | 2190.29 | 561.45 | 114.92 | 58.64 | 65.42 | 1756.92 | 75.56 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/60.61/62.07.
Separate latency-pass duration: 1.94 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=128.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — d-fine, images, batch 4

Status: **complete**. Load 19.45 s. Device `cuda:0`, dtype `float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 10.37 ± 0.05 | 384.93 | 384.26/386.26/386.32 | 3237.36 | 514.69/714.00 | 32/160 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 10.25 ± 0.11 | 393.09 | 393.04/401.78/402.91 | 3237.36 | 488.93/714.00 | 32/160 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 10.09 ± 0.12 | 392.79 | 390.96/399.69/400.98 | 3274.51 | 490.85/714.00 | 32/160 | PASS |
| managed-stream | 10.22 ± 0.06 | 395.17 | 389.81/396.89/398.25 | 3237.46 | 485.75/714.00 | 32/160 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.115026 | 10.27 | 389.85 | 3236.44 | 509.54 | 101.92 | 7.23 | 35.59 | 2040.00 | 71.00 |
| 2 | 3.103901 | 10.31 | 391.26 | 3236.91 | 512.12 | 101.94 | 7.39 | 36.60 | 2040.00 | 72.10 |
| 3 | 3.091519 | 10.35 | 382.15 | 3236.92 | 513.04 | 101.94 | 5.02 | 36.14 | 2040.00 | 72.61 |
| 4 | 3.065597 | 10.44 | 378.26 | 3237.20 | 514.69 | 101.91 | 5.74 | 36.94 | 2040.00 | 73.84 |
| 5 | 3.086043 | 10.37 | 384.93 | 3237.20 | 509.54 | 101.97 | 6.08 | 36.62 | 2040.00 | 73.95 |
| 6 | 3.082698 | 10.38 | 385.62 | 3237.36 | 510.32 | 101.94 | 7.03 | 37.24 | 2040.00 | 74.95 |
| 7 | 3.078264 | 10.40 | 383.29 | 3237.36 | 509.54 | 101.97 | 5.56 | 36.99 | 2040.00 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/385.25/386.24.
Separate latency-pass duration: 3.07 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

1.10% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.149162 | 10.16 | 391.25 | 3236.45 | 484.19 | 101.92 | 4.85 | 35.25 | 2040.00 | 71.10 |
| 2 | 3.162976 | 10.12 | 408.22 | 3236.86 | 485.44 | 102.02 | 4.32 | 36.79 | 2040.00 | 72.00 |
| 3 | 3.185286 | 10.05 | 404.10 | 3236.95 | 479.44 | 101.89 | 7.75 | 35.63 | 2040.00 | 72.92 |
| 4 | 3.095359 | 10.34 | 393.09 | 3237.20 | 486.61 | 101.99 | 5.36 | 36.37 | 2040.00 | 73.41 |
| 5 | 3.105996 | 10.30 | 386.79 | 3237.20 | 486.61 | 101.92 | 7.40 | 36.65 | 2040.00 | 74.00 |
| 6 | 3.106963 | 10.30 | 393.95 | 3237.36 | 488.36 | 102.01 | 5.94 | 36.75 | 2040.00 | 74.89 |
| 7 | 3.121020 | 10.25 | 384.58 | 3237.36 | 488.93 | 101.95 | 4.52 | 37.54 | 2040.00 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/397.00/401.91.
Separate latency-pass duration: 3.15 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-non-stream

2.55% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.219165 | 9.94 | 395.26 | 3274.00 | 490.85 | 102.22 | 7.61 | 35.84 | 2040.00 | 71.30 |
| 2 | 3.194045 | 10.02 | 392.79 | 3274.29 | 490.85 | 102.23 | 7.14 | 34.95 | 2040.00 | 72.00 |
| 3 | 3.170277 | 10.09 | 395.45 | 3274.51 | 490.85 | 102.28 | 3.81 | 36.84 | 2040.00 | 72.97 |
| 4 | 3.124429 | 10.24 | 388.08 | 3273.00 | 490.85 | 102.28 | 7.66 | 37.15 | 2040.00 | 73.06 |
| 5 | 3.126039 | 10.24 | 391.33 | 3273.02 | 490.85 | 102.23 | 5.31 | 36.44 | 2040.00 | 74.00 |
| 6 | 3.127947 | 10.23 | 390.44 | 3273.16 | 490.85 | 102.25 | 4.31 | 36.27 | 2040.00 | 74.50 |
| 7 | 3.171431 | 10.09 | 403.74 | 3273.18 | 490.85 | 102.20 | 7.19 | 37.74 | 2040.00 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/395.15/399.85.
Separate latency-pass duration: 3.14 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

1.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.151318 | 10.15 | 398.84 | 3236.60 | 485.75 | 102.00 | 5.87 | 35.99 | 2040.00 | 71.92 |
| 2 | 3.139968 | 10.19 | 393.34 | 3236.71 | 478.33 | 101.89 | 5.15 | 35.57 | 2040.00 | 71.95 |
| 3 | 3.151234 | 10.15 | 401.67 | 3237.10 | 482.00 | 101.97 | 6.66 | 36.60 | 2040.00 | 73.00 |
| 4 | 3.123670 | 10.24 | 386.21 | 3237.18 | 484.83 | 101.94 | 5.37 | 36.87 | 2040.00 | 73.00 |
| 5 | 3.131365 | 10.22 | 395.17 | 3237.27 | 480.33 | 101.99 | 5.10 | 37.25 | 2040.00 | 74.00 |
| 6 | 3.095707 | 10.34 | 383.78 | 3237.34 | 481.78 | 101.89 | 6.75 | 36.93 | 2040.00 | 74.07 |
| 7 | 3.127679 | 10.23 | 397.60 | 3237.46 | 481.10 | 102.01 | 4.68 | 37.58 | 2040.00 | 75.03 |

Ready result delivery-interval P50/P95/P99 ms: 0.03/391.73/396.74.
Separate latency-pass duration: 3.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

First-repeat metadata: per-frame UID uniqueness=True; distinct source identities=32.
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo, video, batch 1

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 107.72 ± 2.01 | 1113.95 | 1129.53/1129.53/1129.53 | 2167.04 | 34.30/62.00 | 120/665 | PASS |
| raw-stream | 107.71 ± 1.71 | 46.23 | 1106.22/1106.22/1106.22 | 2167.04 | 34.30/62.00 | 120/665 | PASS |
| native-non-stream | 100.70 ± 1.75 | 1191.57 | 1178.02/1178.02/1178.02 | 2167.04 | 34.30/62.00 | 120/665 | PASS |
| native-stream | 98.94 ± 1.50 | 1212.67 | 1224.37/1224.37/1224.37 | 2167.04 | 34.25/62.00 | 120/665 | PASS |
| managed-non-stream | 95.65 ± 1.68 | 1254.42 | 1233.91/1233.91/1233.91 | 1861.08 | 24.66/62.00 | 120/665 | PASS |
| managed-stream | 94.55 ± 2.84 | 84.11 | 1257.00/1257.00/1257.00 | 1861.08 | 24.66/62.00 | 120/665 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.161827 | 103.29 | 1161.74 | 2156.66 | 34.30 | 154.05 | 14.33 | 40.94 | 2040.00 | 77.00 |
| 2 | 1.144755 | 104.83 | 1144.66 | 2165.68 | 34.30 | 144.67 | 14.13 | 41.84 | 2040.00 | 77.83 |
| 3 | 1.114029 | 107.72 | 1113.95 | 2165.74 | 34.30 | 146.06 | 13.35 | 40.77 | 2040.00 | 77.78 |
| 4 | 1.109163 | 108.19 | 1109.07 | 2166.88 | 34.30 | 146.24 | 14.04 | 41.52 | 2040.00 | 77.00 |
| 5 | 1.109950 | 108.11 | 1109.87 | 2166.88 | 34.30 | 146.70 | 13.35 | 40.97 | 2040.00 | 77.00 |
| 6 | 1.105042 | 108.59 | 1104.94 | 2167.04 | 34.30 | 145.20 | 13.91 | 41.42 | 2040.00 | 76.09 |
| 7 | 1.118154 | 107.32 | 1118.07 | 2167.04 | 34.30 | 144.57 | 13.61 | 40.59 | 2040.00 | 76.00 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2556347500018699, 'inference': 6.738541933323934, 'postprocess': 1.2675191250082207}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.20% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.114124 | 107.71 | 45.73 | 2156.99 | 34.30 | 146.30 | 14.04 | 41.70 | 2040.00 | 77.30 |
| 2 | 1.139587 | 105.30 | 46.00 | 2163.05 | 34.30 | 145.02 | 13.70 | 41.22 | 2040.00 | 77.83 |
| 3 | 1.143568 | 104.93 | 46.53 | 2165.74 | 34.30 | 144.64 | 13.70 | 41.77 | 2040.00 | 77.74 |
| 4 | 1.143505 | 104.94 | 46.71 | 2166.88 | 34.30 | 145.37 | 14.09 | 40.88 | 2040.00 | 77.17 |
| 5 | 1.107751 | 108.33 | 46.00 | 2166.88 | 34.30 | 144.84 | 13.91 | 41.89 | 2040.00 | 77.00 |
| 6 | 1.104198 | 108.68 | 47.19 | 2164.40 | 34.30 | 145.03 | 13.36 | 40.66 | 2040.00 | 76.41 |
| 7 | 1.110356 | 108.07 | 46.23 | 2167.04 | 34.30 | 144.61 | 14.65 | 41.19 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 8.76/10.20/11.66.
Separate latency-pass duration: 1.11 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.199571666643351, 'inference': 6.4565911082998655, 'postprocess': 1.2178382833326395}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

5.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.199890 | 100.01 | 1199.78 | 2157.15 | 34.30 | 143.33 | 15.08 | 41.78 | 2040.00 | 77.58 |
| 2 | 1.204399 | 99.63 | 1204.29 | 2162.42 | 34.30 | 151.91 | 14.58 | 41.47 | 2040.00 | 77.83 |
| 3 | 1.233159 | 97.31 | 1233.04 | 2165.74 | 34.30 | 141.35 | 13.88 | 41.79 | 2040.00 | 77.76 |
| 4 | 1.186548 | 101.13 | 1186.44 | 2166.55 | 34.30 | 152.73 | 14.58 | 41.62 | 2040.00 | 77.67 |
| 5 | 1.169403 | 102.62 | 1169.31 | 2166.88 | 34.30 | 143.20 | 14.79 | 41.76 | 2040.00 | 76.92 |
| 6 | 1.191686 | 100.70 | 1191.57 | 2164.07 | 34.30 | 152.92 | 14.17 | 41.00 | 2040.00 | 76.67 |
| 7 | 1.176078 | 102.03 | 1175.96 | 2167.04 | 34.30 | 143.25 | 14.50 | 41.30 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
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


### native-stream

8.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.226246 | 97.86 | 1226.17 | 2157.62 | 34.25 | 141.61 | 13.68 | 41.01 | 2040.00 | 77.72 |
| 2 | 1.231198 | 97.47 | 1231.13 | 1858.80 | 34.25 | 141.02 | 13.52 | 41.21 | 2040.00 | 77.84 |
| 3 | 1.252544 | 95.81 | 1252.46 | 2165.78 | 34.25 | 140.82 | 12.24 | 41.06 | 2040.00 | 77.84 |
| 4 | 1.204472 | 99.63 | 1204.40 | 1860.57 | 34.25 | 141.79 | 13.17 | 41.10 | 2040.00 | 77.71 |
| 5 | 1.198845 | 100.10 | 1198.76 | 2166.88 | 34.25 | 139.94 | 13.33 | 41.32 | 2040.00 | 76.83 |
| 6 | 1.207071 | 99.41 | 1207.00 | 1866.94 | 34.25 | 143.19 | 13.62 | 40.41 | 2040.00 | 76.62 |
| 7 | 1.212829 | 98.94 | 1212.67 | 2167.04 | 34.25 | 141.18 | 13.28 | 40.56 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.22 s; not used for throughput.

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

11.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.268448 | 94.60 | 1268.36 | 1851.75 | 24.66 | 139.68 | 12.77 | 40.52 | 2040.00 | 77.77 |
| 2 | 1.254518 | 95.65 | 1254.42 | 1858.91 | 24.66 | 138.84 | 13.24 | 40.71 | 2040.00 | 77.84 |
| 3 | 1.259360 | 95.29 | 1259.28 | 1857.59 | 24.66 | 140.57 | 12.68 | 40.44 | 2040.00 | 77.76 |
| 4 | 1.212707 | 98.95 | 1212.57 | 1860.95 | 24.66 | 140.92 | 13.28 | 41.09 | 2040.00 | 77.76 |
| 5 | 1.234398 | 97.21 | 1234.31 | 1858.39 | 24.66 | 140.45 | 12.96 | 40.96 | 2040.00 | 76.84 |
| 6 | 1.229543 | 97.60 | 1229.44 | 1861.08 | 24.66 | 140.78 | 13.40 | 40.47 | 2040.00 | 76.76 |
| 7 | 1.269438 | 94.53 | 1269.32 | 1858.49 | 24.66 | 139.97 | 12.54 | 40.25 | 2040.00 | 75.92 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.23 s; not used for throughput.

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

13.37% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.307140 | 91.80 | 84.49 | 1855.43 | 24.66 | 139.52 | 12.00 | 40.77 | 2040.00 | 77.81 |
| 2 | 1.334677 | 89.91 | 83.54 | 1858.50 | 24.66 | 139.15 | 12.22 | 39.53 | 2040.00 | 77.78 |
| 3 | 1.256691 | 95.49 | 85.30 | 1857.86 | 24.66 | 139.28 | 13.16 | 40.94 | 2040.00 | 77.68 |
| 4 | 1.214325 | 98.82 | 82.21 | 1860.79 | 24.66 | 140.81 | 12.96 | 40.47 | 2040.00 | 77.64 |
| 5 | 1.258386 | 95.36 | 85.85 | 1858.30 | 24.66 | 138.93 | 12.96 | 40.48 | 2040.00 | 76.76 |
| 6 | 1.272003 | 94.34 | 83.45 | 1861.08 | 24.66 | 140.42 | 12.31 | 40.01 | 2040.00 | 76.73 |
| 7 | 1.269114 | 94.55 | 84.11 | 1858.46 | 24.66 | 141.53 | 12.08 | 40.37 | 2040.00 | 75.85 |

Ready result delivery-interval P50/P95/P99 ms: 9.69/11.23/12.55.
Separate latency-pass duration: 1.26 s; not used for throughput.

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

## Detail — yolo, video, batch 4

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 226.75 ± 8.67 | 529.12 | 539.72/539.72/539.72 | 2531.07 | 93.11/150.00 | 120/665 | PASS |
| raw-stream | 222.78 ± 2.18 | 56.71 | 540.91/540.91/540.91 | 2472.11 | 93.11/150.00 | 120/665 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 199.09 ± 4.63 | 602.65 | 605.05/605.05/605.05 | 2470.70 | 64.78/150.00 | 120/665 | PASS |
| managed-stream | 192.75 ± 2.95 | 97.16 | 626.14/626.14/626.14 | 2501.57 | 64.78/150.00 | 120/665 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.583206 | 205.76 | 583.09 | 2531.07 | 93.11 | 208.40 | 14.00 | 41.71 | 2040.00 | 75.67 |
| 2 | 0.534303 | 224.59 | 534.21 | 2263.91 | 93.11 | 198.52 | 13.82 | 42.12 | 2040.00 | 76.27 |
| 3 | 0.541713 | 221.52 | 541.62 | 2225.82 | 93.11 | 196.86 | 10.00 | 41.97 | 2040.00 | 76.55 |
| 4 | 0.523323 | 229.30 | 523.24 | 2265.02 | 93.11 | 201.84 | 13.64 | 43.95 | 2040.00 | 76.82 |
| 5 | 0.518654 | 231.37 | 518.55 | 2224.29 | 93.11 | 204.63 | 9.00 | 43.52 | 2040.00 | 76.91 |
| 6 | 0.529207 | 226.75 | 529.12 | 2225.12 | 93.11 | 204.61 | 14.00 | 42.57 | 2040.00 | 77.64 |
| 7 | 0.524754 | 228.68 | 524.67 | 2225.12 | 93.11 | 203.55 | 12.82 | 43.14 | 2040.00 | 77.36 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.54 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2260668916572588, 'inference': 2.179125400008767, 'postprocess': 1.0652504666647171}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

1.71% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.540393 | 222.06 | 70.32 | 2472.11 | 93.11 | 213.77 | 12.45 | 43.01 | 2040.00 | 75.64 |
| 2 | 0.543446 | 220.81 | 59.60 | 2253.35 | 93.11 | 203.21 | 13.73 | 42.45 | 2040.00 | 76.27 |
| 3 | 0.538639 | 222.78 | 56.22 | 2223.30 | 93.11 | 199.01 | 13.73 | 41.59 | 2040.00 | 76.55 |
| 4 | 0.529082 | 226.81 | 55.91 | 2264.90 | 93.11 | 201.64 | 13.36 | 43.80 | 2040.00 | 76.82 |
| 5 | 0.532240 | 225.46 | 56.66 | 2224.29 | 93.11 | 197.28 | 13.64 | 43.99 | 2040.00 | 77.18 |
| 6 | 0.541774 | 221.49 | 58.91 | 2254.64 | 93.11 | 197.22 | 14.73 | 43.10 | 2040.00 | 77.64 |
| 7 | 0.535923 | 223.91 | 56.71 | 2225.12 | 93.11 | 198.27 | 15.09 | 43.39 | 2040.00 | 77.73 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/17.00/19.78.
Separate latency-pass duration: 0.54 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1009046499907527, 'inference': 1.879546700016969, 'postprocess': 1.1534164333321921}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

12.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.625225 | 191.93 | 625.12 | 2470.70 | 64.78 | 187.37 | 13.69 | 43.00 | 2040.00 | 75.69 |
| 2 | 0.600493 | 199.84 | 600.40 | 2252.69 | 64.78 | 189.45 | 12.58 | 42.29 | 2040.00 | 75.92 |
| 3 | 0.602745 | 199.09 | 602.65 | 2223.29 | 64.78 | 190.51 | 13.42 | 43.05 | 2040.00 | 76.67 |
| 4 | 0.590316 | 203.28 | 590.21 | 2254.48 | 64.78 | 188.59 | 13.00 | 42.62 | 2040.00 | 76.67 |
| 5 | 0.617809 | 194.23 | 617.72 | 2224.18 | 64.78 | 194.43 | 13.08 | 43.88 | 2040.00 | 77.00 |
| 6 | 0.613474 | 195.61 | 613.38 | 2254.60 | 64.78 | 189.38 | 12.92 | 43.42 | 2040.00 | 77.15 |
| 7 | 0.587312 | 204.32 | 587.19 | 2225.02 | 64.78 | 188.29 | 13.25 | 43.13 | 2040.00 | 77.58 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.61 s; not used for throughput.

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

18.64% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.614416 | 195.31 | 96.84 | 2501.57 | 64.78 | 191.77 | 12.69 | 41.65 | 2040.00 | 75.69 |
| 2 | 0.635045 | 188.96 | 102.30 | 2501.56 | 64.78 | 191.96 | 10.69 | 41.81 | 2040.00 | 75.62 |
| 3 | 0.618981 | 193.87 | 97.16 | 2251.46 | 64.78 | 182.88 | 11.54 | 42.13 | 2040.00 | 76.54 |
| 4 | 0.633391 | 189.46 | 94.81 | 2254.45 | 64.78 | 187.00 | 10.38 | 42.02 | 2040.00 | 76.62 |
| 5 | 0.630342 | 190.37 | 99.04 | 2251.83 | 64.78 | 187.67 | 11.46 | 43.41 | 2040.00 | 77.15 |
| 6 | 0.610614 | 196.52 | 94.00 | 2254.58 | 64.78 | 189.49 | 13.00 | 42.29 | 2040.00 | 77.23 |
| 7 | 0.622576 | 192.75 | 99.46 | 2252.00 | 64.78 | 190.38 | 11.85 | 42.61 | 2040.00 | 77.69 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/18.52/21.03.
Separate latency-pass duration: 0.63 s; not used for throughput.

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
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 87.28 ± 0.66 | 1374.81 | 1457.66/1457.66/1457.66 | 2555.78 | 64.67/90.00 | 120/659 | PASS |
| raw-stream | 86.52 ± 1.51 | 48.82 | 1385.19/1385.19/1385.19 | 2555.79 | 64.67/90.00 | 120/659 | PASS |
| native-non-stream | 82.84 ± 0.96 | 1448.51 | 1454.19/1454.19/1454.19 | 2555.43 | 64.67/90.00 | 120/659 | PASS |
| native-stream | 82.84 ± 1.40 | 1448.54 | 1469.20/1469.20/1469.20 | 2550.48 | 64.67/90.00 | 120/659 | PASS |
| managed-non-stream | 80.43 ± 1.64 | 1491.88 | 1504.68/1504.68/1504.68 | 2251.92 | 56.12/90.00 | 120/659 | PASS |
| managed-stream | 79.00 ± 0.80 | 85.06 | 1513.27/1513.27/1513.27 | 2251.51 | 56.12/90.00 | 120/659 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.381398 | 86.87 | 1381.31 | 2548.14 | 64.67 | 146.46 | 13.71 | 40.39 | 2040.00 | 77.07 |
| 2 | 1.394455 | 86.06 | 1394.37 | 2555.70 | 64.67 | 136.05 | 12.93 | 40.41 | 2040.00 | 76.00 |
| 3 | 1.371454 | 87.50 | 1371.37 | 2555.78 | 64.67 | 137.48 | 13.93 | 39.79 | 2040.00 | 76.00 |
| 4 | 1.368570 | 87.68 | 1368.49 | 2244.00 | 64.67 | 137.24 | 13.57 | 40.45 | 2040.00 | 75.93 |
| 5 | 1.374896 | 87.28 | 1374.81 | 2244.00 | 64.67 | 136.95 | 13.39 | 39.81 | 2040.00 | 76.00 |
| 6 | 1.384674 | 86.66 | 1384.59 | 2244.57 | 64.67 | 137.69 | 13.32 | 40.48 | 2040.00 | 76.86 |
| 7 | 1.364150 | 87.97 | 1364.06 | 2244.57 | 64.67 | 136.88 | 13.00 | 39.97 | 2040.00 | 76.79 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1948221916516861, 'inference': 8.645438883331735, 'postprocess': 1.2396350916712133}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.40% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.386956 | 86.52 | 48.96 | 2548.29 | 64.67 | 137.63 | 13.18 | 40.79 | 2040.00 | 77.54 |
| 2 | 1.405439 | 85.38 | 49.34 | 2555.38 | 64.67 | 135.81 | 13.14 | 40.44 | 2040.00 | 76.89 |
| 3 | 1.361734 | 88.12 | 49.48 | 2555.79 | 64.67 | 138.44 | 13.57 | 40.47 | 2040.00 | 76.00 |
| 4 | 1.364175 | 87.97 | 48.03 | 2242.97 | 64.67 | 137.11 | 12.43 | 40.04 | 2040.00 | 75.86 |
| 5 | 1.408766 | 85.18 | 48.04 | 2244.00 | 64.67 | 136.94 | 13.14 | 40.44 | 2040.00 | 76.00 |
| 6 | 1.414091 | 84.86 | 48.82 | 2244.57 | 64.67 | 136.32 | 13.10 | 40.45 | 2040.00 | 76.83 |
| 7 | 1.358874 | 88.31 | 48.54 | 2244.57 | 64.67 | 136.64 | 13.48 | 40.53 | 2040.00 | 76.78 |

Ready result delivery-interval P50/P95/P99 ms: 11.08/12.09/13.62.
Separate latency-pass duration: 1.39 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.23528878336856, 'inference': 8.639215816659393, 'postprocess': 1.2538945250071265}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

5.41% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.448623 | 82.84 | 1448.51 | 2549.98 | 64.67 | 136.32 | 13.66 | 40.57 | 2040.00 | 77.72 |
| 2 | 1.480577 | 81.05 | 1480.48 | 2554.78 | 64.67 | 144.02 | 12.87 | 40.53 | 2040.00 | 76.87 |
| 3 | 1.446033 | 82.99 | 1445.92 | 2555.43 | 64.67 | 134.77 | 13.86 | 40.38 | 2040.00 | 75.90 |
| 4 | 1.427703 | 84.05 | 1427.61 | 2243.42 | 64.67 | 135.54 | 13.83 | 40.24 | 2040.00 | 75.90 |
| 5 | 1.460078 | 82.19 | 1459.96 | 2244.00 | 64.67 | 135.39 | 14.21 | 40.63 | 2040.00 | 76.66 |
| 6 | 1.453709 | 82.55 | 1453.62 | 2244.22 | 64.67 | 135.67 | 13.34 | 40.66 | 2040.00 | 76.90 |
| 7 | 1.437984 | 83.45 | 1437.90 | 2244.57 | 64.67 | 134.89 | 13.79 | 40.72 | 2040.00 | 76.83 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.45 s; not used for throughput.

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

5.48% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.443197 | 83.15 | 1443.12 | 2550.48 | 64.67 | 134.57 | 12.52 | 40.20 | 2040.00 | 77.59 |
| 2 | 1.428028 | 84.03 | 1427.96 | 2251.24 | 64.67 | 134.47 | 12.93 | 40.44 | 2040.00 | 76.93 |
| 3 | 1.483244 | 80.90 | 1483.17 | 2243.18 | 64.67 | 132.93 | 11.97 | 40.09 | 2040.00 | 75.87 |
| 4 | 1.499310 | 80.04 | 1499.23 | 2240.77 | 64.67 | 134.50 | 12.27 | 39.87 | 2040.00 | 75.87 |
| 5 | 1.450255 | 82.74 | 1450.18 | 2244.00 | 64.67 | 134.97 | 12.45 | 39.92 | 2040.00 | 76.79 |
| 6 | 1.448608 | 82.84 | 1448.54 | 2248.31 | 64.67 | 135.02 | 12.79 | 40.44 | 2040.00 | 76.90 |
| 7 | 1.446126 | 82.98 | 1446.06 | 2244.57 | 64.67 | 134.63 | 11.83 | 40.39 | 2040.00 | 76.79 |

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


### managed-non-stream

8.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.480064 | 81.08 | 1479.96 | 2242.52 | 56.12 | 132.72 | 12.77 | 40.31 | 2040.00 | 77.07 |
| 2 | 1.460718 | 82.15 | 1460.62 | 2251.92 | 56.12 | 134.22 | 12.79 | 40.13 | 2040.00 | 77.00 |
| 3 | 1.493904 | 80.33 | 1493.82 | 2232.83 | 56.12 | 133.48 | 12.30 | 39.52 | 2040.00 | 75.83 |
| 4 | 1.491971 | 80.43 | 1491.88 | 2248.56 | 56.12 | 133.99 | 12.57 | 39.65 | 2040.00 | 75.83 |
| 5 | 1.495296 | 80.25 | 1495.20 | 2233.45 | 56.12 | 135.15 | 12.20 | 40.02 | 2040.00 | 76.80 |
| 6 | 1.489561 | 80.56 | 1489.47 | 2248.68 | 56.12 | 133.77 | 12.53 | 40.06 | 2040.00 | 76.87 |
| 7 | 1.561673 | 76.84 | 1561.58 | 2237.79 | 56.12 | 134.06 | 12.26 | 40.09 | 2040.00 | 76.74 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.50 s; not used for throughput.

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

9.20% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.498456 | 80.08 | 85.57 | 2248.49 | 56.12 | 134.00 | 12.60 | 40.13 | 2040.00 | 77.00 |
| 2 | 1.522768 | 78.80 | 84.20 | 2251.51 | 56.12 | 134.26 | 10.97 | 39.34 | 2040.00 | 77.00 |
| 3 | 1.518937 | 79.00 | 85.68 | 2245.55 | 56.12 | 134.00 | 12.42 | 39.56 | 2040.00 | 75.87 |
| 4 | 1.522420 | 78.82 | 83.92 | 2248.49 | 56.12 | 133.31 | 11.42 | 39.29 | 2040.00 | 75.81 |
| 5 | 1.490340 | 80.52 | 85.06 | 2245.85 | 56.12 | 133.92 | 12.73 | 39.80 | 2040.00 | 76.80 |
| 6 | 1.492494 | 80.40 | 84.33 | 2248.63 | 56.12 | 132.66 | 11.80 | 39.54 | 2040.00 | 76.87 |
| 7 | 1.521711 | 78.86 | 86.20 | 2249.79 | 56.12 | 133.75 | 12.32 | 39.66 | 2040.00 | 76.74 |

Ready result delivery-interval P50/P95/P99 ms: 11.86/12.86/14.62.
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

## Detail — yolo11, video, batch 4

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 198.00 ± 8.08 | 605.97 | 590.32/590.32/590.32 | 2794.89 | 126.08/196.00 | 120/659 | PASS |
| raw-stream | 196.86 ± 4.23 | 63.77 | 585.38/585.38/585.38 | 2536.41 | 126.08/196.00 | 120/659 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 179.61 ± 4.12 | 668.00 | 660.83/660.83/660.83 | 2527.36 | 99.43/196.00 | 120/659 | PASS |
| managed-stream | 173.85 ± 2.96 | 101.34 | 678.86/678.86/678.86 | 2527.08 | 99.43/196.00 | 120/659 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.613604 | 195.57 | 613.52 | 2794.89 | 126.08 | 201.24 | 12.92 | 42.07 | 2040.00 | 77.00 |
| 2 | 0.606054 | 198.00 | 605.97 | 2500.73 | 126.08 | 192.52 | 13.54 | 43.45 | 2040.00 | 76.85 |
| 3 | 0.669672 | 179.19 | 669.58 | 2500.77 | 126.08 | 197.47 | 10.36 | 42.56 | 2040.00 | 76.86 |
| 4 | 0.595083 | 201.65 | 595.00 | 2498.87 | 126.08 | 188.15 | 13.67 | 43.50 | 2040.00 | 77.33 |
| 5 | 0.598285 | 200.57 | 598.20 | 2501.56 | 126.08 | 193.02 | 9.33 | 42.94 | 2040.00 | 77.17 |
| 6 | 0.590450 | 203.23 | 590.36 | 2536.37 | 126.08 | 189.62 | 13.00 | 42.97 | 2040.00 | 77.00 |
| 7 | 0.612740 | 195.84 | 612.63 | 2500.39 | 126.08 | 190.87 | 12.92 | 43.32 | 2040.00 | 76.85 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.59 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1269080333249804, 'inference': 2.4393695750025777, 'postprocess': 1.0596798416562099}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

1.06% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.620094 | 193.52 | 73.09 | 2492.06 | 126.08 | 200.25 | 12.69 | 42.85 | 2040.00 | 77.15 |
| 2 | 0.614410 | 195.31 | 64.24 | 2525.18 | 126.08 | 185.08 | 12.69 | 43.53 | 2040.00 | 77.00 |
| 3 | 0.591672 | 202.82 | 58.26 | 2498.15 | 126.08 | 192.98 | 13.25 | 42.39 | 2040.00 | 77.00 |
| 4 | 0.624739 | 192.08 | 64.78 | 2498.54 | 126.08 | 201.03 | 13.08 | 44.35 | 2040.00 | 77.08 |
| 5 | 0.591513 | 202.87 | 62.18 | 2499.70 | 126.08 | 193.41 | 13.08 | 43.61 | 2040.00 | 77.25 |
| 6 | 0.609584 | 196.86 | 63.77 | 2536.41 | 126.08 | 205.65 | 12.77 | 43.82 | 2040.00 | 77.00 |
| 7 | 0.607061 | 197.67 | 58.47 | 2500.78 | 126.08 | 186.34 | 13.15 | 43.10 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/18.48/19.84.
Separate latency-pass duration: 0.59 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.112109291678583, 'inference': 2.5031225416758693, 'postprocess': 1.1766764416734077}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

10.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.662397 | 181.16 | 662.30 | 2492.70 | 99.43 | 179.66 | 12.93 | 42.16 | 2040.00 | 76.86 |
| 2 | 0.668097 | 179.61 | 668.00 | 2527.36 | 99.43 | 186.26 | 12.86 | 41.97 | 2040.00 | 77.00 |
| 3 | 0.699262 | 171.61 | 699.16 | 2500.68 | 99.43 | 182.53 | 12.57 | 43.19 | 2040.00 | 76.86 |
| 4 | 0.657398 | 182.54 | 657.30 | 2218.73 | 99.43 | 183.44 | 12.86 | 44.09 | 2040.00 | 76.86 |
| 5 | 0.693057 | 173.15 | 692.95 | 2501.45 | 99.43 | 181.33 | 12.29 | 43.43 | 2040.00 | 77.14 |
| 6 | 0.667646 | 179.74 | 667.55 | 2218.78 | 99.43 | 179.86 | 12.86 | 42.15 | 2040.00 | 77.00 |
| 7 | 0.678626 | 176.83 | 678.53 | 2500.98 | 99.43 | 185.73 | 12.64 | 43.59 | 2040.00 | 76.86 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.66 s; not used for throughput.

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

14.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.701728 | 171.01 | 101.34 | 2523.91 | 99.43 | 183.17 | 12.00 | 41.10 | 2040.00 | 76.86 |
| 2 | 0.688172 | 174.37 | 97.67 | 2527.08 | 99.43 | 177.89 | 10.00 | 42.41 | 2040.00 | 76.71 |
| 3 | 0.698700 | 171.75 | 101.40 | 2215.77 | 99.43 | 180.46 | 12.86 | 42.36 | 2040.00 | 76.86 |
| 4 | 0.672164 | 178.53 | 96.07 | 2218.71 | 99.43 | 178.74 | 9.71 | 40.89 | 2040.00 | 76.71 |
| 5 | 0.687535 | 174.54 | 103.79 | 2216.02 | 99.43 | 183.94 | 10.71 | 42.77 | 2040.00 | 76.86 |
| 6 | 0.690258 | 173.85 | 96.24 | 2218.75 | 99.43 | 180.86 | 12.36 | 41.08 | 2040.00 | 76.86 |
| 7 | 0.708078 | 169.47 | 101.89 | 2215.69 | 99.43 | 181.50 | 12.71 | 42.77 | 2040.00 | 76.71 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/20.22/20.65.
Separate latency-pass duration: 0.68 s; not used for throughput.

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
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 80.93 ± 1.21 | 1482.62 | 1488.19/1488.19/1488.19 | 2491.78 | 62.86/90.00 | 120/643 | PASS |
| raw-stream | 80.15 ± 0.98 | 49.51 | 1469.67/1469.67/1469.67 | 2491.78 | 62.86/90.00 | 120/643 | PASS |
| native-non-stream | 77.18 ± 0.81 | 1554.65 | 1544.98/1544.98/1544.98 | 2491.78 | 62.86/90.00 | 120/643 | PASS |
| native-stream | 76.06 ± 0.41 | 1577.71 | 1603.46/1603.46/1603.46 | 2491.78 | 62.80/90.00 | 120/643 | PASS |
| managed-non-stream | 75.45 ± 0.63 | 1590.37 | 1658.71/1658.71/1658.71 | 2184.07 | 55.34/90.00 | 120/643 | PASS |
| managed-stream | 74.19 ± 0.75 | 86.66 | 1636.97/1636.97/1636.97 | 2189.15 | 55.34/90.00 | 120/643 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.482703 | 80.93 | 1482.62 | 2482.27 | 62.86 | 142.56 | 13.40 | 39.73 | 2040.00 | 75.87 |
| 2 | 1.531918 | 78.33 | 1531.84 | 2490.40 | 62.86 | 135.01 | 12.84 | 40.15 | 2040.00 | 76.87 |
| 3 | 1.481812 | 80.98 | 1481.73 | 2490.45 | 62.86 | 134.99 | 12.77 | 39.60 | 2040.00 | 76.83 |
| 4 | 1.483129 | 80.91 | 1483.05 | 2491.30 | 62.86 | 134.09 | 13.40 | 40.45 | 2040.00 | 77.83 |
| 5 | 1.497408 | 80.14 | 1497.32 | 2491.31 | 62.86 | 134.16 | 12.80 | 40.03 | 2040.00 | 77.80 |
| 6 | 1.466680 | 81.82 | 1466.60 | 2491.77 | 62.86 | 134.52 | 13.77 | 40.07 | 2040.00 | 76.00 |
| 7 | 1.465688 | 81.87 | 1465.60 | 2491.78 | 62.86 | 134.67 | 13.60 | 39.58 | 2040.00 | 75.87 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1888062332976308, 'inference': 9.539479558342615, 'postprocess': 1.2145012500354824}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

1.51% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.506907 | 79.63 | 50.58 | 2482.59 | 62.86 | 134.04 | 12.87 | 39.91 | 2040.00 | 75.90 |
| 2 | 1.497215 | 80.15 | 49.28 | 2490.40 | 62.86 | 134.70 | 13.40 | 40.17 | 2040.00 | 76.87 |
| 3 | 1.495144 | 80.26 | 49.41 | 2490.46 | 62.86 | 133.49 | 13.80 | 40.35 | 2040.00 | 76.87 |
| 4 | 1.523760 | 78.75 | 49.51 | 2490.98 | 62.86 | 134.76 | 13.16 | 39.95 | 2040.00 | 77.77 |
| 5 | 1.520991 | 78.90 | 51.07 | 2491.31 | 62.86 | 136.50 | 12.97 | 40.33 | 2040.00 | 77.84 |
| 6 | 1.488892 | 80.60 | 53.26 | 2491.77 | 62.86 | 134.12 | 13.07 | 39.78 | 2040.00 | 76.00 |
| 7 | 1.471272 | 81.56 | 49.05 | 2491.78 | 62.86 | 135.09 | 14.20 | 40.12 | 2040.00 | 75.93 |

Ready result delivery-interval P50/P95/P99 ms: 11.80/12.68/15.44.
Separate latency-pass duration: 1.47 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2887531249930362, 'inference': 9.619078925023436, 'postprocess': 1.2343665999992481}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

5.28% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.560977 | 76.87 | 1560.88 | 2483.87 | 62.86 | 133.95 | 13.16 | 40.06 | 2040.00 | 75.87 |
| 2 | 1.545466 | 77.65 | 1545.38 | 2489.59 | 62.86 | 141.23 | 13.45 | 40.13 | 2040.00 | 76.87 |
| 3 | 1.568833 | 76.49 | 1568.74 | 2490.46 | 62.86 | 133.52 | 13.12 | 40.25 | 2040.00 | 76.94 |
| 4 | 1.538464 | 78.00 | 1538.37 | 2488.29 | 62.86 | 140.73 | 13.39 | 40.42 | 2040.00 | 77.71 |
| 5 | 1.548787 | 77.48 | 1548.68 | 2491.31 | 62.86 | 133.52 | 13.32 | 40.52 | 2040.00 | 77.48 |
| 6 | 1.554750 | 77.18 | 1554.65 | 2491.20 | 62.86 | 141.22 | 13.71 | 40.08 | 2040.00 | 76.00 |
| 7 | 1.587696 | 75.58 | 1587.58 | 2491.78 | 62.86 | 132.64 | 13.25 | 39.98 | 2040.00 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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

6.39% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.577473 | 76.07 | 1577.40 | 2484.36 | 62.80 | 132.50 | 11.94 | 39.81 | 2040.00 | 75.88 |
| 2 | 1.587129 | 75.61 | 1587.06 | 2177.75 | 62.80 | 133.02 | 12.38 | 39.79 | 2040.00 | 76.81 |
| 3 | 1.593957 | 75.28 | 1593.87 | 2490.54 | 62.80 | 131.60 | 12.03 | 39.69 | 2040.00 | 76.88 |
| 4 | 1.577603 | 76.06 | 1577.52 | 2192.19 | 62.80 | 132.26 | 12.50 | 40.22 | 2040.00 | 77.00 |
| 5 | 1.568429 | 76.51 | 1568.36 | 2491.31 | 62.80 | 131.79 | 12.56 | 40.16 | 2040.00 | 77.00 |
| 6 | 1.577779 | 76.06 | 1577.71 | 2184.34 | 62.80 | 132.78 | 12.75 | 39.71 | 2040.00 | 76.00 |
| 7 | 1.587456 | 75.59 | 1587.37 | 2491.78 | 62.80 | 132.42 | 12.62 | 39.64 | 2040.00 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.60 s; not used for throughput.

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

7.37% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.619659 | 74.09 | 1619.57 | 2176.86 | 55.34 | 132.74 | 12.21 | 39.53 | 2040.00 | 76.00 |
| 2 | 1.581716 | 75.87 | 1581.63 | 2180.60 | 55.34 | 131.50 | 12.62 | 39.72 | 2040.00 | 76.66 |
| 3 | 1.581543 | 75.88 | 1581.46 | 2182.11 | 55.34 | 132.76 | 12.62 | 39.66 | 2040.00 | 76.94 |
| 4 | 1.592459 | 75.36 | 1592.36 | 2184.07 | 55.34 | 131.85 | 13.06 | 39.93 | 2040.00 | 77.00 |
| 5 | 1.586226 | 75.65 | 1586.13 | 2182.66 | 55.34 | 131.09 | 12.84 | 40.00 | 2040.00 | 77.00 |
| 6 | 1.600911 | 74.96 | 1600.82 | 2182.53 | 55.34 | 132.00 | 12.91 | 39.67 | 2040.00 | 76.12 |
| 7 | 1.590459 | 75.45 | 1590.37 | 2183.18 | 55.34 | 131.57 | 12.56 | 39.54 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
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

8.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.617362 | 74.19 | 88.02 | 2183.15 | 55.34 | 130.79 | 12.64 | 39.11 | 2040.00 | 76.00 |
| 2 | 1.620277 | 74.06 | 86.66 | 2186.14 | 55.34 | 130.84 | 11.00 | 39.15 | 2040.00 | 76.00 |
| 3 | 1.626821 | 73.76 | 87.73 | 2179.40 | 55.34 | 132.70 | 12.55 | 39.86 | 2040.00 | 77.00 |
| 4 | 1.604779 | 74.78 | 84.83 | 2184.05 | 55.34 | 132.36 | 12.19 | 39.34 | 2040.00 | 77.00 |
| 5 | 1.616821 | 74.22 | 87.61 | 2186.34 | 55.34 | 131.27 | 12.58 | 39.65 | 2040.00 | 76.94 |
| 6 | 1.636508 | 73.33 | 85.57 | 2189.15 | 55.34 | 131.25 | 12.18 | 39.26 | 2040.00 | 76.85 |
| 7 | 1.585990 | 75.66 | 86.44 | 2186.50 | 55.34 | 131.33 | 12.62 | 39.39 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.76/15.65/16.10.
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

## Detail — yolo26, video, batch 4

Status: **complete**. Load 0.13 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 190.11 ± 2.25 | 631.12 | 630.06/630.06/630.06 | 2520.02 | 120.45/186.00 | 120/643 | PASS |
| raw-stream | 189.97 ± 1.18 | 60.16 | 621.58/621.58/621.58 | 2515.19 | 120.45/186.00 | 120/643 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 172.81 ± 3.48 | 694.31 | 696.67/696.67/696.67 | 2510.51 | 97.52/186.00 | 120/643 | PASS |
| managed-stream | 166.35 ± 2.28 | 100.60 | 720.57/720.57/720.57 | 2510.46 | 97.52/186.00 | 120/643 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.636112 | 188.65 | 636.02 | 2469.17 | 120.45 | 197.93 | 14.08 | 42.05 | 2040.00 | 75.69 |
| 2 | 0.619525 | 193.70 | 619.44 | 2482.91 | 120.45 | 188.99 | 14.00 | 42.94 | 2040.00 | 75.85 |
| 3 | 0.618380 | 194.06 | 618.30 | 2480.43 | 120.45 | 185.83 | 11.69 | 41.83 | 2040.00 | 75.85 |
| 4 | 0.627460 | 191.25 | 627.37 | 2483.52 | 120.45 | 184.45 | 13.54 | 43.32 | 2040.00 | 76.69 |
| 5 | 0.631206 | 190.11 | 631.12 | 2483.52 | 120.45 | 189.68 | 10.92 | 43.00 | 2040.00 | 76.77 |
| 6 | 0.633262 | 189.49 | 633.16 | 2520.02 | 120.45 | 185.57 | 13.85 | 43.70 | 2040.00 | 77.31 |
| 7 | 0.635998 | 188.68 | 635.91 | 2482.02 | 120.45 | 184.97 | 12.77 | 42.41 | 2040.00 | 77.46 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.63 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1698907166495094, 'inference': 2.715862175008018, 'postprocess': 1.0462886916836094}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.627861 | 191.13 | 59.91 | 2476.67 | 120.45 | 188.76 | 13.77 | 42.02 | 2040.00 | 75.85 |
| 2 | 0.631674 | 189.97 | 61.29 | 2475.52 | 120.45 | 189.57 | 14.08 | 42.93 | 2040.00 | 75.77 |
| 3 | 0.632308 | 189.78 | 63.75 | 2480.55 | 120.45 | 187.95 | 13.77 | 42.12 | 2040.00 | 76.23 |
| 4 | 0.633470 | 189.43 | 61.32 | 2511.41 | 120.45 | 185.50 | 13.46 | 42.45 | 2040.00 | 76.69 |
| 5 | 0.626296 | 191.60 | 59.82 | 2481.57 | 120.45 | 183.96 | 14.08 | 43.00 | 2040.00 | 76.77 |
| 6 | 0.624890 | 192.03 | 59.66 | 2515.19 | 120.45 | 186.93 | 13.31 | 42.85 | 2040.00 | 77.69 |
| 7 | 0.635145 | 188.93 | 60.16 | 2482.02 | 120.45 | 187.87 | 12.92 | 43.13 | 2040.00 | 77.77 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/19.69/20.01.
Separate latency-pass duration: 0.62 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1632860166704024, 'inference': 2.6858106833590987, 'postprocess': 1.0271722083037578}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

10.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.709828 | 169.06 | 709.72 | 2478.46 | 97.52 | 178.66 | 11.87 | 42.56 | 2040.00 | 75.73 |
| 2 | 0.685639 | 175.02 | 685.55 | 2452.90 | 97.52 | 176.68 | 13.07 | 41.93 | 2040.00 | 75.79 |
| 3 | 0.718286 | 167.06 | 718.18 | 2479.08 | 97.52 | 177.19 | 12.33 | 42.40 | 2040.00 | 76.27 |
| 4 | 0.687583 | 174.52 | 687.49 | 2510.51 | 97.52 | 180.90 | 12.64 | 42.15 | 2040.00 | 76.50 |
| 5 | 0.694399 | 172.81 | 694.31 | 2480.70 | 97.52 | 176.15 | 11.86 | 42.92 | 2040.00 | 76.71 |
| 6 | 0.686169 | 174.88 | 686.07 | 2453.87 | 97.52 | 176.15 | 12.79 | 41.57 | 2040.00 | 76.93 |
| 7 | 0.714409 | 167.97 | 714.32 | 2481.16 | 97.52 | 173.35 | 12.60 | 43.30 | 2040.00 | 77.40 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.70 s; not used for throughput.

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

13.79% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.721360 | 166.35 | 101.46 | 2449.40 | 97.52 | 182.10 | 11.20 | 41.66 | 2040.00 | 75.67 |
| 2 | 0.709683 | 169.09 | 95.69 | 2452.41 | 97.52 | 177.30 | 11.33 | 40.92 | 2040.00 | 75.73 |
| 3 | 0.735979 | 163.05 | 105.22 | 2507.54 | 97.52 | 180.03 | 11.40 | 42.15 | 2040.00 | 76.27 |
| 4 | 0.728915 | 164.63 | 97.73 | 2510.46 | 97.52 | 172.42 | 11.33 | 41.46 | 2040.00 | 76.47 |
| 5 | 0.712479 | 168.43 | 100.82 | 2451.07 | 97.52 | 179.06 | 12.33 | 42.41 | 2040.00 | 76.87 |
| 6 | 0.711226 | 168.72 | 96.23 | 2453.80 | 97.52 | 178.33 | 11.07 | 41.74 | 2040.00 | 76.73 |
| 7 | 0.723724 | 165.81 | 100.60 | 2508.84 | 97.52 | 178.48 | 10.67 | 42.74 | 2040.00 | 77.73 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/21.77/22.62.
Separate latency-pass duration: 0.72 s; not used for throughput.

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

Status: **complete**. Load 5.79 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 76.87 ± 0.66 | 1560.88 | 1564.36/1564.36/1564.36 | 2764.55 | 115.88/134.00 | 120/708 | PASS |
| raw-stream | 77.74 ± 1.85 | 49.91 | 1574.05/1574.05/1574.05 | 2753.89 | 115.88/134.00 | 120/708 | PASS |
| native-non-stream | 75.55 ± 0.73 | 1588.34 | 1641.29/1641.29/1641.29 | 2754.22 | 115.88/134.00 | 120/708 | PASS |
| native-stream | 72.32 ± 0.94 | 1659.20 | 1644.83/1644.83/1644.83 | 2754.24 | 115.88/134.00 | 120/708 | PASS |
| managed-non-stream | 73.53 ± 0.74 | 1631.96 | 1650.63/1650.63/1650.63 | 2447.39 | 107.97/134.00 | 120/708 | PASS |
| managed-stream | 72.74 ± 1.88 | 86.39 | 1654.86/1654.86/1654.86 | 2447.39 | 107.97/134.00 | 120/708 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.562898 | 76.78 | 1562.81 | 2729.27 | 115.88 | 140.92 | 18.66 | 46.23 | 2040.00 | 78.84 |
| 2 | 1.560993 | 76.87 | 1560.88 | 2752.30 | 115.88 | 133.17 | 18.71 | 45.30 | 2040.00 | 76.58 |
| 3 | 1.561132 | 76.87 | 1561.05 | 2764.55 | 115.88 | 140.70 | 17.74 | 44.38 | 2040.00 | 75.87 |
| 4 | 1.547364 | 77.55 | 1547.28 | 2753.02 | 115.88 | 132.41 | 18.77 | 45.49 | 2040.00 | 75.81 |
| 5 | 1.569137 | 76.48 | 1569.05 | 2753.03 | 115.88 | 134.08 | 18.25 | 44.44 | 2040.00 | 75.56 |
| 6 | 1.530039 | 78.43 | 1529.95 | 2753.36 | 115.88 | 132.59 | 19.26 | 45.91 | 2040.00 | 76.32 |
| 7 | 1.548417 | 77.50 | 1548.32 | 2753.36 | 115.88 | 132.99 | 17.19 | 45.02 | 2040.00 | 76.35 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2114712499548357, 'inference': 10.129358800031696, 'postprocess': 1.2646839250237463}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.25% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.550599 | 77.39 | 50.53 | 2729.62 | 115.88 | 134.24 | 19.06 | 46.84 | 2040.00 | 78.87 |
| 2 | 1.586870 | 75.62 | 49.91 | 2751.95 | 115.88 | 132.12 | 18.19 | 45.62 | 2040.00 | 76.88 |
| 3 | 1.520856 | 78.90 | 49.78 | 2753.89 | 115.88 | 133.52 | 19.03 | 45.85 | 2040.00 | 75.94 |
| 4 | 1.543516 | 77.74 | 49.50 | 2752.96 | 115.88 | 132.90 | 18.58 | 45.25 | 2040.00 | 75.71 |
| 5 | 1.517124 | 79.10 | 49.89 | 2753.03 | 115.88 | 133.08 | 19.19 | 44.89 | 2040.00 | 75.74 |
| 6 | 1.538620 | 77.99 | 50.44 | 2753.30 | 115.88 | 132.76 | 18.90 | 45.43 | 2040.00 | 76.10 |
| 7 | 1.622724 | 73.95 | 52.40 | 2753.36 | 115.88 | 133.46 | 18.18 | 45.49 | 2040.00 | 76.76 |

Ready result delivery-interval P50/P95/P99 ms: 12.64/13.80/16.59.
Separate latency-pass duration: 1.57 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2275067749821271, 'inference': 10.03184462504123, 'postprocess': 1.2382439999782946}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.80% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.588432 | 75.55 | 1588.34 | 2731.45 | 115.88 | 131.87 | 18.62 | 46.46 | 2040.00 | 78.88 |
| 2 | 1.604745 | 74.78 | 1604.67 | 2749.00 | 115.88 | 138.60 | 19.09 | 45.60 | 2040.00 | 76.94 |
| 3 | 1.572625 | 76.31 | 1572.52 | 2754.22 | 115.88 | 131.21 | 19.75 | 45.86 | 2040.00 | 75.88 |
| 4 | 1.590842 | 75.43 | 1590.73 | 2752.61 | 115.88 | 141.11 | 19.06 | 45.63 | 2040.00 | 75.81 |
| 5 | 1.565005 | 76.68 | 1564.90 | 2753.03 | 115.88 | 132.65 | 19.25 | 45.88 | 2040.00 | 75.81 |
| 6 | 1.587546 | 75.59 | 1587.45 | 2753.03 | 115.88 | 139.32 | 19.47 | 45.34 | 2040.00 | 75.94 |
| 7 | 1.606527 | 74.70 | 1606.43 | 2753.36 | 115.88 | 131.53 | 18.62 | 45.69 | 2040.00 | 76.81 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
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


### native-stream

6.58% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.677525 | 71.53 | 1677.46 | 2731.91 | 115.88 | 131.47 | 17.41 | 45.94 | 2040.00 | 78.88 |
| 2 | 1.659476 | 72.31 | 1659.41 | 2444.14 | 115.88 | 132.00 | 17.36 | 45.15 | 2040.00 | 77.00 |
| 3 | 1.659276 | 72.32 | 1659.20 | 2754.24 | 115.88 | 131.48 | 17.82 | 44.77 | 2040.00 | 75.88 |
| 4 | 1.649112 | 72.77 | 1649.04 | 2446.39 | 115.88 | 131.09 | 17.94 | 44.92 | 2040.00 | 75.76 |
| 5 | 1.704808 | 70.39 | 1704.74 | 2753.03 | 115.88 | 132.03 | 16.71 | 44.41 | 2040.00 | 75.82 |
| 6 | 1.656937 | 72.42 | 1656.87 | 2447.38 | 115.88 | 131.21 | 17.82 | 44.51 | 2040.00 | 75.88 |
| 7 | 1.637590 | 73.28 | 1637.52 | 2753.36 | 115.88 | 131.36 | 17.88 | 45.32 | 2040.00 | 76.85 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
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


### managed-non-stream

5.40% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.616288 | 74.24 | 1616.20 | 2425.56 | 107.97 | 132.86 | 18.24 | 45.77 | 2040.00 | 78.82 |
| 2 | 1.622992 | 73.94 | 1622.90 | 2443.86 | 107.97 | 131.19 | 18.33 | 45.70 | 2040.00 | 77.88 |
| 3 | 1.629422 | 73.65 | 1629.33 | 2445.59 | 107.97 | 130.97 | 17.48 | 44.49 | 2040.00 | 75.82 |
| 4 | 1.634472 | 73.42 | 1634.35 | 2446.23 | 107.97 | 130.47 | 18.09 | 44.81 | 2040.00 | 75.82 |
| 5 | 1.659871 | 72.29 | 1659.78 | 2447.24 | 107.97 | 130.63 | 17.58 | 44.54 | 2040.00 | 75.82 |
| 6 | 1.657693 | 72.39 | 1657.60 | 2447.39 | 107.97 | 131.55 | 17.27 | 44.46 | 2040.00 | 75.88 |
| 7 | 1.632054 | 73.53 | 1631.96 | 2447.39 | 107.97 | 131.69 | 18.03 | 45.25 | 2040.00 | 76.88 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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

6.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.630770 | 73.58 | 89.92 | 2440.27 | 107.97 | 131.24 | 18.06 | 45.30 | 2040.00 | 78.06 |
| 2 | 1.637725 | 73.27 | 85.97 | 2443.45 | 107.97 | 131.11 | 17.91 | 44.96 | 2040.00 | 77.85 |
| 3 | 1.662171 | 72.19 | 87.39 | 2445.87 | 107.97 | 130.49 | 17.24 | 44.32 | 2040.00 | 75.82 |
| 4 | 1.649732 | 72.74 | 85.61 | 2446.20 | 107.97 | 130.12 | 16.55 | 43.93 | 2040.00 | 75.82 |
| 5 | 1.693703 | 70.85 | 89.24 | 2447.22 | 107.97 | 130.85 | 17.41 | 44.67 | 2040.00 | 75.82 |
| 6 | 1.758389 | 68.24 | 86.39 | 2447.38 | 107.97 | 130.15 | 15.66 | 43.57 | 2040.00 | 75.83 |
| 7 | 1.643793 | 73.00 | 84.80 | 2447.39 | 107.97 | 130.19 | 17.79 | 45.21 | 2040.00 | 76.88 |

Ready result delivery-interval P50/P95/P99 ms: 13.09/13.95/14.62.
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

## Detail — yolo-world, video, batch 4

Status: **complete**. Load 5.84 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `59dca6fd7f6542bec73a94624757865ff187abc5cda8dfef02a58dcd1fe28c47`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 187.12 ± 4.00 | 641.20 | 646.21/646.21/646.21 | 2693.27 | 209.49/292.00 | 120/709 | PASS |
| raw-stream | 186.63 ± 2.84 | 61.95 | 639.36/639.36/639.36 | 2686.75 | 209.49/292.00 | 120/709 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 171.96 ± 2.25 | 697.72 | 712.16/712.16/712.16 | 2686.76 | 188.26/292.00 | 120/709 | PASS |
| managed-stream | 164.54 ± 3.80 | 100.51 | 727.11/727.11/727.11 | 2686.62 | 188.26/292.00 | 120/709 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.632435 | 189.74 | 632.35 | 2650.36 | 209.49 | 197.13 | 26.38 | 53.76 | 2040.00 | 77.23 |
| 2 | 0.647360 | 185.37 | 647.28 | 2693.27 | 209.49 | 181.49 | 25.54 | 53.94 | 2040.00 | 78.23 |
| 3 | 0.634808 | 189.03 | 634.72 | 2657.17 | 209.49 | 185.98 | 24.54 | 52.15 | 2040.00 | 78.08 |
| 4 | 0.641283 | 187.12 | 641.20 | 2657.00 | 209.49 | 187.35 | 25.77 | 51.61 | 2040.00 | 79.23 |
| 5 | 0.660262 | 181.75 | 660.16 | 2665.23 | 209.49 | 194.97 | 24.43 | 53.92 | 2040.00 | 78.71 |
| 6 | 0.638196 | 188.03 | 638.11 | 2667.64 | 209.49 | 179.39 | 25.08 | 55.40 | 2040.00 | 78.92 |
| 7 | 0.670560 | 178.96 | 670.48 | 2667.68 | 209.49 | 181.67 | 25.29 | 51.04 | 2040.00 | 78.71 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 0.65 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1134624333711447, 'inference': 2.750629216673891, 'postprocess': 1.0432654583307037}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.61% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.638661 | 187.89 | 61.95 | 2650.80 | 209.49 | 186.72 | 26.08 | 54.09 | 2040.00 | 77.23 |
| 2 | 0.653140 | 183.73 | 61.39 | 2682.72 | 209.49 | 188.59 | 26.69 | 52.34 | 2040.00 | 77.69 |
| 3 | 0.645215 | 185.98 | 61.04 | 2655.04 | 209.49 | 185.65 | 25.00 | 54.10 | 2040.00 | 78.38 |
| 4 | 0.637120 | 188.35 | 61.06 | 2686.75 | 209.49 | 182.85 | 27.23 | 52.43 | 2040.00 | 78.77 |
| 5 | 0.664299 | 180.64 | 73.45 | 2666.04 | 209.49 | 189.19 | 23.86 | 51.92 | 2040.00 | 79.14 |
| 6 | 0.637384 | 188.27 | 65.47 | 2663.66 | 209.49 | 198.53 | 25.69 | 54.59 | 2040.00 | 79.38 |
| 7 | 0.642982 | 186.63 | 62.05 | 2665.97 | 209.49 | 186.94 | 25.23 | 53.29 | 2040.00 | 79.23 |

Ready result delivery-interval P50/P95/P99 ms: 0.03/20.33/20.91.
Separate latency-pass duration: 0.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1426149250155504, 'inference': 2.7627746166672296, 'postprocess': 1.0461350249822015}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

8.18% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.711907 | 168.56 | 711.82 | 2650.68 | 188.26 | 175.37 | 23.73 | 52.05 | 2040.00 | 77.07 |
| 2 | 0.697817 | 171.96 | 697.72 | 2558.56 | 188.26 | 177.26 | 24.57 | 51.31 | 2040.00 | 77.57 |
| 3 | 0.712591 | 168.40 | 712.49 | 2655.00 | 188.26 | 174.23 | 24.20 | 53.34 | 2040.00 | 78.20 |
| 4 | 0.693720 | 172.98 | 693.63 | 2686.76 | 188.26 | 177.29 | 24.14 | 52.62 | 2040.00 | 78.57 |
| 5 | 0.707728 | 169.56 | 707.60 | 2666.24 | 188.26 | 176.30 | 22.53 | 53.53 | 2040.00 | 79.27 |
| 6 | 0.691917 | 173.43 | 691.82 | 2453.28 | 188.26 | 176.65 | 24.79 | 51.95 | 2040.00 | 79.21 |
| 7 | 0.692791 | 173.21 | 692.68 | 2665.84 | 188.26 | 175.94 | 24.14 | 54.49 | 2040.00 | 78.57 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.71 s; not used for throughput.

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

10.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.729309 | 164.54 | 101.13 | 2554.96 | 188.26 | 175.98 | 23.33 | 50.93 | 2040.00 | 77.07 |
| 2 | 0.741310 | 161.88 | 97.25 | 2558.25 | 188.26 | 173.79 | 22.40 | 50.15 | 2040.00 | 77.20 |
| 3 | 0.733173 | 163.67 | 102.12 | 2683.64 | 188.26 | 175.43 | 21.60 | 51.86 | 2040.00 | 78.07 |
| 4 | 0.702495 | 170.82 | 95.65 | 2686.62 | 188.26 | 173.17 | 25.00 | 50.70 | 2040.00 | 78.29 |
| 5 | 0.731698 | 164.00 | 100.51 | 2450.54 | 188.26 | 177.56 | 21.20 | 51.51 | 2040.00 | 79.07 |
| 6 | 0.698519 | 171.79 | 97.71 | 2453.25 | 188.26 | 174.69 | 20.79 | 51.32 | 2040.00 | 79.00 |
| 7 | 0.717147 | 167.33 | 102.72 | 2640.59 | 188.26 | 178.91 | 23.33 | 52.50 | 2040.00 | 78.47 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/21.90/23.08.
Separate latency-pass duration: 0.73 s; not used for throughput.

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

## Ten-minute managed-stream stability

GPU: NVIDIA L4; model yolov8n.pt; elapsed 600.10 s; frames 49680.
Pass: **True**. Exceptions: `[]`. Failed gates: `[]`.
Each loop verifies frame count and order, consumes without retaining frames, closes the stream and collects unreachable Python objects. The model remains loaded; allocator caches are not emptied between loops.

| Metric | First 60-second median | Last 60-second median | Change % |
| --- | ---: | ---: | ---: |
| rss_mib | 1700.84 | 1702.32 | 0.09 |
| vram_allocated_mib | 12.10 | 12.10 | 0.00 |
| fps | 92.11 | 91.94 | -0.19 |
| open_fds | 44.00 | 44.00 | 0.00 |

Whole-check throughput including between-loop cleanup: 82.79 FPS. Per-loop FPS above excludes between-loop cleanup.

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
