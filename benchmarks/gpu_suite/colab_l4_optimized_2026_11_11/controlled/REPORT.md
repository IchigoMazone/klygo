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

## FPS matrix — images, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 99.84 | 99.92 | 89.99 | 90.97 | 95.33 | 95.82 |
| yolo11 | 87.29 | 88.05 | 79.84 | 80.61 | 84.01 | 84.19 |
| yolo26 | 78.63 | 78.76 | 72.63 | 72.63 | 76.28 | 76.51 |
| yolo-world | 78.80 | 78.85 | 73.99 | 73.91 | 77.55 | 77.60 |
| grounding-dino | 4.89 | N/A | 4.84 | N/A | 4.85 | 4.85 |
| grounding-dino-base | 3.73 | N/A | 3.70 | N/A | 3.70 | 3.68 |
| faster-rcnn | 44.87 | N/A | 42.66 | N/A | 41.00 | 41.22 |
| d-fine | 2.66 | N/A | 2.64 | N/A | 2.65 | 2.64 |
| locate-anything | 2.47 | N/A | 2.46 | N/A | 2.44 | 2.46 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — images, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 225.23 | 225.40 | 196.82 | 196.08 | 217.52 | 218.60 |
| yolo11 | 200.78 | 195.01 | 177.56 | 177.73 | 193.52 | 195.45 |
| yolo26 | 194.44 | 189.29 | 170.55 | 171.19 | 188.32 | 185.51 |
| yolo-world | 191.63 | 192.99 | 184.19 | 184.06 | 199.71 | 199.62 |
| grounding-dino | 5.05 | N/A | 5.05 | N/A | 5.01 | 5.03 |
| grounding-dino-base | 3.70 | N/A | 3.70 | N/A | 3.69 | 3.70 |
| faster-rcnn | 72.03 | N/A | 69.89 | N/A | 68.59 | 68.74 |
| d-fine | 10.21 | N/A | 10.07 | N/A | 10.01 | 10.04 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 107.83 | 105.96 | 100.89 | 100.40 | 99.21 | 97.47 |
| yolo11 | 87.42 | 86.99 | 82.38 | 82.92 | 81.53 | 80.12 |
| yolo26 | 78.80 | 78.78 | 75.72 | 74.03 | 74.63 | 73.00 |
| yolo-world | 77.43 | 77.14 | 74.56 | 72.74 | 72.49 | 72.45 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — video, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 223.68 | 218.43 | N/A | N/A | 200.72 | 197.03 |
| yolo11 | 197.33 | 198.40 | N/A | N/A | 182.84 | 176.70 |
| yolo26 | 185.28 | 184.57 | N/A | N/A | 171.28 | 164.15 |
| yolo-world | 185.90 | 188.09 | N/A | N/A | 174.96 | 168.96 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, images, batch 1

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 99.84 ± 0.71 | 14.23 | 9.68/9.87/10.53 | 1829.99 | 29.74/60.00 | 64/320 | PASS |
| raw-stream | 99.92 ± 0.81 | 14.16 | 9.69/9.98/12.21 | 1829.99 | 29.74/60.00 | 64/320 | PASS |
| native-non-stream | 89.99 ± 0.83 | 16.01 | 10.85/11.86/14.29 | 1829.99 | 29.71/60.00 | 64/320 | PASS |
| native-stream | 90.97 ± 1.48 | 12.91 | 10.82/11.32/15.53 | 1776.65 | 29.71/60.00 | 64/320 | PASS |
| managed-non-stream | 95.33 ± 0.74 | 12.22 | 10.33/12.54/12.78 | 1775.52 | 29.71/60.00 | 64/320 | PASS |
| managed-stream | 95.82 ± 1.65 | 12.15 | 10.19/10.68/11.33 | 1775.59 | 29.71/60.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.642402 | 99.63 | 12.00 | 1825.50 | 29.74 | 100.46 | 14.31 | 34.71 | 2040.00 | 48.00 |
| 2 | 0.638127 | 100.29 | 14.31 | 1829.41 | 29.74 | 100.37 | 11.69 | 35.50 | 2040.00 | 49.85 |
| 3 | 0.637550 | 100.38 | 14.23 | 1829.41 | 29.74 | 100.42 | 11.15 | 35.08 | 2040.00 | 50.00 |
| 4 | 0.645211 | 99.19 | 14.59 | 1829.82 | 29.74 | 100.47 | 11.69 | 35.42 | 2040.00 | 51.85 |
| 5 | 0.645103 | 99.21 | 14.66 | 1829.82 | 29.74 | 100.48 | 11.08 | 34.89 | 2040.00 | 51.77 |
| 6 | 0.632527 | 101.18 | 14.23 | 1829.99 | 29.74 | 100.51 | 13.08 | 36.01 | 2040.00 | 53.00 |
| 7 | 0.641013 | 99.84 | 14.22 | 1829.99 | 29.74 | 100.48 | 11.77 | 35.80 | 2040.00 | 53.46 |

Ready result delivery-interval P50/P95/P99 ms: 9.68/9.87/10.53.
Separate latency-pass duration: 0.62 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6184740780668108, 'inference': 6.7944589687556345, 'postprocess': 1.247641359498175}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.07% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.640525 | 99.92 | 14.16 | 1828.13 | 29.74 | 100.48 | 13.85 | 35.08 | 2040.00 | 48.15 |
| 2 | 0.639276 | 100.11 | 11.21 | 1826.77 | 29.74 | 100.44 | 12.00 | 35.07 | 2040.00 | 49.85 |
| 3 | 0.636890 | 100.49 | 14.27 | 1829.41 | 29.74 | 100.47 | 13.00 | 35.18 | 2040.00 | 50.23 |
| 4 | 0.652778 | 98.04 | 11.36 | 1826.30 | 29.74 | 100.55 | 11.85 | 35.12 | 2040.00 | 51.69 |
| 5 | 0.642261 | 99.65 | 14.98 | 1829.82 | 29.74 | 100.46 | 11.08 | 35.69 | 2040.00 | 51.85 |
| 6 | 0.646073 | 99.06 | 11.26 | 1826.47 | 29.74 | 100.61 | 12.15 | 35.61 | 2040.00 | 53.00 |
| 7 | 0.640540 | 99.92 | 14.29 | 1829.99 | 29.74 | 100.45 | 11.08 | 35.30 | 2040.00 | 53.77 |

Ready result delivery-interval P50/P95/P99 ms: 9.70/9.98/12.19.
Separate latency-pass duration: 0.63 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6119476874459906, 'inference': 6.739759015488289, 'postprocess': 1.2850747186234912}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

10.70% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.704766 | 90.81 | 16.14 | 1828.13 | 29.71 | 100.47 | 12.71 | 35.07 | 2040.00 | 48.43 |
| 2 | 0.715325 | 89.47 | 12.99 | 1775.93 | 29.71 | 100.51 | 10.73 | 34.37 | 2040.00 | 49.80 |
| 3 | 0.705762 | 90.68 | 16.01 | 1829.41 | 29.71 | 100.40 | 11.86 | 35.48 | 2040.00 | 50.43 |
| 4 | 0.717873 | 89.15 | 12.98 | 1776.50 | 29.71 | 100.50 | 12.93 | 34.75 | 2040.00 | 51.53 |
| 5 | 0.713354 | 89.72 | 16.14 | 1829.82 | 29.71 | 100.51 | 11.07 | 35.93 | 2040.00 | 52.00 |
| 6 | 0.711187 | 89.99 | 12.90 | 1776.65 | 29.71 | 100.53 | 13.13 | 34.89 | 2040.00 | 53.00 |
| 7 | 0.699792 | 91.46 | 16.23 | 1829.99 | 29.71 | 100.48 | 11.00 | 35.41 | 2040.00 | 53.71 |

Ready result delivery-interval P50/P95/P99 ms: 10.85/11.84/14.29.
Separate latency-pass duration: 0.71 s; not used for throughput.

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

9.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.706879 | 90.54 | 12.65 | 1773.90 | 29.71 | 100.44 | 13.36 | 34.66 | 2040.00 | 48.79 |
| 2 | 0.725694 | 88.19 | 13.71 | 1775.84 | 29.71 | 100.51 | 13.00 | 34.70 | 2040.00 | 49.53 |
| 3 | 0.699751 | 91.46 | 12.91 | 1775.03 | 29.71 | 100.48 | 13.29 | 34.72 | 2040.00 | 50.79 |
| 4 | 0.703513 | 90.97 | 13.37 | 1776.50 | 29.71 | 100.49 | 11.57 | 35.55 | 2040.00 | 51.00 |
| 5 | 0.700302 | 91.39 | 12.80 | 1775.34 | 29.71 | 100.50 | 13.00 | 34.98 | 2040.00 | 52.00 |
| 6 | 0.722883 | 88.53 | 13.30 | 1776.65 | 29.71 | 100.53 | 11.13 | 35.63 | 2040.00 | 52.87 |
| 7 | 0.696223 | 91.92 | 12.44 | 1775.51 | 29.71 | 100.46 | 10.71 | 35.70 | 2040.00 | 53.79 |

Ready result delivery-interval P50/P95/P99 ms: 10.82/11.32/15.48.
Separate latency-pass duration: 0.71 s; not used for throughput.

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

4.07% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.664900 | 96.26 | 12.17 | 1774.03 | 29.71 | 100.54 | 12.14 | 34.86 | 2040.00 | 48.86 |
| 2 | 0.680220 | 94.09 | 12.62 | 1774.49 | 29.71 | 100.55 | 13.43 | 34.57 | 2040.00 | 49.21 |
| 3 | 0.671597 | 95.30 | 12.11 | 1775.12 | 29.71 | 100.56 | 13.00 | 35.08 | 2040.00 | 50.86 |
| 4 | 0.666896 | 95.97 | 12.35 | 1775.33 | 29.71 | 100.47 | 11.57 | 35.46 | 2040.00 | 51.00 |
| 5 | 0.671382 | 95.33 | 12.22 | 1775.35 | 29.71 | 100.49 | 13.86 | 35.35 | 2040.00 | 52.36 |
| 6 | 0.672213 | 95.21 | 13.44 | 1775.48 | 29.71 | 100.56 | 11.36 | 35.66 | 2040.00 | 52.79 |
| 7 | 0.665918 | 96.11 | 12.04 | 1775.52 | 29.71 | 100.56 | 13.14 | 35.59 | 2040.00 | 53.86 |

Ready result delivery-interval P50/P95/P99 ms: 10.33/12.54/12.78.
Separate latency-pass duration: 0.68 s; not used for throughput.

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

4.31% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.673071 | 95.09 | 12.08 | 1774.21 | 29.71 | 100.51 | 11.71 | 35.20 | 2040.00 | 48.86 |
| 2 | 0.691575 | 92.54 | 12.23 | 1774.31 | 29.71 | 100.52 | 10.71 | 34.61 | 2040.00 | 48.86 |
| 3 | 0.665059 | 96.23 | 12.11 | 1775.29 | 29.71 | 100.52 | 12.15 | 35.49 | 2040.00 | 50.85 |
| 4 | 0.665101 | 96.23 | 12.18 | 1775.33 | 29.71 | 100.53 | 11.43 | 34.96 | 2040.00 | 50.79 |
| 5 | 0.668547 | 95.73 | 12.16 | 1775.43 | 29.71 | 100.56 | 13.21 | 35.84 | 2040.00 | 52.64 |
| 6 | 0.667934 | 95.82 | 12.15 | 1775.48 | 29.71 | 100.92 | 11.00 | 35.45 | 2040.00 | 52.77 |
| 7 | 0.652816 | 98.04 | 11.89 | 1775.59 | 29.71 | 100.46 | 13.54 | 35.64 | 2040.00 | 53.85 |

Ready result delivery-interval P50/P95/P99 ms: 10.20/10.68/11.32.
Separate latency-pass duration: 0.66 s; not used for throughput.

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

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 87.29 ± 0.82 | 15.69 | 11.18/11.55/12.40 | 1910.02 | 59.81/88.00 | 64/320 | PASS |
| raw-stream | 88.05 ± 0.71 | 15.52 | 11.19/12.50/14.65 | 1910.02 | 59.81/88.00 | 64/320 | PASS |
| native-non-stream | 79.84 ± 1.57 | 17.39 | 12.24/14.78/15.91 | 1910.02 | 59.77/88.00 | 64/320 | PASS |
| native-stream | 80.61 ± 1.13 | 14.47 | 12.22/13.05/17.23 | 1856.95 | 59.77/88.00 | 64/320 | PASS |
| managed-non-stream | 84.01 ± 0.81 | 13.79 | 11.67/12.17/13.11 | 1856.11 | 59.77/88.00 | 64/320 | PASS |
| managed-stream | 84.19 ± 0.32 | 13.70 | 11.83/12.43/12.99 | 1856.19 | 59.77/88.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.749097 | 85.44 | 13.51 | 1905.45 | 59.81 | 100.48 | 11.80 | 35.62 | 2040.00 | 55.87 |
| 2 | 0.729850 | 87.69 | 15.61 | 1909.43 | 59.81 | 100.53 | 13.93 | 35.98 | 2040.00 | 56.87 |
| 3 | 0.728807 | 87.81 | 15.69 | 1909.43 | 59.81 | 100.48 | 11.73 | 35.55 | 2040.00 | 57.00 |
| 4 | 0.733337 | 87.27 | 15.62 | 1909.83 | 59.81 | 100.47 | 13.33 | 36.14 | 2040.00 | 58.00 |
| 5 | 0.729866 | 87.69 | 16.60 | 1909.83 | 59.81 | 100.44 | 11.47 | 35.63 | 2040.00 | 58.33 |
| 6 | 0.735841 | 86.98 | 15.89 | 1910.02 | 59.81 | 100.49 | 12.80 | 36.32 | 2040.00 | 59.53 |
| 7 | 0.733174 | 87.29 | 15.99 | 1910.02 | 59.81 | 100.59 | 11.73 | 35.84 | 2040.00 | 59.60 |

Ready result delivery-interval P50/P95/P99 ms: 11.18/11.55/12.40.
Separate latency-pass duration: 0.72 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6198960780068319, 'inference': 8.405523671967785, 'postprocess': 1.2405350312292285}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.89% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.735022 | 87.07 | 15.60 | 1908.08 | 59.81 | 100.48 | 13.33 | 35.88 | 2040.00 | 55.87 |
| 2 | 0.725187 | 88.25 | 12.77 | 1908.55 | 59.81 | 100.47 | 12.40 | 35.83 | 2040.00 | 56.87 |
| 3 | 0.738532 | 86.66 | 15.52 | 1909.43 | 59.81 | 100.49 | 13.33 | 36.17 | 2040.00 | 57.00 |
| 4 | 0.721456 | 88.71 | 12.67 | 1908.95 | 59.81 | 100.50 | 13.33 | 35.89 | 2040.00 | 58.00 |
| 5 | 0.730276 | 87.64 | 15.69 | 1909.83 | 59.81 | 100.47 | 12.53 | 36.12 | 2040.00 | 58.73 |
| 6 | 0.726850 | 88.05 | 12.97 | 1909.14 | 59.81 | 100.49 | 13.20 | 36.02 | 2040.00 | 59.00 |
| 7 | 0.726625 | 88.08 | 15.65 | 1910.02 | 59.81 | 100.48 | 12.27 | 36.40 | 2040.00 | 59.80 |

Ready result delivery-interval P50/P95/P99 ms: 11.19/12.52/14.62.
Separate latency-pass duration: 0.73 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6131563906706106, 'inference': 8.225786796600687, 'postprocess': 1.2534644374397885}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

8.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.804502 | 79.55 | 17.80 | 1908.08 | 59.77 | 100.43 | 11.06 | 35.57 | 2040.00 | 55.88 |
| 2 | 0.789059 | 81.11 | 14.21 | 1855.93 | 59.77 | 100.44 | 13.06 | 35.18 | 2040.00 | 56.81 |
| 3 | 0.801647 | 79.84 | 17.50 | 1909.43 | 59.77 | 100.44 | 10.88 | 35.93 | 2040.00 | 57.00 |
| 4 | 0.801755 | 79.82 | 14.63 | 1856.50 | 59.77 | 100.50 | 13.00 | 35.65 | 2040.00 | 58.00 |
| 5 | 0.838096 | 76.36 | 17.39 | 1909.83 | 59.77 | 100.86 | 11.53 | 36.04 | 2040.00 | 58.71 |
| 6 | 0.795952 | 80.41 | 14.54 | 1856.95 | 59.77 | 100.45 | 11.25 | 35.57 | 2040.00 | 59.00 |
| 7 | 0.792530 | 80.75 | 17.40 | 1910.02 | 59.77 | 100.41 | 13.25 | 36.25 | 2040.00 | 59.81 |

Ready result delivery-interval P50/P95/P99 ms: 12.24/14.80/15.90.
Separate latency-pass duration: 0.80 s; not used for throughput.

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

8.58% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.787104 | 81.31 | 14.12 | 1853.85 | 59.77 | 100.45 | 11.50 | 35.07 | 2040.00 | 56.00 |
| 2 | 0.793964 | 80.61 | 14.85 | 1855.84 | 59.77 | 100.45 | 11.25 | 35.12 | 2040.00 | 56.75 |
| 3 | 0.804038 | 79.60 | 14.47 | 1855.02 | 59.77 | 100.46 | 11.00 | 35.72 | 2040.00 | 57.50 |
| 4 | 0.787314 | 81.29 | 14.83 | 1856.50 | 59.77 | 100.48 | 11.25 | 35.30 | 2040.00 | 58.00 |
| 5 | 0.817715 | 78.27 | 14.45 | 1855.65 | 59.77 | 100.59 | 10.65 | 35.77 | 2040.00 | 58.82 |
| 6 | 0.786882 | 81.33 | 14.78 | 1856.95 | 59.77 | 100.46 | 12.94 | 36.09 | 2040.00 | 59.00 |
| 7 | 0.796111 | 80.39 | 14.41 | 1855.83 | 59.77 | 100.44 | 11.69 | 36.14 | 2040.00 | 59.88 |

Ready result delivery-interval P50/P95/P99 ms: 12.21/13.05/17.14.
Separate latency-pass duration: 0.80 s; not used for throughput.

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

3.93% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.761966 | 83.99 | 13.63 | 1853.94 | 59.77 | 100.49 | 14.00 | 35.20 | 2040.00 | 56.00 |
| 2 | 0.776602 | 82.41 | 13.79 | 1854.48 | 59.77 | 100.51 | 12.50 | 35.63 | 2040.00 | 56.88 |
| 3 | 0.761775 | 84.01 | 13.67 | 1855.11 | 59.77 | 100.51 | 13.69 | 35.67 | 2040.00 | 57.81 |
| 4 | 0.761275 | 84.07 | 13.88 | 1855.32 | 59.77 | 100.51 | 13.25 | 35.45 | 2040.00 | 57.88 |
| 5 | 0.758542 | 84.37 | 13.81 | 1855.93 | 59.77 | 100.48 | 11.33 | 35.43 | 2040.00 | 58.73 |
| 6 | 0.755580 | 84.70 | 13.85 | 1856.07 | 59.77 | 100.45 | 12.40 | 35.96 | 2040.00 | 59.00 |
| 7 | 0.771455 | 82.96 | 13.76 | 1856.11 | 59.77 | 100.53 | 11.25 | 35.69 | 2040.00 | 59.88 |

Ready result delivery-interval P50/P95/P99 ms: 11.67/12.16/13.10.
Separate latency-pass duration: 0.75 s; not used for throughput.

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

3.88% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.760199 | 84.19 | 13.59 | 1854.12 | 59.77 | 100.47 | 11.25 | 35.36 | 2040.00 | 56.12 |
| 2 | 0.760360 | 84.17 | 13.70 | 1854.31 | 59.77 | 100.52 | 12.00 | 35.30 | 2040.00 | 56.62 |
| 3 | 0.764924 | 83.67 | 13.57 | 1855.29 | 59.77 | 100.53 | 11.31 | 35.76 | 2040.00 | 57.81 |
| 4 | 0.755862 | 84.67 | 13.74 | 1855.32 | 59.77 | 100.47 | 11.00 | 35.44 | 2040.00 | 57.87 |
| 5 | 0.759718 | 84.24 | 13.79 | 1856.05 | 59.77 | 100.49 | 13.50 | 35.71 | 2040.00 | 58.88 |
| 6 | 0.757733 | 84.46 | 13.56 | 1856.07 | 59.77 | 100.47 | 11.27 | 35.73 | 2040.00 | 58.87 |
| 7 | 0.761590 | 84.03 | 15.89 | 1856.19 | 59.77 | 100.51 | 13.56 | 35.84 | 2040.00 | 60.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.83/12.44/12.97.
Separate latency-pass duration: 0.76 s; not used for throughput.

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

Status: **complete**. Load 0.13 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 78.63 ± 0.52 | 17.01 | 12.41/15.20/15.44 | 1842.52 | 59.02/88.00 | 64/320 | PASS |
| raw-stream | 78.76 ± 0.97 | 16.79 | 12.44/15.29/16.02 | 1842.52 | 59.02/88.00 | 64/320 | PASS |
| native-non-stream | 72.63 ± 1.28 | 18.52 | 13.60/16.59/17.47 | 1842.52 | 58.99/88.00 | 64/320 | PASS |
| native-stream | 72.63 ± 1.39 | 15.86 | 13.47/13.96/18.43 | 1789.44 | 58.99/88.00 | 64/320 | PASS |
| managed-non-stream | 76.28 ± 0.70 | 14.86 | 13.03/15.87/16.26 | 1788.61 | 58.99/88.00 | 64/320 | PASS |
| managed-stream | 76.51 ± 0.48 | 14.85 | 13.18/14.11/15.61 | 1788.66 | 58.99/88.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.814702 | 78.56 | 14.57 | 1840.62 | 59.02 | 100.43 | 12.35 | 36.06 | 2040.00 | 61.00 |
| 2 | 0.808793 | 79.13 | 16.80 | 1841.86 | 59.02 | 100.46 | 13.41 | 36.27 | 2040.00 | 62.47 |
| 3 | 0.821448 | 77.91 | 16.91 | 1841.86 | 59.02 | 100.48 | 11.76 | 36.07 | 2040.00 | 62.71 |
| 4 | 0.810982 | 78.92 | 17.02 | 1842.30 | 59.02 | 100.48 | 12.82 | 36.99 | 2040.00 | 63.76 |
| 5 | 0.814340 | 78.59 | 17.08 | 1842.33 | 59.02 | 100.74 | 11.53 | 36.48 | 2040.00 | 63.76 |
| 6 | 0.804244 | 79.58 | 17.32 | 1842.52 | 59.02 | 100.46 | 13.75 | 36.86 | 2040.00 | 64.75 |
| 7 | 0.813977 | 78.63 | 17.01 | 1842.52 | 59.02 | 100.48 | 11.65 | 36.66 | 2040.00 | 64.82 |

Ready result delivery-interval P50/P95/P99 ms: 12.41/15.20/15.44.
Separate latency-pass duration: 0.81 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.5996588126038205, 'inference': 9.551775796808215, 'postprocess': 1.2124982342811563}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.817615 | 78.28 | 16.79 | 1840.62 | 59.02 | 100.50 | 13.18 | 36.51 | 2040.00 | 61.47 |
| 2 | 0.819630 | 78.08 | 14.00 | 1840.98 | 59.02 | 100.42 | 12.82 | 36.51 | 2040.00 | 62.00 |
| 3 | 0.811350 | 78.88 | 16.94 | 1841.86 | 59.02 | 100.46 | 13.47 | 36.40 | 2040.00 | 62.82 |
| 4 | 0.834214 | 76.72 | 14.29 | 1840.54 | 59.02 | 100.49 | 11.59 | 36.56 | 2040.00 | 63.18 |
| 5 | 0.803282 | 79.67 | 16.80 | 1842.33 | 59.02 | 100.45 | 13.19 | 37.11 | 2040.00 | 63.81 |
| 6 | 0.807102 | 79.30 | 13.90 | 1839.88 | 59.02 | 100.45 | 13.06 | 37.09 | 2040.00 | 64.56 |
| 7 | 0.812624 | 78.76 | 16.93 | 1842.52 | 59.02 | 100.45 | 13.65 | 36.98 | 2040.00 | 64.88 |

Ready result delivery-interval P50/P95/P99 ms: 12.44/15.29/15.99.
Separate latency-pass duration: 0.82 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6044868593448882, 'inference': 9.516483312552282, 'postprocess': 1.263628515516757}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

8.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.890279 | 71.89 | 18.52 | 1840.62 | 58.99 | 100.48 | 12.83 | 35.77 | 2040.00 | 61.61 |
| 2 | 0.872421 | 73.36 | 15.75 | 1788.65 | 58.99 | 100.42 | 13.11 | 36.40 | 2040.00 | 62.00 |
| 3 | 0.914629 | 69.97 | 18.53 | 1841.86 | 58.99 | 100.51 | 11.89 | 36.41 | 2040.00 | 62.79 |
| 4 | 0.872064 | 73.39 | 15.70 | 1789.25 | 58.99 | 100.44 | 12.00 | 36.64 | 2040.00 | 63.00 |
| 5 | 0.881188 | 72.63 | 18.68 | 1842.33 | 58.99 | 100.47 | 12.83 | 36.27 | 2040.00 | 63.83 |
| 6 | 0.870634 | 73.51 | 15.47 | 1789.44 | 58.99 | 100.42 | 12.67 | 36.95 | 2040.00 | 64.00 |
| 7 | 0.893338 | 71.64 | 18.87 | 1842.52 | 58.99 | 100.51 | 11.22 | 36.92 | 2040.00 | 64.83 |

Ready result delivery-interval P50/P95/P99 ms: 13.60/16.59/17.46.
Separate latency-pass duration: 0.89 s; not used for throughput.

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

7.50% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.886408 | 72.20 | 15.44 | 1786.69 | 58.99 | 100.47 | 12.83 | 35.98 | 2040.00 | 61.83 |
| 2 | 0.869480 | 73.61 | 15.87 | 1788.57 | 58.99 | 100.45 | 12.94 | 36.10 | 2040.00 | 62.00 |
| 3 | 0.881148 | 72.63 | 15.54 | 1787.77 | 58.99 | 100.44 | 12.67 | 36.49 | 2040.00 | 62.83 |
| 4 | 0.869237 | 73.63 | 15.95 | 1789.25 | 58.99 | 100.43 | 12.83 | 36.76 | 2040.00 | 63.00 |
| 5 | 0.893262 | 71.65 | 15.46 | 1788.16 | 58.99 | 100.48 | 12.67 | 36.81 | 2040.00 | 63.83 |
| 6 | 0.862757 | 74.18 | 15.86 | 1789.44 | 58.99 | 100.39 | 13.33 | 36.93 | 2040.00 | 64.00 |
| 7 | 0.912069 | 70.17 | 17.52 | 1788.33 | 58.99 | 100.49 | 12.11 | 36.34 | 2040.00 | 64.89 |

Ready result delivery-interval P50/P95/P99 ms: 13.47/14.00/18.35.
Separate latency-pass duration: 0.88 s; not used for throughput.

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

3.14% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.839149 | 76.27 | 14.83 | 1787.07 | 58.99 | 100.51 | 12.35 | 36.27 | 2040.00 | 61.88 |
| 2 | 0.834180 | 76.72 | 14.86 | 1787.50 | 58.99 | 100.40 | 13.00 | 36.45 | 2040.00 | 61.94 |
| 3 | 0.839433 | 76.24 | 14.87 | 1788.15 | 58.99 | 100.44 | 12.53 | 36.46 | 2040.00 | 62.88 |
| 4 | 0.838982 | 76.28 | 14.97 | 1788.36 | 58.99 | 100.46 | 12.65 | 36.60 | 2040.00 | 63.00 |
| 5 | 0.834886 | 76.66 | 14.74 | 1788.44 | 58.99 | 100.47 | 12.29 | 36.80 | 2040.00 | 63.88 |
| 6 | 0.834982 | 76.65 | 14.80 | 1788.55 | 58.99 | 100.44 | 13.41 | 37.02 | 2040.00 | 64.00 |
| 7 | 0.856607 | 74.71 | 15.10 | 1788.61 | 58.99 | 100.47 | 12.41 | 36.92 | 2040.00 | 64.88 |

Ready result delivery-interval P50/P95/P99 ms: 13.03/15.87/16.26.
Separate latency-pass duration: 0.86 s; not used for throughput.

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

3.14% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.832147 | 76.91 | 14.78 | 1787.26 | 58.99 | 100.47 | 13.35 | 36.52 | 2040.00 | 61.82 |
| 2 | 0.837634 | 76.41 | 14.90 | 1787.34 | 58.99 | 100.45 | 12.06 | 36.01 | 2040.00 | 61.88 |
| 3 | 0.839824 | 76.21 | 14.85 | 1788.32 | 58.99 | 100.41 | 13.18 | 36.71 | 2040.00 | 62.88 |
| 4 | 0.836444 | 76.51 | 14.85 | 1788.36 | 58.99 | 100.45 | 11.59 | 36.16 | 2040.00 | 62.88 |
| 5 | 0.829092 | 77.19 | 14.81 | 1788.52 | 58.99 | 100.42 | 13.12 | 37.06 | 2040.00 | 64.00 |
| 6 | 0.830016 | 77.11 | 14.89 | 1788.54 | 58.99 | 100.41 | 11.59 | 36.44 | 2040.00 | 64.00 |
| 7 | 0.843207 | 75.90 | 14.97 | 1788.66 | 58.99 | 100.51 | 12.65 | 37.26 | 2040.00 | 65.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.18/14.11/15.62.
Separate latency-pass duration: 0.86 s; not used for throughput.

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

Status: **complete**. Load 5.79 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 78.80 ± 0.42 | 16.95 | 12.50/15.53/15.65 | 2098.16 | 115.04/132.00 | 64/384 | PASS |
| raw-stream | 78.85 ± 0.73 | 16.91 | 12.51/14.59/15.90 | 2098.16 | 115.04/132.00 | 64/384 | PASS |
| native-non-stream | 73.99 ± 1.27 | 18.38 | 13.31/14.68/16.35 | 2098.16 | 115.01/132.00 | 64/384 | PASS |
| native-stream | 73.91 ± 0.82 | 15.51 | 13.49/14.82/15.38 | 2045.11 | 115.01/132.00 | 64/384 | PASS |
| managed-non-stream | 77.55 ± 0.60 | 14.58 | 12.74/13.04/13.97 | 2044.27 | 115.01/132.00 | 64/384 | PASS |
| managed-stream | 77.60 ± 0.60 | 14.68 | 12.75/13.03/13.76 | 2044.32 | 115.01/132.00 | 64/384 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.808949 | 79.12 | 14.49 | 2096.24 | 115.04 | 100.46 | 20.94 | 43.72 | 2040.00 | 67.76 |
| 2 | 0.813865 | 78.64 | 16.95 | 2097.54 | 115.04 | 100.50 | 17.35 | 45.25 | 2040.00 | 68.76 |
| 3 | 0.812459 | 78.77 | 17.05 | 2097.56 | 115.04 | 100.47 | 17.35 | 44.05 | 2040.00 | 68.76 |
| 4 | 0.812142 | 78.80 | 16.92 | 2098.01 | 115.04 | 100.48 | 20.29 | 44.92 | 2040.00 | 70.53 |
| 5 | 0.820775 | 77.98 | 16.98 | 2098.01 | 115.04 | 100.85 | 17.41 | 43.81 | 2040.00 | 70.53 |
| 6 | 0.808920 | 79.12 | 16.81 | 2098.16 | 115.04 | 100.46 | 20.65 | 44.87 | 2040.00 | 71.76 |
| 7 | 0.808173 | 79.19 | 17.00 | 2098.16 | 115.04 | 100.48 | 17.82 | 43.68 | 2040.00 | 71.76 |

Ready result delivery-interval P50/P95/P99 ms: 12.50/15.53/15.65.
Separate latency-pass duration: 0.83 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6097477657078798, 'inference': 9.417385765743802, 'postprocess': 1.212849156303264}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.12% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.815232 | 78.51 | 16.91 | 2096.24 | 115.04 | 100.49 | 19.00 | 44.81 | 2040.00 | 67.82 |
| 2 | 0.798935 | 80.11 | 13.83 | 2094.91 | 115.04 | 100.44 | 19.81 | 45.17 | 2040.00 | 68.88 |
| 3 | 0.811457 | 78.87 | 17.03 | 2097.56 | 115.04 | 100.47 | 18.71 | 44.27 | 2040.00 | 69.41 |
| 4 | 0.811713 | 78.85 | 13.88 | 2098.01 | 115.04 | 100.51 | 18.29 | 44.82 | 2040.00 | 70.24 |
| 5 | 0.814443 | 78.58 | 17.49 | 2098.02 | 115.04 | 100.51 | 20.18 | 45.69 | 2040.00 | 70.76 |
| 6 | 0.802613 | 79.74 | 14.05 | 2095.52 | 115.04 | 100.45 | 19.12 | 45.81 | 2040.00 | 71.62 |
| 7 | 0.820388 | 78.01 | 16.93 | 2098.16 | 115.04 | 100.52 | 20.24 | 44.75 | 2040.00 | 71.76 |

Ready result delivery-interval P50/P95/P99 ms: 12.51/14.57/15.87.
Separate latency-pass duration: 0.82 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6146951874086426, 'inference': 9.433944546771045, 'postprocess': 1.2662418124591568}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

6.94% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.874090 | 73.22 | 18.69 | 2096.24 | 115.01 | 100.49 | 16.00 | 44.44 | 2040.00 | 67.78 |
| 2 | 0.891092 | 71.82 | 15.55 | 2044.01 | 115.01 | 100.48 | 16.11 | 44.02 | 2040.00 | 68.78 |
| 3 | 0.860480 | 74.38 | 18.38 | 2097.56 | 115.01 | 100.47 | 16.89 | 44.81 | 2040.00 | 69.17 |
| 4 | 0.893537 | 71.63 | 15.25 | 2044.69 | 115.01 | 100.57 | 19.22 | 44.19 | 2040.00 | 69.89 |
| 5 | 0.863910 | 74.08 | 18.52 | 2098.02 | 115.01 | 100.46 | 18.78 | 44.69 | 2040.00 | 70.67 |
| 6 | 0.865029 | 73.99 | 16.07 | 2045.11 | 115.01 | 100.50 | 18.56 | 45.10 | 2040.00 | 71.56 |
| 7 | 0.854350 | 74.91 | 18.65 | 2098.16 | 115.01 | 100.49 | 18.59 | 45.69 | 2040.00 | 71.88 |

Ready result delivery-interval P50/P95/P99 ms: 13.31/14.69/16.34.
Separate latency-pass duration: 0.86 s; not used for throughput.

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

7.04% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.876642 | 73.01 | 15.21 | 2041.98 | 115.01 | 100.46 | 16.83 | 44.68 | 2040.00 | 67.78 |
| 2 | 0.868940 | 73.65 | 15.97 | 2043.92 | 115.01 | 100.44 | 16.56 | 43.56 | 2040.00 | 68.78 |
| 3 | 0.856846 | 74.69 | 15.27 | 2043.15 | 115.01 | 100.43 | 16.53 | 44.26 | 2040.00 | 69.76 |
| 4 | 0.858067 | 74.59 | 15.71 | 2044.69 | 115.01 | 100.48 | 19.65 | 44.78 | 2040.00 | 69.88 |
| 5 | 0.883871 | 72.41 | 15.51 | 2043.85 | 115.01 | 100.47 | 18.72 | 43.80 | 2040.00 | 70.83 |
| 6 | 0.865905 | 73.91 | 15.96 | 2045.11 | 115.01 | 100.46 | 19.33 | 45.33 | 2040.00 | 71.11 |
| 7 | 0.865742 | 73.93 | 14.94 | 2043.99 | 115.01 | 100.46 | 18.11 | 45.48 | 2040.00 | 71.89 |

Ready result delivery-interval P50/P95/P99 ms: 13.49/14.82/15.33.
Separate latency-pass duration: 0.87 s; not used for throughput.

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

2.02% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.825299 | 77.55 | 14.55 | 2042.09 | 115.01 | 100.49 | 19.88 | 44.41 | 2040.00 | 67.88 |
| 2 | 0.824318 | 77.64 | 14.58 | 2042.54 | 115.01 | 100.54 | 17.53 | 45.17 | 2040.00 | 68.65 |
| 3 | 0.816271 | 78.41 | 14.54 | 2043.22 | 115.01 | 100.44 | 17.47 | 43.87 | 2040.00 | 69.76 |
| 4 | 0.833251 | 76.81 | 14.57 | 2043.52 | 115.01 | 100.45 | 19.76 | 45.78 | 2040.00 | 69.76 |
| 5 | 0.823289 | 77.74 | 14.69 | 2044.12 | 115.01 | 100.52 | 19.29 | 45.00 | 2040.00 | 70.76 |
| 6 | 0.833599 | 76.78 | 14.77 | 2044.23 | 115.01 | 100.44 | 19.06 | 45.61 | 2040.00 | 71.00 |
| 7 | 0.832192 | 76.91 | 14.69 | 2044.27 | 115.01 | 100.43 | 19.12 | 45.26 | 2040.00 | 72.12 |

Ready result delivery-interval P50/P95/P99 ms: 12.74/13.04/13.97.
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


### managed-stream

1.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.820492 | 78.00 | 14.51 | 2042.29 | 115.01 | 100.48 | 17.59 | 44.94 | 2040.00 | 68.06 |
| 2 | 0.824755 | 77.60 | 14.65 | 2042.38 | 115.01 | 100.52 | 18.71 | 43.63 | 2040.00 | 68.35 |
| 3 | 0.819177 | 78.13 | 14.48 | 2043.43 | 115.01 | 100.52 | 19.76 | 45.17 | 2040.00 | 69.76 |
| 4 | 0.819672 | 78.08 | 14.71 | 2043.52 | 115.01 | 100.49 | 17.65 | 44.13 | 2040.00 | 69.71 |
| 5 | 0.831978 | 76.93 | 14.68 | 2044.18 | 115.01 | 100.48 | 19.94 | 44.65 | 2040.00 | 70.76 |
| 6 | 0.834677 | 76.68 | 14.69 | 2044.23 | 115.01 | 100.50 | 17.18 | 44.41 | 2040.00 | 70.82 |
| 7 | 0.830423 | 77.07 | 14.79 | 2044.32 | 115.01 | 100.50 | 18.71 | 45.39 | 2040.00 | 72.29 |

Ready result delivery-interval P50/P95/P99 ms: 12.75/13.03/13.75.
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

## Detail — grounding-dino, images, batch 1

Status: **complete**. Load 12.43 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 4.89 ± 0.02 | 203.64 | 203.75/206.17/206.82 | 2896.80 | 1366.93/1424.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 4.84 ± 0.01 | 206.02 | 206.11/207.10/207.19 | 2896.80 | 1366.92/1424.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 4.85 ± 0.01 | 206.04 | 206.91/209.31/210.10 | 2902.27 | 1366.92/1424.00 | 8/40 | PASS |
| managed-stream | 4.85 ± 0.02 | 203.54 | 206.62/209.92/210.60 | 2903.02 | 1366.92/1424.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.623568 | 4.93 | 201.61 | 2875.57 | 1366.93 | 100.41 | 76.85 | 49.20 | 1436.36 | 76.36 |
| 2 | 1.634756 | 4.89 | 203.40 | 2896.80 | 1366.93 | 100.42 | 79.64 | 55.03 | 1422.27 | 79.18 |
| 3 | 1.636791 | 4.89 | 205.17 | 2896.80 | 1366.93 | 100.44 | 57.58 | 52.42 | 1571.82 | 79.03 |
| 4 | 1.633227 | 4.90 | 203.64 | 2896.80 | 1366.93 | 100.48 | 38.52 | 68.35 | 1581.36 | 79.79 |
| 5 | 1.634991 | 4.89 | 204.01 | 2896.80 | 1366.93 | 100.49 | 49.70 | 50.41 | 1438.64 | 80.00 |
| 6 | 1.640019 | 4.88 | 203.81 | 2895.23 | 1366.93 | 100.46 | 79.82 | 66.44 | 1611.36 | 77.48 |
| 7 | 1.624492 | 4.92 | 202.17 | 2867.93 | 1366.93 | 100.42 | 80.36 | 54.43 | 1493.18 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 203.75/206.17/206.82.
Separate latency-pass duration: 1.63 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

1.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.642483 | 4.87 | 206.02 | 2895.32 | 1366.92 | 100.41 | 69.82 | 49.50 | 1509.55 | 76.67 |
| 2 | 1.649788 | 4.85 | 202.27 | 2896.80 | 1366.92 | 100.44 | 78.97 | 65.43 | 1583.18 | 77.97 |
| 3 | 1.657036 | 4.83 | 206.71 | 2895.88 | 1366.92 | 100.49 | 40.85 | 64.25 | 1546.36 | 78.85 |
| 4 | 1.652813 | 4.84 | 202.30 | 2896.80 | 1366.92 | 100.48 | 47.27 | 60.06 | 1530.91 | 79.42 |
| 5 | 1.651321 | 4.84 | 207.42 | 2896.80 | 1366.92 | 100.49 | 70.15 | 57.00 | 1544.09 | 78.88 |
| 6 | 1.652101 | 4.84 | 205.54 | 2896.80 | 1366.92 | 100.48 | 69.45 | 55.32 | 1547.73 | 77.76 |
| 7 | 1.648269 | 4.85 | 207.17 | 2896.40 | 1366.92 | 100.46 | 70.67 | 49.73 | 1444.09 | 77.88 |

Ready result delivery-interval P50/P95/P99 ms: 206.11/207.11/207.19.
Separate latency-pass duration: 1.65 s; not used for throughput.

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

1.27% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.646503 | 4.86 | 206.22 | 2901.74 | 1366.92 | 100.44 | 54.76 | 54.84 | 1580.45 | 76.73 |
| 2 | 1.646068 | 4.86 | 202.65 | 2902.27 | 1366.92 | 100.43 | 69.58 | 55.32 | 1576.36 | 78.03 |
| 3 | 1.650330 | 4.85 | 206.04 | 2900.84 | 1366.92 | 100.47 | 44.42 | 56.37 | 1476.82 | 79.73 |
| 4 | 1.653909 | 4.84 | 207.98 | 2901.49 | 1366.92 | 100.47 | 68.00 | 50.94 | 1438.64 | 79.94 |
| 5 | 1.656551 | 4.83 | 209.12 | 2899.67 | 1366.92 | 100.46 | 79.79 | 65.46 | 1586.82 | 78.67 |
| 6 | 1.655556 | 4.83 | 205.40 | 2901.49 | 1366.92 | 100.46 | 42.94 | 60.43 | 1520.45 | 78.12 |
| 7 | 1.650206 | 4.85 | 204.65 | 2900.84 | 1366.92 | 100.46 | 48.48 | 56.53 | 1465.00 | 76.97 |

Ready result delivery-interval P50/P95/P99 ms: 206.91/209.31/210.10.
Separate latency-pass duration: 1.65 s; not used for throughput.

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

0.81% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.647699 | 4.86 | 203.42 | 2901.48 | 1366.92 | 100.46 | 40.39 | 64.78 | 1567.73 | 76.94 |
| 2 | 1.641171 | 4.87 | 202.97 | 2903.02 | 1366.92 | 100.42 | 50.42 | 52.23 | 1486.36 | 77.88 |
| 3 | 1.649626 | 4.85 | 203.54 | 2902.66 | 1366.92 | 100.47 | 63.58 | 53.49 | 1479.09 | 79.58 |
| 4 | 1.651339 | 4.84 | 205.71 | 2900.84 | 1366.92 | 100.47 | 80.30 | 64.42 | 1572.73 | 79.64 |
| 5 | 1.648160 | 4.85 | 202.06 | 2902.66 | 1366.92 | 100.49 | 73.45 | 51.39 | 1453.18 | 78.94 |
| 6 | 1.653043 | 4.84 | 205.18 | 2900.84 | 1366.92 | 100.47 | 43.42 | 60.69 | 1559.55 | 77.73 |
| 7 | 1.660648 | 4.82 | 204.81 | 2901.49 | 1366.92 | 100.48 | 39.18 | 64.75 | 1562.73 | 77.03 |

Ready result delivery-interval P50/P95/P99 ms: 206.63/209.92/210.60.
Separate latency-pass duration: 1.65 s; not used for throughput.

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

Status: **complete**. Load 12.64 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 3.73 ± 0.02 | 266.36 | 266.81/270.03/270.81 | 2909.54 | 1600.91/1662.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 3.70 ± 0.02 | 269.97 | 269.59/270.40/270.50 | 2909.54 | 1600.90/1662.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 3.70 ± 0.02 | 271.02 | 270.28/271.15/271.17 | 2918.92 | 1600.90/1662.00 | 8/40 | PASS |
| managed-stream | 3.68 ± 0.01 | 278.13 | 271.02/274.80/275.60 | 2918.92 | 1600.90/1662.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.157673 | 3.71 | 277.38 | 2909.53 | 1600.91 | 100.51 | 74.84 | 62.76 | 1374.77 | 77.40 |
| 2 | 2.159797 | 3.70 | 266.36 | 2909.54 | 1600.91 | 100.43 | 74.33 | 62.52 | 1353.14 | 80.28 |
| 3 | 2.146404 | 3.73 | 267.50 | 2909.54 | 1600.91 | 100.48 | 74.19 | 64.06 | 1353.84 | 80.40 |
| 4 | 2.147532 | 3.73 | 268.96 | 2909.54 | 1600.91 | 100.44 | 73.95 | 64.07 | 1367.79 | 77.58 |
| 5 | 2.141047 | 3.74 | 264.92 | 2909.54 | 1600.91 | 100.42 | 73.72 | 63.37 | 1355.93 | 77.40 |
| 6 | 2.137003 | 3.74 | 265.92 | 2909.54 | 1600.91 | 100.43 | 69.49 | 61.07 | 1353.49 | 75.67 |
| 7 | 2.131165 | 3.75 | 266.20 | 2909.54 | 1600.91 | 100.45 | 72.09 | 62.15 | 1374.07 | 75.33 |

Ready result delivery-interval P50/P95/P99 ms: 266.81/270.02/270.80.
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

0.66% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.161287 | 3.70 | 269.97 | 2909.53 | 1600.90 | 100.45 | 75.07 | 62.49 | 1381.05 | 78.02 |
| 2 | 2.170690 | 3.69 | 275.06 | 2909.54 | 1600.90 | 100.45 | 73.77 | 62.82 | 1389.55 | 80.50 |
| 3 | 2.154730 | 3.71 | 269.06 | 2909.54 | 1600.90 | 100.39 | 73.70 | 62.90 | 1373.72 | 79.91 |
| 4 | 2.169197 | 3.69 | 276.00 | 2909.54 | 1600.90 | 100.44 | 73.41 | 62.61 | 1393.64 | 78.00 |
| 5 | 2.165168 | 3.69 | 269.74 | 2909.54 | 1600.90 | 100.44 | 73.27 | 61.80 | 1403.18 | 77.14 |
| 6 | 2.163734 | 3.70 | 278.07 | 2909.54 | 1600.90 | 100.48 | 69.21 | 62.27 | 1383.84 | 75.91 |
| 7 | 2.145176 | 3.73 | 266.42 | 2909.54 | 1600.90 | 100.47 | 73.12 | 64.30 | 1400.58 | 75.33 |

Ready result delivery-interval P50/P95/P99 ms: 269.58/270.40/270.50.
Separate latency-pass duration: 2.15 s; not used for throughput.

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
| 1 | 2.168154 | 3.69 | 269.02 | 2918.90 | 1600.90 | 100.45 | 73.09 | 63.48 | 1372.16 | 78.66 |
| 2 | 2.179669 | 3.67 | 276.40 | 2918.91 | 1600.90 | 100.44 | 71.55 | 62.41 | 1358.52 | 80.14 |
| 3 | 2.160319 | 3.70 | 270.36 | 2918.92 | 1600.90 | 100.44 | 72.74 | 63.54 | 1367.09 | 79.40 |
| 4 | 2.180212 | 3.67 | 279.65 | 2918.92 | 1600.90 | 100.44 | 70.09 | 61.96 | 1362.27 | 78.25 |
| 5 | 2.160823 | 3.70 | 271.02 | 2918.92 | 1600.90 | 100.41 | 71.63 | 62.99 | 1383.14 | 76.91 |
| 6 | 2.163845 | 3.70 | 277.10 | 2918.92 | 1600.90 | 100.44 | 71.14 | 61.73 | 1405.47 | 76.05 |
| 7 | 2.157919 | 3.71 | 268.70 | 2918.92 | 1600.90 | 100.53 | 70.53 | 62.19 | 1385.58 | 75.33 |

Ready result delivery-interval P50/P95/P99 ms: 270.28/271.15/271.17.
Separate latency-pass duration: 2.16 s; not used for throughput.

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

1.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.180281 | 3.67 | 279.00 | 2918.90 | 1600.90 | 100.46 | 70.73 | 62.33 | 1382.05 | 79.27 |
| 2 | 2.188928 | 3.65 | 277.20 | 2918.91 | 1600.90 | 100.45 | 70.41 | 62.83 | 1380.68 | 79.70 |
| 3 | 2.167707 | 3.69 | 279.97 | 2918.92 | 1600.90 | 100.47 | 69.91 | 61.86 | 1390.91 | 78.95 |
| 4 | 2.182304 | 3.67 | 277.24 | 2918.92 | 1600.90 | 100.43 | 71.57 | 62.67 | 1358.18 | 78.55 |
| 5 | 2.169455 | 3.69 | 278.32 | 2918.92 | 1600.90 | 100.42 | 69.82 | 62.02 | 1395.00 | 76.55 |
| 6 | 2.171898 | 3.68 | 277.39 | 2918.92 | 1600.90 | 100.44 | 72.27 | 62.06 | 1392.95 | 76.34 |
| 7 | 2.166489 | 3.69 | 278.13 | 2918.92 | 1600.90 | 100.42 | 69.41 | 62.23 | 1411.36 | 75.18 |

Ready result delivery-interval P50/P95/P99 ms: 271.03/274.77/275.55.
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

## Detail — faster-rcnn, images, batch 1

Status: **complete**. Load 1.65 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 44.87 ± 0.55 | 23.87 | 22.37/23.64/25.74 | 1900.94 | 202.88/324.00 | 32/256 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 42.66 ± 0.48 | 25.41 | 23.43/26.88/28.51 | 1900.94 | 202.85/324.00 | 32/256 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 41.00 ± 0.21 | 26.06 | 24.80/25.99/28.57 | 1938.42 | 202.85/324.00 | 32/256 | PASS |
| managed-stream | 41.22 ± 0.38 | 28.26 | 24.17/24.97/27.34 | 1938.34 | 202.85/324.00 | 32/256 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.710487 | 45.04 | 26.36 | 1899.95 | 202.88 | 124.54 | 35.20 | 53.47 | 2040.00 | 74.07 |
| 2 | 0.697719 | 45.86 | 23.53 | 1900.67 | 202.88 | 124.60 | 30.00 | 54.43 | 2040.00 | 75.50 |
| 3 | 0.717414 | 44.60 | 23.63 | 1900.68 | 202.88 | 124.39 | 33.73 | 52.78 | 2040.00 | 75.60 |
| 4 | 0.724386 | 44.18 | 23.59 | 1900.86 | 202.88 | 124.02 | 29.00 | 54.05 | 2040.00 | 76.73 |
| 5 | 0.721244 | 44.37 | 24.76 | 1900.86 | 202.88 | 124.22 | 33.40 | 52.89 | 2040.00 | 77.13 |
| 6 | 0.713178 | 44.87 | 23.87 | 1900.94 | 202.88 | 124.34 | 29.07 | 54.31 | 2040.00 | 78.47 |
| 7 | 0.713148 | 44.87 | 24.03 | 1900.94 | 202.88 | 124.22 | 31.20 | 54.39 | 2040.00 | 78.47 |

Ready result delivery-interval P50/P95/P99 ms: 22.37/23.63/25.73.
Separate latency-pass duration: 0.72 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

5.53% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.756648 | 42.29 | 25.01 | 1899.95 | 202.85 | 123.19 | 29.12 | 53.14 | 2040.00 | 74.25 |
| 2 | 0.741558 | 43.15 | 27.34 | 1900.58 | 202.85 | 123.48 | 31.67 | 53.59 | 2040.00 | 75.27 |
| 3 | 0.743171 | 43.06 | 25.16 | 1900.69 | 202.85 | 123.30 | 32.67 | 53.77 | 2040.00 | 75.53 |
| 4 | 0.750179 | 42.66 | 27.81 | 1900.86 | 202.85 | 123.48 | 33.67 | 53.22 | 2040.00 | 76.47 |
| 5 | 0.744226 | 43.00 | 25.04 | 1900.86 | 202.85 | 123.27 | 33.73 | 55.29 | 2040.00 | 77.13 |
| 6 | 0.752622 | 42.52 | 27.69 | 1900.93 | 202.85 | 123.31 | 34.00 | 53.80 | 2040.00 | 77.87 |
| 7 | 0.765285 | 41.81 | 25.41 | 1900.94 | 202.85 | 123.23 | 33.62 | 55.60 | 2040.00 | 78.69 |

Ready result delivery-interval P50/P95/P99 ms: 23.43/26.89/28.52.
Separate latency-pass duration: 0.76 s; not used for throughput.

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

9.29% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.776627 | 41.20 | 25.95 | 1937.57 | 202.85 | 124.67 | 33.00 | 52.52 | 2040.00 | 74.12 |
| 2 | 0.782271 | 40.91 | 28.25 | 1936.79 | 202.85 | 125.49 | 29.06 | 53.64 | 2040.00 | 74.81 |
| 3 | 0.780401 | 41.00 | 25.65 | 1937.03 | 202.85 | 124.60 | 27.69 | 54.41 | 2040.00 | 76.12 |
| 4 | 0.776944 | 41.19 | 28.01 | 1938.34 | 202.85 | 124.76 | 27.88 | 52.54 | 2040.00 | 76.50 |
| 5 | 0.788275 | 40.59 | 26.06 | 1937.20 | 202.85 | 124.20 | 27.81 | 56.14 | 2040.00 | 77.06 |
| 6 | 0.779010 | 41.08 | 28.32 | 1938.42 | 202.85 | 124.72 | 28.00 | 53.29 | 2040.00 | 77.88 |
| 7 | 0.782251 | 40.91 | 25.89 | 1937.27 | 202.85 | 124.65 | 28.94 | 54.69 | 2040.00 | 78.69 |

Ready result delivery-interval P50/P95/P99 ms: 24.80/25.98/28.57.
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


### managed-stream

8.79% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.772904 | 41.40 | 28.05 | 1937.68 | 202.85 | 124.88 | 27.75 | 51.86 | 2040.00 | 74.50 |
| 2 | 0.775579 | 41.26 | 28.28 | 1937.84 | 202.85 | 124.79 | 31.69 | 51.28 | 2040.00 | 74.75 |
| 3 | 0.771418 | 41.48 | 27.93 | 1938.31 | 202.85 | 124.99 | 32.62 | 54.97 | 2040.00 | 75.94 |
| 4 | 0.776378 | 41.22 | 28.01 | 1938.34 | 202.85 | 125.17 | 29.50 | 52.91 | 2040.00 | 76.00 |
| 5 | 0.780407 | 41.00 | 28.53 | 1937.22 | 202.85 | 124.67 | 31.12 | 55.05 | 2040.00 | 77.31 |
| 6 | 0.779543 | 41.05 | 28.26 | 1937.24 | 202.85 | 124.76 | 30.50 | 53.21 | 2040.00 | 77.50 |
| 7 | 0.793193 | 40.34 | 30.02 | 1937.29 | 202.85 | 124.35 | 30.12 | 54.23 | 2040.00 | 78.69 |

Ready result delivery-interval P50/P95/P99 ms: 24.17/24.96/27.30.
Separate latency-pass duration: 0.78 s; not used for throughput.

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

Status: **complete**. Load 16.78 s. Device `cuda:0`, dtype `float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.66 ± 0.01 | 377.39 | 377.74/380.14/380.29 | 3167.76 | 146.85/252.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.64 ± 0.01 | 377.41 | 383.47/385.04/385.26 | 3167.77 | 153.40/252.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.65 ± 0.01 | 377.80 | 378.27/380.58/380.62 | 3176.89 | 154.25/252.00 | 8/40 | PASS |
| managed-stream | 2.64 ± 0.01 | 380.19 | 377.28/381.63/381.89 | 3176.93 | 150.29/252.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.018872 | 2.65 | 379.87 | 3167.20 | 145.90 | 101.78 | 2.40 | 35.31 | 2040.00 | 75.00 |
| 2 | 2.992224 | 2.67 | 373.34 | 3167.41 | 139.84 | 101.78 | 2.25 | 35.47 | 2040.00 | 76.00 |
| 3 | 3.030704 | 2.64 | 374.77 | 3167.42 | 139.68 | 101.78 | 2.17 | 35.42 | 2040.00 | 76.00 |
| 4 | 3.024371 | 2.65 | 379.57 | 3167.60 | 146.85 | 101.80 | 2.30 | 35.54 | 2040.00 | 76.00 |
| 5 | 2.989988 | 2.68 | 378.28 | 3167.61 | 143.93 | 101.76 | 2.20 | 35.67 | 2040.00 | 76.03 |
| 6 | 3.003444 | 2.66 | 374.78 | 3167.75 | 143.93 | 101.75 | 2.34 | 35.76 | 2040.00 | 77.00 |
| 7 | 3.003563 | 2.66 | 377.39 | 3167.76 | 139.42 | 101.74 | 2.22 | 35.92 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 377.74/380.14/380.29.
Separate latency-pass duration: 3.02 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.71% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.040313 | 2.63 | 380.67 | 3167.22 | 152.55 | 101.79 | 2.20 | 35.22 | 2040.00 | 75.00 |
| 2 | 3.004173 | 2.66 | 376.32 | 3167.11 | 153.40 | 101.77 | 2.33 | 35.49 | 2040.00 | 76.00 |
| 3 | 3.031980 | 2.64 | 377.41 | 3167.45 | 153.30 | 101.76 | 2.25 | 35.58 | 2040.00 | 76.00 |
| 4 | 3.020481 | 2.65 | 378.07 | 3167.29 | 152.29 | 101.80 | 2.18 | 35.55 | 2040.00 | 76.00 |
| 5 | 3.027294 | 2.64 | 377.35 | 3167.63 | 152.15 | 101.75 | 2.20 | 35.59 | 2040.00 | 76.63 |
| 6 | 3.047500 | 2.63 | 386.88 | 3167.44 | 153.04 | 101.81 | 2.33 | 35.80 | 2040.00 | 77.00 |
| 7 | 3.027012 | 2.64 | 376.96 | 3167.77 | 152.27 | 101.77 | 2.28 | 35.78 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 383.48/385.03/385.26.
Separate latency-pass duration: 3.06 s; not used for throughput.

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

0.59% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.023268 | 2.65 | 377.80 | 3176.34 | 148.64 | 101.88 | 2.15 | 35.29 | 2040.00 | 75.17 |
| 2 | 3.043316 | 2.63 | 375.88 | 3176.44 | 148.64 | 101.88 | 2.28 | 35.44 | 2040.00 | 76.00 |
| 3 | 3.006264 | 2.66 | 375.21 | 3176.58 | 154.25 | 101.89 | 2.32 | 35.57 | 2040.00 | 76.00 |
| 4 | 3.017227 | 2.65 | 377.82 | 3176.64 | 151.58 | 101.89 | 2.37 | 35.68 | 2040.00 | 76.00 |
| 5 | 3.035642 | 2.64 | 376.33 | 3176.73 | 148.64 | 101.98 | 2.23 | 35.60 | 2040.00 | 76.98 |
| 6 | 3.021097 | 2.65 | 379.07 | 3176.79 | 152.52 | 101.88 | 2.08 | 35.83 | 2040.00 | 77.00 |
| 7 | 3.040426 | 2.63 | 384.34 | 3176.89 | 146.07 | 101.87 | 2.23 | 35.72 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 378.26/380.58/380.62.
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


### managed-stream

0.65% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.050065 | 2.62 | 381.23 | 3176.38 | 142.60 | 101.91 | 2.28 | 35.39 | 2040.00 | 75.60 |
| 2 | 3.046321 | 2.63 | 380.37 | 3176.41 | 142.60 | 101.91 | 2.25 | 35.39 | 2040.00 | 75.95 |
| 3 | 3.028245 | 2.64 | 377.21 | 3176.61 | 143.95 | 101.91 | 2.33 | 35.53 | 2040.00 | 76.00 |
| 4 | 3.021556 | 2.65 | 380.19 | 3176.63 | 150.29 | 101.90 | 2.32 | 35.52 | 2040.00 | 76.00 |
| 5 | 3.026333 | 2.64 | 380.08 | 3176.77 | 142.60 | 101.91 | 2.33 | 35.72 | 2040.00 | 77.00 |
| 6 | 3.017201 | 2.65 | 377.95 | 3176.79 | 143.95 | 101.92 | 2.28 | 35.79 | 2040.00 | 77.00 |
| 7 | 3.023137 | 2.65 | 381.96 | 3176.93 | 142.60 | 101.90 | 2.18 | 35.72 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 377.28/381.60/381.85.
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

## Detail — locate-anything, images, batch 1

Status: **complete**. Load 19.51 s. Device `cuda:0`, dtype `torch.float16`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.47 ± 0.01 | 404.15 | 403.58/403.58/403.58 | 4093.07 | 7910.11/8364.00 | 1/4 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.46 ± 0.01 | 406.70 | 407.72/407.72/407.72 | 4093.07 | 7910.11/8364.00 | 1/4 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.44 ± 0.03 | 409.64 | 410.94/410.94/410.94 | 4093.07 | 7910.11/8364.00 | 1/4 | PASS |
| managed-stream | 2.46 ± 0.03 | 406.34 | 405.12/405.12/405.12 | 4093.07 | 7910.11/8364.00 | 1/4 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.406257 | 2.46 | 406.21 | 4093.07 | 7910.11 | 102.59 | 43.11 | 51.74 | 1780.00 | 78.11 |
| 2 | 0.406294 | 2.46 | 406.24 | 4093.07 | 7910.11 | 102.47 | 41.33 | 53.51 | 1790.00 | 79.33 |
| 3 | 0.404622 | 2.47 | 404.57 | 4093.07 | 7910.11 | 102.52 | 29.00 | 49.44 | 1711.67 | 79.78 |
| 4 | 0.404202 | 2.47 | 404.15 | 4093.07 | 7910.11 | 102.50 | 28.22 | 54.24 | 1770.00 | 80.11 |
| 5 | 0.403570 | 2.48 | 403.52 | 4093.07 | 7910.11 | 102.49 | 35.11 | 52.92 | 1816.67 | 80.44 |
| 6 | 0.402348 | 2.49 | 402.30 | 4093.07 | 7910.11 | 102.41 | 35.62 | 50.66 | 1745.62 | 79.50 |
| 7 | 0.403904 | 2.48 | 403.85 | 4093.07 | 7910.11 | 102.47 | 29.00 | 52.36 | 1695.00 | 79.78 |

Ready result delivery-interval P50/P95/P99 ms: 403.57/403.57/403.57.
Separate latency-pass duration: 0.40 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.408975 | 2.45 | 408.92 | 4093.07 | 7910.11 | 102.36 | 41.33 | 52.46 | 1800.00 | 78.33 |
| 2 | 0.405837 | 2.46 | 405.78 | 4093.07 | 7910.11 | 102.44 | 43.22 | 52.72 | 1770.00 | 79.11 |
| 3 | 0.410942 | 2.43 | 410.88 | 4093.07 | 7910.11 | 102.40 | 30.89 | 52.50 | 1738.33 | 79.78 |
| 4 | 0.408075 | 2.45 | 408.02 | 4093.07 | 7910.11 | 102.47 | 28.00 | 49.52 | 1691.67 | 80.67 |
| 5 | 0.404867 | 2.47 | 404.79 | 4093.07 | 7910.11 | 102.41 | 49.67 | 51.48 | 1738.33 | 80.33 |
| 6 | 0.406754 | 2.46 | 406.70 | 4093.07 | 7910.11 | 102.38 | 48.67 | 49.70 | 1681.67 | 80.00 |
| 7 | 0.404659 | 2.47 | 404.60 | 4093.07 | 7910.11 | 102.47 | 25.33 | 52.70 | 1773.33 | 79.11 |

Ready result delivery-interval P50/P95/P99 ms: 407.71/407.71/407.71.
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

1.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.420831 | 2.38 | 420.77 | 4093.07 | 7910.11 | 102.32 | 27.67 | 48.74 | 1685.00 | 79.00 |
| 2 | 0.405096 | 2.47 | 405.04 | 4093.07 | 7910.11 | 102.39 | 41.56 | 51.46 | 1705.00 | 79.44 |
| 3 | 0.410648 | 2.44 | 410.59 | 4093.07 | 7910.11 | 102.57 | 36.67 | 52.67 | 1776.67 | 79.56 |
| 4 | 0.409702 | 2.44 | 409.64 | 4093.07 | 7910.11 | 102.42 | 41.33 | 52.54 | 1796.67 | 80.11 |
| 5 | 0.416182 | 2.40 | 416.11 | 4093.07 | 7910.11 | 102.31 | 41.33 | 52.02 | 1773.33 | 80.11 |
| 6 | 0.407804 | 2.45 | 407.75 | 4093.07 | 7910.11 | 102.47 | 38.22 | 52.93 | 1790.00 | 79.89 |
| 7 | 0.404901 | 2.47 | 404.84 | 4093.07 | 7910.11 | 102.37 | 32.44 | 52.52 | 1715.00 | 79.00 |

Ready result delivery-interval P50/P95/P99 ms: 410.92/410.92/410.92.
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


### managed-stream

0.57% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.415187 | 2.41 | 415.07 | 4093.07 | 7910.11 | 102.40 | 32.56 | 52.66 | 1745.00 | 78.78 |
| 2 | 0.408628 | 2.45 | 408.53 | 4093.07 | 7910.11 | 102.43 | 34.44 | 53.15 | 1786.67 | 78.78 |
| 3 | 0.416480 | 2.40 | 416.37 | 4093.07 | 7910.11 | 102.39 | 41.56 | 49.84 | 1698.33 | 80.22 |
| 4 | 0.403859 | 2.48 | 403.77 | 4093.07 | 7910.11 | 102.39 | 50.22 | 53.04 | 1741.67 | 79.78 |
| 5 | 0.405107 | 2.47 | 405.01 | 4093.07 | 7910.11 | 102.45 | 29.00 | 49.97 | 1698.33 | 80.67 |
| 6 | 0.406436 | 2.46 | 406.34 | 4093.07 | 7910.11 | 102.43 | 23.78 | 52.47 | 1740.00 | 79.89 |
| 7 | 0.405027 | 2.47 | 404.92 | 4093.07 | 7910.11 | 102.40 | 49.78 | 48.90 | 1765.00 | 79.00 |

Ready result delivery-interval P50/P95/P99 ms: 405.07/405.07/405.07.
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
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 225.23 ± 18.13 | 29.20 | 17.26/17.92/18.68 | 2003.57 | 76.00/186.00 | 256/1280 | PASS |
| raw-stream | 225.40 ± 16.07 | 28.85 | 17.17/20.24/24.08 | 2003.57 | 76.00/186.00 | 256/1280 | PASS |
| native-non-stream | 196.82 ± 11.73 | 31.72 | 20.27/22.59/26.90 | 2003.57 | 75.87/186.00 | 256/1280 | PASS |
| native-stream | 196.08 ± 11.24 | 19.74 | 20.94/22.27/25.56 | 1789.11 | 75.87/186.00 | 256/1280 | PASS |
| managed-non-stream | 217.52 ± 2.87 | 19.19 | 18.71/20.43/22.09 | 1789.12 | 75.87/186.00 | 256/1280 | PASS |
| managed-stream | 218.60 ± 1.88 | 19.40 | 18.66/20.05/21.93 | 1789.12 | 75.87/186.00 | 256/1280 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.268392 | 201.83 | 18.39 | 1986.32 | 76.00 | 100.45 | 18.23 | 45.68 | 2040.00 | 74.92 |
| 2 | 1.353741 | 189.11 | 29.20 | 1991.09 | 76.00 | 100.50 | 15.37 | 42.78 | 2040.00 | 75.89 |
| 3 | 1.298332 | 197.18 | 29.53 | 1991.36 | 76.00 | 100.46 | 15.38 | 43.01 | 2040.00 | 75.85 |
| 4 | 1.108732 | 230.89 | 28.82 | 2003.35 | 76.00 | 100.46 | 18.82 | 47.17 | 2040.00 | 76.95 |
| 5 | 1.114320 | 229.74 | 29.39 | 2003.35 | 76.00 | 100.50 | 17.74 | 44.95 | 2040.00 | 76.96 |
| 6 | 1.111246 | 230.37 | 29.16 | 2003.57 | 76.00 | 100.49 | 18.52 | 47.76 | 2040.00 | 77.78 |
| 7 | 1.136600 | 225.23 | 29.62 | 2003.57 | 76.00 | 100.73 | 16.57 | 44.60 | 2040.00 | 77.39 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/17.41/18.00.
Separate latency-pass duration: 1.11 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.0887085898829696, 'inference': 1.788766214851023, 'postprocess': 1.0295508046667123}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.08% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.286536 | 198.98 | 28.98 | 1986.32 | 76.00 | 100.48 | 16.85 | 46.28 | 2040.00 | 75.00 |
| 2 | 1.276019 | 200.62 | 20.90 | 1998.00 | 76.00 | 100.51 | 17.77 | 44.77 | 2040.00 | 75.92 |
| 3 | 1.287511 | 198.83 | 29.31 | 1991.36 | 76.00 | 100.48 | 16.08 | 45.58 | 2040.00 | 75.88 |
| 4 | 1.122105 | 228.14 | 18.01 | 2003.35 | 76.00 | 100.52 | 19.57 | 46.04 | 2040.00 | 76.78 |
| 5 | 1.119555 | 228.66 | 28.85 | 2003.35 | 76.00 | 100.50 | 18.30 | 46.59 | 2040.00 | 77.30 |
| 6 | 1.092682 | 234.29 | 17.52 | 1996.54 | 76.00 | 100.51 | 19.64 | 48.70 | 2040.00 | 77.91 |
| 7 | 1.135738 | 225.40 | 30.67 | 2003.57 | 76.00 | 100.50 | 18.74 | 47.64 | 2040.00 | 77.35 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/17.58/20.22.
Separate latency-pass duration: 1.12 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1021927577914425, 'inference': 1.7904755312514453, 'postprocess': 1.0677923164195136}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

8.99% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.382393 | 185.19 | 31.72 | 1986.32 | 75.87 | 100.50 | 15.43 | 46.25 | 2040.00 | 75.00 |
| 2 | 1.397061 | 183.24 | 24.05 | 1779.40 | 75.87 | 100.48 | 15.07 | 45.18 | 2040.00 | 75.89 |
| 3 | 1.360759 | 188.13 | 32.34 | 1991.37 | 75.87 | 100.67 | 16.52 | 44.27 | 2040.00 | 75.89 |
| 4 | 1.300677 | 196.82 | 21.09 | 1785.38 | 75.87 | 100.48 | 16.54 | 47.62 | 2040.00 | 76.69 |
| 5 | 1.284909 | 199.24 | 31.97 | 2003.35 | 75.87 | 100.45 | 17.42 | 47.68 | 2040.00 | 77.27 |
| 6 | 1.179401 | 217.06 | 20.55 | 1785.60 | 75.87 | 100.52 | 17.83 | 47.74 | 2040.00 | 77.88 |
| 7 | 1.270956 | 201.42 | 32.43 | 2003.57 | 75.87 | 100.45 | 17.77 | 47.18 | 2040.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/21.29/23.05.
Separate latency-pass duration: 1.30 s; not used for throughput.

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

9.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.392189 | 183.88 | 23.35 | 1775.66 | 75.87 | 100.46 | 16.14 | 45.51 | 2040.00 | 75.00 |
| 2 | 1.389664 | 184.22 | 19.88 | 1782.51 | 75.87 | 100.47 | 16.00 | 46.29 | 2040.00 | 75.86 |
| 3 | 1.305561 | 196.08 | 19.74 | 1784.20 | 75.87 | 100.48 | 16.38 | 44.01 | 2040.00 | 75.96 |
| 4 | 1.329816 | 192.51 | 19.74 | 1788.89 | 75.87 | 100.50 | 16.30 | 44.96 | 2040.00 | 76.44 |
| 5 | 1.287645 | 198.81 | 19.57 | 1785.39 | 75.87 | 100.47 | 18.46 | 48.67 | 2040.00 | 77.65 |
| 6 | 1.179317 | 217.07 | 19.73 | 1789.11 | 75.87 | 100.49 | 20.21 | 48.08 | 2040.00 | 77.92 |
| 7 | 1.287498 | 198.84 | 19.48 | 1785.61 | 75.87 | 100.51 | 16.50 | 46.98 | 2040.00 | 76.92 |

Ready result delivery-interval P50/P95/P99 ms: 0.49/20.37/21.13.
Separate latency-pass duration: 1.32 s; not used for throughput.

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

2.53% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.180409 | 216.87 | 21.70 | 1780.23 | 75.87 | 100.48 | 19.21 | 44.98 | 2040.00 | 75.50 |
| 2 | 1.185718 | 215.90 | 19.66 | 1781.89 | 75.87 | 100.52 | 19.00 | 46.67 | 2040.00 | 75.88 |
| 3 | 1.176910 | 217.52 | 18.91 | 1787.78 | 75.87 | 100.51 | 18.50 | 46.30 | 2040.00 | 76.17 |
| 4 | 1.174619 | 217.94 | 19.19 | 1788.89 | 75.87 | 100.49 | 18.00 | 46.35 | 2040.00 | 76.42 |
| 5 | 1.163299 | 220.06 | 18.80 | 1788.90 | 75.87 | 100.49 | 17.87 | 46.71 | 2040.00 | 77.43 |
| 6 | 1.210509 | 211.48 | 19.27 | 1789.11 | 75.87 | 100.50 | 19.00 | 48.72 | 2040.00 | 77.79 |
| 7 | 1.165405 | 219.67 | 18.78 | 1789.12 | 75.87 | 100.46 | 16.78 | 47.33 | 2040.00 | 76.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/19.43/20.91.
Separate latency-pass duration: 1.18 s; not used for throughput.

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

2.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.171072 | 218.60 | 19.25 | 1780.89 | 75.87 | 100.44 | 17.67 | 47.16 | 2040.00 | 75.75 |
| 2 | 1.185828 | 215.88 | 19.40 | 1781.36 | 75.87 | 100.53 | 17.71 | 44.19 | 2040.00 | 75.79 |
| 3 | 1.173766 | 218.10 | 19.49 | 1788.43 | 75.87 | 100.52 | 17.12 | 46.68 | 2040.00 | 76.33 |
| 4 | 1.172740 | 218.29 | 19.44 | 1788.88 | 75.87 | 100.54 | 17.25 | 44.87 | 2040.00 | 76.33 |
| 5 | 1.162119 | 220.29 | 19.10 | 1788.90 | 75.87 | 100.44 | 18.96 | 46.45 | 2040.00 | 77.61 |
| 6 | 1.153692 | 221.90 | 19.23 | 1789.11 | 75.87 | 100.48 | 16.35 | 44.93 | 2040.00 | 77.65 |
| 7 | 1.168681 | 219.05 | 19.49 | 1789.12 | 75.87 | 100.53 | 19.08 | 47.65 | 2040.00 | 76.88 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/19.15/20.46.
Separate latency-pass duration: 1.18 s; not used for throughput.

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
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 200.78 ± 11.89 | 31.08 | 19.37/22.92/23.30 | 2055.15 | 112.51/184.00 | 256/1344 | PASS |
| raw-stream | 195.01 ± 12.09 | 30.95 | 19.22/21.58/26.30 | 2055.15 | 112.51/184.00 | 256/1344 | PASS |
| native-non-stream | 177.56 ± 6.43 | 34.05 | 22.77/25.77/30.07 | 2055.15 | 112.39/184.00 | 256/1344 | PASS |
| native-stream | 177.73 ± 4.02 | 21.92 | 22.92/24.25/24.55 | 1840.68 | 112.39/184.00 | 256/1344 | PASS |
| managed-non-stream | 193.52 ± 2.99 | 21.40 | 20.86/21.96/22.97 | 1840.70 | 112.39/184.00 | 256/1344 | PASS |
| managed-stream | 195.45 ± 2.02 | 21.43 | 20.82/22.20/22.95 | 1840.70 | 112.39/184.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.387042 | 184.57 | 20.43 | 2034.10 | 112.51 | 100.47 | 16.79 | 45.49 | 2040.00 | 75.93 |
| 2 | 1.410033 | 181.56 | 31.41 | 2042.63 | 112.51 | 100.48 | 16.29 | 45.28 | 2040.00 | 77.29 |
| 3 | 1.439191 | 177.88 | 31.08 | 2042.89 | 112.51 | 100.67 | 14.97 | 44.15 | 2040.00 | 77.14 |
| 4 | 1.266574 | 202.12 | 30.83 | 2054.95 | 112.51 | 100.49 | 18.27 | 46.74 | 2040.00 | 76.81 |
| 5 | 1.275006 | 200.78 | 31.30 | 2054.95 | 112.51 | 100.48 | 16.08 | 44.35 | 2040.00 | 76.88 |
| 6 | 1.245763 | 205.50 | 31.21 | 2055.15 | 112.51 | 100.48 | 18.08 | 46.80 | 2040.00 | 76.48 |
| 7 | 1.254806 | 204.02 | 30.94 | 2055.15 | 112.51 | 100.45 | 17.32 | 43.17 | 2040.00 | 76.32 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/19.78/23.08.
Separate latency-pass duration: 1.27 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.0893571679417846, 'inference': 2.270038078115988, 'postprocess': 1.0223182851376578}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.54% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.418252 | 180.50 | 31.20 | 2037.61 | 112.51 | 100.47 | 17.14 | 45.70 | 2040.00 | 76.21 |
| 2 | 1.402437 | 182.54 | 22.86 | 2040.16 | 112.51 | 100.47 | 15.54 | 45.17 | 2040.00 | 77.50 |
| 3 | 1.399217 | 182.96 | 30.95 | 2042.91 | 112.51 | 100.46 | 16.21 | 45.52 | 2040.00 | 77.07 |
| 4 | 1.247704 | 205.18 | 23.00 | 2047.92 | 112.51 | 100.48 | 17.72 | 47.25 | 2040.00 | 76.88 |
| 5 | 1.312775 | 195.01 | 32.46 | 2054.95 | 112.51 | 100.48 | 17.41 | 46.21 | 2040.00 | 76.85 |
| 6 | 1.230735 | 208.01 | 19.71 | 2051.63 | 112.51 | 100.45 | 18.00 | 46.07 | 2040.00 | 76.56 |
| 7 | 1.249022 | 204.96 | 31.03 | 2055.15 | 112.51 | 100.48 | 18.80 | 46.88 | 2040.00 | 76.28 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/19.43/22.15.
Separate latency-pass duration: 1.26 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1059277929561517, 'inference': 2.295235957021191, 'postprocess': 1.0788225312197142}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

12.66% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.549629 | 165.20 | 34.69 | 2037.61 | 112.39 | 100.46 | 15.65 | 44.54 | 2040.00 | 76.19 |
| 2 | 1.527183 | 167.63 | 25.87 | 1830.90 | 112.39 | 100.54 | 16.74 | 45.10 | 2040.00 | 77.00 |
| 3 | 1.479640 | 173.02 | 34.05 | 2042.91 | 112.39 | 100.49 | 16.27 | 46.19 | 2040.00 | 76.87 |
| 4 | 1.426966 | 179.40 | 25.77 | 1836.98 | 112.39 | 100.48 | 17.24 | 45.63 | 2040.00 | 76.69 |
| 5 | 1.441752 | 177.56 | 34.45 | 2054.95 | 112.39 | 100.44 | 17.34 | 46.48 | 2040.00 | 76.86 |
| 6 | 1.407994 | 181.82 | 25.66 | 1837.18 | 112.39 | 100.41 | 16.11 | 45.13 | 2040.00 | 76.54 |
| 7 | 1.424924 | 179.66 | 34.13 | 2055.15 | 112.39 | 100.47 | 17.66 | 46.05 | 2040.00 | 75.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/23.29/25.98.
Separate latency-pass duration: 1.45 s; not used for throughput.

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

11.80% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.480945 | 172.86 | 23.95 | 1827.60 | 112.39 | 100.44 | 15.53 | 44.43 | 2040.00 | 76.63 |
| 2 | 1.499609 | 170.71 | 25.99 | 1833.97 | 112.39 | 100.43 | 15.60 | 46.63 | 2040.00 | 77.07 |
| 3 | 1.440421 | 177.73 | 22.09 | 1835.53 | 112.39 | 100.44 | 16.83 | 45.14 | 2040.00 | 76.97 |
| 4 | 1.433130 | 178.63 | 21.77 | 1840.50 | 112.39 | 100.46 | 16.97 | 45.20 | 2040.00 | 76.72 |
| 5 | 1.425501 | 179.59 | 21.92 | 1836.98 | 112.39 | 100.46 | 17.79 | 45.49 | 2040.00 | 76.86 |
| 6 | 1.501047 | 170.55 | 21.88 | 1840.68 | 112.39 | 100.72 | 15.47 | 45.85 | 2040.00 | 76.60 |
| 7 | 1.431227 | 178.87 | 21.70 | 1837.18 | 112.39 | 100.45 | 17.28 | 45.13 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.49/22.19/22.87.
Separate latency-pass duration: 1.44 s; not used for throughput.

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

2.34% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.283816 | 199.41 | 23.88 | 1831.56 | 112.39 | 100.49 | 16.88 | 44.72 | 2040.00 | 76.81 |
| 2 | 1.311411 | 195.21 | 21.42 | 1833.33 | 112.39 | 100.48 | 17.46 | 45.85 | 2040.00 | 76.96 |
| 3 | 1.322865 | 193.52 | 21.05 | 1839.31 | 112.39 | 100.48 | 16.93 | 46.46 | 2040.00 | 76.85 |
| 4 | 1.296231 | 197.50 | 21.40 | 1840.47 | 112.39 | 100.47 | 16.96 | 46.99 | 2040.00 | 76.88 |
| 5 | 1.342782 | 190.65 | 21.07 | 1840.50 | 112.39 | 100.49 | 17.67 | 46.16 | 2040.00 | 76.89 |
| 6 | 1.324604 | 193.27 | 21.62 | 1840.68 | 112.39 | 100.48 | 18.37 | 46.62 | 2040.00 | 76.85 |
| 7 | 1.327728 | 192.81 | 21.39 | 1840.70 | 112.39 | 100.51 | 18.22 | 45.23 | 2040.00 | 76.07 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/21.52/22.09.
Separate latency-pass duration: 1.31 s; not used for throughput.

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

2.36% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.290883 | 198.31 | 21.35 | 1832.26 | 112.39 | 100.49 | 17.92 | 45.60 | 2040.00 | 76.92 |
| 2 | 1.331062 | 192.33 | 21.25 | 1832.81 | 112.39 | 100.39 | 16.59 | 43.90 | 2040.00 | 76.89 |
| 3 | 1.318017 | 194.23 | 21.40 | 1839.98 | 112.39 | 100.49 | 18.46 | 46.58 | 2040.00 | 76.92 |
| 4 | 1.296518 | 197.45 | 21.59 | 1840.45 | 112.39 | 100.50 | 15.54 | 44.51 | 2040.00 | 76.85 |
| 5 | 1.310339 | 195.37 | 21.47 | 1840.50 | 112.39 | 100.46 | 18.31 | 46.15 | 2040.00 | 76.88 |
| 6 | 1.302281 | 196.58 | 21.54 | 1840.68 | 112.39 | 100.45 | 16.81 | 43.29 | 2040.00 | 76.88 |
| 7 | 1.309786 | 195.45 | 21.43 | 1840.70 | 112.39 | 100.46 | 16.92 | 45.26 | 2040.00 | 76.23 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/21.52/22.26.
Separate latency-pass duration: 1.32 s; not used for throughput.

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
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 194.44 ± 12.22 | 31.95 | 20.07/20.77/21.12 | 1990.20 | 111.72/184.00 | 256/1344 | PASS |
| raw-stream | 189.29 ± 11.05 | 31.89 | 20.09/20.66/27.12 | 1990.20 | 111.72/184.00 | 256/1344 | PASS |
| native-non-stream | 170.55 ± 3.88 | 34.81 | 23.39/25.81/30.50 | 1990.20 | 111.60/184.00 | 256/1344 | PASS |
| native-stream | 171.19 ± 5.24 | 22.84 | 23.65/25.27/25.94 | 1776.38 | 111.60/184.00 | 256/1344 | PASS |
| managed-non-stream | 188.32 ± 1.57 | 22.33 | 21.72/23.04/24.15 | 1776.38 | 111.60/184.00 | 256/1344 | PASS |
| managed-stream | 185.51 ± 2.70 | 22.32 | 22.04/23.64/24.49 | 1776.38 | 111.60/184.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.445717 | 177.07 | 21.25 | 1969.13 | 111.72 | 100.45 | 16.10 | 46.03 | 2040.00 | 76.03 |
| 2 | 1.511549 | 169.36 | 32.73 | 1977.68 | 111.72 | 100.46 | 16.60 | 44.93 | 2040.00 | 77.67 |
| 3 | 1.469585 | 174.20 | 31.96 | 1977.96 | 111.72 | 100.65 | 16.20 | 43.41 | 2040.00 | 77.60 |
| 4 | 1.300822 | 196.80 | 31.95 | 1989.96 | 111.72 | 100.43 | 17.62 | 44.18 | 2040.00 | 76.92 |
| 5 | 1.308328 | 195.67 | 31.91 | 1989.96 | 111.72 | 100.48 | 17.42 | 43.69 | 2040.00 | 76.88 |
| 6 | 1.316603 | 194.44 | 31.88 | 1990.20 | 111.72 | 100.48 | 17.67 | 45.22 | 2040.00 | 76.78 |
| 7 | 1.299691 | 196.97 | 31.96 | 1990.20 | 111.72 | 100.41 | 16.77 | 44.27 | 2040.00 | 76.62 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/20.27/20.78.
Separate latency-pass duration: 1.29 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.0461100116927469, 'inference': 2.523056847671512, 'postprocess': 0.9873859453009004}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

1.57% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.463476 | 174.93 | 31.89 | 1972.64 | 111.72 | 100.43 | 16.83 | 45.12 | 2040.00 | 76.41 |
| 2 | 1.480434 | 172.92 | 23.78 | 1981.76 | 111.72 | 100.48 | 15.43 | 45.50 | 2040.00 | 77.67 |
| 3 | 1.492665 | 171.51 | 32.57 | 1977.96 | 111.72 | 100.48 | 15.83 | 45.33 | 2040.00 | 77.63 |
| 4 | 1.329222 | 192.59 | 21.20 | 1986.44 | 111.72 | 100.52 | 18.59 | 45.86 | 2040.00 | 76.85 |
| 5 | 1.352437 | 189.29 | 32.97 | 1989.96 | 111.72 | 100.47 | 17.04 | 45.55 | 2040.00 | 76.93 |
| 6 | 1.294389 | 197.78 | 21.63 | 1985.68 | 111.72 | 100.42 | 19.08 | 44.87 | 2040.00 | 76.69 |
| 7 | 1.326082 | 193.05 | 31.99 | 1990.20 | 111.72 | 100.49 | 17.44 | 46.64 | 2040.00 | 76.56 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/20.25/20.68.
Separate latency-pass duration: 1.30 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.0515287265704387, 'inference': 2.5330421328106922, 'postprocess': 1.0347066405884675}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

13.32% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.559487 | 164.16 | 34.81 | 1972.64 | 111.60 | 100.48 | 16.52 | 44.77 | 2040.00 | 76.65 |
| 2 | 1.558334 | 164.28 | 26.67 | 1765.95 | 111.60 | 100.45 | 16.26 | 44.47 | 2040.00 | 77.45 |
| 3 | 1.548700 | 165.30 | 35.61 | 1977.96 | 111.60 | 100.46 | 15.84 | 43.11 | 2040.00 | 77.55 |
| 4 | 1.499966 | 170.67 | 24.59 | 1771.98 | 111.60 | 100.44 | 16.50 | 44.84 | 2040.00 | 76.93 |
| 5 | 1.482637 | 172.67 | 35.86 | 1989.96 | 111.60 | 100.47 | 16.37 | 44.49 | 2040.00 | 76.87 |
| 6 | 1.501043 | 170.55 | 24.31 | 1772.23 | 111.60 | 100.44 | 16.80 | 45.12 | 2040.00 | 76.37 |
| 7 | 1.482950 | 172.63 | 34.95 | 1990.20 | 111.60 | 100.49 | 16.53 | 46.21 | 2040.00 | 76.33 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/24.28/26.40.
Separate latency-pass duration: 1.49 s; not used for throughput.

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

12.07% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.574383 | 162.60 | 25.17 | 1762.00 | 111.60 | 100.44 | 15.56 | 45.12 | 2040.00 | 76.88 |
| 2 | 1.585848 | 161.43 | 22.84 | 1769.59 | 111.60 | 100.48 | 15.41 | 43.78 | 2040.00 | 77.22 |
| 3 | 1.481497 | 172.80 | 22.69 | 1770.57 | 111.60 | 100.42 | 15.73 | 45.72 | 2040.00 | 77.60 |
| 4 | 1.479585 | 173.02 | 22.96 | 1776.07 | 111.60 | 100.46 | 15.73 | 45.90 | 2040.00 | 76.93 |
| 5 | 1.466275 | 174.59 | 22.38 | 1771.98 | 111.60 | 100.46 | 16.34 | 45.29 | 2040.00 | 76.86 |
| 6 | 1.517589 | 168.69 | 22.99 | 1776.38 | 111.60 | 100.44 | 15.45 | 45.35 | 2040.00 | 76.68 |
| 7 | 1.495435 | 171.19 | 22.35 | 1772.23 | 111.60 | 100.47 | 15.87 | 45.48 | 2040.00 | 76.20 |

Ready result delivery-interval P50/P95/P99 ms: 0.48/22.89/23.88.
Separate latency-pass duration: 1.49 s; not used for throughput.

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

3.78% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.359362 | 188.32 | 25.07 | 1767.18 | 111.60 | 100.48 | 18.15 | 45.17 | 2040.00 | 76.85 |
| 2 | 1.357784 | 188.54 | 26.92 | 1768.93 | 111.60 | 100.45 | 18.04 | 46.75 | 2040.00 | 77.59 |
| 3 | 1.358471 | 188.45 | 21.82 | 1774.95 | 111.60 | 100.45 | 18.26 | 45.50 | 2040.00 | 77.59 |
| 4 | 1.362036 | 187.95 | 22.33 | 1776.07 | 111.60 | 100.42 | 18.30 | 46.04 | 2040.00 | 77.00 |
| 5 | 1.357797 | 188.54 | 21.67 | 1776.13 | 111.60 | 100.46 | 17.19 | 45.88 | 2040.00 | 76.85 |
| 6 | 1.382712 | 185.14 | 22.70 | 1776.37 | 111.60 | 100.45 | 17.82 | 46.22 | 2040.00 | 76.61 |
| 7 | 1.382301 | 185.20 | 21.91 | 1776.38 | 111.60 | 100.49 | 17.04 | 45.22 | 2040.00 | 76.25 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/22.41/23.03.
Separate latency-pass duration: 1.37 s; not used for throughput.

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

3.89% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.359969 | 188.24 | 22.19 | 1767.87 | 111.60 | 100.50 | 16.85 | 46.10 | 2040.00 | 76.89 |
| 2 | 1.364190 | 187.66 | 22.44 | 1768.40 | 111.60 | 100.45 | 17.19 | 44.77 | 2040.00 | 77.11 |
| 3 | 1.394908 | 183.52 | 22.14 | 1775.61 | 111.60 | 100.45 | 16.39 | 45.90 | 2040.00 | 77.46 |
| 4 | 1.405668 | 182.12 | 23.21 | 1776.06 | 111.60 | 100.44 | 16.82 | 44.24 | 2040.00 | 76.96 |
| 5 | 1.385821 | 184.73 | 22.32 | 1776.13 | 111.60 | 100.50 | 18.07 | 45.94 | 2040.00 | 76.86 |
| 6 | 1.379990 | 185.51 | 22.36 | 1776.37 | 111.60 | 100.72 | 17.14 | 44.14 | 2040.00 | 76.64 |
| 7 | 1.350200 | 189.60 | 22.19 | 1776.38 | 111.60 | 100.48 | 17.41 | 45.81 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/23.04/23.66.
Separate latency-pass duration: 1.40 s; not used for throughput.

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

Status: **complete**. Load 5.86 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 191.63 ± 2.05 | 32.17 | 20.57/22.07/24.66 | 2252.02 | 214.50/304.00 | 256/1344 | PASS |
| raw-stream | 192.99 ± 2.72 | 31.69 | 20.73/21.81/27.99 | 2252.02 | 214.50/304.00 | 256/1344 | PASS |
| native-non-stream | 184.19 ± 2.18 | 34.70 | 21.83/23.95/29.74 | 2252.02 | 214.38/304.00 | 256/1344 | PASS |
| native-stream | 184.06 ± 3.00 | 22.41 | 21.71/22.67/23.62 | 2038.14 | 214.38/304.00 | 256/1344 | PASS |
| managed-non-stream | 199.71 ± 3.22 | 22.19 | 19.88/22.32/24.94 | 2038.14 | 214.38/304.00 | 256/1344 | PASS |
| managed-stream | 199.62 ± 2.11 | 22.03 | 20.08/21.31/22.33 | 2038.14 | 214.38/304.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.314487 | 194.75 | 21.20 | 2245.11 | 214.50 | 100.43 | 40.00 | 60.73 | 2012.22 | 79.04 |
| 2 | 1.332351 | 192.14 | 31.98 | 2250.09 | 214.50 | 100.48 | 37.59 | 58.95 | 2001.67 | 79.89 |
| 3 | 1.352230 | 189.32 | 32.34 | 2250.34 | 214.50 | 100.48 | 33.89 | 56.36 | 2005.56 | 79.30 |
| 4 | 1.333070 | 192.04 | 32.14 | 2251.79 | 214.50 | 100.47 | 38.59 | 59.59 | 2009.44 | 76.78 |
| 5 | 1.352601 | 189.26 | 32.17 | 2251.80 | 214.50 | 100.46 | 37.04 | 53.93 | 2011.11 | 76.26 |
| 6 | 1.335932 | 191.63 | 32.37 | 2252.02 | 214.50 | 100.46 | 36.37 | 56.84 | 2020.56 | 74.96 |
| 7 | 1.352260 | 189.31 | 32.38 | 2252.02 | 214.50 | 100.46 | 34.00 | 53.13 | 2004.44 | 74.78 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/20.97/23.12.
Separate latency-pass duration: 1.33 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8543319882505784, 'inference': 2.567931265623713, 'postprocess': 1.0323055469214637}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.95% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.366180 | 187.38 | 31.69 | 2245.11 | 214.50 | 100.48 | 38.00 | 60.16 | 2007.86 | 79.36 |
| 2 | 1.327961 | 192.78 | 21.11 | 2246.95 | 214.50 | 100.46 | 38.78 | 61.19 | 1999.44 | 80.26 |
| 3 | 1.326512 | 192.99 | 32.17 | 2250.37 | 214.50 | 100.69 | 39.19 | 58.84 | 1999.44 | 79.30 |
| 4 | 1.320388 | 193.88 | 21.61 | 2251.79 | 214.50 | 100.44 | 36.33 | 60.28 | 2013.33 | 76.93 |
| 5 | 1.358138 | 188.49 | 32.48 | 2251.80 | 214.50 | 100.47 | 36.07 | 58.65 | 2011.11 | 76.22 |
| 6 | 1.319430 | 194.02 | 21.03 | 2252.02 | 214.50 | 100.46 | 39.22 | 57.95 | 2023.89 | 75.04 |
| 7 | 1.324245 | 193.32 | 32.18 | 2252.02 | 214.50 | 100.45 | 39.07 | 57.68 | 2018.33 | 74.93 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/21.07/22.27.
Separate latency-pass duration: 1.35 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8729840977039771, 'inference': 2.642236500008721, 'postprocess': 1.1100989219414714}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

2.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.412494 | 181.24 | 34.94 | 2245.12 | 214.38 | 100.45 | 34.39 | 59.29 | 2022.86 | 79.46 |
| 2 | 1.400304 | 182.82 | 24.89 | 2035.58 | 214.38 | 100.55 | 34.46 | 58.57 | 2000.36 | 80.46 |
| 3 | 1.389867 | 184.19 | 34.70 | 2250.37 | 214.38 | 100.45 | 35.14 | 57.41 | 2021.79 | 78.43 |
| 4 | 1.364449 | 187.62 | 23.66 | 2037.91 | 214.38 | 100.48 | 38.86 | 57.79 | 2013.21 | 77.25 |
| 5 | 1.397053 | 183.24 | 35.03 | 2251.80 | 214.38 | 100.41 | 35.71 | 58.06 | 2016.43 | 76.07 |
| 6 | 1.375483 | 186.12 | 23.54 | 2038.14 | 214.38 | 100.63 | 34.11 | 57.50 | 2017.50 | 75.00 |
| 7 | 1.380037 | 185.50 | 34.75 | 2252.02 | 214.38 | 100.48 | 34.86 | 56.70 | 2019.64 | 74.71 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/22.11/24.74.
Separate latency-pass duration: 1.42 s; not used for throughput.

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

3.70% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.373396 | 186.40 | 22.46 | 2028.21 | 214.38 | 100.44 | 37.29 | 57.88 | 2007.86 | 80.14 |
| 2 | 1.378158 | 185.76 | 22.33 | 2035.09 | 214.38 | 100.45 | 37.18 | 62.99 | 2003.04 | 80.50 |
| 3 | 1.381213 | 185.34 | 22.47 | 2032.41 | 214.38 | 100.49 | 36.25 | 60.49 | 2005.71 | 78.43 |
| 4 | 1.390819 | 184.06 | 22.41 | 2037.91 | 214.38 | 100.46 | 37.89 | 59.60 | 2023.93 | 77.32 |
| 5 | 1.402708 | 182.50 | 22.50 | 2033.82 | 214.38 | 100.42 | 38.11 | 59.26 | 2017.50 | 75.86 |
| 6 | 1.442183 | 177.51 | 22.41 | 2038.14 | 214.38 | 100.61 | 32.90 | 57.74 | 2026.55 | 75.31 |
| 7 | 1.394994 | 183.51 | 22.30 | 2034.05 | 214.38 | 100.48 | 36.18 | 58.39 | 2022.86 | 74.68 |

Ready result delivery-interval P50/P95/P99 ms: 0.44/20.77/21.43.
Separate latency-pass duration: 1.40 s; not used for throughput.

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

-4.53% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.277165 | 200.44 | 22.39 | 2032.75 | 214.38 | 100.47 | 41.62 | 62.28 | 1961.54 | 80.15 |
| 2 | 1.269034 | 201.73 | 22.02 | 2034.46 | 214.38 | 100.45 | 37.65 | 66.95 | 1970.77 | 80.77 |
| 3 | 1.291510 | 198.22 | 22.25 | 2036.79 | 214.38 | 100.51 | 41.54 | 57.04 | 2038.85 | 78.15 |
| 4 | 1.329198 | 192.60 | 22.15 | 2037.91 | 214.38 | 100.52 | 39.15 | 59.50 | 1992.22 | 77.74 |
| 5 | 1.281872 | 199.71 | 22.35 | 2037.91 | 214.38 | 100.42 | 37.62 | 59.23 | 1985.77 | 75.92 |
| 6 | 1.266239 | 202.17 | 22.19 | 2038.14 | 214.38 | 100.44 | 38.69 | 61.49 | 2012.31 | 75.31 |
| 7 | 1.291055 | 198.29 | 21.95 | 2038.14 | 214.38 | 100.45 | 40.69 | 56.72 | 2039.42 | 74.77 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/20.28/23.24.
Separate latency-pass duration: 1.30 s; not used for throughput.

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

-3.78% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.273711 | 200.99 | 21.97 | 2033.40 | 214.38 | 100.42 | 41.96 | 63.44 | 1950.00 | 80.77 |
| 2 | 1.303368 | 196.41 | 22.88 | 2033.94 | 214.38 | 100.46 | 34.15 | 61.35 | 1986.35 | 80.62 |
| 3 | 1.305378 | 196.11 | 22.03 | 2037.42 | 214.38 | 100.46 | 39.19 | 65.36 | 2010.00 | 78.04 |
| 4 | 1.282644 | 199.59 | 21.92 | 2037.89 | 214.38 | 100.48 | 36.65 | 52.50 | 2038.85 | 77.69 |
| 5 | 1.282424 | 199.62 | 22.14 | 2037.91 | 214.38 | 100.44 | 40.19 | 66.40 | 2028.46 | 75.50 |
| 6 | 1.279916 | 200.01 | 22.04 | 2038.12 | 214.38 | 100.47 | 36.46 | 54.75 | 2040.00 | 74.69 |
| 7 | 1.270424 | 201.51 | 21.94 | 2038.14 | 214.38 | 100.48 | 39.77 | 58.71 | 2035.38 | 75.08 |

Ready result delivery-interval P50/P95/P99 ms: 0.03/20.33/21.33.
Separate latency-pass duration: 1.29 s; not used for throughput.

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

Status: **complete**. Load 12.46 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 5.05 ± 0.02 | 788.72 | 792.73/795.72/796.16 | 2979.10 | 3486.48/4298.00 | 32/176 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 5.05 ± 0.02 | 793.84 | 794.23/797.27/798.10 | 2979.11 | 3486.46/4298.00 | 32/176 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 5.01 ± 0.02 | 791.98 | 797.58/802.26/803.30 | 3016.73 | 3486.46/4298.00 | 32/176 | PASS |
| managed-stream | 5.03 ± 0.01 | 788.15 | 795.92/797.64/797.77 | 3012.14 | 3486.46/4298.00 | 32/176 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.358328 | 5.03 | 786.27 | 2978.42 | 3486.48 | 100.47 | 80.72 | 66.47 | 1338.19 | 80.31 |
| 2 | 6.307286 | 5.07 | 788.75 | 2978.45 | 3486.48 | 100.45 | 76.13 | 64.05 | 1396.67 | 75.17 |
| 3 | 6.287716 | 5.09 | 783.72 | 2978.46 | 3486.48 | 100.42 | 76.82 | 63.66 | 1390.44 | 75.04 |
| 4 | 6.345247 | 5.04 | 788.49 | 2978.78 | 3486.48 | 100.49 | 76.13 | 62.41 | 1375.87 | 77.22 |
| 5 | 6.336777 | 5.05 | 792.81 | 2973.75 | 3486.48 | 100.48 | 80.51 | 62.83 | 1391.79 | 77.10 |
| 6 | 6.348511 | 5.04 | 788.72 | 2979.10 | 3486.48 | 100.52 | 71.70 | 63.02 | 1399.96 | 76.51 |
| 7 | 6.329932 | 5.06 | 790.27 | 2979.10 | 3486.48 | 100.43 | 71.75 | 62.52 | 1392.86 | 76.69 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/794.62/795.76.
Separate latency-pass duration: 6.34 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.376386 | 5.02 | 798.56 | 2978.43 | 3486.46 | 100.42 | 80.79 | 62.06 | 1385.79 | 80.46 |
| 2 | 6.336874 | 5.05 | 783.18 | 2978.45 | 3486.46 | 100.41 | 76.06 | 64.68 | 1391.19 | 75.40 |
| 3 | 6.338321 | 5.05 | 793.84 | 2978.46 | 3486.46 | 100.43 | 74.85 | 64.53 | 1405.00 | 74.95 |
| 4 | 6.340895 | 5.05 | 787.07 | 2978.76 | 3486.46 | 100.44 | 78.48 | 63.41 | 1387.62 | 77.25 |
| 5 | 6.374767 | 5.02 | 799.93 | 2978.79 | 3486.46 | 100.54 | 73.09 | 64.89 | 1374.33 | 76.80 |
| 6 | 6.339312 | 5.05 | 787.14 | 2978.17 | 3486.46 | 100.48 | 80.44 | 62.96 | 1396.79 | 76.35 |
| 7 | 6.378409 | 5.02 | 796.83 | 2979.11 | 3486.46 | 100.54 | 77.31 | 65.65 | 1388.86 | 76.63 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/795.15/797.38.
Separate latency-pass duration: 6.35 s; not used for throughput.

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

0.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.390064 | 5.01 | 800.59 | 3011.24 | 3486.46 | 100.44 | 77.23 | 65.65 | 1355.55 | 78.81 |
| 2 | 6.355330 | 5.04 | 787.25 | 2997.20 | 3486.46 | 100.44 | 79.81 | 63.97 | 1404.45 | 75.93 |
| 3 | 6.340567 | 5.05 | 791.98 | 3016.03 | 3486.46 | 100.41 | 78.96 | 63.37 | 1396.18 | 75.43 |
| 4 | 6.356969 | 5.03 | 790.90 | 3002.20 | 3486.46 | 100.44 | 77.41 | 65.32 | 1391.10 | 76.79 |
| 5 | 6.387203 | 5.01 | 796.27 | 3011.65 | 3486.46 | 100.51 | 71.35 | 61.54 | 1393.35 | 76.72 |
| 6 | 6.381996 | 5.01 | 789.25 | 3011.88 | 3486.46 | 100.48 | 79.75 | 63.05 | 1396.06 | 76.42 |
| 7 | 6.397840 | 5.00 | 799.97 | 3016.73 | 3486.46 | 100.58 | 79.48 | 63.58 | 1415.98 | 76.68 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/799.77/802.38.
Separate latency-pass duration: 6.39 s; not used for throughput.

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

0.46% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.365086 | 5.03 | 789.57 | 3001.87 | 3486.46 | 100.42 | 71.75 | 62.81 | 1389.33 | 77.78 |
| 2 | 6.351052 | 5.04 | 787.65 | 3001.88 | 3486.46 | 100.42 | 72.25 | 62.08 | 1374.33 | 76.73 |
| 3 | 6.358783 | 5.03 | 789.26 | 2997.37 | 3486.46 | 100.53 | 70.79 | 62.19 | 1380.00 | 75.89 |
| 4 | 6.363589 | 5.03 | 788.06 | 3002.17 | 3486.46 | 100.45 | 70.96 | 61.66 | 1375.16 | 76.35 |
| 5 | 6.366234 | 5.03 | 788.15 | 3002.38 | 3486.46 | 100.47 | 72.76 | 63.77 | 1385.31 | 76.48 |
| 6 | 6.360850 | 5.03 | 787.36 | 3002.49 | 3486.46 | 100.49 | 71.91 | 63.34 | 1390.75 | 76.43 |
| 7 | 6.385874 | 5.01 | 792.82 | 3012.14 | 3486.46 | 100.51 | 79.69 | 62.66 | 1400.43 | 76.64 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/797.19/797.57.
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

Status: **complete**. Load 12.69 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 3.70 ± 0.01 | 1077.92 | 1079.64/1081.15/1081.51 | 2986.48 | 3731.84/4524.00 | 32/168 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 3.70 ± 0.01 | 1074.63 | 1079.39/1081.63/1082.11 | 2986.50 | 3731.82/4524.00 | 32/168 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 3.69 ± 0.01 | 1079.55 | 1080.27/1083.40/1083.60 | 3023.20 | 3731.82/4524.00 | 32/168 | PASS |
| managed-stream | 3.70 ± 0.01 | 1074.32 | 1082.29/1088.55/1088.72 | 3023.20 | 3731.82/4524.00 | 32/168 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.668117 | 3.69 | 1077.92 | 2985.68 | 3731.84 | 100.45 | 83.03 | 66.75 | 1278.03 | 79.01 |
| 2 | 8.636699 | 3.71 | 1076.28 | 2985.70 | 3731.84 | 100.52 | 82.77 | 65.84 | 1295.32 | 76.61 |
| 3 | 8.650623 | 3.70 | 1081.78 | 2985.71 | 3731.84 | 100.50 | 83.74 | 65.79 | 1288.78 | 77.24 |
| 4 | 8.634398 | 3.71 | 1077.45 | 2986.04 | 3731.84 | 100.51 | 82.19 | 65.31 | 1299.51 | 75.88 |
| 5 | 8.644591 | 3.70 | 1079.96 | 2986.05 | 3731.84 | 100.51 | 82.49 | 66.00 | 1299.51 | 75.90 |
| 6 | 8.640729 | 3.70 | 1082.73 | 2986.46 | 3731.84 | 100.46 | 82.72 | 65.73 | 1294.88 | 76.36 |
| 7 | 8.633875 | 3.71 | 1076.15 | 2986.48 | 3731.84 | 100.46 | 83.12 | 65.88 | 1291.57 | 76.57 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1080.24/1081.17.
Separate latency-pass duration: 8.63 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

-0.03% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.667977 | 3.69 | 1083.86 | 2985.69 | 3731.82 | 100.45 | 82.29 | 65.73 | 1294.51 | 77.39 |
| 2 | 8.631332 | 3.71 | 1070.64 | 2985.70 | 3731.82 | 100.46 | 82.60 | 65.87 | 1294.45 | 75.73 |
| 3 | 8.683191 | 3.69 | 1085.00 | 2985.71 | 3731.82 | 100.59 | 82.70 | 66.18 | 1285.84 | 77.65 |
| 4 | 8.639446 | 3.70 | 1072.30 | 2985.96 | 3731.82 | 100.48 | 81.51 | 65.43 | 1303.34 | 76.02 |
| 5 | 8.639835 | 3.70 | 1080.34 | 2986.09 | 3731.82 | 100.50 | 82.33 | 65.10 | 1303.52 | 76.21 |
| 6 | 8.632187 | 3.71 | 1072.57 | 2986.41 | 3731.82 | 100.44 | 83.13 | 65.53 | 1303.60 | 76.39 |
| 7 | 8.631234 | 3.71 | 1074.63 | 2986.50 | 3731.82 | 100.44 | 83.00 | 66.04 | 1300.64 | 76.69 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1080.17/1081.66.
Separate latency-pass duration: 8.63 s; not used for throughput.

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

0.42% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.634536 | 3.71 | 1079.55 | 3023.20 | 3731.82 | 100.45 | 81.58 | 66.05 | 1310.93 | 75.97 |
| 2 | 8.633224 | 3.71 | 1073.98 | 3023.20 | 3731.82 | 100.47 | 83.96 | 65.54 | 1309.01 | 74.96 |
| 3 | 8.711708 | 3.67 | 1089.44 | 3013.85 | 3731.82 | 100.56 | 81.76 | 65.79 | 1295.29 | 77.63 |
| 4 | 8.670253 | 3.69 | 1074.50 | 3018.78 | 3731.82 | 100.55 | 82.27 | 65.27 | 1301.88 | 76.38 |
| 5 | 8.684146 | 3.68 | 1088.06 | 3018.43 | 3731.82 | 100.51 | 81.57 | 66.18 | 1299.62 | 76.58 |
| 6 | 8.650908 | 3.70 | 1072.55 | 3000.55 | 3731.82 | 100.46 | 83.06 | 65.27 | 1298.02 | 76.57 |
| 7 | 8.669857 | 3.69 | 1086.15 | 3019.38 | 3731.82 | 100.46 | 82.95 | 65.74 | 1300.32 | 76.71 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1082.92/1083.40.
Separate latency-pass duration: 8.65 s; not used for throughput.

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

0.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.632746 | 3.71 | 1074.32 | 3013.82 | 3731.82 | 100.47 | 81.95 | 65.63 | 1311.45 | 74.96 |
| 2 | 8.633550 | 3.71 | 1072.22 | 3023.20 | 3731.82 | 100.47 | 81.98 | 65.01 | 1307.79 | 74.69 |
| 3 | 8.708492 | 3.67 | 1089.25 | 3014.01 | 3731.82 | 100.54 | 82.54 | 66.01 | 1292.43 | 77.51 |
| 4 | 8.686000 | 3.68 | 1073.49 | 3018.78 | 3731.82 | 100.57 | 82.10 | 65.84 | 1297.37 | 76.90 |
| 5 | 8.652966 | 3.70 | 1074.66 | 3014.39 | 3731.82 | 100.46 | 81.62 | 65.60 | 1299.88 | 76.76 |
| 6 | 8.677229 | 3.69 | 1072.50 | 3018.54 | 3731.82 | 100.46 | 81.94 | 65.71 | 1293.12 | 76.82 |
| 7 | 8.644745 | 3.70 | 1074.58 | 3020.84 | 3731.82 | 100.45 | 82.56 | 65.77 | 1301.08 | 76.49 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/1086.39/1088.45.
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

Status: **complete**. Load 1.67 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 72.03 ± 0.16 | 55.33 | 55.55/56.28/56.64 | 2044.98 | 561.57/1006.00 | 128/1120 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 69.89 ± 0.17 | 57.24 | 57.27/59.28/60.02 | 2044.98 | 561.45/1006.00 | 128/1120 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 68.59 ± 1.20 | 58.13 | 58.25/59.90/60.48 | 2046.74 | 561.45/1006.00 | 128/1120 | PASS |
| managed-stream | 68.74 ± 0.14 | 57.77 | 57.97/58.67/58.76 | 2046.34 | 561.45/1006.00 | 128/1120 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.774954 | 72.11 | 55.19 | 2036.01 | 561.57 | 119.21 | 68.53 | 67.07 | 1677.50 | 77.00 |
| 2 | 1.777017 | 72.03 | 55.48 | 2035.00 | 561.57 | 119.20 | 63.89 | 65.28 | 1618.33 | 79.39 |
| 3 | 1.775326 | 72.10 | 55.32 | 2035.00 | 561.57 | 119.26 | 64.53 | 65.97 | 1641.25 | 78.53 |
| 4 | 1.783494 | 71.77 | 55.39 | 2044.98 | 561.57 | 119.24 | 65.58 | 67.87 | 1642.50 | 77.17 |
| 5 | 1.780756 | 71.88 | 55.43 | 2044.98 | 561.57 | 119.43 | 63.33 | 63.93 | 1670.83 | 77.06 |
| 6 | 1.771010 | 72.28 | 55.28 | 2042.78 | 561.57 | 119.29 | 68.19 | 68.65 | 1673.33 | 76.31 |
| 7 | 1.777776 | 72.00 | 55.33 | 2042.78 | 561.57 | 119.44 | 66.61 | 63.68 | 1655.00 | 76.17 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/55.93/56.26.
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

3.18% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.830762 | 69.92 | 57.38 | 1921.30 | 561.45 | 115.09 | 63.73 | 68.15 | 1674.73 | 77.51 |
| 2 | 1.834533 | 69.77 | 56.93 | 2034.61 | 561.45 | 115.03 | 59.84 | 68.11 | 1688.11 | 79.35 |
| 3 | 1.831847 | 69.87 | 57.50 | 2035.00 | 561.45 | 115.05 | 63.05 | 67.60 | 1695.81 | 78.62 |
| 4 | 1.830776 | 69.92 | 57.24 | 2044.97 | 561.45 | 115.00 | 62.84 | 66.29 | 1693.78 | 77.59 |
| 5 | 1.842594 | 69.47 | 57.71 | 2044.98 | 561.45 | 115.12 | 66.59 | 67.85 | 1697.03 | 77.05 |
| 6 | 1.829351 | 69.97 | 57.19 | 2042.78 | 561.45 | 115.09 | 67.43 | 65.92 | 1679.19 | 76.46 |
| 7 | 1.831407 | 69.89 | 57.22 | 2042.78 | 561.45 | 115.16 | 61.24 | 67.29 | 1681.62 | 75.84 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/57.66/59.53.
Separate latency-pass duration: 1.84 s; not used for throughput.

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

5.12% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.950868 | 65.61 | 57.95 | 2030.19 | 561.45 | 118.40 | 58.90 | 65.44 | 1733.85 | 78.03 |
| 2 | 1.868113 | 68.52 | 58.13 | 2046.74 | 561.45 | 114.83 | 61.92 | 65.21 | 1668.55 | 78.97 |
| 3 | 1.866296 | 68.59 | 58.17 | 2038.40 | 561.45 | 114.78 | 68.08 | 66.84 | 1707.24 | 78.13 |
| 4 | 1.852020 | 69.11 | 57.72 | 2044.97 | 561.45 | 114.42 | 60.54 | 64.89 | 1676.35 | 77.46 |
| 5 | 1.865437 | 68.62 | 58.59 | 2044.98 | 561.45 | 114.73 | 65.26 | 66.25 | 1690.26 | 76.79 |
| 6 | 1.871929 | 68.38 | 58.53 | 2046.29 | 561.45 | 114.75 | 65.11 | 65.92 | 1721.05 | 76.34 |
| 7 | 1.854396 | 69.03 | 57.81 | 2046.30 | 561.45 | 114.75 | 60.19 | 65.16 | 1709.19 | 75.68 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/58.65/60.00.
Separate latency-pass duration: 1.87 s; not used for throughput.

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

4.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.854019 | 69.04 | 57.49 | 2043.51 | 561.45 | 114.51 | 66.30 | 64.68 | 1667.03 | 78.41 |
| 2 | 1.860083 | 68.81 | 57.77 | 2046.34 | 561.45 | 114.50 | 59.89 | 62.65 | 1697.84 | 78.78 |
| 3 | 1.863106 | 68.70 | 57.64 | 2038.76 | 561.45 | 114.45 | 63.70 | 66.67 | 1665.41 | 77.70 |
| 4 | 1.865558 | 68.61 | 57.56 | 2044.97 | 561.45 | 114.52 | 60.42 | 63.79 | 1687.11 | 77.53 |
| 5 | 1.859807 | 68.82 | 58.14 | 2046.15 | 561.45 | 114.57 | 64.78 | 67.23 | 1692.16 | 76.73 |
| 6 | 1.862060 | 68.74 | 58.38 | 2046.29 | 561.45 | 114.69 | 62.05 | 65.17 | 1697.43 | 76.43 |
| 7 | 1.863189 | 68.70 | 58.37 | 2046.30 | 561.45 | 114.70 | 60.76 | 64.99 | 1697.03 | 75.76 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/58.23/58.59.
Separate latency-pass duration: 1.86 s; not used for throughput.

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

Status: **complete**. Load 19.27 s. Device `cuda:0`, dtype `float32`; observed after complete warmup, before measured repeats.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 10.21 ± 0.07 | 390.52 | 392.86/399.48/399.85 | 3199.68 | 509.82/752.00 | 32/160 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 10.07 ± 0.06 | 396.66 | 406.76/410.41/410.49 | 3199.68 | 477.43/752.00 | 32/160 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 10.01 ± 0.10 | 401.23 | 399.43/401.91/402.26 | 3237.20 | 490.85/752.00 | 32/160 | PASS |
| managed-stream | 10.04 ± 0.07 | 404.56 | 395.46/402.48/404.51 | 3237.27 | 495.01/752.00 | 32/160 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.131628 | 10.22 | 403.24 | 3197.90 | 495.55 | 101.94 | 6.24 | 35.80 | 2040.00 | 71.00 |
| 2 | 3.135533 | 10.21 | 390.52 | 3199.24 | 509.82 | 101.87 | 4.24 | 35.92 | 2040.00 | 72.56 |
| 3 | 3.132502 | 10.22 | 388.87 | 3199.25 | 509.82 | 101.86 | 7.76 | 36.31 | 2040.00 | 72.95 |
| 4 | 3.133053 | 10.21 | 388.34 | 3199.50 | 509.82 | 101.90 | 5.82 | 36.81 | 2040.00 | 73.95 |
| 5 | 3.192210 | 10.02 | 403.57 | 3199.50 | 509.06 | 101.91 | 7.60 | 37.44 | 2040.00 | 74.00 |
| 6 | 3.163288 | 10.12 | 402.27 | 3199.67 | 498.39 | 101.89 | 3.83 | 37.11 | 2040.00 | 75.00 |
| 7 | 3.132725 | 10.21 | 389.13 | 3199.68 | 509.82 | 101.88 | 6.61 | 37.34 | 2040.00 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/396.91/399.51.
Separate latency-pass duration: 3.16 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

1.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.155406 | 10.14 | 393.72 | 3198.76 | 477.43 | 101.83 | 5.97 | 36.45 | 2040.00 | 71.32 |
| 2 | 3.187725 | 10.04 | 403.93 | 3199.18 | 477.43 | 102.01 | 7.67 | 35.53 | 2040.00 | 72.16 |
| 3 | 3.197236 | 10.01 | 394.87 | 3199.27 | 477.43 | 101.95 | 3.60 | 36.92 | 2040.00 | 72.94 |
| 4 | 3.169657 | 10.10 | 402.80 | 3199.50 | 477.43 | 101.99 | 6.38 | 36.15 | 2040.00 | 73.92 |
| 5 | 3.176648 | 10.07 | 393.41 | 3199.50 | 477.43 | 101.93 | 5.27 | 37.62 | 2040.00 | 74.00 |
| 6 | 3.215362 | 9.95 | 401.58 | 3199.66 | 477.43 | 101.96 | 4.48 | 38.18 | 2040.00 | 74.94 |
| 7 | 3.170758 | 10.09 | 396.66 | 3199.68 | 477.43 | 101.91 | 5.65 | 37.56 | 2040.00 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/409.37/410.38.
Separate latency-pass duration: 3.24 s; not used for throughput.

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

1.87% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.224673 | 9.92 | 398.29 | 3236.33 | 490.85 | 102.34 | 6.97 | 35.24 | 2040.00 | 71.90 |
| 2 | 3.173310 | 10.08 | 401.23 | 3236.61 | 490.85 | 102.29 | 6.11 | 36.78 | 2040.00 | 72.00 |
| 3 | 3.190932 | 10.03 | 398.98 | 3236.84 | 490.85 | 102.20 | 5.27 | 35.94 | 2040.00 | 73.00 |
| 4 | 3.201421 | 10.00 | 406.22 | 3236.99 | 490.85 | 102.28 | 7.40 | 37.53 | 2040.00 | 73.32 |
| 5 | 3.185470 | 10.05 | 396.01 | 3237.02 | 490.85 | 102.25 | 3.56 | 36.01 | 2040.00 | 74.00 |
| 6 | 3.196444 | 10.01 | 406.03 | 3237.15 | 490.85 | 102.28 | 7.83 | 36.60 | 2040.00 | 74.79 |
| 7 | 3.273038 | 9.78 | 416.29 | 3237.20 | 490.85 | 102.20 | 4.78 | 37.00 | 2040.00 | 75.06 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/400.69/401.94.
Separate latency-pass duration: 3.20 s; not used for throughput.

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

1.64% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.194285 | 10.02 | 417.11 | 3236.43 | 490.85 | 102.28 | 6.71 | 35.04 | 2040.00 | 71.95 |
| 2 | 3.183358 | 10.05 | 403.40 | 3236.52 | 490.85 | 102.26 | 4.08 | 36.96 | 2040.00 | 72.00 |
| 3 | 3.188067 | 10.04 | 403.20 | 3236.91 | 495.01 | 102.27 | 3.67 | 36.63 | 2040.00 | 73.00 |
| 4 | 3.198859 | 10.00 | 403.45 | 3236.99 | 490.85 | 102.29 | 4.63 | 35.51 | 2040.00 | 73.22 |
| 5 | 3.243289 | 9.87 | 405.90 | 3237.07 | 490.85 | 102.31 | 6.02 | 37.02 | 2040.00 | 74.12 |
| 6 | 3.182910 | 10.05 | 404.56 | 3237.14 | 490.85 | 102.31 | 5.78 | 36.71 | 2040.00 | 74.46 |
| 7 | 3.184062 | 10.05 | 405.77 | 3237.27 | 490.85 | 102.27 | 3.65 | 37.25 | 2040.00 | 75.29 |

Ready result delivery-interval P50/P95/P99 ms: 0.03/396.95/402.65.
Separate latency-pass duration: 3.17 s; not used for throughput.

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

Status: **complete**. Load 0.10 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 107.83 ± 1.97 | 1112.83 | 1168.44/1168.44/1168.44 | 2167.85 | 34.30/62.00 | 120/665 | PASS |
| raw-stream | 105.96 ± 1.34 | 47.10 | 1127.81/1127.81/1127.81 | 2167.98 | 34.30/62.00 | 120/665 | PASS |
| native-non-stream | 100.89 ± 1.53 | 1189.32 | 1179.82/1179.82/1179.82 | 2167.98 | 34.30/62.00 | 120/665 | PASS |
| native-stream | 100.40 ± 0.97 | 50.36 | 1222.94/1222.94/1222.94 | 2167.98 | 34.25/62.00 | 120/665 | PASS |
| managed-non-stream | 99.21 ± 0.93 | 1209.42 | 1226.98/1226.98/1226.98 | 1861.13 | 24.66/62.00 | 120/665 | PASS |
| managed-stream | 97.47 ± 1.27 | 83.69 | 1250.38/1250.38/1250.38 | 1861.12 | 24.66/62.00 | 120/665 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.132736 | 105.94 | 1132.65 | 2155.58 | 34.30 | 154.71 | 13.87 | 41.03 | 2040.00 | 77.13 |
| 2 | 1.111145 | 108.00 | 1111.06 | 2166.85 | 34.30 | 145.82 | 14.39 | 41.92 | 2040.00 | 77.83 |
| 3 | 1.111304 | 107.98 | 1111.22 | 2166.96 | 34.30 | 145.15 | 13.87 | 41.16 | 2040.00 | 77.83 |
| 4 | 1.166701 | 102.85 | 1166.59 | 2167.08 | 34.30 | 143.73 | 13.71 | 41.95 | 2040.00 | 76.92 |
| 5 | 1.105831 | 108.52 | 1105.75 | 2167.08 | 34.30 | 148.10 | 13.73 | 40.64 | 2040.00 | 76.91 |
| 6 | 1.112913 | 107.83 | 1112.83 | 2167.85 | 34.30 | 145.00 | 14.22 | 41.47 | 2040.00 | 76.00 |
| 7 | 1.128051 | 106.38 | 1127.97 | 2167.85 | 34.30 | 146.21 | 13.00 | 40.91 | 2040.00 | 75.91 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
Separate latency-pass duration: 1.17 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.197576250054529, 'inference': 6.6092631499789904, 'postprocess': 1.2201732584344427}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.42% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.098893 | 109.20 | 45.91 | 2158.22 | 34.30 | 144.81 | 14.27 | 42.12 | 2040.00 | 77.55 |
| 2 | 1.115843 | 107.54 | 47.10 | 2166.85 | 34.30 | 145.11 | 14.04 | 42.05 | 2040.00 | 77.87 |
| 3 | 1.134322 | 105.79 | 46.74 | 2166.97 | 34.30 | 144.39 | 14.39 | 41.99 | 2040.00 | 77.78 |
| 4 | 1.132508 | 105.96 | 47.37 | 2164.45 | 34.30 | 146.75 | 12.74 | 41.72 | 2040.00 | 76.87 |
| 5 | 1.135202 | 105.71 | 49.17 | 2167.08 | 34.30 | 145.24 | 14.00 | 41.75 | 2040.00 | 76.87 |
| 6 | 1.137204 | 105.52 | 49.13 | 2165.21 | 34.30 | 143.51 | 13.78 | 41.34 | 2040.00 | 76.00 |
| 7 | 1.128229 | 106.36 | 46.43 | 2167.98 | 34.30 | 145.30 | 14.04 | 41.75 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 8.90/10.72/11.55.
Separate latency-pass duration: 1.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1686943333643285, 'inference': 6.372771583528447, 'postprocess': 1.2053319000945824}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

5.04% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.174432 | 102.18 | 1174.32 | 2158.24 | 34.30 | 143.69 | 14.96 | 42.23 | 2040.00 | 77.75 |
| 2 | 1.199975 | 100.00 | 1199.87 | 2166.18 | 34.30 | 153.02 | 14.17 | 41.86 | 2040.00 | 77.83 |
| 3 | 1.167337 | 102.80 | 1167.23 | 2166.97 | 34.30 | 143.65 | 14.17 | 41.85 | 2040.00 | 77.75 |
| 4 | 1.189939 | 100.85 | 1189.81 | 2164.18 | 34.30 | 153.98 | 14.08 | 41.77 | 2040.00 | 77.00 |
| 5 | 1.189438 | 100.89 | 1189.32 | 2167.08 | 34.30 | 141.59 | 14.17 | 41.62 | 2040.00 | 76.88 |
| 6 | 1.211275 | 99.07 | 1211.19 | 2164.81 | 34.30 | 153.75 | 13.76 | 41.67 | 2040.00 | 76.00 |
| 7 | 1.161574 | 103.31 | 1161.50 | 2167.98 | 34.30 | 143.92 | 14.83 | 41.59 | 2040.00 | 76.00 |

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

6.27% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.191623 | 100.70 | 47.37 | 2158.71 | 34.25 | 140.05 | 13.42 | 41.79 | 2040.00 | 77.79 |
| 2 | 1.180814 | 101.62 | 50.36 | 1858.00 | 34.25 | 141.31 | 13.92 | 41.36 | 2040.00 | 77.88 |
| 3 | 1.211105 | 99.08 | 50.74 | 2166.97 | 34.25 | 143.88 | 12.36 | 41.73 | 2040.00 | 77.56 |
| 4 | 1.195241 | 100.40 | 51.57 | 1860.15 | 34.25 | 140.21 | 13.21 | 41.92 | 2040.00 | 77.00 |
| 5 | 1.192392 | 100.64 | 47.89 | 2167.08 | 34.25 | 140.99 | 13.00 | 40.93 | 2040.00 | 76.83 |
| 6 | 1.214020 | 98.85 | 52.10 | 1860.76 | 34.25 | 141.17 | 13.12 | 41.19 | 2040.00 | 76.00 |
| 7 | 1.197447 | 100.21 | 48.74 | 2167.98 | 34.25 | 140.15 | 12.92 | 40.75 | 2040.00 | 75.92 |

Ready result delivery-interval P50/P95/P99 ms: 9.53/11.67/17.81.
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

7.81% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.189823 | 100.86 | 1189.73 | 1853.33 | 24.66 | 142.56 | 13.33 | 41.47 | 2040.00 | 77.83 |
| 2 | 1.222072 | 98.19 | 1221.99 | 1858.12 | 24.66 | 143.05 | 13.24 | 40.96 | 2040.00 | 77.84 |
| 3 | 1.209514 | 99.21 | 1209.42 | 1858.57 | 24.66 | 143.34 | 12.79 | 41.39 | 2040.00 | 77.46 |
| 4 | 1.194695 | 100.44 | 1194.60 | 1860.52 | 24.66 | 140.85 | 13.71 | 41.05 | 2040.00 | 77.00 |
| 5 | 1.211478 | 99.05 | 1211.39 | 1858.50 | 24.66 | 140.88 | 12.96 | 41.15 | 2040.00 | 76.80 |
| 6 | 1.199845 | 100.01 | 1199.76 | 1861.13 | 24.66 | 140.47 | 13.29 | 40.53 | 2040.00 | 76.00 |
| 7 | 1.211952 | 99.01 | 1211.86 | 1859.36 | 24.66 | 142.69 | 13.40 | 41.07 | 2040.00 | 75.92 |

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

9.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.231091 | 97.47 | 83.47 | 1853.37 | 24.66 | 140.36 | 13.04 | 41.40 | 2040.00 | 77.84 |
| 2 | 1.230521 | 97.52 | 84.71 | 1857.56 | 24.66 | 142.85 | 12.40 | 40.42 | 2040.00 | 77.84 |
| 3 | 1.242711 | 96.56 | 83.71 | 1858.55 | 24.66 | 141.42 | 12.48 | 41.28 | 2040.00 | 77.00 |
| 4 | 1.210572 | 99.13 | 82.58 | 1860.49 | 24.66 | 140.85 | 12.32 | 40.23 | 2040.00 | 77.00 |
| 5 | 1.204557 | 99.62 | 83.69 | 1858.33 | 24.66 | 141.48 | 12.67 | 40.96 | 2040.00 | 76.67 |
| 6 | 1.247767 | 96.17 | 81.09 | 1861.12 | 24.66 | 140.73 | 10.88 | 39.74 | 2040.00 | 76.00 |
| 7 | 1.234836 | 97.18 | 83.75 | 1859.93 | 24.66 | 140.20 | 12.80 | 40.69 | 2040.00 | 75.84 |

Ready result delivery-interval P50/P95/P99 ms: 9.46/11.32/13.79.
Separate latency-pass duration: 1.25 s; not used for throughput.

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
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 223.68 ± 4.04 | 536.41 | 547.08/547.08/547.08 | 2395.85 | 93.11/150.00 | 120/665 | PASS |
| raw-stream | 218.43 ± 6.52 | 59.42 | 541.60/541.60/541.60 | 2370.40 | 93.11/150.00 | 120/665 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 200.72 ± 3.62 | 597.74 | 603.91/603.91/603.91 | 2350.16 | 64.78/150.00 | 120/665 | PASS |
| managed-stream | 197.03 ± 1.87 | 96.73 | 625.88/625.88/625.88 | 2258.38 | 64.78/150.00 | 120/665 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.542982 | 221.00 | 542.87 | 2395.85 | 93.11 | 216.35 | 13.36 | 42.90 | 2040.00 | 75.64 |
| 2 | 0.529620 | 226.58 | 529.54 | 2386.51 | 93.11 | 201.74 | 14.09 | 43.06 | 2040.00 | 76.27 |
| 3 | 0.557190 | 215.37 | 557.10 | 2348.30 | 93.11 | 201.61 | 12.67 | 43.25 | 2040.00 | 76.33 |
| 4 | 0.547883 | 219.02 | 547.80 | 2350.28 | 93.11 | 197.93 | 13.09 | 42.08 | 2040.00 | 76.64 |
| 5 | 0.536490 | 223.68 | 536.41 | 2350.29 | 93.11 | 203.82 | 11.45 | 42.46 | 2040.00 | 76.82 |
| 6 | 0.531839 | 225.63 | 531.75 | 2349.64 | 93.11 | 202.64 | 14.64 | 44.23 | 2040.00 | 77.73 |
| 7 | 0.533940 | 224.74 | 533.85 | 2349.64 | 93.11 | 202.42 | 9.36 | 43.11 | 2040.00 | 77.45 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.55 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.0964174833740497, 'inference': 1.9629590083695803, 'postprocess': 1.0652891416611965}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

2.31% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.555524 | 216.01 | 56.95 | 2344.32 | 93.11 | 191.36 | 13.83 | 44.37 | 2040.00 | 75.67 |
| 2 | 0.549376 | 218.43 | 61.10 | 2370.40 | 93.11 | 208.20 | 12.55 | 42.91 | 2040.00 | 76.09 |
| 3 | 0.532999 | 225.14 | 56.88 | 2348.30 | 93.11 | 199.80 | 13.18 | 42.98 | 2040.00 | 76.64 |
| 4 | 0.541628 | 221.55 | 61.27 | 2339.61 | 93.11 | 206.83 | 13.36 | 42.22 | 2040.00 | 76.64 |
| 5 | 0.573358 | 209.29 | 59.42 | 2350.29 | 93.11 | 194.60 | 13.33 | 43.42 | 2040.00 | 76.83 |
| 6 | 0.550382 | 218.03 | 63.39 | 2331.46 | 93.11 | 207.91 | 13.09 | 42.85 | 2040.00 | 77.18 |
| 7 | 0.522871 | 229.50 | 56.46 | 2349.64 | 93.11 | 198.44 | 13.82 | 43.55 | 2040.00 | 77.64 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/16.86/17.80.
Separate latency-pass duration: 0.54 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.3339206916346786, 'inference': 1.8684939249851595, 'postprocess': 1.064876616631712}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

10.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.592039 | 202.69 | 591.95 | 2344.37 | 64.78 | 190.25 | 11.92 | 42.57 | 2040.00 | 75.58 |
| 2 | 0.584915 | 205.16 | 584.83 | 2252.14 | 64.78 | 196.93 | 13.00 | 42.50 | 2040.00 | 76.08 |
| 3 | 0.603272 | 198.92 | 603.17 | 2348.29 | 64.78 | 191.99 | 11.00 | 43.30 | 2040.00 | 76.67 |
| 4 | 0.610313 | 196.62 | 610.22 | 2253.46 | 64.78 | 193.47 | 13.15 | 42.22 | 2040.00 | 76.62 |
| 5 | 0.590077 | 203.36 | 589.98 | 2350.16 | 64.78 | 189.93 | 12.75 | 43.95 | 2040.00 | 77.00 |
| 6 | 0.614266 | 195.35 | 614.17 | 2253.68 | 64.78 | 190.03 | 12.54 | 43.03 | 2040.00 | 77.15 |
| 7 | 0.597836 | 200.72 | 597.74 | 2349.52 | 64.78 | 193.84 | 12.67 | 44.23 | 2040.00 | 77.67 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.60 s; not used for throughput.

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

13.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.607551 | 197.51 | 96.72 | 2248.69 | 64.78 | 190.24 | 12.23 | 42.36 | 2040.00 | 75.69 |
| 2 | 0.619635 | 193.66 | 94.08 | 2251.87 | 64.78 | 189.64 | 8.62 | 42.00 | 2040.00 | 75.85 |
| 3 | 0.620476 | 193.40 | 96.73 | 2250.34 | 64.78 | 190.29 | 12.69 | 42.69 | 2040.00 | 76.62 |
| 4 | 0.608657 | 197.16 | 92.68 | 2253.44 | 64.78 | 187.78 | 10.69 | 41.72 | 2040.00 | 76.69 |
| 5 | 0.613252 | 195.68 | 98.02 | 2250.95 | 64.78 | 189.38 | 10.69 | 43.10 | 2040.00 | 76.69 |
| 6 | 0.606134 | 197.98 | 98.56 | 2253.66 | 64.78 | 189.53 | 8.85 | 42.50 | 2040.00 | 76.77 |
| 7 | 0.609051 | 197.03 | 98.15 | 2258.38 | 64.78 | 190.17 | 12.54 | 43.72 | 2040.00 | 77.77 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/18.37/19.12.
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

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 87.42 ± 2.14 | 1372.55 | 1380.32/1380.32/1380.32 | 2556.53 | 64.67/92.00 | 120/659 | PASS |
| raw-stream | 86.99 ± 0.94 | 48.68 | 1429.79/1429.79/1429.79 | 2556.53 | 64.67/92.00 | 120/659 | PASS |
| native-non-stream | 82.38 ± 1.34 | 1456.62 | 1500.69/1500.69/1500.69 | 2556.18 | 64.67/92.00 | 120/659 | PASS |
| native-stream | 82.92 ± 1.21 | 51.52 | 1515.57/1515.57/1515.57 | 2549.64 | 64.67/92.00 | 120/659 | PASS |
| managed-non-stream | 81.53 ± 1.31 | 1471.83 | 1468.42/1468.42/1468.42 | 2248.49 | 56.12/92.00 | 120/659 | PASS |
| managed-stream | 80.12 ± 0.44 | 85.33 | 1492.26/1492.26/1492.26 | 2253.88 | 56.12/92.00 | 120/659 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.402829 | 85.54 | 1402.74 | 2543.83 | 64.67 | 146.11 | 13.86 | 40.36 | 2040.00 | 77.00 |
| 2 | 1.354188 | 88.61 | 1354.10 | 2556.43 | 64.67 | 137.06 | 13.70 | 40.76 | 2040.00 | 76.85 |
| 3 | 1.372637 | 87.42 | 1372.55 | 2556.53 | 64.67 | 136.66 | 12.71 | 40.01 | 2040.00 | 76.14 |
| 4 | 1.351003 | 88.82 | 1350.92 | 2243.84 | 64.67 | 137.27 | 13.74 | 40.65 | 2040.00 | 75.93 |
| 5 | 1.353191 | 88.68 | 1353.11 | 2243.84 | 64.67 | 137.57 | 13.19 | 39.85 | 2040.00 | 75.93 |
| 6 | 1.422788 | 84.34 | 1422.71 | 2244.38 | 64.67 | 135.93 | 13.38 | 40.56 | 2040.00 | 76.00 |
| 7 | 1.430866 | 83.87 | 1430.78 | 2244.38 | 64.67 | 137.69 | 11.90 | 39.37 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.38 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2021670416591708, 'inference': 8.81792389157757, 'postprocess': 1.2488604999058832}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.64% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.356800 | 88.44 | 49.93 | 2549.10 | 64.67 | 137.41 | 13.56 | 40.90 | 2040.00 | 77.00 |
| 2 | 1.378313 | 87.06 | 48.68 | 2556.36 | 64.67 | 137.20 | 13.54 | 40.56 | 2040.00 | 76.89 |
| 3 | 1.381467 | 86.86 | 48.44 | 2556.53 | 64.67 | 137.45 | 14.11 | 40.30 | 2040.00 | 76.00 |
| 4 | 1.370196 | 87.58 | 48.42 | 2243.84 | 64.67 | 137.49 | 12.89 | 40.26 | 2040.00 | 75.89 |
| 5 | 1.379451 | 86.99 | 48.33 | 2243.84 | 64.67 | 137.15 | 13.50 | 40.40 | 2040.00 | 76.00 |
| 6 | 1.381265 | 86.88 | 48.70 | 2244.38 | 64.67 | 136.97 | 12.71 | 40.25 | 2040.00 | 76.00 |
| 7 | 1.406727 | 85.30 | 48.92 | 2244.38 | 64.67 | 137.71 | 13.14 | 40.32 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.39/12.60/14.84.
Separate latency-pass duration: 1.43 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1922370999703464, 'inference': 8.473509374956242, 'postprocess': 1.2189447416555292}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

5.16% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.446916 | 82.94 | 1446.80 | 2549.17 | 64.67 | 136.04 | 14.03 | 41.02 | 2040.00 | 77.00 |
| 2 | 1.456733 | 82.38 | 1456.62 | 2555.46 | 64.67 | 143.68 | 13.21 | 40.55 | 2040.00 | 76.90 |
| 3 | 1.477736 | 81.21 | 1477.65 | 2556.18 | 64.67 | 135.30 | 12.97 | 40.37 | 2040.00 | 76.00 |
| 4 | 1.479390 | 81.11 | 1479.28 | 2243.52 | 64.67 | 135.66 | 13.20 | 40.38 | 2040.00 | 75.87 |
| 5 | 1.423061 | 84.33 | 1422.97 | 2243.84 | 64.67 | 135.79 | 13.93 | 40.63 | 2040.00 | 76.00 |
| 6 | 1.439436 | 83.37 | 1439.34 | 2244.05 | 64.67 | 135.12 | 13.90 | 40.34 | 2040.00 | 76.28 |
| 7 | 1.487165 | 80.69 | 1487.06 | 2244.38 | 64.67 | 135.12 | 13.50 | 40.53 | 2040.00 | 76.00 |

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


### native-stream

4.07% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.459960 | 82.19 | 52.11 | 2549.64 | 64.67 | 136.21 | 12.00 | 40.55 | 2040.00 | 77.00 |
| 2 | 1.447115 | 82.92 | 52.07 | 2244.42 | 64.67 | 136.31 | 12.97 | 40.30 | 2040.00 | 76.93 |
| 3 | 1.402912 | 85.54 | 50.14 | 2243.42 | 64.67 | 134.35 | 12.86 | 39.90 | 2040.00 | 76.00 |
| 4 | 1.462646 | 82.04 | 50.88 | 2240.24 | 64.67 | 135.29 | 12.76 | 39.81 | 2040.00 | 75.93 |
| 5 | 1.426723 | 84.11 | 49.66 | 2243.84 | 64.67 | 134.23 | 12.03 | 40.17 | 2040.00 | 76.00 |
| 6 | 1.449355 | 82.80 | 52.26 | 2240.34 | 64.67 | 135.07 | 13.38 | 40.17 | 2040.00 | 76.07 |
| 7 | 1.442922 | 83.16 | 51.52 | 2244.38 | 64.67 | 134.54 | 11.97 | 39.67 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.05/13.83/14.53.
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

7.01% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.501101 | 79.94 | 1501.01 | 2244.00 | 56.12 | 133.94 | 12.90 | 40.15 | 2040.00 | 77.00 |
| 2 | 1.457858 | 82.31 | 1457.77 | 2246.37 | 56.12 | 134.65 | 12.79 | 40.02 | 2040.00 | 77.00 |
| 3 | 1.458794 | 82.26 | 1458.70 | 2233.06 | 56.12 | 134.02 | 12.79 | 40.09 | 2040.00 | 75.93 |
| 4 | 1.512674 | 79.33 | 1512.58 | 2248.40 | 56.12 | 134.27 | 12.30 | 39.78 | 2040.00 | 75.90 |
| 5 | 1.471923 | 81.53 | 1471.83 | 2233.30 | 56.12 | 135.10 | 12.73 | 40.03 | 2040.00 | 76.00 |
| 6 | 1.447058 | 82.93 | 1446.97 | 2248.49 | 56.12 | 134.91 | 12.83 | 39.95 | 2040.00 | 76.21 |
| 7 | 1.472211 | 81.51 | 1472.12 | 2233.83 | 56.12 | 134.44 | 13.00 | 40.09 | 2040.00 | 76.00 |

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


### managed-stream

7.81% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.494825 | 80.28 | 85.02 | 2250.89 | 56.12 | 134.56 | 13.23 | 40.25 | 2040.00 | 77.00 |
| 2 | 1.498424 | 80.08 | 87.31 | 2253.88 | 56.12 | 134.63 | 11.73 | 39.61 | 2040.00 | 77.00 |
| 3 | 1.479775 | 81.09 | 85.33 | 2245.38 | 56.12 | 135.04 | 13.00 | 39.53 | 2040.00 | 75.90 |
| 4 | 1.491182 | 80.47 | 83.06 | 2248.38 | 56.12 | 134.70 | 11.73 | 39.30 | 2040.00 | 75.87 |
| 5 | 1.506142 | 79.67 | 85.72 | 2245.68 | 56.12 | 135.53 | 11.93 | 39.81 | 2040.00 | 76.13 |
| 6 | 1.498965 | 80.06 | 86.12 | 2248.48 | 56.12 | 134.74 | 11.67 | 39.41 | 2040.00 | 76.00 |
| 7 | 1.497785 | 80.12 | 85.21 | 2245.84 | 56.12 | 134.06 | 13.00 | 39.82 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.65/12.63/14.80.
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

## Detail — yolo11, video, batch 4

Status: **complete**. Load 0.11 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 197.33 ± 4.88 | 608.04 | 594.54/594.54/594.54 | 2804.52 | 126.55/188.00 | 120/659 | PASS |
| raw-stream | 198.40 ± 2.41 | 61.69 | 602.35/602.35/602.35 | 2530.40 | 126.55/188.00 | 120/659 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 182.84 ± 2.04 | 656.22 | 677.79/677.79/677.79 | 2530.41 | 99.31/188.00 | 120/659 | PASS |
| managed-stream | 176.70 ± 2.14 | 96.87 | 678.12/678.12/678.12 | 2530.34 | 99.31/188.00 | 120/659 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.634132 | 189.24 | 634.01 | 2804.52 | 126.55 | 198.76 | 12.85 | 43.24 | 2040.00 | 76.69 |
| 2 | 0.608120 | 197.33 | 608.04 | 2539.45 | 126.55 | 195.26 | 13.15 | 43.56 | 2040.00 | 76.77 |
| 3 | 0.608124 | 197.33 | 608.02 | 2501.01 | 126.55 | 192.78 | 13.46 | 42.49 | 2040.00 | 76.69 |
| 4 | 0.604802 | 198.41 | 604.71 | 2501.37 | 126.55 | 193.63 | 14.17 | 43.36 | 2040.00 | 77.42 |
| 5 | 0.629693 | 190.57 | 629.61 | 2502.70 | 126.55 | 192.48 | 12.08 | 41.91 | 2040.00 | 77.23 |
| 6 | 0.590037 | 203.38 | 589.94 | 2500.57 | 126.55 | 189.51 | 13.83 | 43.68 | 2040.00 | 77.67 |
| 7 | 0.618919 | 193.89 | 618.83 | 2503.20 | 126.55 | 195.17 | 13.54 | 42.55 | 2040.00 | 77.69 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1942354666947115, 'inference': 2.509328283273741, 'postprocess': 1.090636341617331}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.65% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.613621 | 195.56 | 61.16 | 2494.79 | 126.55 | 186.41 | 13.08 | 43.40 | 2040.00 | 76.69 |
| 2 | 0.604843 | 198.40 | 61.27 | 2526.96 | 126.55 | 192.18 | 12.83 | 42.81 | 2040.00 | 76.67 |
| 3 | 0.601341 | 199.55 | 61.48 | 2498.43 | 126.55 | 189.49 | 12.67 | 42.98 | 2040.00 | 76.83 |
| 4 | 0.594831 | 201.74 | 61.69 | 2490.81 | 126.55 | 189.91 | 15.00 | 41.82 | 2040.00 | 77.50 |
| 5 | 0.593030 | 202.35 | 65.14 | 2503.03 | 126.55 | 197.03 | 14.00 | 42.89 | 2040.00 | 77.50 |
| 6 | 0.607333 | 197.59 | 62.45 | 2530.40 | 126.55 | 188.69 | 13.77 | 41.97 | 2040.00 | 77.69 |
| 7 | 0.606903 | 197.73 | 67.29 | 2503.20 | 126.55 | 194.92 | 13.54 | 43.33 | 2040.00 | 77.85 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/19.00/19.98.
Separate latency-pass duration: 0.60 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2362729500485632, 'inference': 2.4429541167592106, 'postprocess': 1.062639091727154}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

8.52% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.640395 | 187.38 | 640.29 | 2494.68 | 99.31 | 180.63 | 13.77 | 43.14 | 2040.00 | 76.69 |
| 2 | 0.660655 | 181.64 | 660.57 | 2465.73 | 99.31 | 182.07 | 12.79 | 43.11 | 2040.00 | 76.71 |
| 3 | 0.659921 | 181.84 | 659.82 | 2501.03 | 99.31 | 182.06 | 13.43 | 43.10 | 2040.00 | 76.86 |
| 4 | 0.656318 | 182.84 | 656.22 | 2473.06 | 99.31 | 182.94 | 13.00 | 43.27 | 2040.00 | 76.86 |
| 5 | 0.653116 | 183.73 | 653.02 | 2500.33 | 99.31 | 185.49 | 13.08 | 43.21 | 2040.00 | 77.62 |
| 6 | 0.650958 | 184.34 | 650.87 | 2530.41 | 99.31 | 180.84 | 12.46 | 41.88 | 2040.00 | 77.69 |
| 7 | 0.659687 | 181.90 | 659.59 | 2500.47 | 99.31 | 180.04 | 13.29 | 44.17 | 2040.00 | 77.86 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
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


### managed-stream

11.88% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.667674 | 179.73 | 96.87 | 2462.28 | 99.31 | 187.27 | 13.29 | 42.39 | 2040.00 | 76.50 |
| 2 | 0.684213 | 175.38 | 94.76 | 2465.46 | 99.31 | 173.93 | 12.93 | 41.66 | 2040.00 | 76.57 |
| 3 | 0.692875 | 173.19 | 99.42 | 2470.08 | 99.31 | 181.59 | 12.43 | 42.01 | 2040.00 | 76.86 |
| 4 | 0.676677 | 177.34 | 95.20 | 2473.01 | 99.31 | 179.59 | 12.14 | 42.17 | 2040.00 | 76.71 |
| 5 | 0.679108 | 176.70 | 98.17 | 2527.62 | 99.31 | 184.51 | 12.36 | 42.87 | 2040.00 | 77.64 |
| 6 | 0.677115 | 177.22 | 93.80 | 2530.34 | 99.31 | 182.47 | 12.36 | 42.63 | 2040.00 | 77.64 |
| 7 | 0.687419 | 174.57 | 97.46 | 2527.80 | 99.31 | 184.35 | 12.71 | 43.67 | 2040.00 | 77.86 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/20.13/20.56.
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
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 78.80 ± 1.21 | 1522.83 | 1576.99/1576.99/1576.99 | 2483.12 | 62.86/90.00 | 120/642 | PASS |
| raw-stream | 78.78 ± 0.86 | 49.74 | 1537.75/1537.75/1537.75 | 2492.81 | 62.86/90.00 | 120/642 | PASS |
| native-non-stream | 75.72 ± 0.95 | 1584.78 | 1590.99/1590.99/1590.99 | 2483.12 | 62.86/90.00 | 120/642 | PASS |
| native-stream | 74.03 ± 0.83 | 52.98 | 1638.86/1638.86/1638.86 | 2483.25 | 62.80/90.00 | 120/642 | PASS |
| managed-non-stream | 74.63 ± 1.09 | 1607.88 | 1623.32/1623.32/1623.32 | 2182.78 | 55.34/90.00 | 120/642 | PASS |
| managed-stream | 73.00 ± 1.38 | 86.01 | 1626.05/1626.05/1626.05 | 2182.80 | 55.34/90.00 | 120/642 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.513950 | 79.26 | 1513.86 | 2472.98 | 62.86 | 141.43 | 13.16 | 39.79 | 2040.00 | 75.90 |
| 2 | 1.522909 | 78.80 | 1522.83 | 2480.58 | 62.86 | 133.71 | 13.23 | 40.02 | 2040.00 | 76.90 |
| 3 | 1.575130 | 76.18 | 1575.03 | 2480.66 | 62.86 | 134.19 | 12.34 | 39.45 | 2040.00 | 76.91 |
| 4 | 1.526353 | 78.62 | 1526.26 | 2481.29 | 62.86 | 134.19 | 12.81 | 40.01 | 2040.00 | 76.87 |
| 5 | 1.547213 | 77.56 | 1547.12 | 2481.29 | 62.86 | 132.86 | 11.90 | 39.60 | 2040.00 | 76.87 |
| 6 | 1.507584 | 79.60 | 1507.48 | 2483.11 | 62.86 | 133.61 | 13.27 | 40.16 | 2040.00 | 76.87 |
| 7 | 1.512645 | 79.33 | 1512.56 | 2483.12 | 62.86 | 133.95 | 12.42 | 39.58 | 2040.00 | 76.87 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1794804998771724, 'inference': 9.766075958411117, 'postprocess': 1.2506669833176904}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.20% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.519813 | 78.96 | 50.08 | 2473.32 | 62.86 | 134.19 | 13.13 | 39.75 | 2040.00 | 75.87 |
| 2 | 1.538647 | 77.99 | 49.12 | 2480.57 | 62.86 | 133.94 | 13.06 | 40.05 | 2040.00 | 76.90 |
| 3 | 1.538334 | 78.01 | 66.46 | 2492.81 | 62.86 | 141.14 | 12.81 | 40.10 | 2040.00 | 76.84 |
| 4 | 1.498851 | 80.06 | 50.60 | 2481.22 | 62.86 | 133.78 | 13.00 | 39.74 | 2040.00 | 76.83 |
| 5 | 1.543583 | 77.74 | 49.72 | 2481.29 | 62.86 | 135.37 | 12.81 | 40.06 | 2040.00 | 76.87 |
| 6 | 1.523288 | 78.78 | 49.74 | 2483.05 | 62.86 | 134.23 | 13.06 | 39.99 | 2040.00 | 76.87 |
| 7 | 1.509683 | 79.49 | 49.20 | 2483.12 | 62.86 | 134.40 | 13.10 | 39.95 | 2040.00 | 76.83 |

Ready result delivery-interval P50/P95/P99 ms: 12.36/13.59/15.13.
Separate latency-pass duration: 1.54 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2149139748544258, 'inference': 9.774063566707506, 'postprocess': 1.2578302751080628}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

4.07% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.566900 | 76.58 | 1566.78 | 2473.33 | 62.86 | 132.04 | 13.62 | 39.74 | 2040.00 | 75.94 |
| 2 | 1.584857 | 75.72 | 1584.78 | 2479.70 | 62.86 | 140.06 | 13.38 | 39.87 | 2040.00 | 76.88 |
| 3 | 1.559370 | 76.95 | 1559.29 | 2482.20 | 62.86 | 132.01 | 13.32 | 40.14 | 2040.00 | 76.84 |
| 4 | 1.607454 | 74.65 | 1607.32 | 2480.87 | 62.86 | 139.34 | 13.38 | 39.73 | 2040.00 | 76.75 |
| 5 | 1.604975 | 74.77 | 1604.86 | 2481.29 | 62.86 | 132.07 | 13.03 | 40.13 | 2040.00 | 76.94 |
| 6 | 1.607731 | 74.64 | 1607.63 | 2482.70 | 62.86 | 139.63 | 13.09 | 39.99 | 2040.00 | 77.00 |
| 7 | 1.582684 | 75.82 | 1582.55 | 2483.12 | 62.86 | 132.85 | 13.12 | 40.03 | 2040.00 | 76.00 |

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

5.34% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.630354 | 73.60 | 51.51 | 2473.80 | 62.80 | 131.96 | 12.30 | 39.39 | 2040.00 | 76.00 |
| 2 | 1.578855 | 76.00 | 57.03 | 2173.17 | 62.80 | 132.12 | 12.75 | 39.78 | 2040.00 | 76.88 |
| 3 | 1.623123 | 73.93 | 52.62 | 2482.59 | 62.80 | 131.46 | 12.45 | 39.48 | 2040.00 | 76.55 |
| 4 | 1.607916 | 74.63 | 52.98 | 2172.71 | 62.80 | 132.48 | 12.38 | 39.55 | 2040.00 | 76.09 |
| 5 | 1.628858 | 73.67 | 53.69 | 2481.29 | 62.80 | 132.87 | 12.09 | 39.63 | 2040.00 | 76.94 |
| 6 | 1.617818 | 74.17 | 53.54 | 2182.41 | 62.80 | 133.09 | 12.42 | 39.91 | 2040.00 | 77.00 |
| 7 | 1.620944 | 74.03 | 51.65 | 2483.25 | 62.80 | 133.93 | 12.00 | 39.24 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.18/13.98/15.36.
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

6.06% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.605626 | 74.74 | 1605.54 | 2168.21 | 55.34 | 131.91 | 12.62 | 39.38 | 2040.00 | 76.00 |
| 2 | 1.591678 | 75.39 | 1591.59 | 2171.03 | 55.34 | 132.16 | 12.75 | 39.50 | 2040.00 | 76.47 |
| 3 | 1.644205 | 72.98 | 1644.11 | 2173.87 | 55.34 | 131.39 | 12.42 | 39.46 | 2040.00 | 76.06 |
| 4 | 1.654412 | 72.53 | 1654.31 | 2180.30 | 55.34 | 132.15 | 12.27 | 39.37 | 2040.00 | 76.00 |
| 5 | 1.596875 | 75.15 | 1596.78 | 2175.34 | 55.34 | 131.55 | 12.75 | 39.67 | 2040.00 | 77.00 |
| 6 | 1.607969 | 74.63 | 1607.88 | 2182.78 | 55.34 | 130.59 | 12.75 | 39.60 | 2040.00 | 77.00 |
| 7 | 1.622086 | 73.98 | 1621.98 | 2177.28 | 55.34 | 131.99 | 12.30 | 39.34 | 2040.00 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.04.
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

8.68% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.669120 | 71.89 | 90.54 | 2174.43 | 55.34 | 133.69 | 12.21 | 39.12 | 2040.00 | 76.00 |
| 2 | 1.609830 | 74.54 | 84.37 | 2177.35 | 55.34 | 132.11 | 11.75 | 39.16 | 2040.00 | 76.00 |
| 3 | 1.620114 | 74.07 | 85.73 | 2177.35 | 55.34 | 131.05 | 12.55 | 39.19 | 2040.00 | 76.00 |
| 4 | 1.676472 | 71.58 | 84.24 | 2180.29 | 55.34 | 131.37 | 11.18 | 39.05 | 2040.00 | 76.00 |
| 5 | 1.627295 | 73.74 | 86.64 | 2179.83 | 55.34 | 131.83 | 12.48 | 39.79 | 2040.00 | 77.00 |
| 6 | 1.691951 | 70.92 | 88.04 | 2182.80 | 55.34 | 131.52 | 11.29 | 39.22 | 2040.00 | 77.00 |
| 7 | 1.643938 | 73.00 | 86.01 | 2180.50 | 55.34 | 131.59 | 12.09 | 39.33 | 2040.00 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 12.77/14.08/14.87.
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

## Detail — yolo26, video, batch 4

Status: **complete**. Load 0.12 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 185.28 ± 2.28 | 647.58 | 652.06/652.06/652.06 | 2523.72 | 119.89/162.00 | 120/643 | PASS |
| raw-stream | 184.57 ± 2.41 | 61.61 | 632.82/632.82/632.82 | 2515.82 | 119.89/162.00 | 120/643 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 171.28 ± 1.47 | 700.51 | 691.35/691.35/691.35 | 2513.54 | 98.52/162.00 | 120/643 | PASS |
| managed-stream | 164.15 ± 5.96 | 100.34 | 752.99/752.99/752.99 | 2513.51 | 98.52/162.00 | 120/643 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.638813 | 187.85 | 638.73 | 2475.91 | 119.89 | 194.20 | 13.31 | 42.62 | 2040.00 | 76.23 |
| 2 | 0.651627 | 184.15 | 651.54 | 2520.40 | 119.89 | 182.34 | 14.08 | 43.08 | 2040.00 | 76.69 |
| 3 | 0.647666 | 185.28 | 647.58 | 2481.30 | 119.89 | 187.29 | 12.00 | 41.87 | 2040.00 | 76.62 |
| 4 | 0.634083 | 189.25 | 633.97 | 2523.72 | 119.89 | 185.18 | 12.92 | 41.89 | 2040.00 | 77.00 |
| 5 | 0.655212 | 183.15 | 655.13 | 2485.14 | 119.89 | 183.60 | 12.46 | 42.68 | 2040.00 | 76.85 |
| 6 | 0.653504 | 183.63 | 653.42 | 2485.08 | 119.89 | 182.46 | 12.85 | 42.54 | 2040.00 | 77.69 |
| 7 | 0.643748 | 186.41 | 643.65 | 2484.91 | 119.89 | 185.58 | 10.92 | 42.22 | 2040.00 | 77.62 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.65 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.0991962583072261, 'inference': 2.8150466249826422, 'postprocess': 1.049162783359255}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.23% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.639581 | 187.62 | 60.42 | 2477.70 | 119.89 | 183.24 | 13.23 | 41.83 | 2040.00 | 76.15 |
| 2 | 0.648868 | 184.94 | 60.74 | 2509.84 | 119.89 | 187.10 | 14.31 | 42.72 | 2040.00 | 76.62 |
| 3 | 0.663752 | 180.79 | 64.29 | 2484.36 | 119.89 | 184.22 | 12.79 | 43.87 | 2040.00 | 76.71 |
| 4 | 0.642631 | 186.73 | 60.22 | 2515.82 | 119.89 | 185.13 | 12.46 | 41.78 | 2040.00 | 76.85 |
| 5 | 0.656707 | 182.73 | 62.74 | 2484.05 | 119.89 | 184.43 | 14.43 | 42.75 | 2040.00 | 77.21 |
| 6 | 0.650159 | 184.57 | 61.61 | 2474.53 | 119.89 | 183.47 | 12.77 | 42.97 | 2040.00 | 77.69 |
| 7 | 0.656921 | 182.67 | 64.29 | 2485.25 | 119.89 | 185.31 | 13.57 | 43.68 | 2040.00 | 77.71 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/19.74/21.92.
Separate latency-pass duration: 0.63 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1517231333376305, 'inference': 2.7695569167159797, 'postprocess': 1.0488260583467006}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

8.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.711035 | 168.77 | 710.94 | 2477.71 | 98.52 | 174.59 | 13.27 | 41.12 | 2040.00 | 76.47 |
| 2 | 0.712143 | 168.51 | 712.05 | 2448.76 | 98.52 | 173.32 | 12.27 | 42.07 | 2040.00 | 76.60 |
| 3 | 0.698135 | 171.89 | 698.04 | 2481.61 | 98.52 | 178.18 | 12.36 | 43.31 | 2040.00 | 76.71 |
| 4 | 0.697330 | 172.08 | 697.24 | 2513.54 | 98.52 | 176.82 | 11.86 | 42.12 | 2040.00 | 76.86 |
| 5 | 0.699772 | 171.48 | 699.65 | 2483.93 | 98.52 | 176.71 | 13.14 | 43.09 | 2040.00 | 77.29 |
| 6 | 0.700604 | 171.28 | 700.51 | 2455.94 | 98.52 | 175.92 | 12.43 | 42.61 | 2040.00 | 77.36 |
| 7 | 0.701441 | 171.08 | 701.35 | 2485.12 | 98.52 | 178.16 | 12.29 | 41.89 | 2040.00 | 77.71 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.69 s; not used for throughput.

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

13.22% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.731540 | 164.04 | 100.34 | 2445.30 | 98.52 | 176.42 | 12.33 | 42.47 | 2040.00 | 76.33 |
| 2 | 0.714280 | 168.00 | 96.21 | 2448.50 | 98.52 | 177.49 | 11.07 | 41.58 | 2040.00 | 76.47 |
| 3 | 0.798606 | 150.26 | 105.98 | 2510.41 | 98.52 | 176.85 | 10.25 | 41.82 | 2040.00 | 76.75 |
| 4 | 0.731022 | 164.15 | 96.41 | 2513.51 | 98.52 | 175.36 | 10.40 | 41.60 | 2040.00 | 76.60 |
| 5 | 0.719114 | 166.87 | 100.45 | 2453.07 | 98.52 | 176.85 | 12.47 | 42.65 | 2040.00 | 77.33 |
| 6 | 0.739927 | 162.18 | 94.93 | 2455.89 | 98.52 | 173.48 | 10.40 | 42.44 | 2040.00 | 77.40 |
| 7 | 0.723832 | 165.78 | 101.36 | 2452.90 | 98.52 | 181.51 | 12.87 | 42.14 | 2040.00 | 77.73 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/22.94/25.53.
Separate latency-pass duration: 0.75 s; not used for throughput.

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

Status: **complete**. Load 5.85 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 77.43 ± 1.27 | 1549.62 | 1533.16/1533.16/1533.16 | 2751.43 | 115.88/134.00 | 120/708 | PASS |
| raw-stream | 77.14 ± 1.35 | 49.80 | 1565.00/1565.00/1565.00 | 2748.27 | 115.88/134.00 | 120/708 | PASS |
| native-non-stream | 74.56 ± 0.26 | 1609.33 | 1660.01/1660.01/1660.01 | 2748.27 | 115.88/134.00 | 120/708 | PASS |
| native-stream | 72.74 ± 1.12 | 51.86 | 1659.24/1659.24/1659.24 | 2748.27 | 115.88/134.00 | 120/708 | PASS |
| managed-non-stream | 72.49 ± 1.07 | 1655.42 | 1640.47/1640.47/1640.47 | 2438.95 | 107.97/134.00 | 120/708 | PASS |
| managed-stream | 72.45 ± 1.14 | 85.26 | 1671.02/1671.02/1671.02 | 2438.95 | 107.97/134.00 | 120/708 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.539525 | 77.95 | 1539.44 | 2716.36 | 115.88 | 140.49 | 19.03 | 46.14 | 2040.00 | 77.00 |
| 2 | 1.549693 | 77.43 | 1549.62 | 2739.15 | 115.88 | 133.07 | 18.84 | 45.83 | 2040.00 | 77.26 |
| 3 | 1.541658 | 77.84 | 1541.58 | 2751.43 | 115.88 | 140.80 | 17.68 | 45.45 | 2040.00 | 76.87 |
| 4 | 1.543645 | 77.74 | 1543.56 | 2747.80 | 115.88 | 134.13 | 19.03 | 46.13 | 2040.00 | 76.77 |
| 5 | 1.599264 | 75.03 | 1599.18 | 2747.80 | 115.88 | 134.19 | 18.12 | 44.39 | 2040.00 | 76.62 |
| 6 | 1.555924 | 77.12 | 1555.83 | 2748.27 | 115.88 | 132.49 | 18.97 | 45.25 | 2040.00 | 75.90 |
| 7 | 1.597818 | 75.10 | 1597.73 | 2748.27 | 115.88 | 132.63 | 16.94 | 44.70 | 2040.00 | 75.88 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1858601666517643, 'inference': 9.985675983307374, 'postprocess': 1.2495094999849243}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.91% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.591494 | 75.40 | 50.13 | 2719.05 | 115.88 | 131.99 | 18.59 | 46.00 | 2040.00 | 77.50 |
| 2 | 1.553915 | 77.22 | 49.37 | 2739.09 | 115.88 | 133.61 | 18.32 | 45.91 | 2040.00 | 77.84 |
| 3 | 1.527688 | 78.55 | 50.37 | 2746.08 | 115.88 | 133.30 | 19.29 | 46.12 | 2040.00 | 76.94 |
| 4 | 1.517349 | 79.09 | 49.24 | 2747.80 | 115.88 | 133.51 | 18.71 | 45.07 | 2040.00 | 76.74 |
| 5 | 1.577218 | 76.08 | 49.59 | 2747.93 | 115.88 | 133.31 | 18.47 | 45.40 | 2040.00 | 76.38 |
| 6 | 1.555661 | 77.14 | 49.80 | 2748.27 | 115.88 | 132.74 | 18.42 | 45.15 | 2040.00 | 75.90 |
| 7 | 1.576204 | 76.13 | 52.60 | 2748.27 | 115.88 | 134.12 | 18.62 | 45.47 | 2040.00 | 75.94 |

Ready result delivery-interval P50/P95/P99 ms: 12.48/14.20/15.12.
Separate latency-pass duration: 1.57 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.2636062917408708, 'inference': 10.261274891627181, 'postprocess': 1.3146393667132845}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.77% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.605599 | 74.74 | 1605.51 | 2719.12 | 115.88 | 131.37 | 18.75 | 45.84 | 2040.00 | 77.88 |
| 2 | 1.608191 | 74.62 | 1608.08 | 2738.47 | 115.88 | 139.07 | 18.88 | 45.83 | 2040.00 | 77.88 |
| 3 | 1.609447 | 74.56 | 1609.33 | 2746.41 | 115.88 | 131.84 | 18.75 | 46.21 | 2040.00 | 76.94 |
| 4 | 1.610120 | 74.53 | 1610.04 | 2747.45 | 115.88 | 140.47 | 18.31 | 44.98 | 2040.00 | 76.75 |
| 5 | 1.609634 | 74.55 | 1609.55 | 2747.93 | 115.88 | 131.96 | 19.12 | 45.30 | 2040.00 | 76.31 |
| 6 | 1.607569 | 74.65 | 1607.48 | 2747.91 | 115.88 | 139.92 | 19.00 | 45.27 | 2040.00 | 75.94 |
| 7 | 1.622663 | 73.95 | 1622.55 | 2748.27 | 115.88 | 132.86 | 18.82 | 45.31 | 2040.00 | 75.94 |

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


### native-stream

5.14% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.688321 | 71.08 | 53.73 | 2719.57 | 115.88 | 132.07 | 18.29 | 45.41 | 2040.00 | 77.88 |
| 2 | 1.649671 | 72.74 | 51.54 | 2432.29 | 115.88 | 131.80 | 18.09 | 45.51 | 2040.00 | 77.94 |
| 3 | 1.656824 | 72.43 | 51.21 | 2746.41 | 115.88 | 129.93 | 17.39 | 45.58 | 2040.00 | 76.91 |
| 4 | 1.622954 | 73.94 | 50.97 | 2438.62 | 115.88 | 131.07 | 18.09 | 45.34 | 2040.00 | 76.42 |
| 5 | 1.643087 | 73.03 | 51.89 | 2747.93 | 115.88 | 130.26 | 17.73 | 45.01 | 2040.00 | 75.91 |
| 6 | 1.616996 | 74.21 | 51.86 | 2439.12 | 115.88 | 130.80 | 18.21 | 45.16 | 2040.00 | 75.88 |
| 7 | 1.672281 | 71.76 | 52.81 | 2748.27 | 115.88 | 131.75 | 17.35 | 44.67 | 2040.00 | 75.91 |

Ready result delivery-interval P50/P95/P99 ms: 13.10/14.92/18.56.
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

5.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.684706 | 71.23 | 1684.62 | 2413.97 | 107.97 | 131.23 | 16.79 | 44.81 | 2040.00 | 77.88 |
| 2 | 1.655508 | 72.49 | 1655.42 | 2432.06 | 107.97 | 131.19 | 17.85 | 45.30 | 2040.00 | 77.88 |
| 3 | 1.649202 | 72.76 | 1649.11 | 2437.76 | 107.97 | 132.20 | 17.45 | 44.76 | 2040.00 | 76.85 |
| 4 | 1.624073 | 73.89 | 1623.97 | 2438.51 | 107.97 | 131.75 | 18.21 | 45.09 | 2040.00 | 76.52 |
| 5 | 1.666807 | 71.99 | 1666.69 | 2438.77 | 107.97 | 131.79 | 17.26 | 44.87 | 2040.00 | 75.91 |
| 6 | 1.614952 | 74.31 | 1614.86 | 2438.91 | 107.97 | 131.23 | 18.00 | 44.30 | 2040.00 | 75.85 |
| 7 | 1.661605 | 72.22 | 1661.47 | 2438.95 | 107.97 | 131.98 | 17.00 | 44.43 | 2040.00 | 75.88 |

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


### managed-stream

6.85% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.686124 | 71.17 | 85.36 | 2428.64 | 107.97 | 130.04 | 17.03 | 44.94 | 2040.00 | 77.91 |
| 2 | 1.640599 | 73.14 | 84.98 | 2431.63 | 107.97 | 130.53 | 16.64 | 44.46 | 2040.00 | 77.88 |
| 3 | 1.647204 | 72.85 | 85.06 | 2438.16 | 107.97 | 132.27 | 17.33 | 44.34 | 2040.00 | 76.79 |
| 4 | 1.673658 | 71.70 | 84.58 | 2438.50 | 107.97 | 131.51 | 17.32 | 43.66 | 2040.00 | 76.35 |
| 5 | 1.654317 | 72.54 | 86.38 | 2438.75 | 107.97 | 131.94 | 17.33 | 44.07 | 2040.00 | 75.91 |
| 6 | 1.717125 | 69.88 | 88.27 | 2438.89 | 107.97 | 131.92 | 16.91 | 43.87 | 2040.00 | 75.89 |
| 7 | 1.656426 | 72.45 | 85.26 | 2438.95 | 107.97 | 131.27 | 17.12 | 44.85 | 2040.00 | 75.88 |

Ready result delivery-interval P50/P95/P99 ms: 13.15/14.36/15.03.
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

## Detail — yolo-world, video, batch 4

Status: **complete**. Load 5.83 s. Device `cuda:0`, dtype `torch.float32`; observed after complete warmup, before measured repeats.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `0ebb1f3d2122f1d3bfc0ae97df9e096da354751fe1ceb71e00f68d230c77f27f`.
Runner SHA256: `5b86db85c01dcf1c652380b156685fd4576db2422344080c492a39ede16a312f`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 185.90 ± 5.34 | 645.41 | 651.76/651.76/651.76 | 2658.64 | 209.49/292.00 | 120/709 | PASS |
| raw-stream | 188.09 ± 1.97 | 61.86 | 659.63/659.63/659.63 | 2658.97 | 209.49/292.00 | 120/709 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 174.96 ± 3.26 | 685.76 | 692.87/692.87/692.87 | 2658.44 | 188.26/292.00 | 120/709 | PASS |
| managed-stream | 168.96 ± 3.39 | 99.38 | 711.80/711.80/711.80 | 2623.86 | 188.26/292.00 | 120/709 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.666543 | 180.03 | 666.45 | 2650.80 | 209.49 | 195.01 | 24.00 | 53.12 | 2040.00 | 76.57 |
| 2 | 0.639229 | 187.73 | 639.14 | 2658.12 | 209.49 | 186.54 | 24.92 | 53.19 | 2040.00 | 77.69 |
| 3 | 0.645494 | 185.90 | 645.41 | 2658.27 | 209.49 | 191.27 | 24.23 | 48.14 | 2040.00 | 77.62 |
| 4 | 0.646430 | 185.63 | 646.34 | 2657.80 | 209.49 | 186.42 | 25.00 | 54.81 | 2040.00 | 78.69 |
| 5 | 0.672128 | 178.54 | 672.04 | 2658.64 | 209.49 | 189.26 | 24.29 | 53.24 | 2040.00 | 79.00 |
| 6 | 0.615938 | 194.82 | 615.85 | 2657.73 | 209.49 | 189.89 | 26.54 | 53.28 | 2040.00 | 79.62 |
| 7 | 0.644279 | 186.25 | 644.19 | 2657.73 | 209.49 | 191.62 | 24.69 | 52.87 | 2040.00 | 79.31 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.65 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1813982415939488, 'inference': 2.8976236666494515, 'postprocess': 1.1149287749958603}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.40% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.637978 | 188.09 | 60.70 | 2651.13 | 209.49 | 182.33 | 25.00 | 53.44 | 2040.00 | 76.85 |
| 2 | 0.643280 | 186.54 | 65.23 | 2648.63 | 209.49 | 188.61 | 27.23 | 54.18 | 2040.00 | 77.92 |
| 3 | 0.642916 | 186.65 | 61.90 | 2656.11 | 209.49 | 184.29 | 25.38 | 49.47 | 2040.00 | 78.08 |
| 4 | 0.653462 | 183.64 | 61.40 | 2647.25 | 209.49 | 186.74 | 24.77 | 54.06 | 2040.00 | 78.77 |
| 5 | 0.636133 | 188.64 | 67.48 | 2658.97 | 209.49 | 187.10 | 26.00 | 54.20 | 2040.00 | 79.31 |
| 6 | 0.633919 | 189.30 | 61.86 | 2657.40 | 209.49 | 190.00 | 24.92 | 52.45 | 2040.00 | 79.46 |
| 7 | 0.635404 | 188.86 | 60.63 | 2657.73 | 209.49 | 188.16 | 25.46 | 55.39 | 2040.00 | 79.54 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/20.57/21.74.
Separate latency-pass duration: 0.66 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.1343851166505676, 'inference': 2.7499815499898737, 'postprocess': 1.0616311333099777}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

5.71% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.695781 | 172.47 | 695.69 | 2651.17 | 188.26 | 183.21 | 24.79 | 52.12 | 2040.00 | 76.93 |
| 2 | 0.674467 | 177.92 | 674.37 | 2622.24 | 188.26 | 180.47 | 25.50 | 51.76 | 2040.00 | 77.29 |
| 3 | 0.685852 | 174.96 | 685.76 | 2656.09 | 188.26 | 178.25 | 24.86 | 52.01 | 2040.00 | 78.29 |
| 4 | 0.714123 | 168.04 | 714.03 | 2624.04 | 188.26 | 175.54 | 22.87 | 51.73 | 2040.00 | 78.47 |
| 5 | 0.692918 | 173.18 | 692.79 | 2658.44 | 188.26 | 177.33 | 24.43 | 54.90 | 2040.00 | 79.21 |
| 6 | 0.681715 | 176.03 | 681.61 | 2571.82 | 188.26 | 179.61 | 23.57 | 51.15 | 2040.00 | 79.43 |
| 7 | 0.681064 | 176.20 | 680.97 | 2657.63 | 188.26 | 181.03 | 25.36 | 55.22 | 2040.00 | 79.43 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.69 s; not used for throughput.

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

11.04% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.690217 | 173.86 | 100.58 | 2618.71 | 188.26 | 177.50 | 23.43 | 50.41 | 2040.00 | 77.29 |
| 2 | 0.716117 | 167.57 | 95.20 | 2621.73 | 188.26 | 177.93 | 23.47 | 48.63 | 2040.00 | 77.20 |
| 3 | 0.716775 | 167.42 | 99.65 | 2620.92 | 188.26 | 174.87 | 22.27 | 51.99 | 2040.00 | 78.33 |
| 4 | 0.733919 | 163.51 | 99.38 | 2623.86 | 188.26 | 178.57 | 22.33 | 51.40 | 2040.00 | 78.40 |
| 5 | 0.710214 | 168.96 | 98.97 | 2569.06 | 188.26 | 175.29 | 25.40 | 52.53 | 2040.00 | 79.07 |
| 6 | 0.703320 | 170.62 | 95.53 | 2571.76 | 188.26 | 180.43 | 22.00 | 51.52 | 2040.00 | 79.14 |
| 7 | 0.698335 | 171.84 | 101.60 | 2623.35 | 188.26 | 181.02 | 23.43 | 52.99 | 2040.00 | 79.29 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/21.41/22.90.
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

## Ten-minute managed-stream stability

GPU: NVIDIA L4; model yolov8n.pt; elapsed 600.35 s; frames 49800.
Pass: **True**. Exceptions: `[]`. Failed gates: `[]`.
Each loop verifies frame count and order, consumes without retaining frames, closes the stream and collects unreachable Python objects. The model remains loaded; allocator caches are not emptied between loops.

| Metric | First 60-second median | Last 60-second median | Change % |
| --- | ---: | ---: | ---: |
| rss_mib | 1705.32 | 1707.35 | 0.12 |
| vram_allocated_mib | 12.10 | 12.10 | 0.00 |
| fps | 92.48 | 91.11 | -1.49 |
| open_fds | 44.00 | 44.00 | 0.00 |

Whole-check throughput including between-loop cleanup: 82.95 FPS. Per-loop FPS above excludes between-loop cleanup.

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
