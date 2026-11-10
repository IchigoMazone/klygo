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

Tested commit(s): `cffb4769b93947f5bbe97d27d9b6c4ebcfcbe9c2`

Additional hardware/runtime details: `{'nvidia_smi': 'Tesla T4, GPU-2b70995d-8000-03e8-1644-f2693fa0f841, 580.82.07, 15360 MiB, 70.00 W', 'lscpu': 'Architecture:                            x86_64\nCPU op-mode(s):                          32-bit, 64-bit\nAddress sizes:                           46 bits physical, 48 bits virtual\nByte Order:                              Little Endian\nCPU(s):                                  2\nOn-line CPU(s) list:                     0,1\nVendor ID:                               GenuineIntel\nModel name:                              Intel(R) Xeon(R) CPU @ 2.00GHz\nCPU family:                              6\nModel:                                   85\nThread(s) per core:                      2\nCore(s) per socket:                      1\nSocket(s):                               1\nStepping:                                3\nBogoMIPS:                                4000.28\nFlags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch ssbd ibrs ibpb stibp fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat md_clear arch_capabilities\nHypervisor vendor:                       KVM\nVirtualization type:                     full\nL1d cache:                               32 KiB (1 instance)\nL1i cache:                               32 KiB (1 instance)\nL2 cache:                                1 MiB (1 instance)\nL3 cache:                                38.5 MiB (1 instance)\nNUMA node(s):                            1\nNUMA node0 CPU(s):                       0,1\nVulnerability Gather data sampling:      Not affected\nVulnerability Indirect target selection: Vulnerable\nVulnerability Itlb multihit:             Not affected\nVulnerability L1tf:                      Mitigation; PTE Inversion\nVulnerability Mds:                       Vulnerable; SMT Host state unknown\nVulnerability Meltdown:                  Vulnerable\nVulnerability Mmio stale data:           Vulnerable\nVulnerability Reg file data sampling:    Not affected\nVulnerability Retbleed:                  Vulnerable\nVulnerability Spec rstack overflow:      Not affected\nVulnerability Spec store bypass:         Vulnerable\nVulnerability Spectre v1:                Vulnerable: __user pointer sanitization and usercopy barriers only; no swapgs barriers\nVulnerability Spectre v2:                Vulnerable; IBPB: disabled; STIBP: disabled; PBRSB-eIBRS: Not affected; BHI: Vulnerable\nVulnerability Srbds:                     Not affected\nVulnerability Tsa:                       Not affected\nVulnerability Tsx async abort:           Vulnerable\nVulnerability Vmscape:                   Not affected\n', 'ffmpeg': 'ffmpeg version 6.1.1-3ubuntu5 Copyright (c) 2000-2023 the FFmpeg developers', 'source_clean': True, 'driver_sha256': '4303242ccb74f8a59060b341f2bfdc864fd68f317a637ca54f4363981d113562'}`.

## FPS matrix — images, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 110.07 | 110.39 | 82.69 | 82.35 | 70.13 | 55.33 |
| yolo11 | 71.03 | 76.06 | 70.47 | 70.42 | 64.99 | 60.85 |
| yolo26 | 81.01 | 82.01 | 64.75 | 61.73 | 50.91 | 54.32 |
| yolo-world | 79.67 | 80.82 | 62.35 | 61.68 | 57.78 | 53.75 |
| grounding-dino | 2.90 | N/A | 2.71 | N/A | 2.80 | 2.82 |
| grounding-dino-base | 2.01 | N/A | 1.94 | N/A | 1.98 | 1.99 |
| faster-rcnn | 32.23 | N/A | 28.48 | N/A | 29.16 | 28.21 |
| d-fine | 2.54 | N/A | 2.40 | N/A | 2.44 | 2.16 |
| locate-anything | 1.54 | N/A | 1.54 | N/A | 1.55 | 1.52 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — images, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 164.93 | 191.98 | 142.49 | 138.59 | 118.09 | 89.65 |
| yolo11 | 147.51 | 193.56 | 141.38 | 140.99 | 115.37 | 79.36 |
| yolo26 | 165.08 | 193.63 | 137.66 | 136.66 | 114.79 | 83.05 |
| yolo-world | 92.86 | 97.93 | 86.09 | 81.56 | 70.43 | 73.27 |
| grounding-dino | 3.05 | N/A | 3.04 | N/A | 3.03 | 3.05 |
| grounding-dino-base | 2.10 | N/A | 2.10 | N/A | 2.09 | 2.09 |
| faster-rcnn | 37.59 | N/A | 36.91 | N/A | 35.19 | 35.21 |
| d-fine | 8.16 | N/A | 8.76 | N/A | 8.73 | 8.53 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 82.58 | 84.59 | 79.65 | 74.98 | 71.50 | 70.17 |
| yolo11 | 72.53 | 69.02 | 69.42 | 64.61 | 49.56 | 62.54 |
| yolo26 | 66.19 | 62.80 | 59.62 | 57.06 | 57.12 | 57.49 |
| yolo-world | 67.88 | 54.57 | 65.20 | 61.63 | 60.29 | 61.42 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## FPS matrix — video, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolo | 148.73 | 153.87 | N/A | N/A | 130.55 | 124.02 |
| yolo11 | 133.06 | 134.43 | N/A | N/A | 121.21 | 115.32 |
| yolo26 | 118.37 | 130.86 | N/A | N/A | 115.37 | 109.13 |
| yolo-world | 97.06 | 105.72 | N/A | N/A | 100.47 | 98.76 |

⚠ means numerical/label parity failed in at least one measured repeat; do not treat that row as an equivalent-output speedup.

## Detail — yolo, images, batch 1

Status: **complete**. Load 0.93 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 110.07 ± 12.83 | 13.79 | 10.63/15.31/17.42 | 1655.00 | 27.84/60.00 | 64/320 | PASS |
| raw-stream | 110.39 ± 16.58 | 13.36 | 11.04/17.89/18.61 | 1654.99 | 27.84/60.00 | 64/320 | PASS |
| native-non-stream | 82.69 ± 0.83 | 16.69 | 24.91/48.97/55.98 | 1673.98 | 27.81/60.00 | 64/320 | PASS |
| native-stream | 82.35 ± 1.70 | 14.97 | 12.01/14.53/18.39 | 1674.49 | 27.81/60.00 | 64/320 | PASS |
| managed-non-stream | 70.13 ± 5.73 | 16.60 | 13.76/17.23/20.44 | 1674.36 | 27.81/60.00 | 64/320 | PASS |
| managed-stream | 55.33 ± 8.12 | 22.84 | 13.78/17.30/23.97 | 1675.64 | 27.81/60.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.581444 | 110.07 | 13.79 | 1649.84 | 27.84 | 113.86 | 39.10 | 38.55 | 931.50 | 55.00 |
| 2 | 0.553241 | 115.68 | 13.61 | 1654.38 | 27.84 | 110.68 | 31.00 | 41.78 | 1269.00 | 57.40 |
| 3 | 0.563672 | 113.54 | 13.50 | 1654.38 | 27.84 | 109.79 | 26.20 | 43.51 | 1305.00 | 57.80 |
| 4 | 0.816282 | 78.40 | 33.13 | 1654.83 | 27.84 | 96.93 | 24.00 | 36.14 | 934.29 | 58.93 |
| 5 | 0.592174 | 108.08 | 14.61 | 1654.83 | 27.84 | 111.92 | 29.27 | 42.90 | 1234.09 | 59.00 |
| 6 | 0.567309 | 112.81 | 13.81 | 1655.00 | 27.84 | 110.86 | 28.91 | 42.45 | 1242.27 | 60.55 |
| 7 | 0.583305 | 109.72 | 13.70 | 1654.99 | 27.84 | 115.52 | 26.90 | 44.16 | 1275.00 | 60.80 |

Ready result delivery-interval P50/P95/P99 ms: 10.63/15.31/17.42.
Separate latency-pass duration: 0.72 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6025455156253301, 'inference': 5.9726156562538435, 'postprocess': 1.1779802656297633}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

2.85% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.592777 | 107.97 | 13.36 | 1653.36 | 27.84 | 116.05 | 28.20 | 42.27 | 1263.00 | 55.70 |
| 2 | 0.569660 | 112.35 | 11.68 | 1649.11 | 27.84 | 109.55 | 35.70 | 38.84 | 1125.00 | 56.90 |
| 3 | 0.579740 | 110.39 | 13.39 | 1654.38 | 27.84 | 110.28 | 29.18 | 43.82 | 1305.00 | 57.91 |
| 4 | 0.957442 | 66.84 | 12.87 | 1654.83 | 27.84 | 95.38 | 28.12 | 35.22 | 897.19 | 58.94 |
| 5 | 0.564836 | 113.31 | 13.76 | 1654.83 | 27.84 | 110.06 | 26.30 | 43.50 | 1275.00 | 59.50 |
| 6 | 0.574683 | 111.37 | 12.11 | 1651.48 | 27.84 | 112.05 | 37.20 | 40.07 | 1042.50 | 60.10 |
| 7 | 0.625980 | 102.24 | 13.78 | 1654.99 | 27.84 | 109.53 | 30.36 | 43.38 | 1275.00 | 60.91 |

Ready result delivery-interval P50/P95/P99 ms: 11.05/17.87/18.57.
Separate latency-pass duration: 0.78 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6030764374989062, 'inference': 6.057943875004135, 'postprocess': 1.2407459218719552}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

33.30% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.775089 | 82.57 | 17.11 | 1670.61 | 27.81 | 109.70 | 23.79 | 39.83 | 1290.00 | 55.86 |
| 2 | 0.773240 | 82.77 | 14.70 | 1672.66 | 27.81 | 111.04 | 31.43 | 36.13 | 917.14 | 56.86 |
| 3 | 0.769017 | 83.22 | 16.69 | 1673.98 | 27.81 | 111.59 | 26.07 | 39.44 | 1158.21 | 58.00 |
| 4 | 0.792944 | 80.71 | 15.23 | 1672.00 | 27.81 | 111.66 | 31.43 | 36.53 | 885.00 | 58.86 |
| 5 | 0.773987 | 82.69 | 16.96 | 1673.19 | 27.81 | 109.55 | 25.21 | 41.20 | 1247.14 | 59.57 |
| 6 | 0.779095 | 82.15 | 14.56 | 1672.16 | 27.81 | 115.51 | 33.15 | 36.05 | 871.15 | 60.00 |
| 7 | 0.771981 | 82.90 | 17.86 | 1673.35 | 27.81 | 110.26 | 26.93 | 39.34 | 1145.36 | 61.00 |

Ready result delivery-interval P50/P95/P99 ms: 24.91/48.97/55.98.
Separate latency-pass duration: 1.86 s; not used for throughput.

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

31.33% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.742182 | 86.23 | 16.74 | 1668.41 | 27.81 | 110.30 | 22.77 | 41.32 | 1255.38 | 55.92 |
| 2 | 0.787258 | 81.29 | 14.86 | 1672.54 | 27.81 | 111.24 | 33.43 | 36.32 | 860.36 | 56.43 |
| 3 | 0.771706 | 82.93 | 14.29 | 1672.88 | 27.81 | 111.43 | 27.93 | 37.70 | 1027.50 | 58.00 |
| 4 | 0.777194 | 82.35 | 14.97 | 1674.34 | 27.81 | 112.34 | 34.86 | 36.38 | 838.93 | 58.21 |
| 5 | 0.777689 | 82.30 | 14.57 | 1674.36 | 27.81 | 110.81 | 29.14 | 38.25 | 995.36 | 59.00 |
| 6 | 0.787082 | 81.31 | 15.94 | 1674.49 | 27.81 | 112.88 | 31.71 | 36.78 | 814.29 | 60.00 |
| 7 | 0.765721 | 83.58 | 15.05 | 1671.00 | 27.81 | 111.02 | 27.38 | 37.32 | 1015.38 | 61.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.01/14.54/18.39.
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

55.27% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.995482 | 64.29 | 16.60 | 1673.49 | 27.81 | 105.77 | 27.00 | 34.92 | 954.71 | 55.94 |
| 2 | 0.912630 | 70.13 | 29.81 | 1673.80 | 27.81 | 107.38 | 32.00 | 34.84 | 751.88 | 56.00 |
| 3 | 0.875237 | 73.12 | 16.53 | 1670.66 | 27.81 | 111.60 | 29.13 | 35.54 | 914.00 | 58.00 |
| 4 | 1.065024 | 60.09 | 21.90 | 1674.36 | 27.81 | 106.20 | 31.06 | 34.05 | 641.67 | 58.00 |
| 5 | 0.874284 | 73.20 | 16.54 | 1672.31 | 27.81 | 111.45 | 27.47 | 36.20 | 887.00 | 59.00 |
| 6 | 1.021677 | 62.64 | 21.57 | 1673.30 | 27.81 | 107.45 | 31.65 | 34.30 | 632.65 | 59.65 |
| 7 | 0.866431 | 73.87 | 15.98 | 1673.35 | 27.81 | 110.63 | 29.07 | 35.36 | 913.00 | 61.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.77/17.22/20.44.
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

99.03% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.321348 | 48.44 | 22.84 | 1672.16 | 27.81 | 100.47 | 22.81 | 33.65 | 730.71 | 56.00 |
| 2 | 1.156676 | 55.33 | 23.48 | 1671.32 | 27.81 | 104.50 | 27.63 | 33.97 | 608.68 | 56.00 |
| 3 | 1.121884 | 57.05 | 19.34 | 1675.50 | 27.81 | 103.72 | 25.37 | 34.15 | 798.16 | 58.00 |
| 4 | 1.394704 | 45.89 | 23.03 | 1672.29 | 27.81 | 96.38 | 25.36 | 33.34 | 599.32 | 58.00 |
| 5 | 1.121879 | 57.05 | 16.56 | 1675.64 | 27.81 | 103.88 | 27.53 | 34.02 | 783.95 | 59.00 |
| 6 | 1.409995 | 45.39 | 41.57 | 1673.60 | 27.81 | 96.61 | 26.05 | 33.38 | 595.23 | 59.00 |
| 7 | 0.937019 | 68.30 | 17.03 | 1672.22 | 27.81 | 110.23 | 29.44 | 35.04 | 809.06 | 61.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.80/17.57/23.96.
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

## Detail — yolo11, images, batch 1

Status: **complete**. Load 0.88 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 71.03 ± 13.77 | 16.14 | 10.50/13.80/16.55 | 1676.71 | 59.92/88.00 | 64/320 | PASS |
| raw-stream | 76.06 ± 11.83 | 15.44 | 10.55/14.17/17.62 | 1676.71 | 59.92/88.00 | 64/320 | PASS |
| native-non-stream | 70.47 ± 12.14 | 19.89 | 14.00/17.54/24.10 | 1696.23 | 59.89/88.00 | 64/320 | PASS |
| native-stream | 70.42 ± 8.30 | 17.40 | 13.78/17.94/22.05 | 1696.20 | 59.89/88.00 | 64/320 | PASS |
| managed-non-stream | 64.99 ± 5.90 | 18.40 | 21.80/34.19/38.72 | 1696.37 | 59.89/88.00 | 64/320 | PASS |
| managed-stream | 60.85 ± 0.81 | 19.42 | 26.14/32.48/40.10 | 1697.36 | 59.89/88.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.683261 | 93.67 | 14.28 | 1673.11 | 59.92 | 113.62 | 40.67 | 38.21 | 876.25 | 62.58 |
| 2 | 0.719677 | 88.93 | 15.74 | 1676.15 | 59.92 | 110.13 | 29.15 | 40.27 | 1095.00 | 64.00 |
| 3 | 0.761204 | 84.08 | 15.73 | 1676.16 | 59.92 | 108.53 | 29.43 | 39.27 | 1092.86 | 64.00 |
| 4 | 0.900980 | 71.03 | 16.14 | 1676.55 | 59.92 | 102.43 | 27.13 | 39.48 | 1110.00 | 65.87 |
| 5 | 1.050407 | 60.93 | 26.93 | 1676.55 | 59.92 | 96.70 | 25.18 | 37.19 | 848.82 | 65.94 |
| 6 | 0.988221 | 64.76 | 23.33 | 1676.70 | 59.92 | 96.55 | 23.00 | 38.97 | 1080.00 | 67.00 |
| 7 | 1.048328 | 61.05 | 25.12 | 1676.71 | 59.92 | 100.49 | 26.65 | 36.19 | 736.76 | 67.00 |

Ready result delivery-interval P50/P95/P99 ms: 10.50/13.80/16.55.
Separate latency-pass duration: 0.71 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6097116406085235, 'inference': 7.577894937494278, 'postprocess': 1.1562321874922787}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-5.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.878474 | 72.85 | 15.44 | 1674.87 | 59.92 | 103.48 | 29.27 | 39.03 | 1080.00 | 62.93 |
| 2 | 0.704046 | 90.90 | 14.22 | 1676.15 | 59.92 | 113.66 | 37.15 | 38.25 | 960.00 | 63.92 |
| 3 | 0.989281 | 64.69 | 24.51 | 1676.15 | 59.92 | 97.55 | 24.88 | 37.10 | 930.94 | 64.00 |
| 4 | 0.705196 | 90.75 | 14.07 | 1676.55 | 59.92 | 111.26 | 33.46 | 39.42 | 1001.54 | 65.23 |
| 5 | 0.989277 | 64.69 | 30.56 | 1676.55 | 59.92 | 107.19 | 27.60 | 35.86 | 703.00 | 65.80 |
| 6 | 0.841420 | 76.06 | 14.27 | 1674.07 | 59.92 | 105.84 | 30.14 | 39.23 | 985.71 | 66.93 |
| 7 | 0.716489 | 89.32 | 16.46 | 1676.71 | 59.92 | 112.68 | 40.31 | 38.40 | 845.77 | 67.00 |

Ready result delivery-interval P50/P95/P99 ms: 10.56/14.18/17.54.
Separate latency-pass duration: 0.71 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6756576093742694, 'inference': 9.8424522187468, 'postprocess': 1.4753410312451365}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

0.79% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.364886 | 46.89 | 27.71 | 1694.48 | 59.89 | 96.05 | 22.32 | 35.32 | 875.45 | 62.95 |
| 2 | 0.873525 | 73.27 | 17.03 | 1690.84 | 59.89 | 111.09 | 32.53 | 36.33 | 840.00 | 63.93 |
| 3 | 1.367301 | 46.81 | 32.57 | 1693.35 | 59.89 | 99.78 | 21.59 | 34.52 | 746.59 | 64.00 |
| 4 | 0.908139 | 70.47 | 16.88 | 1694.91 | 59.89 | 111.38 | 30.94 | 36.90 | 810.94 | 65.00 |
| 5 | 0.888350 | 72.04 | 20.24 | 1696.08 | 59.89 | 111.80 | 29.44 | 37.81 | 915.00 | 66.00 |
| 6 | 0.909190 | 70.39 | 16.77 | 1696.21 | 59.89 | 111.49 | 33.94 | 37.07 | 816.56 | 66.94 |
| 7 | 0.888452 | 72.04 | 19.89 | 1696.23 | 59.89 | 109.72 | 26.56 | 39.49 | 1065.00 | 67.00 |

Ready result delivery-interval P50/P95/P99 ms: 14.00/17.54/24.10.
Separate latency-pass duration: 0.94 s; not used for throughput.

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

21.02% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.107435 | 57.79 | 19.11 | 1692.36 | 59.89 | 107.69 | 30.17 | 35.40 | 655.00 | 63.00 |
| 2 | 0.870934 | 73.48 | 16.92 | 1691.91 | 59.89 | 111.39 | 35.80 | 36.84 | 765.00 | 63.20 |
| 3 | 0.923798 | 69.28 | 21.70 | 1694.57 | 59.89 | 112.74 | 36.50 | 36.29 | 727.50 | 64.25 |
| 4 | 1.226016 | 52.20 | 17.12 | 1692.56 | 59.89 | 88.91 | 26.68 | 35.59 | 724.74 | 65.00 |
| 5 | 0.884531 | 72.35 | 17.40 | 1696.08 | 59.89 | 110.92 | 30.56 | 37.60 | 892.50 | 66.00 |
| 6 | 0.908783 | 70.42 | 20.13 | 1696.20 | 59.89 | 113.97 | 35.19 | 37.53 | 765.00 | 66.75 |
| 7 | 0.891809 | 71.76 | 17.20 | 1695.07 | 59.89 | 109.83 | 27.81 | 39.31 | 1021.88 | 67.12 |

Ready result delivery-interval P50/P95/P99 ms: 13.78/17.98/22.06.
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

28.95% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000908 | 63.94 | 19.43 | 1696.01 | 59.89 | 111.65 | 33.39 | 35.85 | 735.00 | 63.00 |
| 2 | 0.992066 | 64.51 | 27.10 | 1694.02 | 59.89 | 112.70 | 34.24 | 35.87 | 692.65 | 63.00 |
| 3 | 0.981580 | 65.20 | 18.40 | 1694.70 | 59.89 | 111.74 | 31.35 | 36.49 | 795.00 | 64.82 |
| 4 | 1.297745 | 49.32 | 18.28 | 1696.36 | 59.89 | 96.39 | 27.05 | 35.53 | 685.50 | 65.00 |
| 5 | 0.980074 | 65.30 | 18.41 | 1696.37 | 59.89 | 112.04 | 31.53 | 36.41 | 791.47 | 66.00 |
| 6 | 0.984743 | 64.99 | 18.37 | 1694.15 | 59.89 | 111.64 | 34.18 | 36.88 | 702.35 | 66.00 |
| 7 | 0.979673 | 65.33 | 18.09 | 1695.08 | 59.89 | 111.87 | 28.82 | 36.87 | 835.59 | 67.24 |

Ready result delivery-interval P50/P95/P99 ms: 21.80/34.19/38.73.
Separate latency-pass duration: 1.42 s; not used for throughput.

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

16.41% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.049099 | 61.00 | 21.07 | 1692.68 | 59.89 | 111.54 | 32.28 | 35.21 | 713.33 | 63.00 |
| 2 | 1.050257 | 60.94 | 19.24 | 1691.83 | 59.89 | 110.27 | 33.28 | 35.72 | 650.83 | 63.00 |
| 3 | 1.051686 | 60.85 | 19.41 | 1694.89 | 59.89 | 111.77 | 31.28 | 35.58 | 725.00 | 64.94 |
| 4 | 1.048813 | 61.02 | 18.95 | 1692.85 | 59.89 | 110.80 | 31.56 | 36.27 | 656.67 | 64.94 |
| 5 | 1.079848 | 59.27 | 19.63 | 1697.36 | 59.89 | 112.21 | 29.26 | 36.32 | 697.89 | 66.00 |
| 6 | 1.077694 | 59.39 | 24.97 | 1696.50 | 59.89 | 112.60 | 32.37 | 35.84 | 641.84 | 66.00 |
| 7 | 1.070466 | 59.79 | 19.42 | 1694.00 | 59.89 | 112.49 | 30.39 | 36.56 | 743.33 | 67.72 |

Ready result delivery-interval P50/P95/P99 ms: 25.89/32.52/40.22.
Separate latency-pass duration: 1.67 s; not used for throughput.

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

Status: **complete**. Load 0.73 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 81.01 ± 1.88 | 17.08 | 11.85/16.74/20.73 | 1701.65 | 59.02/88.00 | 64/320 | PASS |
| raw-stream | 82.01 ± 2.50 | 17.82 | 11.92/16.16/19.71 | 1703.47 | 59.02/88.00 | 64/320 | PASS |
| native-non-stream | 64.75 ± 4.79 | 19.50 | 16.16/19.89/29.18 | 1702.12 | 58.99/88.00 | 64/320 | PASS |
| native-stream | 61.73 ± 9.46 | 19.70 | 15.89/21.53/34.29 | 1703.77 | 58.99/88.00 | 64/320 | PASS |
| managed-non-stream | 50.91 ± 9.97 | 19.05 | 17.34/24.36/27.09 | 1704.02 | 58.99/88.00 | 64/320 | PASS |
| managed-stream | 54.32 ± 7.45 | 20.15 | 18.53/23.36/26.29 | 1701.97 | 58.99/88.00 | 64/320 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.766655 | 83.48 | 17.57 | 1701.65 | 59.02 | 111.96 | 37.00 | 38.73 | 855.00 | 69.00 |
| 2 | 0.798111 | 80.19 | 16.99 | 1683.11 | 59.02 | 110.84 | 32.29 | 40.12 | 1005.00 | 70.00 |
| 3 | 0.792439 | 80.76 | 17.47 | 1683.12 | 59.02 | 110.20 | 30.36 | 40.12 | 1030.71 | 70.00 |
| 4 | 0.788073 | 81.21 | 16.80 | 1683.96 | 59.02 | 115.33 | 32.21 | 41.41 | 1050.00 | 70.93 |
| 5 | 0.774786 | 82.60 | 16.59 | 1683.95 | 59.02 | 111.23 | 30.21 | 40.83 | 1069.29 | 71.00 |
| 6 | 0.790036 | 81.01 | 17.08 | 1684.14 | 59.02 | 112.03 | 34.07 | 40.18 | 972.86 | 72.00 |
| 7 | 0.825092 | 77.57 | 17.10 | 1684.14 | 59.02 | 112.78 | 29.50 | 41.21 | 1020.00 | 72.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.85/16.74/20.72.
Separate latency-pass duration: 0.80 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.5962099531213738, 'inference': 8.751026390632077, 'postprocess': 1.1563067343729472}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-0.14% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.765751 | 83.58 | 18.88 | 1681.19 | 59.02 | 110.94 | 32.79 | 41.52 | 1075.71 | 69.43 |
| 2 | 0.780425 | 82.01 | 14.35 | 1682.23 | 59.02 | 112.24 | 34.21 | 38.89 | 909.64 | 70.00 |
| 3 | 0.791327 | 80.88 | 18.08 | 1683.12 | 59.02 | 111.19 | 30.93 | 40.57 | 1050.00 | 70.14 |
| 4 | 0.778015 | 82.26 | 17.82 | 1703.29 | 59.02 | 113.67 | 34.00 | 38.73 | 903.46 | 70.46 |
| 5 | 0.778381 | 82.22 | 16.81 | 1683.95 | 59.02 | 111.95 | 31.29 | 42.60 | 1080.00 | 71.00 |
| 6 | 0.837729 | 76.40 | 19.85 | 1703.47 | 59.02 | 113.76 | 34.33 | 39.14 | 853.00 | 71.93 |
| 7 | 0.814177 | 78.61 | 17.34 | 1684.14 | 59.02 | 111.98 | 31.43 | 40.00 | 1007.14 | 72.00 |

Ready result delivery-interval P50/P95/P99 ms: 11.94/16.12/19.72.
Separate latency-pass duration: 0.80 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6250813906181207, 'inference': 8.796765656246208, 'postprocess': 1.2163374531226623}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

26.92% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.973072 | 65.77 | 19.14 | 1699.06 | 58.99 | 109.62 | 26.24 | 40.35 | 1095.00 | 69.88 |
| 2 | 1.168176 | 54.79 | 20.44 | 1700.25 | 58.99 | 106.49 | 31.37 | 36.84 | 643.42 | 70.00 |
| 3 | 0.983417 | 65.08 | 19.90 | 1700.41 | 58.99 | 110.65 | 27.00 | 39.00 | 962.65 | 70.00 |
| 4 | 1.034057 | 61.89 | 17.72 | 1700.95 | 58.99 | 111.37 | 32.44 | 37.81 | 765.00 | 70.00 |
| 5 | 0.980179 | 65.29 | 19.50 | 1702.12 | 58.99 | 113.15 | 28.00 | 39.61 | 983.82 | 71.00 |
| 6 | 0.988488 | 64.75 | 17.05 | 1700.29 | 58.99 | 112.58 | 33.18 | 37.57 | 752.65 | 71.71 |
| 7 | 1.155713 | 55.38 | 20.87 | 1700.84 | 58.99 | 110.61 | 24.85 | 39.25 | 1020.00 | 72.00 |

Ready result delivery-interval P50/P95/P99 ms: 16.17/19.89/29.18.
Separate latency-pass duration: 1.07 s; not used for throughput.

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

31.57% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.987867 | 64.79 | 20.44 | 1701.62 | 58.99 | 113.54 | 26.94 | 38.06 | 939.71 | 70.00 |
| 2 | 1.431496 | 44.71 | 31.16 | 1702.45 | 58.99 | 98.36 | 25.04 | 35.80 | 659.35 | 70.00 |
| 3 | 0.985902 | 64.92 | 17.36 | 1700.45 | 58.99 | 112.03 | 31.06 | 37.93 | 838.24 | 70.00 |
| 4 | 1.036852 | 61.73 | 17.85 | 1703.29 | 58.99 | 112.59 | 37.56 | 37.71 | 642.50 | 70.00 |
| 5 | 1.039618 | 61.56 | 19.70 | 1701.27 | 58.99 | 111.09 | 29.61 | 37.94 | 847.50 | 71.00 |
| 6 | 0.985361 | 64.95 | 17.41 | 1701.09 | 58.99 | 113.63 | 35.41 | 37.74 | 709.41 | 71.35 |
| 7 | 1.448953 | 44.17 | 20.48 | 1703.77 | 58.99 | 102.20 | 23.52 | 36.95 | 754.57 | 72.00 |

Ready result delivery-interval P50/P95/P99 ms: 15.89/21.54/34.28.
Separate latency-pass duration: 1.08 s; not used for throughput.

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

57.51% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.081684 | 59.17 | 18.74 | 1702.12 | 58.99 | 112.75 | 31.11 | 37.17 | 774.47 | 69.95 |
| 2 | 1.257129 | 50.91 | 18.23 | 1701.65 | 58.99 | 107.67 | 31.52 | 36.26 | 653.57 | 70.00 |
| 3 | 1.116994 | 57.30 | 19.05 | 1703.54 | 58.99 | 112.54 | 29.65 | 36.30 | 745.50 | 70.00 |
| 4 | 1.695341 | 37.75 | 27.44 | 1702.72 | 58.99 | 98.56 | 23.78 | 35.40 | 585.00 | 70.00 |
| 5 | 1.282654 | 49.90 | 18.48 | 1703.89 | 58.99 | 113.24 | 28.73 | 36.76 | 735.00 | 71.00 |
| 6 | 1.068214 | 59.91 | 19.50 | 1704.02 | 58.99 | 111.81 | 35.11 | 37.75 | 655.26 | 71.00 |
| 7 | 1.801647 | 35.52 | 29.06 | 1702.60 | 58.99 | 99.54 | 22.14 | 34.99 | 585.00 | 72.00 |

Ready result delivery-interval P50/P95/P99 ms: 17.33/24.36/27.14.
Separate latency-pass duration: 1.16 s; not used for throughput.

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

52.45% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.168799 | 54.76 | 20.15 | 1697.88 | 58.99 | 111.04 | 30.90 | 36.29 | 687.75 | 70.00 |
| 2 | 1.155059 | 55.41 | 20.11 | 1701.50 | 58.99 | 113.38 | 33.10 | 36.56 | 625.50 | 70.00 |
| 3 | 1.168754 | 54.76 | 20.14 | 1700.48 | 58.99 | 113.98 | 32.20 | 36.88 | 664.50 | 70.00 |
| 4 | 1.571672 | 40.72 | 18.52 | 1700.68 | 58.99 | 101.54 | 27.08 | 36.08 | 586.80 | 70.00 |
| 5 | 1.688410 | 37.91 | 27.74 | 1700.79 | 58.99 | 98.93 | 24.46 | 35.43 | 589.62 | 71.00 |
| 6 | 1.374005 | 46.58 | 30.19 | 1701.97 | 58.99 | 105.36 | 27.91 | 36.57 | 608.48 | 71.00 |
| 7 | 1.178262 | 54.32 | 20.50 | 1700.92 | 58.99 | 112.63 | 34.80 | 36.96 | 598.50 | 72.00 |

Ready result delivery-interval P50/P95/P99 ms: 18.47/23.38/26.28.
Separate latency-pass duration: 1.23 s; not used for throughput.

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

Status: **complete**. Load 13.65 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 79.67 ± 10.72 | 17.00 | 11.59/15.53/20.57 | 1914.54 | 114.82/136.00 | 64/384 | PASS |
| raw-stream | 80.82 ± 10.04 | 18.18 | 16.76/26.86/28.47 | 1933.16 | 114.82/136.00 | 64/384 | PASS |
| native-non-stream | 62.35 ± 9.14 | 20.03 | 32.40/107.94/129.82 | 1934.18 | 114.79/136.00 | 64/384 | PASS |
| native-stream | 61.68 ± 6.89 | 22.25 | 17.59/61.15/89.82 | 1934.05 | 114.79/136.00 | 64/384 | PASS |
| managed-non-stream | 57.78 ± 1.32 | 18.83 | 16.61/22.22/24.36 | 1935.21 | 114.79/136.00 | 64/384 | PASS |
| managed-stream | 53.75 ± 0.50 | 20.85 | 19.26/24.90/28.99 | 1935.44 | 114.79/136.00 | 64/384 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.770046 | 83.11 | 14.78 | 1908.91 | 114.82 | 114.20 | 52.15 | 64.13 | 1310.77 | 74.92 |
| 2 | 0.951610 | 67.25 | 16.70 | 1913.80 | 114.82 | 105.12 | 50.50 | 63.07 | 1343.44 | 79.06 |
| 3 | 1.092256 | 58.59 | 24.52 | 1913.81 | 114.82 | 98.55 | 35.00 | 63.14 | 1380.00 | 79.69 |
| 4 | 1.040807 | 61.49 | 29.23 | 1914.39 | 114.82 | 101.94 | 37.76 | 63.21 | 1350.00 | 81.94 |
| 5 | 0.803292 | 79.67 | 19.82 | 1914.39 | 114.82 | 113.09 | 48.21 | 63.34 | 1220.36 | 81.93 |
| 6 | 0.779998 | 82.05 | 17.00 | 1914.54 | 114.82 | 114.02 | 55.00 | 65.71 | 1290.00 | 78.62 |
| 7 | 0.776337 | 82.44 | 16.62 | 1914.54 | 114.82 | 114.60 | 52.69 | 65.84 | 1271.54 | 78.08 |

Ready result delivery-interval P50/P95/P99 ms: 11.59/15.52/20.57.
Separate latency-pass duration: 0.79 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.607948609363973, 'inference': 8.798865250005505, 'postprocess': 1.234090109374364}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.88% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.016475 | 62.96 | 18.18 | 1912.42 | 114.82 | 103.37 | 44.65 | 64.30 | 1424.12 | 75.12 |
| 2 | 0.789688 | 81.04 | 18.02 | 1933.16 | 114.82 | 115.80 | 54.23 | 67.09 | 1250.77 | 79.00 |
| 3 | 0.998778 | 64.08 | 24.53 | 1913.81 | 114.82 | 104.70 | 36.62 | 64.98 | 1383.75 | 79.94 |
| 4 | 1.058632 | 60.46 | 19.54 | 1914.40 | 114.82 | 98.30 | 39.22 | 66.90 | 1365.83 | 82.00 |
| 5 | 0.791860 | 80.82 | 16.68 | 1914.39 | 114.82 | 113.35 | 53.86 | 65.59 | 1224.64 | 81.93 |
| 6 | 0.789548 | 81.06 | 21.59 | 1914.55 | 114.82 | 114.26 | 54.50 | 61.63 | 1223.57 | 78.36 |
| 7 | 0.783196 | 81.72 | 17.08 | 1914.55 | 114.82 | 115.15 | 55.64 | 62.25 | 1255.71 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 16.80/26.90/29.12.
Separate latency-pass duration: 1.18 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.7727956874976627, 'inference': 11.705347140617661, 'postprocess': 1.7156890937464198}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

22.98% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.428777 | 44.79 | 25.94 | 1929.74 | 114.79 | 98.32 | 28.86 | 62.60 | 1528.64 | 75.91 |
| 2 | 1.026484 | 62.35 | 21.10 | 1933.00 | 114.79 | 113.89 | 41.06 | 65.04 | 1403.33 | 78.94 |
| 3 | 0.984391 | 65.01 | 20.03 | 1929.67 | 114.79 | 111.93 | 45.24 | 66.13 | 1378.24 | 80.06 |
| 4 | 1.156189 | 55.35 | 17.67 | 1932.87 | 114.79 | 109.14 | 39.11 | 65.25 | 1431.32 | 82.05 |
| 5 | 0.987860 | 64.79 | 19.37 | 1932.88 | 114.79 | 114.42 | 45.41 | 64.28 | 1351.76 | 81.88 |
| 6 | 1.436726 | 44.55 | 23.19 | 1934.18 | 114.79 | 99.87 | 31.67 | 53.02 | 1248.12 | 78.46 |
| 7 | 1.009578 | 63.39 | 19.48 | 1930.68 | 114.79 | 111.95 | 42.59 | 63.17 | 1420.59 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 32.39/108.24/129.82.
Separate latency-pass duration: 2.85 s; not used for throughput.

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

29.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.184385 | 54.04 | 23.61 | 1931.98 | 114.79 | 108.06 | 36.70 | 55.83 | 1302.00 | 75.65 |
| 2 | 1.093247 | 58.54 | 23.26 | 1930.50 | 114.79 | 114.03 | 39.00 | 64.14 | 1432.50 | 78.22 |
| 3 | 1.016357 | 62.97 | 22.25 | 1933.24 | 114.79 | 114.26 | 42.71 | 64.75 | 1371.18 | 80.71 |
| 4 | 1.024878 | 62.45 | 17.92 | 1930.53 | 114.79 | 113.56 | 42.71 | 62.77 | 1346.47 | 81.94 |
| 5 | 1.037582 | 61.68 | 17.62 | 1934.05 | 114.79 | 111.95 | 40.89 | 63.46 | 1361.67 | 81.00 |
| 6 | 1.452120 | 44.07 | 34.45 | 1933.01 | 114.79 | 98.69 | 28.39 | 63.89 | 1527.39 | 79.83 |
| 7 | 1.030766 | 62.09 | 20.07 | 1930.68 | 114.79 | 114.98 | 41.18 | 64.75 | 1402.06 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 17.59/61.15/89.83.
Separate latency-pass duration: 1.56 s; not used for throughput.

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

36.68% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.071493 | 59.73 | 18.83 | 1931.31 | 114.79 | 113.75 | 38.83 | 63.95 | 1430.83 | 76.78 |
| 2 | 1.107638 | 57.78 | 18.73 | 1933.21 | 114.79 | 114.03 | 36.32 | 62.83 | 1438.42 | 77.95 |
| 3 | 1.114683 | 57.42 | 18.69 | 1932.54 | 114.79 | 116.74 | 36.83 | 66.91 | 1424.17 | 80.94 |
| 4 | 1.095780 | 58.41 | 18.25 | 1935.21 | 114.79 | 112.57 | 37.63 | 64.61 | 1386.32 | 81.74 |
| 5 | 1.135997 | 56.34 | 19.32 | 1932.88 | 114.79 | 112.95 | 39.42 | 64.55 | 1402.89 | 81.00 |
| 6 | 1.066094 | 60.03 | 19.06 | 1933.01 | 114.79 | 112.43 | 39.11 | 64.74 | 1393.33 | 79.94 |
| 7 | 1.111875 | 57.56 | 24.35 | 1934.50 | 114.79 | 113.22 | 38.37 | 61.03 | 1421.84 | 77.74 |

Ready result delivery-interval P50/P95/P99 ms: 16.62/22.20/24.37.
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

47.84% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.200394 | 53.32 | 21.55 | 1933.81 | 114.79 | 112.52 | 35.52 | 63.08 | 1459.29 | 76.95 |
| 2 | 1.183315 | 54.09 | 20.22 | 1930.72 | 114.79 | 112.77 | 33.75 | 62.98 | 1444.50 | 77.25 |
| 3 | 1.195522 | 53.53 | 20.18 | 1932.66 | 114.79 | 112.72 | 34.10 | 59.94 | 1414.50 | 80.95 |
| 4 | 1.190629 | 53.75 | 21.83 | 1931.17 | 114.79 | 113.65 | 32.25 | 60.64 | 1391.25 | 81.15 |
| 5 | 1.187605 | 53.89 | 20.07 | 1932.95 | 114.79 | 112.90 | 35.70 | 64.60 | 1420.50 | 80.70 |
| 6 | 1.167114 | 54.84 | 20.85 | 1933.30 | 114.79 | 113.06 | 34.20 | 61.47 | 1405.50 | 79.95 |
| 7 | 1.193320 | 53.63 | 23.97 | 1935.44 | 114.79 | 111.50 | 35.05 | 62.79 | 1438.57 | 77.10 |

Ready result delivery-interval P50/P95/P99 ms: 19.07/25.10/28.99.
Separate latency-pass duration: 1.27 s; not used for throughput.

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

Status: **complete**. Load 27.27 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.90 ± 0.10 | 348.74 | 342.71/345.54/346.01 | 2822.68 | 1366.93/1424.00 | 8/48 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.71 ± 0.14 | 348.17 | 344.90/348.37/348.56 | 2822.68 | 1366.92/1424.00 | 8/48 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.80 ± 0.12 | 355.26 | 380.52/398.78/400.38 | 2829.95 | 1366.92/1424.00 | 8/48 | PASS |
| managed-stream | 2.82 ± 0.08 | 347.56 | 345.50/347.93/348.02 | 2831.13 | 1366.92/1424.00 | 8/48 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.638997 | 3.03 | 329.92 | 2822.68 | 1366.93 | 117.57 | 76.88 | 65.21 | 1016.07 | 74.33 |
| 2 | 2.941279 | 2.72 | 379.41 | 2822.68 | 1366.93 | 110.83 | 76.59 | 61.83 | 909.13 | 80.09 |
| 3 | 2.850254 | 2.81 | 353.68 | 2811.37 | 1366.93 | 116.30 | 77.72 | 64.04 | 889.57 | 80.89 |
| 4 | 2.767805 | 2.89 | 352.24 | 2818.52 | 1366.93 | 115.83 | 77.83 | 64.76 | 929.02 | 78.09 |
| 5 | 2.757886 | 2.90 | 344.79 | 2818.52 | 1366.93 | 115.99 | 77.43 | 63.48 | 930.98 | 77.78 |
| 6 | 2.726436 | 2.93 | 348.74 | 2818.52 | 1366.93 | 116.91 | 77.11 | 66.68 | 979.09 | 75.50 |
| 7 | 2.712669 | 2.95 | 343.30 | 2818.52 | 1366.93 | 116.06 | 77.04 | 65.34 | 980.67 | 75.20 |

Ready result delivery-interval P50/P95/P99 ms: 342.71/345.54/346.01.
Separate latency-pass duration: 2.74 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

3.63% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.737996 | 2.92 | 341.94 | 2818.52 | 1366.92 | 114.65 | 76.07 | 64.89 | 986.09 | 75.17 |
| 2 | 3.007959 | 2.66 | 348.17 | 2822.68 | 1366.92 | 107.88 | 73.96 | 62.73 | 938.62 | 79.55 |
| 3 | 2.953652 | 2.71 | 371.23 | 2796.03 | 1366.92 | 115.16 | 76.45 | 64.48 | 894.18 | 81.14 |
| 4 | 3.119451 | 2.56 | 364.18 | 2818.52 | 1366.92 | 100.17 | 70.84 | 64.99 | 970.33 | 78.73 |
| 5 | 2.806354 | 2.85 | 347.83 | 2818.52 | 1366.92 | 115.36 | 75.93 | 64.66 | 948.59 | 77.09 |
| 6 | 3.005189 | 2.66 | 372.39 | 2806.33 | 1366.92 | 100.89 | 68.70 | 63.05 | 1043.86 | 75.57 |
| 7 | 2.727756 | 2.93 | 340.92 | 2822.68 | 1366.92 | 115.00 | 75.80 | 62.40 | 988.00 | 75.38 |

Ready result delivery-interval P50/P95/P99 ms: 344.90/348.38/348.58.
Separate latency-pass duration: 2.76 s; not used for throughput.

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

6.81% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.818822 | 2.84 | 342.19 | 2829.95 | 1366.92 | 115.13 | 73.20 | 63.31 | 985.91 | 76.14 |
| 2 | 2.845633 | 2.81 | 349.20 | 2826.43 | 1366.92 | 116.65 | 76.17 | 62.73 | 915.00 | 78.70 |
| 3 | 3.105055 | 2.58 | 367.78 | 2829.95 | 1366.92 | 103.86 | 72.65 | 66.48 | 945.94 | 81.17 |
| 4 | 2.860591 | 2.80 | 365.81 | 2821.37 | 1366.92 | 115.22 | 78.23 | 63.55 | 909.69 | 79.42 |
| 5 | 3.013424 | 2.65 | 355.26 | 2828.77 | 1366.92 | 99.58 | 71.23 | 64.80 | 1043.18 | 76.77 |
| 6 | 2.752362 | 2.91 | 340.00 | 2829.95 | 1366.92 | 114.87 | 75.24 | 63.44 | 985.33 | 75.87 |
| 7 | 3.047852 | 2.62 | 382.39 | 2827.61 | 1366.92 | 100.26 | 68.24 | 63.94 | 1052.61 | 75.52 |

Ready result delivery-interval P50/P95/P99 ms: 380.52/398.77/400.36.
Separate latency-pass duration: 3.04 s; not used for throughput.

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

1.71% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.929635 | 2.73 | 376.46 | 2828.78 | 1366.92 | 108.01 | 73.16 | 62.28 | 984.67 | 76.82 |
| 2 | 2.838409 | 2.82 | 347.56 | 2823.37 | 1366.92 | 114.54 | 76.06 | 61.58 | 959.04 | 77.79 |
| 3 | 2.968217 | 2.70 | 382.35 | 2805.37 | 1366.92 | 114.06 | 78.61 | 64.92 | 897.86 | 80.65 |
| 4 | 2.913256 | 2.75 | 362.24 | 2802.64 | 1366.92 | 114.39 | 77.20 | 63.13 | 909.18 | 80.02 |
| 5 | 2.780708 | 2.88 | 344.41 | 2829.95 | 1366.92 | 115.23 | 76.63 | 63.44 | 974.67 | 76.20 |
| 6 | 2.773139 | 2.88 | 345.91 | 2828.78 | 1366.92 | 115.79 | 74.57 | 64.33 | 984.78 | 75.98 |
| 7 | 2.754448 | 2.90 | 337.35 | 2831.13 | 1366.92 | 115.89 | 73.87 | 63.85 | 994.57 | 75.78 |

Ready result delivery-interval P50/P95/P99 ms: 345.59/347.89/348.00.
Separate latency-pass duration: 2.76 s; not used for throughput.

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

Status: **complete**. Load 24.74 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.01 ± 0.04 | 497.17 | 526.87/541.76/544.28 | 2793.93 | 1600.91/1662.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 1.94 ± 0.09 | 510.05 | 502.21/514.58/515.24 | 2793.93 | 1600.90/1662.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 1.98 ± 0.08 | 497.72 | 496.37/502.57/502.84 | 2802.36 | 1600.90/1662.00 | 8/40 | PASS |
| managed-stream | 1.99 ± 0.08 | 508.52 | 526.40/537.87/539.68 | 2802.36 | 1600.90/1662.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.047751 | 1.98 | 510.67 | 2793.89 | 1600.91 | 116.25 | 85.73 | 68.30 | 827.05 | 78.88 |
| 2 | 4.074956 | 1.96 | 500.59 | 2793.92 | 1600.91 | 110.66 | 85.31 | 68.62 | 832.50 | 78.47 |
| 3 | 4.029061 | 1.99 | 523.05 | 2793.92 | 1600.91 | 111.90 | 85.20 | 62.44 | 855.92 | 77.66 |
| 4 | 3.827061 | 2.09 | 477.02 | 2793.92 | 1600.91 | 113.86 | 85.51 | 64.82 | 900.48 | 75.06 |
| 5 | 3.967386 | 2.02 | 477.58 | 2793.93 | 1600.91 | 107.68 | 82.29 | 65.66 | 900.51 | 75.37 |
| 6 | 3.966756 | 2.02 | 495.48 | 2793.93 | 1600.91 | 115.62 | 84.95 | 63.84 | 848.86 | 77.85 |
| 7 | 3.978027 | 2.01 | 497.17 | 2793.93 | 1600.91 | 115.45 | 84.73 | 66.12 | 848.41 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 526.88/541.75/544.26.
Separate latency-pass duration: 4.19 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

2.72% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.168389 | 1.92 | 515.15 | 2793.89 | 1600.90 | 115.13 | 85.52 | 64.58 | 804.35 | 80.06 |
| 2 | 4.132742 | 1.94 | 525.88 | 2793.92 | 1600.90 | 114.58 | 84.09 | 62.98 | 826.96 | 79.16 |
| 3 | 3.942884 | 2.03 | 506.05 | 2793.92 | 1600.90 | 117.59 | 83.60 | 68.20 | 875.71 | 76.98 |
| 4 | 3.931272 | 2.03 | 518.25 | 2793.92 | 1600.90 | 114.62 | 83.77 | 65.07 | 909.23 | 75.06 |
| 5 | 3.914471 | 2.04 | 496.75 | 2793.93 | 1600.90 | 115.52 | 84.06 | 64.47 | 881.31 | 75.74 |
| 6 | 4.452237 | 1.80 | 510.05 | 2793.93 | 1600.90 | 87.69 | 74.57 | 61.71 | 953.81 | 77.25 |
| 7 | 4.217095 | 1.90 | 509.47 | 2793.93 | 1600.90 | 102.15 | 79.24 | 64.12 | 883.33 | 77.87 |

Ready result delivery-interval P50/P95/P99 ms: 502.19/514.61/515.29.
Separate latency-pass duration: 4.04 s; not used for throughput.

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

1.74% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.391149 | 1.82 | 527.24 | 2802.33 | 1600.90 | 104.98 | 81.11 | 64.45 | 834.00 | 80.95 |
| 2 | 4.285451 | 1.87 | 560.46 | 2802.34 | 1600.90 | 108.71 | 83.53 | 65.96 | 824.09 | 80.23 |
| 3 | 4.063619 | 1.97 | 494.52 | 2802.35 | 1600.90 | 106.97 | 82.59 | 62.15 | 900.00 | 76.27 |
| 4 | 4.043611 | 1.98 | 490.38 | 2802.35 | 1600.90 | 105.04 | 80.65 | 62.12 | 926.37 | 75.06 |
| 5 | 3.909310 | 2.05 | 491.89 | 2802.36 | 1600.90 | 115.51 | 85.72 | 64.58 | 875.31 | 75.98 |
| 6 | 3.960209 | 2.02 | 506.56 | 2802.36 | 1600.90 | 114.63 | 84.91 | 63.01 | 865.15 | 76.98 |
| 7 | 4.047389 | 1.98 | 497.72 | 2802.36 | 1600.90 | 114.92 | 84.48 | 59.63 | 852.54 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 496.37/502.57/502.84.
Separate latency-pass duration: 3.98 s; not used for throughput.

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

2.10% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.324705 | 1.85 | 536.96 | 2802.34 | 1600.90 | 115.81 | 84.39 | 64.19 | 776.88 | 81.88 |
| 2 | 4.306910 | 1.86 | 535.41 | 2802.35 | 1600.90 | 112.66 | 84.53 | 63.90 | 790.29 | 81.47 |
| 3 | 3.966656 | 2.02 | 537.45 | 2802.36 | 1600.90 | 112.29 | 83.91 | 64.46 | 892.38 | 75.94 |
| 4 | 3.907454 | 2.05 | 493.21 | 2802.36 | 1600.90 | 114.33 | 83.37 | 64.73 | 896.77 | 75.58 |
| 5 | 4.130514 | 1.94 | 497.34 | 2802.36 | 1600.90 | 105.05 | 81.29 | 63.37 | 908.47 | 76.21 |
| 6 | 3.951295 | 2.02 | 500.53 | 2802.36 | 1600.90 | 114.46 | 84.26 | 63.24 | 867.00 | 76.78 |
| 7 | 4.027077 | 1.99 | 508.52 | 2802.36 | 1600.90 | 115.56 | 84.55 | 61.16 | 857.01 | 77.90 |

Ready result delivery-interval P50/P95/P99 ms: 526.38/537.80/539.66.
Separate latency-pass duration: 4.15 s; not used for throughput.

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

Status: **complete**. Load 2.30 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 32.23 ± 5.30 | 36.19 | 29.13/36.48/45.65 | 1688.27 | 202.88/324.00 | 32/256 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 28.48 ± 5.59 | 36.95 | 34.92/41.96/48.32 | 1688.27 | 202.85/324.00 | 32/256 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 29.16 ± 3.81 | 35.60 | 33.63/41.57/44.13 | 1726.86 | 202.85/324.00 | 32/256 | PASS |
| managed-stream | 28.21 ± 0.47 | 37.47 | 33.32/41.87/50.33 | 1726.91 | 202.85/324.00 | 32/256 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.992942 | 32.23 | 34.71 | 1687.40 | 202.88 | 159.05 | 57.94 | 62.54 | 1160.62 | 76.00 |
| 2 | 1.299249 | 24.63 | 60.43 | 1688.00 | 202.88 | 131.09 | 45.11 | 57.03 | 1092.63 | 77.42 |
| 3 | 0.988085 | 32.39 | 36.19 | 1688.00 | 202.88 | 160.04 | 56.65 | 65.43 | 1151.47 | 77.94 |
| 4 | 1.178566 | 27.15 | 35.93 | 1688.21 | 202.88 | 136.19 | 52.22 | 63.83 | 1210.83 | 79.00 |
| 5 | 1.662162 | 19.25 | 54.72 | 1688.21 | 202.88 | 104.85 | 30.35 | 58.29 | 1430.22 | 78.96 |
| 6 | 0.984375 | 32.51 | 38.53 | 1688.27 | 202.88 | 160.34 | 59.25 | 63.63 | 1137.19 | 78.00 |
| 7 | 0.964959 | 33.16 | 36.05 | 1688.27 | 202.88 | 158.32 | 60.81 | 64.50 | 1142.81 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 29.13/36.48/45.65.
Separate latency-pass duration: 0.97 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

7.43% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.073764 | 29.80 | 32.66 | 1687.41 | 202.85 | 159.01 | 53.76 | 64.22 | 1264.41 | 76.53 |
| 2 | 1.876455 | 17.05 | 50.35 | 1685.57 | 202.85 | 103.85 | 28.21 | 59.27 | 1444.29 | 77.68 |
| 3 | 1.061494 | 30.15 | 36.95 | 1688.01 | 202.85 | 161.31 | 55.41 | 62.20 | 1192.06 | 78.00 |
| 4 | 1.123706 | 28.48 | 35.55 | 1685.86 | 202.85 | 153.28 | 50.67 | 66.35 | 1275.00 | 79.00 |
| 5 | 1.123653 | 28.48 | 54.24 | 1688.21 | 202.85 | 152.38 | 50.17 | 60.54 | 1215.00 | 78.83 |
| 6 | 1.055981 | 30.30 | 34.55 | 1685.92 | 202.85 | 158.54 | 53.06 | 63.42 | 1215.00 | 78.00 |
| 7 | 1.659667 | 19.28 | 36.95 | 1688.27 | 202.85 | 113.85 | 36.92 | 56.96 | 1328.65 | 77.46 |

Ready result delivery-interval P50/P95/P99 ms: 34.93/41.97/48.35.
Separate latency-pass duration: 1.16 s; not used for throughput.

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

9.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.097281 | 29.16 | 33.23 | 1724.99 | 202.85 | 158.32 | 51.89 | 60.41 | 1273.33 | 76.89 |
| 2 | 1.095438 | 29.21 | 37.89 | 1726.46 | 202.85 | 162.49 | 52.56 | 65.02 | 1255.00 | 77.56 |
| 3 | 1.086521 | 29.45 | 34.64 | 1725.53 | 202.85 | 158.44 | 53.11 | 60.66 | 1204.17 | 78.44 |
| 4 | 1.106411 | 28.92 | 35.72 | 1726.86 | 202.85 | 158.61 | 53.00 | 63.10 | 1233.16 | 78.95 |
| 5 | 1.101931 | 29.04 | 33.59 | 1725.72 | 202.85 | 159.60 | 52.33 | 66.85 | 1214.17 | 78.72 |
| 6 | 1.085317 | 29.48 | 35.60 | 1725.75 | 202.85 | 158.71 | 53.94 | 63.42 | 1225.83 | 78.11 |
| 7 | 1.672458 | 19.13 | 56.11 | 1725.79 | 202.85 | 116.97 | 31.40 | 56.27 | 1231.80 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 33.63/41.58/44.13.
Separate latency-pass duration: 1.11 s; not used for throughput.

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

11.91% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.130623 | 28.30 | 34.86 | 1726.28 | 202.85 | 155.46 | 49.74 | 65.71 | 1265.53 | 76.95 |
| 2 | 1.152407 | 27.77 | 36.49 | 1726.32 | 202.85 | 154.28 | 47.00 | 64.90 | 1280.53 | 77.05 |
| 3 | 1.105736 | 28.94 | 37.08 | 1725.64 | 202.85 | 158.93 | 52.44 | 60.78 | 1208.33 | 78.78 |
| 4 | 1.134426 | 28.21 | 38.91 | 1725.68 | 202.85 | 153.20 | 47.74 | 61.40 | 1227.63 | 78.95 |
| 5 | 1.139659 | 28.08 | 37.47 | 1726.91 | 202.85 | 156.57 | 50.84 | 59.68 | 1256.84 | 78.37 |
| 6 | 1.128459 | 28.36 | 39.82 | 1726.91 | 202.85 | 156.67 | 49.68 | 63.00 | 1230.00 | 78.05 |
| 7 | 1.165563 | 27.45 | 41.67 | 1725.80 | 202.85 | 154.46 | 49.84 | 64.83 | 1260.79 | 77.05 |

Ready result delivery-interval P50/P95/P99 ms: 33.18/41.84/50.32.
Separate latency-pass duration: 1.14 s; not used for throughput.

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

Status: **complete**. Load 29.19 s. Device `cuda:0`, dtype `float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.54 ± 0.32 | 399.28 | 384.93/401.27/402.58 | 2997.32 | 139.72/210.00 | 8/40 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.40 ± 0.22 | 387.15 | 393.24/409.70/409.95 | 2997.34 | 141.17/210.00 | 8/40 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.44 ± 0.22 | 396.74 | 580.18/605.37/607.24 | 3006.21 | 139.55/210.00 | 8/40 | PASS |
| managed-stream | 2.16 ± 0.30 | 402.45 | 419.43/677.39/743.65 | 3006.24 | 142.59/210.00 | 8/40 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.100269 | 1.95 | 398.81 | 2996.79 | 139.47 | 106.29 | 5.78 | 33.06 | 585.00 | 71.00 |
| 2 | 4.173856 | 1.92 | 571.86 | 2996.98 | 134.44 | 103.37 | 5.48 | 33.46 | 585.00 | 72.00 |
| 3 | 3.085192 | 2.59 | 403.76 | 2996.99 | 133.08 | 114.55 | 7.42 | 33.83 | 585.00 | 72.12 |
| 4 | 3.148944 | 2.54 | 399.28 | 2997.14 | 129.73 | 114.01 | 7.11 | 34.10 | 585.00 | 73.98 |
| 5 | 3.062192 | 2.61 | 376.41 | 2997.15 | 139.72 | 114.65 | 7.54 | 34.36 | 585.00 | 74.00 |
| 6 | 3.848419 | 2.08 | 392.07 | 2997.30 | 131.23 | 105.51 | 6.05 | 34.26 | 585.00 | 75.00 |
| 7 | 3.115699 | 2.57 | 419.28 | 2997.32 | 137.79 | 114.79 | 7.26 | 34.58 | 585.00 | 75.00 |

Ready result delivery-interval P50/P95/P99 ms: 384.93/401.26/402.57.
Separate latency-pass duration: 3.10 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

3.03% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.376373 | 2.37 | 516.32 | 2996.81 | 137.75 | 113.60 | 6.70 | 33.16 | 585.00 | 71.00 |
| 2 | 3.336226 | 2.40 | 387.15 | 2996.67 | 135.44 | 114.38 | 6.91 | 33.59 | 585.00 | 72.00 |
| 3 | 3.271680 | 2.45 | 381.93 | 2997.02 | 135.55 | 115.05 | 7.13 | 33.76 | 585.00 | 72.95 |
| 4 | 3.567266 | 2.24 | 597.02 | 2996.83 | 141.17 | 110.54 | 6.51 | 34.17 | 585.00 | 73.44 |
| 5 | 4.305924 | 1.86 | 545.54 | 2997.18 | 139.60 | 104.01 | 5.38 | 34.13 | 585.00 | 74.00 |
| 6 | 3.221358 | 2.48 | 377.78 | 2996.99 | 130.81 | 115.09 | 7.13 | 34.70 | 585.00 | 75.00 |
| 7 | 3.210260 | 2.49 | 376.47 | 2997.34 | 133.19 | 114.40 | 7.17 | 34.61 | 585.00 | 74.63 |

Ready result delivery-interval P50/P95/P99 ms: 393.24/409.70/409.95.
Separate latency-pass duration: 3.16 s; not used for throughput.

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

4.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.318912 | 2.41 | 386.41 | 3005.68 | 134.18 | 114.72 | 6.91 | 33.24 | 585.00 | 71.00 |
| 2 | 3.220883 | 2.48 | 412.40 | 3005.78 | 136.63 | 114.70 | 6.95 | 33.67 | 585.00 | 72.00 |
| 3 | 4.231488 | 1.89 | 411.07 | 3005.92 | 135.34 | 104.02 | 5.51 | 33.57 | 585.00 | 73.00 |
| 4 | 3.628409 | 2.20 | 389.85 | 3005.96 | 136.85 | 109.93 | 6.31 | 34.18 | 585.00 | 73.00 |
| 5 | 3.277204 | 2.44 | 396.74 | 3006.06 | 133.19 | 115.07 | 6.91 | 34.38 | 585.00 | 74.00 |
| 6 | 3.167876 | 2.53 | 397.95 | 3006.12 | 130.09 | 114.93 | 7.17 | 34.41 | 585.00 | 75.00 |
| 7 | 3.261341 | 2.45 | 394.03 | 3006.21 | 139.55 | 114.24 | 7.09 | 34.28 | 585.00 | 74.00 |

Ready result delivery-interval P50/P95/P99 ms: 580.18/605.37/607.25.
Separate latency-pass duration: 4.20 s; not used for throughput.

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

6.01% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.704493 | 2.16 | 402.45 | 3005.71 | 134.20 | 109.46 | 6.33 | 33.31 | 585.00 | 71.07 |
| 2 | 3.863299 | 2.07 | 662.34 | 3005.74 | 141.28 | 108.67 | 5.76 | 33.58 | 585.00 | 71.95 |
| 3 | 3.279974 | 2.44 | 386.91 | 3005.95 | 136.92 | 114.49 | 6.87 | 33.99 | 585.00 | 73.00 |
| 4 | 3.181988 | 2.51 | 396.11 | 3005.96 | 131.65 | 114.97 | 6.93 | 33.90 | 585.00 | 73.00 |
| 5 | 3.246134 | 2.46 | 386.39 | 3006.09 | 130.66 | 114.80 | 7.00 | 34.44 | 585.00 | 74.00 |
| 6 | 4.619407 | 1.73 | 424.01 | 3006.12 | 131.14 | 99.67 | 5.06 | 34.25 | 585.00 | 74.25 |
| 7 | 4.089717 | 1.96 | 575.39 | 3006.24 | 142.59 | 106.30 | 5.58 | 33.97 | 585.00 | 74.00 |

Ready result delivery-interval P50/P95/P99 ms: 419.42/677.41/743.71.
Separate latency-pass duration: 3.79 s; not used for throughput.

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

Status: **complete**. Load 106.07 s. Device `cuda:0`, dtype `torch.float16`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 1.54 ± 0.13 | 650.32 | 758.38/758.38/758.38 | 3899.34 | 7910.11/8364.00 | 1/4 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 1.54 ± 0.13 | 647.36 | 810.78/810.78/810.78 | 3899.34 | 7910.11/8364.00 | 1/4 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 1.55 ± 0.14 | 646.37 | 660.49/660.49/660.49 | 3899.34 | 7910.11/8364.00 | 1/4 | PASS |
| managed-stream | 1.52 ± 0.11 | 658.90 | 661.72/661.72/661.72 | 3899.34 | 7910.11/8364.00 | 1/4 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.682048 | 1.47 | 681.99 | 3899.34 | 7910.11 | 122.92 | 55.45 | 65.51 | 1255.91 | 61.45 |
| 2 | 0.638314 | 1.57 | 638.25 | 3899.34 | 7910.11 | 118.14 | 55.55 | 60.84 | 1138.64 | 64.09 |
| 3 | 0.634109 | 1.58 | 634.06 | 3899.34 | 7910.11 | 120.83 | 59.36 | 61.84 | 1141.36 | 64.82 |
| 4 | 0.757963 | 1.32 | 757.90 | 3899.34 | 7910.11 | 111.00 | 51.62 | 67.41 | 1206.92 | 66.92 |
| 5 | 0.794734 | 1.26 | 794.67 | 3899.34 | 7910.11 | 102.67 | 51.75 | 60.92 | 1257.50 | 67.42 |
| 6 | 0.636444 | 1.57 | 636.39 | 3899.34 | 7910.11 | 116.26 | 61.18 | 61.48 | 1086.82 | 69.27 |
| 7 | 0.650468 | 1.54 | 650.32 | 3899.34 | 7910.11 | 116.72 | 57.82 | 60.37 | 1080.00 | 69.91 |

Ready result delivery-interval P50/P95/P99 ms: 758.37/758.37/758.37.
Separate latency-pass duration: 0.76 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

-0.43% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.630534 | 1.59 | 630.47 | 3899.34 | 7910.11 | 117.45 | 52.64 | 62.10 | 1186.36 | 61.91 |
| 2 | 0.635569 | 1.57 | 635.51 | 3899.34 | 7910.11 | 117.57 | 53.91 | 62.58 | 1159.09 | 63.91 |
| 3 | 0.638722 | 1.57 | 638.66 | 3899.34 | 7910.11 | 116.99 | 57.82 | 60.94 | 1163.18 | 64.91 |
| 4 | 0.648340 | 1.54 | 648.28 | 3899.34 | 7910.11 | 130.75 | 58.00 | 65.73 | 1078.50 | 66.70 |
| 5 | 0.819488 | 1.22 | 819.42 | 3899.34 | 7910.11 | 100.94 | 48.23 | 60.78 | 1268.08 | 67.69 |
| 6 | 0.658818 | 1.52 | 658.76 | 3899.34 | 7910.11 | 116.51 | 56.00 | 65.99 | 1084.09 | 68.91 |
| 7 | 0.647417 | 1.54 | 647.36 | 3899.34 | 7910.11 | 114.48 | 56.64 | 63.46 | 1093.64 | 70.00 |

Ready result delivery-interval P50/P95/P99 ms: 810.73/810.73/810.73.
Separate latency-pass duration: 0.81 s; not used for throughput.

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

1.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.635156 | 1.57 | 635.09 | 3899.34 | 7910.11 | 117.59 | 57.73 | 64.82 | 1175.45 | 62.18 |
| 2 | 0.842568 | 1.19 | 842.50 | 3899.34 | 7910.11 | 107.64 | 50.00 | 56.65 | 1302.50 | 63.58 |
| 3 | 0.646438 | 1.55 | 646.37 | 3899.34 | 7910.11 | 116.73 | 58.50 | 63.01 | 1163.75 | 65.25 |
| 4 | 0.638553 | 1.57 | 638.49 | 3899.34 | 7910.11 | 118.25 | 58.27 | 63.01 | 1114.09 | 66.00 |
| 5 | 0.642379 | 1.56 | 642.32 | 3899.34 | 7910.11 | 116.75 | 61.82 | 68.68 | 1110.00 | 67.91 |
| 6 | 0.659544 | 1.52 | 659.48 | 3899.34 | 7910.11 | 117.26 | 60.27 | 60.99 | 1092.27 | 68.91 |
| 7 | 0.658105 | 1.52 | 658.03 | 3899.34 | 7910.11 | 115.16 | 62.36 | 66.82 | 1054.09 | 70.00 |

Ready result delivery-interval P50/P95/P99 ms: 660.47/660.47/660.47.
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
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.


### managed-stream

3.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.726722 | 1.38 | 726.25 | 3899.34 | 7910.11 | 111.73 | 58.42 | 61.81 | 1257.50 | 62.67 |
| 2 | 0.736117 | 1.36 | 734.42 | 3899.34 | 7910.11 | 100.83 | 52.50 | 60.38 | 1286.25 | 63.08 |
| 3 | 0.659196 | 1.52 | 658.90 | 3899.34 | 7910.11 | 116.83 | 59.27 | 70.16 | 1130.45 | 65.73 |
| 4 | 0.639911 | 1.56 | 639.61 | 3899.34 | 7910.11 | 119.21 | 54.09 | 60.79 | 1118.18 | 65.91 |
| 5 | 0.649318 | 1.54 | 649.01 | 3899.34 | 7910.11 | 119.20 | 61.45 | 68.86 | 1096.36 | 68.27 |
| 6 | 0.655359 | 1.53 | 655.09 | 3899.34 | 7910.11 | 119.16 | 58.36 | 62.09 | 1070.45 | 68.82 |
| 7 | 0.779814 | 1.28 | 779.54 | 3899.34 | 7910.11 | 105.11 | 53.77 | 60.85 | 1190.77 | 70.77 |

Ready result delivery-interval P50/P95/P99 ms: 661.50/661.50/661.50.
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
Image-pack frame indices restart per public API call; cross-path source representations are not asserted to have identical identity strings.

## Detail — yolo, images, batch 4

Status: **complete**. Load 0.46 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 164.93 ± 11.30 | 26.57 | 24.73/37.53/47.94 | 2059.84 | 76.00/186.00 | 256/1280 | PASS |
| raw-stream | 191.98 ± 14.20 | 27.50 | 20.48/23.17/28.77 | 2040.15 | 76.00/186.00 | 256/1280 | PASS |
| native-non-stream | 142.49 ± 2.76 | 29.82 | 27.48/30.59/33.89 | 2040.16 | 75.87/186.00 | 256/1280 | PASS |
| native-stream | 138.59 ± 3.20 | 37.56 | 27.04/34.84/36.73 | 2048.94 | 75.87/186.00 | 256/1280 | PASS |
| managed-non-stream | 118.09 ± 6.26 | 32.43 | 32.56/37.62/44.93 | 2068.88 | 75.87/186.00 | 256/1280 | PASS |
| managed-stream | 89.65 ± 13.02 | 45.39 | 56.32/216.39/251.53 | 2221.04 | 75.87/186.00 | 256/1280 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.552146 | 164.93 | 33.61 | 2059.84 | 76.00 | 110.58 | 44.83 | 58.99 | 1258.04 | 74.22 |
| 2 | 1.352185 | 189.32 | 22.63 | 1826.11 | 76.00 | 117.29 | 48.29 | 59.93 | 1342.86 | 79.62 |
| 3 | 1.562764 | 163.81 | 28.29 | 1826.20 | 76.00 | 98.47 | 39.64 | 67.22 | 1392.00 | 80.04 |
| 4 | 1.438292 | 177.99 | 21.81 | 1831.90 | 76.00 | 107.38 | 44.38 | 64.84 | 1395.00 | 78.96 |
| 5 | 1.626083 | 157.43 | 28.68 | 1831.91 | 76.00 | 100.10 | 37.27 | 64.51 | 1403.08 | 78.85 |
| 6 | 1.503006 | 170.33 | 22.95 | 1951.67 | 76.00 | 103.30 | 43.44 | 64.94 | 1412.40 | 75.96 |
| 7 | 1.606612 | 159.34 | 26.57 | 1951.68 | 76.00 | 100.55 | 35.15 | 65.24 | 1431.35 | 76.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/31.57/39.23.
Separate latency-pass duration: 1.71 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.743588910154358, 'inference': 2.890444757813526, 'postprocess': 1.4190005859378019}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-8.31% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.625289 | 157.51 | 31.45 | 1940.58 | 76.00 | 98.34 | 38.15 | 63.39 | 1448.33 | 75.37 |
| 2 | 1.333491 | 191.98 | 26.55 | 1914.00 | 76.00 | 111.58 | 46.78 | 60.67 | 1321.30 | 78.87 |
| 3 | 1.483500 | 172.56 | 29.50 | 1826.20 | 76.00 | 106.47 | 41.84 | 66.48 | 1362.60 | 80.40 |
| 4 | 1.318708 | 194.13 | 27.50 | 1920.38 | 76.00 | 114.35 | 46.18 | 66.99 | 1369.09 | 78.95 |
| 5 | 1.331461 | 192.27 | 22.35 | 1831.91 | 76.00 | 111.89 | 45.35 | 69.12 | 1391.74 | 78.96 |
| 6 | 1.334675 | 191.81 | 28.02 | 2040.15 | 76.00 | 113.56 | 45.22 | 68.48 | 1376.09 | 75.91 |
| 7 | 1.324497 | 193.28 | 23.40 | 1951.68 | 76.00 | 113.15 | 47.09 | 66.97 | 1371.82 | 76.32 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/21.02/23.22.
Separate latency-pass duration: 1.33 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8094528085935693, 'inference': 2.861180859379253, 'postprocess': 1.5304524257828689}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

20.43% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.869235 | 136.95 | 28.25 | 2012.69 | 75.87 | 113.59 | 34.32 | 63.22 | 1432.26 | 75.94 |
| 2 | 1.841694 | 139.00 | 33.96 | 1914.94 | 75.87 | 114.02 | 35.45 | 60.43 | 1429.35 | 78.03 |
| 3 | 1.792886 | 142.79 | 29.17 | 1913.64 | 75.87 | 113.08 | 35.20 | 63.60 | 1420.50 | 80.90 |
| 4 | 1.784495 | 143.46 | 34.10 | 1920.39 | 75.87 | 113.66 | 35.47 | 62.54 | 1404.50 | 78.87 |
| 5 | 1.767300 | 144.85 | 28.41 | 1920.39 | 75.87 | 113.51 | 36.30 | 58.56 | 1419.00 | 78.93 |
| 6 | 1.822649 | 140.45 | 34.10 | 2040.15 | 75.87 | 114.66 | 35.87 | 60.26 | 1438.55 | 75.32 |
| 7 | 1.796628 | 142.49 | 29.82 | 2040.16 | 75.87 | 113.54 | 35.43 | 59.62 | 1419.50 | 76.03 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/29.22/31.71.
Separate latency-pass duration: 1.78 s; not used for throughput.

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

20.43% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.869241 | 136.95 | 34.00 | 2023.18 | 75.87 | 114.38 | 34.23 | 61.49 | 1456.45 | 76.55 |
| 2 | 1.847151 | 138.59 | 37.74 | 1921.16 | 75.87 | 116.19 | 38.03 | 55.24 | 1255.00 | 77.30 |
| 3 | 1.782784 | 143.60 | 35.92 | 1918.94 | 75.87 | 113.97 | 36.27 | 61.05 | 1422.00 | 81.23 |
| 4 | 1.857896 | 137.79 | 37.56 | 1928.34 | 75.87 | 113.26 | 37.29 | 56.87 | 1254.68 | 78.52 |
| 5 | 1.760992 | 145.37 | 33.60 | 1920.39 | 75.87 | 114.28 | 36.10 | 60.80 | 1417.76 | 78.83 |
| 6 | 1.852776 | 138.17 | 37.56 | 2048.94 | 75.87 | 113.89 | 37.19 | 55.51 | 1300.16 | 74.77 |
| 7 | 1.828387 | 140.01 | 38.99 | 2040.16 | 75.87 | 112.25 | 35.45 | 64.02 | 1434.68 | 76.29 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/29.31/35.33.
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

39.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.167858 | 118.09 | 31.98 | 2029.60 | 75.87 | 113.59 | 31.16 | 52.70 | 1320.41 | 76.78 |
| 2 | 2.450817 | 104.45 | 47.86 | 1943.70 | 75.87 | 113.41 | 38.41 | 45.78 | 836.34 | 76.22 |
| 3 | 2.160406 | 118.50 | 32.36 | 1918.57 | 75.87 | 113.57 | 31.31 | 61.26 | 1325.83 | 81.39 |
| 4 | 2.318665 | 110.41 | 38.91 | 1923.36 | 75.87 | 114.11 | 35.49 | 45.71 | 947.31 | 78.67 |
| 5 | 2.124504 | 120.50 | 32.04 | 1925.99 | 75.87 | 114.52 | 31.17 | 58.03 | 1379.17 | 77.97 |
| 6 | 2.320199 | 110.34 | 41.05 | 2068.88 | 75.87 | 113.50 | 36.87 | 46.39 | 945.38 | 74.33 |
| 7 | 2.124952 | 120.47 | 32.43 | 2045.14 | 75.87 | 113.60 | 31.06 | 57.11 | 1380.83 | 76.42 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/33.84/37.94.
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


### managed-stream

77.74% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.598071 | 98.53 | 47.54 | 2209.12 | 75.87 | 114.34 | 31.02 | 44.66 | 1011.28 | 76.02 |
| 2 | 3.699989 | 69.19 | 42.24 | 2209.90 | 75.87 | 100.39 | 28.12 | 41.41 | 707.11 | 76.00 |
| 3 | 2.625889 | 97.49 | 45.39 | 2119.64 | 75.87 | 112.64 | 30.52 | 46.74 | 1008.07 | 80.77 |
| 4 | 3.605556 | 71.00 | 56.23 | 2114.95 | 75.87 | 102.19 | 30.53 | 43.43 | 640.79 | 79.44 |
| 5 | 2.669061 | 95.91 | 39.44 | 2216.23 | 75.87 | 111.57 | 30.56 | 46.91 | 1054.53 | 76.67 |
| 6 | 3.393435 | 75.44 | 52.65 | 2219.95 | 75.87 | 104.94 | 32.44 | 40.82 | 656.83 | 75.00 |
| 7 | 2.855530 | 89.65 | 45.23 | 2221.04 | 75.87 | 107.72 | 29.32 | 44.97 | 1018.72 | 75.45 |

Ready result delivery-interval P50/P95/P99 ms: 0.05/155.65/219.42.
Separate latency-pass duration: 5.85 s; not used for throughput.

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

Status: **complete**. Load 0.34 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 147.51 ± 20.06 | 31.68 | 21.00/27.47/29.54 | 1965.61 | 112.51/184.00 | 256/1344 | PASS |
| raw-stream | 193.56 ± 17.62 | 23.34 | 20.27/22.88/26.01 | 1969.13 | 112.51/184.00 | 256/1344 | PASS |
| native-non-stream | 141.38 ± 1.50 | 28.60 | 27.75/31.84/40.00 | 2047.94 | 112.39/184.00 | 256/1344 | PASS |
| native-stream | 140.99 ± 2.59 | 35.75 | 27.66/34.70/40.93 | 2047.94 | 112.39/184.00 | 256/1344 | PASS |
| managed-non-stream | 115.37 ± 4.36 | 34.13 | 33.44/39.07/53.52 | 2070.22 | 112.39/184.00 | 256/1344 | PASS |
| managed-stream | 79.36 ± 11.13 | 54.28 | 58.05/71.15/84.86 | 2232.52 | 112.39/184.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.780249 | 143.80 | 33.42 | 1943.68 | 112.51 | 99.01 | 38.78 | 59.23 | 1297.22 | 78.78 |
| 2 | 1.440872 | 177.67 | 21.96 | 1883.58 | 112.51 | 110.10 | 46.04 | 60.81 | 1391.88 | 78.25 |
| 3 | 1.768629 | 144.74 | 31.68 | 1883.73 | 112.51 | 98.19 | 34.66 | 63.03 | 1450.34 | 77.97 |
| 4 | 1.735491 | 147.51 | 22.48 | 1965.61 | 112.51 | 100.87 | 38.89 | 61.56 | 1442.14 | 75.79 |
| 5 | 1.604954 | 159.51 | 46.63 | 1965.61 | 112.51 | 103.80 | 37.41 | 66.69 | 1457.78 | 76.04 |
| 6 | 1.871963 | 136.75 | 33.37 | 1847.23 | 112.51 | 97.37 | 34.17 | 62.46 | 1476.50 | 77.80 |
| 7 | 1.340316 | 191.00 | 23.67 | 1847.23 | 112.51 | 113.79 | 45.96 | 69.21 | 1380.00 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/23.28/28.00.
Separate latency-pass duration: 1.40 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.7681998281263702, 'inference': 3.3425495820349482, 'postprocess': 1.5234750429673483}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-13.93% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.633061 | 156.76 | 36.93 | 1827.00 | 112.51 | 104.93 | 38.19 | 63.64 | 1410.00 | 79.88 |
| 2 | 1.308554 | 195.64 | 26.67 | 1969.13 | 112.51 | 115.01 | 47.77 | 65.21 | 1445.45 | 78.77 |
| 3 | 1.343752 | 190.51 | 28.86 | 1883.72 | 112.51 | 113.11 | 45.26 | 61.40 | 1400.87 | 78.35 |
| 4 | 1.310844 | 195.29 | 20.76 | 1965.61 | 112.51 | 113.46 | 48.26 | 64.62 | 1414.57 | 75.65 |
| 5 | 1.321148 | 193.77 | 22.40 | 1965.61 | 112.51 | 114.59 | 48.14 | 68.30 | 1416.82 | 76.77 |
| 6 | 1.611146 | 158.89 | 21.76 | 1843.71 | 112.51 | 105.69 | 40.08 | 69.86 | 1464.81 | 77.42 |
| 7 | 1.322619 | 193.56 | 23.34 | 1847.23 | 112.51 | 114.51 | 48.05 | 71.94 | 1429.77 | 78.64 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/21.10/22.88.
Separate latency-pass duration: 1.33 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8305031679700647, 'inference': 3.249691417972045, 'postprocess': 1.3140925585934937}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.80% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.847854 | 138.54 | 27.43 | 1911.75 | 112.39 | 113.45 | 35.81 | 63.35 | 1444.35 | 80.84 |
| 2 | 1.820673 | 140.61 | 34.72 | 1968.59 | 112.39 | 113.07 | 36.17 | 63.28 | 1456.50 | 78.07 |
| 3 | 1.833048 | 139.66 | 28.60 | 1973.23 | 112.39 | 114.04 | 36.32 | 64.82 | 1462.74 | 78.13 |
| 4 | 1.798086 | 142.37 | 29.86 | 2037.39 | 112.39 | 113.58 | 36.60 | 62.78 | 1475.50 | 75.00 |
| 5 | 1.804470 | 141.87 | 28.55 | 2047.94 | 112.39 | 114.31 | 34.23 | 64.80 | 1477.00 | 76.97 |
| 6 | 1.794804 | 142.63 | 27.27 | 1928.68 | 112.39 | 113.98 | 35.93 | 61.30 | 1443.00 | 76.90 |
| 7 | 1.810775 | 141.38 | 40.02 | 1928.68 | 112.39 | 114.62 | 36.37 | 61.78 | 1444.00 | 78.90 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/29.13/32.27.
Separate latency-pass duration: 1.81 s; not used for throughput.

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

7.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.815692 | 140.99 | 33.45 | 1919.91 | 112.39 | 112.96 | 35.83 | 61.74 | 1414.50 | 80.97 |
| 2 | 1.823288 | 140.41 | 39.10 | 1967.19 | 112.39 | 113.32 | 37.10 | 57.11 | 1342.00 | 78.03 |
| 3 | 1.810753 | 141.38 | 38.05 | 1973.25 | 112.39 | 113.97 | 36.80 | 61.69 | 1451.50 | 78.00 |
| 4 | 1.865151 | 137.25 | 35.75 | 2047.94 | 112.39 | 112.86 | 36.61 | 56.79 | 1379.52 | 74.71 |
| 5 | 1.800118 | 142.21 | 27.82 | 2047.94 | 112.39 | 113.73 | 36.80 | 60.10 | 1426.00 | 76.97 |
| 6 | 1.889200 | 135.51 | 38.74 | 1939.22 | 112.39 | 112.81 | 37.50 | 56.85 | 1297.50 | 76.28 |
| 7 | 1.801500 | 142.10 | 28.72 | 1928.68 | 112.39 | 113.41 | 36.53 | 60.94 | 1426.50 | 78.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/29.34/35.10.
Separate latency-pass duration: 1.81 s; not used for throughput.

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

30.61% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.169951 | 117.98 | 32.53 | 1922.52 | 112.39 | 114.53 | 32.06 | 53.81 | 1341.67 | 81.11 |
| 2 | 2.301048 | 111.25 | 40.92 | 1986.80 | 112.39 | 113.43 | 36.56 | 46.66 | 951.15 | 78.00 |
| 3 | 2.138513 | 119.71 | 32.40 | 1973.75 | 112.39 | 111.67 | 31.50 | 58.39 | 1387.50 | 77.81 |
| 4 | 2.266774 | 112.94 | 40.02 | 2070.22 | 112.39 | 114.08 | 36.29 | 48.43 | 994.74 | 74.00 |
| 5 | 2.218879 | 115.37 | 34.13 | 2053.80 | 112.39 | 114.47 | 31.68 | 53.85 | 1353.24 | 77.03 |
| 6 | 2.365746 | 108.21 | 40.73 | 1939.22 | 112.39 | 113.85 | 34.98 | 47.62 | 1011.75 | 75.97 |
| 7 | 2.146205 | 119.28 | 32.55 | 1934.54 | 112.39 | 113.89 | 31.06 | 58.47 | 1419.17 | 78.97 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/36.17/44.55.
Separate latency-pass duration: 2.22 s; not used for throughput.

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

94.18% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.579840 | 99.23 | 54.84 | 2218.26 | 112.39 | 112.04 | 30.42 | 49.69 | 1108.95 | 80.86 |
| 2 | 3.594894 | 71.21 | 51.98 | 2224.77 | 112.39 | 101.56 | 29.95 | 41.70 | 682.91 | 79.25 |
| 3 | 2.757372 | 92.84 | 39.03 | 2232.52 | 112.39 | 109.45 | 30.24 | 46.55 | 1061.33 | 76.42 |
| 4 | 3.370021 | 75.96 | 58.79 | 2231.84 | 112.39 | 105.75 | 32.91 | 41.12 | 652.36 | 74.55 |
| 5 | 3.225840 | 79.36 | 54.28 | 2125.73 | 112.39 | 103.31 | 26.76 | 42.90 | 975.00 | 76.38 |
| 6 | 2.857626 | 89.58 | 66.01 | 2125.97 | 112.39 | 111.00 | 35.79 | 42.89 | 731.49 | 76.00 |
| 7 | 3.592074 | 71.27 | 45.14 | 2130.72 | 112.39 | 100.99 | 25.02 | 43.59 | 921.39 | 77.67 |

Ready result delivery-interval P50/P95/P99 ms: 0.05/61.61/73.97.
Separate latency-pass duration: 3.60 s; not used for throughput.

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

Status: **complete**. Load 0.32 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 165.08 ± 24.53 | 22.83 | 20.77/27.80/32.04 | 1924.85 | 111.72/184.00 | 256/1344 | PASS |
| raw-stream | 193.63 ± 22.79 | 22.64 | 20.15/22.72/27.03 | 1951.34 | 111.72/184.00 | 256/1344 | PASS |
| native-non-stream | 137.66 ± 6.57 | 28.06 | 28.62/32.47/40.26 | 1952.51 | 111.60/184.00 | 256/1344 | PASS |
| native-stream | 136.66 ± 3.37 | 30.11 | 29.11/36.49/38.48 | 1962.29 | 111.60/184.00 | 256/1344 | PASS |
| managed-non-stream | 114.79 ± 6.00 | 39.45 | 33.78/41.88/48.59 | 1961.65 | 111.60/184.00 | 256/1344 | PASS |
| managed-stream | 83.05 ± 12.71 | 43.63 | 57.51/74.44/81.69 | 2229.69 | 111.60/184.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.366374 | 187.36 | 27.88 | 1924.85 | 111.72 | 114.37 | 47.87 | 61.58 | 1300.43 | 76.39 |
| 2 | 1.304042 | 196.31 | 21.13 | 1922.77 | 111.72 | 113.64 | 49.77 | 63.90 | 1324.09 | 78.00 |
| 3 | 1.744106 | 146.78 | 22.83 | 1922.90 | 111.72 | 100.01 | 33.28 | 64.05 | 1463.79 | 77.97 |
| 4 | 1.550777 | 165.08 | 22.07 | 1886.82 | 111.72 | 105.29 | 40.50 | 61.58 | 1449.23 | 77.19 |
| 5 | 1.721169 | 148.74 | 27.05 | 1886.82 | 111.72 | 101.64 | 33.66 | 64.92 | 1465.86 | 76.97 |
| 6 | 1.796655 | 142.49 | 35.27 | 1892.04 | 111.72 | 97.53 | 33.93 | 61.43 | 1479.00 | 76.57 |
| 7 | 1.285808 | 199.10 | 21.71 | 1892.05 | 111.72 | 114.64 | 45.82 | 65.89 | 1357.50 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/22.86/27.84.
Separate latency-pass duration: 1.40 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6349577304698428, 'inference': 2.847701312505002, 'postprocess': 1.0897626054697795}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-4.63% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.809595 | 141.47 | 33.77 | 1883.64 | 111.72 | 97.99 | 34.67 | 65.54 | 1492.00 | 77.40 |
| 2 | 1.319454 | 194.02 | 23.28 | 1951.34 | 111.72 | 114.02 | 47.65 | 63.91 | 1335.65 | 77.91 |
| 3 | 1.663273 | 153.91 | 31.79 | 1922.91 | 111.72 | 103.83 | 36.36 | 63.11 | 1420.71 | 78.29 |
| 4 | 1.321361 | 193.74 | 21.91 | 1886.82 | 111.72 | 115.34 | 47.05 | 56.75 | 1335.68 | 76.95 |
| 5 | 1.322101 | 193.63 | 21.88 | 1886.82 | 111.72 | 113.11 | 46.18 | 68.49 | 1420.91 | 77.59 |
| 6 | 1.376211 | 186.02 | 22.64 | 1892.04 | 111.72 | 112.79 | 46.52 | 66.76 | 1411.30 | 76.09 |
| 7 | 1.289463 | 198.53 | 21.41 | 1892.05 | 111.72 | 113.74 | 48.50 | 63.18 | 1348.64 | 77.09 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/21.06/23.51.
Separate latency-pass duration: 1.31 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.6972556249991158, 'inference': 3.8236802617177545, 'postprocess': 1.4752133945341583}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

21.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.115147 | 121.03 | 36.69 | 1900.38 | 111.60 | 109.04 | 29.09 | 58.47 | 1442.57 | 77.89 |
| 2 | 1.858449 | 137.75 | 29.14 | 1950.80 | 111.60 | 113.92 | 36.19 | 55.49 | 1289.03 | 77.35 |
| 3 | 1.859646 | 137.66 | 27.39 | 1952.51 | 111.60 | 113.30 | 32.97 | 62.53 | 1478.23 | 78.97 |
| 4 | 1.879652 | 136.20 | 28.06 | 1916.87 | 111.60 | 114.36 | 34.47 | 56.96 | 1359.84 | 76.94 |
| 5 | 1.850750 | 138.32 | 26.82 | 1912.19 | 111.60 | 114.36 | 35.10 | 63.17 | 1488.87 | 77.87 |
| 6 | 1.881290 | 136.08 | 36.79 | 1926.78 | 111.60 | 112.53 | 33.97 | 57.42 | 1432.97 | 75.69 |
| 7 | 1.814831 | 141.06 | 27.25 | 1922.10 | 111.60 | 113.92 | 35.48 | 63.29 | 1482.58 | 77.90 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/30.24/32.67.
Separate latency-pass duration: 1.86 s; not used for throughput.

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

21.65% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.811687 | 141.30 | 29.72 | 1911.00 | 111.60 | 114.21 | 35.13 | 59.82 | 1447.74 | 78.42 |
| 2 | 1.926124 | 132.91 | 31.77 | 1962.29 | 111.60 | 113.58 | 36.97 | 53.47 | 1200.00 | 77.44 |
| 3 | 1.929173 | 132.70 | 30.65 | 1957.88 | 111.60 | 113.03 | 34.42 | 54.87 | 1306.82 | 78.94 |
| 4 | 1.886518 | 135.70 | 32.40 | 1928.00 | 111.60 | 112.79 | 36.44 | 54.14 | 1255.31 | 76.88 |
| 5 | 1.873295 | 136.66 | 30.11 | 1916.88 | 111.60 | 114.05 | 33.84 | 56.24 | 1385.62 | 77.22 |
| 6 | 1.847906 | 138.54 | 29.97 | 1922.09 | 111.60 | 112.29 | 37.87 | 53.13 | 1289.03 | 75.03 |
| 7 | 1.825745 | 140.22 | 27.57 | 1922.10 | 111.60 | 113.89 | 34.61 | 62.37 | 1438.06 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/33.00/36.69.
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


### managed-non-stream

47.21% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.167286 | 118.12 | 32.78 | 1904.45 | 111.60 | 114.02 | 30.17 | 57.23 | 1354.17 | 78.83 |
| 2 | 2.540579 | 100.76 | 61.07 | 1961.65 | 111.60 | 112.48 | 35.19 | 45.66 | 842.50 | 77.00 |
| 3 | 2.227141 | 114.95 | 33.27 | 1958.31 | 111.60 | 112.98 | 31.71 | 55.13 | 1190.13 | 78.76 |
| 4 | 2.282867 | 112.14 | 40.13 | 1945.60 | 111.60 | 114.40 | 36.03 | 45.98 | 941.54 | 77.00 |
| 5 | 2.165136 | 118.24 | 32.96 | 1920.98 | 111.60 | 112.16 | 31.30 | 55.66 | 1311.49 | 77.05 |
| 6 | 2.312804 | 110.69 | 39.45 | 1926.19 | 111.60 | 113.11 | 36.26 | 45.43 | 931.92 | 74.97 |
| 7 | 2.230202 | 114.79 | 48.83 | 1931.18 | 111.60 | 113.22 | 31.45 | 50.43 | 1216.97 | 77.97 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/35.69/42.38.
Separate latency-pass duration: 2.21 s; not used for throughput.

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

88.08% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.569898 | 99.61 | 39.71 | 2197.15 | 111.60 | 114.24 | 31.05 | 45.25 | 1020.35 | 78.00 |
| 2 | 3.536219 | 72.39 | 39.57 | 2197.89 | 111.60 | 101.93 | 29.38 | 42.16 | 695.36 | 77.91 |
| 3 | 2.574340 | 99.44 | 39.18 | 2229.69 | 111.60 | 114.01 | 32.81 | 44.85 | 949.53 | 78.00 |
| 4 | 3.560566 | 71.90 | 60.40 | 2227.43 | 111.60 | 101.62 | 30.68 | 41.07 | 633.21 | 77.18 |
| 5 | 3.082534 | 83.05 | 43.63 | 2192.38 | 111.60 | 105.67 | 28.00 | 42.72 | 952.81 | 76.08 |
| 6 | 3.032014 | 84.43 | 62.55 | 2194.95 | 111.60 | 108.09 | 34.16 | 41.80 | 680.51 | 75.00 |
| 7 | 3.688445 | 69.41 | 50.49 | 2197.31 | 111.60 | 99.35 | 25.50 | 42.06 | 821.12 | 77.05 |

Ready result delivery-interval P50/P95/P99 ms: 0.05/64.20/76.22.
Separate latency-pass duration: 3.61 s; not used for throughput.

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

Status: **complete**. Load 8.32 s. Device `cpu`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 92.86 ± 2.96 | 40.47 | 40.78/42.88/43.51 | 2168.15 | 214.50/304.00 | 256/1344 | PASS |
| raw-stream | 97.93 ± 3.23 | 40.18 | 45.52/53.41/54.09 | 2103.08 | 214.50/304.00 | 256/1344 | PASS |
| native-non-stream | 86.09 ± 1.33 | 45.57 | 47.35/51.64/52.82 | 2192.44 | 214.38/304.00 | 256/1344 | PASS |
| native-stream | 81.56 ± 5.47 | 52.62 | 47.62/53.31/56.13 | 2207.83 | 214.38/304.00 | 256/1344 | PASS |
| managed-non-stream | 70.43 ± 5.27 | 51.96 | 51.52/54.64/55.79 | 2232.59 | 214.38/304.00 | 256/1344 | PASS |
| managed-stream | 73.27 ± 2.65 | 63.74 | 65.58/82.72/84.60 | 2456.97 | 214.38/304.00 | 256/1344 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.756723 | 92.86 | 48.95 | 2168.15 | 214.50 | 109.18 | 69.52 | 64.73 | 994.43 | 80.80 |
| 2 | 2.609604 | 98.10 | 40.44 | 2100.87 | 214.50 | 115.04 | 71.09 | 62.81 | 1045.81 | 76.09 |
| 3 | 2.800740 | 91.40 | 40.47 | 2100.09 | 214.50 | 101.19 | 62.17 | 63.40 | 1088.93 | 75.98 |
| 4 | 2.650743 | 96.58 | 40.15 | 2102.62 | 214.50 | 113.35 | 69.86 | 66.32 | 1041.63 | 76.79 |
| 5 | 2.771167 | 92.38 | 46.73 | 2102.62 | 214.50 | 102.48 | 64.44 | 64.81 | 1056.98 | 76.91 |
| 6 | 2.822462 | 90.70 | 40.03 | 2103.08 | 214.50 | 102.95 | 65.33 | 64.98 | 1056.98 | 77.58 |
| 7 | 2.645362 | 96.77 | 53.83 | 2103.08 | 214.50 | 113.75 | 69.09 | 64.49 | 1038.75 | 77.02 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/42.03/43.02.
Separate latency-pass duration: 2.62 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8359661796912832, 'inference': 7.662608828127304, 'postprocess': 1.296210941410969}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.28% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.859814 | 89.52 | 47.53 | 2035.84 | 214.50 | 104.62 | 67.18 | 63.71 | 981.48 | 81.11 |
| 2 | 2.601449 | 98.41 | 34.95 | 2097.35 | 214.50 | 114.08 | 70.37 | 61.50 | 1032.91 | 76.95 |
| 3 | 2.605623 | 98.25 | 43.25 | 2100.09 | 214.50 | 115.21 | 69.67 | 64.34 | 1048.95 | 75.98 |
| 4 | 2.616735 | 97.83 | 36.31 | 2102.62 | 214.50 | 115.52 | 70.58 | 65.97 | 1042.33 | 76.23 |
| 5 | 2.607428 | 98.18 | 40.18 | 2102.62 | 214.50 | 114.90 | 72.40 | 64.62 | 1036.74 | 77.07 |
| 6 | 2.621731 | 97.65 | 35.31 | 2099.57 | 214.50 | 114.65 | 71.30 | 64.95 | 1032.91 | 77.93 |
| 7 | 2.614055 | 97.93 | 40.63 | 2103.08 | 214.50 | 115.65 | 70.86 | 64.49 | 1034.65 | 77.09 |

Ready result delivery-interval P50/P95/P99 ms: 0.03/49.27/53.37.
Separate latency-pass duration: 2.95 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 0.8793431445361932, 'inference': 7.796639878910661, 'postprocess': 1.4569203593737257}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

11.51% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.074095 | 83.28 | 48.13 | 2132.58 | 214.38 | 114.69 | 61.54 | 63.67 | 1035.90 | 81.66 |
| 2 | 2.973802 | 86.09 | 41.20 | 2172.68 | 214.38 | 114.19 | 60.47 | 62.57 | 1102.65 | 76.98 |
| 3 | 2.923296 | 87.57 | 45.07 | 2178.61 | 214.38 | 115.03 | 60.94 | 63.31 | 1117.81 | 75.60 |
| 4 | 2.964835 | 86.35 | 41.87 | 2178.50 | 214.38 | 115.19 | 59.98 | 64.15 | 1120.71 | 75.98 |
| 5 | 2.986172 | 85.73 | 45.57 | 2187.29 | 214.38 | 114.84 | 60.08 | 62.59 | 1098.37 | 77.49 |
| 6 | 2.952468 | 86.71 | 47.42 | 2192.44 | 214.38 | 115.08 | 61.18 | 63.84 | 1091.33 | 77.78 |
| 7 | 2.973985 | 86.08 | 51.00 | 2187.75 | 214.38 | 114.61 | 60.53 | 62.84 | 1092.24 | 77.14 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/48.10/51.10.
Separate latency-pass duration: 3.04 s; not used for throughput.

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

13.86% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.138750 | 81.56 | 56.03 | 2147.66 | 214.38 | 114.54 | 59.62 | 61.63 | 1032.40 | 81.46 |
| 2 | 3.516313 | 72.80 | 61.02 | 2207.83 | 214.38 | 101.65 | 48.15 | 61.10 | 1174.81 | 77.43 |
| 3 | 2.989622 | 85.63 | 41.36 | 2173.10 | 214.38 | 115.27 | 59.20 | 62.60 | 1130.82 | 75.37 |
| 4 | 3.289397 | 77.83 | 52.62 | 2178.50 | 214.38 | 106.31 | 51.65 | 61.28 | 1150.29 | 75.63 |
| 5 | 3.075943 | 83.23 | 41.39 | 2191.98 | 214.38 | 111.92 | 57.98 | 61.59 | 1120.00 | 77.84 |
| 6 | 3.029293 | 84.51 | 56.94 | 2202.40 | 214.38 | 113.34 | 58.42 | 60.63 | 1088.70 | 77.72 |
| 7 | 3.538222 | 72.35 | 44.92 | 2192.44 | 214.38 | 104.61 | 49.62 | 60.27 | 1156.36 | 77.05 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/49.36/53.57.
Separate latency-pass duration: 3.08 s; not used for throughput.

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

32.95% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.634855 | 70.43 | 46.54 | 2148.20 | 214.38 | 108.00 | 50.24 | 61.16 | 1073.79 | 80.71 |
| 2 | 3.289291 | 77.83 | 51.96 | 2232.59 | 214.38 | 113.62 | 52.36 | 65.97 | 1128.27 | 78.02 |
| 3 | 3.948996 | 64.83 | 46.57 | 2198.52 | 214.38 | 102.75 | 42.48 | 61.98 | 1212.05 | 75.20 |
| 4 | 3.524039 | 72.64 | 52.05 | 2187.59 | 214.38 | 110.58 | 49.14 | 57.04 | 1161.05 | 75.32 |
| 5 | 3.989097 | 64.17 | 55.59 | 2201.94 | 214.38 | 103.84 | 42.43 | 59.15 | 1162.50 | 77.63 |
| 6 | 3.917563 | 65.35 | 48.72 | 2197.42 | 214.38 | 102.19 | 43.97 | 61.09 | 1193.11 | 77.80 |
| 7 | 3.461438 | 73.96 | 74.43 | 2202.40 | 214.38 | 113.22 | 49.84 | 64.84 | 1160.36 | 76.88 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/53.10/54.89.
Separate latency-pass duration: 3.30 s; not used for throughput.

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

30.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.856073 | 66.39 | 80.21 | 2456.48 | 214.38 | 109.08 | 44.93 | 57.59 | 1128.20 | 79.52 |
| 2 | 3.496214 | 73.22 | 52.10 | 2456.97 | 214.38 | 114.48 | 47.84 | 62.99 | 1135.60 | 78.67 |
| 3 | 3.492735 | 73.29 | 83.65 | 2267.28 | 214.38 | 114.55 | 48.31 | 62.60 | 1192.76 | 75.10 |
| 4 | 3.471781 | 73.74 | 47.73 | 2267.77 | 214.38 | 114.17 | 47.16 | 64.98 | 1190.00 | 75.21 |
| 5 | 3.493783 | 73.27 | 63.74 | 2281.71 | 214.38 | 114.45 | 48.74 | 60.72 | 1155.26 | 77.86 |
| 6 | 3.470738 | 73.76 | 51.43 | 2278.09 | 214.38 | 114.53 | 47.82 | 62.55 | 1161.32 | 77.91 |
| 7 | 3.526196 | 72.60 | 71.29 | 2282.01 | 214.38 | 114.64 | 48.88 | 61.45 | 1156.29 | 76.95 |

Ready result delivery-interval P50/P95/P99 ms: 0.05/74.17/83.20.
Separate latency-pass duration: 4.22 s; not used for throughput.

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

Status: **complete**. Load 24.94 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 3.05 ± 0.05 | 1299.89 | 1298.69/1327.03/1331.79 | 2887.83 | 3486.48/4298.00 | 32/184 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 3.04 ± 0.05 | 1300.28 | 1302.15/1379.12/1382.48 | 2887.83 | 3486.46/4298.00 | 32/184 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 3.03 ± 0.02 | 1291.41 | 1297.75/1402.07/1414.51 | 2926.14 | 3486.46/4298.00 | 32/184 | PASS |
| managed-stream | 3.05 ± 0.01 | 1290.59 | 1298.53/1384.47/1389.70 | 2888.02 | 3486.46/4298.00 | 32/184 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.924051 | 2.93 | 1435.00 | 2886.75 | 3486.48 | 114.01 | 83.84 | 62.73 | 782.01 | 79.69 |
| 2 | 10.385490 | 3.08 | 1299.89 | 2886.91 | 3486.48 | 113.67 | 85.05 | 63.15 | 829.79 | 77.58 |
| 3 | 10.568336 | 3.03 | 1405.57 | 2886.92 | 3486.48 | 113.31 | 83.26 | 63.79 | 846.54 | 77.86 |
| 4 | 10.489263 | 3.05 | 1382.44 | 2887.41 | 3486.48 | 112.25 | 83.08 | 63.35 | 868.49 | 76.85 |
| 5 | 10.508125 | 3.05 | 1286.68 | 2887.42 | 3486.48 | 111.00 | 82.97 | 62.75 | 862.32 | 76.89 |
| 6 | 10.418022 | 3.07 | 1282.27 | 2887.82 | 3486.48 | 110.87 | 82.87 | 63.51 | 856.92 | 76.86 |
| 7 | 10.434880 | 3.07 | 1295.70 | 2887.83 | 3486.48 | 110.74 | 82.90 | 63.59 | 862.68 | 76.86 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1307.85/1327.66.
Separate latency-pass duration: 10.42 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

0.55% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.697076 | 2.99 | 1416.75 | 2886.76 | 3486.46 | 112.84 | 82.73 | 63.52 | 836.88 | 77.66 |
| 2 | 10.475631 | 3.05 | 1291.78 | 2886.82 | 3486.46 | 111.25 | 83.11 | 63.14 | 845.84 | 76.82 |
| 3 | 10.626510 | 3.01 | 1300.28 | 2886.93 | 3486.46 | 110.99 | 82.40 | 63.30 | 854.91 | 77.40 |
| 4 | 10.357966 | 3.09 | 1275.04 | 2887.36 | 3486.46 | 113.69 | 83.79 | 63.05 | 860.06 | 76.88 |
| 5 | 10.886824 | 2.94 | 1303.76 | 2887.42 | 3486.46 | 106.77 | 80.47 | 62.39 | 879.88 | 76.99 |
| 6 | 10.461790 | 3.06 | 1367.32 | 2887.77 | 3486.46 | 113.01 | 83.14 | 63.93 | 849.76 | 77.02 |
| 7 | 10.533242 | 3.04 | 1290.00 | 2887.83 | 3486.46 | 110.83 | 82.51 | 63.00 | 869.91 | 76.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1343.06/1379.56.
Separate latency-pass duration: 10.56 s; not used for throughput.

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

0.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.570405 | 3.03 | 1310.49 | 2924.99 | 3486.46 | 110.88 | 81.39 | 62.53 | 868.04 | 76.14 |
| 2 | 10.488124 | 3.05 | 1282.44 | 2915.66 | 3486.46 | 111.15 | 82.78 | 62.99 | 889.02 | 75.96 |
| 3 | 10.583928 | 3.02 | 1311.98 | 2897.15 | 3486.46 | 111.20 | 81.86 | 62.97 | 858.68 | 76.95 |
| 4 | 10.535115 | 3.04 | 1285.48 | 2920.88 | 3486.46 | 110.94 | 82.88 | 63.18 | 862.63 | 76.85 |
| 5 | 10.642049 | 3.01 | 1301.08 | 2897.59 | 3486.46 | 110.55 | 81.76 | 62.48 | 844.20 | 77.24 |
| 6 | 10.429244 | 3.07 | 1277.65 | 2921.29 | 3486.46 | 113.57 | 83.93 | 62.81 | 840.88 | 77.06 |
| 7 | 10.618047 | 3.01 | 1291.41 | 2926.14 | 3486.46 | 110.28 | 81.74 | 63.66 | 853.76 | 76.96 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1333.31/1403.80.
Separate latency-pass duration: 10.57 s; not used for throughput.

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

0.51% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10.479054 | 3.05 | 1271.64 | 2876.33 | 3486.46 | 109.78 | 82.13 | 63.05 | 892.99 | 75.83 |
| 2 | 10.472958 | 3.06 | 1275.33 | 2886.79 | 3486.46 | 111.10 | 82.37 | 62.18 | 888.68 | 75.80 |
| 3 | 10.537728 | 3.04 | 1277.53 | 2887.19 | 3486.46 | 110.33 | 81.69 | 62.74 | 858.66 | 76.86 |
| 4 | 10.495041 | 3.05 | 1299.36 | 2884.88 | 3486.46 | 110.24 | 82.46 | 62.55 | 870.54 | 76.83 |
| 5 | 10.589605 | 3.02 | 1291.50 | 2865.17 | 3486.46 | 111.78 | 82.84 | 63.31 | 863.79 | 77.43 |
| 6 | 10.580569 | 3.02 | 1297.07 | 2887.74 | 3486.46 | 111.27 | 82.32 | 62.63 | 859.20 | 77.07 |
| 7 | 10.487957 | 3.05 | 1290.59 | 2888.02 | 3486.46 | 112.62 | 83.60 | 62.76 | 855.62 | 77.22 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/1341.29/1385.07.
Separate latency-pass duration: 10.55 s; not used for throughput.

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

Status: **complete**. Load 18.87 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 2.10 ± 0.02 | 1897.49 | 1886.00/1915.79/1921.54 | 2865.18 | 3731.84/4524.00 | 32/168 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 2.10 ± 0.02 | 1881.74 | 1902.73/1984.88/1987.67 | 2865.19 | 3731.82/4524.00 | 32/168 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 2.09 ± 0.01 | 1891.10 | 1905.47/1944.98/1959.10 | 2903.58 | 3731.82/4524.00 | 32/168 | PASS |
| managed-stream | 2.09 ± 0.01 | 1903.59 | 1904.51/1997.04/2011.15 | 2865.46 | 3731.82/4524.00 | 32/168 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.959922 | 2.14 | 1880.36 | 2863.86 | 3731.84 | 112.34 | 88.39 | 63.69 | 803.22 | 75.64 |
| 2 | 15.184134 | 2.11 | 1958.43 | 2863.92 | 3731.84 | 110.30 | 87.97 | 63.79 | 800.56 | 76.89 |
| 3 | 15.225248 | 2.10 | 1871.71 | 2863.95 | 3731.84 | 112.47 | 88.10 | 64.73 | 802.78 | 76.88 |
| 4 | 15.240616 | 2.10 | 1897.49 | 2864.59 | 3731.84 | 110.57 | 87.83 | 63.64 | 790.71 | 76.99 |
| 5 | 15.327484 | 2.09 | 1965.77 | 2863.01 | 3731.84 | 111.29 | 87.87 | 63.68 | 797.99 | 76.91 |
| 6 | 15.308630 | 2.09 | 1883.92 | 2865.16 | 3731.84 | 112.43 | 88.49 | 64.23 | 783.34 | 77.10 |
| 7 | 15.361989 | 2.08 | 1979.58 | 2865.18 | 3731.84 | 109.80 | 88.34 | 63.90 | 791.90 | 77.01 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1894.44/1916.59.
Separate latency-pass duration: 15.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

-0.35% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.955150 | 2.14 | 1829.91 | 2863.88 | 3731.82 | 111.93 | 87.80 | 63.91 | 808.38 | 75.43 |
| 2 | 15.337296 | 2.09 | 1897.60 | 2863.92 | 3731.82 | 110.70 | 87.48 | 63.47 | 800.34 | 76.90 |
| 3 | 15.150270 | 2.11 | 1874.16 | 2863.95 | 3731.82 | 112.21 | 88.02 | 63.58 | 799.86 | 76.89 |
| 4 | 15.360657 | 2.08 | 1907.05 | 2864.48 | 3731.82 | 112.41 | 87.90 | 64.46 | 777.62 | 77.09 |
| 5 | 15.258572 | 2.10 | 1881.74 | 2864.64 | 3731.82 | 112.51 | 87.54 | 63.99 | 792.17 | 76.99 |
| 6 | 15.255619 | 2.10 | 1875.34 | 2827.27 | 3731.82 | 112.48 | 88.39 | 64.05 | 796.71 | 77.01 |
| 7 | 15.241999 | 2.10 | 1884.91 | 2865.19 | 3731.82 | 109.29 | 88.42 | 64.20 | 783.40 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1958.46/1985.24.
Separate latency-pass duration: 15.37 s; not used for throughput.

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

0.72% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.097942 | 2.12 | 1869.29 | 2860.09 | 3731.82 | 110.69 | 87.80 | 64.14 | 806.17 | 76.41 |
| 2 | 15.315954 | 2.09 | 1891.10 | 2888.07 | 3731.82 | 111.66 | 87.78 | 63.91 | 773.94 | 76.93 |
| 3 | 15.263588 | 2.10 | 1878.54 | 2892.98 | 3731.82 | 111.92 | 87.32 | 63.86 | 795.24 | 76.91 |
| 4 | 15.380825 | 2.08 | 1905.78 | 2898.01 | 3731.82 | 112.32 | 87.50 | 63.70 | 789.56 | 77.18 |
| 5 | 15.437314 | 2.07 | 1912.11 | 2888.89 | 3731.82 | 112.50 | 87.33 | 64.23 | 791.07 | 77.01 |
| 6 | 15.248279 | 2.10 | 1889.20 | 2897.26 | 3731.82 | 112.40 | 87.67 | 64.23 | 781.84 | 76.93 |
| 7 | 15.310893 | 2.09 | 1899.57 | 2903.58 | 3731.82 | 111.98 | 87.35 | 64.17 | 783.90 | 77.04 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/1909.90/1946.97.
Separate latency-pass duration: 15.30 s; not used for throughput.

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

0.63% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.516278 | 2.06 | 1950.27 | 2863.89 | 3731.82 | 111.02 | 87.69 | 63.76 | 785.93 | 77.62 |
| 2 | 15.483697 | 2.07 | 1910.60 | 2863.90 | 3731.82 | 112.26 | 87.21 | 64.29 | 776.30 | 77.60 |
| 3 | 15.383326 | 2.08 | 1897.30 | 2864.34 | 3731.82 | 109.84 | 87.43 | 63.64 | 790.37 | 76.98 |
| 4 | 15.336652 | 2.09 | 1903.59 | 2852.67 | 3731.82 | 112.12 | 87.94 | 63.40 | 785.28 | 77.21 |
| 5 | 15.339739 | 2.09 | 1895.49 | 2847.10 | 3731.82 | 112.01 | 88.33 | 63.97 | 795.73 | 77.00 |
| 6 | 15.296293 | 2.09 | 1977.94 | 2865.05 | 3731.82 | 109.82 | 88.35 | 63.77 | 787.62 | 76.90 |
| 7 | 15.222951 | 2.10 | 1878.78 | 2865.46 | 3731.82 | 111.97 | 87.94 | 63.51 | 791.08 | 76.96 |

Ready result delivery-interval P50/P95/P99 ms: 0.05/1937.44/1997.48.
Separate latency-pass duration: 15.39 s; not used for throughput.

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

Status: **complete**. Load 2.25 s. Device `cuda:0`, dtype `torch.float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 37.59 ± 1.90 | 101.82 | 107.12/137.76/166.68 | 1753.47 | 561.57/1006.00 | 128/1120 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 36.91 ± 2.93 | 105.57 | 108.52/113.42/114.98 | 1902.07 | 561.45/1006.00 | 128/1120 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 35.19 ± 2.96 | 107.02 | 108.29/113.80/119.62 | 1908.21 | 561.45/1006.00 | 128/1120 | PASS |
| managed-stream | 35.21 ± 2.72 | 105.81 | 128.73/153.46/158.06 | 1914.36 | 561.45/1006.00 | 128/1120 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.404776 | 37.59 | 110.20 | 1680.66 | 561.57 | 132.02 | 78.31 | 63.88 | 867.27 | 78.60 |
| 2 | 3.579519 | 35.76 | 100.65 | 1753.47 | 561.57 | 125.28 | 73.25 | 63.38 | 923.30 | 77.52 |
| 3 | 3.649006 | 35.08 | 142.82 | 1743.05 | 561.57 | 123.08 | 68.48 | 65.20 | 960.80 | 77.12 |
| 4 | 3.297237 | 38.82 | 101.82 | 1743.87 | 561.57 | 133.80 | 77.26 | 56.53 | 927.50 | 76.22 |
| 5 | 3.299460 | 38.79 | 99.22 | 1743.87 | 561.57 | 133.97 | 76.44 | 64.76 | 933.00 | 76.25 |
| 6 | 3.305208 | 38.73 | 103.59 | 1750.96 | 561.57 | 135.88 | 77.04 | 72.40 | 895.10 | 77.21 |
| 7 | 3.717100 | 34.44 | 98.34 | 1750.96 | 561.57 | 117.37 | 65.19 | 72.87 | 1001.67 | 77.28 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/122.73/141.55.
Separate latency-pass duration: 3.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

3.29% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.146859 | 30.87 | 106.05 | 1693.82 | 561.45 | 116.36 | 59.32 | 66.59 | 986.84 | 79.16 |
| 2 | 3.455957 | 37.04 | 111.39 | 1902.07 | 561.45 | 133.72 | 73.91 | 67.45 | 928.64 | 77.95 |
| 3 | 3.432556 | 37.29 | 104.86 | 1753.48 | 561.45 | 132.41 | 73.70 | 66.03 | 932.68 | 77.00 |
| 4 | 4.111075 | 31.14 | 105.57 | 1754.29 | 561.45 | 112.59 | 58.07 | 60.61 | 1046.50 | 76.17 |
| 5 | 3.713509 | 34.47 | 102.25 | 1761.32 | 561.45 | 128.49 | 66.19 | 63.21 | 984.05 | 76.29 |
| 6 | 3.413911 | 37.49 | 103.30 | 1761.51 | 561.45 | 134.85 | 71.67 | 64.83 | 937.09 | 77.11 |
| 7 | 3.468150 | 36.91 | 123.96 | 1761.51 | 561.45 | 133.28 | 73.46 | 63.78 | 917.95 | 77.11 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/111.41/113.58.
Separate latency-pass duration: 3.48 s; not used for throughput.

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

9.16% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.716598 | 34.44 | 107.02 | 1833.60 | 561.45 | 135.94 | 68.39 | 62.45 | 940.42 | 79.37 |
| 2 | 3.555882 | 36.00 | 121.12 | 1901.62 | 561.45 | 133.86 | 70.36 | 65.68 | 917.33 | 78.07 |
| 3 | 3.476010 | 36.82 | 104.63 | 1908.21 | 561.45 | 133.31 | 71.86 | 62.63 | 955.00 | 76.93 |
| 4 | 3.636993 | 35.19 | 102.95 | 1899.61 | 561.45 | 136.25 | 67.15 | 63.06 | 974.49 | 76.19 |
| 5 | 3.967915 | 32.26 | 128.75 | 1901.96 | 561.45 | 120.33 | 61.03 | 65.38 | 983.25 | 76.45 |
| 6 | 4.510242 | 28.38 | 107.13 | 1900.97 | 561.45 | 115.24 | 52.19 | 60.38 | 1117.97 | 76.92 |
| 7 | 3.537562 | 36.18 | 105.47 | 1903.31 | 561.45 | 136.10 | 71.04 | 64.89 | 919.29 | 77.21 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/111.49/113.84.
Separate latency-pass duration: 3.48 s; not used for throughput.

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

6.77% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.635273 | 35.21 | 105.81 | 1851.32 | 561.45 | 133.35 | 69.97 | 64.65 | 914.49 | 79.02 |
| 2 | 4.173347 | 30.67 | 110.13 | 1837.77 | 561.45 | 115.88 | 56.64 | 63.44 | 989.22 | 78.33 |
| 3 | 4.239072 | 30.20 | 110.59 | 1914.36 | 561.45 | 112.36 | 56.66 | 59.94 | 1046.07 | 76.52 |
| 4 | 3.486360 | 36.71 | 103.23 | 1843.32 | 561.45 | 132.56 | 69.68 | 63.84 | 965.53 | 76.28 |
| 5 | 3.546789 | 36.09 | 103.37 | 1907.70 | 561.45 | 131.37 | 68.88 | 63.83 | 981.98 | 76.79 |
| 6 | 3.526608 | 36.30 | 102.05 | 1838.90 | 561.45 | 131.88 | 68.74 | 65.91 | 967.89 | 76.95 |
| 7 | 3.823110 | 33.48 | 114.25 | 1909.06 | 561.45 | 125.45 | 66.31 | 63.26 | 1006.02 | 77.25 |

Ready result delivery-interval P50/P95/P99 ms: 0.05/143.95/153.58.
Separate latency-pass duration: 4.17 s; not used for throughput.

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

Status: **complete**. Load 20.64 s. Device `cuda:0`, dtype `float32`.
Input scope: predecoded 640x480 PIL packs; same effective inference batch in all supported modes. Input SHA256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 8.16 ± 0.83 | 457.03 | 459.94/471.16/471.70 | 3004.44 | 489.49/702.00 | 32/160 | PASS |
| raw-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| native-non-stream | 8.76 ± 0.75 | 441.26 | 457.51/472.66/475.31 | 3006.68 | 493.54/702.00 | 32/160 | PASS |
| native-stream | N/A | — | — | — | — | — | No upstream native source-stream engine |
| managed-non-stream | 8.73 ± 0.89 | 417.38 | 632.29/694.69/699.32 | 3042.00 | 532.97/702.00 | 32/160 | PASS |
| managed-stream | 8.53 ± 0.82 | 446.24 | 457.88/464.98/465.32 | 3041.68 | 501.83/702.00 | 32/160 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.067112 | 7.87 | 416.59 | 3003.42 | 456.20 | 106.54 | 14.14 | 36.18 | 585.00 | 72.77 |
| 2 | 3.920175 | 8.16 | 753.17 | 3003.97 | 471.87 | 111.71 | 15.00 | 37.04 | 585.00 | 74.02 |
| 3 | 3.484025 | 9.18 | 484.66 | 3003.99 | 453.29 | 115.18 | 16.07 | 36.56 | 585.00 | 74.38 |
| 4 | 4.038765 | 7.92 | 409.39 | 3004.27 | 453.73 | 108.57 | 14.20 | 36.60 | 585.00 | 76.00 |
| 5 | 3.604585 | 8.88 | 639.87 | 3004.27 | 471.82 | 113.95 | 16.22 | 37.30 | 585.00 | 76.00 |
| 6 | 3.454769 | 9.26 | 457.03 | 3004.44 | 466.42 | 115.81 | 16.16 | 37.53 | 585.00 | 77.00 |
| 7 | 4.594248 | 6.97 | 446.31 | 3004.44 | 489.49 | 107.20 | 12.67 | 36.96 | 585.00 | 77.19 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/466.26/471.22.
Separate latency-pass duration: 3.66 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

### native-non-stream

1.29% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.561888 | 8.98 | 486.32 | 3006.08 | 493.54 | 114.46 | 15.81 | 36.86 | 585.00 | 73.00 |
| 2 | 4.323006 | 7.40 | 437.64 | 3003.91 | 492.78 | 108.27 | 13.84 | 36.14 | 585.00 | 74.00 |
| 3 | 4.446679 | 7.20 | 463.00 | 3006.68 | 484.10 | 109.26 | 12.59 | 36.34 | 585.00 | 74.99 |
| 4 | 3.611900 | 8.86 | 417.66 | 3004.27 | 482.75 | 115.01 | 15.42 | 36.91 | 585.00 | 75.98 |
| 5 | 3.651082 | 8.76 | 451.32 | 3004.29 | 475.81 | 114.92 | 15.59 | 37.29 | 585.00 | 76.00 |
| 6 | 3.639011 | 8.79 | 441.26 | 3004.44 | 480.22 | 116.21 | 15.55 | 37.72 | 585.00 | 77.00 |
| 7 | 3.680492 | 8.69 | 429.22 | 3004.45 | 462.36 | 114.59 | 15.58 | 37.67 | 585.00 | 77.84 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/464.81/473.01.
Separate latency-pass duration: 3.67 s; not used for throughput.

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

-9.25% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.536793 | 9.05 | 413.22 | 3041.04 | 487.41 | 115.42 | 16.39 | 35.94 | 585.00 | 73.00 |
| 2 | 3.472750 | 9.21 | 408.47 | 3041.39 | 532.97 | 115.66 | 16.43 | 36.64 | 585.00 | 74.00 |
| 3 | 3.948536 | 8.10 | 704.37 | 3041.62 | 474.34 | 112.22 | 14.58 | 36.22 | 585.00 | 75.00 |
| 4 | 3.665381 | 8.73 | 469.22 | 3041.88 | 458.55 | 115.13 | 15.47 | 36.67 | 585.00 | 75.45 |
| 5 | 4.418451 | 7.24 | 417.38 | 3041.85 | 440.50 | 107.09 | 13.18 | 36.70 | 585.00 | 76.06 |
| 6 | 4.590598 | 6.97 | 435.93 | 3041.99 | 465.14 | 105.46 | 12.63 | 36.78 | 585.00 | 77.00 |
| 7 | 3.650476 | 8.77 | 408.31 | 3042.00 | 468.17 | 115.32 | 16.16 | 37.44 | 585.00 | 77.98 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/662.24/695.31.
Separate latency-pass duration: 4.62 s; not used for throughput.

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

3.34% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.202981 | 7.61 | 444.21 | 3003.66 | 485.28 | 109.32 | 13.59 | 35.90 | 585.00 | 73.14 |
| 2 | 3.731286 | 8.58 | 613.52 | 3003.77 | 467.38 | 113.18 | 15.33 | 36.04 | 585.00 | 73.97 |
| 3 | 3.596266 | 8.90 | 422.74 | 3041.68 | 486.74 | 115.60 | 15.83 | 37.44 | 585.00 | 75.00 |
| 4 | 4.391465 | 7.29 | 437.01 | 3004.27 | 488.12 | 106.83 | 13.39 | 36.32 | 585.00 | 75.00 |
| 5 | 3.750824 | 8.53 | 578.64 | 3004.37 | 501.83 | 113.88 | 15.29 | 37.66 | 585.00 | 76.73 |
| 6 | 3.603226 | 8.88 | 446.24 | 3004.44 | 468.68 | 114.89 | 15.80 | 37.45 | 585.00 | 76.98 |
| 7 | 4.654571 | 6.87 | 472.65 | 3004.53 | 470.66 | 104.61 | 12.51 | 37.15 | 585.00 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.06/463.65/464.73.
Separate latency-pass duration: 3.62 s; not used for throughput.

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

Status: **complete**. Load 0.29 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 82.58 ± 10.53 | 1453.06 | 1558.98/1558.98/1558.98 | 2250.71 | 34.30/62.00 | 120/658 | PASS |
| raw-stream | 84.59 ± 16.42 | 60.13 | 1810.57/1810.57/1810.57 | 2250.71 | 34.30/62.00 | 120/658 | PASS |
| native-non-stream | 79.65 ± 6.30 | 1506.43 | 2316.54/2316.54/2316.54 | 2250.71 | 34.30/62.00 | 120/658 | PASS |
| native-stream | 74.98 ± 8.64 | 1600.39 | 1605.68/1605.68/1605.68 | 2257.34 | 34.25/62.00 | 120/658 | PASS |
| managed-non-stream | 71.50 ± 6.22 | 1678.13 | 1666.01/1666.01/1666.01 | 2257.35 | 24.66/62.00 | 120/658 | PASS |
| managed-stream | 70.17 ± 11.79 | 99.84 | 1686.81/1686.81/1686.81 | 2257.35 | 24.66/62.00 | 120/658 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.419793 | 84.52 | 1419.71 | 2227.52 | 34.30 | 155.85 | 34.91 | 40.30 | 834.13 | 77.00 |
| 2 | 1.453138 | 82.58 | 1453.06 | 2246.37 | 34.30 | 155.29 | 33.33 | 40.31 | 839.38 | 77.58 |
| 3 | 1.418426 | 84.60 | 1418.35 | 2249.19 | 34.30 | 146.03 | 32.26 | 40.61 | 886.96 | 77.26 |
| 4 | 1.829756 | 65.58 | 1829.66 | 2250.71 | 34.30 | 120.15 | 26.28 | 39.47 | 875.17 | 77.00 |
| 5 | 2.033957 | 59.00 | 2033.86 | 2250.71 | 34.30 | 116.24 | 30.09 | 38.21 | 637.03 | 77.00 |
| 6 | 1.434876 | 83.63 | 1434.79 | 2250.66 | 34.30 | 147.27 | 32.54 | 40.11 | 894.38 | 77.00 |
| 7 | 1.468989 | 81.69 | 1468.88 | 2250.66 | 34.30 | 145.63 | 31.72 | 39.79 | 823.20 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.05.
Separate latency-pass duration: 1.56 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.5289987666771292, 'inference': 8.424329291661556, 'postprocess': 1.3394125916647681}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.57% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.397484 | 85.87 | 61.43 | 2241.21 | 34.30 | 147.64 | 32.75 | 40.71 | 915.00 | 77.00 |
| 2 | 2.351001 | 51.04 | 99.01 | 2245.78 | 34.30 | 104.35 | 24.50 | 38.09 | 655.00 | 77.47 |
| 3 | 1.380527 | 86.92 | 54.64 | 2249.78 | 34.30 | 148.50 | 33.00 | 40.32 | 895.43 | 77.00 |
| 4 | 1.405148 | 85.40 | 55.02 | 2250.71 | 34.30 | 148.37 | 32.21 | 40.75 | 905.62 | 77.00 |
| 5 | 1.492109 | 80.42 | 60.13 | 2250.71 | 34.30 | 149.58 | 34.00 | 39.93 | 808.80 | 77.00 |
| 6 | 1.418596 | 84.59 | 59.94 | 2250.33 | 34.30 | 148.09 | 32.42 | 40.09 | 861.88 | 77.00 |
| 7 | 2.332046 | 51.46 | 97.13 | 2250.66 | 34.30 | 102.56 | 26.20 | 37.93 | 685.29 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.12/22.58/27.26.
Separate latency-pass duration: 1.81 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.5593766416373, 'inference': 8.131848791667077, 'postprocess': 1.3386440083081652}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

4.38% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.481973 | 80.97 | 1481.87 | 2241.23 | 34.30 | 147.01 | 32.58 | 40.61 | 873.75 | 77.00 |
| 2 | 1.873849 | 64.04 | 1873.72 | 2245.47 | 34.30 | 137.76 | 31.43 | 38.97 | 772.50 | 77.97 |
| 3 | 1.471030 | 81.58 | 1470.93 | 2249.84 | 34.30 | 144.23 | 34.48 | 40.43 | 901.80 | 77.00 |
| 4 | 1.506526 | 79.65 | 1506.43 | 2250.66 | 34.30 | 144.47 | 35.00 | 40.43 | 848.40 | 77.00 |
| 5 | 1.476622 | 81.27 | 1476.49 | 2250.71 | 34.30 | 145.40 | 34.20 | 40.28 | 873.60 | 77.00 |
| 6 | 1.599606 | 75.02 | 1599.51 | 2250.28 | 34.30 | 149.37 | 32.85 | 39.37 | 813.33 | 77.00 |
| 7 | 1.604693 | 74.78 | 1604.60 | 2250.66 | 34.30 | 142.18 | 35.22 | 39.58 | 764.44 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 2.32 s; not used for throughput.

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

10.17% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.280712 | 52.62 | 2280.62 | 2241.72 | 34.25 | 111.64 | 22.20 | 38.60 | 815.57 | 77.00 |
| 2 | 1.667762 | 71.95 | 1667.68 | 1940.80 | 34.25 | 144.46 | 31.36 | 39.38 | 744.64 | 77.93 |
| 3 | 1.584059 | 75.75 | 1583.99 | 2249.89 | 34.25 | 145.96 | 28.00 | 39.87 | 908.33 | 77.00 |
| 4 | 1.557489 | 77.05 | 1557.42 | 2257.34 | 34.25 | 146.72 | 34.04 | 39.62 | 775.38 | 77.00 |
| 5 | 1.603535 | 74.83 | 1603.43 | 2250.76 | 34.25 | 146.12 | 28.70 | 40.02 | 870.00 | 77.00 |
| 6 | 1.580734 | 75.91 | 1580.66 | 1949.41 | 34.25 | 145.66 | 33.44 | 39.27 | 746.11 | 77.00 |
| 7 | 1.600472 | 74.98 | 1600.39 | 2250.66 | 34.25 | 144.71 | 31.41 | 39.52 | 840.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
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


### managed-non-stream

15.49% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.069395 | 57.99 | 2069.29 | 2242.00 | 24.66 | 122.34 | 29.88 | 38.54 | 647.27 | 77.09 |
| 2 | 1.678252 | 71.50 | 1678.13 | 1940.61 | 24.66 | 144.47 | 32.82 | 39.44 | 729.11 | 77.89 |
| 3 | 1.917653 | 62.58 | 1917.54 | 2250.09 | 24.66 | 132.94 | 30.00 | 39.00 | 725.81 | 77.00 |
| 4 | 1.631007 | 73.57 | 1630.92 | 2257.35 | 24.66 | 146.34 | 33.85 | 39.60 | 745.00 | 77.00 |
| 5 | 1.637108 | 73.30 | 1637.01 | 1944.29 | 24.66 | 145.75 | 32.04 | 39.39 | 752.22 | 77.00 |
| 6 | 1.855725 | 64.66 | 1855.62 | 1944.65 | 24.66 | 134.64 | 31.90 | 38.39 | 665.50 | 76.83 |
| 7 | 1.655688 | 72.48 | 1655.60 | 1944.41 | 24.66 | 144.99 | 32.56 | 38.87 | 733.89 | 77.00 |

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


### managed-stream

17.65% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.670436 | 71.84 | 99.32 | 1937.73 | 24.66 | 145.02 | 31.86 | 39.20 | 750.00 | 77.64 |
| 2 | 1.715101 | 69.97 | 90.26 | 1938.07 | 24.66 | 145.35 | 32.55 | 39.08 | 676.03 | 77.83 |
| 3 | 2.514418 | 47.72 | 148.08 | 2257.06 | 24.66 | 107.07 | 24.86 | 37.57 | 587.84 | 77.00 |
| 4 | 1.710028 | 70.17 | 103.37 | 2257.35 | 24.66 | 143.67 | 32.59 | 39.08 | 689.48 | 77.00 |
| 5 | 1.688524 | 71.07 | 101.27 | 1951.08 | 24.66 | 145.33 | 31.57 | 39.14 | 763.93 | 77.00 |
| 6 | 2.614650 | 45.90 | 97.43 | 1944.53 | 24.66 | 104.25 | 25.32 | 37.37 | 597.00 | 77.00 |
| 7 | 1.677443 | 71.54 | 99.84 | 1951.24 | 24.66 | 144.81 | 34.43 | 39.16 | 708.21 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.01/15.52/26.10.
Separate latency-pass duration: 1.69 s; not used for throughput.

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

Status: **complete**. Load 0.29 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 148.73 ± 28.62 | 806.75 | 814.79/814.79/814.79 | 2235.28 | 90.50/146.00 | 120/658 | PASS |
| raw-stream | 153.87 ± 29.88 | 82.57 | 765.17/765.17/765.17 | 2120.26 | 90.50/146.00 | 120/658 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 130.55 ± 18.45 | 919.07 | 888.55/888.55/888.55 | 2118.34 | 63.09/146.00 | 120/658 | PASS |
| managed-stream | 124.02 ± 23.16 | 116.53 | 882.99/882.99/882.99 | 2089.29 | 63.09/146.00 | 120/658 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.850177 | 141.15 | 850.04 | 2235.28 | 90.50 | 176.89 | 32.29 | 47.01 | 1018.93 | 77.64 |
| 2 | 0.782640 | 153.33 | 782.56 | 1929.14 | 90.50 | 171.28 | 28.38 | 52.14 | 1248.46 | 78.77 |
| 3 | 0.806267 | 148.83 | 806.18 | 1929.60 | 90.50 | 173.18 | 25.36 | 54.38 | 1290.00 | 78.93 |
| 4 | 0.757214 | 158.48 | 757.13 | 1930.41 | 90.50 | 173.69 | 26.38 | 53.70 | 1262.31 | 79.31 |
| 5 | 0.806832 | 148.73 | 806.75 | 1930.43 | 90.50 | 177.68 | 24.69 | 52.17 | 1305.00 | 79.69 |
| 6 | 1.469603 | 81.65 | 1469.50 | 2180.77 | 90.50 | 103.09 | 17.81 | 46.60 | 1173.57 | 78.00 |
| 7 | 1.118912 | 107.25 | 1118.83 | 2120.23 | 90.50 | 129.28 | 30.18 | 42.54 | 720.00 | 77.41 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 0.81 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.603693791673777, 'inference': 3.1777930916822092, 'postprocess': 1.5698752249932113}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

1.92% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.584869 | 75.72 | 144.55 | 2045.40 | 90.50 | 90.76 | 16.87 | 42.48 | 1074.13 | 77.57 |
| 2 | 0.799783 | 150.04 | 82.57 | 1910.49 | 90.50 | 171.58 | 31.31 | 50.24 | 1095.00 | 78.38 |
| 3 | 0.880673 | 136.26 | 80.28 | 1929.60 | 90.50 | 155.04 | 25.53 | 50.57 | 1194.00 | 79.00 |
| 4 | 0.771723 | 155.50 | 83.56 | 1929.79 | 90.50 | 177.17 | 32.92 | 48.84 | 1068.46 | 78.92 |
| 5 | 0.779886 | 153.87 | 79.59 | 1930.43 | 90.50 | 173.85 | 26.69 | 52.02 | 1277.31 | 79.69 |
| 6 | 0.759804 | 157.94 | 81.26 | 1932.46 | 90.50 | 175.03 | 32.08 | 53.55 | 1226.54 | 78.85 |
| 7 | 0.754049 | 159.14 | 87.35 | 2120.26 | 90.50 | 173.34 | 31.38 | 50.21 | 1226.54 | 77.85 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/26.02/35.60.
Separate latency-pass duration: 0.77 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 4.056809100003041, 'inference': 5.71954581666508, 'postprocess': 2.2392039916667272}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

15.76% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.359509 | 88.27 | 1359.38 | 1927.25 | 63.09 | 123.61 | 28.24 | 40.56 | 637.14 | 77.43 |
| 2 | 0.915061 | 131.14 | 914.96 | 1888.50 | 63.09 | 168.86 | 29.00 | 46.10 | 1020.00 | 78.00 |
| 3 | 0.933313 | 128.57 | 933.20 | 1929.60 | 63.09 | 163.98 | 26.00 | 49.25 | 1170.00 | 79.00 |
| 4 | 1.128815 | 106.31 | 1128.72 | 1892.27 | 63.09 | 145.74 | 31.67 | 42.93 | 725.83 | 78.83 |
| 5 | 0.919219 | 130.55 | 919.07 | 1930.43 | 63.09 | 167.78 | 21.75 | 54.24 | 1346.25 | 79.75 |
| 6 | 0.882974 | 135.90 | 882.84 | 1889.73 | 63.09 | 168.42 | 28.93 | 49.08 | 1125.00 | 78.87 |
| 7 | 0.866953 | 138.42 | 866.86 | 2118.34 | 63.09 | 164.86 | 27.60 | 51.55 | 1305.00 | 77.73 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 0.89 s; not used for throughput.

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

14.30% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.971711 | 123.49 | 119.43 | 1889.21 | 63.09 | 166.63 | 29.44 | 46.36 | 922.50 | 77.81 |
| 2 | 0.967578 | 124.02 | 116.58 | 1889.49 | 63.09 | 166.91 | 25.56 | 44.10 | 945.00 | 77.88 |
| 3 | 1.158509 | 103.58 | 116.53 | 1889.30 | 63.09 | 139.16 | 23.06 | 44.00 | 1087.50 | 79.00 |
| 4 | 1.633127 | 73.48 | 185.01 | 1892.26 | 63.09 | 100.51 | 21.00 | 41.35 | 724.57 | 78.91 |
| 5 | 0.902290 | 132.99 | 115.55 | 1889.64 | 63.09 | 170.40 | 24.29 | 51.88 | 1312.50 | 79.21 |
| 6 | 0.880386 | 136.30 | 113.22 | 1889.73 | 63.09 | 168.09 | 25.50 | 46.18 | 1084.29 | 79.00 |
| 7 | 0.868369 | 138.19 | 112.69 | 2089.29 | 63.09 | 166.28 | 27.47 | 53.34 | 1263.00 | 77.60 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/27.24/39.17.
Separate latency-pass duration: 0.88 s; not used for throughput.

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

Status: **complete**. Load 0.36 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 72.53 ± 4.72 | 1654.30 | 1736.99/1736.99/1736.99 | 2593.73 | 64.97/90.00 | 120/658 | PASS |
| raw-stream | 69.02 ± 12.92 | 60.56 | 1637.64/1637.64/1637.64 | 2593.64 | 64.97/90.00 | 120/658 | PASS |
| native-non-stream | 69.42 ± 3.51 | 1728.62 | 1730.84/1730.84/1730.84 | 2593.43 | 64.97/90.00 | 120/658 | PASS |
| native-stream | 64.61 ± 0.94 | 1857.20 | 1909.33/1909.33/1909.33 | 2592.64 | 64.94/90.00 | 120/658 | PASS |
| managed-non-stream | 49.56 ± 10.87 | 2421.05 | 2906.25/2906.25/2906.25 | 2334.95 | 56.35/90.00 | 120/658 | PASS |
| managed-stream | 62.54 ± 7.07 | 92.38 | 1859.91/1859.91/1859.91 | 2281.12 | 56.35/90.00 | 120/658 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.673591 | 71.70 | 1673.50 | 2573.81 | 64.97 | 147.70 | 35.14 | 38.31 | 744.64 | 74.00 |
| 2 | 1.965304 | 61.06 | 1965.23 | 2592.52 | 64.97 | 124.53 | 32.29 | 38.16 | 681.77 | 76.00 |
| 3 | 1.642078 | 73.08 | 1642.00 | 2592.60 | 64.97 | 141.79 | 32.32 | 39.00 | 798.75 | 76.00 |
| 4 | 1.654407 | 72.53 | 1654.30 | 2593.66 | 64.97 | 141.55 | 35.21 | 39.57 | 724.29 | 77.61 |
| 5 | 1.768945 | 67.84 | 1768.80 | 2593.73 | 64.97 | 142.16 | 30.80 | 39.39 | 777.00 | 77.93 |
| 6 | 1.619895 | 74.08 | 1619.80 | 2285.05 | 64.97 | 142.67 | 35.22 | 39.52 | 762.78 | 78.00 |
| 7 | 1.620810 | 74.04 | 1620.73 | 2285.05 | 64.97 | 142.80 | 32.67 | 39.81 | 804.44 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.74 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.5536125833136794, 'inference': 10.57752965000418, 'postprocess': 1.3011934583232687}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

3.89% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.738698 | 69.02 | 62.96 | 2582.04 | 64.97 | 141.95 | 33.52 | 38.26 | 772.76 | 74.00 |
| 2 | 2.240471 | 53.56 | 54.71 | 2592.52 | 64.97 | 112.64 | 24.89 | 37.84 | 787.29 | 76.00 |
| 3 | 1.646651 | 72.88 | 61.13 | 2592.60 | 64.97 | 141.37 | 31.68 | 39.26 | 807.86 | 76.00 |
| 4 | 2.663665 | 45.05 | 57.98 | 2593.64 | 64.97 | 103.04 | 27.12 | 37.35 | 588.29 | 77.00 |
| 5 | 1.720984 | 69.73 | 60.56 | 2282.18 | 64.97 | 143.44 | 34.34 | 39.43 | 730.86 | 78.00 |
| 6 | 2.650344 | 45.28 | 92.82 | 2285.03 | 64.97 | 103.85 | 23.55 | 37.93 | 678.57 | 78.00 |
| 7 | 1.626585 | 73.77 | 55.35 | 2285.05 | 64.97 | 141.65 | 34.54 | 39.70 | 770.89 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 12.68/15.97/27.93.
Separate latency-pass duration: 1.64 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.4493271416654352, 'inference': 11.022237166673676, 'postprocess': 1.4278856166773342}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

3.23% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.727729 | 69.46 | 1727.61 | 2584.98 | 64.97 | 140.07 | 35.45 | 38.74 | 750.52 | 74.03 |
| 2 | 1.795948 | 66.82 | 1795.86 | 2592.21 | 64.97 | 144.54 | 34.90 | 38.41 | 729.00 | 75.90 |
| 3 | 1.681741 | 71.35 | 1681.67 | 2592.60 | 64.97 | 142.18 | 33.18 | 39.55 | 824.46 | 76.21 |
| 4 | 1.961321 | 61.18 | 1961.23 | 2593.43 | 64.97 | 146.14 | 33.12 | 38.75 | 687.27 | 77.00 |
| 5 | 1.710593 | 70.15 | 1710.49 | 2282.79 | 64.97 | 140.61 | 34.28 | 39.68 | 779.48 | 78.00 |
| 6 | 1.834441 | 65.42 | 1834.24 | 2284.97 | 64.97 | 137.95 | 35.16 | 39.33 | 754.35 | 78.00 |
| 7 | 1.728716 | 69.42 | 1728.62 | 2285.05 | 64.97 | 140.18 | 33.66 | 39.42 | 808.97 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.73 s; not used for throughput.

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

11.58% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.823011 | 65.83 | 1822.86 | 2585.48 | 64.94 | 139.91 | 30.94 | 38.17 | 781.45 | 74.58 |
| 2 | 1.846081 | 65.00 | 1846.00 | 2283.89 | 64.94 | 140.17 | 34.68 | 38.23 | 690.97 | 75.29 |
| 3 | 1.832179 | 65.50 | 1832.10 | 2592.64 | 64.94 | 138.88 | 31.45 | 38.73 | 707.42 | 76.81 |
| 4 | 1.857272 | 64.61 | 1857.20 | 2285.89 | 64.94 | 141.31 | 33.84 | 38.52 | 687.10 | 77.00 |
| 5 | 1.907851 | 62.90 | 1907.77 | 2282.79 | 64.94 | 137.09 | 30.47 | 39.27 | 743.91 | 78.00 |
| 6 | 1.859463 | 64.53 | 1859.38 | 2274.41 | 64.94 | 139.29 | 33.23 | 39.03 | 693.87 | 78.00 |
| 7 | 1.858092 | 64.58 | 1858.02 | 2285.06 | 64.94 | 141.27 | 32.48 | 39.11 | 689.03 | 78.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.02.
Separate latency-pass duration: 1.91 s; not used for throughput.

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

44.67% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.421159 | 49.56 | 2421.05 | 2334.95 | 56.35 | 115.85 | 25.87 | 37.03 | 704.21 | 74.95 |
| 2 | 1.872101 | 64.10 | 1872.01 | 2279.26 | 56.35 | 140.20 | 35.39 | 37.89 | 667.26 | 75.00 |
| 3 | 2.424768 | 49.49 | 2424.65 | 2289.61 | 56.35 | 120.09 | 26.59 | 37.71 | 678.46 | 76.92 |
| 4 | 1.884823 | 63.67 | 1884.73 | 2281.12 | 56.35 | 139.02 | 34.31 | 38.67 | 678.28 | 77.00 |
| 5 | 2.973810 | 40.35 | 2973.70 | 2274.27 | 56.35 | 101.23 | 24.95 | 37.22 | 593.18 | 78.00 |
| 6 | 1.892707 | 63.40 | 1892.56 | 2274.42 | 56.35 | 138.73 | 33.06 | 39.20 | 702.19 | 78.00 |
| 7 | 3.030495 | 39.60 | 3030.35 | 2274.46 | 56.35 | 101.65 | 24.52 | 37.01 | 589.89 | 77.59 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.07.
Separate latency-pass duration: 2.91 s; not used for throughput.

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

17.44% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.309886 | 51.95 | 154.17 | 2276.64 | 56.35 | 119.92 | 29.00 | 36.91 | 627.97 | 75.00 |
| 2 | 1.833239 | 65.46 | 87.97 | 2277.93 | 56.35 | 138.03 | 33.00 | 38.37 | 705.00 | 75.00 |
| 3 | 2.484924 | 48.29 | 140.34 | 2280.36 | 56.35 | 118.85 | 27.26 | 37.28 | 608.08 | 76.97 |
| 4 | 1.942865 | 61.76 | 100.84 | 2281.12 | 56.35 | 139.93 | 31.97 | 38.22 | 647.81 | 77.00 |
| 5 | 1.826719 | 65.69 | 90.55 | 2274.27 | 56.35 | 139.45 | 34.06 | 39.18 | 663.39 | 78.00 |
| 6 | 1.842478 | 65.13 | 92.38 | 2274.42 | 56.35 | 139.14 | 32.23 | 39.01 | 699.68 | 78.00 |
| 7 | 1.918765 | 62.54 | 90.11 | 2274.46 | 56.35 | 135.98 | 34.03 | 38.87 | 654.38 | 77.53 |

Ready result delivery-interval P50/P95/P99 ms: 14.30/17.54/30.76.
Separate latency-pass duration: 1.86 s; not used for throughput.

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

Status: **complete**. Load 0.30 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 133.06 ± 24.90 | 901.73 | 1440.88/1440.88/1440.88 | 2560.11 | 125.73/172.00 | 120/658 | PASS |
| raw-stream | 134.43 ± 8.17 | 86.84 | 1467.17/1467.17/1467.17 | 2553.27 | 125.73/172.00 | 120/658 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 121.21 ± 18.15 | 989.92 | 987.95/987.95/987.95 | 2542.73 | 99.31/172.00 | 120/658 | PASS |
| managed-stream | 115.32 ± 9.71 | 120.65 | 992.50/992.50/992.50 | 2541.45 | 99.31/172.00 | 120/658 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.936059 | 128.20 | 935.89 | 2560.11 | 125.73 | 172.73 | 29.44 | 44.08 | 990.00 | 76.94 |
| 2 | 1.143409 | 104.95 | 1143.30 | 2519.93 | 125.73 | 140.07 | 27.28 | 43.64 | 1020.00 | 77.78 |
| 3 | 1.602705 | 74.87 | 1602.53 | 2520.00 | 125.73 | 100.00 | 23.39 | 41.63 | 671.09 | 77.04 |
| 4 | 0.901824 | 133.06 | 901.73 | 2520.30 | 125.73 | 165.98 | 27.80 | 47.93 | 1062.00 | 78.00 |
| 5 | 0.876294 | 136.94 | 876.19 | 2520.96 | 125.73 | 168.09 | 26.40 | 49.38 | 1080.00 | 78.00 |
| 6 | 0.834403 | 143.82 | 834.32 | 2521.20 | 125.73 | 165.09 | 29.14 | 48.37 | 1140.00 | 77.43 |
| 7 | 0.851407 | 140.94 | 851.32 | 2521.22 | 125.73 | 166.60 | 25.29 | 47.28 | 1074.64 | 77.00 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.7184613333332284, 'inference': 4.158157425005508, 'postprocess': 1.3475647083320534}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.02% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.824264 | 145.58 | 86.84 | 2512.29 | 125.73 | 165.42 | 31.36 | 43.67 | 1078.93 | 77.00 |
| 2 | 0.906129 | 132.43 | 86.13 | 2553.27 | 125.73 | 162.79 | 30.13 | 43.99 | 910.00 | 77.40 |
| 3 | 1.012227 | 118.55 | 164.44 | 2520.00 | 125.73 | 154.27 | 30.19 | 43.11 | 724.69 | 77.56 |
| 4 | 0.892631 | 134.43 | 84.60 | 2541.16 | 125.73 | 164.47 | 30.53 | 45.87 | 914.00 | 77.93 |
| 5 | 0.881200 | 136.18 | 84.07 | 2520.96 | 125.73 | 163.34 | 31.33 | 45.03 | 1028.00 | 78.00 |
| 6 | 0.892883 | 134.40 | 88.02 | 2541.43 | 125.73 | 162.17 | 30.60 | 46.44 | 1053.00 | 77.20 |
| 7 | 0.865763 | 138.61 | 96.16 | 2521.22 | 125.73 | 166.07 | 28.20 | 48.15 | 1125.00 | 77.07 |

Ready result delivery-interval P50/P95/P99 ms: 0.03/51.77/65.41.
Separate latency-pass duration: 1.47 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.6302945250117773, 'inference': 3.586406916656415, 'postprocess': 1.1790952166696418}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

10.63% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.939127 | 127.78 | 939.02 | 2513.21 | 99.31 | 166.87 | 26.81 | 50.62 | 1170.00 | 77.06 |
| 2 | 1.021866 | 117.43 | 1021.77 | 2542.73 | 99.31 | 164.04 | 29.82 | 42.56 | 900.00 | 77.06 |
| 3 | 0.990028 | 121.21 | 989.92 | 2520.02 | 99.31 | 161.38 | 27.06 | 44.60 | 1020.00 | 77.88 |
| 4 | 0.997655 | 120.28 | 997.55 | 2541.17 | 99.31 | 162.89 | 30.65 | 43.89 | 923.82 | 77.88 |
| 5 | 1.603211 | 74.85 | 1603.09 | 2520.97 | 99.31 | 99.75 | 16.38 | 44.19 | 1021.88 | 78.00 |
| 6 | 0.968333 | 123.92 | 968.24 | 2541.44 | 99.31 | 164.77 | 29.12 | 44.97 | 914.06 | 77.00 |
| 7 | 0.982355 | 122.16 | 982.26 | 2521.22 | 99.31 | 161.69 | 27.00 | 45.15 | 1065.94 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 0.99 s; not used for throughput.

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

12.75% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.992571 | 120.90 | 135.27 | 2539.41 | 99.31 | 167.24 | 26.53 | 46.28 | 1085.29 | 77.06 |
| 2 | 1.068974 | 112.26 | 114.50 | 2539.78 | 99.31 | 155.01 | 27.29 | 42.63 | 878.82 | 77.00 |
| 3 | 1.040553 | 115.32 | 141.39 | 2540.88 | 99.31 | 161.12 | 27.00 | 43.64 | 960.88 | 77.88 |
| 4 | 1.016837 | 118.01 | 116.19 | 2541.17 | 99.31 | 163.19 | 26.76 | 42.57 | 930.00 | 77.82 |
| 5 | 1.308376 | 91.72 | 178.92 | 2541.18 | 99.31 | 131.25 | 28.00 | 39.63 | 641.25 | 77.30 |
| 6 | 1.022963 | 117.31 | 120.45 | 2541.45 | 99.31 | 162.75 | 29.65 | 43.14 | 843.53 | 77.00 |
| 7 | 1.053384 | 113.92 | 120.65 | 2541.44 | 99.31 | 161.31 | 27.06 | 42.96 | 877.06 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/32.02/47.64.
Separate latency-pass duration: 0.99 s; not used for throughput.

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

Status: **complete**. Load 0.30 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 66.19 ± 10.05 | 1812.79 | 1855.96/1855.96/1855.96 | 2264.02 | 62.86/90.00 | 120/648 | PASS |
| raw-stream | 62.80 ± 8.93 | 57.19 | 1770.26/1770.26/1770.26 | 2264.02 | 62.86/90.00 | 120/648 | PASS |
| native-non-stream | 59.62 ± 7.04 | 2012.80 | 2862.82/2862.82/2862.82 | 2264.02 | 62.86/90.00 | 120/648 | PASS |
| native-stream | 57.06 ± 5.14 | 2102.81 | 1982.12/1982.12/1982.12 | 2264.07 | 62.80/90.00 | 120/648 | PASS |
| managed-non-stream | 57.12 ± 5.64 | 2100.67 | 1998.73/1998.73/1998.73 | 1958.93 | 55.57/90.00 | 120/648 | PASS |
| managed-stream | 57.49 ± 6.32 | 93.62 | 2046.08/2046.08/2046.08 | 1957.06 | 55.57/90.00 | 120/648 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.665150 | 45.03 | 2665.05 | 2257.28 | 62.86 | 114.29 | 27.80 | 36.95 | 651.22 | 76.00 |
| 2 | 2.624999 | 45.71 | 2624.90 | 2263.89 | 62.86 | 106.52 | 26.29 | 37.26 | 650.12 | 77.00 |
| 3 | 1.811162 | 66.26 | 1811.08 | 2264.02 | 62.86 | 141.19 | 33.81 | 38.42 | 730.16 | 77.00 |
| 4 | 1.805891 | 66.45 | 1805.81 | 2261.06 | 62.86 | 139.00 | 34.94 | 39.15 | 711.29 | 78.00 |
| 5 | 1.899558 | 63.17 | 1899.48 | 2261.06 | 62.86 | 138.42 | 31.97 | 38.63 | 719.53 | 78.00 |
| 6 | 1.791945 | 66.97 | 1791.88 | 2261.40 | 62.86 | 140.17 | 35.60 | 38.75 | 724.50 | 77.00 |
| 7 | 1.812872 | 66.19 | 1812.79 | 2261.40 | 62.86 | 141.21 | 33.17 | 38.82 | 741.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.86 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.8566589250175032, 'inference': 17.114737199983665, 'postprocess': 1.8899125666469747}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.96% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.999622 | 60.01 | 94.80 | 2257.31 | 62.86 | 132.30 | 31.88 | 37.83 | 674.55 | 76.00 |
| 2 | 1.911826 | 62.77 | 57.19 | 2263.88 | 62.86 | 137.83 | 35.56 | 38.69 | 681.09 | 77.00 |
| 3 | 1.775697 | 67.58 | 55.62 | 2264.02 | 62.86 | 140.21 | 35.10 | 38.97 | 749.50 | 77.00 |
| 4 | 2.857932 | 41.99 | 96.23 | 2261.06 | 62.86 | 104.19 | 23.45 | 37.66 | 660.00 | 78.00 |
| 5 | 1.810800 | 66.27 | 58.34 | 2261.06 | 62.86 | 139.63 | 34.43 | 38.96 | 750.00 | 78.00 |
| 6 | 1.910947 | 62.80 | 56.23 | 2261.40 | 62.86 | 138.76 | 33.38 | 38.69 | 701.25 | 77.00 |
| 7 | 1.784795 | 67.23 | 55.99 | 2261.40 | 62.86 | 141.40 | 32.80 | 39.09 | 773.00 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 13.78/16.27/32.30.
Separate latency-pass duration: 1.77 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.6671916916720875, 'inference': 12.667043266689385, 'postprocess': 1.420876775000579}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

4.24% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.012892 | 59.62 | 2012.80 | 2257.57 | 62.86 | 135.92 | 33.32 | 37.90 | 723.97 | 76.00 |
| 2 | 2.060951 | 58.23 | 2060.86 | 2263.57 | 62.86 | 142.04 | 34.15 | 38.10 | 668.82 | 76.94 |
| 3 | 1.887886 | 63.56 | 1887.75 | 2264.02 | 62.86 | 138.62 | 34.00 | 38.82 | 756.09 | 77.00 |
| 4 | 1.980625 | 60.59 | 1980.46 | 2261.02 | 62.86 | 144.51 | 35.06 | 38.54 | 687.73 | 78.00 |
| 5 | 1.922109 | 62.43 | 1922.02 | 2261.06 | 62.86 | 139.30 | 34.69 | 38.82 | 709.69 | 78.00 |
| 6 | 2.759793 | 43.48 | 2759.71 | 2261.37 | 62.86 | 109.34 | 26.77 | 37.34 | 633.49 | 77.00 |
| 7 | 2.288399 | 52.44 | 2288.26 | 2261.40 | 62.86 | 120.63 | 30.14 | 37.96 | 744.58 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 2.86 s; not used for throughput.

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

12.34% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.102890 | 57.06 | 2102.81 | 2258.08 | 62.80 | 138.47 | 30.00 | 38.11 | 710.00 | 76.00 |
| 2 | 2.077781 | 57.75 | 2077.71 | 1954.89 | 62.80 | 138.03 | 33.26 | 37.88 | 634.71 | 76.46 |
| 3 | 1.999629 | 60.01 | 1999.49 | 2264.07 | 62.80 | 139.41 | 32.03 | 38.30 | 727.06 | 77.29 |
| 4 | 2.028720 | 59.15 | 2028.65 | 1958.86 | 62.80 | 137.48 | 33.62 | 38.15 | 669.71 | 78.00 |
| 5 | 2.477518 | 48.44 | 2477.41 | 2261.06 | 62.80 | 120.97 | 27.93 | 38.01 | 673.88 | 78.00 |
| 6 | 2.121836 | 56.55 | 2121.71 | 1956.93 | 62.80 | 132.68 | 34.74 | 38.37 | 654.00 | 77.00 |
| 7 | 2.537159 | 47.30 | 2537.09 | 2261.40 | 62.80 | 117.73 | 28.75 | 37.43 | 614.62 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.98 s; not used for throughput.

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

11.26% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.158050 | 55.61 | 2157.95 | 1952.38 | 55.57 | 137.22 | 33.25 | 37.67 | 605.42 | 76.00 |
| 2 | 2.100763 | 57.12 | 2100.67 | 1956.12 | 55.57 | 137.23 | 34.49 | 38.01 | 620.57 | 76.03 |
| 3 | 2.534564 | 47.35 | 2534.43 | 1958.39 | 55.57 | 118.75 | 29.02 | 37.64 | 644.25 | 77.58 |
| 4 | 2.009301 | 59.72 | 2009.20 | 1958.93 | 55.57 | 137.76 | 34.53 | 38.47 | 640.59 | 78.00 |
| 5 | 2.554043 | 46.98 | 2553.95 | 1957.88 | 55.57 | 116.49 | 28.12 | 37.66 | 605.25 | 77.90 |
| 6 | 2.054087 | 58.42 | 2053.98 | 1957.00 | 55.57 | 137.83 | 34.21 | 38.09 | 658.68 | 77.00 |
| 7 | 1.985887 | 60.43 | 1985.80 | 1958.17 | 55.57 | 137.40 | 34.00 | 38.34 | 673.18 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.07.
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


### managed-stream

11.15% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.592234 | 46.29 | 93.08 | 1952.09 | 55.57 | 121.20 | 30.63 | 37.19 | 597.07 | 76.00 |
| 2 | 2.605055 | 46.06 | 137.10 | 1955.60 | 55.57 | 120.49 | 27.98 | 37.02 | 599.29 | 76.00 |
| 3 | 2.532502 | 47.38 | 135.57 | 1956.57 | 55.57 | 117.40 | 28.40 | 37.54 | 613.88 | 77.72 |
| 4 | 2.007228 | 59.78 | 93.62 | 1956.86 | 55.57 | 135.83 | 32.32 | 38.31 | 660.00 | 77.94 |
| 5 | 2.063600 | 58.15 | 91.61 | 1956.71 | 55.57 | 136.39 | 33.97 | 38.44 | 657.35 | 77.62 |
| 6 | 2.087339 | 57.49 | 90.70 | 1957.00 | 55.57 | 137.37 | 33.03 | 38.48 | 632.14 | 77.14 |
| 7 | 2.076586 | 57.79 | 104.64 | 1957.06 | 55.57 | 137.86 | 32.00 | 37.86 | 696.18 | 77.00 |

Ready result delivery-interval P50/P95/P99 ms: 16.08/19.71/28.27.
Separate latency-pass duration: 2.05 s; not used for throughput.

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

Status: **complete**. Load 0.37 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `None`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 118.37 ± 20.33 | 1013.68 | 979.63/979.63/979.63 | 2550.49 | 119.89/162.00 | 120/648 | PASS |
| raw-stream | 130.86 ± 21.32 | 86.51 | 887.35/887.35/887.35 | 2527.39 | 119.89/162.00 | 120/648 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 115.37 ± 8.21 | 1040.03 | 995.97/995.97/995.97 | 2529.98 | 98.52/162.00 | 120/648 | PASS |
| managed-stream | 109.13 ± 17.11 | 124.63 | 1003.15/1003.15/1003.15 | 2528.30 | 98.52/162.00 | 120/648 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.013780 | 118.37 | 1013.68 | 2550.49 | 119.89 | 173.15 | 30.25 | 42.70 | 860.62 | 77.00 |
| 2 | 0.887804 | 135.16 | 887.72 | 2503.95 | 119.89 | 165.80 | 31.27 | 45.78 | 1030.00 | 77.93 |
| 3 | 0.935261 | 128.31 | 935.18 | 2504.04 | 119.89 | 166.81 | 25.81 | 46.44 | 1050.00 | 78.00 |
| 4 | 1.071557 | 111.99 | 1071.47 | 2504.40 | 119.89 | 145.51 | 34.24 | 42.73 | 724.41 | 77.88 |
| 5 | 0.881855 | 136.08 | 881.77 | 2504.75 | 119.89 | 163.05 | 27.60 | 45.35 | 1026.00 | 77.93 |
| 6 | 1.047973 | 114.51 | 1047.84 | 2506.75 | 119.89 | 146.76 | 29.24 | 43.20 | 1005.00 | 78.00 |
| 7 | 1.564661 | 76.69 | 1564.55 | 2506.75 | 119.89 | 104.06 | 22.87 | 40.86 | 710.22 | 77.87 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 0.98 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.791471916677286, 'inference': 4.7489219333366846, 'postprocess': 1.3463523832986841}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-1.95% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.941999 | 127.39 | 87.93 | 2502.37 | 119.89 | 164.74 | 32.38 | 43.04 | 907.50 | 77.19 |
| 2 | 0.909247 | 131.98 | 84.43 | 2518.64 | 119.89 | 160.63 | 34.50 | 44.62 | 916.88 | 77.81 |
| 3 | 0.916980 | 130.86 | 86.51 | 2504.04 | 119.89 | 163.31 | 30.53 | 44.46 | 971.00 | 78.00 |
| 4 | 1.553981 | 77.22 | 149.65 | 2527.24 | 119.89 | 101.93 | 24.91 | 39.43 | 638.86 | 78.00 |
| 5 | 0.889175 | 134.96 | 80.68 | 2504.75 | 119.89 | 167.82 | 30.80 | 46.70 | 1080.00 | 78.00 |
| 6 | 0.905443 | 132.53 | 82.03 | 2527.39 | 119.89 | 167.17 | 31.80 | 45.61 | 919.00 | 78.00 |
| 7 | 1.134181 | 105.80 | 149.41 | 2506.75 | 119.89 | 142.70 | 31.39 | 42.78 | 701.67 | 77.61 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/28.34/36.99.
Separate latency-pass duration: 0.89 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.5480433833242084, 'inference': 4.565024149997043, 'postprocess': 1.2170432833348361}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

8.82% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.026530 | 116.90 | 1026.42 | 2499.00 | 98.52 | 158.35 | 26.59 | 44.05 | 975.00 | 77.35 |
| 2 | 1.122440 | 106.91 | 1122.30 | 2528.76 | 98.52 | 149.75 | 29.83 | 41.25 | 709.17 | 77.56 |
| 3 | 1.017717 | 117.91 | 1017.62 | 2504.05 | 98.52 | 159.19 | 30.41 | 42.06 | 879.71 | 78.00 |
| 4 | 1.264254 | 94.92 | 1264.14 | 2527.24 | 98.52 | 138.74 | 27.35 | 41.79 | 849.00 | 78.00 |
| 5 | 1.040137 | 115.37 | 1040.03 | 2504.76 | 98.52 | 159.05 | 27.59 | 45.36 | 953.82 | 78.00 |
| 6 | 1.061036 | 113.10 | 1060.94 | 2529.98 | 98.52 | 156.70 | 29.82 | 43.46 | 793.24 | 78.00 |
| 7 | 1.034370 | 116.01 | 1034.27 | 2506.76 | 98.52 | 159.31 | 27.47 | 43.10 | 915.00 | 77.82 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.04.
Separate latency-pass duration: 1.00 s; not used for throughput.

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

12.98% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.245517 | 96.35 | 121.17 | 2527.88 | 98.52 | 141.38 | 25.90 | 41.92 | 900.00 | 77.50 |
| 2 | 1.789623 | 67.05 | 202.00 | 2528.30 | 98.52 | 101.40 | 20.16 | 39.45 | 723.60 | 77.00 |
| 3 | 1.056677 | 113.56 | 117.46 | 2526.99 | 98.52 | 160.15 | 28.50 | 43.75 | 915.00 | 78.00 |
| 4 | 1.024875 | 117.09 | 133.97 | 2527.29 | 98.52 | 163.78 | 28.94 | 43.47 | 867.35 | 78.00 |
| 5 | 1.099644 | 109.13 | 124.63 | 2527.25 | 98.52 | 162.65 | 29.11 | 42.99 | 809.17 | 78.00 |
| 6 | 1.120803 | 107.07 | 127.88 | 2527.37 | 98.52 | 157.44 | 30.44 | 42.91 | 730.83 | 78.00 |
| 7 | 1.086237 | 110.47 | 122.26 | 2527.35 | 98.52 | 161.89 | 26.94 | 41.28 | 866.47 | 77.65 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/33.19/46.66.
Separate latency-pass duration: 1.00 s; not used for throughput.

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

Status: **complete**. Load 6.52 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 67.88 ± 6.98 | 1767.78 | 2682.71/2682.71/2682.71 | 2797.22 | 115.19/140.00 | 120/709 | PASS |
| raw-stream | 54.57 ± 10.24 | 59.62 | 1710.75/1710.75/1710.75 | 2789.97 | 115.19/140.00 | 120/709 | PASS |
| native-non-stream | 65.20 ± 4.91 | 1840.32 | 1782.27/1782.27/1782.27 | 2790.11 | 115.19/140.00 | 120/709 | PASS |
| native-stream | 61.63 ± 2.52 | 1946.95 | 1939.29/1939.29/1939.29 | 2790.11 | 115.13/140.00 | 120/709 | PASS |
| managed-non-stream | 60.29 ± 9.29 | 1990.35 | 1908.35/1908.35/1908.35 | 2483.15 | 107.97/140.00 | 120/709 | PASS |
| managed-stream | 61.42 ± 4.75 | 89.89 | 2087.47/2087.47/2087.47 | 2483.14 | 107.97/140.00 | 120/709 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.767864 | 67.88 | 1767.78 | 2784.84 | 115.19 | 147.86 | 36.50 | 65.14 | 1451.50 | 80.80 |
| 2 | 1.728567 | 69.42 | 1728.50 | 2787.77 | 115.19 | 140.75 | 37.86 | 64.48 | 1475.36 | 77.96 |
| 3 | 1.706617 | 70.31 | 1706.54 | 2787.86 | 115.19 | 141.65 | 36.39 | 65.30 | 1476.96 | 77.93 |
| 4 | 1.793062 | 66.92 | 1792.99 | 2797.22 | 115.19 | 145.85 | 35.80 | 63.94 | 1504.00 | 75.73 |
| 5 | 2.057154 | 58.33 | 2057.06 | 2789.61 | 115.19 | 131.86 | 31.15 | 63.27 | 1517.65 | 75.74 |
| 6 | 1.710174 | 70.17 | 1710.09 | 2789.96 | 115.19 | 141.33 | 37.79 | 65.97 | 1477.24 | 77.21 |
| 7 | 2.294009 | 52.31 | 2293.90 | 2789.96 | 115.19 | 114.35 | 27.35 | 64.35 | 1545.81 | 77.76 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.05.
Separate latency-pass duration: 2.68 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.5480115666415866, 'inference': 11.32445032499542, 'postprocess': 1.3024435166585135}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

0.63% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.198964 | 54.57 | 57.07 | 2784.86 | 115.19 | 119.02 | 31.66 | 62.59 | 1493.57 | 81.11 |
| 2 | 1.728259 | 69.43 | 77.01 | 2787.71 | 115.19 | 139.37 | 36.69 | 65.71 | 1462.76 | 77.93 |
| 3 | 2.356975 | 50.91 | 59.62 | 2787.86 | 115.19 | 109.90 | 28.05 | 63.75 | 1533.16 | 77.21 |
| 4 | 1.700711 | 70.56 | 54.77 | 2788.71 | 115.19 | 141.39 | 37.72 | 65.05 | 1490.69 | 75.31 |
| 5 | 2.591665 | 46.30 | 90.57 | 2789.69 | 115.19 | 107.06 | 28.95 | 52.82 | 1285.61 | 74.66 |
| 6 | 1.721009 | 69.73 | 54.95 | 2789.89 | 115.19 | 140.29 | 36.03 | 64.02 | 1481.90 | 76.93 |
| 7 | 2.225530 | 53.92 | 115.83 | 2789.97 | 115.19 | 126.35 | 33.67 | 54.68 | 1274.17 | 77.11 |

Ready result delivery-interval P50/P95/P99 ms: 13.42/16.11/24.42.
Separate latency-pass duration: 1.71 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.934510783329794, 'inference': 13.724061616646091, 'postprocess': 1.5930904416563862}. These are upstream timers, not an additive decomposition of total wall time.


### native-non-stream

4.89% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.241075 | 53.55 | 2241.00 | 2784.93 | 115.19 | 116.44 | 34.92 | 56.22 | 1255.00 | 81.08 |
| 2 | 1.840435 | 65.20 | 1840.32 | 2787.41 | 115.19 | 146.25 | 37.00 | 64.01 | 1469.03 | 77.97 |
| 3 | 1.949118 | 61.57 | 1949.02 | 2787.86 | 115.19 | 129.94 | 37.30 | 58.43 | 1339.09 | 76.58 |
| 4 | 1.880784 | 63.80 | 1880.71 | 2788.64 | 115.19 | 144.69 | 35.03 | 62.48 | 1484.06 | 75.00 |
| 5 | 1.761036 | 68.14 | 1760.96 | 2790.11 | 115.19 | 138.80 | 38.87 | 65.19 | 1475.50 | 75.40 |
| 6 | 1.790378 | 67.02 | 1790.30 | 2789.64 | 115.19 | 147.23 | 36.57 | 66.13 | 1496.00 | 76.63 |
| 7 | 1.826272 | 65.71 | 1826.20 | 2789.41 | 115.19 | 137.55 | 37.39 | 65.16 | 1485.00 | 77.84 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.03.
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


### native-stream

10.13% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.947023 | 61.63 | 1946.95 | 2785.39 | 115.13 | 139.16 | 33.45 | 63.57 | 1460.00 | 81.18 |
| 2 | 2.106370 | 56.97 | 2106.30 | 2478.63 | 115.13 | 131.08 | 35.66 | 55.84 | 1274.57 | 77.63 |
| 3 | 2.075569 | 57.82 | 2075.45 | 2787.89 | 115.13 | 135.21 | 31.57 | 65.08 | 1506.00 | 76.71 |
| 4 | 1.962085 | 61.16 | 1962.02 | 2480.16 | 115.13 | 136.36 | 34.97 | 57.25 | 1406.36 | 74.91 |
| 5 | 1.877454 | 63.92 | 1877.35 | 2790.11 | 115.13 | 140.62 | 34.47 | 63.90 | 1507.50 | 75.91 |
| 6 | 1.923833 | 62.38 | 1923.75 | 2483.74 | 115.13 | 139.78 | 33.59 | 62.50 | 1491.09 | 75.94 |
| 7 | 1.932499 | 62.10 | 1932.42 | 2789.45 | 115.13 | 139.45 | 33.09 | 64.01 | 1498.12 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.03.
Separate latency-pass duration: 1.94 s; not used for throughput.

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

10.47% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.913354 | 62.72 | 1913.27 | 2477.48 | 107.97 | 141.70 | 34.09 | 63.04 | 1431.09 | 80.94 |
| 2 | 2.692789 | 44.56 | 2692.68 | 2478.38 | 107.97 | 107.71 | 27.30 | 55.00 | 1325.93 | 78.58 |
| 3 | 1.885320 | 63.65 | 1885.16 | 2479.57 | 107.97 | 141.34 | 36.32 | 62.13 | 1450.16 | 76.32 |
| 4 | 2.781129 | 43.15 | 2781.04 | 2480.16 | 107.97 | 103.14 | 26.11 | 50.41 | 1294.09 | 74.75 |
| 5 | 1.882965 | 63.73 | 1882.77 | 2482.98 | 107.97 | 138.33 | 33.97 | 63.06 | 1500.94 | 75.91 |
| 6 | 2.426985 | 49.44 | 2426.87 | 2483.15 | 107.97 | 118.08 | 33.36 | 49.48 | 1164.62 | 75.21 |
| 7 | 1.990499 | 60.29 | 1990.35 | 2480.88 | 107.97 | 138.22 | 35.79 | 57.23 | 1377.79 | 77.91 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.01/0.01.
Separate latency-pass duration: 1.91 s; not used for throughput.

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

8.97% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.879242 | 63.86 | 87.41 | 2477.76 | 107.97 | 140.36 | 34.62 | 64.52 | 1467.19 | 80.66 |
| 2 | 1.956403 | 61.34 | 87.73 | 2478.10 | 107.97 | 136.16 | 33.33 | 64.32 | 1471.36 | 79.94 |
| 3 | 1.926330 | 62.29 | 99.61 | 2479.86 | 107.97 | 138.18 | 32.88 | 65.16 | 1513.59 | 76.00 |
| 4 | 1.953820 | 61.42 | 89.89 | 2480.15 | 107.97 | 137.13 | 33.45 | 60.83 | 1457.73 | 75.94 |
| 5 | 1.892435 | 63.41 | 90.73 | 2482.97 | 107.97 | 141.25 | 34.91 | 63.40 | 1502.81 | 75.94 |
| 6 | 2.399873 | 50.00 | 87.87 | 2483.14 | 107.97 | 117.54 | 28.77 | 58.07 | 1433.46 | 75.67 |
| 7 | 1.962297 | 61.15 | 93.04 | 2480.88 | 107.97 | 137.98 | 33.71 | 63.52 | 1475.74 | 77.94 |

Ready result delivery-interval P50/P95/P99 ms: 15.20/27.41/36.38.
Separate latency-pass duration: 2.09 s; not used for throughput.

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

Status: **complete**. Load 6.24 s. Device `cpu`, dtype `torch.float32`.
Input scope: end-to-end 1280x720 H.264 controlled moving bus clip; includes source decoding. Input SHA256: `93a55fad4d49c9cade5dcd9837ff00f68810e12bfee88137062a08a77e69eb14`.
Runner SHA256: `95e7cfd5a934635940d474c561ab44d2091482b71a47ac5b525b3de929d9afc8`. Prompt: `['person', 'bus', 'car']`.

| Mode | Median FPS ± SD | Host TTFF ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Peak allocated/reserved VRAM MiB | Frames/boxes | Parity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw-non-stream | 97.06 ± 13.64 | 1236.22 | 1139.00/1139.00/1139.00 | 2645.49 | 209.49/292.00 | 120/709 | PASS |
| raw-stream | 105.72 ± 12.11 | 88.75 | 1125.13/1125.13/1125.13 | 2645.49 | 209.49/292.00 | 120/709 | PASS |
| native-non-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| native-stream | N/A | — | — | — | — | — | Native video batch is locked to 1 by the public API |
| managed-non-stream | 100.47 ± 4.34 | 1194.33 | 1196.30/1196.30/1196.30 | 2645.49 | 188.26/292.00 | 120/709 | PASS |
| managed-stream | 98.76 ± 1.82 | 121.37 | 1191.06/1191.06/1191.06 | 2541.38 | 188.26/292.00 | 120/709 | PASS |

### raw-non-stream

0.00% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.094275 | 109.66 | 1094.19 | 2451.17 | 209.49 | 160.71 | 56.59 | 58.62 | 1179.71 | 75.65 |
| 2 | 1.259734 | 95.26 | 1259.66 | 2455.97 | 209.49 | 137.73 | 45.35 | 62.68 | 1243.50 | 78.60 |
| 3 | 1.108261 | 108.28 | 1108.18 | 2456.10 | 209.49 | 158.63 | 51.89 | 61.16 | 1141.67 | 78.83 |
| 4 | 1.554421 | 77.20 | 1554.32 | 2459.50 | 209.49 | 103.30 | 35.86 | 60.20 | 1266.82 | 80.59 |
| 5 | 1.233906 | 97.25 | 1233.79 | 2459.53 | 209.49 | 148.19 | 47.45 | 60.05 | 1114.50 | 79.90 |
| 6 | 1.236335 | 97.06 | 1236.22 | 2642.86 | 209.49 | 136.72 | 48.26 | 60.18 | 1206.32 | 79.00 |
| 7 | 1.601319 | 74.94 | 1601.20 | 2645.49 | 209.49 | 103.26 | 34.24 | 62.98 | 1260.71 | 78.90 |

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

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.550563599994348, 'inference': 5.949069166672416, 'postprocess': 1.1474361916725684}. These are upstream timers, not an additive decomposition of total wall time.


### raw-stream

-8.19% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.092441 | 109.85 | 83.68 | 2454.66 | 209.49 | 155.15 | 56.39 | 66.30 | 1156.67 | 75.94 |
| 2 | 1.610389 | 74.52 | 156.88 | 2447.32 | 209.49 | 103.41 | 34.45 | 54.73 | 1322.73 | 77.91 |
| 3 | 1.129150 | 106.27 | 82.96 | 2456.10 | 209.49 | 148.62 | 52.28 | 64.03 | 1156.67 | 79.00 |
| 4 | 1.250425 | 95.97 | 88.75 | 2456.00 | 209.49 | 141.13 | 52.10 | 59.84 | 1203.00 | 80.65 |
| 5 | 1.123828 | 106.78 | 88.58 | 2459.51 | 209.49 | 154.43 | 58.61 | 62.49 | 1081.67 | 79.89 |
| 6 | 1.135076 | 105.72 | 92.06 | 2642.05 | 209.49 | 161.77 | 56.17 | 60.67 | 1134.17 | 79.06 |
| 7 | 1.199954 | 100.00 | 120.05 | 2645.49 | 209.49 | 145.90 | 50.00 | 56.52 | 1159.74 | 78.84 |

Ready result delivery-interval P50/P95/P99 ms: 0.02/37.60/46.21.
Separate latency-pass duration: 1.13 s; not used for throughput.

Parity for every measured repeat:

- Repeat 1: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 2: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 3: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 4: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 5: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 6: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`
- Repeat 7: `{'pass': True, 'max_box_error_px': 0.0, 'max_score_error': 0.0, 'frame_label_parity': True}`

Upstream-reported first-repeat stage averages (ms/frame): {'preprocess': 1.531511524982913, 'inference': 5.9454719499929825, 'postprocess': 1.1468378916788424}. These are upstream timers, not an additive decomposition of total wall time.


### managed-non-stream

-3.05% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.160257 | 103.43 | 1160.11 | 2454.68 | 188.26 | 154.98 | 54.16 | 61.39 | 1185.00 | 76.11 |
| 2 | 1.333010 | 90.02 | 1332.89 | 2416.98 | 188.26 | 134.81 | 46.65 | 68.59 | 1269.00 | 77.55 |
| 3 | 1.183768 | 101.37 | 1183.68 | 2456.11 | 188.26 | 154.08 | 51.05 | 61.67 | 1165.26 | 79.58 |
| 4 | 1.194438 | 100.47 | 1194.33 | 2416.08 | 188.26 | 151.31 | 51.40 | 61.74 | 1155.75 | 80.20 |
| 5 | 1.196235 | 100.31 | 1196.11 | 2462.79 | 188.26 | 154.17 | 53.00 | 61.61 | 1149.00 | 80.00 |
| 6 | 1.197927 | 100.17 | 1197.80 | 2363.00 | 188.26 | 155.89 | 53.58 | 61.72 | 1134.47 | 79.11 |
| 7 | 1.187874 | 101.02 | 1187.78 | 2645.49 | 188.26 | 154.14 | 54.70 | 58.87 | 1144.50 | 78.75 |

Ready result delivery-interval P50/P95/P99 ms: 0.01/0.02/0.03.
Separate latency-pass duration: 1.20 s; not used for throughput.

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

-2.15% median paired elapsed-time overhead vs raw non-stream. Negative means lower measured elapsed time.

| Repeat | Seconds | FPS | First host frame ms | RSS MiB | Allocated VRAM MiB | CPU %/one core | GPU util % | Power W | SM MHz | Temp C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.180103 | 101.69 | 119.59 | 2416.34 | 188.26 | 154.08 | 49.95 | 62.25 | 1175.25 | 76.85 |
| 2 | 1.251343 | 95.90 | 133.10 | 2416.70 | 188.26 | 151.97 | 45.84 | 65.39 | 1195.26 | 77.00 |
| 3 | 1.216923 | 98.61 | 121.69 | 2415.75 | 188.26 | 152.95 | 52.10 | 61.63 | 1145.25 | 79.80 |
| 4 | 1.215074 | 98.76 | 120.80 | 2416.08 | 188.26 | 153.42 | 50.10 | 62.11 | 1151.25 | 79.95 |
| 5 | 1.207394 | 99.39 | 121.37 | 2360.01 | 188.26 | 154.04 | 53.58 | 59.02 | 1134.47 | 79.68 |
| 6 | 1.193687 | 100.53 | 116.70 | 2360.39 | 188.26 | 154.79 | 50.95 | 59.37 | 1142.25 | 79.60 |
| 7 | 1.220166 | 98.35 | 131.28 | 2541.38 | 188.26 | 150.95 | 52.15 | 59.10 | 1163.25 | 78.50 |

Ready result delivery-interval P50/P95/P99 ms: 0.04/37.64/44.71.
Separate latency-pass duration: 1.19 s; not used for throughput.

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

GPU: Tesla T4; model yolov8n.pt; elapsed 601.07 s; frames 38760.
Pass: **True**. Exceptions: `[]`. Failed gates: `[]`.
Each loop verifies frame count and order, consumes without retaining frames, closes the stream and collects unreachable Python objects. The model remains loaded; allocator caches are not emptied between loops.

| Metric | First 60-second median | Last 60-second median | Change % |
| --- | ---: | ---: | ---: |
| rss_mib | 1604.45 | 1613.27 | 0.55 |
| vram_allocated_mib | 12.10 | 12.10 | 0.00 |
| fps | 75.33 | 76.17 | 1.11 |
| open_fds | 45.00 | 45.00 | 0.00 |

Whole-check throughput including between-loop cleanup: 64.48 FPS. Per-loop FPS above excludes between-loop cleanup.

## Reproduction and limits

Use the versioned runner and notebook with the recorded package versions. Every raw timing sample, diagnostic interval, telemetry sample and first-repeat normalized output is retained in the companion JSON.
No CPU result has been substituted for a failed GPU job. N/A is a capability limitation; ERROR is an attempted run that failed. Model-load and download costs are separate from prediction FPS.
Stage costs not instrumented independently (including per-framework transfer/NMS/restoration/locking) are not guessed. The upstream stage averages above are the available stage evidence.
