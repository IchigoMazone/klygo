# Optimized NVIDIA L4 benchmark evidence — 2026-11-11

Executed on Google Colab Pro (NVIDIA L4, `GPU-a2494f9a-318e-2a48-df44-9f892f39c3f0`, 22,563.12 MiB VRAM, CUDA 13.0, PyTorch `2.14.1+cu130`, Ultralytics `8.4.208`, Transformers `4.57.1`, Keras `3.13.2`, KerasHub `0.26.0`, OpenCV `5.0.0.93`) at commit [`7729cb719cc493fc45fc2c885c848b909d770ecd`](https://github.com/IchigoMazone/klygo/commit/7729cb719cc493fc45fc2c885c848b909d770ecd) (`benchmark_models_klygo`), directly comparable on the same physical GPU against the baseline [`8b872985e9f7a9910e0736b79bed6c0effe0b518`](../colab_l4_complete_2026_11_11/README.md). Executed Colab notebook: [Google Colab](https://colab.research.google.com/drive/1B0C4NMDUY9eoXnTPzaJXokI3lX8E36vj).

33 configurations, 172 supported cases, 1,204 individual measured passes (seven repeats each); **100% parity PASS** across every supported configuration.

| Dataset | Configurations | Supported cases | Measured passes | All parity pass | Report |
|---|---:|---:|---:|---:|---|
| Controlled images/video | 25 | 124 | 868 | `true` | [Full report](controlled/REPORT.md) |
| People video | 4 | 24 | 168 | `true` | [Full report](people/REPORT.md) |
| Cars video | 4 | 24 | 168 | `true` | [Full report](cars/REPORT.md) |

Repeat-level statistics (mean, sample SD, CV, exploratory bootstrap intervals, and paired elapsed overhead): [controlled](controlled/STATISTICS.md), [people](people/STATISTICS.md), [cars](cars/STATISTICS.md). Each directory also contains machine-readable `raw_statistics.json`.

## Original evidence

- [controlled_raw_evidence.zip](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_optimized_2026_11_11__controlled_raw_evidence.zip) (SHA-256 `b2cdb9548422c57b5189e92c7192d18d650f944180b8d5feb26b4512a75ceaed`): full `bundle.json`, individual timing/latency/telemetry samples, 25 job logs, environment freeze, `nvidia_smi.csv`, `driver.log`, `jobs.json`, `completion.json`, 600s `stability.json`/`stability.log`, and verified SHA-256 `manifest.json` (34 entries).
- [extension_evidence.zip](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_optimized_2026_11_11__extension_evidence.zip) (SHA-256 `5447010a0210d3e8a309d3c03b6aefc44036a2c9a48dcf84c1d3619946d2edd6`): `real_people` and `real_cars` `bundle.json`, `source.json` provenance/hashes, `REPORT.md`, per-model logs, `real_jobs.json`, and verified SHA-256 `manifest.json` (15 entries).
- [validation.json](validation.json): verified dataset counts, parity flags, and archive SHA-256 digests.

## Stability probe (600 s managed stream)

- Model: `yolov8n.pt` on `NVIDIA L4`, `video_b1/bus_motion_h264.mp4` (120 frames/loop)
- Elapsed: `600.35 s`, total frames: `49,800` (`415` loops), `pass: true`, `failed_gates: []`
- Throughput stability: `92.48` FPS (first 60s window) $\rightarrow$ `91.11` FPS (last 60s window)
- Memory & FD stability: RSS `1705.32` $\rightarrow$ `1707.35 MiB` (`+0.12%`), allocated VRAM growth `0.00%`, open FDs `44` $\rightarrow$ `44` (`0.00%`)

## Before (`8b87298`) vs. After (`7729cb7`) on the same NVIDIA L4

Commit `7729cb7` replaced recursive `copy.deepcopy` in hot prediction/output paths with structural cloning (`clone_value` / `_clone_meta_value`), removed per-`Detection` `__dict__` allocation and redundant registry copies, switched PIL-to-BGR conversion to `cv2.cvtColor`, eliminated redundant RGB PIL copies in Ultralytics managed inputs and native in-memory fallbacks, preserved finite stream length (`_SizedStream`) to avoid single-batch thread-pool startup, and enabled true lazy streaming in `native-stream`.

### 1. Fast image throughput & managed overhead (`images`, 640x480)

| Workload | Model | Mode | Before FPS (`8b87298`) | After FPS (`7729cb7`) | Speedup | Overhead vs. Raw (`Before -> After`) |
|---|---|---|---:|---:|---:|---:|
| `images b=1` | `yolo` | `native-non-stream` | 79.32 | 89.99 | +13.5% | +27.97% $\rightarrow$ **+10.70%** |
| `images b=1` | `yolo` | `managed-non-stream` | 73.02 | 95.33 | +30.6% | +37.47% $\rightarrow$ **+4.07%** |
| `images b=1` | `yolo` | `managed-stream` | 70.34 | 95.82 | **+36.2%** | +41.67% $\rightarrow$ **+4.31%** |
| `images b=1` | `yolo11` | `managed-stream` | 61.31 | 84.19 | **+37.3%** | +40.82% $\rightarrow$ **+3.88%** |
| `images b=1` | `yolo26` | `managed-stream` | 58.45 | 76.51 | **+30.9%** | +33.76% $\rightarrow$ **+3.14%** |
| `images b=1` | `yolo-world` | `managed-stream` | 58.08 | 77.60 | **+33.6%** | +32.32% $\rightarrow$ **+1.36%** |
| `images b=1` | `faster-rcnn` | `managed-stream` | 38.77 | 41.22 | +6.3% | +13.13% $\rightarrow$ **+8.79%** |
| `images b=4` | `yolo` | `native-non-stream` | 171.56 | 196.82 | +14.7% | +42.84% $\rightarrow$ **+8.99%** |
| `images b=4` | `yolo` | `managed-non-stream` | 140.21 | 217.52 | **+55.1%** | +75.13% $\rightarrow$ **+2.53%** |
| `images b=4` | `yolo` | `managed-stream` | 131.37 | 218.60 | **+66.4%** | +87.54% $\rightarrow$ **+2.82%** |
| `images b=4` | `yolo11` | `managed-stream` | 124.19 | 195.45 | **+57.4%** | +67.98% $\rightarrow$ **+2.36%** |
| `images b=4` | `yolo26` | `managed-stream` | 118.32 | 185.51 | **+56.8%** | +64.99% $\rightarrow$ **+3.89%** |
| `images b=4` | `yolo-world` | `managed-stream` | 118.31 | 199.62 | **+68.7%** | +61.39% $\rightarrow$ **-3.78%** |
| `images b=4` | `faster-rcnn` | `managed-stream` | 65.99 | 68.74 | +4.2% | +9.28% $\rightarrow$ **+4.67%** |

### 2. True lazy `native-stream` Time-to-First-Frame (`video b=1`)

| Dataset | Model | `native-stream` TTFF before (ms) | `native-stream` TTFF after (ms) | Reduction |
|---|---|---:|---:|---:|
| `controlled` | `yolo` / `yolo11` / `yolo26` / `yolo-world` | 1,212.7 / 1,448.5 / 1,577.7 / 1,659.2 | **50.4 / 51.5 / 53.0 / 51.9** | **24.1x – 32.0x** |
| `people` | `yolo` / `yolo11` / `yolo26` / `yolo-world` | 1,130.6 / 1,376.5 / 1,508.7 / 1,541.8 | **24.7 / 26.5 / 27.9 / 27.7** | **45.7x – 55.7x** |
| `cars` | `yolo` / `yolo11` / `yolo26` / `yolo-world` | 1,098.4 / 1,342.8 / 1,522.0 / 1,530.7 | **23.6 / 25.4 / 27.2 / 26.9** | **46.6x – 56.9x** |

### 3. Remaining improvement directions

- **KerasHub `d-fine` fixed per-call overhead (`~375 ms/call`)**: `d-fine` scales `3.88x` from `b=1` (`2.66 FPS`) to `b=4` (`10.21 FPS`) in almost identical wall time; compiling or streamlining Keras 3 `preprocessor` / `decode_predictions` dispatch targets single-frame latency.
- **FP16 / BF16 autocast profiles for Transformer detectors**: `grounding-dino` (`4.89 -> 5.05 FPS`) and `grounding-dino-base` (`3.73 -> 3.70 FPS`) remain compute-bound in FP32 with near-zero `b=4` scaling.
- **Native `batch > 1` video & `preprocessed` trace benchmark (`C6`)**: Currently native video is locked to `batch=1` by API contract; adding a dedicated `C6` trace-restoration and `FP16` comparison profile will further broaden coverage.
