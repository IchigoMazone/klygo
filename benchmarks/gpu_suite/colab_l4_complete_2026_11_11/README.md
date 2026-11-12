# Complete NVIDIA L4 benchmark evidence — 2026-11-11

Executed on Google Colab Pro (NVIDIA L4, 22,563.12 MiB VRAM, CUDA 13.0, PyTorch `2.14.1+cu130`, Ultralytics `8.4.208`, Transformers `4.57.1`, Keras `3.13.2`, KerasHub `0.26.0`, OpenCV `5.0.0.93`) at pinned commit [`8b872985e9f7a9910e0736b79bed6c0effe0b518`](https://github.com/IchigoMazone/klygo/commit/8b872985e9f7a9910e0736b79bed6c0effe0b518) (`perf/managed-input-pipeline`). Executed Colab notebook: [`Klygo_L4_Benchmarks_2026_11_11.ipynb`](https://colab.research.google.com/drive/1B0C4NMDUY9eoXnTPzaJXokI3lX8E36vj?usp=sharing).

33 configurations, 172 supported cases, 1,204 individual measured passes (seven repeats each); **100% parity PASS** across every supported configuration. Raw non-stream is retained for every configuration. Unsupported cases remain N/A.

| Dataset | Configurations | Supported cases | Measured passes | All parity pass | Report |
|---|---:|---:|---:|---:|---|
| Controlled images/video | 25 | 124 | 868 | `true` | [Full report](controlled/REPORT.md) |
| People video | 4 | 24 | 168 | `true` | [Full report](people/REPORT.md) |
| Cars video | 4 | 24 | 168 | `true` | [Full report](cars/REPORT.md) |

Repeat-level statistics (mean, sample SD, CV, exploratory bootstrap intervals, and paired elapsed overhead): [controlled](controlled/STATISTICS.md), [people](people/STATISTICS.md), [cars](cars/STATISTICS.md). Each directory also contains machine-readable `raw_statistics.json`.

## Original evidence

- [controlled_raw_evidence.zip](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_complete_2026_11_11__controlled_raw_evidence.zip) (SHA-256 `8230f7780fd47e80552d67ac589e1bacd4a4a149a2b5e5775506e792f47fa350`): full `bundle.json`, individual timing/latency/telemetry samples, 25 job logs, environment freeze, `nvidia_smi.csv`, `driver.log`, `jobs.json`, `completion.json`, 600s `stability.json`/`stability.log`, and verified SHA-256 `manifest.json` (34 entries).
- [extension_evidence.zip](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_complete_2026_11_11__extension_evidence.zip) (SHA-256 `89b7d60951e4b92074a9ef9e2593c0cf213ac345ce98849e05557dc9f93a35c8`): `real_people` and `real_cars` `bundle.json`, `source.json` provenance/hashes, `REPORT.md`, per-model logs, `real_jobs.json`, and verified SHA-256 `manifest.json` (15 entries).
- [validation.json](validation.json): verified dataset counts, parity flags, and archive SHA-256 digests.

## Stability probe (600 s managed stream)

- Model: `yolov8n.pt` on `NVIDIA L4`, `video_b1/bus_motion_h264.mp4` (120 frames/loop)
- Elapsed: `600.10 s`, total frames: `49,680` (`414` loops), `pass: true`, `failed_gates: []`
- Throughput stability: `92.11` FPS (first 60s window) $\rightarrow$ `91.94` FPS (last 60s window), `-0.19%` drift
- Memory & FD stability: RSS `1700.84` $\rightarrow$ `1702.32 MiB` (`+0.09%`), allocated VRAM `12.10` $\rightarrow$ `12.10 MiB` (`0.00%`), open FDs `44` $\rightarrow$ `44` (`0.00%`)

## Key findings & L4 vs. T4 comparison

1. **Compute scaling on NVIDIA L4 (`8b87298`) vs. Tesla T4 (`cffb476`/`f83352d`)**:
   - Batched and heavy architectures scale strongly on L4: `yolo` `images b=4` raw `248.41` vs `164.93 FPS` (`1.51x`), `video b=4` raw `226.75` vs `148.73 FPS` (`1.52x`), `faster-rcnn b=4` raw `71.81` vs `37.59 FPS` (`1.91x`), `grounding-dino-base b=1` `3.72` vs `2.01 FPS` (`1.85x`), `grounding-dino b=1` `4.89` vs `2.90 FPS` (`1.69x`), and `locate-anything` `2.44` vs `1.54 FPS` (`1.59x`).
   - Managed video consistency improved after `237efc7`: on L4, all four YOLO variants maintain `5.4%–13.4%` (`b=1`) and `8.2%–18.6%` (`b=4`) paired overhead in controlled video, and `4.5%–9.4%` on real `people`/`cars` video, eliminating the anomalous `yolo11 video b=1` managed-non-stream spike (`+44.67%` on T4 pre-`237efc7`, now `+8.76%` on L4).
2. **Where Klygo overhead concentrates on fast models**:
   - At `images b=4` (`yolo`), throughput drops from `248.41 FPS` (raw) $\rightarrow$ `171.56 FPS` (`native-non-stream`, `+42.84%` paired elapsed) $\rightarrow$ `140.21 FPS` (`managed-non-stream`, `+75.13%`) $\rightarrow$ `131.37 FPS` (`managed-stream`, `+87.54%`).
   - In contrast, heavy models (`grounding-dino`, `grounding-dino-base`, `d-fine`, `locate-anything`) incur only `0.05%–2.55%` overhead across all Klygo modes.

## Recommended improvement roadmap

1. **Eliminate redundant `Detection._clone()` and per-frame `deepcopy(config)` in `klygo/models/detection/base.py`**:
   - `_infer_batch` calls `YOLODetect.forward()`, which constructs an owned `Detections` via `outputs.build_detections(...)`, causing `_pack_detection` (`det._owner is not None`) to clone every `Detection` immediately afterward, followed by `frame.metadata = deepcopy(config)` per frame and repeated `deepcopy` in `_remember_prediction`/`_config_snapshot`. Removing intermediate ownership/cloning and sharing immutable snapshot metadata directly targets the `+27%–43%` `raw -> native` gap on `images`.
2. **Avoid redundant PIL $\rightarrow$ contiguous BGR copies in `_materialize_model_image` & per-batch lock/context re-entry in `managed-stream`**:
   - Pre-converted `PIL.Image` batches incur `np.ascontiguousarray(np.asarray(image)[:, :, ::-1])` allocations on the host plus per-batch `_infer_batch` lock/context setup in `generate()`, driving the `+22%–32%` `native -> managed` delta at `images b=4`.
3. **Stream `native-stream` lazily instead of buffering with `list(frames())`**:
   - `native-stream` (`model_stream=True`) currently materializes all frames in `_predict_native` (`list(frames())`), producing `1,130–1,213 ms` Host TTFF on 120-frame video vs `22–46 ms` (`raw-stream`) and `30–84 ms` (`managed-stream`). Yielding lazily via `Detections(frames(), ..., stream=True)` will align native-stream TTFF with real streaming.
4. **Unlock `batch > 1` for Ultralytics `native` video mode**:
   - `video b=4` native rows are currently `N/A` because `_predict_native` locks `batch == 1`. Allowing backend-supported `batch` in native video unlocks `2x` video throughput for native callers.
5. **Reduce per-call KerasHub `d-fine` preprocessing/postprocessing overhead**:
   - On L4, `d-fine` takes `374.5 ms/pack` at `b=1` (`2.67 FPS`) and `385.8 ms/pack` at `b=4` (`10.37 FPS`, `3.88x` scaling), showing ~370 ms of fixed per-call Keras 3 `preprocessor` / `decode_predictions` host/graph dispatch overhead rather than GPU compute saturation.
6. **Add optional `FP16`/`BF16` autocast & `torch.compile` profiles for Transformer detectors**:
   - `grounding-dino` (`4.89 -> 5.06 FPS`) and `grounding-dino-base` (`3.72 -> 3.71 FPS`) show near-zero `b=4` scaling under FP32 compute saturation. Supporting opt-in `amp`/`fp16` precision profiles is the highest-leverage speedup for open-vocabulary Transformers.
