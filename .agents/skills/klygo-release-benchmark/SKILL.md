---
name: klygo-release-benchmark
description: Run and verify the full Klygo detection-evaluation benchmark suite (`benchmarks/gpu_suite/evaluate_release.py`, `run.py`, `stability.py`, `export_bundle.py`) on NVIDIA L4, T4, or local GPUs. Use when benchmarking all 9 detector models (`yolo`, `yolo11`, `yolo26`, `yolo-world`, `grounding-dino`, `grounding-dino-base`, `faster-rcnn`, `d-fine`, `locate-anything`) across 6 execution modes, `float32`/`float16`, scaling (`batch`/`workers`/`prefetch`/`packed`), GPU + loader component benchmarks (`metrics`, `outputs`, `postprocessing`, `mining`, `datasets.load`, `media.load`, `models.load`, `preprocessing`), large-scale stress (`--large-scale`), 600s stability, packaging a complete `.zip` evidence bundle (`--zip`), and uploading it as a GitHub Release asset.
---

# Klygo Release Benchmark & Evaluation Skill

This skill defines the exact measurement protocol, required metrics, execution recipes, and **mandatory `.zip` artifact packaging** for Klygo GPU/CPU release benchmarks using `benchmarks/gpu_suite/`.

> **Note:** For lightweight 5-model `examples/integration/multi_backend_benchmark.py` runs, see `klygo-multi-gpu-benchmark`. Use **this** skill (`klygo-release-benchmark`) for full `detection-evaluation` / Colab L4 / T4 / release benchmarking.

---

## 1. What Must Be Measured (Checklist)

Every full release benchmark run must capture and verify **all 7 dimensions**:

1. **Environment & Precision Pinning** (recorded automatically by `evaluate_release.py` -> `summary.json["environment"]`):
   - Exact `git branch`, `git commit` (`HEAD`), `dirty` status.
   - OS, Python version, CPU model & core count.
   - GPU name, total VRAM (MB), `is_nvidia_l4`, CUDA runtime, `torch_cuda`, `cupy_cuda`.
   - Exact package versions (`torch`, `torchvision`, `ultralytics`, `transformers`, `keras`, `keras-hub`, `numpy`, `cupy`, `pillow`, `opencv`).
   - TF32 / cuDNN / deterministic policy and **actual executing CUDA device & model dtype** (`actual_model_dtype`).

2. **9 Registered Detector Models** across **2 Precisions** (`float32`, `float16`):
   - `yolo` (YOLOv8n)
   - `yolo11` (YOLO11n)
   - `yolo26` (YOLO26n)
   - `yolo-world` (YOLO-World v2 small)
   - `grounding-dino` (Grounding DINO tiny)
   - `grounding-dino-base` (Grounding DINO base)
   - `faster-rcnn` (Faster R-CNN MobileNetV3 FPN)
   - `d-fine` (KerasHub D-FINE Nano)
   - `locate-anything` (LocateAnything-3B — requires `--allow-remote-code`; if `transformers` version conflicts with the main environment, run in an isolated venv with `transformers==4.57.1` and merge/download its JSONs alongside the main run).

3. **6 Execution Modes** (per model × dtype, `repeats=7` after 1 complete warmup):
   - `raw-non-stream`, `raw-stream` (upstream framework baseline; `N/A` with explicit reason if upstream lacks a native source-stream engine)
   - `native-non-stream`, `native-stream` (`native=True`)
   - `managed-non-stream`, `managed-stream` (`native=False`, full Klygo `media` + trace pipeline)
   - Required metrics per mode: `fps_median`, `ready_interval_p50_ms` (plus P95/P99 latency diagnostic), `all_parity_pass` (0.01 px box / 0.0001 score tolerance), VRAM allocated/reserved peaks, RSS/CPU, and optional NVML telemetry.

4. **GPU Compute & Loader Component Benchmarks** (built into `evaluate_release.py`, focused on GPU + `load` + `mining`; CPU-only visual/tabulate helpers are opt-in via `--include-cpu-utils`):
   - **Loaders (`load`)**:
     - `datasets.load` (real `coco128` by default via `--dataset coco128`, or custom/COCO-val2017 via `--dataset-dir`, plus `2,048`-image scaled dataset load & `Detections[i].image` random-jump image decode under `--large-scale`)
     - `media.load` (eager `stream=False`, streaming `stream=True`, plus `indexed=True`, `sample_rate=32` strided decode, and `64/2,048` `MediaFrames[i]` random-jump decode under `--large-scale`)
     - `models.load` (cold load latency `load_seconds` + `actual_model_dtype` summary in `model_loads.json`)
     - `preprocessing.compose` (letterbox + autocontrast trace transform)
   - **GPU Compute (CPU vs CUDA/CuPy)**:
     - `klygo.metrics` (`ConfusionMatrix`, `DetectionMetrics`, `evaluate`, `compose` across FP16/FP32/FP64 & packed/unpacked, plus `xlarge_5000_2000` & `batch_100f_10k` under `--large-scale`)
     - `klygo.outputs` & `klygo.postprocessing` (`group().keep`, `clip`/`threshold`/`nms`/`top`, `combine`, `pack`, `astype(16)`, `cuda(0)`/`cpu()` transfer, scaled to 100 frames × 120 pred under `--large-scale`)
     - `klygo.mining`: CPU vs CUDA `Detections` (`stats`, `score`, `select`, `balance`, `compare`) plus disk-backed `mining.stats(quality=True)`, `mining.find_duplicates`, `mining.cluster` on the real dataset (`coco128` or `--dataset-dir`).

5. **Batch / Workers / Prefetch / Packed Scaling** (`--include-scaling`):
   - Default scaling models: `yolo,yolo11,yolo26,yolo-world`
   - Tested matrix `(batch, workers, prefetch, packed)`: `(1,1,1,True)`, `(4,2,2,False)`, `(4,2,2,True)`, `(8,4,4,False)`, `(8,4,4,True)`

6. **600-Second Streaming Stability Probe** (when qualifying long-running memory stability):
   - Run `benchmarks/gpu_suite/stability.py` (non-retaining 600s stream) to verify bounded RSS/VRAM and zero handle/FD leaks.

7. **Complete Raw Evidence & ZIP Packaging (Mandatory)**:
   - Never download or keep only `summary.json`. Always preserve and zip the **entire** output directory (`summary.json`, `model_runs.json`, `model_modes.json`, `scaling_runs.json`, and all per-model raw JSONs under `models/` and `scaling/`).

---

## 1.5. Pre-Flight Condition (Before Running on Colab / Remote)

Before launching a Colab or remote GPU run, **always check local Git state**:
```bash
git status --short
git log -1 --oneline
```
- If `benchmarks/gpu_suite/`, `klygo/`, or `.agents/skills/klygo-release-benchmark/` has uncommitted or unpushed changes, **commit and push to the active remote branch first** so Colab clones the exact updated code and runner.

---

## 2. Standard Execution Commands

### A. Full 9-Model Release Evaluation (NVIDIA L4 / CUDA GPU)

```bash
PYTHONPATH=. python benchmarks/gpu_suite/evaluate_release.py \
  --output artifacts/benchmarks/colab_l4_full \
  --dataset coco128 \
  --repeats 7 \
  --large-scale \
  --include-models \
  --models yolo,yolo11,yolo26,yolo-world,grounding-dino,grounding-dino-base,faster-rcnn,d-fine,locate-anything \
  --dtypes float32,float16 \
  --allow-remote-code \
  --locate-python /content/venv_locate/bin/python \
  --include-scaling \
  --scaling-models yolo,yolo11,yolo26,yolo-world \
  --zip artifacts/benchmarks/colab_l4_full_results.zip
```

If `locate-anything` needs a pinned `transformers==4.57.1` environment, ensure `opencv-python-headless` and `-e .` are also installed in that venv and pass `--locate-python /content/venv_locate/bin/python` so `evaluate_release.py` automatically runs `locate-anything` in that venv and merges all **9/9 models** into `summary.json`, `REPORT.md`, and `charts/*.png` in one pass.

### B. 600s Stability Probe (Optional / Full Qualification)

```bash
PYTHONPATH=. python benchmarks/gpu_suite/stability.py \
  --model yolo \
  --duration 600 \
  --output artifacts/benchmarks/colab_l4_full/stability.json
```

### C. Package & Validate `.zip` Bundle (Mandatory at End of Run)

Always package the complete output folder into a `.zip` using Klygo's `archive.compress` (or `export_bundle.py` for validated raw six-mode reports) so all raw per-model JSONs, scaling JSONs, and `summary.json` stay together:

```python
from klygo import archive

archive.compress(
    "artifacts/benchmarks/colab_l4_full",
    "artifacts/benchmarks/colab_l4_full_results.zip",
    include_root=True,
    overwrite=True,
)
```

For validated strict raw six-mode bundles with SHA-256 `manifest.json` and `REPORT.md`:

```bash
PYTHONPATH=. python benchmarks/gpu_suite/export_bundle.py \
  --report artifacts/benchmarks/colab_l4_full/models/yolo_float32/yolo.json \
  --expected-repeats 7 \
  --output artifacts/benchmarks/colab_l4_validated.zip
```
*(Repeat `--report` for each selected configuration, and add `--stability ...` if `stability.json` was generated.)*

### D. Downloading from Google Colab

When running via Colab, **always** download the `.zip` archive (not just `summary.json`):

```python
from google.colab import files as colab_files
colab_files.download("/content/klygo/artifacts/benchmarks/colab_l4_full_results.zip")
```

### E. Upload `.zip` Bundle as a GitHub Release Asset

**Never** commit `.zip` or raw benchmark artifacts to Git history. Instead, attach the `.zip` (and optionally `summary.json`) to a GitHub Release using `gh`:

1. **Verify authentication & exact commit SHA**:
   ```bash
   gh auth status
   SHORT_SHA=$(git rev-parse --short HEAD)
   TAG="benchmark-l4-$(date +%Y%m%d)-${SHORT_SHA}"
   ```
   *(In headless/Colab environments without interactive login, set `GH_TOKEN` / `GITHUB_TOKEN` first.)*

2. **Create a new GitHub Release and attach the `.zip` bundle** (or upload to an existing release tag):
   ```bash
   # Create a release pinned to the exact tested commit and upload the .zip asset
   gh release create "$TAG" \
     artifacts/benchmarks/colab_l4_full_results.zip \
     --target "$(git rev-parse HEAD)" \
     --title "Benchmark L4 (${SHORT_SHA})" \
     --notes "Automated Klygo 9-model / 6-mode L4 release benchmark on commit ${SHORT_SHA}." \
     --prerelease
   ```
   To attach or overwrite the `.zip` on an **existing** release tag instead:
   ```bash
   gh release upload "$TAG" artifacts/benchmarks/colab_l4_full_results.zip --clobber
   ```

3. **Verify the uploaded release asset URL**:
   ```bash
   gh release view "$TAG" --json url,assets --jq '{url: .url, assets: [.assets[].name]}'
   ```

4. **Browser Fallback (when `gh` CLI / `GH_TOKEN` is unavailable locally)**:
   - Stage the `.zip` into `<Artifact Directory Path>/browser_sandbox/uploads/<filename>.zip`.
   - Use the `browser` subagent to open `https://github.com/<owner>/<repo>/releases/new`, create tag `$TAG` targeting the tested branch/commit, mark **Set as a pre-release**, attach `<filename>.zip`, and click **Publish release** (pausing only if GitHub login is not yet signed in).

### F. Auto-Push Benchmark Reports to GitHub (After Completion)

Once the benchmark finishes and the `.zip` is uploaded to GitHub Releases:
1. Store the lightweight Markdown reports (`REPORT.md`, `STATISTICS.md` if generated), generated PNG charts (`charts/*.png`), executed notebooks (`.ipynb`), and summary JSONs under `benchmarks/results/` (never commit `.zip` files or `artifacts/`):
   - `benchmarks/results/performance/<gpu>_<dataset>_<YYYY_MM_DD>/` — Full **GPU & core performance** suite (`evaluate_release.py`: 9-model 6-mode FPS, scaling, `metrics` CPU vs CUDA latency, `outputs_post`, `mining` speed, 600s `stability.json`)
   - `benchmarks/results/cpu_loaders/<dataset>_<YYYY_MM_DD>/` — **CPU & I/O loaders** (`datasets.load`, `media.load`, strided/random-jump `MediaFrames` & `Detections`, `preprocessing.compose`, `--include-cpu-utils`)
   - `benchmarks/results/video/<gpu>_<YYYY_MM_DD>/` — **Video streaming** (`--profile video`: H.264 decode, TTFF, P50/P95/P99)
   - `benchmarks/results/metrics/<dataset>_<YYYY_MM_DD>/` — **Detection accuracy (`klygo.metrics`)**: `mAP@50`, `mAP@50-95`, Precision, Recall, F1
   - `benchmarks/results/mining/<dataset>_<YYYY_MM_DD>/` — **Curation training accuracy (`curation_accuracy_benchmark.py`)**: mAP comparison across `klygo.mining` selection strategies
2. Commit and push `benchmarks/results/` (using `git add -f` for `charts/*.png` if `.gitignore` ignores `.png`) to the active branch:
   ```bash
   git add benchmarks/results/
   git add -f benchmarks/results/**/charts/*.png
   git commit -m "Update <GPU> benchmark reports (<YYYY-MM-DD>)"
   git push origin HEAD
   ```

---

## 3. Verification & Reporting Rules

- Report **exact** observed values (`fps_median`, `p50_ms`, `all_parity_pass`, `actual_model_dtype`, `git commit`, GPU name/VRAM). Never extrapolate or round away failures.
- Keep `artifacts/benchmarks/` out of Git unless explicitly asked to commit a recorded report.
- Follow Klygo dogfooding rules (`klygo.files`, `klygo.archive`, `klygo.utils.tabulate`) when writing helper cells or post-run scripts.
