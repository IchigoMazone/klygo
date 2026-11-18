---
name: klygo-multi-gpu-benchmark
description: Run and interpret the Klygo release benchmark suite for models, metrics, mining, outputs, postprocessing, datasets, and media. Use for release validation, GPU comparisons, six-mode inference, batch/worker/prefetch scaling, or reproducing the NVIDIA L4 COCO val2017 report. Use the canonical evaluate_release.py suite and preserve raw results, environment metadata, parity checks, charts, and failures.
---

# Klygo release benchmarking

## Use one implementation

Run from the Klygo repository root. Read `AGENTS.md`, then inspect the current CLI
before choosing a run. Treat these files as the source of truth:

- `benchmarks/gpu_suite/evaluate_release.py`: comprehensive release orchestrator.
- `benchmarks/gpu_suite/run.py`: per-model six-mode inference and scaling engine.
- `benchmarks/gpu_suite/README.md`: inference protocol and report field definitions.
- [references/multi-gpu-guide.md](references/multi-gpu-guide.md): setup, hardware
  selection, reproduction limits, and result validation.

Keep `scripts/run_benchmark.py` as a thin launcher for the orchestrator. Do not
copy benchmark logic into this skill or introduce a second release entry point.
The launcher forwards the canonical CLI unchanged and resolves relative paths
from the repository root, including output and dataset paths.

```bash
python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py --help
```

## API selection inside benchmarks

Apply root `AGENTS.md` Rule 0 and `docs/API_LEVELS.md`. Time the owning module's
public API for a public-API benchmark rather than reimplementing it from smaller
operations. Keep direct upstream/backend calls only in the explicitly named raw
reference cases where using a Klygo wrapper would invalidate the comparison.
That benchmark-specific exception is not an ordinary application usage example.

## Establish the run contract

1. Record the branch, commit, dirty state, Python executable, package versions,
   physical GPU selection, dataset provenance, image count, and run parameters.
   Preserve a patch for a dirty checkout alongside the run metadata.
2. Create a fresh, explicitly named output directory under `artifacts/benchmarks/`.
   Existing paths are overwritten by the runner; do not reuse an earlier result.
3. Start with CPU synthetic validation or one model before a full GPU run. Check
   the selected model's optional dependencies in `pyproject.toml`.
4. Use matching source, model weights, precision, image size, batch, repeats,
   worker/prefetch settings, and packed state for comparisons. Keep progress
   disabled in timed mining/metrics/combine calls (`verbose=False` by default).
5. Read `environment.json` after the run. CUDA utility workloads depend on CuPy
   availability; PyTorch CUDA alone does not enable them. Never label CPU results
   as GPU measurements or claim unsupported modes succeeded.

## Choose an execution profile

### CPU validation without downloads or models

```bash
CUDA_VISIBLE_DEVICES="" python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py \
  --device cpu --dataset synthetic --repeats 1 --include-cpu-utils \
  --output artifacts/benchmarks/cpu_smoke
```

This is a correctness/smoke profile, not a published throughput result.
Omit `--include-models` and `--include-scaling` to avoid model inference.
Hide CUDA devices as shown: `--device cpu` alone does not disable utility CUDA
auto-detection. On PowerShell set `$env:CUDA_VISIBLE_DEVICES=""` first.

### GPU preflight with one model

```bash
CUDA_VISIBLE_DEVICES=0 python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py \
  --device cuda:0 --dataset synthetic --repeats 1 \
  --include-models --models yolo --dtypes float32 \
  --output artifacts/benchmarks/gpu_preflight
```

### Real-dataset release evaluation

Use a prepared YOLO dataset containing `data.yaml`, images, and matching labels.
For COCO val2017, supply the prepared 5,000-image dataset; the runner does not
convert or download val2017 for this option.

```bash
CUDA_VISIBLE_DEVICES=0 python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py \
  --device cuda:0 --repeats 3 --large-scale \
  --dataset-dir /content/coco_val2017_yolo \
  --include-models \
  --models yolo,yolo11,yolo26,yolo-world,grounding-dino,grounding-dino-base,faster-rcnn,d-fine,locate-anything \
  --dtypes float32,float16 \
  --require-locate-results --locate-isolated --allow-remote-code \
  --include-scaling --scaling-models yolo,yolo11,yolo26,yolo-world \
  --output artifacts/benchmarks/release_gpu0_val2017 \
  --zip artifacts/benchmarks/release_gpu0_val2017.zip
```

For a full release, require Locate Anything before generating the combined
report and charts. `--require-locate-results` adds the model if omitted and
implies `--include-models`. It validates every requested precision: complete raw
report, matching dtype/device/repeats, four measured supported modes with positive
FPS and passing parity, and two explicit upstream-stream `N/A` modes.
A requested Locate model is gated even without the explicit require flag.
CPU and single-model smoke profiles intentionally do not require it.

Use `--locate-isolated` to force a managed venv. Alternatively supply
`--locate-python /absolute/path/to/venv/bin/python`. The preflight checks `cv2`,
PyTorch, torchvision, Klygo, Transformers and Locate dependencies before inference.
`--locate-auto-fix` defaults to enabled: on known dependency errors, create a
managed venv, install the repository's Locate extra with Transformers 4.57.1,
repair headless OpenCV if necessary, and retry up to `--locate-retries 2` per
precision. Never modify the main or supplied interpreter's installed packages.
The managed venv shares system site packages to reuse PyTorch, but dependency
overrides are installed inside the venv. Keep it outside the report/ZIP tree;
use `--locate-venv` to choose a location. Use `--no-locate-auto-fix` to prohibit
automatic dependency installation.

Use `--allow-remote-code` only when execution of the model's remote code is within
the user's authorized task. The runner refuses Locate inference without it.
Retain every stdout/stderr log, command, raw attempt, and interpreter preflight in
`locate_runtime/run_<id>/`. Missing `cv2` is repaired as a known dependency failure.
For an unknown model/driver/OOM error, inspect the preserved traceback and fix the
cause; do not silently change dtype, batch, workload, repeats or parity criteria.
If the cause cannot be fixed, leave the release incomplete and report the error.

### Repair only Locate and merge before chart generation

When a full run reaches Locate but fails, keep the completed utility and other
model results. Re-run only Locate in the same output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py \
  --locate-only --locate-isolated --allow-remote-code \
  --output artifacts/benchmarks/release_gpu0_val2017
```

This restores dtype/device/repeats from `run_config.json`, retains other models
and scaling rows, replaces failed Locate rows, and regenerates all combined charts
and the report only after every required Locate precision passes. Repeat on the
same physical GPU and compatible software environment. The inference runner hash
must match. This recovery needs the configuration/raw outputs retained by the new
runner; old published summary-only folders cannot establish that provenance.
Do not merge unrelated GPUs or silently fabricate a configuration for old results.

On failure, retain `summary.partial.json` and `completion.json` with status
`incomplete`; final `summary.json`, `REPORT.md`, charts, and the final ZIP are not
created. Previous finalized artifacts are archived before an in-place repair.
The original ZIP destination is remembered for successful recovery. A venv or ZIP
inside the source report directory is rejected to avoid bundling environments or
recursively packaging output.

## Respect the actual scope

- Expect nine model IDs, two requested precisions, and six named modes. Some
  upstream stream modes are `N/A`; preserve that distinction from failures.
- Expect scaling configurations `(batch, workers, prefetch, packed)` of
  `(1,1,1,True)`, `(4,2,2,False)`, `(4,2,2,True)`, `(8,4,4,False)`,
  `(8,4,4,True)`. Check the current source before assuming they are unchanged.
- Treat `--include-cpu-utils` as effective only on the synthetic dataset path in
  the current runner. The real-data branch returns before those optional helpers;
  run the synthetic CPU profile separately to cover them.
- Treat `--dataset-dir` as the source for loader, disk-mining, and access tests.
  Model inference uses its own fixtures and default iterations in `run.py`.
  Never describe the report as 5,000-image inference per model without evidence.
- Treat `--device` as the requested model device. Utility GPU workloads are
  auto-detected and use the visible GPU. For comparable per-GPU runs, expose one
  physical device with `CUDA_VISIBLE_DEVICES` and use logical `cuda:0`.
- Do not pass legacy `--device all-gpus`, `--device auto`, `--models all`, or
  `--modes` to this launcher. The release CLI does not implement those aliases.
  Do not claim MPS support from the old skill; this utility suite targets CPU/CUDA.

## Validate and deliver evidence

Check `summary.json`, `REPORT.md`, `environment.json`, per-module JSON files,
`model_runs.json`, `model_modes.json`, `model_loads.json`, `scaling_runs.json`,
per-model `inference/` and `scaling/` results, and `charts/` when generated.
Model/scaling files are conditional on the corresponding include flags.

Inspect every requested model/dtype, case status, parity result, actual dtype,
missing result, and scaling error. A zero orchestrator exit code or the word
`passed` on a subprocess row is insufficient evidence of release correctness.
Report failures explicitly; never replace missing measurements with zeros from
error rows or silently remove failed models from a comparison.

Separate output parity from detector accuracy; this suite does not establish
COCO mAP quality for every model. Distinguish throughput FPS, first-result
latency, and ready intervals: the report's `p50_ms` is a ready interval, not an
independent per-frame inference latency. Batch/worker/prefetch scaling changes
several parameters together and cannot isolate each parameter's contribution.
Distinguish lazy media setup from actual decoding and cache-warm access tests.

Preserve the complete run and optional ZIP before a Colab runtime expires.
For the historical baseline inspect
`benchmarks/gpu_suite/colab_l4_val2017_2026_11_15/REPORT.md` and adjacent JSONs.
Copy only deliberately selected reports/charts into a versioned benchmark
folder when authorized; keep weights, datasets, and generated runtime artifacts
out of source commits. Do not claim benchmark execution when only the skill,
CLI, or launcher has been validated.
