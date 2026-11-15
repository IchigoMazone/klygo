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
copy benchmark logic into this skill or use the legacy
`examples/integration/multi_backend_benchmark.py` as the release entry point.
The launcher forwards the canonical CLI unchanged and resolves relative paths
from the repository root, including output and dataset paths.

```bash
python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py --help
```

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
  --models yolo,yolo11,yolo26,yolo-world,grounding-dino,grounding-dino-base,faster-rcnn,d-fine \
  --dtypes float32,float16 \
  --include-scaling --scaling-models yolo,yolo11,yolo26,yolo-world \
  --output artifacts/benchmarks/release_gpu0_val2017 \
  --zip artifacts/benchmarks/release_gpu0_val2017.zip
```

Add `locate-anything` to `--models` only when its environment is ready. Pass
`--locate-python /absolute/path/to/venv/bin/python` for its isolated interpreter.
Use `--allow-remote-code` only when execution of that model's remote code is
within the user's authorized task. Verify the isolated environment imports
`cv2`, Klygo, PyTorch, and the required model dependencies first. Missing `cv2`
caused both Locate Anything cases to fail in the historical val2017 run.

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
