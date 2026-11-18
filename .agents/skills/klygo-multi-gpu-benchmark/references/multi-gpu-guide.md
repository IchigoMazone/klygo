# Release benchmark setup and interpretation

## Environment preflight

Read `pyproject.toml` for the dependency extras required by each selected model.
Use the same Klygo checkout in each interpreter. Capture exact installed versions
instead of assuming the historical L4 package set is appropriate for a new GPU.
Install a PyTorch build compatible with the device/driver and a compatible CuPy
package for CUDA utility workloads. Avoid installing multiple CuPy variants.
Set the Keras backend before importing Keras when running D-FINE.

Check the runtime before a GPU run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import cupy; print(cupy.__version__, cupy.cuda.runtime.getDeviceCount())"
```

Use a separate environment for Locate Anything when its Transformers requirement
conflicts with the main environment. The current extra specifies
`transformers==4.57.1`; inspect the current dependency definition before setup.
Install Klygo and the Locate Anything extra there, then verify:

```bash
/path/to/locate/bin/python -c "import cv2, klygo, torch, transformers; print(transformers.__version__, torch.cuda.is_available())"
```

Pass that executable to `--locate-python`; this affects Locate Anything subprocesses
only. Use `--locate-isolated` to force the managed environment instead. Known
dependency failures automatically fall back to a managed venv; the main and
supplied environments are never modified. Remote-code execution still requires the separate `--allow-remote-code` flag.
Do not assume the isolated interpreter inherits packages from the main interpreter.

## Select a physical GPU

The orchestrator's utility code detects visible CUDA capabilities, and several
utility paths use device 0. Isolate one physical GPU per fresh process rather than
assuming `--device cuda:1` retargets all utility sections.

```bash
CUDA_VISIBLE_DEVICES=1 python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py \
  --device cuda:0 --dataset synthetic --repeats 1 \
  --output artifacts/benchmarks/physical_gpu1_smoke
```

On PowerShell, set `$env:CUDA_VISIBLE_DEVICES="1"` before the command. The visible
physical GPU is addressed as logical `cuda:0`. Repeat with a different visible GPU
and a fresh output directory for each device. Keep comparisons sequential unless
resource isolation has been verified. The release launcher does not implement an
`all-gpus` option or a merged multi-GPU report.

Use the CPU synthetic smoke profile from `SKILL.md` when CUDA is unavailable.
For a CPU-only run on a GPU machine, hide CUDA devices with
`CUDA_VISIBLE_DEVICES=""`; `--device cpu` controls model subprocesses, not
utility auto-detection.
Start with batch 1 on low-memory GPUs. Use the lower-level engine for explicit
model-specific image size, iteration, batch, or video control; those flags are
not forwarded by the release orchestrator:

```bash
PYTHONPATH=. python benchmarks/gpu_suite/run.py \
  --model yolo --device cuda:0 --dtype float32 \
  --imgsz 320 --batch 1 --workers 1 --prefetch 1 \
  --iterations 2 --repeats 1 \
  --output artifacts/benchmarks/low_memory_yolo
```

Consult `benchmarks/gpu_suite/README.md` before using the video profile. Do not
promise constant memory use merely because a call uses streaming; inspect what
the benchmark retains. Treat allocation failures as failed measurements and
record any changes of precision, batch, image size, or workload used on reruns.

## Dataset and workload boundaries

`--dataset-dir` takes precedence over `--dataset`. Supply a prepared YOLO tree
with matching image/label stems and `data.yaml`; a flat `images/`, `labels/` tree
is supported. The runner also recognizes `images/train2017`, `labels/train2017`.
Flatten/repartition other layouts with dataset APIs before benchmarking.

- `--dataset synthetic`: no external dataset download; useful for validation.
- `--dataset coco128`: request COCO128 explicitly; download failures must surface.
- `--dataset auto`: may fall back to synthetic; inspect `dataset_source` before
  describing a result as a real-data run.
- `--dataset-dir /content/coco_val2017_yolo`: use local real data for loader,
  quality, duplicate, clustering, and indexed/strided/random access workloads.
- `--large-scale`: expand synthetic metrics/groups/mining and access workloads;
  it is not a request to run every model across the supplied real dataset.

`--include-cpu-utils` currently takes effect only on the synthetic dataset
branch. With real data the runner returns before optional media.info, save_images,
visual, and tabulate tests; cover these with a separate synthetic CPU run.

The loader benchmark's eager/stream/indexed rows can include lazy containers;
compare actual decoding rows separately. Record cache state and measurement order
when comparing random access. `detections.random_jump_index` times metadata/index
access, while `detections.random_jump_image_read` includes image reads.

## Check output completeness

Always retain:

| Output | Purpose |
| --- | --- |
| `environment.json` | Hardware capabilities, package versions, dataset source, Git state |
| `metrics_benchmark.json` | Precision/device/size workloads and numerical drift |
| `outputs_post_benchmark.json` | Group operations, postprocessing, combine, packing, transfers |
| `mining_benchmark.json` | Synthetic annotation-based mining workloads |
| `datasets_media_visual_benchmark.json` | Real/synthetic loaders, disk mining, access, optional CPU helpers |
| `summary.json`, `REPORT.md` | Aggregated results and readable report |

When inference is enabled, retain `model_runs.json`, `model_modes.json`,
`model_loads.json`, and `inference/<model>_<dtype>/`. When scaling is enabled,
retain `scaling_runs.json` and `scaling/<configuration>/`. Retain generated charts
and a complete ZIP when requested. Chart availability depends on plotting support;
check actual files rather than assuming all five charts exist.

Inspect subprocess failures, per-case statuses, parity flags, actual dtype,
missing models, and missing repetitions. Report unsupported upstream stream modes
as `N/A`, not failures or successful zero-FPS cases. Scaling failure rows can
contain zero FPS placeholders; exclude them from numeric rankings and show the
failure separately. Do not use a successful orchestrator exit as the only gate.

`parity` compares implementations on the benchmark fixture; it is not a model
accuracy score. For CUDA timing, retain the canonical runner's synchronization
and warmup logic. Avoid additional synchronization or verbose output in timed
regions. CPU versus CUDA speed depends on work granularity and transfer overhead;
do not assume CUDA wins for small frames or many small operations.

## Historical L4 baseline

Use `benchmarks/gpu_suite/colab_l4_val2017_2026_11_15/` as historical evidence,
not a template to copy result values. Its real-data section records 5,000 COCO
images; inference has an independent fixture/iteration protocol. Its environment
records a dirty checkout, and both Locate Anything precisions failed due to
missing `cv2`. Do not describe that run as nine fully passing models.
Re-run failed cases after fixing the environment in a new directory and retain
both the original failure and the rerun provenance.

## Current synthetic mining protocol

The annotation-only mining fixture has no source images. Use explicit scoring
weights `rarity=0.5`, `coverage=0.3`, `boxes=0.2`; its weighted-selection measurement
includes recomputing these scores. Record `score_weights` and
`select_includes_scoring` in each mining row. Keep image-quality scoring in the
separate disk-backed section. This makes the current fixture runnable without
source pixels; do not compare its scoring/selection timings directly with old
rows whose scoring protocol is unrecorded.


## Locate repair and mandatory aggregation gate

For the full release recipe use `--require-locate-results --locate-isolated` and
supply `--allow-remote-code` when authorized. The managed environment lives outside
the report directory and reuses system site packages; local pinned overrides
provide the required Transformers version. `--locate-auto-fix` is enabled by
default, with two known-dependency retries per precision. It checks OpenCV and
other imports, installs the current repository Locate extra, and can install
headless OpenCV for missing `cv2`/libGL errors. It preserves all attempts and does
not automatically alter the benchmark precision, input size or work count.

Unknown errors remain explicit failures for the executing agent to investigate.
After correcting the cause, `--locate-only --output <existing-run>` restores the
original precision/device/repeat contract and merges Locate measurements with
existing utility, model, and scaling records. It requires `run_config.json` and an
unchanged inference engine hash. Use the same hardware/software protocol; old
summary-only published runs need a fresh, provenance-complete run.

`completion.json` is the machine-readable gate. If any requested Locate precision
has missing/partial output, failed parity, incorrect actual dtype/device, missing
mode records or repeat measurements, retain the partial summary and exit with an
error before generating final charts/report/ZIP. The two unsupported upstream
stream modes remain N/A. Archive previous final charts/reports during repair;
never display a stale final artifact as the current completed result.
