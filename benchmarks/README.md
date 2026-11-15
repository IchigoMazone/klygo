# Klygo benchmark suite

Use the release suite in [gpu_suite](gpu_suite/README.md) for current performance
and correctness evaluation. It covers metrics, mining, outputs/postprocessing,
dataset/media operations, optional model inference and scaling, and combined
reports and charts.

## Current entry points

| Entry point | Purpose |
| --- | --- |
| `gpu_suite/evaluate_release.py` | Comprehensive release orchestration and aggregation |
| `gpu_suite/run.py` | Per-model six-mode inference and explicit image/video configuration |
| `gpu_suite/locate_runtime.py` | Isolated Locate Anything execution and result validation |
| [Repository skill](../.agents/skills/klygo-multi-gpu-benchmark/SKILL.md) | Run profiles, dependency repair, recovery, and evidence requirements |

The skill's launcher calls `evaluate_release.py` directly. Keep one canonical
release implementation. The superseded Round 1/2 runners and reports have been
removed from the current tree; their history remains available in Git.

### CPU validation without external downloads

Run from the repository root with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES="" python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py \
  --device cpu --dataset synthetic --repeats 1 --include-cpu-utils \
  --output artifacts/benchmarks/cpu_smoke
```

On PowerShell, set `$env:CUDA_VISIBLE_DEVICES=""` before the Python command.
This is a validation profile, not a published GPU performance result.

### Inspect options

```bash
python .agents/skills/klygo-multi-gpu-benchmark/scripts/run_benchmark.py --help
```

For full GPU release commands, prepared YOLO datasets, precision and scaling,
follow [the repository skill](../.agents/skills/klygo-multi-gpu-benchmark/SKILL.md).
Use `--require-locate-results` for a full release. Requested Locate measurements
must pass before combined charts/report/ZIP are produced. Failed runs retain
partial results and logs; `--locate-only` repairs that section without rerunning
other models when the saved configuration and environment match.

`--dataset-dir` controls real-data loading, mining, and access tests. Model
inference has an independent fixture and iteration protocol. Do not describe a
5,000-image dataset report as 5,000-image inference per model without evidence.

## Specialized benchmarks

| Directory | Purpose |
| --- | --- |
| [outputs_cpu_gpu](outputs_cpu_gpu/README.md) | Focused NumPy/CuPy NMS, storage, precision and device-transfer measurements |
| [mining](mining/README.md) | Mining parity/latency and curation accuracy against held-out ground truth |
| [managed_pipeline](managed_pipeline/README.md) | Preparation costs and before/after managed-pipeline optimization evidence |

These answer narrower questions than the release suite and remain useful.
Curation accuracy measures selection quality, rather than inference throughput.

## Recorded evidence

Keep historical hardware and optimization results distinct from current release
validation. Compare runs only when source, inputs, precision, device, timing and
repeat protocols match.

| Report | Scope |
| --- | --- |
| [L4 COCO val2017, 2026-11-15](gpu_suite/colab_l4_val2017_2026_11_15/REPORT.md) | Real-data loader/mining/access report on 5,000 images; both Locate precisions failed due to missing `cv2` |
| [L4 evaluation, 2026-11-15](gpu_suite/colab_l4_2026_11_15/REPORT.md) | Earlier release evaluation; retain its own recorded environment and dataset scope |
| [T4 six-mode report](gpu_suite/colab_t4_2026_11_10/README.md) | Earlier inference/parity summaries and stability telemetry |
| [T4 complete suite](gpu_suite/colab_t4_complete_2026_11_10/README.md) | Controlled, people and cars workloads with separately downloadable raw evidence |
| [L4 complete suite](gpu_suite/colab_l4_complete_2026_11_11/README.md) | Baseline inference, parity and stability evidence |
| [L4 optimized suite](gpu_suite/colab_l4_optimized_2026_11_11/README.md) | Optimization comparison against the L4 baseline |
| [Mining curation accuracy](mining/curation_l4_2026_11_11/README.md) | Real-data strategy evaluation on noisy/long-tail COCO inputs and clean held-out frames |

The historical val2017 report is incomplete for Locate. Keep it as recorded
evidence until a fresh, provenance-complete full release succeeds; do not relabel
its failed cases as passing or fabricate raw configuration for old summary files.

## Runtime outputs and optional downloads

Write new runs under `artifacts/benchmarks/`. Preserve raw JSON, environment and
run configuration, logs, generated reports/charts, and the optional ZIP. Publish
selected evidence only when requested. Keep downloaded weights, generated inputs,
managed environments and runtime artifacts outside source commits.

Original ZIP evidence is hosted in GitHub Releases. See [downloads and
checksums](DOWNLOADS.md). Retain the release manifest and checksums with the
recorded reports so the original evidence can still be verified.
