# Reproduce and report the six-mode benchmark on another GPU

Use the same measurement protocol for each accelerator and keep one independent
run directory per GPU and session. The full raw reports are the source of truth;
the printed FPS lines are only progress summaries.

## 1. Freeze the comparison

Record an exact Git commit, not only a branch name. Use the same runner revision,
model checkpoint/revision, input file SHA256, prompts, thresholds, resize policy,
precision and batch settings on every GPU being compared. A run with a different
configuration is a separate experiment, even if its GPU is faster.

Use a fresh Python environment and install a Torch build supported by that GPU
and driver. On Colab, preserve the working CUDA Torch/Torchvision installation.
Install the model extras from the repository and record the resolved package
versions. The recorded T4 baseline uses Ultralytics 8.4.208 and Transformers
4.57.1; other package versions must also be included in the environment manifest.
Do not silently update the model framework during a hardware comparison.

From a checkout at your selected commit:

```bash
python -m pip install -e . ultralytics==8.4.208 transformers==4.57.1 accelerate \
  keras-hub timm open_clip_torch nvidia-ml-py decord lmdb
mkdir -p artifacts/benchmarks/session
python -m pip freeze > artifacts/benchmarks/session/environment.txt
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,power.limit --format=csv
```

The freeze file records the actual resolved environment; it is not an instruction
to install a T4-specific CUDA wheel on arbitrary hardware. Save it alongside the
run. The model and video downloads must finish before steady-state timing begins;
model loading/download times are reported separately. Keep the CPU thread policy,
codec and host hardware visible because decode and preprocessing also affect FPS.

## 2. Select exactly one GPU

For CUDA GPUs, expose one physical device by UUID and use logical `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_DEVICE_UUID PYTHONPATH=. \
python benchmarks/gpu_suite/run.py --model yolo --device cuda:0 \
  --profile images --batch 1 --imgsz 640 --iterations 64 --repeats 7 \
  --output artifacts/benchmarks/gpu-name/session-01/images_b1
```

On PowerShell, set `$env:CUDA_VISIBLE_DEVICES` and `$env:PYTHONPATH` before running
the same Python command. Do not run competing model jobs on that GPU during a
comparison. Record device UUID/name, VRAM, driver, CUDA/Torch versions, CPU,
thread counts and sampled power/clocks/temperature.

The current runner maps NVML telemetry through the active CUDA UUID. The older
`cffb476` runner used physical NVML index zero; its recorded single-T4 run is
unambiguous, but do not use that older telemetry mapping for a remapped multi-GPU
run. Missing NVML data must stay unavailable rather than being attributed to a
different device. CUDA UUID addressing does not imply multi-GPU inference.

This suite is CUDA-focused. CPU/MPS/ROCm require explicit validation of timing
synchronization, model support and memory counters; changing the device string
alone does not establish an equivalent benchmark. Keep accelerator families in
separate reports until their measurement paths are validated.
The stability helper currently uses Linux file-descriptor counters. On Windows,
adapt that gate to process handles and document the different metric; do not
label a failed or unimplemented FD gate as passing.

## 3. Run the same matrix

The image profile uses predecoded PIL packs. The video profile includes source
loading and H.264 decode. Measure both batch 1 and batch 4 when supported.

| Model key | Image batches | Iterations in the complete T4 rerun | Video batches |
|---|---|---:|---|
| yolo | 1, 4 | 64 | 1, 4 |
| yolo11 | 1, 4 | 64 | 1, 4 |
| yolo26 | 1, 4 | 64 | 1, 4 |
| yolo-world | 1, 4 | 64 | 1, 4 |
| grounding-dino | 1, 4 | 8 | Not measured |
| grounding-dino-base | 1, 4 | 8 | Not measured |
| faster-rcnn | 1, 4 | 32 | Not measured |
| d-fine | 1, 4 | 8 | Not measured |
| locate-anything | 1 | 1 | Not measured |

All configurations use seven measured repeats after a complete warmup of each
supported mode. Alternate forward/reverse mode order. The video workload contains
120 controlled 1280x720 H.264 frames; `--iterations` does not repeat that clip in
the video profile. LocateAnything uses FP16 and one query; other families use
FP32. Preserve these differences explicitly in the report.

Example video command:

```bash
PYTHONPATH=. python benchmarks/gpu_suite/run.py --model yolo --device cuda:0 \
  --profile video --video-frames 120 --batch 4 --imgsz 640 --repeats 7 \
  --output artifacts/benchmarks/gpu-name/session-01/video_b4
```

For LocateAnything, add `--allow-remote-code` only for the declared pinned NVIDIA
checkpoint after reviewing that model integration. Install `decord` and `lmdb`
before the run. Record any failed/OOM attempt and every configuration change.
Do not select only a faster retry. Native video batch 4 is N/A under the tested
API; frameworks without a source-stream engine keep raw/native stream N/A.
Raw non-stream remains a required baseline for every successfully measured model.

On a smaller GPU, first run a separate one-repeat smoke test. If the standard
batch/resolution cannot fit, record OOM for that configuration and add a clearly
separate smaller profile. Do not overwrite the standard experiment or compare
its reduced input directly with a larger-GPU standard row. On a faster GPU, keep
the standard profile; add an extended-duration run if passes become too short.

## 4. Keep complete files and check them

The runner saves each repeat's timing/parity records before the job completes.
Keep those JSON files, stdout logs, `pip freeze`, input/checkpoint hashes and
source/runner hashes. After the selected jobs finish, explicitly list them:

```bash
PYTHONPATH=. python benchmarks/gpu_suite/export_bundle.py \
  --report artifacts/benchmarks/gpu-name/session-01/images_b1/yolo.json \
  --report artifacts/benchmarks/gpu-name/session-01/video_b4/yolo.json \
  --expected-repeats 7 \
  --output artifacts/benchmarks/gpu-name/session-01/complete.zip
```

Add one `--report` per successful selected configuration; include failed attempts
with `--failed-report`. Add `--stability path/to/stability.json` when measured.
The export checks completeness/internal consistency and packages `bundle.json`,
`REPORT.md` and a SHA256 manifest. Keep each input JSON and log as well.
Download the ZIP as a file in Colab; do not rely on a multi-megabyte HTML output.

Extract the ZIP and compute statistics locally without a GPU:

```bash
PYTHONPATH=. python benchmarks/gpu_suite/raw_statistics.py \
  artifacts/benchmarks/gpu-name/session-01/extracted/bundle.json \
  --output artifacts/benchmarks/gpu-name/session-01/statistics
```

The tables retain raw non-stream rows, median/mean/sample SD, min/max, CV,
exploratory repeat-bootstrap intervals, paired elapsed overhead, TTFF and the
separate synchronized latency diagnostic. A parity failure remains FAIL. N/A is
a capability limit; ERROR/OOM is an attempted configuration that failed.

## 5. Interpret and write the benchmark

Use one row per GPU, model, input/profile, batch, precision and mode. State the
number of frames/calls per pass and number of repeats. Compare equal rows first.
Hardware comparisons should use fresh independent sessions, preferably several
sessions per GPU with run order varied. Seven repeats in one warm process do not
capture between-session variance. Bootstrap intervals from that small serial
sample are exploratory and do not establish statistical significance.

Describe the timing boundary accurately: this runner synchronizes CUDA around a
complete pass and includes source handling, prediction and result consumption.
Normalization/parity conversion is outside the timer, while construction of
Klygo outputs inside prediction remains part of the path. Throughput passes retain
results for later checking, so their memory peaks differ from a non-retaining
live stream. Report the ten-minute stability probe separately. Host delivery
intervals and API-pack quantiles are not isolated kernel latency or a production
P99 guarantee. Sampled RSS/NVML counters can miss short peaks.

A suitable results paragraph must name the actual measurements, for example:

> We evaluated [commit] on [GPU and VRAM], with [driver, CUDA, Torch/framework
> versions] and [CPU/thread policy]. Each [input digest/profile, resolution,
> batch, precision] configuration had one full warmup per mode and seven measured
> repeats in alternating mode order. We report median FPS, mean and sample SD,
> min/max and parity against raw non-stream. Full raw repeats and telemetry are
> archived under [run identifier/checksum]. [Number] configurations completed;
> [number] failed and [number] modes were unsupported. Results concern this
> workload and runtime; ground-truth accuracy and production latency were not
> evaluated.

Replace every bracket with verified values from the raw reports. Do not claim a
universal wrapper speedup, detector accuracy, or linear GPU scaling from the
controlled bus workload. To extend workload coverage, add fixed real videos with
documented digests and consistent preprocessing, and keep their results separate
from the historical controlled input.
