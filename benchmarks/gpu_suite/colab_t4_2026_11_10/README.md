# Recorded Colab Tesla T4 benchmark

Imported on 2026-11-10 from the shared [Colab notebook](https://colab.research.google.com/drive/1l2ejVBidUcgaw907giH5V6WIdD46hxMo?usp=sharing).

The saved notebook contains 25 selected configurations across nine model variants and four backend families, 124 supported mode cases and seven repeats per case (868 reported measured passes). Every selected printed parity flag is PASS. These are imported GPU measurements, not a new run.

## Evidence and completeness

See [STATISTICS.md](STATISTICS.md) and [statistics.json](statistics.json) for
derived overhead, throughput retention, batch scaling and stability quantiles.
These preserve the distinction between observed medians and unavailable raw
matrix timing samples; no confidence intervals are fabricated.

The statistics/export follow-up passed eight focused checks, including mode
routing, incomplete-report rejection and overhead formula direction. A synthetic
full-report ZIP was exported, extracted and verified against its manifest.
All 13 module audits passed. Full local discovery ran 505 tests with 21 skips
and the same six missing-extra import errors; no new GPU run was performed.

- [Colab_Executed.ipynb](Colab_Executed.ipynb): all 18 original cell sources and available saved outputs, including historical before/after experiments, failures, retries and control checks. Transient cell metadata and executable download outputs were removed. This is an execution archive; do not use Run all as a clean reproduction recipe.
- [summary.json](summary.json): the 124 median FPS values and parity flags printed to three decimal places, selected LocateAnything retry, six separate control-verification values, source cells and commit pins. Unsupported statuses follow the pinned runner capability rules.
- [stability.json](stability.json): complete recovered stability telemetry and gate results. The embedded compact JSON SHA256 was verified as `77b72b61d13918946b10199806bbadc9672db8305d940e4fceefd5beb3514e60`.

**Missing matrix raw data:** cell 15 contains only an "Output hidden" placeholder in the saved notebook. The original `bundle.json`, 868 individual timing records, per-case latency percentiles and detailed telemetry are not included in this import. Rounded medians cannot reconstruct those samples. Do not pass `summary.json` to `../analyze.py`, which expects the original full bundle schema.

## Environment and protocol

Tesla T4, 15360 MiB device memory (14.56 GiB reported by Torch), driver 580.82.07, 70 W limit. Torch 2.14.1+cu130, Ultralytics 8.4.208, Transformers 4.57.1. Matrix source: `313897b4b40c8ebfd366ac1d99460727ed48919e`; control verification and stability: `5f9aef9a0b7ed1fa93b51425bb232ad7d7de64e4`. The notebook checks that `klygo/` and `pyproject.toml` are identical between these revisions.

See the [suite protocol](../README.md). Complete warmup precedes seven alternating timed repeats; CUDA is synchronized around complete passes. Images are predecoded 640x480 PIL packs, repeated eight times; video is a 120-frame controlled 1280x720 H.264 bus clip including decode. FP32 is used except LocateAnything FP16, one query, fast generation, 512-token limit and one pack per repeat. These are different workloads and precision settings.

N/A means no upstream source-stream engine, or native video batch locked to one. Native-stream buffers into a materialized Klygo result at these revisions. Parity is cross-path numerical/label equivalence, not ground-truth detection accuracy.

## Median FPS: images, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
|---|---:|---:|---:|---:|---:|---:|
|yolo|99.568|97.336|70.885|71.132|63.101|58.624|
|yolo11|77.048|76.940|58.242|59.657|55.705|50.436|
|yolo26|71.300|73.680|55.188|51.060|49.424|42.364|
|yolo-world|61.561|68.617|58.975|56.068|53.515|50.271|
|grounding-dino|3.075|N/A|3.078|N/A|3.085|2.962|
|grounding-dino-base|2.158|N/A|2.079|N/A|2.094|2.121|
|faster-rcnn|30.731|N/A|28.519|N/A|28.273|27.003|
|d-fine|2.311|N/A|2.183|N/A|2.125|2.187|
|locate-anything|1.561|N/A|1.546|N/A|1.550|1.573|

## Median FPS: images, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
|---|---:|---:|---:|---:|---:|---:|
|yolo|156.265|151.244|107.758|108.747|98.595|89.237|
|yolo11|157.958|154.403|108.491|108.003|101.729|92.212|
|yolo26|162.393|159.109|108.978|114.069|99.755|86.569|
|yolo-world|99.317|100.669|86.816|84.431|70.897|58.098|
|grounding-dino|3.264|N/A|3.255|N/A|3.266|3.251|
|grounding-dino-base|2.316|N/A|2.290|N/A|2.284|2.286|
|faster-rcnn|39.732|N/A|37.106|N/A|36.085|34.516|
|d-fine|8.639|N/A|8.170|N/A|7.930|8.466|

## Median FPS: video, batch 1

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
|---|---:|---:|---:|---:|---:|---:|
|yolo|82.134|84.032|74.410|71.231|70.508|70.033|
|yolo11|70.499|63.475|62.021|62.787|59.043|59.375|
|yolo26|62.954|63.750|60.860|55.878|54.967|56.028|
|yolo-world|64.260|64.675|62.198|58.484|58.049|56.799|

## Median FPS: video, batch 4

| Model | raw-non-stream | raw-stream | native-non-stream | native-stream | managed-non-stream | managed-stream |
|---|---:|---:|---:|---:|---:|---:|
|yolo|138.746|148.175|N/A|N/A|128.142|122.721|
|yolo11|139.956|137.682|N/A|N/A|117.277|111.254|
|yolo26|122.154|120.980|N/A|N/A|111.036|107.239|
|yolo-world|108.092|108.777|N/A|N/A|101.799|98.001|

## Independent routing control check

The final adapter passed two Detector routing tests and a separate seven-repeat YOLO image batch-1 control. These values are additional measurements and are not substituted into the selected matrix.

| Mode | Median FPS | Parity |
|---|---:|---|
|raw-non-stream|89.117|PASS|
|raw-stream|90.881|PASS|
|native-non-stream|68.847|PASS|
|native-stream|71.668|PASS|
|managed-non-stream|60.352|PASS|
|managed-stream|54.248|PASS|

## Ten-minute managed-stream stability

YOLOv8n completed 601.77 seconds and 37440 ordered frames, with no exceptions. Memory/FD gates passed. This non-retaining probe covers one model and pipeline, not every matrix mode.

| Metric | First-minute median | Last-minute median | Change % |
|---|---:|---:|---:|
|rss_mib|1608.402|1608.715|0.019|
|vram_allocated_mib|12.095|12.095|0.000|
|fps|72.453|74.719|3.127|
|open_fds|45.000|45.000|0.000|

## Interpretation and attempt audit

YOLOv8n video batch 1 measured 82.134 FPS raw non-stream, 74.410 FPS native non-stream and 70.508 FPS managed non-stream. At batch 4, raw non-stream measured 138.746 FPS and managed non-stream 128.142 FPS. The native batch-4 video path is unsupported. Compare equivalent profile/batch rows; the short image-pack controls and video numbers measure different API overhead.

Grounding DINO, D-FINE and LocateAnything show small cross-path throughput differences on this workload. Rounded medians without raw repeat samples do not establish statistical significance. Grounding DINO base and LocateAnything are not expected to provide YOLO-level throughput.

The first LocateAnything attempt failed because decord and lmdb were missing; the successful dependency-repaired retry is selected. Sixteen later control probes failed because they explicitly overrode the frozen model group during managed prediction. Those probes are excluded from the tables and retained in the archived outputs. The final routing test verifies transient native overrides and the frozen managed default.

Stability samples show FPS fluctuations despite passing memory/FD gates. These controlled public-bus results do not establish production accuracy, a live-camera latency SLA or a universal speedup. Historical before/after results remain in the managed-pipeline report and must not be combined with this matrix as a single experiment.

## Reproduction

Use the [versioned runner](../run.py) at the recorded revisions and the [CLI recipes](../README.md), matching package versions and settings. Historical code is preserved exactly in the archived cell source; remote model code requires the explicit runner opt-in. For new runs, save runtime outputs under `artifacts/benchmarks/` and retain `bundle.json` as a downloadable file rather than relying on a large HTML output.

## Import validation

All archived Python cells compile; JSON parsing, relative report links and
`git diff --check` pass. The import verifies 25 configurations, 124 printed
supported cases, seven repeat markers per selected job and all printed parity
flags. The embedded stability payload hash matches the notebook export hash.
The public API audit passed for all 13 modules, and the two benchmark routing
tests passed locally. Full local test discovery ran 499 tests with 21 skips and
six import errors because Ultralytics/Transformers extras are absent. No GPU
measurements were rerun during this documentation/data import.
The four-backend integration entry point could not start because Keras is absent.
