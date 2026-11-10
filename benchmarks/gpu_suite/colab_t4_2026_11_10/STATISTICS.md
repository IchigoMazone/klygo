# Descriptive benchmark statistics

Derived from the companion saved median-FPS summary and complete stability samples. No additional GPU run was performed.

**Statistical limit:** per-case matrix SD, confidence intervals and significance cannot be recovered from medians. Ratios below are ratios of rounded medians, not medians of paired ratios. Differences near the rounding precision are not meaningful.

Throughput retained = mode FPS / raw non-stream FPS. Elapsed overhead = raw FPS / mode FPS − 1 for equal frame counts. Negative overhead means lower measured elapsed time. These are end-to-end comparisons, not isolated wrapper cost.

## Against raw non-stream: images, batch 1

| Model | Mode | FPS | Raw throughput retained % | Elapsed overhead % |
|---|---|---:|---:|---:|
|yolo|raw-stream|97.336|97.76|+2.29|
|yolo|native-non-stream|70.885|71.19|+40.46|
|yolo|native-stream|71.132|71.44|+39.98|
|yolo|managed-non-stream|63.101|63.37|+57.79|
|yolo|managed-stream|58.624|58.88|+69.84|
|yolo11|raw-stream|76.940|99.86|+0.14|
|yolo11|native-non-stream|58.242|75.59|+32.29|
|yolo11|native-stream|59.657|77.43|+29.15|
|yolo11|managed-non-stream|55.705|72.30|+38.31|
|yolo11|managed-stream|50.436|65.46|+52.76|
|yolo26|raw-stream|73.680|103.34|-3.23|
|yolo26|native-non-stream|55.188|77.40|+29.19|
|yolo26|native-stream|51.060|71.61|+39.64|
|yolo26|managed-non-stream|49.424|69.32|+44.26|
|yolo26|managed-stream|42.364|59.42|+68.30|
|yolo-world|raw-stream|68.617|111.46|-10.28|
|yolo-world|native-non-stream|58.975|95.80|+4.38|
|yolo-world|native-stream|56.068|91.08|+9.80|
|yolo-world|managed-non-stream|53.515|86.93|+15.04|
|yolo-world|managed-stream|50.271|81.66|+22.46|
|grounding-dino|native-non-stream|3.078|100.10|-0.10|
|grounding-dino|managed-non-stream|3.085|100.33|-0.32|
|grounding-dino|managed-stream|2.962|96.33|+3.81|
|grounding-dino-base|native-non-stream|2.079|96.34|+3.80|
|grounding-dino-base|managed-non-stream|2.094|97.03|+3.06|
|grounding-dino-base|managed-stream|2.121|98.29|+1.74|
|faster-rcnn|native-non-stream|28.519|92.80|+7.76|
|faster-rcnn|managed-non-stream|28.273|92.00|+8.69|
|faster-rcnn|managed-stream|27.003|87.87|+13.81|
|d-fine|native-non-stream|2.183|94.46|+5.86|
|d-fine|managed-non-stream|2.125|91.95|+8.75|
|d-fine|managed-stream|2.187|94.63|+5.67|
|locate-anything|native-non-stream|1.546|99.04|+0.97|
|locate-anything|managed-non-stream|1.550|99.30|+0.71|
|locate-anything|managed-stream|1.573|100.77|-0.76|

## Against raw non-stream: images, batch 4

| Model | Mode | FPS | Raw throughput retained % | Elapsed overhead % |
|---|---|---:|---:|---:|
|yolo|raw-stream|151.244|96.79|+3.32|
|yolo|native-non-stream|107.758|68.96|+45.01|
|yolo|native-stream|108.747|69.59|+43.70|
|yolo|managed-non-stream|98.595|63.09|+58.49|
|yolo|managed-stream|89.237|57.11|+75.11|
|yolo11|raw-stream|154.403|97.75|+2.30|
|yolo11|native-non-stream|108.491|68.68|+45.60|
|yolo11|native-stream|108.003|68.37|+46.25|
|yolo11|managed-non-stream|101.729|64.40|+55.27|
|yolo11|managed-stream|92.212|58.38|+71.30|
|yolo26|raw-stream|159.109|97.98|+2.06|
|yolo26|native-non-stream|108.978|67.11|+49.01|
|yolo26|native-stream|114.069|70.24|+42.36|
|yolo26|managed-non-stream|99.755|61.43|+62.79|
|yolo26|managed-stream|86.569|53.31|+87.59|
|yolo-world|raw-stream|100.669|101.36|-1.34|
|yolo-world|native-non-stream|86.816|87.41|+14.40|
|yolo-world|native-stream|84.431|85.01|+17.63|
|yolo-world|managed-non-stream|70.897|71.38|+40.09|
|yolo-world|managed-stream|58.098|58.50|+70.95|
|grounding-dino|native-non-stream|3.255|99.72|+0.28|
|grounding-dino|managed-non-stream|3.266|100.06|-0.06|
|grounding-dino|managed-stream|3.251|99.60|+0.40|
|grounding-dino-base|native-non-stream|2.290|98.88|+1.14|
|grounding-dino-base|managed-non-stream|2.284|98.62|+1.40|
|grounding-dino-base|managed-stream|2.286|98.70|+1.31|
|faster-rcnn|native-non-stream|37.106|93.39|+7.08|
|faster-rcnn|managed-non-stream|36.085|90.82|+10.11|
|faster-rcnn|managed-stream|34.516|86.87|+15.11|
|d-fine|native-non-stream|8.170|94.57|+5.74|
|d-fine|managed-non-stream|7.930|91.79|+8.94|
|d-fine|managed-stream|8.466|98.00|+2.04|

## Against raw non-stream: video, batch 1

| Model | Mode | FPS | Raw throughput retained % | Elapsed overhead % |
|---|---|---:|---:|---:|
|yolo|raw-stream|84.032|102.31|-2.26|
|yolo|native-non-stream|74.410|90.60|+10.38|
|yolo|native-stream|71.231|86.73|+15.31|
|yolo|managed-non-stream|70.508|85.85|+16.49|
|yolo|managed-stream|70.033|85.27|+17.28|
|yolo11|raw-stream|63.475|90.04|+11.07|
|yolo11|native-non-stream|62.021|87.97|+13.67|
|yolo11|native-stream|62.787|89.06|+12.28|
|yolo11|managed-non-stream|59.043|83.75|+19.40|
|yolo11|managed-stream|59.375|84.22|+18.74|
|yolo26|raw-stream|63.750|101.26|-1.25|
|yolo26|native-non-stream|60.860|96.67|+3.44|
|yolo26|native-stream|55.878|88.76|+12.66|
|yolo26|managed-non-stream|54.967|87.31|+14.53|
|yolo26|managed-stream|56.028|89.00|+12.36|
|yolo-world|raw-stream|64.675|100.65|-0.64|
|yolo-world|native-non-stream|62.198|96.79|+3.32|
|yolo-world|native-stream|58.484|91.01|+9.88|
|yolo-world|managed-non-stream|58.049|90.33|+10.70|
|yolo-world|managed-stream|56.799|88.39|+13.14|

## Against raw non-stream: video, batch 4

| Model | Mode | FPS | Raw throughput retained % | Elapsed overhead % |
|---|---|---:|---:|---:|
|yolo|raw-stream|148.175|106.80|-6.36|
|yolo|managed-non-stream|128.142|92.36|+8.28|
|yolo|managed-stream|122.721|88.45|+13.06|
|yolo11|raw-stream|137.682|98.38|+1.65|
|yolo11|managed-non-stream|117.277|83.80|+19.34|
|yolo11|managed-stream|111.254|79.49|+25.80|
|yolo26|raw-stream|120.980|99.04|+0.97|
|yolo26|managed-non-stream|111.036|90.90|+10.01|
|yolo26|managed-stream|107.239|87.79|+13.91|
|yolo-world|raw-stream|108.777|100.63|-0.63|
|yolo-world|managed-non-stream|101.799|94.18|+6.18|
|yolo-world|managed-stream|98.001|90.66|+10.30|

## Batch scaling

Batch 4 / batch 1 is a throughput comparison across different batch settings. It is not a same-workload optimization gain or a latency speedup. Image packs use eight calls in both settings, so batch 4 also processes four times as many images.

| Model | Profile | Mode | Batch 1 FPS | Batch 4 FPS | Throughput ratio |
|---|---|---|---:|---:|---:|
|yolo|images|raw-non-stream|99.568|156.265|1.569|
|yolo|images|raw-stream|97.336|151.244|1.554|
|yolo|images|native-non-stream|70.885|107.758|1.520|
|yolo|images|native-stream|71.132|108.747|1.529|
|yolo|images|managed-non-stream|63.101|98.595|1.562|
|yolo|images|managed-stream|58.624|89.237|1.522|
|yolo11|images|raw-non-stream|77.048|157.958|2.050|
|yolo11|images|raw-stream|76.940|154.403|2.007|
|yolo11|images|native-non-stream|58.242|108.491|1.863|
|yolo11|images|native-stream|59.657|108.003|1.810|
|yolo11|images|managed-non-stream|55.705|101.729|1.826|
|yolo11|images|managed-stream|50.436|92.212|1.828|
|yolo26|images|raw-non-stream|71.300|162.393|2.278|
|yolo26|images|raw-stream|73.680|159.109|2.159|
|yolo26|images|native-non-stream|55.188|108.978|1.975|
|yolo26|images|native-stream|51.060|114.069|2.234|
|yolo26|images|managed-non-stream|49.424|99.755|2.018|
|yolo26|images|managed-stream|42.364|86.569|2.043|
|yolo-world|images|raw-non-stream|61.561|99.317|1.613|
|yolo-world|images|raw-stream|68.617|100.669|1.467|
|yolo-world|images|native-non-stream|58.975|86.816|1.472|
|yolo-world|images|native-stream|56.068|84.431|1.506|
|yolo-world|images|managed-non-stream|53.515|70.897|1.325|
|yolo-world|images|managed-stream|50.271|58.098|1.156|
|grounding-dino|images|raw-non-stream|3.075|3.264|1.061|
|grounding-dino|images|native-non-stream|3.078|3.255|1.058|
|grounding-dino|images|managed-non-stream|3.085|3.266|1.059|
|grounding-dino|images|managed-stream|2.962|3.251|1.098|
|grounding-dino-base|images|raw-non-stream|2.158|2.316|1.073|
|grounding-dino-base|images|native-non-stream|2.079|2.290|1.101|
|grounding-dino-base|images|managed-non-stream|2.094|2.284|1.091|
|grounding-dino-base|images|managed-stream|2.121|2.286|1.078|
|faster-rcnn|images|raw-non-stream|30.731|39.732|1.293|
|faster-rcnn|images|native-non-stream|28.519|37.106|1.301|
|faster-rcnn|images|managed-non-stream|28.273|36.085|1.276|
|faster-rcnn|images|managed-stream|27.003|34.516|1.278|
|d-fine|images|raw-non-stream|2.311|8.639|3.738|
|d-fine|images|native-non-stream|2.183|8.170|3.743|
|d-fine|images|managed-non-stream|2.125|7.930|3.732|
|d-fine|images|managed-stream|2.187|8.466|3.871|
|yolo|video|raw-non-stream|82.134|138.746|1.689|
|yolo|video|raw-stream|84.032|148.175|1.763|
|yolo|video|managed-non-stream|70.508|128.142|1.817|
|yolo|video|managed-stream|70.033|122.721|1.752|
|yolo11|video|raw-non-stream|70.499|139.956|1.985|
|yolo11|video|raw-stream|63.475|137.682|2.169|
|yolo11|video|managed-non-stream|59.043|117.277|1.986|
|yolo11|video|managed-stream|59.375|111.254|1.874|
|yolo26|video|raw-non-stream|62.954|122.154|1.940|
|yolo26|video|raw-stream|63.750|120.980|1.898|
|yolo26|video|managed-non-stream|54.967|111.036|2.020|
|yolo26|video|managed-stream|56.028|107.239|1.914|
|yolo-world|video|raw-non-stream|64.260|108.092|1.682|
|yolo-world|video|raw-stream|64.675|108.777|1.682|
|yolo-world|video|managed-non-stream|58.049|101.799|1.754|
|yolo-world|video|managed-stream|56.799|98.001|1.725|

## Stability distribution

312 consecutive loops from one loaded model. Quantiles describe observed loop samples; samples are correlated. They are not frame-level latency percentiles or production tail guarantees. SD is descriptive sample SD, not uncertainty of the matrix medians.

| Metric | Min | P05 | Median | P95 | Max | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|---:|---:|
|fps|47.224|49.914|73.234|77.234|78.266|69.452|8.797|
|rss_mib|1608.266|1608.404|1608.707|1619.156|1619.250|1612.399|4.469|
|vram_allocated_mib|12.095|12.095|12.095|12.095|12.095|12.095|0.000|
|vram_reserved_mib|60.000|60.000|60.000|60.000|60.000|60.000|0.000|
|open_fds|45.000|45.000|45.000|45.000|45.000|45.000|0.000|

Whole-check throughput including loop cleanup: **62.217 FPS**. Loop FPS above excludes cleanup. First/last-minute memory and FD gates are retained in the original report; min/max here additionally expose intermediate variation.

## Evidence completeness

| Evidence | Availability |
|---|---|
| Median FPS / printed parity | 124 supported cases, available |
| Unsupported modes | 26 N/A cells, recorded capability reasons |
| Seven repeat markers per selected job | Available in archived stdout |
| 868 individual matrix timing / parity records | Missing from saved output |
| Matrix SD / confidence intervals / TTFF / P95 / P99 | Cannot independently reconstruct |
| Stability telemetry | Complete: 312 samples, hash verified on import |
| Ground-truth accuracy / diverse real video | Not evaluated |

The existing experiment supports descriptive performance comparison and a ten-minute memory/FD check. Full statistical review requires the original matrix bundle or a fresh run with file-based export. Use `export_bundle.py` to reject incomplete raw reports and preserve the next complete run as a ZIP.
