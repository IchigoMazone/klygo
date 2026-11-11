# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|111.341|111.093 ± 2.450|108.231–114.892|2.21|108.489–113.131|0.00|PASS|
|yolo|raw-stream|7|110.815|110.191 ± 3.886|101.974–113.494|3.53|109.883–113.005|0.11|PASS|
|yolo|native-non-stream|7|107.743|106.340 ± 3.302|99.990–109.579|3.11|104.411–108.853|3.66|PASS|
|yolo|native-stream|7|104.815|103.967 ± 3.117|97.245–106.589|3.00|103.798–105.989|5.30|PASS|
|yolo|managed-non-stream|7|104.048|103.605 ± 2.313|100.503–106.533|2.23|100.823–105.749|7.01|PASS|
|yolo|managed-stream|7|106.521|105.311 ± 2.793|99.258–107.461|2.65|104.843–106.658|4.39|PASS|
|yolo11|raw-non-stream|7|92.009|91.024 ± 2.255|86.150–92.753|2.48|90.708–92.278|0.00|PASS|
|yolo11|raw-stream|7|92.265|92.161 ± 0.796|90.936–93.324|0.86|91.456–92.780|-1.26|PASS|
|yolo11|native-non-stream|7|89.675|89.616 ± 1.218|87.832–91.117|1.36|88.512–91.009|1.17|PASS|
|yolo11|native-stream|7|87.599|87.664 ± 1.449|85.871–90.154|1.65|86.141–88.533|5.52|PASS|
|yolo11|managed-non-stream|7|88.777|87.984 ± 1.309|86.121–89.214|1.49|86.415–89.086|3.73|PASS|
|yolo11|managed-stream|7|88.311|87.988 ± 1.045|85.730–88.842|1.19|87.948–88.645|3.64|PASS|
|yolo26|raw-non-stream|7|82.882|82.924 ± 0.637|81.968–83.639|0.77|82.313–83.476|0.00|PASS|
|yolo26|raw-stream|7|83.367|83.178 ± 1.132|81.704–85.095|1.36|82.137–83.830|-0.43|PASS|
|yolo26|native-non-stream|7|80.159|80.174 ± 0.836|79.105–81.357|1.04|79.252–81.012|3.20|PASS|
|yolo26|native-stream|7|80.069|79.983 ± 0.534|79.275–80.569|0.67|79.371–80.502|2.96|PASS|
|yolo26|managed-non-stream|7|80.220|80.126 ± 0.734|78.735–81.213|0.92|79.927–80.335|4.05|PASS|
|yolo26|managed-stream|7|79.578|79.698 ± 0.721|78.768–80.911|0.90|79.075–80.284|4.18|PASS|
|yolo-world|raw-non-stream|7|82.320|82.602 ± 0.796|81.593–83.579|0.96|82.029–83.526|0.00|PASS|
|yolo-world|raw-stream|7|82.977|82.523 ± 1.456|79.714–83.665|1.76|81.440–83.565|-0.66|PASS|
|yolo-world|native-non-stream|7|81.356|80.959 ± 1.009|78.947–81.750|1.25|80.275–81.569|1.19|PASS|
|yolo-world|native-stream|7|79.912|79.489 ± 0.856|78.141–80.338|1.08|78.861–80.242|3.97|PASS|
|yolo-world|managed-non-stream|7|78.943|79.353 ± 1.181|77.426–80.703|1.49|78.773–80.523|3.73|PASS|
|yolo-world|managed-stream|7|79.739|79.994 ± 1.047|78.694–81.603|1.31|79.063–80.793|3.59|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.077771|1077.682|1125.448/1125.448/1125.448|1926.45|34.27/62.00|
|yolo|raw-stream|1.082886|23.764|1066.997/1066.997/1066.997|1926.48|34.27/62.00|
|yolo|native-non-stream|1.113757|1113.653|1111.540/1111.540/1111.540|1930.76|34.27/62.00|
|yolo|native-stream|1.144870|23.579|1137.890/1137.890/1137.890|1926.45|34.25/62.00|
|yolo|managed-non-stream|1.153316|1153.215|1113.892/1113.892/1113.892|1824.30|24.66/62.00|
|yolo|managed-stream|1.126542|30.133|1123.508/1123.508/1123.508|1824.29|24.66/62.00|
|yolo11|raw-non-stream|1.304221|1304.136|1288.646/1288.646/1288.646|2012.24|64.67/90.00|
|yolo11|raw-stream|1.300605|24.341|1318.746/1318.746/1318.746|2012.21|64.67/90.00|
|yolo11|native-non-stream|1.338169|1338.077|1360.765/1360.765/1360.765|2012.25|64.67/90.00|
|yolo11|native-stream|1.369880|25.445|1353.106/1353.106/1353.106|2012.27|64.67/90.00|
|yolo11|managed-non-stream|1.351702|1351.604|1362.900/1362.900/1362.900|1903.78|56.12/90.00|
|yolo11|managed-stream|1.358836|31.782|1384.717/1384.717/1384.717|1905.21|56.12/90.00|
|yolo26|raw-non-stream|1.447834|1447.755|1444.426/1444.426/1444.426|1947.03|62.82/90.00|
|yolo26|raw-stream|1.439423|25.558|1461.522/1461.522/1461.522|1947.07|62.82/90.00|
|yolo26|native-non-stream|1.497022|1496.919|1475.959/1475.959/1475.959|1947.05|62.82/90.00|
|yolo26|native-stream|1.498701|27.212|1484.824/1484.824/1484.824|1947.04|62.80/90.00|
|yolo26|managed-non-stream|1.495893|1495.794|1486.054/1486.054/1486.054|1838.85|55.34/90.00|
|yolo26|managed-stream|1.507948|33.419|1513.291/1513.291/1513.291|1845.10|55.34/90.00|
|yolo-world|raw-non-stream|1.457729|1457.626|1457.655/1457.655/1457.655|2192.99|115.88/134.00|
|yolo-world|raw-stream|1.446192|25.732|1452.021/1452.021/1452.021|2192.96|115.88/134.00|
|yolo-world|native-non-stream|1.475004|1474.918|1477.838/1477.838/1477.838|2192.97|115.88/134.00|
|yolo-world|native-stream|1.501644|26.879|1564.880/1564.880/1564.880|2192.96|115.88/134.00|
|yolo-world|managed-non-stream|1.520092|1519.995|1467.253/1467.253/1467.253|2084.65|107.97/134.00|
|yolo-world|managed-stream|1.504917|32.434|1520.191/1520.191/1520.191|2091.82|107.97/134.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
