# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|109.235|107.888 ± 3.177|100.961–109.887|2.94|107.245–109.775|0.00|PASS|
|yolo|raw-stream|7|109.669|109.246 ± 1.395|106.351–110.734|1.28|108.804–109.836|-0.39|PASS|
|yolo|native-non-stream|7|105.018|104.773 ± 1.354|102.006–106.025|1.29|104.372–105.800|4.31|PASS|
|yolo|native-stream|7|102.579|102.762 ± 2.247|99.305–105.433|2.19|100.678–104.890|4.84|PASS|
|yolo|managed-non-stream|7|104.700|102.642 ± 3.238|97.830–105.041|3.15|98.136–104.800|4.33|PASS|
|yolo|managed-stream|7|102.432|100.180 ± 3.547|96.353–103.898|3.54|96.403–103.084|7.12|PASS|
|yolo11|raw-non-stream|7|90.090|90.305 ± 1.715|87.417–92.402|1.90|89.407–91.718|0.00|PASS|
|yolo11|raw-stream|7|91.098|90.875 ± 1.790|87.345–93.288|1.97|90.514–91.573|-0.90|PASS|
|yolo11|native-non-stream|7|88.144|88.414 ± 1.664|85.748–90.915|1.88|87.556–89.951|2.72|PASS|
|yolo11|native-stream|7|87.317|87.615 ± 0.977|86.310–89.441|1.11|87.130–87.976|3.78|PASS|
|yolo11|managed-non-stream|7|87.206|87.676 ± 0.957|86.562–89.104|1.09|86.910–88.576|3.66|PASS|
|yolo11|managed-stream|7|86.847|85.900 ± 2.200|82.335–88.471|2.56|84.008–87.492|5.40|PASS|
|yolo26|raw-non-stream|7|80.459|80.117 ± 1.387|78.085–81.480|1.73|78.294–81.251|0.00|PASS|
|yolo26|raw-stream|7|80.267|80.277 ± 0.916|78.783–81.544|1.14|79.613–80.935|-0.36|PASS|
|yolo26|native-non-stream|7|77.995|77.590 ± 1.509|74.950–78.892|1.94|76.042–78.820|3.08|PASS|
|yolo26|native-stream|7|77.730|77.347 ± 1.043|75.822–78.481|1.35|76.022–78.098|3.26|PASS|
|yolo26|managed-non-stream|7|77.830|77.636 ± 1.028|75.799–78.865|1.32|76.801–78.407|3.29|PASS|
|yolo26|managed-stream|7|76.956|76.806 ± 1.479|74.136–78.576|1.93|75.976–78.241|5.28|PASS|
|yolo-world|raw-non-stream|7|79.843|80.413 ± 1.703|78.651–82.541|2.12|78.758–82.304|0.00|PASS|
|yolo-world|raw-stream|7|79.600|79.608 ± 0.816|78.217–80.646|1.02|79.131–80.287|0.69|PASS|
|yolo-world|native-non-stream|7|77.008|76.788 ± 0.747|75.452–77.575|0.97|76.407–77.563|5.82|PASS|
|yolo-world|native-stream|7|76.262|76.421 ± 1.871|73.974–79.356|2.45|74.441–77.837|4.01|PASS|
|yolo-world|managed-non-stream|7|76.462|77.035 ± 1.926|74.631–80.472|2.50|75.887–78.301|4.28|PASS|
|yolo-world|managed-stream|7|75.678|76.329 ± 2.279|74.118–79.902|2.99|74.277–78.404|5.45|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.098549|1098.462|1081.545/1081.545/1081.545|1927.40|34.28/62.00|
|yolo|raw-stream|1.094206|23.551|1156.011/1156.011/1156.011|1933.99|34.28/62.00|
|yolo|native-non-stream|1.142660|1142.574|1141.697/1141.697/1141.697|1934.00|34.28/62.00|
|yolo|native-stream|1.169824|24.729|1161.166/1161.166/1161.166|1927.40|34.25/62.00|
|yolo|managed-non-stream|1.146135|1146.042|1145.615/1145.615/1145.615|1826.18|24.66/62.00|
|yolo|managed-stream|1.171508|32.576|1164.589/1164.589/1164.589|1826.34|24.66/62.00|
|yolo11|raw-non-stream|1.332008|1331.929|1295.880/1295.880/1295.880|2005.83|64.67/90.00|
|yolo11|raw-stream|1.317262|25.378|1307.149/1307.149/1307.149|2005.83|64.67/90.00|
|yolo11|native-non-stream|1.361413|1361.291|1325.931/1325.931/1325.931|2009.16|64.67/90.00|
|yolo11|native-stream|1.374308|26.464|1329.658/1329.658/1329.658|2005.80|64.67/90.00|
|yolo11|managed-non-stream|1.376060|1375.959|1324.178/1324.178/1324.178|1904.57|56.12/90.00|
|yolo11|managed-stream|1.381736|33.504|1361.653/1361.653/1361.653|1904.64|56.12/90.00|
|yolo26|raw-non-stream|1.491439|1491.359|1564.366/1564.366/1564.366|1937.64|62.83/90.00|
|yolo26|raw-stream|1.495003|26.923|1571.067/1571.067/1571.067|1944.16|62.83/90.00|
|yolo26|native-non-stream|1.538562|1538.460|1559.785/1559.785/1559.785|1944.20|62.83/90.00|
|yolo26|native-stream|1.543813|27.878|1591.448/1591.448/1591.448|1937.61|62.80/90.00|
|yolo26|managed-non-stream|1.541821|1541.720|1575.489/1575.489/1575.489|1836.27|55.34/90.00|
|yolo26|managed-stream|1.559336|35.795|1606.854/1606.854/1606.854|1836.28|55.34/90.00|
|yolo-world|raw-non-stream|1.502956|1502.864|1484.330/1484.330/1484.330|2193.12|115.88/134.00|
|yolo-world|raw-stream|1.507534|27.128|1473.633/1473.633/1473.633|2193.12|115.88/134.00|
|yolo-world|native-non-stream|1.558283|1558.175|1537.816/1537.816/1537.816|2193.10|115.88/134.00|
|yolo-world|native-stream|1.573517|27.664|1512.411/1512.411/1512.411|2193.09|115.88/134.00|
|yolo-world|managed-non-stream|1.569417|1569.318|1510.580/1510.580/1510.580|2083.80|107.97/134.00|
|yolo-world|managed-stream|1.585675|35.102|1513.825/1513.825/1513.825|2083.84|107.97/134.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
