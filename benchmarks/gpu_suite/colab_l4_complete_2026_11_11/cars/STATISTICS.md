# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|116.049|116.097 ± 1.774|112.937–118.339|1.53|115.089–117.460|0.00|PASS|
|yolo|raw-stream|7|115.643|115.511 ± 2.021|111.616–117.695|1.75|114.927–117.473|0.55|PASS|
|yolo|native-non-stream|7|111.910|111.075 ± 1.479|108.479–112.339|1.33|109.916–112.314|4.56|PASS|
|yolo|native-stream|7|109.239|109.307 ± 1.786|106.259–111.114|1.63|107.993–110.860|5.97|PASS|
|yolo|managed-non-stream|7|107.482|107.266 ± 1.224|105.405–109.190|1.14|106.134–107.879|7.59|PASS|
|yolo|managed-stream|7|107.339|107.166 ± 1.224|105.221–108.816|1.14|106.074–108.180|9.40|PASS|
|yolo11|raw-non-stream|7|93.517|93.388 ± 0.755|92.180–94.340|0.81|92.584–93.883|0.00|PASS|
|yolo11|raw-stream|7|93.489|93.473 ± 0.438|92.991–94.227|0.47|93.112–93.735|0.16|PASS|
|yolo11|native-non-stream|7|91.038|91.344 ± 1.126|89.904–92.851|1.23|90.473–92.755|2.98|PASS|
|yolo11|native-stream|7|89.363|88.955 ± 1.163|86.766–90.016|1.31|87.961–89.693|5.10|PASS|
|yolo11|managed-non-stream|7|87.559|87.396 ± 1.661|84.813–89.830|1.90|86.255–88.753|7.22|PASS|
|yolo11|managed-stream|7|88.350|88.366 ± 0.895|87.274–89.797|1.01|87.435–89.137|6.08|PASS|
|yolo26|raw-non-stream|7|82.950|82.706 ± 1.024|80.632–83.650|1.24|82.242–83.508|0.00|PASS|
|yolo26|raw-stream|7|82.773|82.700 ± 1.188|80.446–84.078|1.44|82.293–83.637|0.23|PASS|
|yolo26|native-non-stream|7|81.157|80.967 ± 0.641|79.737–81.601|0.79|80.510–81.450|2.36|PASS|
|yolo26|native-stream|7|78.840|78.880 ± 0.716|77.969–80.237|0.91|78.414–79.096|5.13|PASS|
|yolo26|managed-non-stream|7|77.623|78.114 ± 1.244|76.736–80.374|1.59|77.350–79.049|7.12|PASS|
|yolo26|managed-stream|7|77.721|77.460 ± 0.857|75.992–78.439|1.11|76.797–78.273|6.87|PASS|
|yolo-world|raw-non-stream|7|82.415|82.289 ± 0.793|80.638–83.189|0.96|82.187–82.644|0.00|PASS|
|yolo-world|raw-stream|7|82.520|82.514 ± 0.976|80.644–83.777|1.18|82.351–83.290|0.21|PASS|
|yolo-world|native-non-stream|7|80.915|80.434 ± 0.945|79.226–81.326|1.18|79.328–81.289|1.67|PASS|
|yolo-world|native-stream|7|78.391|78.691 ± 0.850|77.622–79.813|1.08|77.971–79.464|4.01|PASS|
|yolo-world|managed-non-stream|7|78.865|79.132 ± 0.772|78.289–80.139|0.98|78.299–79.874|4.50|PASS|
|yolo-world|managed-stream|7|78.538|78.360 ± 1.099|76.766–79.833|1.40|77.396–79.386|5.23|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.034045|1033.961|1037.618/1037.618/1037.618|1933.85|34.27/62.00|
|yolo|raw-stream|1.037675|22.073|1028.202/1028.202/1028.202|1933.82|34.27/62.00|
|yolo|native-non-stream|1.072286|1072.196|1071.343/1071.343/1071.343|1933.95|34.27/62.00|
|yolo|native-stream|1.098508|1098.435|1086.297/1086.297/1086.297|1933.79|34.25/62.00|
|yolo|managed-non-stream|1.116462|1116.375|1122.216/1122.216/1122.216|1825.00|24.66/62.00|
|yolo|managed-stream|1.117948|29.884|1130.437/1130.437/1130.437|1825.03|24.66/62.00|
|yolo11|raw-non-stream|1.283189|1283.106|1280.373/1280.373/1280.373|2011.84|64.67/90.00|
|yolo11|raw-stream|1.283576|24.192|1295.965/1295.965/1295.965|2012.01|64.67/90.00|
|yolo11|native-non-stream|1.318131|1318.045|1327.937/1327.937/1327.937|2011.82|64.67/90.00|
|yolo11|native-stream|1.342833|1342.761|1340.482/1340.482/1340.482|2011.87|64.67/90.00|
|yolo11|managed-non-stream|1.370512|1370.406|1364.043/1364.043/1364.043|1902.35|56.12/90.00|
|yolo11|managed-stream|1.358239|32.608|1392.961/1392.961/1392.961|1903.66|56.12/90.00|
|yolo26|raw-non-stream|1.446661|1446.576|1437.409/1437.409/1437.409|1946.11|62.82/90.00|
|yolo26|raw-stream|1.449740|25.890|1428.481/1428.481/1428.481|1945.88|62.82/90.00|
|yolo26|native-non-stream|1.478611|1478.510|1475.907/1475.907/1475.907|1945.89|62.82/90.00|
|yolo26|native-stream|1.522061|1521.990|1508.711/1508.711/1508.711|1945.88|62.80/90.00|
|yolo26|managed-non-stream|1.545930|1545.813|1560.563/1560.563/1560.563|1837.10|55.34/90.00|
|yolo26|managed-stream|1.543978|33.977|1608.713/1608.713/1608.713|1844.27|55.34/90.00|
|yolo-world|raw-non-stream|1.456050|1455.966|1463.350/1463.350/1463.350|2190.01|115.88/132.00|
|yolo-world|raw-stream|1.454192|26.062|1465.428/1465.428/1465.428|2190.39|115.88/132.00|
|yolo-world|native-non-stream|1.483034|1482.951|1469.849/1469.849/1469.849|2190.30|115.88/132.00|
|yolo-world|native-stream|1.530786|1530.712|1530.091/1530.091/1530.091|2185.35|115.88/132.00|
|yolo-world|managed-non-stream|1.521596|1521.507|1514.092/1514.092/1514.092|2081.45|107.97/132.00|
|yolo-world|managed-stream|1.527929|33.485|1545.491/1545.491/1545.491|2082.82|107.97/132.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
