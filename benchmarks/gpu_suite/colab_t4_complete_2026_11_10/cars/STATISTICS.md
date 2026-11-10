# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|96.400|89.305 ± 14.578|63.136–102.714|16.32|81.243–102.269|0.00|PASS|
|yolo|raw-stream|7|101.817|96.984 ± 13.689|66.165–104.136|14.11|99.879–103.892|-2.24|PASS|
|yolo|native-non-stream|7|97.229|96.011 ± 3.372|90.716–99.767|3.51|92.131–98.803|1.61|PASS|
|yolo|native-stream|7|92.834|84.688 ± 14.792|58.763–95.360|17.47|68.321–94.073|8.38|PASS|
|yolo|managed-non-stream|7|92.755|85.701 ± 12.433|65.833–96.113|14.51|70.221–93.850|4.05|PASS|
|yolo|managed-stream|7|89.864|87.736 ± 9.031|67.958–93.917|10.29|87.687–93.881|7.27|PASS|
|yolo11|raw-non-stream|7|78.376|76.279 ± 9.736|54.852–83.420|12.76|77.207–82.083|0.00|PASS|
|yolo11|raw-stream|7|79.625|74.484 ± 12.377|54.470–84.690|16.62|59.248–84.493|-0.50|PASS|
|yolo11|native-non-stream|7|79.957|77.322 ± 5.558|66.592–81.249|7.19|72.753–81.057|2.19|PASS|
|yolo11|native-stream|7|74.618|68.397 ± 12.596|49.720–78.165|18.42|50.389–76.160|7.95|PASS|
|yolo11|managed-non-stream|7|74.912|75.018 ± 1.569|72.877–77.347|2.09|73.784–76.427|4.18|PASS|
|yolo11|managed-stream|7|74.669|71.114 ± 10.446|48.435–78.628|14.69|69.222–76.769|2.44|PASS|
|yolo26|raw-non-stream|7|72.719|67.078 ± 8.972|55.120–74.795|13.37|55.696–74.497|0.00|PASS|
|yolo26|raw-stream|7|75.199|69.498 ± 10.255|51.330–76.221|14.76|58.264–75.821|-1.77|PASS|
|yolo26|native-non-stream|7|71.815|70.959 ± 2.247|66.430–73.055|3.17|69.701–72.277|2.20|PASS|
|yolo26|native-stream|7|68.389|68.476 ± 1.206|66.337–70.118|1.76|68.149–69.597|5.35|PASS|
|yolo26|managed-non-stream|7|67.310|64.869 ± 6.397|50.766–69.089|9.86|65.115–68.392|8.93|PASS|
|yolo26|managed-stream|7|57.354|58.885 ± 9.355|43.267–69.210|15.89|52.339–68.990|13.93|PASS|
|yolo-world|raw-non-stream|7|78.093|71.700 ± 11.005|52.173–78.927|15.35|59.756–78.518|0.00|PASS|
|yolo-world|raw-stream|7|77.441|74.862 ± 5.117|66.687–79.919|6.84|70.338–78.881|0.06|PASS|
|yolo-world|native-non-stream|7|75.986|71.714 ± 10.142|51.793–79.170|14.14|63.988–77.908|3.33|PASS|
|yolo-world|native-stream|7|72.805|69.106 ± 8.361|50.792–74.319|12.10|68.291–74.003|7.26|PASS|
|yolo-world|managed-non-stream|7|72.802|69.881 ± 8.966|49.840–75.997|12.83|70.885–73.693|7.26|PASS|
|yolo-world|managed-stream|7|71.142|67.053 ± 8.147|53.371–72.783|12.15|57.169–72.346|9.10|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.244809|1244.732|1170.079/1170.079/1170.079|1722.37|34.27/62.00|
|yolo|raw-stream|1.178583|24.001|1201.614/1201.614/1201.614|1722.31|34.27/62.00|
|yolo|native-non-stream|1.234205|1234.112|1299.327/1299.327/1299.327|1722.32|34.27/62.00|
|yolo|native-stream|1.292624|1292.545|1964.878/1964.878/1964.878|1722.16|34.25/62.00|
|yolo|managed-non-stream|1.293728|1293.579|1481.089/1481.089/1481.089|1612.34|24.66/62.00|
|yolo|managed-stream|1.335354|31.623|1330.145/1330.145/1330.145|1612.34|24.66/62.00|
|yolo11|raw-non-stream|1.531079|1530.993|1489.451/1489.451/1489.451|1748.62|64.94/90.00|
|yolo11|raw-stream|1.507055|26.966|2316.648/2316.648/2316.648|1748.62|64.94/90.00|
|yolo11|native-non-stream|1.500808|1500.685|1563.352/1563.352/1563.352|1748.64|64.94/90.00|
|yolo11|native-stream|1.608190|1608.111|1570.348/1570.348/1570.348|1748.62|64.94/90.00|
|yolo11|managed-non-stream|1.601876|1601.773|1573.588/1573.588/1573.588|1696.46|56.35/90.00|
|yolo11|managed-stream|1.607099|32.614|1588.770/1588.770/1588.770|1696.49|56.35/90.00|
|yolo26|raw-non-stream|1.650190|1650.100|1689.495/1689.495/1689.495|1798.97|62.91/90.00|
|yolo26|raw-stream|1.595760|28.152|2158.410/2158.410/2158.410|1798.05|62.91/90.00|
|yolo26|native-non-stream|1.670952|1670.840|1999.583/1999.583/1999.583|1798.04|62.91/90.00|
|yolo26|native-stream|1.754660|1754.569|1847.054/1847.054/1847.054|1798.02|62.89/90.00|
|yolo26|managed-non-stream|1.782787|1782.683|1739.540/1739.540/1739.540|1688.62|55.57/90.00|
|yolo26|managed-stream|2.092253|36.262|1781.781/1781.781/1781.781|1688.69|55.57/90.00|
|yolo-world|raw-non-stream|1.536630|1536.544|1757.601/1757.601/1757.601|2000.77|115.15/140.00|
|yolo-world|raw-stream|1.549569|26.357|2315.754/2315.754/2315.754|1953.67|115.15/140.00|
|yolo-world|native-non-stream|1.579238|1579.159|1582.225/1582.225/1582.225|1953.64|115.15/140.00|
|yolo-world|native-stream|1.648228|1648.159|1622.988/1622.988/1622.988|1953.65|115.13/140.00|
|yolo-world|managed-non-stream|1.648297|1648.150|1616.974/1616.974/1616.974|1891.96|107.97/140.00|
|yolo-world|managed-stream|1.686770|33.315|1626.114/1626.114/1626.114|1843.21|107.97/140.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
