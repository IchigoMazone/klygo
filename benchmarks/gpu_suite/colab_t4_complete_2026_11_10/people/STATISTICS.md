# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|97.213|97.242 ± 4.128|89.528–101.150|4.24|95.277–100.766|0.00|PASS|
|yolo|raw-stream|7|99.407|93.630 ± 14.489|61.318–103.407|15.47|95.215–99.937|0.52|PASS|
|yolo|native-non-stream|7|91.998|83.994 ± 14.585|62.451–95.066|17.36|63.223–94.212|7.17|PASS|
|yolo|native-stream|7|88.691|86.073 ± 7.224|71.544–92.569|8.39|82.063–91.118|13.61|PASS|
|yolo|managed-non-stream|7|89.437|87.208 ± 6.939|72.132–92.600|7.96|86.166–90.817|10.57|PASS|
|yolo|managed-stream|7|86.337|78.445 ± 13.681|58.853–90.412|17.44|60.150–89.144|13.04|PASS|
|yolo11|raw-non-stream|7|81.883|75.587 ± 11.801|56.804–83.700|15.61|59.974–82.598|0.00|PASS|
|yolo11|raw-stream|7|80.098|77.662 ± 6.478|65.429–83.526|8.34|72.829–82.763|-1.06|PASS|
|yolo11|native-non-stream|7|77.529|70.579 ± 13.258|49.696–79.756|18.78|52.884–79.550|3.35|PASS|
|yolo11|native-stream|7|72.725|73.384 ± 2.469|69.486–76.595|3.36|72.204–76.057|9.95|PASS|
|yolo11|managed-non-stream|7|73.865|69.818 ± 10.412|46.478–76.111|14.91|71.433–74.780|10.22|PASS|
|yolo11|managed-stream|7|72.779|68.800 ± 8.783|51.678–76.054|12.77|62.418–74.253|10.22|PASS|
|yolo26|raw-non-stream|7|58.147|62.596 ± 11.502|48.103–75.751|18.37|52.241–74.530|0.00|PASS|
|yolo26|raw-stream|7|75.211|72.948 ± 6.042|59.418–76.745|8.28|73.662–75.783|-4.80|PASS|
|yolo26|native-non-stream|7|73.536|73.266 ± 1.910|69.614–75.143|2.61|72.106–74.762|-18.75|PASS|
|yolo26|native-stream|7|68.688|68.735 ± 1.103|67.747–70.948|1.60|67.889–69.074|-15.35|PASS|
|yolo26|managed-non-stream|7|68.247|66.928 ± 2.992|60.851–69.496|4.47|65.720–69.078|-11.52|PASS|
|yolo26|managed-stream|7|53.495|57.199 ± 10.843|42.497–69.629|18.96|46.944–67.298|8.70|PASS|
|yolo-world|raw-non-stream|7|76.476|76.411 ± 1.433|74.674–78.227|1.88|75.092–78.153|0.00|PASS|
|yolo-world|raw-stream|7|77.430|69.838 ± 13.535|48.455–79.048|19.38|51.793–78.415|-0.33|PASS|
|yolo-world|native-non-stream|7|70.812|67.787 ± 9.962|51.822–78.228|14.70|56.376–74.661|8.00|PASS|
|yolo-world|native-stream|7|61.238|62.525 ± 8.442|52.204–71.913|13.50|54.765–71.879|27.74|PASS|
|yolo-world|managed-non-stream|7|71.478|64.826 ± 9.256|52.366–73.175|14.28|55.976–72.068|8.81|PASS|
|yolo-world|managed-stream|7|70.908|69.264 ± 5.831|56.225–72.924|8.42|70.180–72.476|7.66|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.234405|1234.312|1334.131/1334.131/1334.131|1892.94|34.28/62.00|
|yolo|raw-stream|1.207154|25.088|1202.627/1202.627/1202.627|1892.94|34.28/62.00|
|yolo|native-non-stream|1.304378|1304.280|1299.006/1299.006/1299.006|1892.97|34.28/62.00|
|yolo|native-stream|1.353014|1352.926|1322.474/1322.474/1322.474|1892.94|34.25/62.00|
|yolo|managed-non-stream|1.341723|1341.583|2028.055/2028.055/2028.055|1782.93|24.66/62.00|
|yolo|managed-stream|1.389906|32.948|1626.221/1626.221/1626.221|1724.30|24.66/62.00|
|yolo11|raw-non-stream|1.465501|1465.426|1490.051/1490.051/1490.051|1881.50|64.96/90.00|
|yolo11|raw-stream|1.498163|27.830|2208.666/2208.666/2208.666|1873.08|64.96/90.00|
|yolo11|native-non-stream|1.547817|1547.701|1712.163/1712.163/1712.163|1872.93|64.96/90.00|
|yolo11|native-stream|1.650052|1649.983|1658.392/1658.392/1658.392|1868.18|64.94/90.00|
|yolo11|managed-non-stream|1.624575|1624.470|1653.701/1653.701/1653.701|1762.11|56.35/90.00|
|yolo11|managed-stream|1.648823|35.490|1642.095/1642.095/1642.095|1762.00|56.35/90.00|
|yolo26|raw-non-stream|2.063733|2063.615|1865.126/1865.126/1865.126|1818.71|62.83/90.00|
|yolo26|raw-stream|1.595517|28.210|2278.743/2278.743/2278.743|1818.71|62.83/90.00|
|yolo26|native-non-stream|1.631850|1631.750|1666.840/1666.840/1666.840|1818.70|62.83/90.00|
|yolo26|native-stream|1.747035|1746.945|1712.767/1712.767/1712.767|1816.73|62.80/90.00|
|yolo26|managed-non-stream|1.758327|1758.215|1663.400/1663.400/1663.400|1764.91|55.57/90.00|
|yolo26|managed-stream|2.243206|36.506|1701.282/1701.282/1701.282|1764.90|55.57/90.00|
|yolo-world|raw-non-stream|1.569127|1569.049|1591.228/1591.228/1591.228|2038.86|115.16/140.00|
|yolo-world|raw-stream|1.549797|27.141|1534.537/1534.537/1534.537|1986.65|115.16/140.00|
|yolo-world|native-non-stream|1.694622|1694.546|1587.813/1587.813/1587.813|1986.54|115.16/140.00|
|yolo-world|native-stream|1.959567|1959.501|1700.514/1700.514/1700.514|1983.24|115.13/140.00|
|yolo-world|managed-non-stream|1.678831|1678.723|1679.294/1679.294/1679.294|1933.24|107.97/140.00|
|yolo-world|managed-stream|1.692342|34.402|2604.558/2604.558/2604.558|1932.18|107.97/140.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
