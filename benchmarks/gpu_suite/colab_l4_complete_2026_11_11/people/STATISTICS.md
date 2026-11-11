# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|113.500|113.466 ± 1.611|111.410–115.807|1.42|112.151–115.255|0.00|PASS|
|yolo|raw-stream|7|113.049|112.683 ± 1.623|110.003–114.849|1.44|111.305–113.951|0.76|PASS|
|yolo|native-non-stream|7|108.150|108.193 ± 1.441|106.422–110.195|1.33|106.534–109.776|3.70|PASS|
|yolo|native-stream|7|106.127|105.768 ± 1.248|103.556–106.891|1.18|104.803–106.859|6.39|PASS|
|yolo|managed-non-stream|7|104.107|104.371 ± 1.522|102.478–107.389|1.46|103.558–104.667|8.61|PASS|
|yolo|managed-stream|7|103.562|103.437 ± 0.923|101.459–104.215|0.89|103.477–104.166|9.02|PASS|
|yolo11|raw-non-stream|7|90.056|89.761 ± 0.802|88.733–90.730|0.89|88.923–90.492|0.00|PASS|
|yolo11|raw-stream|7|91.619|91.404 ± 0.976|89.640–92.419|1.07|90.665–92.233|-1.64|PASS|
|yolo11|native-non-stream|7|88.158|88.259 ± 0.409|87.748–88.895|0.46|87.958–88.646|2.35|PASS|
|yolo11|native-stream|7|87.176|86.644 ± 0.953|84.902–87.386|1.10|85.935–87.376|3.89|PASS|
|yolo11|managed-non-stream|7|85.811|85.463 ± 0.812|83.938–86.099|0.95|84.743–86.013|5.45|PASS|
|yolo11|managed-stream|7|85.144|85.076 ± 0.636|84.077–86.091|0.75|84.685–85.477|5.54|PASS|
|yolo26|raw-non-stream|7|82.704|82.737 ± 0.912|80.905–83.689|1.10|82.612–83.428|0.00|PASS|
|yolo26|raw-stream|7|83.614|83.521 ± 0.350|83.024–83.917|0.42|83.075–83.770|-0.55|PASS|
|yolo26|native-non-stream|7|80.591|80.351 ± 1.381|77.683–81.548|1.72|79.473–81.414|2.62|PASS|
|yolo26|native-stream|7|79.535|79.418 ± 0.753|78.262–80.368|0.95|78.570–79.934|4.00|PASS|
|yolo26|managed-non-stream|7|78.819|78.749 ± 0.864|77.240–79.772|1.10|78.254–79.667|4.82|PASS|
|yolo26|managed-stream|7|78.919|78.869 ± 0.717|78.085–80.248|0.91|78.281–79.062|4.80|PASS|
|yolo-world|raw-non-stream|7|82.103|82.394 ± 0.946|81.000–83.828|1.15|81.856–83.099|0.00|PASS|
|yolo-world|raw-stream|7|82.219|82.213 ± 1.226|80.180–84.054|1.49|81.328–83.008|-0.35|PASS|
|yolo-world|native-non-stream|7|80.597|80.424 ± 0.744|79.602–81.429|0.93|79.637–80.990|1.52|PASS|
|yolo-world|native-stream|7|77.826|77.760 ± 0.527|77.124–78.671|0.68|77.148–77.930|5.44|PASS|
|yolo-world|managed-non-stream|7|78.639|78.407 ± 0.953|77.162–79.918|1.22|77.549–78.966|5.03|PASS|
|yolo-world|managed-stream|7|78.491|78.145 ± 0.989|76.916–79.536|1.27|76.986–78.772|5.40|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.057267|1057.184|1069.686/1069.686/1069.686|1926.74|34.28/62.00|
|yolo|raw-stream|1.061487|23.079|1086.448/1086.448/1086.448|1933.03|34.28/62.00|
|yolo|native-non-stream|1.109572|1109.486|1118.103/1118.103/1118.103|1933.04|34.28/62.00|
|yolo|native-stream|1.130718|1130.646|1131.913/1131.913/1131.913|1926.73|34.25/62.00|
|yolo|managed-non-stream|1.152664|1152.557|1141.151/1141.151/1141.151|1825.30|24.66/62.00|
|yolo|managed-stream|1.158725|32.910|1183.289/1183.289/1183.289|1825.30|24.66/62.00|
|yolo11|raw-non-stream|1.332504|1332.414|1321.055/1321.055/1321.055|2007.48|64.67/90.00|
|yolo11|raw-stream|1.309766|25.277|1334.931/1334.931/1334.931|2008.00|64.67/90.00|
|yolo11|native-non-stream|1.361189|1361.099|1372.639/1372.639/1372.639|2009.98|64.67/90.00|
|yolo11|native-stream|1.376533|1376.463|1369.476/1369.476/1369.476|2008.00|64.67/90.00|
|yolo11|managed-non-stream|1.398414|1398.307|1398.800/1398.800/1398.800|1899.72|56.12/90.00|
|yolo11|managed-stream|1.409376|34.815|1422.868/1422.868/1422.868|1904.51|56.12/90.00|
|yolo26|raw-non-stream|1.450953|1450.867|1425.250/1425.250/1425.250|1941.13|62.83/90.00|
|yolo26|raw-stream|1.435162|26.460|1446.670/1446.670/1446.670|1943.52|62.83/90.00|
|yolo26|native-non-stream|1.488992|1488.906|1489.062/1489.062/1489.062|1943.44|62.83/90.00|
|yolo26|native-stream|1.508779|1508.707|1492.889/1492.889/1492.889|1941.13|62.80/90.00|
|yolo26|managed-non-stream|1.522485|1522.381|1540.190/1540.190/1540.190|1834.45|55.34/90.00|
|yolo26|managed-stream|1.520541|36.129|1559.183/1559.183/1559.183|1840.04|55.34/90.00|
|yolo-world|raw-non-stream|1.461578|1461.493|1442.908/1442.908/1442.908|2191.18|115.88/134.00|
|yolo-world|raw-stream|1.459512|26.265|1440.704/1440.704/1440.704|2191.18|115.88/134.00|
|yolo-world|native-non-stream|1.488886|1488.795|1483.937/1483.937/1483.937|2191.19|115.88/134.00|
|yolo-world|native-stream|1.541906|1541.835|1524.372/1524.372/1524.372|2191.18|115.88/134.00|
|yolo-world|managed-non-stream|1.525962|1525.869|1516.997/1516.997/1516.997|2083.07|107.97/134.00|
|yolo-world|managed-stream|1.528836|35.532|1559.524/1559.524/1559.524|2090.27|107.97/134.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
