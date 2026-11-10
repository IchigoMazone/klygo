# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## images, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|110.071|106.901 ± 12.830|78.404–115.682|12.00|108.076–113.541|0.00|PASS|
|yolo|raw-stream|7|110.394|103.495 ± 16.578|66.845–113.307|16.02|102.240–112.348|2.85|PASS|
|yolo|native-non-stream|7|82.689|82.431 ± 0.825|80.712–83.223|1.00|82.147–82.904|33.30|PASS|
|yolo|native-stream|7|82.348|82.857 ± 1.699|81.295–86.232|2.05|81.313–83.581|31.33|PASS|
|yolo|managed-non-stream|7|70.127|68.192 ± 5.730|60.093–73.866|8.40|62.642–73.203|55.27|PASS|
|yolo|managed-stream|7|55.331|53.920 ± 8.121|45.390–68.302|15.06|45.888–57.047|99.03|PASS|
|yolo11|raw-non-stream|7|71.034|74.921 ± 13.772|60.929–93.668|18.38|61.050–88.929|0.00|PASS|
|yolo11|raw-stream|7|76.062|78.469 ± 11.834|64.693–90.903|15.08|64.694–90.755|-5.82|PASS|
|yolo11|native-non-stream|7|70.474|64.558 ± 12.138|46.808–73.266|18.80|46.890–72.044|0.79|PASS|
|yolo11|native-stream|7|70.424|66.757 ± 8.304|52.202–73.484|12.44|57.791–72.355|21.02|PASS|
|yolo11|managed-non-stream|7|64.992|62.656 ± 5.904|49.316–65.328|9.42|63.942–65.301|28.95|PASS|
|yolo11|managed-stream|7|60.855|60.323 ± 0.805|59.268–61.021|1.34|59.386–61.005|16.41|PASS|
|yolo26|raw-non-stream|7|81.009|80.975 ± 1.882|77.567–83.480|2.32|80.189–82.603|0.00|PASS|
|yolo26|raw-stream|7|82.007|80.850 ± 2.500|76.397–83.578|3.09|78.607–82.261|-0.14|PASS|
|yolo26|native-non-stream|7|64.745|61.849 ± 4.793|54.786–65.771|7.75|55.377–65.294|26.92|PASS|
|yolo26|native-stream|7|61.725|58.117 ± 9.457|44.170–64.951|16.27|44.708–64.915|31.57|PASS|
|yolo26|managed-non-stream|7|50.910|50.065 ± 9.966|35.523–59.913|19.91|37.751–59.167|57.51|PASS|
|yolo26|managed-stream|7|54.317|49.207 ± 7.448|37.905–55.408|15.14|40.721–54.759|52.45|PASS|
|yolo-world|raw-non-stream|7|79.672|73.516 ± 10.715|58.594–83.112|14.58|61.491–82.438|0.00|PASS|
|yolo-world|raw-stream|7|80.822|73.163 ± 10.036|60.455–81.716|13.72|62.963–81.059|0.88|PASS|
|yolo-world|native-non-stream|7|62.349|57.177 ± 9.137|44.546–65.015|15.98|44.794–64.787|22.98|PASS|
|yolo-world|native-stream|7|61.682|57.977 ± 6.888|44.074–62.970|11.88|54.036–62.446|29.17|PASS|
|yolo-world|managed-non-stream|7|57.781|58.180 ± 1.317|56.338–60.032|2.26|57.415–59.730|36.68|PASS|
|yolo-world|managed-stream|7|53.753|53.864 ± 0.495|53.316–54.836|0.92|53.533–54.085|47.84|PASS|
|grounding-dino|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|raw-non-stream|7|2.901|2.890 ± 0.101|2.720–3.031|3.50|2.807–2.949|0.00|PASS|
|grounding-dino|native-non-stream|7|2.709|2.757 ± 0.144|2.565–2.933|5.23|2.660–2.922|3.63|PASS|
|grounding-dino|managed-non-stream|7|2.797|2.744 ± 0.124|2.576–2.907|4.53|2.625–2.838|6.81|PASS|
|grounding-dino|managed-stream|7|2.818|2.808 ± 0.084|2.695–2.904|3.00|2.731–2.885|1.71|PASS|
|grounding-dino-base|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|raw-non-stream|7|2.011|2.009 ± 0.042|1.963–2.090|2.08|1.976–2.017|0.00|PASS|
|grounding-dino-base|native-non-stream|7|1.936|1.951 ± 0.091|1.797–2.044|4.66|1.897–2.035|2.72|PASS|
|grounding-dino-base|managed-non-stream|7|1.977|1.954 ± 0.081|1.822–2.046|4.14|1.867–2.020|1.74|PASS|
|grounding-dino-base|managed-stream|7|1.987|1.960 ± 0.081|1.850–2.047|4.11|1.857–2.025|2.10|PASS|
|faster-rcnn|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|raw-non-stream|7|32.227|28.760 ± 5.302|19.252–33.162|18.43|24.630–32.508|0.00|PASS|
|faster-rcnn|native-non-stream|7|28.479|26.220 ± 5.587|17.053–30.304|21.31|19.281–30.146|7.43|PASS|
|faster-rcnn|managed-non-stream|7|29.163|27.772 ± 3.815|19.134–29.484|13.74|28.922–29.452|9.96|PASS|
|faster-rcnn|managed-stream|7|28.208|28.158 ± 0.470|27.455–28.940|1.67|27.768–28.357|11.91|PASS|
|d-fine|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|raw-non-stream|7|2.541|2.323 ± 0.323|1.917–2.613|13.92|1.951–2.593|0.00|PASS|
|d-fine|native-non-stream|7|2.398|2.327 ± 0.223|1.858–2.492|9.60|2.243–2.483|3.03|PASS|
|d-fine|managed-non-stream|7|2.441|2.344 ± 0.225|1.891–2.525|9.59|2.205–2.484|4.67|PASS|
|d-fine|managed-stream|7|2.160|2.191 ± 0.295|1.732–2.514|13.47|1.956–2.464|6.01|PASS|
|locate-anything|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|locate-anything|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|locate-anything|raw-non-stream|7|1.537|1.471 ± 0.131|1.258–1.577|8.91|1.319–1.571|0.00|PASS|
|locate-anything|native-non-stream|7|1.545|1.507 ± 0.129|1.220–1.586|8.53|1.518–1.573|-0.43|PASS|
|locate-anything|managed-non-stream|7|1.547|1.495 ± 0.138|1.187–1.574|9.21|1.516–1.566|1.17|PASS|
|locate-anything|managed-stream|7|1.517|1.452 ± 0.110|1.282–1.563|7.60|1.358–1.540|3.96|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|0.581444|13.785|10.633/15.308/17.419|1655.00|27.84/60.00|
|yolo|raw-stream|0.579740|13.363|11.043/17.887/18.608|1654.99|27.84/60.00|
|yolo|native-non-stream|0.773987|16.692|24.913/48.967/55.981|1673.98|27.81/60.00|
|yolo|native-stream|0.777194|14.966|12.014/14.531/18.394|1674.49|27.81/60.00|
|yolo|managed-non-stream|0.912630|16.601|13.760/17.229/20.443|1674.36|27.81/60.00|
|yolo|managed-stream|1.156676|22.838|13.784/17.301/23.973|1675.64|27.81/60.00|
|yolo11|raw-non-stream|0.900980|16.138|10.500/13.797/16.550|1676.71|59.92/88.00|
|yolo11|raw-stream|0.841420|15.444|10.553/14.169/17.624|1676.71|59.92/88.00|
|yolo11|native-non-stream|0.908139|19.894|13.999/17.540/24.103|1696.23|59.89/88.00|
|yolo11|native-stream|0.908783|17.397|13.776/17.939/22.054|1696.20|59.89/88.00|
|yolo11|managed-non-stream|0.984743|18.402|21.795/34.188/38.724|1696.37|59.89/88.00|
|yolo11|managed-stream|1.051686|19.421|26.144/32.479/40.097|1697.36|59.89/88.00|
|yolo26|raw-non-stream|0.790036|17.078|11.846/16.738/20.725|1701.65|59.02/88.00|
|yolo26|raw-stream|0.780425|17.820|11.918/16.159/19.712|1703.47|59.02/88.00|
|yolo26|native-non-stream|0.988488|19.496|16.161/19.893/29.176|1702.12|58.99/88.00|
|yolo26|native-stream|1.036852|19.701|15.888/21.527/34.285|1703.77|58.99/88.00|
|yolo26|managed-non-stream|1.257129|19.047|17.338/24.363/27.092|1704.02|58.99/88.00|
|yolo26|managed-stream|1.178262|20.155|18.534/23.356/26.286|1701.97|58.99/88.00|
|yolo-world|raw-non-stream|0.803292|16.996|11.591/15.530/20.570|1914.54|114.82/136.00|
|yolo-world|raw-stream|0.791860|18.176|16.760/26.863/28.471|1933.16|114.82/136.00|
|yolo-world|native-non-stream|1.026484|20.032|32.397/107.936/129.819|1934.18|114.79/136.00|
|yolo-world|native-stream|1.037582|22.246|17.588/61.145/89.825|1934.05|114.79/136.00|
|yolo-world|managed-non-stream|1.107638|18.831|16.609/22.216/24.364|1935.21|114.79/136.00|
|yolo-world|managed-stream|1.190629|20.848|19.256/24.898/28.994|1935.44|114.79/136.00|
|grounding-dino|raw-non-stream|2.757886|348.742|342.707/345.542/346.011|2822.68|1366.93/1424.00|
|grounding-dino|native-non-stream|2.953652|348.166|344.904/348.370/348.564|2822.68|1366.92/1424.00|
|grounding-dino|managed-non-stream|2.860591|355.262|380.516/398.780/400.377|2829.95|1366.92/1424.00|
|grounding-dino|managed-stream|2.838409|347.555|345.502/347.929/348.022|2831.13|1366.92/1424.00|
|grounding-dino-base|raw-non-stream|3.978027|497.166|526.874/541.756/544.278|2793.93|1600.91/1662.00|
|grounding-dino-base|native-non-stream|4.132742|510.051|502.209/514.576/515.240|2793.93|1600.90/1662.00|
|grounding-dino-base|managed-non-stream|4.047389|497.724|496.368/502.571/502.843|2802.36|1600.90/1662.00|
|grounding-dino-base|managed-stream|4.027077|508.521|526.401/537.870/539.676|2802.36|1600.90/1662.00|
|faster-rcnn|raw-non-stream|0.992942|36.195|29.129/36.482/45.648|1688.27|202.88/324.00|
|faster-rcnn|native-non-stream|1.123653|36.949|34.924/41.963/48.322|1688.27|202.85/324.00|
|faster-rcnn|managed-non-stream|1.097281|35.602|33.631/41.571/44.130|1726.86|202.85/324.00|
|faster-rcnn|managed-stream|1.134426|37.469|33.315/41.871/50.331|1726.91|202.85/324.00|
|d-fine|raw-non-stream|3.148944|399.282|384.927/401.269/402.577|2997.32|139.72/210.00|
|d-fine|native-non-stream|3.336226|387.150|393.239/409.695/409.948|2997.34|141.17/210.00|
|d-fine|managed-non-stream|3.277204|396.743|580.175/605.365/607.241|3006.21|139.55/210.00|
|d-fine|managed-stream|3.704493|402.446|419.430/677.385/743.655|3006.24|142.59/210.00|
|locate-anything|raw-non-stream|0.650468|650.323|758.379/758.379/758.379|3899.34|7910.11/8364.00|
|locate-anything|native-non-stream|0.647417|647.357|810.777/810.777/810.777|3899.34|7910.11/8364.00|
|locate-anything|managed-non-stream|0.646438|646.369|660.489/660.489/660.489|3899.34|7910.11/8364.00|
|locate-anything|managed-stream|0.659196|658.897|661.721/661.721/661.721|3899.34|7910.11/8364.00|

## images, batch 4

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|164.933|169.023 ± 11.300|157.434–189.323|6.69|159.342–177.989|0.00|PASS|
|yolo|raw-stream|7|191.977|184.791 ± 14.200|157.510–194.129|7.68|172.565–193.281|-8.31|PASS|
|yolo|native-non-stream|7|142.489|141.428 ± 2.759|136.954–144.854|1.95|139.002–143.458|20.43|PASS|
|yolo|native-stream|7|138.592|140.070 ± 3.195|136.954–145.373|2.28|137.790–143.596|20.43|PASS|
|yolo|managed-non-stream|7|118.089|114.679 ± 6.262|104.455–120.499|5.46|110.335–120.473|39.67|PASS|
|yolo|managed-stream|7|89.651|85.317 ± 13.016|69.189–98.535|15.26|71.002–97.491|77.74|PASS|
|yolo11|raw-non-stream|7|147.509|157.284 ± 20.055|136.755–191.000|12.75|143.800–177.670|0.00|PASS|
|yolo11|raw-stream|7|193.555|183.489 ± 17.619|156.761–195.636|9.60|158.893–195.294|-13.93|PASS|
|yolo11|native-non-stream|7|141.376|141.008 ± 1.498|138.539–142.634|1.06|139.658–142.374|3.80|PASS|
|yolo11|native-stream|7|140.993|139.979 ± 2.585|135.507–142.213|1.85|137.254–142.104|7.47|PASS|
|yolo11|managed-non-stream|7|115.374|114.963 ± 4.356|108.211–119.709|3.79|111.254–119.280|30.61|PASS|
|yolo11|managed-stream|7|79.359|82.780 ± 11.126|71.212–99.231|13.44|71.268–92.842|94.18|PASS|
|yolo26|raw-non-stream|7|165.079|169.407 ± 24.530|142.487–199.097|14.48|146.780–196.313|0.00|PASS|
|yolo26|raw-stream|7|193.631|180.189 ± 22.788|141.468–198.532|12.65|153.913–194.020|-4.63|PASS|
|yolo26|native-non-stream|7|137.661|135.442 ± 6.567|121.032–141.060|4.85|136.077–138.322|21.21|PASS|
|yolo26|native-stream|7|136.658|136.860 ± 3.370|132.699–141.305|2.46|132.909–140.217|21.65|PASS|
|yolo26|managed-non-stream|7|114.788|112.812 ± 6.002|100.764–118.237|5.32|110.688–118.120|47.21|PASS|
|yolo26|managed-stream|7|83.049|82.891 ± 12.708|69.406–99.615|15.33|71.899–99.443|88.08|PASS|
|yolo-world|raw-non-stream|7|92.864|94.114 ± 2.960|90.701–98.099|3.15|91.404–96.773|0.00|PASS|
|yolo-world|raw-stream|7|97.932|96.823 ± 3.233|89.516–98.407|3.34|97.645–98.249|-1.28|PASS|
|yolo-world|native-non-stream|7|86.085|85.971 ± 1.329|83.277–87.572|1.55|85.728–86.707|11.51|PASS|
|yolo-world|native-stream|7|81.561|79.701 ± 5.466|72.353–85.630|6.86|72.804–84.508|13.86|PASS|
|yolo-world|managed-non-stream|7|70.429|69.887 ± 5.266|64.175–77.828|7.53|64.827–73.958|32.95|PASS|
|yolo-world|managed-stream|7|73.273|72.325 ± 2.646|66.389–73.760|3.66|72.599–73.737|30.97|PASS|
|grounding-dino|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|raw-non-stream|7|3.051|3.039 ± 0.052|2.929–3.081|1.70|3.028–3.072|0.00|PASS|
|grounding-dino|native-non-stream|7|3.038|3.026 ± 0.050|2.939–3.089|1.65|2.991–3.059|0.55|PASS|
|grounding-dino|managed-non-stream|7|3.027|3.033 ± 0.021|3.007–3.068|0.71|3.014–3.051|0.44|PASS|
|grounding-dino|managed-stream|7|3.049|3.042 ± 0.014|3.022–3.055|0.46|3.024–3.054|0.51|PASS|
|grounding-dino-base|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|raw-non-stream|7|2.100|2.101 ± 0.019|2.083–2.139|0.89|2.088–2.107|0.00|PASS|
|grounding-dino-base|native-non-stream|7|2.098|2.102 ± 0.019|2.083–2.140|0.90|2.086–2.112|-0.35|PASS|
|grounding-dino-base|managed-non-stream|7|2.090|2.092 ± 0.015|2.073–2.119|0.71|2.081–2.099|0.72|PASS|
|grounding-dino-base|managed-stream|7|2.086|2.082 ± 0.014|2.062–2.102|0.67|2.067–2.092|0.63|PASS|
|faster-rcnn|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|raw-non-stream|7|37.594|37.030 ± 1.901|34.435–38.820|5.13|35.078–38.794|0.00|PASS|
|faster-rcnn|native-non-stream|7|36.907|35.028 ± 2.931|30.867–37.494|8.37|31.135–37.290|3.29|PASS|
|faster-rcnn|managed-non-stream|7|35.194|34.182 ± 2.965|28.380–36.824|8.67|32.259–36.183|9.16|PASS|
|faster-rcnn|managed-stream|7|35.211|34.094 ± 2.715|30.195–36.715|7.96|30.671–36.295|6.77|PASS|
|d-fine|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|raw-non-stream|7|8.163|8.321 ± 0.834|6.965–9.263|10.02|7.868–9.185|0.00|PASS|
|d-fine|native-non-stream|7|8.765|8.385 ± 0.749|7.196–8.984|8.94|7.402–8.860|1.29|PASS|
|d-fine|managed-non-stream|7|8.730|8.297 ± 0.887|6.971–9.215|10.69|7.242–9.048|-9.25|PASS|
|d-fine|managed-stream|7|8.531|8.095 ± 0.822|6.875–8.898|10.16|7.287–8.881|3.34|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.552146|26.573|24.732/37.528/47.943|2059.84|76.00/186.00|
|yolo|raw-stream|1.333491|27.500|20.479/23.174/28.772|2040.15|76.00/186.00|
|yolo|native-non-stream|1.796628|29.823|27.481/30.586/33.889|2040.16|75.87/186.00|
|yolo|native-stream|1.847151|37.558|27.042/34.844/36.726|2048.94|75.87/186.00|
|yolo|managed-non-stream|2.167858|32.430|32.557/37.621/44.927|2068.88|75.87/186.00|
|yolo|managed-stream|2.855530|45.388|56.320/216.391/251.531|2221.04|75.87/186.00|
|yolo11|raw-non-stream|1.735491|31.680|20.996/27.473/29.540|1965.61|112.51/184.00|
|yolo11|raw-stream|1.322619|23.342|20.271/22.877/26.008|1969.13|112.51/184.00|
|yolo11|native-non-stream|1.810775|28.596|27.749/31.843/40.002|2047.94|112.39/184.00|
|yolo11|native-stream|1.815692|35.748|27.655/34.698/40.926|2047.94|112.39/184.00|
|yolo11|managed-non-stream|2.218879|34.128|33.441/39.069/53.517|2070.22|112.39/184.00|
|yolo11|managed-stream|3.225840|54.280|58.052/71.154/84.861|2232.52|112.39/184.00|
|yolo26|raw-non-stream|1.550777|22.831|20.774/27.802/32.036|1924.85|111.72/184.00|
|yolo26|raw-stream|1.322101|22.642|20.146/22.716/27.031|1951.34|111.72/184.00|
|yolo26|native-non-stream|1.859646|28.060|28.617/32.475/40.257|1952.51|111.60/184.00|
|yolo26|native-stream|1.873295|30.107|29.108/36.490/38.484|1962.29|111.60/184.00|
|yolo26|managed-non-stream|2.230202|39.446|33.776/41.877/48.587|1961.65|111.60/184.00|
|yolo26|managed-stream|3.082534|43.631|57.514/74.438/81.693|2229.69|111.60/184.00|
|yolo-world|raw-non-stream|2.756723|40.471|40.784/42.876/43.511|2168.15|214.50/304.00|
|yolo-world|raw-stream|2.614055|40.180|45.520/53.408/54.086|2103.08|214.50/304.00|
|yolo-world|native-non-stream|2.973802|45.569|47.348/51.644/52.817|2192.44|214.38/304.00|
|yolo-world|native-stream|3.138750|52.616|47.618/53.309/56.133|2207.83|214.38/304.00|
|yolo-world|managed-non-stream|3.634855|51.955|51.523/54.636/55.790|2232.59|214.38/304.00|
|yolo-world|managed-stream|3.493783|63.740|65.584/82.717/84.601|2456.97|214.38/304.00|
|grounding-dino|raw-non-stream|10.489263|1299.894|1298.689/1327.029/1331.791|2887.83|3486.48/4298.00|
|grounding-dino|native-non-stream|10.533242|1300.282|1302.151/1379.120/1382.480|2887.83|3486.46/4298.00|
|grounding-dino|managed-non-stream|10.570405|1291.407|1297.753/1402.071/1414.510|2926.14|3486.46/4298.00|
|grounding-dino|managed-stream|10.495041|1290.586|1298.531/1384.469/1389.701|2888.02|3486.46/4298.00|
|grounding-dino-base|raw-non-stream|15.240616|1897.492|1885.997/1915.790/1921.543|2865.18|3731.84/4524.00|
|grounding-dino-base|native-non-stream|15.255619|1881.735|1902.734/1984.877/1987.670|2865.19|3731.82/4524.00|
|grounding-dino-base|managed-non-stream|15.310893|1891.100|1905.465/1944.976/1959.097|2903.58|3731.82/4524.00|
|grounding-dino-base|managed-stream|15.339739|1903.589|1904.513/1997.043/2011.145|2865.46|3731.82/4524.00|
|faster-rcnn|raw-non-stream|3.404776|101.818|107.122/137.761/166.680|1753.47|561.57/1006.00|
|faster-rcnn|native-non-stream|3.468150|105.566|108.522/113.416/114.983|1902.07|561.45/1006.00|
|faster-rcnn|managed-non-stream|3.636993|107.021|108.287/113.796/119.622|1908.21|561.45/1006.00|
|faster-rcnn|managed-stream|3.635273|105.807|128.734/153.464/158.059|1914.36|561.45/1006.00|
|d-fine|raw-non-stream|3.920175|457.032|459.941/471.156/471.702|3004.44|489.49/702.00|
|d-fine|native-non-stream|3.651082|441.264|457.515/472.664/475.307|3006.68|493.54/702.00|
|d-fine|managed-non-stream|3.665381|417.376|632.287/694.690/699.318|3042.00|532.97/702.00|
|d-fine|managed-stream|3.750824|446.244|457.877/464.976/465.324|3041.68|501.83/702.00|

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|82.580|77.372 ± 10.527|58.998–84.601|13.61|65.583–84.519|0.00|PASS|
|yolo|raw-stream|7|84.591|75.101 ± 16.422|51.042–86.923|21.87|51.457–85.869|-1.57|PASS|
|yolo|native-non-stream|7|79.653|76.758 ± 6.303|64.039–81.576|8.21|74.781–81.267|4.38|PASS|
|yolo|native-stream|7|74.978|71.871 ± 8.636|52.615–77.047|12.02|71.953–75.914|10.17|PASS|
|yolo|managed-non-stream|7|71.503|68.012 ± 6.222|57.988–73.574|9.15|62.577–73.300|15.49|PASS|
|yolo|managed-stream|7|70.174|64.029 ± 11.794|45.895–71.838|18.42|47.725–71.537|17.65|PASS|
|yolo11|raw-non-stream|7|72.534|70.618 ± 4.720|61.059–74.079|6.68|67.837–74.037|0.00|PASS|
|yolo11|raw-stream|7|69.017|61.326 ± 12.915|45.051–73.774|21.06|45.277–72.875|3.89|PASS|
|yolo11|native-non-stream|7|69.416|67.685 ± 3.508|61.183–71.355|5.18|65.415–70.151|3.23|PASS|
|yolo11|native-stream|7|64.611|64.707 ± 0.939|62.898–65.825|1.45|64.535–65.496|11.58|PASS|
|yolo11|managed-non-stream|7|49.563|52.881 ± 10.869|39.597–64.099|20.55|40.352–63.666|44.67|PASS|
|yolo11|managed-stream|7|62.540|60.118 ± 7.069|48.291–65.692|11.76|51.951–65.458|17.44|PASS|
|yolo26|raw-non-stream|7|66.193|59.968 ± 10.050|45.026–66.966|16.76|45.714–66.449|0.00|PASS|
|yolo26|raw-stream|7|62.796|61.235 ± 8.928|41.988–67.579|14.58|60.011–67.235|-1.96|PASS|
|yolo26|native-non-stream|7|59.616|57.192 ± 7.037|43.482–63.563|12.30|52.438–62.431|4.24|PASS|
|yolo26|native-stream|7|57.064|55.181 ± 5.145|47.297–60.011|9.32|48.436–59.151|12.34|PASS|
|yolo26|managed-non-stream|7|57.122|55.089 ± 5.644|46.984–60.426|10.24|47.345–59.722|11.26|PASS|
|yolo26|managed-stream|7|57.489|53.279 ± 6.321|46.064–59.784|11.86|46.292–58.151|11.15|PASS|
|yolo-world|raw-non-stream|7|67.879|65.050 ± 6.976|52.310–70.315|10.72|58.333–70.168|0.00|PASS|
|yolo-world|raw-stream|7|54.571|59.346 ± 10.237|46.302–70.559|17.25|50.913–69.727|0.63|PASS|
|yolo-world|native-non-stream|7|65.202|63.570 ± 4.910|53.546–68.142|7.72|61.566–67.025|4.89|PASS|
|yolo-world|native-stream|7|61.633|60.852 ± 2.525|56.970–63.916|4.15|57.815–62.375|10.13|PASS|
|yolo-world|managed-non-stream|7|60.286|55.363 ± 9.290|43.148–63.729|16.78|44.563–63.650|10.47|PASS|
|yolo-world|managed-stream|7|61.418|60.496 ± 4.745|50.003–63.856|7.84|61.153–63.410|8.97|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.453138|1453.059|1558.976/1558.976/1558.976|2250.71|34.30/62.00|
|yolo|raw-stream|1.418596|60.134|1810.571/1810.571/1810.571|2250.71|34.30/62.00|
|yolo|native-non-stream|1.506526|1506.429|2316.543/2316.543/2316.543|2250.71|34.30/62.00|
|yolo|native-stream|1.600472|1600.393|1605.681/1605.681/1605.681|2257.34|34.25/62.00|
|yolo|managed-non-stream|1.678252|1678.130|1666.015/1666.015/1666.015|2257.35|24.66/62.00|
|yolo|managed-stream|1.710028|99.838|1686.806/1686.806/1686.806|2257.35|24.66/62.00|
|yolo11|raw-non-stream|1.654407|1654.296|1736.994/1736.994/1736.994|2593.73|64.97/90.00|
|yolo11|raw-stream|1.738698|60.563|1637.640/1637.640/1637.640|2593.64|64.97/90.00|
|yolo11|native-non-stream|1.728716|1728.619|1730.839/1730.839/1730.839|2593.43|64.97/90.00|
|yolo11|native-stream|1.857272|1857.199|1909.334/1909.334/1909.334|2592.64|64.94/90.00|
|yolo11|managed-non-stream|2.421159|2421.047|2906.251/2906.251/2906.251|2334.95|56.35/90.00|
|yolo11|managed-stream|1.918765|92.382|1859.907/1859.907/1859.907|2281.12|56.35/90.00|
|yolo26|raw-non-stream|1.812872|1812.792|1855.957/1855.957/1855.957|2264.02|62.86/90.00|
|yolo26|raw-stream|1.910947|57.192|1770.256/1770.256/1770.256|2264.02|62.86/90.00|
|yolo26|native-non-stream|2.012892|2012.799|2862.817/2862.817/2862.817|2264.02|62.86/90.00|
|yolo26|native-stream|2.102890|2102.811|1982.121/1982.121/1982.121|2264.07|62.80/90.00|
|yolo26|managed-non-stream|2.100763|2100.671|1998.726/1998.726/1998.726|1958.93|55.57/90.00|
|yolo26|managed-stream|2.087339|93.616|2046.081/2046.081/2046.081|1957.06|55.57/90.00|
|yolo-world|raw-non-stream|1.767864|1767.782|2682.706/2682.706/2682.706|2797.22|115.19/140.00|
|yolo-world|raw-stream|2.198964|59.624|1710.745/1710.745/1710.745|2789.97|115.19/140.00|
|yolo-world|native-non-stream|1.840435|1840.320|1782.272/1782.272/1782.272|2790.11|115.19/140.00|
|yolo-world|native-stream|1.947023|1946.947|1939.288/1939.288/1939.288|2790.11|115.13/140.00|
|yolo-world|managed-non-stream|1.990499|1990.350|1908.352/1908.352/1908.352|2483.15|107.97/140.00|
|yolo-world|managed-stream|1.953820|89.889|2087.474/2087.474/2087.474|2483.14|107.97/140.00|

## video, batch 4

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo|raw-non-stream|7|148.730|134.202 ± 28.624|81.655–158.476|21.33|107.247–153.327|0.00|PASS|
|yolo|raw-stream|7|153.869|141.208 ± 29.882|75.716–159.141|21.16|136.259–157.935|1.92|PASS|
|yolo|managed-non-stream|7|130.546|122.736 ± 18.453|88.267–138.416|15.03|106.306–135.904|15.76|PASS|
|yolo|managed-stream|7|124.021|118.866 ± 23.162|73.479–138.190|19.49|103.581–136.304|14.30|PASS|
|yolo11|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo11|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo11|raw-non-stream|7|133.064|123.255 ± 24.898|74.873–143.815|20.20|104.949–140.943|0.00|PASS|
|yolo11|raw-stream|7|134.434|134.311 ± 8.175|118.550–145.584|6.09|132.431–138.606|-1.02|PASS|
|yolo11|managed-non-stream|7|121.209|115.376 ± 18.155|74.850–127.778|15.74|117.432–123.924|10.63|PASS|
|yolo11|managed-stream|7|115.323|112.776 ± 9.708|91.717–120.898|8.61|112.257–118.013|12.75|PASS|
|yolo26|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo26|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo26|raw-non-stream|7|118.369|117.301 ± 20.326|76.694–136.077|17.33|111.987–135.165|0.00|PASS|
|yolo26|raw-stream|7|130.864|120.106 ± 21.324|77.221–134.957|17.75|105.803–132.532|-1.95|PASS|
|yolo26|managed-non-stream|7|115.369|111.588 ± 8.207|94.918–117.911|7.35|106.910–116.899|8.82|PASS|
|yolo26|managed-stream|7|109.126|102.959 ± 17.109|67.053–117.087|16.62|96.346–113.564|12.98|PASS|
|yolo-world|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo-world|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo-world|raw-non-stream|7|97.061|94.235 ± 13.638|74.938–109.662|14.47|77.199–108.278|0.00|PASS|
|yolo-world|raw-stream|7|105.720|99.872 ± 12.114|74.516–109.846|12.13|95.967–106.778|-8.19|PASS|
|yolo-world|managed-non-stream|7|100.466|99.542 ± 4.342|90.022–103.425|4.36|100.173–101.371|-3.05|PASS|
|yolo-world|managed-stream|7|98.759|99.031 ± 1.824|95.897–101.686|1.84|98.347–100.529|-2.15|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|0.806832|806.748|814.787/814.787/814.787|2235.28|90.50/146.00|
|yolo|raw-stream|0.779886|82.568|765.166/765.166/765.166|2120.26|90.50/146.00|
|yolo|managed-non-stream|0.919219|919.068|888.551/888.551/888.551|2118.34|63.09/146.00|
|yolo|managed-stream|0.967578|116.534|882.993/882.993/882.993|2089.29|63.09/146.00|
|yolo11|raw-non-stream|0.901824|901.730|1440.881/1440.881/1440.881|2560.11|125.73/172.00|
|yolo11|raw-stream|0.892631|86.840|1467.173/1467.173/1467.173|2553.27|125.73/172.00|
|yolo11|managed-non-stream|0.990028|989.925|987.949/987.949/987.949|2542.73|99.31/172.00|
|yolo11|managed-stream|1.040553|120.648|992.505/992.505/992.505|2541.45|99.31/172.00|
|yolo26|raw-non-stream|1.013780|1013.677|979.635/979.635/979.635|2550.49|119.89/162.00|
|yolo26|raw-stream|0.916980|86.507|887.350/887.350/887.350|2527.39|119.89/162.00|
|yolo26|managed-non-stream|1.040137|1040.033|995.969/995.969/995.969|2529.98|98.52/162.00|
|yolo26|managed-stream|1.099644|124.633|1003.150/1003.150/1003.150|2528.30|98.52/162.00|
|yolo-world|raw-non-stream|1.236335|1236.224|1139.003/1139.003/1139.003|2645.49|209.49/292.00|
|yolo-world|raw-stream|1.135076|88.749|1125.128/1125.128/1125.128|2645.49|209.49/292.00|
|yolo-world|managed-non-stream|1.194438|1194.334|1196.295/1196.295/1196.295|2645.49|188.26/292.00|
|yolo-world|managed-stream|1.215074|121.368|1191.065/1191.065/1191.065|2541.38|188.26/292.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
