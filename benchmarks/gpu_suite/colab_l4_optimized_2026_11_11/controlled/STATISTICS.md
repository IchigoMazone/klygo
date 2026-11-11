# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## images, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|99.842|99.961 ± 0.714|99.192–101.181|0.71|99.209–100.384|0.00|PASS|
|yolo|raw-stream|7|99.916|99.598 ± 0.814|98.043–100.488|0.82|99.060–100.113|-0.07|PASS|
|yolo|native-non-stream|7|89.990|90.183 ± 0.825|89.152–91.456|0.92|89.470–90.810|10.70|PASS|
|yolo|native-stream|7|90.972|90.430 ± 1.479|88.191–91.925|1.64|88.534–91.461|9.76|PASS|
|yolo|managed-non-stream|7|95.326|95.464 ± 0.742|94.087–96.255|0.78|95.208–96.108|4.07|PASS|
|yolo|managed-stream|7|95.818|95.667 ± 1.652|92.542–98.037|1.73|95.087–96.232|4.31|PASS|
|yolo11|raw-non-stream|7|87.292|87.167 ± 0.819|85.436–87.815|0.94|86.975–87.689|0.00|PASS|
|yolo11|raw-stream|7|88.051|87.780 ± 0.711|86.658–88.710|0.81|87.072–88.253|-0.89|PASS|
|yolo11|native-non-stream|7|79.836|79.692 ± 1.570|76.364–81.109|1.97|79.552–80.754|8.17|PASS|
|yolo11|native-stream|7|80.608|80.400 ± 1.135|78.267–81.334|1.41|79.598–81.311|8.58|PASS|
|yolo11|managed-non-stream|7|84.014|83.789 ± 0.810|82.410–84.703|0.97|82.960–84.372|3.93|PASS|
|yolo11|managed-stream|7|84.188|84.205 ± 0.317|83.668–84.672|0.38|84.035–84.463|3.88|PASS|
|yolo26|raw-non-stream|7|78.626|78.759 ± 0.523|77.911–79.578|0.66|78.556–79.130|0.00|PASS|
|yolo26|raw-stream|7|78.757|78.527 ± 0.967|76.719–79.673|1.23|78.084–79.296|0.36|PASS|
|yolo26|native-non-stream|7|72.629|72.341 ± 1.283|69.974–73.510|1.77|71.641–73.389|8.26|PASS|
|yolo26|native-stream|7|72.633|72.581 ± 1.387|70.170–74.181|1.91|71.647–73.628|7.50|PASS|
|yolo26|managed-non-stream|7|76.283|76.219 ± 0.696|74.713–76.722|0.91|76.242–76.657|3.14|PASS|
|yolo26|managed-stream|7|76.514|76.605 ± 0.482|75.901–77.193|0.63|76.206–77.107|3.14|PASS|
|yolo-world|raw-non-stream|7|78.804|78.802 ± 0.420|77.975–79.191|0.53|78.637–79.118|0.00|PASS|
|yolo-world|raw-stream|7|78.846|78.952 ± 0.729|78.012–80.107|0.92|78.505–79.740|-0.12|PASS|
|yolo-world|native-non-stream|7|73.986|73.432 ± 1.272|71.625–74.911|1.73|71.822–74.377|6.94|PASS|
|yolo-world|native-stream|7|73.911|73.740 ± 0.817|72.409–74.693|1.11|73.006–74.586|7.04|PASS|
|yolo-world|managed-non-stream|7|77.548|77.403 ± 0.604|76.776–78.405|0.78|76.808–77.737|2.02|PASS|
|yolo-world|managed-stream|7|77.599|77.497 ± 0.604|76.676–78.127|0.78|76.925–78.080|1.36|PASS|
|grounding-dino|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|raw-non-stream|7|4.894|4.900 ± 0.019|4.878–4.927|0.38|4.888–4.925|0.00|PASS|
|grounding-dino|native-non-stream|7|4.845|4.847 ± 0.013|4.828–4.871|0.27|4.840–4.854|1.17|PASS|
|grounding-dino|managed-non-stream|7|4.848|4.845 ± 0.012|4.829–4.860|0.25|4.832–4.859|1.27|PASS|
|grounding-dino|managed-stream|7|4.850|4.848 ± 0.017|4.817–4.875|0.36|4.840–4.855|0.81|PASS|
|grounding-dino-base|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|raw-non-stream|7|3.727|3.728 ± 0.018|3.704–3.754|0.49|3.708–3.744|0.00|PASS|
|grounding-dino-base|native-non-stream|7|3.697|3.701 ± 0.015|3.685–3.729|0.41|3.688–3.713|0.66|PASS|
|grounding-dino-base|managed-non-stream|7|3.697|3.691 ± 0.016|3.669–3.707|0.42|3.670–3.703|0.92|PASS|
|grounding-dino-base|managed-stream|7|3.683|3.678 ± 0.014|3.655–3.693|0.39|3.666–3.691|1.35|PASS|
|faster-rcnn|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|raw-non-stream|7|44.870|44.827 ± 0.549|44.175–45.864|1.22|44.368–45.040|0.00|PASS|
|faster-rcnn|native-non-stream|7|42.656|42.641 ± 0.480|41.814–43.152|1.13|42.292–43.059|5.53|PASS|
|faster-rcnn|managed-non-stream|7|41.005|40.983 ± 0.209|40.595–41.204|0.51|40.907–41.187|9.29|PASS|
|faster-rcnn|managed-stream|7|41.217|41.108 ± 0.379|40.343–41.482|0.92|41.004–41.402|8.79|PASS|
|d-fine|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|raw-non-stream|7|2.664|2.659 ± 0.014|2.640–2.676|0.53|2.645–2.674|0.00|PASS|
|d-fine|native-non-stream|7|2.643|2.642 ± 0.012|2.625–2.663|0.46|2.631–2.649|0.71|PASS|
|d-fine|managed-non-stream|7|2.646|2.643 ± 0.012|2.629–2.661|0.45|2.631–2.651|0.59|PASS|
|d-fine|managed-stream|7|2.643|2.640 ± 0.011|2.623–2.651|0.42|2.626–2.648|0.65|PASS|
|locate-anything|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|locate-anything|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|locate-anything|raw-non-stream|7|2.474|2.472 ± 0.009|2.461–2.485|0.35|2.461–2.478|0.00|PASS|
|locate-anything|native-non-stream|7|2.458|2.456 ± 0.014|2.433–2.471|0.56|2.445–2.470|0.67|PASS|
|locate-anything|managed-non-stream|7|2.441|2.435 ± 0.035|2.376–2.470|1.42|2.403–2.469|1.36|PASS|
|locate-anything|managed-stream|7|2.460|2.447 ± 0.030|2.401–2.476|1.24|2.409–2.469|0.57|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|0.641013|14.230|9.680/9.865/10.529|1829.99|29.74/60.00|
|yolo|raw-stream|0.640540|14.157|9.694/9.976/12.211|1829.99|29.74/60.00|
|yolo|native-non-stream|0.711187|16.006|10.849/11.856/14.295|1829.99|29.71/60.00|
|yolo|native-stream|0.703513|12.913|10.816/11.321/15.531|1776.65|29.71/60.00|
|yolo|managed-non-stream|0.671382|12.222|10.331/12.539/12.783|1775.52|29.71/60.00|
|yolo|managed-stream|0.667934|12.152|10.195/10.682/11.330|1775.59|29.71/60.00|
|yolo11|raw-non-stream|0.733174|15.688|11.179/11.553/12.400|1910.02|59.81/88.00|
|yolo11|raw-stream|0.726850|15.518|11.189/12.502/14.654|1910.02|59.81/88.00|
|yolo11|native-non-stream|0.801647|17.395|12.242/14.779/15.911|1910.02|59.77/88.00|
|yolo11|native-stream|0.793964|14.466|12.225/13.046/17.235|1856.95|59.77/88.00|
|yolo11|managed-non-stream|0.761775|13.786|11.674/12.166/13.106|1856.11|59.77/88.00|
|yolo11|managed-stream|0.760199|13.696|11.827/12.434/12.986|1856.19|59.77/88.00|
|yolo26|raw-non-stream|0.813977|17.006|12.407/15.202/15.438|1842.52|59.02/88.00|
|yolo26|raw-stream|0.812624|16.793|12.444/15.285/16.019|1842.52|59.02/88.00|
|yolo26|native-non-stream|0.881188|18.520|13.599/16.589/17.466|1842.52|58.99/88.00|
|yolo26|native-stream|0.881148|15.864|13.467/13.957/18.430|1789.44|58.99/88.00|
|yolo26|managed-non-stream|0.838982|14.859|13.034/15.869/16.264|1788.61|58.99/88.00|
|yolo26|managed-stream|0.836444|14.854|13.181/14.109/15.612|1788.66|58.99/88.00|
|yolo-world|raw-non-stream|0.812142|16.948|12.500/15.531/15.653|2098.16|115.04/132.00|
|yolo-world|raw-stream|0.811713|16.905|12.508/14.592/15.897|2098.16|115.04/132.00|
|yolo-world|native-non-stream|0.865029|18.381|13.307/14.682/16.351|2098.16|115.01/132.00|
|yolo-world|native-stream|0.865905|15.511|13.492/14.816/15.380|2045.11|115.01/132.00|
|yolo-world|managed-non-stream|0.825299|14.584|12.740/13.043/13.974|2044.27|115.01/132.00|
|yolo-world|managed-stream|0.824755|14.684|12.751/13.031/13.764|2044.32|115.01/132.00|
|grounding-dino|raw-non-stream|1.634756|203.644|203.745/206.171/206.820|2896.80|1366.93/1424.00|
|grounding-dino|native-non-stream|1.651321|206.025|206.113/207.104/207.185|2896.80|1366.92/1424.00|
|grounding-dino|managed-non-stream|1.650330|206.042|206.912/209.310/210.102|2902.27|1366.92/1424.00|
|grounding-dino|managed-stream|1.649626|203.537|206.624/209.917/210.598|2903.02|1366.92/1424.00|
|grounding-dino-base|raw-non-stream|2.146404|266.357|266.810/270.025/270.805|2909.54|1600.91/1662.00|
|grounding-dino-base|native-non-stream|2.163734|269.967|269.588/270.402/270.502|2909.54|1600.90/1662.00|
|grounding-dino-base|managed-non-stream|2.163845|271.016|270.282/271.151/271.171|2918.92|1600.90/1662.00|
|grounding-dino-base|managed-stream|2.171898|278.128|271.024/274.799/275.599|2918.92|1600.90/1662.00|
|faster-rcnn|raw-non-stream|0.713178|23.874|22.374/23.637/25.743|1900.94|202.88/324.00|
|faster-rcnn|native-non-stream|0.750179|25.409|23.430/26.884/28.507|1900.94|202.85/324.00|
|faster-rcnn|managed-non-stream|0.780401|26.060|24.804/25.988/28.569|1938.42|202.85/324.00|
|faster-rcnn|managed-stream|0.776378|28.257|24.174/24.968/27.339|1938.34|202.85/324.00|
|d-fine|raw-non-stream|3.003563|377.392|377.738/380.135/380.291|3167.76|146.85/252.00|
|d-fine|native-non-stream|3.027294|377.407|383.475/385.038/385.263|3167.77|153.40/252.00|
|d-fine|managed-non-stream|3.023268|377.803|378.267/380.580/380.621|3176.89|154.25/252.00|
|d-fine|managed-stream|3.026333|380.193|377.276/381.629/381.888|3176.93|150.29/252.00|
|locate-anything|raw-non-stream|0.404202|404.152|403.581/403.581/403.581|4093.07|7910.11/8364.00|
|locate-anything|native-non-stream|0.406754|406.696|407.723/407.723/407.723|4093.07|7910.11/8364.00|
|locate-anything|managed-non-stream|0.409702|409.643|410.935/410.935/410.935|4093.07|7910.11/8364.00|
|locate-anything|managed-stream|0.406436|406.344|405.124/405.124/405.124|4093.07|7910.11/8364.00|

## images, batch 4

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|225.233|214.907 ± 18.131|189.106–230.894|8.44|197.176–230.372|0.00|PASS|
|yolo|raw-stream|7|225.404|216.419 ± 16.073|198.833–234.286|7.43|198.984–228.662|-0.08|PASS|
|yolo|native-non-stream|7|196.821|195.871 ± 11.730|183.242–217.059|5.99|185.186–201.423|8.99|PASS|
|yolo|native-stream|7|196.084|195.916 ± 11.239|183.883–217.075|5.74|184.217–198.835|9.76|PASS|
|yolo|managed-non-stream|7|217.519|217.064 ± 2.866|211.481–220.064|1.32|215.903–219.666|2.53|PASS|
|yolo|managed-stream|7|218.603|218.873 ± 1.876|215.883–221.896|0.86|218.101–220.287|2.82|PASS|
|yolo11|raw-non-stream|7|200.783|193.774 ± 11.887|177.878–205.497|6.13|181.556–204.016|0.00|PASS|
|yolo11|raw-stream|7|195.007|194.165 ± 12.093|180.504–208.006|6.23|182.539–205.177|-0.54|PASS|
|yolo11|native-non-stream|7|177.562|174.898 ± 6.434|165.201–181.819|3.68|167.629–179.659|12.66|PASS|
|yolo11|native-stream|7|177.726|175.562 ± 4.024|170.548–179.586|2.29|170.711–178.868|11.80|PASS|
|yolo11|managed-non-stream|7|193.519|194.622 ± 2.989|190.649–199.406|1.54|192.811–197.496|2.34|PASS|
|yolo11|managed-stream|7|195.452|195.675 ± 2.015|192.328–198.314|1.03|194.231–197.452|2.36|PASS|
|yolo26|raw-non-stream|7|194.440|186.359 ± 12.224|169.363–196.970|6.56|174.199–196.799|0.00|PASS|
|yolo26|raw-stream|7|189.288|184.580 ± 11.048|171.505–197.777|5.99|172.922–193.050|1.57|PASS|
|yolo26|native-non-stream|7|170.548|168.607 ± 3.876|164.157–172.665|2.30|164.278–172.629|13.32|PASS|
|yolo26|native-stream|7|171.188|169.188 ± 5.240|161.428–174.592|3.10|162.603–173.021|12.07|PASS|
|yolo26|managed-non-stream|7|188.324|187.450 ± 1.570|185.143–188.542|0.84|185.198–188.541|3.78|PASS|
|yolo26|managed-stream|7|185.509|185.911 ± 2.699|182.120–189.601|1.45|183.525–188.240|3.89|PASS|
|yolo-world|raw-non-stream|7|191.627|191.208 ± 2.051|189.265–194.753|1.07|189.313–192.142|0.00|PASS|
|yolo-world|raw-stream|7|192.987|191.838 ± 2.719|187.384–194.023|1.42|188.493–193.882|-0.95|PASS|
|yolo-world|native-non-stream|7|184.190|184.390 ± 2.176|181.240–187.622|1.18|182.817–186.116|2.96|PASS|
|yolo-world|native-stream|7|184.064|183.584 ± 3.001|177.509–186.399|1.63|182.504–185.755|3.70|PASS|
|yolo-world|managed-non-stream|7|199.708|199.022 ± 3.220|192.597–202.173|1.62|198.218–201.728|-4.53|PASS|
|yolo-world|managed-stream|7|199.622|199.178 ± 2.114|196.112–201.508|1.06|196.414–200.988|-3.78|PASS|
|grounding-dino|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|raw-non-stream|7|5.050|5.055 ± 0.020|5.033–5.089|0.39|5.041–5.073|0.00|PASS|
|grounding-dino|native-non-stream|7|5.047|5.035 ± 0.016|5.017–5.050|0.32|5.019–5.049|0.47|PASS|
|grounding-dino|managed-non-stream|7|5.014|5.021 ± 0.017|5.002–5.047|0.34|5.008–5.035|0.76|PASS|
|grounding-dino|managed-stream|7|5.029|5.028 ± 0.008|5.011–5.039|0.17|5.027–5.032|0.46|PASS|
|grounding-dino-base|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|raw-non-stream|7|3.703|3.702 ± 0.005|3.692–3.706|0.14|3.699–3.706|0.00|PASS|
|grounding-dino-base|native-non-stream|7|3.704|3.701 ± 0.009|3.685–3.707|0.24|3.692–3.707|-0.03|PASS|
|grounding-dino-base|managed-non-stream|7|3.691|3.693 ± 0.012|3.673–3.707|0.32|3.685–3.706|0.42|PASS|
|grounding-dino-base|managed-stream|7|3.698|3.694 ± 0.012|3.675–3.707|0.33|3.684–3.706|0.13|PASS|
|faster-rcnn|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|raw-non-stream|7|72.031|72.024 ± 0.165|71.769–72.275|0.23|71.880–72.115|0.00|PASS|
|faster-rcnn|native-non-stream|7|69.892|69.830 ± 0.171|69.467–69.970|0.24|69.773–69.916|3.18|PASS|
|faster-rcnn|managed-non-stream|7|68.585|68.264 ± 1.200|65.612–69.114|1.76|68.379–69.025|5.12|PASS|
|faster-rcnn|managed-stream|7|68.741|68.776 ± 0.137|68.612–69.039|0.20|68.699–68.824|4.67|PASS|
|d-fine|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|raw-non-stream|7|10.214|10.173 ± 0.075|10.024–10.218|0.74|10.116–10.215|0.00|PASS|
|d-fine|native-non-stream|7|10.074|10.057 ± 0.063|9.952–10.141|0.63|10.009–10.096|1.21|PASS|
|d-fine|managed-non-stream|7|10.011|9.981 ± 0.103|9.777–10.084|1.03|9.923–10.046|1.87|PASS|
|d-fine|managed-stream|7|10.037|10.012 ± 0.067|9.867–10.054|0.67|10.004–10.052|1.64|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.136600|29.199|17.258/17.918/18.682|2003.57|76.00/186.00|
|yolo|raw-stream|1.135738|28.854|17.167/20.237/24.084|2003.57|76.00/186.00|
|yolo|native-non-stream|1.300677|31.718|20.269/22.593/26.895|2003.57|75.87/186.00|
|yolo|native-stream|1.305561|19.737|20.938/22.272/25.555|1789.11|75.87/186.00|
|yolo|managed-non-stream|1.176910|19.189|18.715/20.432/22.094|1789.12|75.87/186.00|
|yolo|managed-stream|1.171072|19.397|18.664/20.053/21.926|1789.12|75.87/186.00|
|yolo11|raw-non-stream|1.275006|31.081|19.373/22.916/23.300|2055.15|112.51/184.00|
|yolo11|raw-stream|1.312775|30.949|19.217/21.579/26.296|2055.15|112.51/184.00|
|yolo11|native-non-stream|1.441752|34.045|22.770/25.767/30.070|2055.15|112.39/184.00|
|yolo11|native-stream|1.440421|21.920|22.918/24.248/24.549|1840.68|112.39/184.00|
|yolo11|managed-non-stream|1.322865|21.395|20.863/21.955/22.970|1840.70|112.39/184.00|
|yolo11|managed-stream|1.309786|21.427|20.818/22.200/22.955|1840.70|112.39/184.00|
|yolo26|raw-non-stream|1.316603|31.950|20.069/20.769/21.122|1990.20|111.72/184.00|
|yolo26|raw-stream|1.352437|31.887|20.093/20.660/27.124|1990.20|111.72/184.00|
|yolo26|native-non-stream|1.501043|34.807|23.392/25.813/30.503|1990.20|111.60/184.00|
|yolo26|native-stream|1.495435|22.844|23.651/25.275/25.935|1776.38|111.60/184.00|
|yolo26|managed-non-stream|1.359362|22.331|21.723/23.040/24.154|1776.38|111.60/184.00|
|yolo26|managed-stream|1.379990|22.319|22.041/23.643/24.492|1776.38|111.60/184.00|
|yolo-world|raw-non-stream|1.335932|32.166|20.569/22.073/24.660|2252.02|214.50/304.00|
|yolo-world|raw-stream|1.326512|31.687|20.733/21.808/27.992|2252.02|214.50/304.00|
|yolo-world|native-non-stream|1.389867|34.695|21.832/23.952/29.738|2252.02|214.38/304.00|
|yolo-world|native-stream|1.390819|22.413|21.712/22.672/23.617|2038.14|214.38/304.00|
|yolo-world|managed-non-stream|1.281872|22.193|19.882/22.324/24.940|2038.14|214.38/304.00|
|yolo-world|managed-stream|1.282424|22.031|20.082/21.312/22.335|2038.14|214.38/304.00|
|grounding-dino|raw-non-stream|6.336777|788.719|792.731/795.719/796.162|2979.10|3486.48/4298.00|
|grounding-dino|native-non-stream|6.340895|793.842|794.226/797.272/798.098|2979.11|3486.46/4298.00|
|grounding-dino|managed-non-stream|6.381996|791.982|797.583/802.263/803.296|3016.73|3486.46/4298.00|
|grounding-dino|managed-stream|6.363589|788.151|795.925/797.642/797.768|3012.14|3486.46/4298.00|
|grounding-dino-base|raw-non-stream|8.640729|1077.923|1079.637/1081.146/1081.506|2986.48|3731.84/4524.00|
|grounding-dino-base|native-non-stream|8.639446|1074.629|1079.393/1081.627/1082.112|2986.50|3731.82/4524.00|
|grounding-dino-base|managed-non-stream|8.669857|1079.552|1080.273/1083.404/1083.597|3023.20|3731.82/4524.00|
|grounding-dino-base|managed-stream|8.652966|1074.321|1082.288/1088.545/1088.717|3023.20|3731.82/4524.00|
|faster-rcnn|raw-non-stream|1.777017|55.328|55.551/56.281/56.642|2044.98|561.57/1006.00|
|faster-rcnn|native-non-stream|1.831407|57.235|57.267/59.280/60.017|2044.98|561.45/1006.00|
|faster-rcnn|managed-non-stream|1.866296|58.125|58.247/59.904/60.477|2046.74|561.45/1006.00|
|faster-rcnn|managed-stream|1.862060|57.770|57.972/58.673/58.763|2046.34|561.45/1006.00|
|d-fine|raw-non-stream|3.133053|390.516|392.859/399.479/399.848|3199.68|509.82/752.00|
|d-fine|native-non-stream|3.176648|396.658|406.761/410.406/410.494|3199.68|477.43/752.00|
|d-fine|managed-non-stream|3.196444|401.235|399.427/401.912/402.264|3237.20|490.85/752.00|
|d-fine|managed-stream|3.188067|404.560|395.465/402.484/404.507|3237.27|495.01/752.00|

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|107.825|106.784 ± 1.971|102.854–108.516|1.85|105.938–107.997|0.00|PASS|
|yolo|raw-stream|7|105.959|106.583 ± 1.337|105.522–109.201|1.25|105.708–107.542|0.42|PASS|
|yolo|native-non-stream|7|100.888|101.298 ± 1.532|99.069–103.308|1.51|100.002–102.798|5.04|PASS|
|yolo|native-stream|7|100.398|100.215 ± 0.966|98.845–101.625|0.96|99.083–100.703|6.27|PASS|
|yolo|managed-non-stream|7|99.213|99.541 ± 0.931|98.194–100.855|0.94|99.014–100.444|7.81|PASS|
|yolo|managed-stream|7|97.475|97.665 ± 1.272|96.172–99.622|1.30|96.563–99.127|9.47|PASS|
|yolo11|raw-non-stream|7|87.423|86.755 ± 2.142|83.865–88.823|2.47|84.341–88.679|0.00|PASS|
|yolo11|raw-stream|7|86.991|87.017 ± 0.942|85.304–88.443|1.08|86.864–87.579|0.64|PASS|
|yolo11|native-non-stream|7|82.376|82.288 ± 1.344|80.690–84.325|1.63|81.114–83.366|5.16|PASS|
|yolo11|native-stream|7|82.924|83.252 ± 1.215|82.043–85.536|1.46|82.194–84.109|4.07|PASS|
|yolo11|managed-non-stream|7|81.526|81.401 ± 1.313|79.330–82.927|1.61|79.941–82.313|7.01|PASS|
|yolo11|managed-stream|7|80.118|80.254 ± 0.443|79.674–81.093|0.55|80.055–80.473|7.81|PASS|
|yolo26|raw-non-stream|7|78.797|78.479 ± 1.213|76.184–79.598|1.55|77.559–79.331|0.00|PASS|
|yolo26|raw-stream|7|78.777|78.717 ± 0.861|77.741–80.061|1.09|77.991–79.487|-0.20|PASS|
|yolo26|native-non-stream|7|75.717|75.591 ± 0.947|74.639–76.954|1.25|74.652–76.584|4.07|PASS|
|yolo26|native-stream|7|74.031|74.292 ± 0.828|73.604–76.004|1.12|73.671–74.631|5.34|PASS|
|yolo26|managed-non-stream|7|74.628|74.200 ± 1.087|72.533–75.392|1.47|72.984–75.147|6.06|PASS|
|yolo26|managed-stream|7|72.995|72.821 ± 1.378|70.924–74.542|1.89|71.579–74.069|8.68|PASS|
|yolo-world|raw-non-stream|7|77.435|76.888 ± 1.273|75.035–77.946|1.66|75.102–77.838|0.00|PASS|
|yolo-world|raw-stream|7|77.138|77.088 ± 1.349|75.401–79.085|1.75|76.083–78.550|-0.91|PASS|
|yolo-world|native-non-stream|7|74.560|74.514 ± 0.257|73.953–74.738|0.35|74.529–74.647|3.77|PASS|
|yolo-world|native-stream|7|72.742|72.741 ± 1.119|71.077–74.212|1.54|71.758–73.939|5.14|PASS|
|yolo-world|managed-non-stream|7|72.485|72.698 ± 1.075|71.229–74.306|1.48|71.994–73.888|5.21|PASS|
|yolo-world|managed-stream|7|72.445|71.961 ± 1.138|69.884–73.144|1.58|71.169–72.851|6.85|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.112913|1112.828|1168.445/1168.445/1168.445|2167.85|34.30/62.00|
|yolo|raw-stream|1.132508|47.097|1127.813/1127.813/1127.813|2167.98|34.30/62.00|
|yolo|native-non-stream|1.189438|1189.321|1179.817/1179.817/1179.817|2167.98|34.30/62.00|
|yolo|native-stream|1.195241|50.357|1222.942/1222.942/1222.942|2167.98|34.25/62.00|
|yolo|managed-non-stream|1.209514|1209.425|1226.984/1226.984/1226.984|1861.13|24.66/62.00|
|yolo|managed-stream|1.231091|83.690|1250.383/1250.383/1250.383|1861.12|24.66/62.00|
|yolo11|raw-non-stream|1.372637|1372.552|1380.318/1380.318/1380.318|2556.53|64.67/92.00|
|yolo11|raw-stream|1.379451|48.678|1429.789/1429.789/1429.789|2556.53|64.67/92.00|
|yolo11|native-non-stream|1.456733|1456.616|1500.689/1500.689/1500.689|2556.18|64.67/92.00|
|yolo11|native-stream|1.447115|51.516|1515.572/1515.572/1515.572|2549.64|64.67/92.00|
|yolo11|managed-non-stream|1.471923|1471.834|1468.415/1468.415/1468.415|2248.49|56.12/92.00|
|yolo11|managed-stream|1.497785|85.333|1492.258/1492.258/1492.258|2253.88|56.12/92.00|
|yolo26|raw-non-stream|1.522909|1522.827|1576.992/1576.992/1576.992|2483.12|62.86/90.00|
|yolo26|raw-stream|1.523288|49.744|1537.752/1537.752/1537.752|2492.81|62.86/90.00|
|yolo26|native-non-stream|1.584857|1584.778|1590.990/1590.990/1590.990|2483.12|62.86/90.00|
|yolo26|native-stream|1.620944|52.985|1638.860/1638.860/1638.860|2483.25|62.80/90.00|
|yolo26|managed-non-stream|1.607969|1607.881|1623.316/1623.316/1623.316|2182.78|55.34/90.00|
|yolo26|managed-stream|1.643938|86.007|1626.049/1626.049/1626.049|2182.80|55.34/90.00|
|yolo-world|raw-non-stream|1.549693|1549.615|1533.164/1533.164/1533.164|2751.43|115.88/134.00|
|yolo-world|raw-stream|1.555661|49.797|1565.003/1565.003/1565.003|2748.27|115.88/134.00|
|yolo-world|native-non-stream|1.609447|1609.326|1660.010/1660.010/1660.010|2748.27|115.88/134.00|
|yolo-world|native-stream|1.649671|51.864|1659.242/1659.242/1659.242|2748.27|115.88/134.00|
|yolo-world|managed-non-stream|1.655508|1655.418|1640.469/1640.469/1640.469|2438.95|107.97/134.00|
|yolo-world|managed-stream|1.656426|85.256|1671.020/1671.020/1671.020|2438.95|107.97/134.00|

## video, batch 4

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo|raw-non-stream|7|223.676|222.289 ± 4.037|215.366–226.578|1.82|219.025–225.632|0.00|PASS|
|yolo|raw-stream|7|218.430|219.709 ± 6.525|209.293–229.502|2.97|216.012–225.141|2.31|PASS|
|yolo|managed-non-stream|7|200.724|200.404 ± 3.622|195.355–205.158|1.81|196.621–203.363|10.44|PASS|
|yolo|managed-stream|7|197.028|196.059 ± 1.866|193.400–197.976|0.95|193.663–197.514|13.97|PASS|
|yolo11|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo11|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo11|raw-non-stream|7|197.328|195.734 ± 4.882|189.235–203.377|2.49|190.569–198.412|0.00|PASS|
|yolo11|raw-stream|7|198.399|198.987 ± 2.410|195.560–202.351|1.21|197.585–201.738|-1.65|PASS|
|yolo11|managed-non-stream|7|182.838|183.383 ± 2.040|181.638–187.384|1.11|181.840–184.344|8.52|PASS|
|yolo11|managed-stream|7|176.703|176.305 ± 2.135|173.192–179.728|1.21|174.566–177.337|11.88|PASS|
|yolo26|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo26|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo26|raw-non-stream|7|185.281|185.673 ± 2.277|183.147–189.250|1.23|183.626–187.849|0.00|PASS|
|yolo26|raw-stream|7|184.570|184.293 ± 2.411|180.790–187.623|1.31|182.670–186.732|0.23|PASS|
|yolo26|managed-non-stream|7|171.281|170.727 ± 1.470|168.506–172.085|0.86|168.768–171.887|8.96|PASS|
|yolo26|managed-stream|7|164.154|163.041 ± 5.959|150.262–168.001|3.66|162.178–166.872|13.22|PASS|
|yolo-world|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo-world|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo-world|raw-non-stream|7|185.904|185.559 ± 5.338|178.537–194.825|2.88|180.034–187.726|0.00|PASS|
|yolo-world|raw-stream|7|188.094|187.389 ± 1.966|183.637–189.299|1.05|186.544–188.856|-0.40|PASS|
|yolo-world|managed-non-stream|7|174.965|174.113 ± 3.258|168.038–177.918|1.87|172.468–176.195|5.71|PASS|
|yolo-world|managed-stream|7|168.963|169.110 ± 3.389|163.506–173.858|2.00|167.417–171.837|11.04|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|0.536490|536.408|547.077/547.077/547.077|2395.85|93.11/150.00|
|yolo|raw-stream|0.549376|59.417|541.596/541.596/541.596|2370.40|93.11/150.00|
|yolo|managed-non-stream|0.597836|597.739|603.909/603.909/603.909|2350.16|64.78/150.00|
|yolo|managed-stream|0.609051|96.725|625.875/625.875/625.875|2258.38|64.78/150.00|
|yolo11|raw-non-stream|0.608124|608.035|594.543/594.543/594.543|2804.52|126.55/188.00|
|yolo11|raw-stream|0.604843|61.695|602.348/602.348/602.348|2530.40|126.55/188.00|
|yolo11|managed-non-stream|0.656318|656.222|677.790/677.790/677.790|2530.41|99.31/188.00|
|yolo11|managed-stream|0.679108|96.872|678.118/678.118/678.118|2530.34|99.31/188.00|
|yolo26|raw-non-stream|0.647666|647.579|652.057/652.057/652.057|2523.72|119.89/162.00|
|yolo26|raw-stream|0.650159|61.611|632.824/632.824/632.824|2515.82|119.89/162.00|
|yolo26|managed-non-stream|0.700604|700.510|691.349/691.349/691.349|2513.54|98.52/162.00|
|yolo26|managed-stream|0.731022|100.340|752.986/752.986/752.986|2513.51|98.52/162.00|
|yolo-world|raw-non-stream|0.645494|645.410|651.755/651.755/651.755|2658.64|209.49/292.00|
|yolo-world|raw-stream|0.637978|61.855|659.630/659.630/659.630|2658.97|209.49/292.00|
|yolo-world|managed-non-stream|0.685852|685.761|692.872/692.872/692.872|2658.44|188.26/292.00|
|yolo-world|managed-stream|0.710214|99.380|711.796/711.796/711.796|2623.86|188.26/292.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
