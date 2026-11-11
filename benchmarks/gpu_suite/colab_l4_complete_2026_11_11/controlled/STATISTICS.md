# Full raw-repeat benchmark statistics

Percentile bootstrap of repeat-level observations within one session. Seven serial, alternating repeats are a small correlated sample. Intervals are exploratory and do not cover between-session/hardware/workload variability or establish significance.

All raw baselines remain visible. SD is sample standard deviation across timed repeats; CV is SD / mean. The paired overhead uses the matching repeat index against raw non-stream, not a ratio of aggregate medians. No failed parity result is discarded.

## images, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|100.372|100.487 ± 1.705|97.982–103.355|1.70|99.343–101.735|0.00|PASS|
|yolo|raw-stream|7|102.369|101.784 ± 1.819|98.898–103.711|1.79|100.199–103.629|-0.71|PASS|
|yolo|native-non-stream|7|79.320|78.912 ± 1.011|77.386–80.087|1.28|77.919–79.743|27.97|PASS|
|yolo|native-stream|7|79.234|79.431 ± 0.705|78.454–80.730|0.89|79.086–79.800|26.27|PASS|
|yolo|managed-non-stream|7|73.015|73.321 ± 0.830|72.418–74.310|1.13|72.593–74.222|37.47|PASS|
|yolo|managed-stream|7|70.336|70.483 ± 0.614|69.720–71.499|0.87|69.912–70.852|41.67|PASS|
|yolo11|raw-non-stream|7|86.077|85.935 ± 1.294|83.222–87.041|1.51|85.713–86.850|0.00|PASS|
|yolo11|raw-stream|7|85.927|85.894 ± 0.759|85.083–86.899|0.88|85.114–86.798|0.74|PASS|
|yolo11|native-non-stream|7|67.861|68.114 ± 0.960|66.751–69.620|1.41|67.496–68.778|26.84|PASS|
|yolo11|native-stream|7|68.122|67.906 ± 1.100|66.282–69.433|1.62|66.628–68.526|25.82|PASS|
|yolo11|managed-non-stream|7|62.166|61.685 ± 1.227|60.135–63.015|1.99|60.315–62.967|39.31|PASS|
|yolo11|managed-stream|7|61.311|61.078 ± 0.647|59.851–61.702|1.06|60.817–61.675|40.82|PASS|
|yolo26|raw-non-stream|7|77.295|77.727 ± 1.763|75.099–80.636|2.27|76.876–78.783|0.00|PASS|
|yolo26|raw-stream|7|78.354|78.479 ± 0.785|77.642–80.117|1.00|78.072–78.601|-0.45|PASS|
|yolo26|native-non-stream|7|63.486|63.339 ± 0.574|62.474–64.083|0.91|62.718–63.820|23.24|PASS|
|yolo26|native-stream|7|64.367|63.906 ± 0.702|62.795–64.539|1.10|63.305–64.468|21.49|PASS|
|yolo26|managed-non-stream|7|59.909|59.751 ± 0.659|58.542–60.475|1.10|59.381–60.400|30.17|PASS|
|yolo26|managed-stream|7|58.454|58.246 ± 0.720|57.037–59.047|1.24|57.496–58.817|33.76|PASS|
|yolo-world|raw-non-stream|7|77.234|76.564 ± 2.361|71.307–78.064|3.08|76.855–77.846|0.00|PASS|
|yolo-world|raw-stream|7|77.587|77.595 ± 1.749|74.982–80.391|2.25|76.192–78.833|-0.39|PASS|
|yolo-world|native-non-stream|7|64.585|64.506 ± 0.727|63.664–65.528|1.13|63.795–65.284|18.96|PASS|
|yolo-world|native-stream|7|63.445|63.185 ± 0.918|61.817–63.940|1.45|61.942–63.889|21.48|PASS|
|yolo-world|managed-non-stream|7|59.588|59.402 ± 0.536|58.306–59.913|0.90|59.237–59.796|29.65|PASS|
|yolo-world|managed-stream|7|58.084|58.183 ± 0.615|57.335–58.882|1.06|57.504–58.863|32.32|PASS|
|grounding-dino|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|raw-non-stream|7|4.894|4.898 ± 0.018|4.879–4.924|0.36|4.881–4.918|0.00|PASS|
|grounding-dino|native-non-stream|7|4.819|4.814 ± 0.017|4.793–4.841|0.34|4.794–4.819|1.79|PASS|
|grounding-dino|managed-non-stream|7|4.811|4.807 ± 0.020|4.778–4.827|0.41|4.786–4.827|1.50|PASS|
|grounding-dino|managed-stream|7|4.813|4.808 ± 0.015|4.775–4.821|0.32|4.805–4.816|1.75|PASS|
|grounding-dino-base|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|raw-non-stream|7|3.725|3.730 ± 0.018|3.709–3.755|0.48|3.716–3.748|0.00|PASS|
|grounding-dino-base|native-non-stream|7|3.678|3.677 ± 0.022|3.640–3.710|0.60|3.663–3.695|1.42|PASS|
|grounding-dino-base|managed-non-stream|7|3.664|3.662 ± 0.013|3.638–3.679|0.36|3.659–3.671|1.97|PASS|
|grounding-dino-base|managed-stream|7|3.656|3.653 ± 0.015|3.623–3.666|0.42|3.646–3.665|1.94|PASS|
|faster-rcnn|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|raw-non-stream|7|43.968|43.908 ± 0.193|43.609–44.112|0.44|43.702–44.079|0.00|PASS|
|faster-rcnn|native-non-stream|7|39.933|40.017 ± 0.390|39.581–40.604|0.97|39.668–40.450|10.11|PASS|
|faster-rcnn|managed-non-stream|7|39.213|39.210 ± 0.266|38.776–39.620|0.68|39.052–39.388|11.99|PASS|
|faster-rcnn|managed-stream|7|38.767|38.580 ± 0.702|37.316–39.287|1.82|38.001–39.240|13.13|PASS|
|d-fine|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|raw-non-stream|7|2.670|2.663 ± 0.017|2.630–2.677|0.63|2.651–2.673|0.00|PASS|
|d-fine|native-non-stream|7|2.643|2.642 ± 0.012|2.622–2.658|0.44|2.635–2.649|0.79|PASS|
|d-fine|managed-non-stream|7|2.646|2.645 ± 0.007|2.631–2.654|0.28|2.642–2.652|0.92|PASS|
|d-fine|managed-stream|7|2.642|2.643 ± 0.012|2.627–2.656|0.45|2.631–2.655|0.76|PASS|
|locate-anything|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|locate-anything|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|locate-anything|raw-non-stream|7|2.440|2.435 ± 0.018|2.404–2.453|0.73|2.419–2.451|0.00|PASS|
|locate-anything|native-non-stream|7|2.442|2.425 ± 0.029|2.365–2.443|1.20|2.411–2.443|0.33|PASS|
|locate-anything|managed-non-stream|7|2.442|2.440 ± 0.016|2.421–2.468|0.66|2.423–2.448|0.05|PASS|
|locate-anything|managed-stream|7|2.424|2.426 ± 0.013|2.403–2.444|0.55|2.420–2.436|0.16|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|0.637626|14.065|9.782/10.659/13.182|1874.73|27.84/60.00|
|yolo|raw-stream|0.625187|14.370|9.694/11.640/12.739|1893.91|27.84/60.00|
|yolo|native-non-stream|0.806858|17.941|12.422/13.181/16.688|1894.07|27.81/60.00|
|yolo|native-stream|0.807732|17.948|12.503/14.545/15.057|1895.22|27.81/60.00|
|yolo|managed-non-stream|0.876527|15.013|14.065/16.038/16.855|1895.27|27.81/60.00|
|yolo|managed-stream|0.909914|15.590|14.674/15.574/18.514|1895.30|27.81/60.00|
|yolo11|raw-non-stream|0.743519|15.901|11.535/13.751/14.408|1923.91|59.81/88.00|
|yolo11|raw-stream|0.744815|15.626|11.704/12.569/14.043|1923.91|59.81/88.00|
|yolo11|native-non-stream|0.943097|19.186|14.617/15.461/18.767|1943.40|59.77/88.00|
|yolo11|native-stream|0.939492|16.843|14.563/15.154/15.943|1944.96|59.77/88.00|
|yolo11|managed-non-stream|1.029508|17.838|15.736/16.189/17.083|1945.01|59.77/88.00|
|yolo11|managed-stream|1.043867|20.969|16.180/16.945/18.737|1945.08|59.77/88.00|
|yolo26|raw-non-stream|0.828001|17.178|12.820/15.710/15.816|1864.23|59.02/88.00|
|yolo26|raw-stream|0.816802|16.868|12.508/15.287/16.232|1876.00|59.02/88.00|
|yolo26|native-non-stream|1.008092|19.852|15.736/17.071/19.345|1874.39|58.99/88.00|
|yolo26|native-stream|0.994305|20.730|15.865/16.879/18.292|1876.29|58.99/88.00|
|yolo26|managed-non-stream|1.068284|18.318|16.907/19.550/19.939|1876.27|58.99/88.00|
|yolo26|managed-stream|1.094881|18.622|17.304/20.170/21.526|1877.34|58.99/88.00|
|yolo-world|raw-non-stream|0.828648|17.387|12.714/13.731/14.795|2108.68|115.04/132.00|
|yolo-world|raw-stream|0.824885|16.686|12.660/13.184/15.162|2108.68|115.04/132.00|
|yolo-world|native-non-stream|0.990939|20.035|15.662/17.951/19.897|2126.85|115.01/132.00|
|yolo-world|native-stream|1.008745|17.098|15.955/17.788/18.978|2126.85|115.01/132.00|
|yolo-world|managed-non-stream|1.074035|18.054|16.956/17.678/19.597|2128.76|115.01/132.00|
|yolo-world|managed-stream|1.101852|18.447|17.373/18.116/21.173|2127.22|115.01/132.00|
|grounding-dino|raw-non-stream|1.634787|203.847|204.123/209.432/210.153|2942.87|1366.93/1424.00|
|grounding-dino|native-non-stream|1.660198|206.522|208.413/211.825/212.169|2942.87|1366.92/1424.00|
|grounding-dino|managed-non-stream|1.662936|206.905|207.451/212.097/213.072|2947.56|1366.92/1424.00|
|grounding-dino|managed-stream|1.662053|204.841|207.727/212.928/214.538|2947.56|1366.92/1424.00|
|grounding-dino-base|raw-non-stream|2.147819|267.957|267.736/271.914/272.923|2910.48|1600.91/1662.00|
|grounding-dino-base|native-non-stream|2.174938|271.891|270.944/274.449/275.672|2910.48|1600.90/1662.00|
|grounding-dino-base|managed-non-stream|2.183517|272.308|272.334/272.905/272.964|2919.86|1600.90/1662.00|
|grounding-dino-base|managed-stream|2.188128|279.405|272.255/276.549/277.772|2919.86|1600.90/1662.00|
|faster-rcnn|raw-non-stream|0.727800|24.977|22.731/24.079/24.411|1916.14|202.88/324.00|
|faster-rcnn|native-non-stream|0.801344|26.435|24.764/25.670/26.331|1951.29|202.85/324.00|
|faster-rcnn|managed-non-stream|0.816050|26.926|28.914/29.935/30.022|1951.27|202.85/324.00|
|faster-rcnn|managed-stream|0.825451|26.896|29.050/30.080/30.866|1953.66|202.85/324.00|
|d-fine|raw-non-stream|2.995936|373.292|376.433/379.728/380.632|3192.07|146.90/238.00|
|d-fine|native-non-stream|3.026384|377.946|378.535/382.270/383.051|3192.09|152.71/238.00|
|d-fine|managed-non-stream|3.023246|377.636|377.625/380.042/380.747|3201.10|154.29/238.00|
|d-fine|managed-stream|3.027959|377.746|376.939/379.611/379.984|3201.22|142.60/238.00|
|locate-anything|raw-non-stream|0.409885|409.837|407.369/407.369/407.369|4172.50|7910.11/8364.00|
|locate-anything|native-non-stream|0.409584|409.526|411.252/411.252/411.252|4172.50|7910.11/8364.00|
|locate-anything|managed-non-stream|0.409546|409.490|425.160/425.160/425.160|4172.50|7910.11/8364.00|
|locate-anything|managed-stream|0.412492|412.268|411.460/411.460/411.460|4172.51|7910.11/8364.00|

## images, batch 4

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|248.406|246.439 ± 3.639|242.290–251.129|1.48|242.691–249.004|0.00|PASS|
|yolo|raw-stream|7|246.081|246.647 ± 2.545|244.181–250.870|1.03|244.235–248.230|-0.46|PASS|
|yolo|native-non-stream|7|171.561|171.428 ± 3.116|166.609–174.691|1.82|168.444–174.321|42.84|PASS|
|yolo|native-stream|7|168.800|169.540 ± 2.207|166.680–172.620|1.30|167.563–171.650|45.44|PASS|
|yolo|managed-non-stream|7|140.205|140.377 ± 1.275|138.497–141.844|0.91|139.480–141.794|75.13|PASS|
|yolo|managed-stream|7|131.370|130.956 ± 1.558|129.043–133.368|1.19|129.410–132.055|87.54|PASS|
|yolo11|raw-non-stream|7|210.243|208.314 ± 5.004|199.868–213.318|2.40|204.105–212.352|0.00|PASS|
|yolo11|raw-stream|7|209.938|209.410 ± 3.624|204.044–214.700|1.73|205.628–211.832|0.70|PASS|
|yolo11|native-non-stream|7|154.169|153.143 ± 4.349|143.823–157.480|2.84|153.028–155.093|37.74|PASS|
|yolo11|native-stream|7|155.124|154.886 ± 1.495|152.254–156.854|0.96|153.873–156.085|35.49|PASS|
|yolo11|managed-non-stream|7|128.691|129.716 ± 2.742|126.560–133.811|2.11|127.067–132.198|63.37|PASS|
|yolo11|managed-stream|7|124.194|123.973 ± 1.688|121.633–126.981|1.36|122.899–124.722|67.98|PASS|
|yolo26|raw-non-stream|7|195.689|196.019 ± 1.751|194.284–198.666|0.89|194.514–197.918|0.00|PASS|
|yolo26|raw-stream|7|196.726|195.707 ± 3.444|191.022–199.547|1.76|191.067–198.631|-0.36|PASS|
|yolo26|native-non-stream|7|146.468|144.504 ± 5.318|133.072–148.897|3.68|143.269–147.163|33.43|PASS|
|yolo26|native-stream|7|144.164|144.459 ± 3.855|137.548–148.810|2.67|142.074–147.423|35.13|PASS|
|yolo26|managed-non-stream|7|126.669|126.176 ± 1.254|124.671–127.600|0.99|124.955–127.467|55.39|PASS|
|yolo26|managed-stream|7|118.317|119.106 ± 2.185|116.055–121.908|1.83|117.573–121.119|64.99|PASS|
|yolo-world|raw-non-stream|7|195.790|195.228 ± 2.468|190.442–198.045|1.26|194.280–197.094|0.00|PASS|
|yolo-world|raw-stream|7|195.838|195.877 ± 2.472|191.935–198.670|1.26|193.391–198.053|-0.52|PASS|
|yolo-world|native-non-stream|7|150.698|150.207 ± 2.410|145.044–152.827|1.60|150.601–150.839|30.24|PASS|
|yolo-world|native-stream|7|144.423|144.595 ± 1.093|143.040–146.150|0.76|143.795–145.653|35.20|PASS|
|yolo-world|managed-non-stream|7|127.688|126.607 ± 2.703|123.122–129.599|2.13|124.099–129.056|53.46|PASS|
|yolo-world|managed-stream|7|118.305|119.354 ± 2.604|116.560–122.573|2.18|116.722–122.037|61.39|PASS|
|grounding-dino|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino|raw-non-stream|7|5.060|5.063 ± 0.010|5.049–5.079|0.20|5.058–5.072|0.00|PASS|
|grounding-dino|native-non-stream|7|5.039|5.038 ± 0.003|5.033–5.043|0.07|5.035–5.040|0.53|PASS|
|grounding-dino|managed-non-stream|7|5.009|5.012 ± 0.010|5.000–5.026|0.19|5.004–5.021|1.13|PASS|
|grounding-dino|managed-stream|7|5.025|5.025 ± 0.006|5.019–5.034|0.11|5.020–5.028|0.76|PASS|
|grounding-dino-base|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|grounding-dino-base|raw-non-stream|7|3.708|3.707 ± 0.005|3.697–3.711|0.12|3.707–3.710|0.00|PASS|
|grounding-dino-base|native-non-stream|7|3.697|3.697 ± 0.006|3.689–3.706|0.17|3.690–3.703|0.32|PASS|
|grounding-dino-base|managed-non-stream|7|3.697|3.694 ± 0.007|3.681–3.700|0.19|3.691–3.699|0.32|PASS|
|grounding-dino-base|managed-stream|7|3.698|3.699 ± 0.008|3.688–3.709|0.21|3.691–3.706|0.26|PASS|
|faster-rcnn|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|faster-rcnn|raw-non-stream|7|71.813|71.904 ± 0.206|71.703–72.261|0.29|71.777–72.122|0.00|PASS|
|faster-rcnn|native-non-stream|7|68.230|68.097 ± 0.354|67.602–68.606|0.52|67.741–68.310|5.56|PASS|
|faster-rcnn|managed-non-stream|7|66.338|66.116 ± 0.670|64.893–66.765|1.01|65.688–66.634|8.44|PASS|
|faster-rcnn|managed-stream|7|65.989|65.680 ± 0.735|64.155–66.271|1.12|65.372–66.122|9.28|PASS|
|d-fine|raw-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|No upstream native source-stream engine|
|d-fine|raw-non-stream|7|10.369|10.360 ± 0.055|10.273–10.438|0.53|10.310–10.395|0.00|PASS|
|d-fine|native-non-stream|7|10.253|10.217 ± 0.110|10.046–10.338|1.07|10.117–10.303|1.10|PASS|
|d-fine|managed-non-stream|7|10.094|10.122 ± 0.119|9.940–10.242|1.17|10.019–10.237|2.55|PASS|
|d-fine|managed-stream|7|10.219|10.219 ± 0.063|10.154–10.337|0.62|10.155–10.244|1.47|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.030572|18.230|16.086/18.075/22.293|2189.70|76.00/186.00|
|yolo|raw-stream|1.040309|18.189|16.457/17.924/20.481|2275.46|76.00/186.00|
|yolo|native-non-stream|1.492181|24.262|22.733/25.395/26.577|2290.47|75.87/186.00|
|yolo|native-stream|1.516584|28.859|23.776/25.143/25.846|2297.68|75.87/186.00|
|yolo|managed-non-stream|1.825892|29.489|28.040/29.148/30.541|2305.23|75.87/186.00|
|yolo|managed-stream|1.948689|30.518|30.782/32.791/34.012|2416.58|75.87/186.00|
|yolo11|raw-non-stream|1.217636|25.669|19.893/20.920/28.637|2295.69|112.51/184.00|
|yolo11|raw-stream|1.219407|25.408|19.198/22.400/25.014|2273.06|112.51/184.00|
|yolo11|native-non-stream|1.660512|31.512|25.781/27.204/31.470|2272.10|112.39/184.00|
|yolo11|native-stream|1.650289|35.557|26.092/28.015/33.906|2288.88|112.39/184.00|
|yolo11|managed-non-stream|1.989259|31.359|30.111/31.733/33.088|2396.39|112.39/184.00|
|yolo11|managed-stream|2.061298|35.687|31.823/33.868/34.818|2465.71|112.39/184.00|
|yolo26|raw-non-stream|1.308201|28.031|20.779/21.633/28.660|2239.93|111.72/184.00|
|yolo26|raw-stream|1.301299|28.584|20.320/24.576/26.249|2119.08|111.72/184.00|
|yolo26|native-non-stream|1.747823|35.273|27.232/29.418/32.892|2119.08|111.60/184.00|
|yolo26|native-stream|1.775755|38.598|25.747/27.364/29.056|2139.33|111.60/184.00|
|yolo26|managed-non-stream|2.021020|33.148|31.133/32.653/34.644|2235.09|111.60/184.00|
|yolo26|managed-stream|2.163687|37.041|33.669/35.028/35.996|2394.56|111.60/184.00|
|yolo-world|raw-non-stream|1.307522|27.510|21.068/21.967/24.814|2624.57|214.50/304.00|
|yolo-world|raw-stream|1.307206|27.330|20.697/21.813/25.330|2383.18|214.50/304.00|
|yolo-world|native-non-stream|1.698761|32.940|27.047/28.578/31.739|2477.22|214.38/304.00|
|yolo-world|native-stream|1.772568|38.245|27.905/29.558/33.188|2478.98|214.38/304.00|
|yolo-world|managed-non-stream|2.004886|32.737|31.446/32.402/32.669|2590.60|214.38/304.00|
|yolo-world|managed-stream|2.163890|37.863|33.280/34.644/36.213|2650.37|214.38/304.00|
|grounding-dino|raw-non-stream|6.323604|789.942|790.060/792.784/793.101|3018.31|3486.48/4298.00|
|grounding-dino|native-non-stream|6.349981|792.477|795.149/796.858/797.349|3018.32|3486.46/4298.00|
|grounding-dino|managed-non-stream|6.388776|796.677|798.123/799.761/800.105|3055.55|3486.46/4298.00|
|grounding-dino|managed-stream|6.367884|793.746|795.791/799.035/799.454|3018.46|3486.46/4298.00|
|grounding-dino-base|raw-non-stream|8.630886|1080.617|1078.426/1085.190/1086.099|3024.92|3731.84/4524.00|
|grounding-dino-base|native-non-stream|8.655260|1081.163|1082.435/1087.814/1088.927|3024.74|3731.82/4524.00|
|grounding-dino-base|managed-non-stream|8.656655|1084.515|1085.581/1087.671/1088.009|3062.24|3731.82/4524.00|
|grounding-dino-base|managed-stream|8.653000|1073.577|1081.554/1088.288/1089.130|3024.98|3731.82/4524.00|
|faster-rcnn|raw-non-stream|1.782411|55.717|55.573/56.436/56.571|2113.17|561.57/1006.00|
|faster-rcnn|native-non-stream|1.875996|59.192|58.759/60.228/61.004|2187.94|561.45/1006.00|
|faster-rcnn|managed-non-stream|1.929524|59.627|60.277/69.395/69.768|2192.93|561.45/1006.00|
|faster-rcnn|managed-stream|1.939709|61.211|60.243/62.576/63.210|2193.38|561.45/1006.00|
|d-fine|raw-non-stream|3.086043|384.929|384.262/386.257/386.317|3237.36|514.69/714.00|
|d-fine|native-non-stream|3.121020|393.093|393.043/401.780/402.910|3237.36|488.93/714.00|
|d-fine|managed-non-stream|3.170277|392.794|390.959/399.688/400.979|3274.51|490.85/714.00|
|d-fine|managed-stream|3.131365|395.175|389.812/396.895/398.248|3237.46|485.75/714.00|

## video, batch 1

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|raw-non-stream|7|107.717|106.863 ± 2.008|103.286–108.593|1.88|104.826–108.190|0.00|PASS|
|yolo|raw-stream|7|107.708|106.852 ± 1.706|104.935–108.676|1.60|104.940–108.328|-0.20|PASS|
|yolo|native-non-stream|7|100.698|100.491 ± 1.753|97.311–102.616|1.74|99.635–102.034|5.36|PASS|
|yolo|native-stream|7|98.942|98.459 ± 1.505|95.805–100.096|1.53|97.466–99.629|8.47|PASS|
|yolo|managed-non-stream|7|95.654|96.262 ± 1.683|94.530–98.952|1.75|94.604–97.597|11.21|PASS|
|yolo|managed-stream|7|94.554|94.325 ± 2.844|89.909–98.820|3.02|91.803–95.489|13.37|PASS|
|yolo11|raw-non-stream|7|87.279|87.145 ± 0.659|86.055–87.967|0.76|86.663–87.683|0.00|PASS|
|yolo11|raw-stream|7|86.520|86.620 ± 1.507|84.860–88.308|1.74|85.181–88.123|0.40|PASS|
|yolo11|native-non-stream|7|82.837|82.730 ± 0.957|81.049–84.051|1.16|82.187–83.450|5.41|PASS|
|yolo11|native-stream|7|82.838|82.383 ± 1.396|80.037–84.032|1.69|80.904–83.149|5.48|PASS|
|yolo11|managed-non-stream|7|80.431|80.234 ± 1.636|76.841–82.151|2.04|80.252–81.078|8.76|PASS|
|yolo11|managed-stream|7|79.003|79.499 ± 0.795|78.804–80.519|1.00|78.822–80.402|9.20|PASS|
|yolo26|raw-non-stream|7|80.933|80.712 ± 1.205|78.333–81.873|1.49|80.138–81.817|0.00|PASS|
|yolo26|raw-stream|7|80.149|79.978 ± 0.983|78.753–81.562|1.23|78.896–80.597|1.51|PASS|
|yolo26|native-non-stream|7|77.183|77.036 ± 0.812|75.581–78.000|1.05|76.490–77.646|5.28|PASS|
|yolo26|native-stream|7|76.056|75.884 ± 0.410|75.284–76.510|0.54|75.593–76.071|6.39|PASS|
|yolo26|managed-non-stream|7|75.450|75.321 ± 0.630|74.090–75.875|0.84|74.957–75.867|7.37|PASS|
|yolo26|managed-stream|7|74.195|74.287 ± 0.752|73.327–75.663|1.01|73.763–74.777|8.21|PASS|
|yolo-world|raw-non-stream|7|76.874|77.211 ± 0.663|76.475–78.429|0.86|76.780–77.551|0.00|PASS|
|yolo-world|raw-stream|7|77.745|77.242 ± 1.849|73.950–79.097|2.39|75.621–78.903|-0.25|PASS|
|yolo-world|native-non-stream|7|75.546|75.575 ± 0.728|74.695–76.677|0.96|74.778–76.306|2.80|PASS|
|yolo-world|native-stream|7|72.321|72.146 ± 0.937|70.389–73.278|1.30|71.534–72.766|6.58|PASS|
|yolo-world|managed-non-stream|7|73.527|73.351 ± 0.742|72.295–74.244|1.01|72.390–73.938|5.40|PASS|
|yolo-world|managed-stream|7|72.739|71.984 ± 1.877|68.244–73.585|2.61|70.851–73.272|6.47|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|1.114029|1113.946|1129.528/1129.528/1129.528|2167.04|34.30/62.00|
|yolo|raw-stream|1.114124|46.226|1106.224/1106.224/1106.224|2167.04|34.30/62.00|
|yolo|native-non-stream|1.191686|1191.573|1178.017/1178.017/1178.017|2167.04|34.30/62.00|
|yolo|native-stream|1.212829|1212.673|1224.370/1224.370/1224.370|2167.04|34.25/62.00|
|yolo|managed-non-stream|1.254518|1254.416|1233.914/1233.914/1233.914|1861.08|24.66/62.00|
|yolo|managed-stream|1.269114|84.113|1257.004/1257.004/1257.004|1861.08|24.66/62.00|
|yolo11|raw-non-stream|1.374896|1374.812|1457.659/1457.659/1457.659|2555.78|64.67/90.00|
|yolo11|raw-stream|1.386956|48.816|1385.192/1385.192/1385.192|2555.79|64.67/90.00|
|yolo11|native-non-stream|1.448623|1448.513|1454.191/1454.191/1454.191|2555.43|64.67/90.00|
|yolo11|native-stream|1.448608|1448.538|1469.201/1469.201/1469.201|2550.48|64.67/90.00|
|yolo11|managed-non-stream|1.491971|1491.879|1504.680/1504.680/1504.680|2251.92|56.12/90.00|
|yolo11|managed-stream|1.518937|85.059|1513.265/1513.265/1513.265|2251.51|56.12/90.00|
|yolo26|raw-non-stream|1.482703|1482.618|1488.187/1488.187/1488.187|2491.78|62.86/90.00|
|yolo26|raw-stream|1.497215|49.508|1469.672/1469.672/1469.672|2491.78|62.86/90.00|
|yolo26|native-non-stream|1.554750|1554.648|1544.984/1544.984/1544.984|2491.78|62.86/90.00|
|yolo26|native-stream|1.577779|1577.708|1603.463/1603.463/1603.463|2491.78|62.80/90.00|
|yolo26|managed-non-stream|1.590459|1590.369|1658.711/1658.711/1658.711|2184.07|55.34/90.00|
|yolo26|managed-stream|1.617362|86.659|1636.974/1636.974/1636.974|2189.15|55.34/90.00|
|yolo-world|raw-non-stream|1.560993|1560.882|1564.361/1564.361/1564.361|2764.55|115.88/134.00|
|yolo-world|raw-stream|1.543516|49.914|1574.051/1574.051/1574.051|2753.89|115.88/134.00|
|yolo-world|native-non-stream|1.588432|1588.342|1641.288/1641.288/1641.288|2754.22|115.88/134.00|
|yolo-world|native-stream|1.659276|1659.203|1644.832/1644.832/1644.832|2754.24|115.88/134.00|
|yolo-world|managed-non-stream|1.632054|1631.957|1650.630/1650.630/1650.630|2447.39|107.97/134.00|
|yolo-world|managed-stream|1.649732|86.386|1654.862/1654.862/1654.862|2447.39|107.97/134.00|

## video, batch 4

| Model | Mode | n | FPS median | Mean ± SD | Min–max | CV % | Median exploratory 95% interval | Paired elapsed overhead % | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|yolo|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo|raw-non-stream|7|226.754|223.997 ± 8.671|205.759–231.368|3.87|221.519–229.304|0.00|PASS|
|yolo|raw-stream|7|222.784|223.334 ± 2.183|220.813–226.808|0.98|221.494–225.462|1.71|PASS|
|yolo|managed-non-stream|7|199.089|198.328 ± 4.626|191.931–204.321|2.33|194.235–203.281|12.39|PASS|
|yolo|managed-stream|7|192.748|192.463 ± 2.952|188.963–196.523|1.53|189.456–195.307|18.64|PASS|
|yolo11|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo11|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo11|raw-non-stream|7|198.002|196.295 ± 8.076|179.192–203.235|4.11|195.566–201.653|0.00|PASS|
|yolo11|raw-stream|7|196.855|197.303 ± 4.228|192.080–202.870|2.14|193.519–202.815|1.06|PASS|
|yolo11|managed-non-stream|7|179.615|177.805 ± 4.117|171.610–182.538|2.32|173.146–181.160|10.47|PASS|
|yolo11|managed-stream|7|173.848|173.359 ± 2.961|169.473–178.528|1.71|171.007–174.537|14.36|PASS|
|yolo26|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo26|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo26|raw-non-stream|7|190.112|190.847 ± 2.254|188.646–194.055|1.18|188.680–193.697|0.00|PASS|
|yolo26|raw-stream|7|189.972|190.411 ± 1.176|188.933–192.034|0.62|189.433–191.603|-0.13|PASS|
|yolo26|managed-non-stream|7|172.811|171.618 ± 3.481|167.064–175.019|2.03|167.971–174.884|10.67|PASS|
|yolo26|managed-stream|7|166.352|166.582 ± 2.281|163.048–169.090|1.37|164.628–168.723|13.79|PASS|
|yolo-world|native-non-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo-world|native-stream|N/A|N/A|N/A|N/A|N/A|N/A|N/A|Native video batch is locked to 1 by the public API|
|yolo-world|raw-non-stream|7|187.125|185.714 ± 4.003|178.955–189.743|2.16|181.746–189.033|0.00|PASS|
|yolo-world|raw-stream|7|186.631|185.928 ± 2.843|180.641–188.348|1.53|183.728–188.270|0.61|PASS|
|yolo-world|managed-non-stream|7|171.965|171.158 ± 2.246|168.400–173.431|1.31|168.561–173.212|8.18|PASS|
|yolo-world|managed-stream|7|164.539|166.290 ± 3.797|161.876–171.792|2.28|163.672–170.820|10.82|PASS|

| Model | Mode | Median pass seconds | Host first delivery median ms | Ready pack P50/P95/P99 ms | Peak RSS MiB | Allocated/reserved VRAM MiB |
|---|---|---:|---:|---:|---:|---:|
|yolo|raw-non-stream|0.529207|529.123|539.723/539.723/539.723|2531.07|93.11/150.00|
|yolo|raw-stream|0.538639|56.708|540.908/540.908/540.908|2472.11|93.11/150.00|
|yolo|managed-non-stream|0.602745|602.649|605.053/605.053/605.053|2470.70|64.78/150.00|
|yolo|managed-stream|0.622576|97.155|626.143/626.143/626.143|2501.57|64.78/150.00|
|yolo11|raw-non-stream|0.606054|605.971|590.322/590.322/590.322|2794.89|126.08/196.00|
|yolo11|raw-stream|0.609584|63.772|585.379/585.379/585.379|2536.41|126.08/196.00|
|yolo11|managed-non-stream|0.668097|668.001|660.827/660.827/660.827|2527.36|99.43/196.00|
|yolo11|managed-stream|0.690258|101.336|678.860/678.860/678.860|2527.08|99.43/196.00|
|yolo26|raw-non-stream|0.631206|631.121|630.059/630.059/630.059|2520.02|120.45/186.00|
|yolo26|raw-stream|0.631674|60.159|621.581/621.581/621.581|2515.19|120.45/186.00|
|yolo26|managed-non-stream|0.694399|694.308|696.668/696.668/696.668|2510.51|97.52/186.00|
|yolo26|managed-stream|0.721360|100.602|720.572/720.572/720.572|2510.46|97.52/186.00|
|yolo-world|raw-non-stream|0.641283|641.199|646.205/646.205/646.205|2693.27|209.49/292.00|
|yolo-world|raw-stream|0.642982|61.954|639.359/639.359/639.359|2686.75|209.49/292.00|
|yolo-world|managed-non-stream|0.697817|697.717|712.165/712.165/712.165|2686.76|188.26/292.00|
|yolo-world|managed-stream|0.729309|100.512|727.109/727.109/727.109|2686.62|188.26/292.00|

## Interpretation limits

- Image packs exclude decode; video includes decode. Batch settings, model precision and input digests must match for a direct comparison.
- Allocated/reserved peaks include the timed runner retention policy and allocator history. They are not minimum required VRAM.
- A pass shorter than one second is especially sensitive to scheduling and clock variation. Extend its workload for a separate qualification run; preserve the current samples.
- Diagnostic pack quantiles describe a small sample of API calls, not a frame-latency SLA. In non-stream modes the first delivery includes the complete call.
- Bootstrap intervals are descriptive within this run. Use additional independent sessions and diverse fixed videos before generalizing.
