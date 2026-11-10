# Reduce image-copy and YOLO video preparation overhead

## Changes

- RGB input normalization retains the independent copy returned by media.to_pil; it skips the second, identical RGB conversion. Non-RGB inputs still convert to RGB.
- Plain OpenCV video with Ultralytics, workers=1 and batch greater than one is read and materialized on the existing ordered source producer. No unused decode executor or per-frame completed Futures are allocated. Source read-ahead stays bounded and decoding stops on stream close.
- Image pipelines, explicit parallel workers and sources with collection transforms/filters keep their existing overlapping source/decode stages. The batch-1 prefetch path is unchanged.

## Validation

68 focused model/pipeline tests passed, including input independence, non-RGB conversion, ordered video reads, decode failures, bounded prefetch, early close, prompt isolation and backend dispatch. Full discovery completed 567 tests with 15 skips; all 13 public-module audits passed. The final focused run also covers preserving separate source/conversion workers for PIL video inputs.

To replay the archived driver from a newer checkout, pin its baseline `git show HEAD:` reads to `5ed34c82fa6db1f700d0e0a591e69e9259538640:` and set the local input/weight paths.

## Local diagnostic measurements

Quadro M1200 (4 GiB), CUDA execution verified from AutoBackend native model parameters, imgsz=640, YOLOv8n and YOLO26n. Eight 640x480 decoded RGB images and an eight-frame local H.264 sample; input and weight hashes and versions are in [raw_comparison.json](raw_comparison.json). The local baseline commit has the same complete tree as published commit 5ed34c8.

Two warmup calls per version and case precede seven alternating timed repeats. CUDA is synchronized around each pass. Parity is checked after timing; boxes, scores, labels and frame order pass for all 84 timed passes (672 frames). The [archived executed driver](executed_driver.txt) preserves the exact local paths and measurement procedure; it is diagnostic evidence, not a portable application example.

### Source preparation only

Seven samples per version, ten calls per sample. Values are median ms per call; model execution is excluded. Plain image-batch preparation remains a control because its parallel pipeline was deliberately retained.

| Stage | Before ms | After ms | Elapsed reduction % |
|---|---:|---:|---:|
|rgb_resolution|9.724|4.823|50.40|
|batch4_preparation|33.303|33.250|0.16|
|single_stream_pack|14.066|13.961|0.74|

### Complete managed-stream passes

| Model | Input | Batch | Before median FPS | After median FPS | FPS change % | Before/after elapsed CV % |
|---|---|---:|---:|---:|---:|---:|
|yolov8n.pt|images|4|10.326|10.352|0.24|0.35/0.27|
|yolov8n.pt|video|1|11.249|11.245|-0.03|0.15/0.28|
|yolov8n.pt|video|4|10.077|10.092|0.15|0.37/0.26|
|yolo26n.pt|images|4|12.794|12.875|0.63|0.33/0.57|
|yolo26n.pt|video|1|13.692|13.683|-0.06|0.21/0.21|
|yolo26n.pt|video|4|12.448|12.470|0.18|0.15/0.25|

RGB normalization takes about half the elapsed time in the isolated test. Complete GPU passes remain close to the baseline; these local results do not establish a meaningful general FPS speedup. The video batch-1 rows act as controls for normal run variation because their prefetch implementation is unchanged.

This is not the Tesla T4 suite and must not replace or be pooled with its results. It does not requalify long-running memory behavior, camera latency, or other accelerators. Use the existing six-mode GPU suite on the target device before making deployment performance claims.
