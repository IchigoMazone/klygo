# Klygo CPU & Disk I/O Loader Benchmark Report

- **CPU**: `Intel(R) Xeon(R) CPU @ 2.20GHz (x86_64)` (12 cores)
- **Dataset**: `coco128`
- **OS / Python**: `Linux-6.6.159+-x86_64-with-glibc2.39` / Python `3.13.16`

## Operations

| operation | median_ms | status |
| --- | --- | --- |
| datasets.load (coco128, 128 imgs) | 198.1545 | passed |
| media.load_eager (coco128, 128 imgs) | 9.7087 | passed |
| media.load_stream (coco128, 128 imgs) | 10.0138 | passed |
| preprocessing.compose (coco128, 32 imgs) | 0.2414 | passed |
| datasets.load_scaled (2048 imgs) | 3007.3202 | passed |
| media.load_indexed (2048 imgs) | 76.2247 | passed |
| media.strided_decode (stride=32, 64 imgs) | 221.3596 | passed |
| media.random_jump_numpy (64/2048 imgs) | 138.7952 | passed |
| media.random_jump_pil (64/2048 imgs) | 129.3331 | passed |
| detections.random_jump_index (64/2048) | 0.0116 | passed |
| detections.random_jump_image_read (64/2048) | 22.5358 | passed |
| media.video_load_indexed (120 frames) | 5.4768 | passed |
| media.video_random_jump_array (32/120) | 1358.3255 | passed |
| media.video_random_jump_pil (32/120) | 1348.0065 | passed |
| detections.export_yolo (128 imgs) | 293.3843 | passed |
| datasets.partition_yolo (128 imgs) | 241.6689 | passed |
| archive.compress_zip (128 imgs) | 255.0917 | passed |
| archive.extract_zip (128 imgs) | 87.0302 | passed |
| files.save_json (128 frames) | 23.443 | passed |
| files.load_json (128 frames) | 4.3368 | passed |
| media.info | 0.1349 | passed |
| media.save_images_letterbox | 62.0123 | passed |
| visual.compare_images | 135.7267 | passed |
| utils.tabulate | 3.7697 | passed |
