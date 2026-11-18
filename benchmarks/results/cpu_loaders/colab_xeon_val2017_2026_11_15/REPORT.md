# Klygo CPU & Disk I/O Loader Benchmark Report

- **CPU**: `Intel(R) Xeon(R) CPU @ 2.20GHz (x86_64)` (12 cores)
- **Dataset**: `custom:coco_val2017_yolo`
- **OS / Python**: `Linux-6.6.159+-x86_64-with-glibc2.39` / Python `3.13.16`

## Operations

| operation | median_ms | status |
| --- | --- | --- |
| datasets.load (custom:/content/coco_val2017_yolo, 5000 imgs) | 7810.8282 | passed |
| media.load_eager (custom:/content/coco_val2017_yolo, 5000 imgs) | 326.7664 | passed |
| media.load_stream (custom:/content/coco_val2017_yolo, 5000 imgs) | 340.6544 | passed |
| preprocessing.compose (custom:/content/coco_val2017_yolo, 32 imgs) | 0.2416 | passed |
| datasets.load_scaled (5000 imgs) | 7922.0788 | passed |
| media.load_indexed (5000 imgs) | 192.2478 | passed |
| media.strided_decode (stride=32, 156 imgs) | 790.8842 | passed |
| media.random_jump_numpy (64/5000 imgs) | 251.2747 | passed |
| media.random_jump_pil (64/5000 imgs) | 239.82 | passed |
| detections.random_jump_index (64/5000) | 0.0113 | passed |
| detections.random_jump_image_read (64/5000) | 22.0376 | passed |
| media.video_load_indexed (120 frames) | 4.4794 | passed |
| media.video_random_jump_array (32/120) | 1027.2384 | passed |
| media.video_random_jump_pil (32/120) | 1023.5332 | passed |
| detections.export_yolo (500 imgs) | 1096.9918 | passed |
| datasets.partition_yolo (500 imgs) | 908.1938 | passed |
| archive.compress_zip (500 imgs) | 946.5946 | passed |
| archive.extract_zip (500 imgs) | 337.3182 | passed |
| files.save_json (5000 frames) | 913.8758 | passed |
| files.load_json (5000 frames) | 191.4489 | passed |
| media.info | 0.116 | passed |
| media.save_images_letterbox | 81.2563 | passed |
| visual.compare_images | 150.0956 | passed |
| utils.tabulate | 3.7029 | passed |
