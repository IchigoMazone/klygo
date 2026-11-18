# Klygo CPU & Disk I/O Loader Benchmark Report

- **CPU**: `Intel(R) Xeon(R) CPU @ 2.20GHz (x86_64)` (12 cores)
- **Dataset**: `custom:voc2007_val_yolo`
- **OS / Python**: `Linux-6.6.159+-x86_64-with-glibc2.39` / Python `3.13.16`

## Operations

| operation | median_ms | status |
| --- | --- | --- |
| datasets.load (custom:voc2007_val_yolo, 2510 imgs) | 3136.556 | passed |
| media.load_eager (custom:voc2007_val_yolo, 2510 imgs) | 158.6343 | passed |
| media.load_stream (custom:voc2007_val_yolo, 2510 imgs) | 163.0562 | passed |
| preprocessing.compose (custom:voc2007_val_yolo, 32 imgs) | 0.2425 | passed |
| datasets.load_scaled (2510 imgs) | 3065.082 | passed |
| media.load_indexed (2510 imgs) | 93.2839 | passed |
| media.strided_decode (stride=32, 78 imgs) | 274.3223 | passed |
| media.random_jump_array (64/2510 imgs) | 137.3797 | passed |
| media.random_jump_pil (64/2510 imgs) | 129.1266 | passed |
| detections.random_jump_index (64/2510) | 0.0117 | passed |
| detections.random_jump_image_read (64/2510) | 13.9708 | passed |
| media.video_load_indexed (120 frames) | 4.4342 | passed |
| media.video_random_jump_array (32/120) | 976.1081 | passed |
| media.video_random_jump_pil (32/120) | 986.9421 | passed |
| detections.export_yolo (500 imgs) | 852.789 | passed |
| datasets.partition_yolo (500 imgs) | 889.7731 | passed |
| archive.compress_zip (500 imgs) | 654.4041 | passed |
| archive.extract_zip (500 imgs) | 302.2808 | passed |
| files.save_json (2510 frames) | 230.7103 | passed |
| files.load_json (2510 frames) | 46.1741 | passed |
| media.info | 0.1125 | passed |
| media.save_images_letterbox | 50.5194 | passed |
| visual.compare_images | 91.2456 | passed |
| utils.tabulate | 7.5743 | passed |
