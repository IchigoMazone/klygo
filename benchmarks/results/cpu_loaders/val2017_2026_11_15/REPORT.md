# CPU Loaders, Preprocessing & Random-Jump Access (val2017_2026_11_15)

- **GPU**: `NVIDIA L4` (22563.1 MB VRAM)
- **CUDA / CuPy**: `torch_cuda=True`, `cupy_cuda=True`, `cuda_runtime=13.0`
- **Git**: `` @ `00ba3cf09b3bc9a561ec0fdb58bf10d924bdc620`

## Loaders & Random-Jump Image Reads

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
