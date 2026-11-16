# CPU Loaders, Preprocessing & Random-Jump Access (coco128_2026_11_15)

- **GPU**: `NVIDIA L4` (22563.1 MB VRAM)
- **CUDA / CuPy**: `torch_cuda=True`, `cupy_cuda=True`, `cuda_runtime=13.0`
- **Git**: `detection-evaluation` @ `ae0eaaaa668446cb4f1d62c30472c6028e3673b3`

## Loaders & Random-Jump Image Reads

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
