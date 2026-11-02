# Klygo Outputs CPU/GPU Benchmark

## Methodology

- Storage backends: NumPy CPU and CuPy CUDA (`cuda:0`).
- NMS measurements: median of five timed runs after warmup.
- Storage-operation measurements: median of three timed runs after CuPy kernel warmup.
- Detection integrity: CPU and GPU retained identical UIDs and box counts.

## NMS throughput

| Distribution | Boxes | Kept | NumPy CPU (ms) | CuPy GPU (ms) | Interpretation |
|---|---:|---:|---:|---:|---|
| Low overlap | 100 | 25 | 1.419 | 2.851 | CPU 2.0x faster |
| Low overlap | 500 | 125 | 8.900 | 3.954 | GPU 2.3x faster |
| Low overlap | 1,000 | 250 | 20.655 | 6.086 | GPU 3.4x faster |
| Low overlap | 2,000 | 500 | 52.648 | 12.280 | GPU 4.3x faster |
| High overlap | 100 | 4 | 0.358 | 3.053 | CPU 8.5x faster |
| High overlap | 500 | 20 | 1.555 | 3.353 | CPU 2.2x faster |
| High overlap | 1,000 | 41 | 3.470 | 4.505 | CPU 1.3x faster |
| High overlap | 2,000 | 80 | 9.500 | 7.257 | GPU 1.3x faster |

## Storage and simple operations

| Boxes | CPU to GPU (ms) | GPU to CPU (ms) | CPU float16 (ms) | GPU float16 (ms) | CPU threshold (ms) | GPU threshold (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.704 | 1.306 | 0.060 | 1.038 | 0.151 | 2.280 |
| 1,000 | 0.776 | 1.646 | 0.104 | 1.105 | 0.579 | 3.923 |
| 2,000 | 0.864 | 1.943 | 0.125 | 1.265 | 0.850 | 3.290 |

## Conclusions

- Keep small collections and simple filters on CPU.
- GPU NMS becomes useful around 500 to 1,000 low-overlap boxes and around 2,000 high-overlap boxes.
- Avoid a CPU-to-GPU transfer for only one simple threshold operation.
- Keep `Detections` on GPU when several heavy postprocessing operations follow inference.
- CPU/GPU round trips preserve boxes, labels, class IDs, UIDs, and float16/32/64 storage.

## Raw metrics

- [`metrics/nms.json`](metrics/nms.json)
- [`metrics/operations.json`](metrics/operations.json)
