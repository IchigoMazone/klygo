"""Collection-wide NumPy and CuPy detection storage tests."""

import os
import tempfile
import unittest

import numpy as np

from klygo import files, outputs, postprocessing


os.environ.setdefault(
    "CUPY_CACHE_DIR",
    str(files.join(tempfile.gettempdir(), "klygo-cupy-cache")),
)

try:
    import cupy as cp

    _CUDA_READY = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, OSError, RuntimeError):
    cp = None
    _CUDA_READY = False


def detection(source_id="gpu", frame_index=0):
    """Create one deterministic CPU frame."""
    return outputs.Detection(
        [
            outputs.Box([0, 0, 10, 10], label="car", score=0.9),
            outputs.Box([1, 1, 9, 9], label="car", score=0.8),
            outputs.Box([12, 12, 18, 18], label="person", score=0.2),
        ],
        source_id=source_id,
        frame_index=frame_index,
        image_size=(20, 20),
    )


@unittest.skipUnless(_CUDA_READY, "CuPy CUDA runtime is unavailable")
class TestGpuStorage(unittest.TestCase):
    def test_packed_storage_converts_base_arrays_without_materializing_frames(self):
        results = outputs.Detections(
            [detection("packed", index) for index in range(3)],
            packed=True,
        )
        live_frame = results[0]
        results.cuda().astype(16)
        self.assertTrue(results.is_packed)
        self.assertEqual(len(results._packed_store.cache), 1)
        self.assertIsInstance(results._packed_store.xyxy, cp.ndarray)
        self.assertEqual(results._packed_store.xyxy.dtype, cp.float16)
        self.assertEqual((live_frame.device, live_frame.dtype), ("cuda:0", 16))
        self.assertIsInstance(live_frame._xyxy, cp.ndarray)

        postprocessing.nms(0.5)(results)
        self.assertEqual([len(frame) for frame in results], [2, 2, 2])
        self.assertEqual(len(results._packed_store.overrides), 3)
        results.cpu()
        self.assertTrue(results.is_packed)
        self.assertIsInstance(results._packed_store.xyxy, np.ndarray)
        self.assertEqual(live_frame.device, "cpu")

    def test_collection_owns_device_and_float_precision(self):
        results = outputs.Detections([detection()])
        frame = results[0]
        cached = frame[0]

        self.assertIs(results.cuda().astype(16), results)
        self.assertEqual((results.device, results.dtype), ("cuda:0", 16))
        self.assertEqual((frame.device, frame.dtype), ("cuda:0", 16))
        self.assertIsInstance(frame._xyxy, cp.ndarray)
        self.assertEqual(frame._xyxy.dtype, cp.float16)
        self.assertEqual(frame._scores.dtype, cp.float16)
        self.assertEqual(frame._class_ids.dtype, cp.int32)
        self.assertEqual(frame._uids.dtype, cp.int64)
        self.assertFalse(hasattr(frame, "astype"))
        self.assertIsNone(cached._owner)

        subset = frame[[1, 0]]
        copied = results.copy()
        self.assertEqual((subset.device, subset.dtype), ("cuda:0", 16))
        self.assertEqual((copied.device, copied.dtype), ("cuda:0", 16))
        self.assertIsInstance(subset._xyxy, cp.ndarray)
        self.assertIsInstance(copied[0]._xyxy, cp.ndarray)

        self.assertIs(results.cpu(), results)
        self.assertEqual(results.device, "cpu")
        self.assertIsInstance(frame._xyxy, np.ndarray)
        self.assertEqual(frame._xyxy.dtype, np.dtype("float16"))
        self.assertEqual(frame._class_ids.dtype, np.dtype("int32"))
        self.assertEqual(frame._uids.dtype, np.dtype("int64"))

    def test_unavailable_cuda_device_does_not_mutate_collection(self):
        results = outputs.Detections([detection()])
        with self.assertRaises(ValueError):
            results.cuda(999)
        self.assertEqual((results.device, results.dtype), ("cpu", 64))
        self.assertIsInstance(results[0]._xyxy, np.ndarray)

    def test_gpu_postprocessing_and_incoming_frame_normalization(self):
        results = outputs.Detections([detection()]).cuda().astype(32)
        results.append(detection("gpu", 1))
        self.assertTrue(all(frame.device == "cuda:0" for frame in results))
        self.assertTrue(all(frame.dtype == 32 for frame in results))

        postprocessing.compose(
            postprocessing.clip(),
            postprocessing.threshold(0.5),
            postprocessing.nms(0.5),
            postprocessing.top(1),
        )(results)
        self.assertEqual([len(frame) for frame in results], [1, 1])
        self.assertEqual([frame.labels for frame in results], [["car"], ["car"]])
        results.cpu()
        self.assertEqual(results[0].boxes, [(0.0, 0.0, 10.0, 10.0)])

    def test_gpu_serialization_stages_without_changing_storage(self):
        results = outputs.Detections([detection()]).cuda().astype(32)
        payload = results.to_dict()
        self.assertEqual(payload["frames"][0]["boxes"][0]["label"], "car")
        with tempfile.TemporaryDirectory() as root:
            results.export(files.join(root, "labels"), format="yolo")
            self.assertTrue(files.exists(files.join(root, "labels", "data.yaml")))
        self.assertEqual((results.device, results.dtype), ("cuda:0", 32))
        self.assertIsInstance(results[0]._xyxy, cp.ndarray)

    def test_large_gpu_nms_stays_in_cupy(self):
        boxes = [
            outputs.Box(
                [index * 2, 0, index * 2 + 1, 1],
                label="object",
                score=1.0 - index / 1000,
            )
            for index in range(600)
        ]
        results = outputs.Detections([
            outputs.Detection(
                boxes,
                source_id="large-nms",
                image_size=(1200, 2),
            )
        ]).cuda().astype(32)
        postprocessing.nms(0.5)(results)
        self.assertEqual(len(results[0]), 600)
        self.assertIsInstance(results[0]._xyxy, cp.ndarray)

    def test_gpu_nms_matches_numpy_for_all_precisions_and_modes(self):
        rng = np.random.default_rng(7)
        boxes = []
        for index in range(300):
            left, top = rng.uniform(0, 400, size=2)
            width, height = rng.uniform(5, 100, size=2)
            boxes.append(
                outputs.Box(
                    [left, top, left + width, top + height],
                    label="car" if index % 2 else "person",
                    score=1.0 - index / 301,
                )
            )
        original = outputs.Detections([
            outputs.Detection(boxes, source_id="parity", image_size=(500, 500))
        ])

        for precision in (16, 32, 64):
            for class_agnostic in (False, True):
                cpu = original.copy().astype(precision)
                gpu = original.copy().astype(precision).cuda()
                postprocessing.nms(0.45, class_agnostic=class_agnostic)(cpu)
                postprocessing.nms(0.45, class_agnostic=class_agnostic)(gpu)
                gpu.cpu()
                self.assertEqual(cpu[0].uids, gpu[0].uids)

    def test_stream_storage_conversion_is_lazy(self):
        consumed = []

        def frames():
            consumed.append(True)
            yield detection("stream", 0)

        results = outputs.Detections(frames(), stream=True)
        results.cuda().astype(16)
        self.assertEqual(consumed, [])
        output = list(results)
        self.assertEqual(consumed, [True])
        self.assertEqual((output[0].device, output[0].dtype), ("cuda:0", 16))
        self.assertIsInstance(output[0]._xyxy, cp.ndarray)

    def test_failed_precision_conversion_is_atomic(self):
        normal = detection("atomic", 0)
        huge = outputs.Detection(
            [outputs.Box([0, 0, 1e39, 10], label="car")],
            source_id="atomic",
            frame_index=1,
            image_size=(20, 20),
        )
        results = outputs.Detections([normal, huge])
        with self.assertRaises(OverflowError):
            results.cuda().astype(16)
        self.assertEqual(results.dtype, 64)
        self.assertEqual([frame.dtype for frame in results], [64, 64])


if __name__ == "__main__":
    unittest.main()
