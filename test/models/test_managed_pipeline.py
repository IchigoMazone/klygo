"""Ordered video preparation, bounded read-ahead, and array output parity."""

from threading import Event, current_thread
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from klygo import files, media, models, preprocessing
from klygo.models.detection import base


class TestManagedPipeline(unittest.TestCase):
    def test_default_prefetch_prepares_next_batch_during_inference(self):
        second_started = Event()

        def decode(image):
            if image == 1:
                second_started.set()
            return image

        with patch.object(base, "_materialize_model_image", side_effect=decode):
            batches = base._prepared_batches(range(3), 1, 1, 1)
            try:
                self.assertEqual(next(batches), ([0], [0]))
                self.assertTrue(second_started.wait(2), "next batch did not overlap inference")
                self.assertEqual(list(batches), [([1], [1]), ([2], [2])])
            finally:
                batches.close()

    def test_default_worker_keeps_source_iteration_on_the_caller_and_closes(self):
        for workers, prefetch in ((0, 0), (1, 0), (1, 1)):
            with self.subTest(workers=workers, prefetch=prefetch):
                caller = current_thread().name
                seen = []
                closed = Event()

                def source():
                    try:
                        for index in range(20):
                            seen.append((index, current_thread().name))
                            yield index
                    finally:
                        closed.set()

                with patch.object(base, "_materialize_model_image", side_effect=lambda image: image):
                    batches = base._prepared_batches(source(), 1, workers, prefetch)
                    self.assertEqual(next(batches), ([0], [0]))
                    batches.close()
                self.assertTrue(closed.is_set())
                self.assertTrue(all(thread == caller for _, thread in seen))
                self.assertLessEqual(len(seen), 2 if prefetch else 1)

    def test_default_worker_propagates_decode_failure_and_closes_source(self):
        closed = Event()

        def source():
            try:
                yield from range(20)
            finally:
                closed.set()

        def decode(image):
            if image == 1:
                raise ValueError("decode failed")
            return image

        with patch.object(base, "_materialize_model_image", side_effect=decode):
            with self.assertRaisesRegex(ValueError, "decode failed"):
                list(base._prepared_batches(source(), 1, 1, 1))
        self.assertTrue(closed.is_set())

    def test_single_batch_still_uses_parallel_workers(self):
        rendezvous = Event()

        def decode(image):
            if image == 1:
                rendezvous.set()
            elif not rendezvous.wait(2):
                raise AssertionError("single batch was decoded serially")
            return image

        with patch.object(base, "_materialize_model_image", side_effect=decode):
            self.assertEqual(list(base._prepared_batches([0, 1], 2, 2, 1)), [([0, 1], [0, 1])])

    def test_prefetch_bounds_source_consumption_and_closes_iterator(self):
        consumed = []
        closed = Event()

        def source():
            try:
                for index in range(1000):
                    consumed.append(index)
                    yield index
            finally:
                closed.set()

        with patch.object(base, "_materialize_model_image", side_effect=lambda image: image):
            batches = base._prepared_batches(source(), 2, 4, 2)
            self.assertEqual(next(batches), ([0, 1], [0, 1]))
            batches.close()
        self.assertLessEqual(len(consumed), 6)
        self.assertTrue(closed.is_set())

    def test_decode_and_source_errors_propagate_and_close(self):
        for source_failure in (False, True):
            closed = Event()

            def source():
                try:
                    yield 0
                    if source_failure:
                        raise ValueError("source failed")
                    yield 1
                finally:
                    closed.set()

            def decode(image):
                if image == 1:
                    raise ValueError("decode failed")
                return image

            with self.subTest(source_failure=source_failure), patch.object(
                base, "_materialize_model_image", side_effect=decode
            ):
                with self.assertRaisesRegex(ValueError, "failed"):
                    list(base._prepared_batches(source(), 1, 2, 1))
                self.assertTrue(closed.is_set())

    def test_video_reads_are_ordered_and_transforms_preserve_trace(self):
        with tempfile.TemporaryDirectory() as root:
            path = files.join(root, "video.mp4")
            reads = []

            class Reader:
                width, height = 20, 12

                def read_frame(self, index):
                    reads.append((index, current_thread().name))
                    return np.full((12, 20, 3), index * 10, dtype=np.uint8)

            reader = Reader()
            originals = [media.LazyImage(path, backend="opencv", frame_index=i, reader=reader) for i in range(8)]
            images = [preprocessing.resize((10, 6))(image) for image in originals]
            batches = list(base._prepared_batches(images, 3, 4, 2, backend="Ultralytics"))
            self.assertEqual([index for index, _ in reads], list(range(8)))
            self.assertTrue(all(name.startswith("klygo-source") for _, name in reads))
            prepared = [image for _, part in batches for image in part]
            self.assertEqual([image.shape for image in prepared], [(6, 10, 3)] * 8)
            self.assertEqual([int(image[0, 0, 0]) for image in prepared], [i * 10 for i in range(8)])
            self.assertTrue(all(image.trace for image in images))
            self.assertTrue(all(not image.is_loaded for image in images))

    def test_video_preparation_keeps_existing_pixel_cache(self):
        image = media.LazyImage("unused.mp4", backend="opencv", frame_index=0)
        pixels = np.zeros((12, 20, 3), dtype=np.uint8)
        image._image = pixels
        batches = list(base._prepared_batches([image], 1, 2, 1, backend="Ultralytics"))
        self.assertIs(image._image, pixels)
        self.assertIs(batches[0][1][0], pixels)

    def test_video_reader_advances_nearby_samples_without_seeking(self):
        class Capture:
            def __init__(self):
                self.position = 0
                self.seeks = []

            def isOpened(self):
                return True

            def grab(self):
                self.position += 1
                return True

            def read(self):
                value = self.position
                self.position += 1
                return True, value

            def set(self, prop, value):
                self.seeks.append(value)
                self.position = value

            def release(self):
                pass

        from threading import Lock

        reader = media.VideoReader.__new__(media.VideoReader)
        reader._cap = Capture()
        reader._current_pos = 0
        reader._lock = Lock()
        reader.total_frames = 100
        self.addCleanup(reader.close)
        self.assertEqual([reader.read_frame(i) for i in (0, 3, 6)], [0, 3, 6])
        self.assertEqual(reader._cap.seeks, [])
        self.assertEqual(reader.read_frame(1), 1)
        self.assertEqual(reader._cap.seeks, [1])
        self.assertEqual(reader.read_frame(80), 80)
        self.assertEqual(reader._cap.seeks, [1, 80])
        with patch.object(reader._cap, "grab", return_value=False):
            self.assertEqual(reader.read_frame(83), 83)
        self.assertEqual(reader._cap.seeks, [1, 80, 83])

    def test_real_video_matches_synchronous_predictions_and_restoration(self):
        class Detector(models.Detector):
            def __init__(self):
                super().__init__(backend="Custom", model_id="video-test")
                self.names = {0: "object"}

            def forward(self, images, prompt=None):
                return [{"boxes": [[1, 1, 4, 4]], "scores": [.9],
                         "class_ids": [0], "labels": ["object"]} for _ in images]

        with tempfile.TemporaryDirectory() as root:
            path = files.join(root, "frames.mp4")
            media.save_video(path, [np.full((12, 20, 3), i * 10, dtype=np.uint8) for i in range(8)],
                             fps=8, overwrite=True, verbose=False)
            for stream in (False, True):
                actual = []
                for workers in (0, 4):
                    source = media.load(path, stream=stream, indexed=not stream, verbose=False)
                    source = source.transform(preprocessing.resize((10, 6))).sample(2)
                    try:
                        results = list(Detector().predict(source, stream=stream, batch=3,
                                                          workers=workers, prefetch=2, verbose=False))
                        actual.append([(frame.frame_index, frame.image_size, frame._xyxy.tolist()) for frame in results])
                        self.assertTrue(all(not frame._source.is_loaded for frame in results))
                    finally:
                        source.close()
                self.assertEqual(actual[0], actual[1])
                self.assertEqual([item[0] for item in actual[1]], [0, 2, 4, 6])
                self.assertTrue(all(item[1] == (20, 12) for item in actual[1]))

    def test_pil_inputs_match_ultralytics_bgr_conversion(self):
        pixels = np.zeros((6, 10, 3), dtype=np.uint8)
        pixels[:, :, 0] = 123
        image = Image.fromarray(pixels)
        prepared = base._materialize_model_image(image, backend="Ultralytics")
        np.testing.assert_array_equal(prepared, pixels[:, :, ::-1])
        self.assertTrue(prepared.flags.c_contiguous)
        np.testing.assert_array_equal(np.asarray(image), pixels)

    def test_native_array_payload_uses_fast_builder_and_owns_results(self):
        detector = models.Detector(backend="Custom", model_id="array-test")
        detector.names = {0: "object"}
        boxes = np.array([[1, 2, 8, 9]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        ids = np.array([0], dtype=np.int32)
        from klygo.outputs import detect

        with patch.object(detect, "_coords", side_effect=AssertionError("slow box path")):
            frame = detector._pack_detection(
                {"boxes": boxes, "scores": scores, "labels": ["object"], "class_ids": ids},
                Image.new("RGB", (20, 20)), 0, 1.0, 1000.0,
            )
        self.assertEqual(frame.labels, ["object"])
        boxes[:] = 0
        scores[:] = 0
        np.testing.assert_array_equal(frame._xyxy, [[1, 2, 8, 9]])
        self.assertGreater(frame.scores[0], 0.8)

    def test_native_yolo_does_not_forward_loading_weights(self):
        detector = models.Detector(backend="Ultralytics", model_id="yolov8n.pt")
        detector.names = {0: "object"}
        options = detector._native_runtime_options(
            {"model": {"weights": "checkpoint.pt", "imgsz": 320}, "post": {}}, None
        )
        self.assertNotIn("weights", options)
        self.assertEqual(options["imgsz"], 320)


if __name__ == "__main__":
    unittest.main()
