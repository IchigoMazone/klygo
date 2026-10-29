"""Remembered inference, immutable provenance, streaming and benchmark isolation."""

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from threading import current_thread
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np

from klygo import files, models, outputs
from klygo.models.detection.yolo import YOLODetect
from klygo.models.detection.yolo_world import YOLOWorldDetect


def definition(backend="Custom"):
    return {"model_id": "test-model", "backend": backend,
            "flags": {"model": False, "post": True},
            "priority": {"post": ["threshold", "iou"]},
            "model": {}, "post": {"threshold": .25, "iou": .5}}


class RecordingDetector(models.Detector):
    def __init__(self):
        super().__init__(definition())
        self.calls = []
        self.fail_at = None

    def forward(self, images, prompt=None):
        self.calls.append((deepcopy(prompt), deepcopy(dict(self.metadata))))
        if self.fail_at == len(self.calls):
            raise RuntimeError("intentional inference failure")
        labels = prompt or ["default"]
        self.names = dict(enumerate(labels))
        return outputs.build_detections(images, [
            {"boxes": [[1, 2, 4, 5]], "scores": [.8], "labels": [labels[0]], "class_ids": [0]}
            for _ in images], names=self.names)


class FakeYOLO:
    def __init__(self, model_id):
        self.names = {0: "person", 1: "car"}
        self.changes = []
        self.options = []
        self.fail = False

    def set_classes(self, names):
        self.changes.append(list(names))
        self.names = dict(enumerate(names))

    def __call__(self, images, **kwargs):
        self.options.append(deepcopy(kwargs))
        if self.fail:
            raise RuntimeError("native inference failed")
        return [SimpleNamespace(boxes=[], names=deepcopy(self.names)) for _ in images]

    def predict(self, source, stream=False, **kwargs):
        self.options.append(deepcopy(kwargs))
        if self.fail:
            raise RuntimeError("native inference failed")
        sources = list(source) if isinstance(source, (list, tuple)) else [source]

        def results():
            for image in sources:
                array = np.asarray(image) if isinstance(image, Image.Image) else np.zeros((12, 12, 3), dtype=np.uint8)
                yield SimpleNamespace(
                    boxes=[],
                    names=deepcopy(self.names),
                    orig_img=array,
                    path=None,
                    speed={"inference": 2.0},
                )

        return results() if stream else list(results())


class TestPredictionLifecycle(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (20, 20))
        self.model = RecordingDetector()

    def test_remember_clear_reset_and_callable(self):
        first = self.model(self.image, prompt=["dog"], threshold=.5, verbose=False)
        second = self.model.predict(self.image, verbose=False)
        self.assertEqual(second[0].metadata["prompt"], ["dog"])
        self.assertEqual(second[0].metadata["threshold"], .5)
        self.model(self.image, threshold=.7, verbose=False)
        self.assertEqual(self.model.predict_config["prompt"], ["dog"])
        self.assertEqual(self.model.predict_config["threshold"], .7)
        self.assertEqual(first[0].metadata["threshold"], .5)
        self.model(self.image, prompt=None, verbose=False)
        self.assertIsNone(self.model.predict_config["prompt"])
        self.model.reset()
        self.assertEqual(self.model.predict_config, {"prompt": None, "threshold": .25, "iou": .5})

    def test_nested_snapshot_and_execution_controls(self):
        self.model(self.image, prompt=["dog"], post_options={"values": [1]}, verbose=False,
                   batch=2, workers=1, prefetch=1, vid_stride=2, max_frames=1)
        saved = self.model.predict_config
        saved["prompt"].append("cat")
        saved["post_options"]["values"].append(2)
        self.assertEqual(self.model.predict_config["post_options"], {"values": [1]})
        self.assertEqual(self.model.predict_config["prompt"], ["dog"])
        for key in ("source", "verbose", "batch", "workers", "prefetch",
                    "vid_stride", "max_frames", "stream"):
            self.assertNotIn(key, self.model.predict_config)

    def test_array_configuration_comparison_is_unambiguous(self):
        self.model.settings["post"]["anchors"] = np.array([1, 2])
        self.model(
            self.image,
            post_anchors=np.array([1, 2]),
            verbose=False,
        )
        np.testing.assert_array_equal(
            self.model.predict_config["post_anchors"],
            np.array([1, 2]),
        )

    def test_validation_and_failure_leave_state_unchanged(self):
        self.model(self.image, prompt="dog", threshold=.6, verbose=False)
        before = self.model.predict_config
        for kwargs in ({"prompt": ["cat"], "threshold": 2}, {"prompt": [3]}, {"prompt": []},
                       {"prompt": " "}, {"prompt": True}, {"threshold": float("nan")},
                       {"iou": -.1}, {"threshold": True}, {"batch": 0}, {"batch": True},
                       {"workers": -1}, {"workers": True}, {"prefetch": -1}, {"prefetch": True},
                       {"vid_stride": -1}, {"max_frames": 0}, {"stream": 1}, {"verbose": 1},
                       {"unknown": 4}, {"post_verbose": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                self.model(self.image, **kwargs)
            self.assertEqual(self.model.predict_config, before)
        self.model.fail_at = len(self.model.calls) + 1
        with self.assertRaises(RuntimeError):
            self.model(self.image, prompt="cat", threshold=.9)
        self.assertEqual(self.model.predict_config, before)

    def test_stream_captures_snapshot_at_creation(self):
        calls_before = len(self.model.calls)
        stream = self.model([self.image] * 3, prompt="old", threshold=.4, stream=True, batch=2, verbose=False)
        self.assertIsInstance(stream, outputs.Detections)
        self.assertTrue(stream.is_stream)
        self.assertEqual(len(self.model.calls), calls_before)
        self.model(self.image, prompt="new", threshold=.8, verbose=False)
        frames = list(stream)
        self.assertEqual([frame.metadata["prompt"] for frame in frames], [["old"]] * 3)
        self.assertEqual([frame.metadata["threshold"] for frame in frames], [.4] * 3)
        self.assertEqual(self.model.predict_config["prompt"], ["new"])
        self.assertEqual(self.model.names, {0: "new"})

    def test_predict_prefetches_decode_and_preserves_order(self):
        from klygo.models.detection import base

        threads = []
        original = base._materialize_model_image

        def record(image):
            threads.append(current_thread().name)
            time.sleep(0.01)
            return original(image)

        with patch.object(base, "_materialize_model_image", side_effect=record):
            result = self.model(
                [self.image] * 5,
                batch=2,
                workers=4,
                prefetch=2,
                verbose=False,
            )
        self.assertEqual([frame.frame_index for frame in result], list(range(5)))
        self.assertTrue(any(name.startswith("klygo-decode") for name in threads))
        self.assertGreater(len(set(threads)), 1)

    def test_predict_can_return_packed_materialized_results(self):
        result = self.model.predict(
            [self.image] * 4,
            packed=True,
            batch=2,
            verbose=False,
        )
        self.assertTrue(result.is_packed)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.total_objects, 4)
        with self.assertRaisesRegex(ValueError, "stream and packed"):
            self.model.predict(self.image, stream=True, packed=True)

    def test_multi_image_source_is_not_mislabeled_as_video(self):
        result = self.model.predict([self.image, self.image], verbose=False)
        self.assertEqual(result.metadata["source_type"], "frames")

    def test_benchmark_is_scoped_and_reports_actual_sample_size(self):
        self.model(self.image, prompt="dog", threshold=.6, verbose=False)
        before = deepcopy(self.model.predict_config)
        names = deepcopy(self.model.names)
        report = self.model.benchmark([self.image] * 2, threshold=.8, iterations=3, warmup=1, verbose=False)
        self.assertEqual(report["images_per_iteration"], 2)
        self.assertEqual(report["predict_config"]["threshold"], .8)
        self.assertIn("media resolution excluded", report["measurement_scope"])
        self.assertTrue(all(call[1]["post"]["threshold"] == .8 for call in self.model.calls[-4:]))
        self.assertEqual(self.model.predict_config, before)
        self.assertEqual(self.model.names, names)
        self.model.fail_at = len(self.model.calls) + 1
        with self.assertRaises(RuntimeError):
            self.model.benchmark(self.image, prompt="cat", warmup=1, iterations=1, verbose=False)
        self.assertEqual(self.model.predict_config, before)
        self.assertEqual(self.model.names, names)

    def test_benchmark_rejects_invalid_counts_and_stream_controls(self):
        for kwargs in ({"iterations": 0}, {"iterations": True}, {"warmup": -1},
                       {"warmup": 1.5}, {"stream": True}, {"workers": 2},
                       {"prefetch": 2}, {"num_runs": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                self.model.benchmark(self.image, verbose=False, **kwargs)

    def test_info_device_dtype_read_only_and_unloaded_stream(self):
        self.assertEqual(self.model.device, "cpu")
        self.assertEqual(self.model.dtype, "float32")
        for attribute in ("device", "dtype", "predict_config"):
            with self.assertRaises(AttributeError):
                setattr(self.model, attribute, None)
        stream = self.model([self.image], stream=True)
        self.model.unload()
        with self.assertRaises(models.errors.InvalidStateError):
            list(stream)
        output = StringIO()
        with redirect_stdout(output):
            self.model.info()
        self.assertIn("Predict Configuration", output.getvalue())

    def test_forward_count_validation(self):
        with patch.object(self.model, "forward", return_value=[]), self.assertRaisesRegex(ValueError, "exactly one"):
            self.model(self.image, prompt="dog")
        self.assertIsNone(self.model.predict_config["prompt"])

    def test_stride_preserves_source_positions_with_batches_and_streams(self):
        for stream in (False, True):
            result = self.model([self.image] * 8, vid_stride=2, batch=2, stream=stream, verbose=False)
            self.assertEqual([frame.frame_index for frame in result], [0, 2, 4, 6])

    def test_reused_backend_objects_do_not_rewrite_previous_results(self):
        shared = outputs.detect.Detection([], image_size=self.image.size, source=self.image,
                                          metadata={"nested": {"values": [1]}})
        with patch.object(self.model, "forward", return_value=[shared]):
            first = self.model(self.image, prompt="dog", threshold=.4, verbose=False)
            self.model(self.image, prompt="cat", threshold=.7, verbose=False)
            shared.metadata["nested"]["values"].append(2)
        self.assertEqual(first[0].metadata["prompt"], ["dog"])
        self.assertEqual(first[0].metadata["threshold"], .4)
        self.assertEqual(first[0].metadata["nested"], {"values": [1]})

    def test_loading_credentials_are_excluded_from_inference_provenance(self):
        self.model.settings["model"]["token"] = "unit-test-secret"
        result = self.model(self.image, verbose=False)
        self.assertNotIn("unit-test-secret", repr(result[0].metadata))
        self.assertNotIn("unit-test-secret", repr(self.model.predict_config))


class TestWrapperLifecycle(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (12, 12))

    def wrapper(self, cls):
        target = (
            "klygo.models.detection.yolo_world.YOLOWorld"
            if cls is YOLOWorldDetect
            else "klygo.models.detection.yolo.YOLO"
        )
        with patch(target, FakeYOLO):
            return cls(definition("Ultralytics"))

    def test_fixed_vocabulary_filter_reuse_clear_and_unknown_label(self):
        model = self.wrapper(YOLODetect)
        model.settings["model"]["classes"] = [0]
        model(self.image, prompt="car", threshold=.4, verbose=False)
        model(self.image, verbose=False)
        self.assertEqual(model.model.options[-1]["classes"], [1])
        model(self.image, prompt=None, verbose=False)
        self.assertEqual(model.model.options[-1]["classes"], [0])
        with self.assertRaisesRegex(ValueError, "Unknown YOLO"):
            model(self.image, prompt="dog")
        self.assertEqual(model.names, {0: "person", 1: "car"})

    def test_managed_yolo_exposes_priority_and_prefixed_groups_to_forward(self):
        model = self.wrapper(YOLODetect)

        model.predict(
            self.image,
            threshold=0.4,
            post_agnostic_nms=True,
            verbose=False,
        )

        self.assertEqual(model.model.options[-1]["conf"], 0.4)
        self.assertTrue(model.model.options[-1]["agnostic_nms"])

    def test_native_yolo_routes_backend_options_and_normalizes_results(self):
        model = self.wrapper(YOLODetect)
        model.settings["model"]["nms"] = True
        model.settings["post"]["extra_post_param"] = 999
        with patch.object(
            model,
            "_config_snapshot",
            wraps=model._config_snapshot,
        ) as config_snapshot:
            result = model.predict(
                [self.image, self.image],
                native=True,
                threshold=0.4,
                iou=0.6,
                packed=True,
                verbose=False,
            )

        self.assertIsInstance(result, outputs.Detections)
        self.assertFalse(result.is_stream)
        self.assertTrue(result.is_packed)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(frame.metadata["native"] for frame in result))
        self.assertEqual(model.model.options[-1]["conf"], 0.4)
        self.assertEqual(model.model.options[-1]["iou"], 0.6)
        self.assertTrue(model.model.options[-1]["nms"])
        self.assertNotIn("extra_post_param", model.model.options[-1])
        self.assertEqual(model.model.options[-1]["vid_stride"], 1)
        self.assertEqual(config_snapshot.call_count, 1)
        result[0].metadata["predict_config"]["threshold"] = 0.1
        self.assertEqual(result[1].metadata["predict_config"]["threshold"], 0.4)

    def test_native_yolo_preserves_folder_source_type(self):
        model = self.wrapper(YOLODetect)
        with tempfile.TemporaryDirectory() as temporary:
            result = model.predict(files.path(temporary), native=True, verbose=False)
        self.assertEqual(result.metadata["source_type"], "folder")

    def test_native_yolo_validates_controls_and_rejects_stream(self):
        model = self.wrapper(YOLODetect)
        with self.assertRaisesRegex(ValueError, "stream is unavailable in native mode"):
            model.predict(
                [self.image, self.image],
                native=True,
                stream=True,
                verbose=False,
            )

        with self.assertRaisesRegex(ValueError, "workers and prefetch"):
            model.predict(self.image, native=True, workers=2)
        with self.assertRaisesRegex(ValueError, "unavailable in native mode"):
            model.predict(self.image, native=True, batch=2)
        with self.assertRaisesRegex(ValueError, "unavailable in native mode"):
            model.predict(self.image, native=True, vid_stride=2)
        with self.assertRaisesRegex(ValueError, "unavailable in native mode"):
            model.predict(self.image, native=True, max_frames=2)
        with self.assertRaisesRegex(ValueError, "neither a priority parameter"):
            model.predict(self.image, native=True, conf=0.4)
        with self.assertRaisesRegex(ValueError, "neither a priority parameter"):
            model.predict(self.image, native=True, native_conf=0.4)
        prompted = model.predict(
            self.image,
            native=True,
            prompt="car",
            verbose=False,
        )
        self.assertEqual(prompted[0].metadata["prompt"], ["car"])
        self.assertEqual(model.model.options[-1]["classes"], [1])

    def test_native_is_supported_by_all_detector_subclasses(self):
        model = RecordingDetector()
        result = model.predict(self.image, native=True, verbose=False)
        self.assertTrue(result[0].metadata["native"])

    def test_generic_native_uses_in_memory_inputs_without_media_resolution(self):
        model = RecordingDetector()

        with patch("klygo.models.detection.base.utils.resolve_images") as resolve_images:
            result = model.predict(
                [self.image, self.image],
                native=True,
                verbose=False,
            )

        resolve_images.assert_not_called()
        self.assertEqual(len(result), 2)
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(all(frame.metadata["native"] for frame in result))

    def test_generic_native_rejects_managed_sources_and_controls(self):
        model = RecordingDetector()

        with self.assertRaisesRegex(TypeError, "in-memory native inputs"):
            model.predict("video.mp4", native=True, verbose=False)
        with self.assertRaisesRegex(ValueError, "workers and prefetch"):
            model.predict(self.image, native=True, workers=2, verbose=False)
        with self.assertRaisesRegex(ValueError, "batch"):
            model.predict(self.image, native=True, batch=2, verbose=False)

    def test_native_source_reuses_backend_image_size(self):
        model = RecordingDetector()

        class NativeSource:
            def __init__(self, path, frame_index=None):
                self.path = path
                self.frame_index = frame_index
                self._cached_size = None

        raw = SimpleNamespace(path="video.mp4", orig_shape=(720, 1280))
        with (
            patch("klygo.files.exists", return_value=True),
            patch("klygo.files.extension", return_value=".mp4"),
            patch("klygo.media.LazyImage", NativeSource),
        ):
            source = model._native_image(raw, 7)

        self.assertEqual(source.frame_index, 7)
        self.assertEqual(source._cached_size, (1280, 720))

    def test_world_changes_only_once_and_restores_native_state(self):
        model = self.wrapper(YOLOWorldDetect)
        model([self.image] * 3, prompt=["dog", "cat"], verbose=False)
        model(self.image, verbose=False)
        self.assertEqual(model.model.changes, [["dog", "cat"]])
        before = model.predict_config
        model.benchmark(self.image, prompt="bird", iterations=2, warmup=1, verbose=False)
        self.assertEqual(model.predict_config, before)
        self.assertEqual(model.model.names, {0: "dog", 1: "cat"})
        self.assertEqual(model.model.changes[-2:], [["bird"], ["dog", "cat"]])
        model.model.fail = True
        with self.assertRaises(RuntimeError):
            model.benchmark(self.image, prompt="fish", iterations=1, warmup=0, verbose=False)
        self.assertEqual(model.model.names, {0: "dog", 1: "cat"})
        model.model.fail = False
        model(self.image, prompt=None, verbose=False)
        self.assertEqual(model.model.names, model.original_names)
        model(self.image, prompt="dog", verbose=False)
        model.reset()
        self.assertEqual(model.model.names, model.original_names)

    def test_world_old_stream_restores_new_vocabulary_between_batches(self):
        model = self.wrapper(YOLOWorldDetect)
        stream = model([self.image] * 2, prompt="dog", stream=True, verbose=False)
        model(self.image, prompt="cat", verbose=False)
        for frame in stream:
            self.assertEqual(frame.metadata["prompt"], ["dog"])
            self.assertEqual(model.model.names, {0: "cat"})
            self.assertEqual(model.predict_config["prompt"], ["cat"])

    def test_world_native_restores_new_vocabulary_per_call(self):
        model = self.wrapper(YOLOWorldDetect)
        with self.assertRaisesRegex(ValueError, "stream is unavailable in native mode"):
            model.predict(
                [self.image] * 2,
                prompt="dog",
                native=True,
                stream=True,
                verbose=False,
            )
        result = model.predict(
            [self.image] * 2,
            prompt="dog",
            native=True,
            verbose=False,
        )
        model(self.image, prompt="cat", verbose=False)
        for frame in result:
            self.assertEqual(frame.metadata["prompt"], ["dog"])
        self.assertEqual(model.model.names, {0: "cat"})
        self.assertEqual(model.predict_config["prompt"], ["cat"])


if __name__ == "__main__":
    unittest.main()
