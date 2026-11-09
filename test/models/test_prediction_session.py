"""Native custom inference, session ownership and legacy import compatibility."""

from contextlib import contextmanager
import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch

from PIL import Image

from klygo import models
from klygo.models import _compat, runtime


class SessionDetector(models.Detector):
    def __init__(self, fail=False):
        super().__init__(backend="Custom", model_id="session-test")
        self.events = []
        self.fail = fail

    @contextmanager
    def _prediction_session(self, options):
        self.events.append("open")

        def infer(images, prompt=None):
            self.events.append("infer")
            if self.fail:
                raise ValueError("inference failed")
            return [{"boxes": []} for _ in images]

        try:
            yield infer
        finally:
            self.events.append("close")

    def forward(self, images, prompt=None):
        raise AssertionError("The session inference callable should be used")


class TestPredictionSession(unittest.TestCase):
    def setUp(self):
        self.images = [Image.new("RGB", (12, 8)) for _ in range(3)]

    def test_one_session_for_multiple_batches(self):
        detector = SessionDetector()
        result = detector.predict(self.images, workers=0, verbose=False)
        self.assertEqual(len(result), 3)
        self.assertEqual(detector.events, ["open", "infer", "infer", "infer", "close"])

    def test_in_memory_native_prediction_uses_and_closes_session(self):
        detector = SessionDetector()
        result = detector.predict(self.images, native=True, verbose=False)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(frame.metadata["native"] for frame in result))
        self.assertEqual(detector.events, ["open", "infer", "close"])

    def test_lazy_session_closes_when_stream_stops_early(self):
        detector = SessionDetector()
        result = detector.predict(self.images, stream=True, workers=0, verbose=False)
        self.assertEqual(detector.events, [])
        iterator = iter(result)
        next(iterator)
        result.close()
        self.assertEqual(detector.events, ["open", "infer", "close"])

    def test_session_closes_on_inference_error(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                detector = SessionDetector(fail=True)
                with self.assertRaisesRegex(ValueError, "inference failed"):
                    result = detector.predict(self.images, stream=stream, workers=0, verbose=False)
                    if stream:
                        list(result)
                self.assertEqual(detector.events, ["open", "infer", "close"])

    def test_forward_only_custom_model_needs_no_execution_adapter(self):
        class Custom(models.Detector):
            def forward(self, images, prompt=None):
                return [{"boxes": []} for _ in images]

        detector = Custom(backend="Custom")
        with patch.object(_compat, "run_inference", side_effect=AssertionError("legacy dispatch")):
            self.assertEqual(len(detector.predict(self.images, verbose=False)), 3)

    def test_builtin_yolo_does_not_use_legacy_result_dispatch(self):
        from types import SimpleNamespace
        from klygo.models.detection.yolo import YOLODetect
        from klygo.models.detection.yolo_world import YOLOWorldDetect

        for cls in (YOLODetect, YOLOWorldDetect):
            with self.subTest(cls=cls.__name__):
                detector = cls.__new__(cls)
                models.Detector.__init__(detector, backend="Ultralytics")
                detector.model = SimpleNamespace()
                with patch.object(_compat, "format_results", side_effect=AssertionError("legacy dispatch")):
                    with detector._prediction_session({}):
                        self.assertEqual(detector.format_results([]), [])

    def test_removed_backend_package_is_not_importable(self):
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("klygo.models.backend")

    def test_runtime_import_does_not_load_frameworks_or_backend_package(self):
        code = '''
import sys
from klygo.models import runtime
assert not {"torch", "keras", "tensorflow", "jax", "ultralytics", "transformers"}.intersection(sys.modules)
assert "klygo.models.backend" not in sys.modules
assert runtime.current_device(None, backend="Custom") == "cpu"
from klygo import models
from PIL import Image
class Custom(models.Detector):
    def forward(self, images, prompt=None):
        return [{"boxes": []} for _ in images]
assert len(Custom().predict(Image.new("RGB", (8, 8)), verbose=False)) == 1
assert "klygo.models.backend" not in sys.modules
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
