"""Framework-native accelerator dispatch for model backends."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klygo.models.backend import common, huggingface, keras, ultralytics
from klygo.models.detection.base import Detector


class FakeJaxDevice:
    platform = "gpu"

    def __str__(self):
        return "gpu:0"


class BenchmarkHarness:
    """Minimal object exercising Detector.benchmark without model loading."""

    backend = "KerasHub"
    model_id = "fake-model"
    task = "object-detection"

    def current_device(self):
        return "/GPU:0"

    def current_dtype(self):
        return "float32"

    def predict(self, **kwargs):
        return {"boxes": []}


class TestBackendRuntimeDispatch(unittest.TestCase):
    def test_offline_backend_names_keep_their_framework_dispatch(self):
        with patch.object(huggingface, "synchronize") as synchronize:
            common.synchronize("Hugging Face (Offline)", device="cuda:0")
        synchronize.assert_called_once_with(device="cuda:0", value=None)

    def test_ultralytics_uses_its_torch_adapter(self):
        with patch.object(ultralytics, "clear_cache") as clear_cache:
            common.clear_cache("Ultralytics")
        clear_cache.assert_called_once_with()

    def test_tensorflow_device_detection_and_synchronization(self):
        to_numpy = Mock(return_value=[])
        tensor = SimpleNamespace(numpy=to_numpy)
        tensorflow = SimpleNamespace(
            config=SimpleNamespace(list_physical_devices=lambda kind: ["GPU:0"]),
        )
        with (
            patch.object(keras, "get_active_backend", return_value="tensorflow"),
            patch.dict(sys.modules, {"tensorflow": tensorflow}),
        ):
            self.assertEqual(keras.current_device(None), "/GPU:0")
            self.assertTrue(keras.is_accelerator_available("/GPU:0"))
            keras.synchronize(device="/GPU:0", value=tensor)
        to_numpy.assert_called_once_with()

    def test_jax_uses_device_objects_barrier_and_native_cache_clear(self):
        device = FakeJaxDevice()
        barrier = Mock()
        clear_caches = Mock()
        jax = SimpleNamespace(
            devices=lambda: [device],
            effects_barrier=barrier,
            clear_caches=clear_caches,
        )
        with (
            patch.object(keras, "get_active_backend", return_value="jax"),
            patch.dict(sys.modules, {"jax": jax}),
        ):
            self.assertIs(keras.current_device(None), device)
            self.assertTrue(keras.is_accelerator_available(device))
            keras.synchronize(device=device)
            keras.clear_cache()
        barrier.assert_called_once_with()
        clear_caches.assert_called_once_with()

    def test_unknown_backend_does_not_fall_back_to_torch(self):
        self.assertFalse(common.is_accelerator_available("CustomBackend"))
        common.synchronize("CustomBackend")
        common.clear_cache("CustomBackend")

    def test_benchmark_accepts_non_torch_devices_and_uses_dispatcher_barriers(self):
        harness = BenchmarkHarness()
        with patch.object(common, "synchronize") as synchronize:
            report = Detector.benchmark(
                harness,
                iterations=2,
                warmup=1,
                verbose=False,
            )
        self.assertEqual(report["device"], "/GPU:0")
        self.assertEqual(synchronize.call_count, 3)
        self.assertEqual(synchronize.call_args_list[0].kwargs["device"], "/GPU:0")


if __name__ == "__main__":
    unittest.main()
