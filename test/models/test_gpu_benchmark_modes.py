"""Benchmark stream controls must honor Detector's frozen model-group semantics."""

from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from klygo import models
from benchmarks.gpu_suite.run import execute


class Backend:
    names = {}

    def __init__(self):
        self.calls = []

    def predict(self, source=None, stream=False, **options):
        self.calls.append(stream)
        frame = SimpleNamespace(orig_img=np.zeros((8,8,3),dtype=np.uint8),
                                path="image0.jpg",names={},boxes=None)
        return iter([frame]) if stream else [frame]

    def __call__(self, images, stream=False, **options):
        return self.predict(source=images,stream=stream,**options)


class Harness(models.Detector):
    def __init__(self, explicit_default=True):
        super().__init__(backend="Ultralytics", metadata={
            "flags":{"model":False,"post":True},
            "model":{"stream":False} if explicit_default else {},"post":{"threshold":0.25,"iou":0.7},
        })
        self.model = Backend()

    def forward(self, images, prompt=None):
        list(self.model(images,**dict(self.metadata["model"])))
        return [{"boxes":[]} for _ in images]


class TestGpuBenchmarkModes(unittest.TestCase):
    def test_modes_use_transient_native_override_and_frozen_managed_default(self):
        detector = Harness()
        args = SimpleNamespace(batch=1)
        for mode in ("native-stream","native-non-stream","managed-stream","managed-non-stream"):
            result = execute(detector,"yolo",None,mode,[Image.new("RGB",(8,8))],args)
            list(result)
            result.close()
        self.assertEqual(detector.model.calls,[True,False,False,False])
        self.assertFalse(detector.metadata["model"]["stream"])
        self.assertFalse(detector.predict_config["model_stream"])

    def test_frozen_model_group_does_not_remember_native_stream_override(self):
        detector = Harness(explicit_default=False)
        detector.predict(Image.new("RGB",(8,8)),native=True,model_stream=True,verbose=False)
        detector.predict(Image.new("RGB",(8,8)),verbose=False)
        self.assertEqual(detector.model.calls,[True,False])
        with self.assertRaisesRegex(ValueError,"locked"):
            detector.predict(Image.new("RGB",(8,8)),model_stream=False,verbose=False)


if __name__=="__main__":
    unittest.main()
