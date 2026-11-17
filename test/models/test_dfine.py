"""D-FINE registry and lifecycle contracts without downloading checkpoints."""
import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from klygo import models, outputs
from klygo.models import detection
from klygo.models.detection.dfine import _restore_boxes
from klygo.models.runtime import keras as runtime


class NativeDFine:
    num_classes = 80
    def __init__(self):
        self.prediction_decoder = SimpleNamespace(confidence_threshold=.5, iou_threshold=.5, bounding_box_format="yxyx")
        self.preprocessor = SimpleNamespace(image_converter=SimpleNamespace(
            image_size=(64,64), crop_to_aspect_ratio=True, pad_to_aspect_ratio=False))
        self.preprocessor.image_converter.get_config = lambda: {"image_size": (64,64)}
        self.stages = []
        self.failure = None
    def preprocess_samples(self, inputs):
        self.stages.append("preprocess")
        if self.failure == "preprocess": raise RuntimeError("preprocess failed")
        return np.zeros((len(inputs),64,64,3), dtype=np.float32)
    def __call__(self, inputs, training=False):
        self.stages.append("forward")
        if self.failure == "forward": raise RuntimeError("forward failed")
        return inputs
    def decode_predictions(self, raw, prepared):
        self.stages.append("decode")
        if self.failure == "decode": raise RuntimeError("decode failed")
        n=len(prepared)
        return dict(boxes=np.tile([[0.,0.,64.,64.],[8.,8.,16.,16.],[0.,0.,0.,0.]],(n,1,1)),
                    confidence=np.tile([.9,.3,-1.],(n,1)), labels=np.tile([0,2,-1],(n,1)))


class TestDFine(unittest.TestCase):
    def detector(self, native=None):
        native=native or NativeDFine()
        keras=SimpleNamespace(backend=SimpleNamespace(backend=lambda:"torch"),
            device=lambda device:nullcontext(), ops=SimpleNamespace(convert_to_tensor=lambda x,dtype:np.asarray(x,dtype=dtype)),
            utils=SimpleNamespace(unpack_x_y_sample_weight=lambda x:(x,None,None)))
        hub=SimpleNamespace(models=SimpleNamespace(DFineObjectDetector=SimpleNamespace(from_preset=Mock(return_value=native))))
        with patch.dict(sys.modules,{"keras_hub":hub}), patch("klygo.models.detection.dfine._require_keras",return_value=keras):
            result=models.load("dfine_nano_coco",device="cpu")
        return result,native

    def test_registry_lazy_class_and_geometry_batch_order(self):
        self.assertIn("DFineDetect",detection.__all__)
        detector,native=self.detector()
        images=[Image.new("RGB",(100,50)),Image.new("RGB",(50,100)),Image.new("RGB",(100,50))]
        result=detector.forward(images)
        self.assertEqual([frame.image_size for frame in result],[(100,50),(50,100),(100,50)])
        self.assertEqual(result[0].boxes[0],(25.,0.,75.,50.))
        self.assertEqual(result[1].boxes[0],(0.,25.,50.,75.))
        self.assertEqual(result[0].labels,["person","car"])
        self.assertEqual(native.stages,["preprocess","forward","decode"]*2)
        self.assertEqual((native.prediction_decoder.confidence_threshold,native.prediction_decoder.iou_threshold),(.5,.5))
        with patch("klygo.models.detection.base.runtime.inference_context",return_value=nullcontext()):
            managed=detector.predict(images,batch=3,threshold=.5,prompt=["person"],verbose=False)
        self.assertIsInstance(managed,outputs.Detections)
        self.assertTrue(all(frame.labels==["person"] for frame in managed))
        self.assertEqual(detector.metadata["post"]["threshold"],.25)

    def test_errors_propagate_and_decoder_state_is_restored(self):
        detector,native=self.detector()
        for stage in ("preprocess","forward","decode"):
            native.failure=stage
            with self.assertRaisesRegex(RuntimeError,stage): detector.forward([Image.new("RGB",(64,64))])
            self.assertEqual((native.prediction_decoder.confidence_threshold,native.prediction_decoder.iou_threshold),(.5,.5))
        native.failure=None
        with self.assertRaises(ValueError): detector.forward([Image.new("RGB",(64,64))],prompt=["unknown"])
        self.assertEqual(list(detector.forward([])),[])

    def test_inverse_resize_crop_pad_and_stretch(self):
        converter=SimpleNamespace(image_size=(64,64),crop_to_aspect_ratio=True,pad_to_aspect_ratio=False)
        self.assertEqual(_restore_boxes([[0,0,64,64]],(101,50),converter),[[25.,0.,75.,50.]])
        converter.crop_to_aspect_ratio=False
        self.assertEqual(_restore_boxes([[0,0,64,64]],(100,50),converter),[[0.,0.,100.,50.]])
        converter.pad_to_aspect_ratio=True
        self.assertEqual(_restore_boxes([[0,16,64,48]],(100,50),converter),[[0.,0.,100.,50.]])

    def test_benchmark_factory_and_independent_native_reference(self):
        from benchmarks.gpu_suite.run import build, raw, canonical
        detector, native = self.detector()
        with patch.object(models, "load", return_value=detector) as load:
            self.assertIs(build("d-fine", SimpleNamespace(device="cpu"))[0], detector)
            load.assert_called_once_with("dfine_nano_coco", device="cpu", threshold=.25, iou=.5)
        # The raw reference must not call the wrapper's inference implementation.
        with patch.object(detector, "run_inference", side_effect=AssertionError("wrapper used by raw reference")), patch.dict(sys.modules, {"keras": detector._keras}):
            detector._keras.ops.convert_to_numpy = np.asarray
            result = raw(detector, "d-fine", [Image.new("RGB", (100,50))], None)
            report = canonical(result[0], "d-fine", True)
        self.assertEqual(report["boxes"][0], [25.,0.,75.,50.])
        self.assertEqual(report["labels"], ["person", "car"])

    def test_runtime_device_uses_keras_variables_before_torch_registration(self):
        import torch
        native=SimpleNamespace(parameters=lambda:iter(()),weights=[SimpleNamespace(value=torch.zeros(1))])
        with patch.object(runtime,"get_active_backend",return_value="torch"), patch.object(torch.cuda,"is_available",return_value=True):
            self.assertEqual(str(runtime.current_device(native)),"cpu")


if __name__ == "__main__": unittest.main()
