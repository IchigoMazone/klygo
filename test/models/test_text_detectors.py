"""Official text-detector contracts without downloading or executing remote code."""

from collections import UserDict
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from klygo import files, models
from klygo.models.detection.locate_anything import LocateAnythingDetect


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.text = None

    def py_apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][1]["text"]

    def process_vision_info(self, messages):
        return [messages[0]["content"][0]["image"]], None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.text = kwargs["text"]
        return UserDict({"pixel_values": torch.ones(1, 3, 4, 4),
                         "input_ids": torch.ones(1, 3, dtype=torch.int64),
                         "attention_mask": torch.ones(1, 3, dtype=torch.int64),
                         "image_grid_hws": torch.ones(1, 2, dtype=torch.int64)})

    def post_process_grounded_object_detection(self, outputs, **kwargs):
        self.post_kwargs = kwargs
        return [{"boxes": [[1, 2, 3, 4]], "scores": [.8],
                 "text_labels": [self.text[index].split(".")[0]]}
                for index in range(len(kwargs["target_sizes"]))]


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.generation = []
        self.response = "<box><100><200><700><800></box><|im_end|>"

    def to(self, device):
        self.requested_device = device
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.last_inputs = kwargs
        return SimpleNamespace(logits=torch.zeros(1))

    def generate(self, **kwargs):
        self.generation.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TestGroundingDino(unittest.TestCase):
    def setUp(self):
        from klygo.models.detection import grounding_dino
        self.processor, self.native = FakeProcessor(), FakeModel()
        metadata = {"model_id": "fake-dino", "backend": "Hugging Face",
                    "flags": {"model": False, "processor": True, "post": True},
                    "priority": {"post": ["threshold", "text_threshold"]},
                    "model": {"torch_dtype": "float32"}, "processor": {},
                    "post": {"threshold": .25, "text_threshold": .3}}
        with patch.object(grounding_dino.AutoProcessor, "from_pretrained", return_value=self.processor), patch.object(grounding_dino.AutoModelForZeroShotObjectDetection, "from_pretrained", return_value=self.native):
            self.model = grounding_dino.GroundingDinoDetect(metadata)
        self.image = Image.new("RGB", (20, 10))

    def test_normalization_reuse_batch_stream_and_processor_ids(self):
        first = self.model([self.image] * 2, prompt=[" DOG. ", "Cat"], batch=2, threshold=.5, verbose=False)
        self.assertEqual(self.model.predict_config["prompt"], ["dog", "cat"])
        self.assertEqual(self.processor.text, ["dog. cat."] * 2)
        self.assertEqual(self.processor.post_kwargs["input_ids"].dtype, torch.int64)
        self.assertNotIn("torch_dtype", self.native.last_inputs)
        stream = self.model([self.image] * 2, stream=True, verbose=False)
        self.model(self.image, prompt="bird", threshold=.7, verbose=False)
        self.assertEqual([item.labels for item in stream], [["dog"], ["dog"]])
        self.assertEqual(self.model.names, {0: "bird"})
        self.assertEqual(first[0].metadata["prompt"], ["dog", "cat"])
        self.assertEqual(first[0].metadata["threshold"], .5)
        native_res = self.model([self.image], prompt="dog", native=True, verbose=False)
        self.assertTrue(native_res[0].metadata["native"])
        dot_res = self.model([self.image], prompt="dog. cat.", verbose=False)
        self.assertEqual(self.model.predict_config["prompt"], ["dog", "cat"])
        self.assertEqual(self.model.names, {0: "dog", 1: "cat"})

    def test_missing_prompt_clear_and_reset_have_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            self.model(self.image)
        self.model(self.image, prompt="dog", verbose=False)
        before = self.model.predict_config
        with self.assertRaisesRegex(ValueError, "requires"):
            self.model(self.image, prompt=None)
        self.assertEqual(self.model.predict_config, before)
        self.model.reset()
        self.assertIsNone(self.model.predict_config["prompt"])
        self.assertEqual(self.model.names, {})


class TestLocateAnything(unittest.TestCase):
    def metadata(self, consent=True):
        entry = deepcopy(files.load("klygo/models/models.json", verbose=False)["locate-anything"])
        entry["model"]["trust_remote_code"] = consent
        return entry

    def make_model(self):
        self.processor, self.native = FakeProcessor(), FakeModel()
        self.loader = Mock(return_value=self.native)
        self.transformers = SimpleNamespace(
            __version__="4.57.1",
            AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=SimpleNamespace(model_type="locateanything"))),
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=object())),
            AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=self.processor)),
            AutoModel=SimpleNamespace(from_pretrained=self.loader))
        with patch.dict("sys.modules", {"transformers": self.transformers}), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            return LocateAnythingDetect(self.metadata())

    def test_loader_consent_optional_dependency_version_and_family(self):
        with self.assertRaisesRegex(ValueError, "trust_remote_code"):
            LocateAnythingDetect(self.metadata(False))
        with patch.dict("sys.modules", {"torch": None}), self.assertRaisesRegex(ImportError, "locate-anything"):
            LocateAnythingDetect(self.metadata())
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(__version__="5.0", AutoConfig=object(), AutoModel=object(), AutoProcessor=object(), AutoTokenizer=object())}), self.assertRaisesRegex(ImportError, "4.57.1"):
            LocateAnythingDetect(self.metadata())
        entry = self.metadata()
        entry["model_id"] = "some-other-model"
        with self.assertRaisesRegex(ValueError, "local copy"):
            LocateAnythingDetect(entry)

    def test_pinned_loader_and_real_output_schema(self):
        model = self.make_model()
        self.assertEqual(self.loader.call_args.kwargs["revision"], "c32291ca5e996f5a7a485845b4f57a233936bba0")
        image = Image.new("RGB", (200, 100))
        result = model(image, prompt=["person", "car"], verbose=False)[0]
        self.assertEqual(result.boxes, [(20., 20., 140., 80.)] * 2)
        self.assertEqual(result.labels, ["person", "car"])
        self.assertFalse(result.metadata["confidence_available"])
        self.assertEqual(result[0].metadata["score_kind"], "presence_only")
        self.assertIsNone(result[0].metadata["native_score"])
        for call in self.native.generation:
            self.assertTrue(call["use_cache"])
            self.assertEqual(call["input_ids"].dtype, torch.int64)
        self.assertIn("person.", self.processor.calls[0]["text"][0])
        before = deepcopy(result.metadata)
        model(image, max_new_tokens=32, verbose=False)
        self.assertEqual(model.predict_config["max_new_tokens"], 32)
        self.assertEqual(result.metadata, before)

    def test_empty_malformed_coordinates_and_options(self):
        model = self.make_model()
        image = Image.new("RGB", (20, 10))
        for response in ("None", "<box>None</box><|im_end|>"):
            self.native.response = response
            self.assertEqual(model(image, prompt="dog", verbose=False).total_objects, 0)
        for response in ("<box><1><2>", "<box><1001><2><9><8></box>",
                         "<box><9><2><1><8></box>", "unstructured text", 3):
            self.native.response = response
            with self.subTest(response=response), self.assertRaises((ValueError, TypeError)):
                model(image, prompt="dog", verbose=False)
        for kwargs in ({"threshold": .5}, {"generation_mode": "invalid"},
                       {"max_new_tokens": 0}, {"temperature": float("nan")}, {"prompt": "<box>"}):
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                model(image, verbose=False, **kwargs)

    def test_benchmark_failure_and_reset(self):
        model = self.make_model()
        image = Image.new("RGB", (20, 10))
        model(image, prompt="dog", verbose=False)
        before = model.predict_config
        model.benchmark(image, prompt="cat", max_new_tokens=16, iterations=1, warmup=0, verbose=False)
        self.assertEqual(model.predict_config, before)
        self.assertEqual(model.names, {0: "dog"})
        self.native.response = RuntimeError("generation failed")
        with self.assertRaises(RuntimeError):
            model.benchmark(image, prompt="cat", iterations=1, warmup=0, verbose=False)
        self.assertEqual(model.predict_config, before)
        self.assertEqual(model.names, {0: "dog"})
        model.reset()
        self.assertIsNone(model.predict_config["prompt"])
        model.unload()
        self.assertIsNone(model.tokenizer)


if __name__ == "__main__":
    unittest.main()
