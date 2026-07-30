import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import klygo
from klygo import models
from klygo.models.base import BaseModel, FunctionalModelWrapper
from klygo.models.registry import registry
from klygo.models.exceptions import KlygoModelError, ModelLoadError, BackendExecutionError

from klygo.models.adapters.detect import DetectAdapter, BoundingBox, DetectionResult
from klygo.models.adapters.classify import ClassifyAdapter, ClassificationResult
from klygo.models.adapters.segment import SegmentAdapter, SegmentedObject, SegmentationResult
from klygo.models.adapters.ocr import OCRAdapter, TextRegion, OCRResult
from klygo.models.adapters.llm import LLMAdapter, LLMResponse
from klygo.models.adapters.speech import SpeechAdapter, SpeechTranscript
from klygo.models.adapters.embedding import EmbeddingAdapter, EmbeddingResult

from klygo.models.backends.torch import TorchBackend
from klygo.models.backends.ultralytics import UltralyticsBackend
from klygo.models.backends.huggingface import HuggingFaceBackend


class TestModelsComprehensive(unittest.TestCase):

    def setUp(self):
        # Setup dummy backend runner mock
        self.mock_backend = MagicMock()
        self.mock_backend.native = MagicMock()
        self.config_data = {
            "model_key": "test-model",
            "model_path": "dummy_path",
            "backend": "mock",
            "task": "detect"
        }

    # ----------------------------------------------------------------
    # 1. Base Model & Functional wrapper tests
    # ----------------------------------------------------------------
    def test_base_model_utilities(self):
        adapter = DetectAdapter(self.mock_backend, self.config_data)
        
        # Test info()
        info = adapter.info()
        self.assertEqual(info["backend"], "MagicMock")
        self.assertEqual(info["task"], "DetectAdapter")
        self.assertEqual(info["config"]["model_key"], "test-model")
        
        # Test save_config() and load_config() using a temporary config file
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.yaml")
            adapter._history["dataset"] = "my_dataset"
            adapter._history["operations"]["train"] = {"lr": 0.01}
            adapter.save_config(cfg_path)
            
            # Load in a new adapter instance
            new_adapter = DetectAdapter(self.mock_backend, self.config_data)
            new_adapter.load_config(cfg_path)
            self.assertEqual(new_adapter._history["dataset"], "my_dataset")
            self.assertEqual(new_adapter._history["operations"]["train"], {"lr": 0.01})

        # Test warmup and unload
        self.mock_backend.model = MagicMock()
        # Warmup should complete without crashing
        adapter.warmup()
        # Unload should call cpu() on torch model
        adapter.unload()
        self.mock_backend.model.cpu.assert_called_once()

    def test_functional_model_wrapper(self):
        func = lambda x, **kwargs: f"func_{x}"
        wrapper = FunctionalModelWrapper(func, "custom-fn")
        self.assertEqual(wrapper.info()["backend"], "Functional")
        self.assertEqual(wrapper.predict(10), "func_10")

    # ----------------------------------------------------------------
    # 2. DetectAdapter tests
    # ----------------------------------------------------------------
    @patch("klygo.media.load")
    def test_detect_adapter(self, mock_media_load):
        img = Image.new("RGB", (100, 100))
        mock_media_load.return_value = [img]
        
        adapter = DetectAdapter(self.mock_backend, self.config_data)
        
        # Scenario A: Hugging Face style outputs
        self.mock_backend.predict.return_value = [{
            "boxes": torch.tensor([[10.0, 15.0, 50.0, 60.0]]),
            "scores": torch.tensor([0.95]),
            "labels": ["cat"]
        }]
        
        result = adapter.predict(img)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0].label, "cat")
        self.assertAlmostEqual(result.objects[0].score, 0.95)
        self.assertEqual(result.objects[0].xmin, 10.0)

        # Test crop() functionality
        crops = adapter.crop(img)
        self.assertEqual(len(crops), 1)
        self.assertIsInstance(crops[0], Image.Image)

        # Scenario B: Ultralytics style outputs
        mock_yolo_box = MagicMock()
        mock_yolo_box.xyxy = [torch.tensor([20.0, 30.0, 80.0, 90.0])]
        mock_yolo_box.conf = [torch.tensor(0.85)]
        mock_yolo_box.cls = [torch.tensor(0)]
        
        mock_yolo_result = MagicMock()
        mock_yolo_result.names = {0: "dog"}
        mock_yolo_result.boxes = [mock_yolo_box]
        
        self.mock_backend.predict.return_value = [mock_yolo_result]
        result_yolo = adapter.predict(img)
        self.assertEqual(len(result_yolo.objects), 1)
        self.assertEqual(result_yolo.objects[0].label, "dog")
        
        # Test dataset exporting (YOLO and Classification format)
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # YOLO export
            adapter._export_dataset(tmpdir, format="yolo", classes=["dog"], source="dummy")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "dataset.yaml")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "images")))
            
            # Classification export
            adapter._export_dataset(tmpdir, format="classification", classes=["dog"], source="dummy")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "dog")))

    # ----------------------------------------------------------------
    # 3. ClassifyAdapter tests
    # ----------------------------------------------------------------
    @patch("klygo.media.load")
    def test_classify_adapter(self, mock_media_load):
        img = Image.new("RGB", (100, 100))
        mock_media_load.return_value = [img]
        
        adapter = ClassifyAdapter(self.mock_backend, self.config_data)
        
        # Scenario A: Hugging Face pipeline output list
        self.mock_backend.predict.return_value = [[
            {"label": "apple", "score": 0.9},
            {"label": "banana", "score": 0.1}
        ]]
        result = adapter.predict(img)
        self.assertIsInstance(result, ClassificationResult)
        self.assertEqual(result.label, "apple")
        self.assertAlmostEqual(result.score, 0.9)
        self.assertEqual(len(result.topk), 2)
        
        # Test top1 & top5 helper methods
        self.assertEqual(adapter.top1(img), {"label": "apple", "score": 0.9})
        self.assertEqual(len(adapter.top5(img)), 2)

        # Scenario B: Ultralytics style classification
        mock_yolo_probs = MagicMock()
        mock_yolo_probs.top5 = [1, 0]
        mock_yolo_probs.top5conf = torch.tensor([0.8, 0.2])
        mock_yolo_result = MagicMock()
        mock_yolo_result.names = {0: "banana", 1: "apple"}
        mock_yolo_result.probs = mock_yolo_probs
        self.mock_backend.predict.return_value = [mock_yolo_result]
        
        result_yolo = adapter.predict(img)
        self.assertEqual(result_yolo.label, "apple")
        self.assertAlmostEqual(result_yolo.score, 0.8)

        # Scenario C: PyTorch raw logits / tensor output
        self.mock_backend.predict.return_value = torch.tensor([[5.0, 1.0]]) # Softmax will favor index 0
        adapter.config["classes"] = ["dog", "cat"]
        result_pt = adapter.predict(img)
        self.assertEqual(result_pt.label, "dog")
        
        # Test classification dataset export
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter._export_dataset(tmpdir, source="dummy")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "dog")))

    # ----------------------------------------------------------------
    # 4. SegmentAdapter tests
    # ----------------------------------------------------------------
    @patch("klygo.media.load")
    def test_segment_adapter(self, mock_media_load):
        img = Image.new("RGB", (100, 100))
        mock_media_load.return_value = [img]
        
        adapter = SegmentAdapter(self.mock_backend, self.config_data)
        
        # Ultralytics style segmentation result mocking
        mock_yolo_masks = MagicMock()
        mock_yolo_masks.data = torch.zeros((1, 100, 100))
        mock_yolo_box = MagicMock()
        mock_yolo_box.xyxy = [torch.tensor([0.0, 0.0, 50.0, 50.0])]
        mock_yolo_box.conf = [torch.tensor(0.9)]
        mock_yolo_box.cls = [torch.tensor(0)]
        
        mock_yolo_res = MagicMock()
        mock_yolo_res.names = {0: "person"}
        mock_yolo_res.masks = mock_yolo_masks
        mock_yolo_res.boxes = [mock_yolo_box]
        
        self.mock_backend.predict.return_value = [mock_yolo_res]
        result = adapter.predict(img)
        self.assertIsInstance(result, SegmentationResult)
        self.assertEqual(len(result.objects), 1)
        self.assertEqual(result.objects[0].label, "person")
        self.assertEqual(result.objects[0].mask.shape, (100, 100))
        
        # Test mask extraction helper method
        masks = adapter.mask(img)
        self.assertEqual(len(masks), 1)

    # ----------------------------------------------------------------
    # 5. OCRAdapter tests
    # ----------------------------------------------------------------
    @patch("klygo.media.load")
    def test_ocr_adapter(self, mock_media_load):
        img = Image.new("RGB", (100, 100))
        mock_media_load.return_value = [img]
        
        adapter = OCRAdapter(self.mock_backend, self.config_data)
        
        # Mock OCR output
        self.mock_backend.predict.return_value = [{
            "text": "Hello World",
            "regions": [
                {"box": [0, 0, 10, 10], "text": "Hello", "score": 0.99},
                {"box": [11, 0, 20, 10], "text": "World", "score": 0.98}
            ]
        }]
        
        result = adapter.predict(img)
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.text, "Hello World")
        self.assertEqual(len(result.regions), 2)
        self.assertEqual(result.regions[0].text, "Hello")
        
        # Test read_text helper method
        text = adapter.read_text(img)
        self.assertEqual(text, "Hello World")

    # ----------------------------------------------------------------
    # 6. LLMAdapter tests
    # ----------------------------------------------------------------
    def test_llm_adapter(self):
        adapter = LLMAdapter(self.mock_backend, self.config_data)
        
        # Mock text generation
        self.mock_backend.generate_text.return_value = {"text": "Generated prompt text", "tokens": 5}
        resp = adapter.generate("Hello")
        self.assertIsInstance(resp, LLMResponse)
        self.assertEqual(resp.text, "Generated prompt text")
        self.assertEqual(resp.token_count, 5)
        
        # Mock chat generation
        self.mock_backend.generate_chat.return_value = {"text": "Chat answer", "tokens": 10}
        resp_chat = adapter.chat([{"role": "user", "content": "Hi"}])
        self.assertEqual(resp_chat.text, "Chat answer")
        
        # Mock streaming
        self.mock_backend.generate_stream.return_value = iter(["token1", "token2"])
        stream_iter = adapter.stream("Hello")
        self.assertEqual(list(stream_iter), ["token1", "token2"])

    # ----------------------------------------------------------------
    # 7. SpeechAdapter tests
    # ----------------------------------------------------------------
    def test_speech_adapter(self):
        adapter = SpeechAdapter(self.mock_backend, self.config_data)
        
        # Mock speech transcription
        self.mock_backend.transcribe.return_value = {"text": "Audio text", "chunks": [{"timestamp": [0.0, 1.0], "text": "Audio"}]}
        result = adapter.predict("audio.mp3")
        self.assertIsInstance(result, SpeechTranscript)
        self.assertEqual(result.text, "Audio text")
        self.assertEqual(len(result.chunks), 1)

    # ----------------------------------------------------------------
    # 8. EmbeddingAdapter tests
    # ----------------------------------------------------------------
    @patch("klygo.media.load")
    @patch("klygo.files.exists")
    def test_embedding_adapter(self, mock_files_exists, mock_media_load):
        img = Image.new("RGB", (100, 100))
        mock_media_load.return_value = [img]
        adapter = EmbeddingAdapter(self.mock_backend, self.config_data)
        
        # Case A: Text input embedding
        mock_files_exists.return_value = False
        self.mock_backend.embed_text.return_value = [0.1, 0.2, 0.3]
        res_text = adapter.predict("hello")
        self.assertIsInstance(res_text, EmbeddingResult)
        self.assertTrue(np.array_equal(res_text.vector, np.array([0.1, 0.2, 0.3])))
        
        # Case B: Image path input embedding
        mock_files_exists.return_value = True
        self.mock_backend.embed_image.return_value = [0.5, 0.6, 0.7]
        res_img = adapter.predict("image.jpg")
        self.assertTrue(np.array_equal(res_img.vector, np.array([0.5, 0.6, 0.7])))

        # Test similarity
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(adapter.similarity(a, a), 1.0)
        self.assertAlmostEqual(adapter.similarity(a, b), 0.0)

    # ----------------------------------------------------------------
    # 9. Backend Tests (Torch, Ultralytics, Hugging Face)
    # ----------------------------------------------------------------
    @patch("klygo.media.to_tensor")
    def test_torch_backend(self, mock_to_tensor):
        mock_raw_model = MagicMock(spec=torch.nn.Module)
        mock_raw_model.to.return_value = mock_raw_model
        
        mock_to_tensor.return_value = torch.zeros((3, 10, 10))
        mock_raw_model.return_value = torch.tensor([[1.0]])
        
        backend = TorchBackend(mock_raw_model)
        
        # Test predict
        out = backend.predict([Image.new("RGB", (10, 10))])
        self.assertTrue(torch.equal(out, torch.tensor([[1.0]])))
        
        # Test save
        with patch("torch.save") as mock_torch_save:
            backend.save("model.pth")
            mock_torch_save.assert_called_once()

    def test_ultralytics_backend(self):
        mock_yolo = MagicMock()
        backend = UltralyticsBackend(mock_yolo)
        
        # Test predict
        backend.predict(["img.jpg"])
        mock_yolo.predict.assert_called_once()
        
        # Test evaluate
        backend.evaluate()
        mock_yolo.val.assert_called_once()
        
        # Test save
        backend.save("yolo.pt")
        mock_yolo.save.assert_called_once()

    @patch("transformers.AutoModel.from_pretrained")
    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_huggingface_backend_load_and_methods(self, mock_tokenizer, mock_processor, mock_model):
        mock_raw_model = MagicMock()
        mock_raw_model.to.return_value = mock_raw_model
        mock_model.return_value = mock_raw_model
        
        # Test backend loading
        backend = HuggingFaceBackend.load("dummy-hf-repo", device="cpu", local_files_only=True)
        self.assertIsInstance(backend, HuggingFaceBackend)
        
        # Test generate_text
        backend.tokenizer = MagicMock()
        backend.tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([[1]])}
        mock_raw_model.generate.return_value = torch.tensor([[1, 2, 3]])
        backend.tokenizer.decode.return_value = "hello"
        
        res = backend.generate_text("Hi", temperature=0.7)
        self.assertEqual(res["text"], "hello")
        self.assertEqual(res["tokens"], 3)

    def test_train_and_evaluate_methods(self):
        # 1. Test BaseModel wrapper train() and evaluate() metadata tracking
        adapter = DetectAdapter(self.mock_backend, self.config_data)
        
        self.mock_backend.train.return_value = "trained_model"
        self.mock_backend.evaluate.return_value = {"acc": 0.99}
        
        # Test training metadata capture
        train_res = adapter.train("my_train_dataset", learning_rate=0.001)
        self.assertEqual(train_res, "trained_model")
        self.assertEqual(adapter._history["dataset"], "my_train_dataset")
        self.assertEqual(adapter._history["operations"]["train"]["learning_rate"], 0.001)
        
        # Test evaluation metadata capture
        eval_res = adapter.evaluate(eval_batch_size=32)
        self.assertEqual(eval_res, {"acc": 0.99})
        self.assertEqual(adapter._history["operations"]["evaluate"]["eval_batch_size"], 32)
        
        # 2. Test TorchBackend train() loop & evaluate()
        mock_torch_model = MagicMock(spec=torch.nn.Module)
        mock_torch_model.to.return_value = mock_torch_model
        torch_backend = TorchBackend(mock_torch_model)
        
        # Mock DataLoader inputs/targets
        mock_loader = [
            (torch.zeros((2, 2)), torch.ones((2,), dtype=torch.long))
        ]
        mock_optimizer = MagicMock()
        mock_loss_fn = MagicMock()
        mock_loss_fn.return_value = torch.tensor(0.5, requires_grad=True)
        
        # Run training loop mock
        trained_native = torch_backend.train(mock_loader, mock_optimizer, mock_loss_fn, epochs=1)
        self.assertEqual(trained_native, mock_torch_model)
        
        # Run evaluation mock
        mock_torch_model.return_value = torch.tensor([[0.1, 0.9]]) # logits favoring class 1
        eval_metrics = torch_backend.evaluate(mock_loader, mock_loss_fn)
        self.assertIn("loss", eval_metrics)
        self.assertIn("accuracy", eval_metrics)
        
        # 3. Test UltralyticsBackend train()
        mock_yolo = MagicMock()
        yolo_backend = UltralyticsBackend(mock_yolo)
        yolo_backend.train(data="coco.yaml", epochs=1)
        mock_yolo.train.assert_called_once_with(data="coco.yaml", epochs=1)
        
        # 4. Test HuggingFaceBackend train() & evaluate()
        mock_hf_model = MagicMock()
        mock_hf_model.to.return_value = mock_hf_model
        hf_backend = HuggingFaceBackend(mock_hf_model)
        
        with patch("transformers.Trainer") as mock_trainer_class:
            mock_trainer_instance = MagicMock()
            mock_trainer_class.return_value = mock_trainer_instance
            mock_trainer_instance.train.return_value = "hf_train_res"
            mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.1}
            
            # HF Train
            hf_train_res = hf_backend.train(dataset="dummy_ds", args="dummy_args")
            self.assertEqual(hf_train_res, "hf_train_res")
            mock_trainer_class.assert_called_once()
            
            # HF Evaluate
            hf_eval_res = hf_backend.evaluate(dataset="dummy_ds", args="dummy_args")
            self.assertEqual(hf_eval_res, {"eval_loss": 0.1})


if __name__ == "__main__":
    unittest.main()

