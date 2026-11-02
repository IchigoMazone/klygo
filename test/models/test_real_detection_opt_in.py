"""Opt-in local-checkpoint smoke tests; never download weights implicitly.

Set KLYGO_REAL_IMAGE and each KLYGO_REAL_<BACKEND> checkpoint to enable.
The Locate Anything flag explicitly opts into that checkpoint's custom code.
"""

import os
import unittest

from klygo import files, models, outputs


class TestRealDetectionOptIn(unittest.TestCase):
    def run_checkpoint(self, key, registry, implementation=None):
        source = os.environ.get("KLYGO_REAL_IMAGE")
        checkpoint = os.environ.get(f"KLYGO_REAL_{key}")
        if not checkpoint or not source:
            self.skipTest(f"Set KLYGO_REAL_{key} and KLYGO_REAL_IMAGE to local files")
        if not files.exists(checkpoint) or not files.is_file(source):
            self.fail("Opt-in smoke paths must exist locally")
        entries = files.load(files.join(files.parent(files.parent(__file__)), "..", "klygo", "models", "models.json"), verbose=False)
        entry = entries[registry]
        entry["model_id"] = checkpoint
        entry.pop("option", None)
        if implementation:
            entry["class"] = implementation
        if key == "LOCATE":
            import torch
            import transformers
            if not torch.cuda.is_available():
                self.skipTest("Locate Anything real inference requires CUDA; this host has CPU-only PyTorch")
            if transformers.__version__ != "4.57.1":
                self.skipTest("Locate Anything real inference requires an isolated Transformers 4.57.1 environment")
            entry["model"].update(trust_remote_code=True, local_files_only=True)
            entry["generation"]["max_new_tokens"] = 512
        detector = models.load(entry)
        try:
            result = detector(source, prompt=["person"], verbose=False)
            self.assertIsInstance(result, outputs.detect.Detections)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].text_prompt, ["person"])
            detector(source, verbose=False)
            before = detector.predict_config
            detector.benchmark(source, iterations=1, warmup=0, verbose=False)
            self.assertEqual(detector.predict_config, before)
        finally:
            detector.unload()

    def test_yolo(self):
        self.run_checkpoint("YOLO", "yolo11?.pt", "klygo.models.detection.YOLODetect")

    def test_yolo_world(self):
        self.run_checkpoint("WORLD", "yolov8*-world*.pt")

    def test_grounding_dino(self):
        self.run_checkpoint("DINO", "grounding-dino-*")

    def test_locate_anything(self):
        self.run_checkpoint("LOCATE", "locate-anything")


if __name__ == "__main__":
    unittest.main()
