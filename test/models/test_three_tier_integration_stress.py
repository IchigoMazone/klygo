"""Comprehensive stress tests for the 3-Tier Model Hierarchy and 3-Level Outputs Contract.

Tests the interaction across:
- Layer 1: BaseModel (metadata, state transitions, unload guards)
- Layer 2: Detector (predict, preprocessing trace restoration, parameter isolation, benchmark)
- Layer 3: Concrete wrappers returning dicts or Detection instances
- Level 1-3 Outputs: Box -> Detection -> Detections synchronization, COW tracking, and postprocessing.
"""

import tempfile
import unittest
from PIL import Image
import numpy as np

from klygo import files, media, models, outputs, postprocessing, preprocessing
from klygo.models import errors


DEFAULT_METADATA = {
    "flags": {"model": False, "post": True},
    "priority": {"post": ["threshold", "iou"]},
    "model": {},
    "post": {"threshold": 0.25, "iou": 0.5},
}


class MockDictDetector(models.Detector):
    """Layer 3 wrapper returning raw prediction dicts."""

    def __init__(self, metadata=None, **kwargs):
        meta = dict(DEFAULT_METADATA) if metadata is None else dict(metadata)
        super().__init__(metadata=meta, backend="Torchvision", **kwargs)
        self.names = {0: "car", 1: "pedestrian"}

    def forward(self, images, prompt=None):
        return [
            {
                "boxes": [[10.0, 10.0, 30.0, 30.0]],
                "scores": [0.85],
                "labels": ["car"],
                "class_ids": [0],
            }
            for _ in images
        ]


class MockDetectionDetector(models.Detector):
    """Layer 3 wrapper returning Detection container objects (standard pattern)."""

    def __init__(self, metadata=None, **kwargs):
        meta = dict(DEFAULT_METADATA) if metadata is None else dict(metadata)
        super().__init__(metadata=meta, backend="Torchvision", **kwargs)
        self.names = {0: "car", 1: "pedestrian"}

    def forward(self, images, prompt=None):
        return outputs.build_detections(
            images,
            [
                {
                    "boxes": [[10.0, 10.0, 30.0, 30.0]],
                    "scores": [0.85],
                    "labels": ["car"],
                    "class_ids": [0],
                }
                for _ in images
            ],
            names=self.names,
        )


class TestThreeTierIntegrationStress(unittest.TestCase):
    """Stress tests across the 3-Tier model hierarchy and 3-Level outputs contract."""

    def test_trace_restoration_parity_between_dict_and_detection_wrappers(self):
        """Verify that preprocessing trace coordinate restoration works identically

        whether Layer 3 forward() returns raw dicts or Detection instances.
        """
        with tempfile.TemporaryDirectory() as root:
            img_path = files.join(root, "sample.png")
            # Create a 200x150 source image
            Image.new("RGB", (200, 150), "blue").save(img_path)

            # Preprocessing pipeline: crop to (20, 30, 120, 130) -> size (100, 100)
            pipeline = preprocessing.compose(
                preprocessing.crop((20, 30, 120, 130)),
            )

            # 1. Test MockDictDetector
            dict_model = MockDictDetector()
            lazy_dict = media.load(img_path)[0]
            proc_dict = pipeline(lazy_dict)
            results_dict = dict_model.predict(proc_dict, verbose=False)
            self.assertEqual(len(results_dict), 1)
            frame_dict = results_dict[0]
            # Restored: box [10, 10, 30, 30] relative to (20, 30) -> [30, 40, 50, 60]
            self.assertEqual(frame_dict.boxes, [(30.0, 40.0, 50.0, 60.0)])
            self.assertEqual(frame_dict.image_size, (200, 150))

            # 2. Test MockDetectionDetector (which returns Detection from forward)
            det_model = MockDetectionDetector()
            lazy_det = media.load(img_path)[0]
            proc_det = pipeline(lazy_det)
            results_det = det_model.predict(proc_det, verbose=False)
            self.assertEqual(len(results_det), 1)
            frame_det = results_det[0]
            # Must also be restored to [30, 40, 50, 60] and original size (200, 150)
            self.assertEqual(frame_det.boxes, [(30.0, 40.0, 50.0, 60.0)])
            self.assertEqual(frame_det.image_size, (200, 150))

            # Parity check
            self.assertEqual(frame_dict.boxes, frame_det.boxes)
            self.assertEqual(frame_dict.image_size, frame_det.image_size)
            results_dict.restore()
            results_det.restore()
            self.assertEqual(frame_dict.boxes, [(30.0, 40.0, 50.0, 60.0)])
            self.assertEqual(frame_det.boxes, [(30.0, 40.0, 50.0, 60.0)])

    def test_chained_transforms_coordinate_restoration_and_postprocessing(self):
        """Test multi-step chained preprocessing restoration through predict -> postprocess."""
        with tempfile.TemporaryDirectory() as root:
            img_path = files.join(root, "multi_step.png")
            Image.new("RGB", (400, 300), "green").save(img_path)

            # Step 1: Crop (50, 50, 350, 250) -> size 300x200
            # Step 2: Letterbox to (100, 100)
            pipeline = preprocessing.compose(
                preprocessing.crop((50, 50, 350, 250)),
                preprocessing.letterbox((100, 100)),
            )

            lazy = media.load(img_path)[0]
            processed = pipeline(lazy)

            model = MockDetectionDetector()
            results = model.predict(processed, verbose=False)
            self.assertEqual(len(results), 1)
            frame = results[0]

            # Original image size must be preserved
            self.assertEqual(frame.image_size, (400, 300))

            # Chain into postprocessing
            post_pipeline = postprocessing.compose(
                postprocessing.clip(image_size=frame.image_size),
                postprocessing.threshold(0.5),
            )
            post_pipeline(results)

            self.assertEqual(len(results), 1)
            self.assertEqual(len(results[0]), 1)
            b = results[0][0]
            # Box must be inside (0, 0, 400, 300)
            self.assertGreaterEqual(b.box[0], 0.0)
            self.assertGreaterEqual(b.box[1], 0.0)
            self.assertLessEqual(b.box[2], 400.0)
            self.assertLessEqual(b.box[3], 300.0)

    def test_base_model_lifecycle_guards_and_state_isolation(self):
        """Verify Layer 1 BaseModel state transitions, unload guards, and output persistence."""
        model = MockDetectionDetector()
        self.assertEqual(model.state, "READY")

        img = Image.new("RGB", (100, 100))
        # 1. Successful prediction
        results = model.predict(img, verbose=False)
        self.assertEqual(len(results), 1)

        # 2. Runtime override kwargs must not mutate model.metadata
        orig_post = dict(model.metadata.get("post", {}))
        model.predict(img, threshold=0.99, verbose=False)
        self.assertEqual(model.metadata.get("post", {}), orig_post)

        # 3. Unload model
        model.unload()
        self.assertEqual(model.state, "UNLOADED")

        # 4. Calling predict, warmup, or benchmark after unload must raise InvalidStateError
        with self.assertRaises(errors.InvalidStateError):
            model.predict(img)

        with self.assertRaises(errors.InvalidStateError):
            model.warmup()

        with self.assertRaises(errors.InvalidStateError):
            model.benchmark(img)

        # 5. Output persistence: previously returned results must remain fully alive and mutable!
        self.assertEqual(len(results), 1)
        results[0].add_box([5, 5, 50, 50], label="pedestrian", score=0.7)
        self.assertEqual(len(results[0]), 2)
        results.relabel({"car": "automobile"})
        self.assertIn("automobile", results[0].labels)

    def test_three_level_bi_directional_synchronization(self):
        """Verify Level 1 Box in-place changes propagate up through Level 2 Detection to Level 3 Detections."""
        box = outputs.Box([10, 10, 50, 50], label="car", score=0.8)
        frame = outputs.Detection([box], source_id="sync_cam", frame_index=0, image_size=(100, 100))
        dets = outputs.Detections([frame])

        # Initially no modified frames
        self.assertEqual(dets.modified_frames, ())

        # Level 1 mutation: mutate box.score directly
        orig_score = box.score
        box.score = 0.999

        # Propagated to Level 2 Detection:
        self.assertAlmostEqual(frame.scores[0], 0.999, places=4)
        self.assertAlmostEqual(frame._scores[0], 0.999, places=4)

        # Propagated to Level 3 Detections:
        self.assertIn(("sync_cam", 0), dets.modified_frames)

        # Revert score: clears modified_frames
        box.score = orig_score
        self.assertNotIn(("sync_cam", 0), dets.modified_frames)

        # Detach box via replace
        new_box = outputs.Box([20, 20, 60, 60], label="bus", score=0.77)
        frame.replace(0, new_box)
        self.assertIsNone(box._owner)

        # Mutating old detached box now does NOT affect frame or dets
        box.score = 0.111
        self.assertNotEqual(frame.scores[0], 0.111)


if __name__ == "__main__":
    unittest.main()
