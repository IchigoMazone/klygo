"""Reprediction tests for materialized detections."""
import unittest
from PIL import Image
from klygo import outputs


def frame(index=0):
    return outputs.Detection(
        [outputs.Box([0, 0, 5, 5], label="old")],
        source_id="src",
        frame_index=index,
        image_size=(10, 10),
        source=Image.new("RGB", (10, 10)),
    )


class FakeModel:
    def predict(self, sources, stream=False, **options):
        frames = [
            outputs.Detection(
                [outputs.Box([1, 1, 8, 8], label="new", score=0.7)],
                source_id=f"generated-{i}",
                frame_index=0,
                image_size=source.size,
                source=source,
            )
            for i, source in enumerate(sources)
        ]
        return outputs.Detections(frames)


class BadModel:
    def predict(self, sources, stream=False, **options):
        return outputs.Detections([])


class TestRepredict(unittest.TestCase):
    def test_selected_frames_keep_identity(self):
        results = outputs.Detections([frame(0), frame(1)])
        results.repredict(FakeModel(), index=1)
        self.assertEqual(results[0].labels, ["old"])
        self.assertEqual(results[1].labels, ["new"])
        self.assertEqual(results[1].identity, ("src", 1))

    def test_failure_is_atomic(self):
        results = outputs.Detections([frame(0), frame(1)])
        before = results.to_dict()
        with self.assertRaises(ValueError):
            results.repredict(BadModel())
        self.assertEqual(results.to_dict(), before)

    def test_metadata_preservation_and_incremental_updates(self):
        f0 = frame(0)
        f0.metadata = {"camera": "front_cam", "fps": 30, "threshold": 0.25}
        results = outputs.Detections([f0])

        class OptionsModel:
            def predict(self, sources, stream=False, **options):
                meta = {"threshold": options.get("threshold", 0.5), "new_feature": True}
                frames = [
                    outputs.Detection(
                        [outputs.Box([2, 2, 8, 8], label="new")],
                        source_id=f"gen-{i}",
                        frame_index=0,
                        image_size=s.size,
                        source=s,
                        metadata=meta,
                    )
                    for i, s in enumerate(sources)
                ]
                return outputs.Detections(frames)

        results.repredict(OptionsModel(), index=0, threshold=0.85)

        # Original custom keys ("camera", "fps") preserved
        self.assertEqual(results[0].metadata["camera"], "front_cam")
        self.assertEqual(results[0].metadata["fps"], 30)
        # Updated key ("threshold") changed to new value
        self.assertEqual(results[0].metadata["threshold"], 0.85)
        # New key ("new_feature") merged in
        self.assertTrue(results[0].metadata["new_feature"])


if __name__ == "__main__":
    unittest.main()
