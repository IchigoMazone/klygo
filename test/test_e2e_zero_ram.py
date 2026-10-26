"""End-to-end materialized prediction and annotation export test."""
import tempfile
import unittest
from PIL import Image
from klygo import files, outputs, postprocessing as post

class MockDetector:
    def predict(self, sources, stream=False, **kwargs):
        if not isinstance(sources, list): sources = [sources]
        frames = [outputs.Detection([outputs.Box([10,10,20,20], label="mock", score=.9)],
                  source_id=f"source-{i}", image_size=source.size, source=source)
                  for i, source in enumerate(sources)]
        return outputs.Detections(frames)

class TestEndToEnd(unittest.TestCase):
    def test_predict_process_and_export(self):
        results = MockDetector().predict([Image.new("RGB",(64,48)) for _ in range(100)])
        results.transform(post.threshold(.5), post.nms(.5))
        self.assertEqual(results.count, 100); self.assertEqual(results.total_objects, 100)
        with tempfile.TemporaryDirectory() as root:
            target = files.join(root, "annotations"); results.export(target)
            self.assertTrue(files.exists(files.join(target, "manifest.json")))

if __name__ == "__main__": unittest.main()
