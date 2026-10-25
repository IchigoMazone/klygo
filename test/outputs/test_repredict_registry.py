"""Reprediction tests for materialized detections."""
import unittest
from PIL import Image
from klygo import outputs

def frame(index=0):
    return outputs.Detection([outputs.Box([0,0,5,5], label="old")], source_id="src",
                             frame_index=index, image_size=(10,10), source=Image.new("RGB",(10,10)))

class FakeModel:
    def predict(self, sources, stream=False, **options):
        frames = [outputs.Detection([outputs.Box([1,1,8,8], label="new", score=.7)],
                 source_id=f"generated-{i}", frame_index=0, image_size=source.size, source=source)
                 for i, source in enumerate(sources)]
        return outputs.Detections(frames)

class BadModel:
    def predict(self, sources, stream=False, **options): return outputs.Detections([])

class TestRepredict(unittest.TestCase):
    def test_selected_frames_keep_identity(self):
        results = outputs.Detections([frame(0), frame(1)]); results.repredict(FakeModel(), index=1)
        self.assertEqual(results[0].labels, ["old"]); self.assertEqual(results[1].labels, ["new"])
        self.assertEqual(results[1].identity, ("src",1))

    def test_failure_is_atomic(self):
        results = outputs.Detections([frame(0), frame(1)]); before = results.to_dict()
        with self.assertRaises(ValueError): results.repredict(BadModel())
        self.assertEqual(results.to_dict(), before)

if __name__ == "__main__": unittest.main()
