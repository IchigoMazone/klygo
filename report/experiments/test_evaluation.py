"""Check operating-point accounting with COCO ignore/crowd and empty predictions."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from prepare_coco import write_json
from run_pilot import evaluate


class EvaluationTests(unittest.TestCase):
    def check_case(self, predictions, expected, crowd=False):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gt.json"
            write_json(path, {
                "info": {}, "images": [{"id": 1, "width": 100, "height": 100}],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                    "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": int(crowd)}],
            })
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(path, predictions, [1])["operating_point"]
            self.assertEqual(tuple(result[key] for key in ("tp", "fp", "fn")), expected)

    def test_empty_predictions_count_false_negative(self):
        self.check_case([], (0, 0, 1))

    def test_duplicate_and_below_threshold(self):
        predictions = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": score}
                       for score in (0.9, 0.8, 0.1)]
        self.check_case(predictions, (1, 1, 0))

    def test_crowd_is_ignored(self):
        self.check_case([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9}],
                        (0, 0, 0), crowd=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
