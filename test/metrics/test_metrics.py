"""Unit tests for klygo.metrics."""

import unittest

from klygo import metrics, outputs


def _make_pair():
    gt = outputs.Detections(
        [
            outputs.Detection(
                [
                    outputs.Box([0, 0, 20, 20], label="person", score=1.0),
                    outputs.Box([40, 40, 60, 60], label="car", score=1.0),
                ],
                source_id="img1.jpg",
                frame_index=0,
                image_size=(100, 100),
            ),
            outputs.Detection(
                [outputs.Box([10, 10, 30, 30], label="person", score=1.0)],
                source_id="img2.jpg",
                frame_index=0,
                image_size=(100, 100),
            ),
        ]
    )
    pred = outputs.Detections(
        [
            outputs.Detection(
                [
                    outputs.Box([0, 0, 20, 20], label="person", score=0.95),
                    outputs.Box([40, 40, 60, 60], label="person", score=0.70),
                    outputs.Box([70, 70, 90, 90], label="car", score=0.40),
                ],
                source_id="img1.jpg",
                frame_index=0,
                image_size=(100, 100),
            ),
            outputs.Detection(
                [outputs.Box([10, 10, 30, 30], label="person", score=0.85)],
                source_id="img2.jpg",
                frame_index=0,
                image_size=(100, 100),
            ),
        ]
    )
    return pred, gt


class TestMetrics(unittest.TestCase):
    def test_evaluate_and_precision_recall(self):
        pred, gt = _make_pair()
        group = metrics.compose(metrics.precision_recall(iou=0.5), metrics.map_score(), metrics.per_class())
        results = metrics.evaluate(pred, gt, group)
        report = results["precision_recall"]
        self.assertEqual((report["tp"], report["fp"], report["fn"]), (2, 2, 1))
        self.assertAlmostEqual(report["precision"], 0.5)
        self.assertAlmostEqual(report["recall"], 2 / 3)
        self.assertAlmostEqual(report["mean_iou"], 1.0)
        self.assertGreater(results["map_score"]["map_50"], 0.0)
        self.assertGreater(results["map_score"]["map"], 0.0)
        self.assertIn("person", [row["label"] for row in results["per_class"]])
        self.assertIn("car", [row["label"] for row in results["per_class"]])

    def test_map_score_and_confusion_matrix(self):
        pred, gt = _make_pair()
        ap = metrics.map_score(pred, gt, thresholds=(0.5, 0.75))
        self.assertEqual([row["iou"] for row in ap["per_iou"]], [0.5, 0.75])
        self.assertAlmostEqual({row["label"]: row["ap"] for row in ap["per_class"]}["person"], 1.0)

        cm = metrics.confusion_matrix(pred, gt, iou=0.5)
        self.assertEqual(cm["labels"], ["car", "person", None])
        # GT car -> predicted person: rows are GT, columns are predictions.
        self.assertEqual(cm["matrix"][0][1], 1)
        # Unmatched car prediction belongs to the background row.
        self.assertEqual(cm["matrix"][2][0], 1)

    def test_match_boxes_and_threshold_sweep(self):
        pred, gt = _make_pair()
        matched = metrics.match_boxes(pred, gt, iou=0.5)
        self.assertEqual(sum(len(row["matches"]) for row in matched), 2)
        self.assertEqual(sum(len(row["unmatched_pred"]) for row in matched), 2)
        self.assertEqual(sum(len(row["unmatched_target"]) for row in matched), 1)
        self.assertEqual(len(matched), 2)

        sweep = metrics.threshold_sweep(pred, gt, thresholds=(0.5, 0.8), iou=0.5)
        self.assertEqual([row["threshold"] for row in sweep], [0.5, 0.8])
        self.assertEqual((sweep[1]["tp"], sweep[1]["fp"], sweep[1]["fn"]), (2, 0, 1))
        self.assertAlmostEqual(sweep[1]["precision"], 1.0)

    def test_requires_identical_image_source_and_materialized(self):
        pred, gt = _make_pair()
        other = outputs.Detections(
            [
                outputs.Detection([], source_id="different.jpg", frame_index=0, image_size=(100, 100)),
                outputs.Detection([], source_id="img2.jpg", frame_index=0, image_size=(100, 100)),
            ]
        )
        with self.assertRaises(ValueError):
            metrics.evaluate(other, gt)

        wrong_size = outputs.Detections(
            [
                outputs.Detection([], source_id="img1.jpg", frame_index=0, image_size=(200, 200)),
                outputs.Detection([], source_id="img2.jpg", frame_index=0, image_size=(100, 100)),
            ]
        )
        with self.assertRaises(ValueError):
            metrics.precision_recall(wrong_size, gt)

        stream_pred = outputs.Detections(iter(pred), stream=True)
        with self.assertRaises(ValueError):
            metrics.evaluate(stream_pred, gt)


if __name__ == "__main__":
    unittest.main()
