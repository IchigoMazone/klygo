"""Identity-aligned group previews through the public visual API."""
import importlib
import unittest
from unittest.mock import patch

from PIL import Image

from klygo import outputs, visual


def collection():
    return outputs.Detections([
        outputs.Detection(
            [], source_id=f"image-{index}", frame_index=0,
            image_size=(40, 30), source=Image.new("RGB", (40, 30), color),
        )
        for index, color in enumerate(("red", "blue", "green", "yellow"))
    ])


class TestDetectionGroupShow(unittest.TestCase):
    def test_shuffled_packed_members_render_the_same_image_without_mutation(self):
        pred, truth = collection(), collection()
        group = outputs.DetectionGroup(pred, truth)
        truth.shuffle(seed=7)
        self.assertNotEqual([f.identity for f in pred], [f.identity for f in truth])
        index = next(i for i in range(len(pred)) if pred[i].identity != truth[i].identity)
        group.pack()
        before = [member.to_dict() for member in group]
        compare_module = importlib.import_module("klygo.visual.compare_images")
        with patch.object(compare_module, "show_image") as display:
            self.assertIsNone(group.show(index))
        display.assert_called_once()
        preview = display.call_args.args[0]
        self.assertEqual(preview.size, (92, 58))
        self.assertEqual(preview.getpixel((0, 28)), preview.getpixel((52, 28)))
        self.assertEqual([member.to_dict() for member in group], before)
        self.assertTrue(all(member.is_packed for member in group))

    def test_named_numeric_and_mixed_selection_use_first_selected_member_order(self):
        a, b, c = collection(), collection(), collection()
        c.shuffle(seed=7)
        group = outputs.DetectionGroup({"yolo": a, "dino": b, "truth": c})
        for selectors, names in (((2, 0), ("truth", "yolo")),
                                 (("truth", "dino"), ("truth", "dino")),
                                 ((-1, "yolo"), ("truth", "yolo"))):
            with self.subTest(selectors=selectors), patch.object(visual, "compare_images") as compare:
                group.show(-1, members=selectors)
                first, second = compare.call_args.args
                self.assertEqual(first.identity, c[-1].identity)
                self.assertEqual(first.identity, second.identity)
                self.assertEqual(compare.call_args.kwargs["titles"], names)

    def test_defaults_renamed_titles_and_display_options(self):
        group = outputs.DetectionGroup(collection(), collection())
        group.rename({"pred": "model", "truth": "ground_truth"})
        with patch.object(visual, "compare_images") as compare:
            group.show()
            self.assertEqual(compare.call_args.kwargs["titles"], ("model", "ground_truth"))
            group.show(1, titles=("Prediction", "GT"), show_labels=False,
                       show_scores=False, backend="matplotlib")
            self.assertEqual(compare.call_args.kwargs, {
                "titles": ("Prediction", "GT"), "show_labels": False,
                "show_scores": False, "backend": "matplotlib",
            })

    def test_invalid_selection_does_not_render(self):
        group = outputs.DetectionGroup(collection(), collection())
        cases = (
            ({"index": True}, TypeError), ({"index": 1.5}, TypeError),
            ({"index": 4}, IndexError), ({"index": -5}, IndexError),
            ({"members": "pred"}, TypeError), ({"members": (0,)}, ValueError),
            ({"members": (0, 1, 0)}, ValueError), ({"members": (0, 2)}, IndexError),
            ({"members": ("missing", 1)}, KeyError),
            ({"members": (0, True)}, TypeError), ({"members": (None, 1)}, TypeError),
            ({"members": (0, "pred")}, ValueError),
        )
        with patch.object(visual, "compare_images") as compare:
            for options, error in cases:
                with self.subTest(options=options), self.assertRaises(error):
                    group.show(**options)
            compare.assert_not_called()
        with self.assertRaises(IndexError):
            outputs.DetectionGroup(collection()).show()
        with self.assertRaises(IndexError):
            outputs.DetectionGroup(outputs.Detections(), outputs.Detections()).show()

    def test_source_alignment_is_rechecked_after_external_changes(self):
        group = outputs.DetectionGroup(collection(), collection())
        group[1].pop()
        with patch.object(visual, "compare_images") as compare:
            with self.assertRaises(ValueError):
                group.show()
            compare.assert_not_called()

    def test_rendering_errors_propagate(self):
        a = outputs.Detections([outputs.Detection([], source_id="missing", image_size=(40, 30))])
        group = outputs.DetectionGroup(a, a.copy())
        with self.assertRaises(RuntimeError):
            group.show()
        group = outputs.DetectionGroup(collection(), collection())
        for options, error in (({"titles": ("one",)}, ValueError),
                               ({"show_scores": 1}, TypeError),
                               ({"backend": None}, TypeError)):
            with self.subTest(options=options), self.assertRaises(error):
                group.show(**options)
