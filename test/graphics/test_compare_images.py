"""Compare Detection sources directly with display and save modes."""
import importlib
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from klygo import files, media, outputs, graphics

module = importlib.import_module("klygo.graphics.compare_images")


def frame(size=(60, 40), color="red", label="cat", identity="image"):
    return outputs.Detection(
        [outputs.Box((5, 5, 25, 25), label=label, score=.9)],
        source=Image.new("RGB", size, color), source_id=identity, image_size=size,
    )


class TestCompareImages(unittest.TestCase):
    def setUp(self):
        display = patch.object(module, "_show")
        display.start()
        self.addCleanup(display.stop)

    def test_show_mode_preserves_scale_order_and_padding(self):
        first, second = frame(), frame((30, 70), "blue")
        first.clear()
        second.clear()
        with patch.object(module, "_show") as display:
            preview = graphics.compare(first, second, mode="show")
        display.assert_called_once_with(preview, title="Image comparison", backend="auto")
        self.assertEqual(preview.mode, "RGB")
        self.assertEqual(preview.size, (102, 98))
        self.assertEqual(preview.getpixel((0, 28)), (255, 0, 0))
        self.assertEqual(preview.getpixel((72, 28)), (0, 0, 255))
        self.assertEqual(preview.getpixel((0, 97)), (255, 255, 255))

    def test_each_panel_uses_its_own_annotations_without_mutation(self):
        first, second = frame(identity="a"), frame((80, 50), "white", "dog", "b")
        before = first.image.tobytes(), second.image.tobytes(), first.to_dict(), second.to_dict()
        with patch.object(module, "_draw", wraps=module._draw) as draw:
            graphics.compare(first, second, mode="show")
        self.assertIs(draw.call_args_list[0].args[1], first)
        self.assertIs(draw.call_args_list[1].args[1], second)
        self.assertEqual(before, (first.image.tobytes(), second.image.tobytes(), first.to_dict(), second.to_dict()))

    def test_default_show_and_options(self):
        with patch.object(module, "_show") as display, patch.object(module, "_draw", wraps=module._draw) as draw:
            preview = graphics.compare(frame(), frame(), show_scores=False, backend="matplotlib")
        display.assert_called_once_with(preview, title="Image comparison", backend="matplotlib")
        self.assertFalse(draw.call_args.kwargs["show_scores"])

    def test_save_mode_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / "nested" / "preview.png"
            with patch.object(module, "_show") as display:
                preview = graphics.compare(frame(), frame(), mode="save", output_path=path)
            display.assert_not_called()
            self.assertEqual(media.info(path)["size"], preview.size)
            with self.assertRaises(FileExistsError):
                graphics.compare(frame(), frame(), mode="save", output_path=path)
            graphics.compare(frame(), frame(), mode="save", output_path=str(path), overwrite=True)

    def test_file_backed_and_packed_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / "source.png"
            media.save(path, Image.new("RGB", (40, 40), "green"))
            first = outputs.Detection([], source_path=str(path), source_id="a", image_size=(40, 40))
            packed = outputs.Detections([first], packed=True)
            preview = graphics.compare(packed[0], frame(), mode="show")
            self.assertEqual(preview.size, (112, 68))
            self.assertTrue(packed.is_packed)

    def test_missing_source_and_mismatched_dimensions(self):
        no_source = outputs.Detection([], source_id="missing", image_size=(40, 40))
        with self.assertRaises(RuntimeError):
            graphics.compare(no_source, frame(), mode="show")
        bad_size = outputs.Detection([], source=Image.new("RGB", (50, 50)), image_size=(40, 40))
        with self.assertRaises(ValueError):
            graphics.compare(bad_size, frame(), mode="show")

    def test_same_source_with_two_label_sets(self):
        first, second = frame(), frame(label="dog")
        preview = graphics.compare(first, second, mode="show", titles=("Prediction", "GT"))
        self.assertEqual(preview.size, (132, 68))

    def test_invalid_modes_paths_and_options(self):
        first, second = frame(), frame()
        for options, error in (({"mode": "return"}, ValueError),
                               ({"mode": "bad"}, ValueError),
                               ({"mode": "save"}, ValueError),
                               ({"mode": "return", "output_path": "unused.png"}, ValueError),
                               ({"mode": "save", "output_path": 1}, TypeError),
                               ({"mode": 1}, TypeError),
                               ({"titles": ["A"]}, ValueError),
                               ({"titles": "AB"}, TypeError),
                               ({"titles": ["A", 1]}, TypeError),
                               ({"show_labels": 1}, TypeError),
                               ({"show_scores": 1}, TypeError),
                               ({"overwrite": 1}, TypeError),
                               ({"backend": None}, TypeError)):
            with self.subTest(options=options), self.assertRaises(error):
                graphics.compare(first, second, **options)
        for invalid in (Image.new("RGB", (40, 40)), outputs.Detections(), None):
            with self.assertRaises(TypeError):
                graphics.compare(invalid, second, mode="show")


if __name__ == "__main__":
    unittest.main()
