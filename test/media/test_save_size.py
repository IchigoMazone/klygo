"""Sized exports letterbox frames lazily and return no paths."""
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from klygo import files, media, outputs, preprocessing, visual


class TestSaveSize(unittest.TestCase):
    def test_mixed_images_letterbox_and_preserve_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            first = Image.new("RGB", (80, 40), "red")
            second = np.zeros((80, 40, 3), dtype=np.uint8)
            second[:, :, 2] = 255
            original = first.tobytes(), second.copy()
            self.assertIsNone(media.save_images(root, iter([first, second]), extension="png", size=(64, 64)))
            for index in (1, 2):
                self.assertEqual(media.info(root / f"frame_{index:06d}.png")["size"], (64, 64))
            a = media.to_pil(media.load(root / "frame_000001.png")[0])
            b = media.to_pil(media.load(root / "frame_000002.png")[0])
            self.assertEqual(a.getpixel((32, 0)), (114, 114, 114))
            self.assertEqual(a.getpixel((32, 32)), (255, 0, 0))
            self.assertEqual(b.getpixel((0, 32)), (114, 114, 114))
            self.assertEqual(b.getpixel((32, 32)), (0, 0, 255))
            self.assertEqual(first.tobytes(), original[0])
            np.testing.assert_array_equal(second, original[1])

    def test_images_without_size_keep_dimensions_and_return_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            self.assertIsNone(media.save_images(root, [Image.new("RGB", (80, 40)), Image.new("RGB", (40, 80))], extension="png"))
            self.assertEqual(media.info(root / "frame_000001.png")["size"], (80, 40))
            self.assertEqual(media.info(root / "frame_000002.png")["size"], (40, 80))
            self.assertIsNone(media.save_images(root / "empty", []))

    def test_video_mixed_sizes_letterbox_and_return_none(self):
        with tempfile.TemporaryDirectory() as directory:
            output = files.path(directory) / "mixed.avi"
            frames = [Image.new("RGB", (80, 40), "red"), Image.new("RGB", (40, 80), "blue")]
            self.assertIsNone(media.save_video(output, iter(frames), size=(64, 64), fourcc="MJPG", fps=5))
            info = media.info(output)
            self.assertEqual((info["size"], info["frame_count"]), ((64, 64), 2))
            with media.load(output, stream=True) as decoded:
                arrays = [media.to_array(image) for image in decoded]
            for array in arrays:
                np.testing.assert_allclose(array[0, 0], (114, 114, 114), atol=5)
            self.assertGreater(int(arrays[0][32, 32, 0]), int(arrays[0][32, 32, 2]) + 100)
            self.assertGreater(int(arrays[1][32, 32, 2]), int(arrays[1][32, 32, 0]) + 100)

    def test_invalid_size_does_not_consume_or_create_output(self):
        for size, error in ((0, TypeError), ((32,), ValueError), ((0, 32), ValueError),
                            ((-1, 32), ValueError), ((True, 32), ValueError),
                            ((32.5, 32), TypeError), (("32", 32), TypeError)):
            for operation in (media.save_images, media.save_video):
                with self.subTest(size=size, operation=operation.__name__), tempfile.TemporaryDirectory() as directory:
                    consumed = []
                    def frames():
                        consumed.append(True)
                        yield Image.new("RGB", (32, 32))
                    output = files.path(directory) / "output"
                    with self.assertRaises(error):
                        operation(output, frames(), size=size)
                    self.assertEqual(consumed, [])
                    self.assertFalse(files.exists(output))

    def test_failed_sized_video_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            output = root / "saved.avi"
            media.save_video(output, [Image.new("RGB", (32, 32))], fourcc="MJPG")
            checksum = files.hash(output)
            def failing():
                yield Image.new("RGB", (40, 20))
                raise RuntimeError("source failed")
            with self.assertRaisesRegex(RuntimeError, "source failed"):
                media.save_video(output, failing(), size=(64, 64), fourcc="MJPG", overwrite=True)
            self.assertEqual(files.hash(output), checksum)
            self.assertEqual(files.find(root, pattern="*.tmp.avi", recursive=False), [])

    def test_mediaframes_forward_size_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            source = root / "source"
            media.save_images(source, [Image.new("RGB", (80, 40)), Image.new("RGB", (40, 80))], extension="png")
            for stream in (False, True):
                with media.load(source, stream=stream) as frames:
                    self.assertIsNone(frames.save_images(root / f"images-{stream}", size=(64, 64)))
                with media.load(source, stream=stream) as frames:
                    self.assertIsNone(frames.save_video(root / f"video-{stream}.avi", fourcc="MJPG", size=(64, 64)))
                self.assertEqual(media.info(root / f"video-{stream}.avi")["size"], (64, 64))

    def test_annotated_exports_scale_rendered_boxes_and_keep_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            frames = [outputs.Detection([outputs.Box((5, 5, 15, 15), label="cat")],
                        source=Image.new("RGB", size, "white"), source_id=str(i), image_size=size)
                      for i, size in enumerate(((40, 20), (20, 40)))]
            for packed in (False, True):
                results = outputs.Detections(frames, packed=packed)
                before = results.to_dict()
                destination = root / f"annotated-{packed}"
                self.assertIsNone(results.save_images(destination, extension="png", size=(80, 80)))
                with media.load(destination / "frame_000001.png") as decoded:
                    actual = media.to_array(decoded[0])
                expected = preprocessing.letterbox((80, 80))(visual.draw_bboxes(results[0].image, results[0]))
                np.testing.assert_array_equal(actual, np.asarray(expected))
                self.assertIsNone(results.save_video(root / f"annotated-{packed}.avi", size=(80, 80), fourcc="MJPG"))
                self.assertEqual(results.to_dict(), before)

    def test_convert_still_returns_destination_with_opencv_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            source, destination = root / "source.avi", root / "converted.mp4"
            media.save_video(source, [Image.new("RGB", (32, 24), "red")], fourcc="MJPG")
            with patch("shutil.which", return_value=None):
                result = media.convert(source, destination, codec="mp4v", gpu=False)
            self.assertIsNone(result)
            self.assertTrue(files.exists(destination))


if __name__ == "__main__":
    unittest.main()
