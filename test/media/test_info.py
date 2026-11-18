"""Public image/video metadata API after the media.info rename."""
import tempfile
import unittest
from PIL import Image, UnidentifiedImageError
from klygo import files, media


class TestMediaInfo(unittest.TestCase):
    def test_image_metadata_and_exports(self):
        self.assertIn("info", media.__all__)
        self.assertNotIn("probe", media.__all__)
        self.assertFalse(hasattr(media, "probe"))
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / "image.png"
            media.save(path, Image.new("RGB", (32, 24)), verbose=False)
            for source in (path, str(path)):
                result = media.info(source)
                self.assertEqual(result["size"], (32, 24))
                self.assertEqual(result["type"], "image")
                self.assertEqual(result["format"], "PNG")
                self.assertEqual(result["mode"], "RGB")
                self.assertEqual(result["name"], "image.png")
                self.assertEqual(result["path"], path)

    def test_video_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / "clip.mp4"
            media.save_video(path, [Image.new("RGB", (32, 24))] * 4, fps=4)
            result = media.info(path)
            self.assertEqual(result["type"], "video")
            self.assertEqual(result["size"], (32, 24))
            self.assertEqual(result["frame_count"], 4)
            self.assertAlmostEqual(result["fps"], 4)
            self.assertEqual(result["duration_seconds"], 1)
            self.assertIn("codec", result)
            self.assertIn("web_ready", result)

    def test_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = files.path(directory)
            with self.assertRaises(FileNotFoundError):
                media.info(root / "missing.png")
            with self.assertRaises(ValueError):
                media.info(root)
            path = root / "invalid.txt"
            files.save(path, "not an image", verbose=False)
            with self.assertRaises(UnidentifiedImageError):
                media.info(path)


if __name__ == "__main__":
    unittest.main()
