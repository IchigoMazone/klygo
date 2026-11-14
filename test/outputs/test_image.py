"""Detection.image resolves source pixels while returning an independent image."""
import tempfile
import unittest
import numpy as np
from PIL import Image
from klygo import files, media, outputs


class TestDetectionImage(unittest.TestCase):
    def test_pil_and_numpy_sources_are_not_modified(self):
        for source in (Image.new("RGB", (40, 30), "red"), np.zeros((30, 40, 3), dtype=np.uint8)):
            frame = outputs.Detection([], source=source, image_size=(40, 30))
            image = frame.image
            original = frame.image.tobytes()
            image.putpixel((0, 0), (1, 2, 3))
            self.assertEqual(frame.image.tobytes(), original)
            self.assertEqual(image.size, (40, 30))

    def test_lazy_source_and_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / "source.png"
            media.save(path, Image.new("RGB", (40, 30), "red"))
            frame = outputs.Detection([], source=media.load(path)[0], image_size=(40, 30))
            self.assertEqual(frame.image.size, (40, 30))
            missing = outputs.Detection([], source_path=str(path.parent / "missing.png"), image_size=(40, 30))
            with self.assertRaises(FileNotFoundError):
                _ = missing.image
        with self.assertRaises(RuntimeError):
            _ = outputs.Detection([], image_size=(40, 30)).image

    def test_video_path_uses_requested_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = files.path(directory) / "clip.mp4"
            media.save_video(path, [Image.new("RGB", (32, 24), color) for color in ("red", "green", "blue")], fps=3)
            frame = outputs.Detection([], source_path=str(path), frame_index=2, image_size=(32, 24))
            r, g, b = frame.image.getpixel((16, 12))
            self.assertGreater(b, r + 100)
            self.assertGreater(b, g + 100)
            unavailable = outputs.Detection([], source_path=str(path), frame_index=9, image_size=(32, 24))
            with self.assertRaises(RuntimeError):
                _ = unavailable.image


if __name__ == "__main__":
    unittest.main()
