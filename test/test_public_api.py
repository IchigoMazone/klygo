"""Top-level public API checks."""
import unittest
from PIL import Image
import klygo
from klygo import archive, config, datasets, files, media, models, outputs, postprocessing, preprocessing, visual

class TestPublicApi(unittest.TestCase):
    def test_public_modules(self):
        self.assertEqual(klygo.__version__, "2.1.0")
        for module in (archive, config, datasets, files, media, models, outputs, postprocessing, preprocessing, visual):
            self.assertIsNotNone(module)

    def test_model_and_media_exports(self):
        for function in (models.load, models.configure, models.details, models.flags,
                         media.probe, media.load, preprocessing.compose,
                         postprocessing.compose, datasets.load):
            self.assertTrue(callable(function))

    def test_three_level_outputs(self):
        image = Image.new("RGB", (64, 48), "white")
        box = outputs.Box([4,5,20,25], label="object", score=.9)
        detection = outputs.Detection([box], image_size=image.size, source=image)
        results = outputs.Detections([detection])
        self.assertIs(results[0][0], box)
        self.assertEqual(results.total_objects, 1)
        self.assertEqual(results[0].labels, ["object"])
        self.assertEqual(results[0].boxes, [(4.,5.,20.,25.)])

if __name__ == "__main__": unittest.main()
