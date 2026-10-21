import unittest
from PIL import Image

import klygo
from klygo import archive, config, datasets, files, media, models, outputs, postprocessing, preprocessing, visual
from klygo.outputs.detect import Box, Crops, Detection, Detections

class TestPublicApi(unittest.TestCase):
    def test_top_level_public_modules(self):
        assert klygo.__version__ == "2.1.0"
        assert klygo.visual is visual
        assert klygo.Config is config.Config

        for module in (archive, config, datasets, files, media, models, outputs, postprocessing, preprocessing, visual):
            assert module is not None

    def test_current_model_api_exports(self):
        assert callable(models.load)
        assert callable(models.configure)
        assert not hasattr(models, "conf")
        assert callable(models.details)
        assert callable(models.flags)
        assert not hasattr(models, "init")
        assert not hasattr(models, "profile")
        assert callable(models.set_backend)
        assert callable(models.get_backend)
        assert models.BaseModel is not None
        assert models.Detector is not None

    def test_archive_and_media_public_exports(self):
        assert klygo.utils.human_size(0) == "0.00 B"
        assert klygo.utils.human_size(1024) == "1.00 KB"
        assert callable(media.probe)
        assert callable(media.load)
        assert "probe" in media.__all__
        assert "stream" not in media.__all__
        assert not hasattr(media, "stream")
        assert callable(preprocessing.compose)
        assert callable(postprocessing.compose)
        assert callable(postprocessing.relabel)
        assert callable(datasets.load)

    def test_detection_output_types(self):
        image = Image.new("RGB", (64, 48), "white")
        box = Box(id=0, label="object", score=0.9, box=[4, 5, 20, 25], parent_image=image)
        detection = Detection(source_image=image, objects=[box])
        results = Detections([detection], source_type="image")

        assert isinstance(detection.crops, Crops)
        assert results.count == 1
        assert results.total_objects == 1
        assert results.labels == ["object"]
        assert results.boxes == [[4.0, 5.0, 20.0, 25.0]]
