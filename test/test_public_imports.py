import unittest
import klygo
from klygo import Config, archive, config, files, media, models, postprocessing, preprocessing, visual

class TestPublicImports(unittest.TestCase):
    def test_top_level_imports(self):
        assert klygo.Config is Config
        assert klygo.visual is visual
        assert all(hasattr(klygo, name) for name in klygo.__all__)

    def test_module_exports_are_present(self):
        for module in (archive, config, files, media, models, postprocessing, preprocessing, visual):
            for name in module.__all__:
                assert hasattr(module, name), f"{module.__name__}.{name} is missing"

    def test_current_public_entry_points(self):
        assert callable(files.load)
        assert callable(config.load)
        assert callable(media.load)
        assert callable(media.probe)
        assert callable(models.load)
        assert callable(postprocessing.compose)
        assert callable(postprocessing.nms)
        assert callable(visual.draw_bboxes)
