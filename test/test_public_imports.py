import unittest
import klygo
from klygo import Config, archive, config, files, media, models, postprocessing, preprocessing, graphics

class TestPublicImports(unittest.TestCase):
    def test_top_level_imports(self):
        assert klygo.Config is Config
        assert klygo.graphics is graphics
        assert all(hasattr(klygo, name) for name in klygo.__all__)

    def test_module_exports_are_present(self):
        for module in (archive, config, files, media, models, postprocessing, preprocessing, graphics):
            for name in module.__all__:
                assert hasattr(module, name), f"{module.__name__}.{name} is missing"

    def test_current_public_entry_points(self):
        assert callable(files.load)
        assert callable(config.load)
        assert callable(media.load)
        assert callable(media.info)
        assert callable(models.load)
        assert callable(postprocessing.compose)
        assert callable(postprocessing.nms)
        assert callable(graphics.draw)

    def test_legacy_visual_module_is_removed(self):
        self.assertNotIn("visual", klygo.__all__)
        self.assertFalse(files.exists(files.parent(klygo.__file__) / "visual"))
