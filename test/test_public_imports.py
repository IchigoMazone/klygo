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
        assert callable(graphics.draw_bboxes)

    def test_deprecated_visual_alias_forwards_to_graphics(self):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            from klygo import visual
            assert visual.draw_bboxes is graphics.draw_bboxes
            assert any(issubclass(w.category, DeprecationWarning) for w in caught)
