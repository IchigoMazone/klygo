import klygo
from klygo import Config, archive, config, files, media, models, visual


def test_top_level_imports():
    assert klygo.Config is Config
    assert klygo.visualize is visual
    assert all(hasattr(klygo, name) for name in klygo.__all__)


def test_module_exports_are_present():
    for module in (archive, config, files, media, models, visual):
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__}.{name} is missing"


def test_current_public_entry_points():
    assert callable(files.load)
    assert callable(config.load)
    assert callable(media.load)
    assert callable(media.probe)
    assert callable(models.load)
    assert callable(visual.draw_bboxes)
