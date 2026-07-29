import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import tempfile
from pathlib import Path
from box import Box

import klygo.config as config
from klygo.config import Config


def test_config_load_save_convert(tmp_path):
    data = {"model": {"name": "yolo", "batch": 16}, "default": {"root": str(tmp_path)}}
    yaml_p = tmp_path / "config.yaml"

    # 1. save & load
    config.save(yaml_p, data, overwrite=True)
    box_cfg = config.load(yaml_p, verbose=False)
    assert isinstance(box_cfg, Box)
    assert box_cfg.model.name == "yolo"

    # 2. convert
    json_p = tmp_path / "config.json"
    conv_p = config.convert(yaml_p, json_p, overwrite=True)
    assert conv_p.exists()
    assert config.load(json_p, verbose=False).model.batch == 16


def test_config_create_defaults(tmp_path):
    # 1. defaults
    def_dict = config.defaults()
    assert "model" in def_dict
    assert def_dict["model"]["name"] == "yolov8n"

    # 2. create
    create_p = tmp_path / "created.yaml"
    cfg_box = config.create(create_p, overwrite=True, verbose=False)
    assert cfg_box.model.name == "yolov8n"


def test_config_merge_update():
    c1 = {"a": 1, "sub": {"x": 10}}
    c2 = {"b": 2, "sub": {"y": 20}}

    merged = config.merge(c1, c2, deep=True)
    assert merged.a == 1
    assert merged.b == 2
    assert merged.sub.x == 10
    assert merged.sub.y == 20

    updated = config.update(c1, {"sub": {"x": 99}}, deep=True)
    assert updated.sub.x == 99


def test_config_access_and_modify():
    cfg = {"model": {"name": "resnet", "hyperparams": {"lr": 0.001}}}

    # 1. get & has
    assert config.has(cfg, "model.name")
    assert config.get(cfg, "model.name") == "resnet"
    assert config.get(cfg, "model.hyperparams.lr") == 0.001
    assert not config.has(cfg, "model.unknown")
    assert config.get(cfg, "model.unknown", default=42) == 42

    # 2. set
    new_cfg = config.set(cfg, "model.hyperparams.lr", 0.01)
    assert new_cfg.model.hyperparams.lr == 0.01

    # 3. delete
    config.delete(new_cfg, "model.hyperparams.lr")
    assert not config.has(new_cfg, "model.hyperparams.lr")


def test_config_keys_values_items():
    cfg = {"a": 1, "b": {"c": 2}}

    assert set(config.keys(cfg)) == {"a", "b"}
    assert set(config.keys(cfg, flat=True)) == {"a", "b.c"}

    assert set(config.values(cfg, flat=True)) == {1, 2}

    items_flat = dict(config.items(cfg, flat=True))
    assert items_flat == {"a": 1, "b.c": 2}


def test_config_validate_export(tmp_path):
    cfg = {"model": {"name": "yolo"}, "dataset": {"path": "/data"}}

    # validate
    assert config.validate(cfg, ["model.name", "dataset.path"])
    try:
        config.validate(cfg, ["model.missing"])
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # export dict
    exported_p = tmp_path / "exp.json"
    config.export(cfg, exported_p, overwrite=True, verbose=False)
    assert exported_p.exists()


def test_config_class_wrapper(tmp_path):
    cfg_p = tmp_path / "test.yaml"
    c_obj = Config.create_default(cfg_p, overwrite=True)
    c_obj.read(verbose=False)

    assert c_obj.get("model.name") == "yolov8n"
    c_obj.set("model.epochs", 200)
    assert c_obj.get("model.epochs") == 200
    assert c_obj.has("model.epochs")

    c_obj.export_file("exported_class", ".toml", output_dir=tmp_path, overwrite=True)
    assert (tmp_path / "exported_class.toml").exists()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp_p = Path(td)
        test_config_load_save_convert(tmp_p)
        test_config_create_defaults(tmp_p)
        test_config_merge_update()
        test_config_access_and_modify()
        test_config_keys_values_items()
        test_config_validate_export(tmp_p)
        test_config_class_wrapper(tmp_p)
        print("ALL KLYGO.CONFIG 16 APIS AND CONFIG CLASS TESTS PASSED SUCCESSFULLY!")
