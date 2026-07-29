import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import pathlib
import tempfile
from pathlib import Path
from PIL import Image

try:
    import pytest
except ImportError:
    class _PytestFallback:
        def raises(self, expected_exception, match=None):
            class _RaisesContext:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {expected_exception} but nothing was raised")
                    return issubclass(exc_type, expected_exception)
            return _RaisesContext()
    pytest = _PytestFallback()

import klygo.files as files
import klygo.media as media
from klygo.config import Config


def test_files_data_operations(tmp_path):
    data = {"name": "klygo", "value": 123, "nested": {"a": [1, 2]}}

    # 1. save and load YAML
    yaml_path = tmp_path / "test.yaml"
    files.save(yaml_path, data, overwrite=True)
    assert files.exists(yaml_path)
    assert files.is_file(yaml_path)
    assert not files.is_dir(yaml_path)
    loaded_yaml = files.load(yaml_path)
    assert dict(loaded_yaml) == data

    # 2. save and load JSON
    json_path = tmp_path / "test.json"
    files.save(json_path, data, overwrite=True)
    assert files.load(json_path) == data

    # 3. overwrite check
    with pytest.raises(FileExistsError):
        files.save(json_path, data, overwrite=False)

    # 4. convert JSON to TOML
    toml_path = tmp_path / "test.toml"
    converted_path = files.convert(json_path, toml_path, overwrite=True)
    assert converted_path.exists()
    assert files.load(toml_path) == data

    # 5. TXT and LOG
    txt_path = tmp_path / "test.txt"
    files.save(txt_path, ["line1", "line2"], overwrite=True)
    assert files.load(txt_path) == "line1\nline2\n"
    assert files.load(txt_path, as_lines=True) == ["line1", "line2"]

    # 6. CSV
    csv_path = tmp_path / "test.csv"
    rows = [{"id": "1", "val": "a"}, {"id": "2", "val": "b"}]
    files.save(csv_path, rows, overwrite=True)
    assert files.load(csv_path) == rows

    # 7. INI / CFG
    ini_path = tmp_path / "test.ini"
    ini_data = {"section1": {"key1": "val1"}, "section2": {"key2": "val2"}}
    files.save(ini_path, ini_data, overwrite=True)
    assert files.load(ini_path) == ini_data

    # 8. ENV
    env_path = tmp_path / ".env"
    env_data = {"PORT": "8080", "HOST": "localhost"}
    files.save(env_path, env_data, overwrite=True)
    assert files.load(env_path) == env_data

    # 9. XML
    xml_path = tmp_path / "test.xml"
    xml_data = {"root": {"item": "val"}}
    files.save(xml_path, xml_data, overwrite=True)
    assert files.load(xml_path) == xml_data

    # 10. Pickle
    pkl_path = tmp_path / "test.pkl"
    files.save(pkl_path, data, overwrite=True)
    assert files.load(pkl_path) == data


def test_media_image_loading(tmp_path):
    img_dir = tmp_path / "images"
    files.mkdir(img_dir)
    img_path = img_dir / "test.png"
    Image.new("RGB", (10, 10), "red").save(img_path)

    # load image via klygo.media
    imgs = media.load(img_path)
    assert len(imgs) == 1
    assert isinstance(imgs[0], Image.Image)

    # load image dir via klygo.media
    imgs_dir = media.load(img_dir)
    assert len(imgs_dir) == 1


def test_files_filesystem_operations(tmp_path):
    sub = tmp_path / "sub"
    files.mkdir(sub)
    assert files.is_dir(sub)

    f1 = sub / "file1.txt"
    f2 = sub / "file2.json"
    files.save(f1, "hello", overwrite=True)
    files.save(f2, {"a": 1}, overwrite=True)

    # list and find
    items = files.list(sub)
    assert len(items) == 2
    found = files.find(sub, "*.txt")
    assert len(found) == 1
    assert found[0].name == "file1.txt"

    # walk
    walked = list(files.walk(sub))
    assert len(walked) >= 1

    # copy
    copied = files.copy(f1, sub / "file1_copy.txt")
    assert files.exists(copied)

    # move
    moved = files.move(copied, sub / "file1_moved.txt")
    assert files.exists(moved)
    assert not files.exists(copied)

    # rename
    renamed = files.rename(moved, "file1_renamed.txt")
    assert files.exists(renamed)

    # remove
    files.remove(renamed)
    assert not files.exists(renamed)


def test_files_info_size_hash_compare(tmp_path):
    f1 = tmp_path / "file1.txt"
    f2 = tmp_path / "file2.txt"
    files.save(f1, "content", overwrite=True)
    files.save(f2, "content", overwrite=True)

    # path helpers
    assert files.name(f1) == "file1.txt"
    assert files.stem(f1) == "file1"
    assert files.extension(f1) == ".txt"
    assert files.parent(f1) == tmp_path

    # info & size & hash
    info_dict = files.info(f1)
    assert info_dict["name"] == "file1.txt"
    assert info_dict["size"] > 0
    assert info_dict["hash"] is not None

    assert files.size(f1) > 0
    assert isinstance(files.size(f1, human=True), str)
    h1 = files.hash(f1)
    assert len(h1) == 32

    # compare
    assert files.compare(f1, f2, by="hash")
    assert files.compare(f1, f2, by="content")


def test_files_download(tmp_path):
    out_file = tmp_path / "python_logo.png"
    # Download a small public file to test download
    downloaded = files.download("https://www.python.org/static/img/python-logo.png", out_file, overwrite=True)
    assert files.exists(downloaded)
    assert files.size(downloaded) > 0


def test_config_module(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg = Config.create_default(cfg_path, overwrite=True)
    assert files.exists(cfg_path)

    data = cfg.read(verbose=False)
    assert data.model.name == "yolov8n"
    assert cfg.to_dict()["model"]["name"] == "yolov8n"
    assert json.loads(cfg.to_json())["model"]["name"] == "yolov8n"

    cfg.export_file("exported", ".json", output_dir=tmp_path, overwrite=True)
    exported = tmp_path / "exported.json"
    assert files.exists(exported)
    assert files.load(exported)["model"]["name"] == "yolov8n"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp_p = Path(td)
        test_files_data_operations(tmp_p)
        test_media_image_loading(tmp_p)
        test_files_filesystem_operations(tmp_p)
        test_files_info_size_hash_compare(tmp_p)
        try:
            test_files_download(tmp_p)
        except Exception as e:
            print(f"Skipping network download test: {e}")
        test_config_module(tmp_p)
        print("ALL KLYGO.FILES AND KLYGO.CONFIG TESTS PASSED SUCCESSFULLY!")

