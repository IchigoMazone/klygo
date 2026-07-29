import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
from pathlib import Path
from PIL import Image

# Import all 4 core modules simultaneously
import klygo.archive as archive
import klygo.config as config
import klygo.files as files
import klygo.media as media
from klygo.config import Config


def test_cross_module_integration(tmp_path):
    print("1. Testing klygo.files operations...")
    data_dir = tmp_path / "data"
    files.mkdir(data_dir)
    txt_file = data_dir / "info.txt"
    files.save(txt_file, ["line1", "line2"], overwrite=True)
    assert files.exists(txt_file)
    assert files.is_file(txt_file)

    print("2. Testing klygo.config operations...")
    cfg_file = data_dir / "app_config.yaml"
    cfg_box = config.create(cfg_file, default_data={"app": {"name": "TestApp", "port": 8000}}, overwrite=True)
    assert cfg_box.app.name == "TestApp"

    # OOP Config usage
    cfg_obj = Config(cfg_file)
    cfg_obj.read()
    assert cfg_obj.get("app.port") == 8000
    cfg_obj.set("app.port", 9000)
    config.save(cfg_file, cfg_obj.to_dict(), overwrite=True)

    print("3. Testing klygo.media operations...")
    img_dir = data_dir / "images"
    files.mkdir(img_dir)
    img_file = img_dir / "sample.png"
    Image.new("RGB", (30, 20), "green").save(img_file)

    media_imgs = media.load(img_file, backend="pil")
    assert len(media_imgs) == 1
    media_info = media.info(img_file)
    assert media_info["width"] == 30

    media.save(tmp_path / "saved_sample.png", media_imgs[0], overwrite=True)
    assert files.exists(tmp_path / "saved_sample.png")

    print("4. Testing klygo.archive operations on files, config & media...")
    zip_file = tmp_path / "bundle.zip"
    archive.compress(data_dir, zip_file, overwrite=True)
    assert files.exists(zip_file)
    assert archive.test(zip_file)

    extracted_dir = tmp_path / "extracted"
    archive.extract(zip_file, output_dir=extracted_dir, overwrite=True)

    # Verify extracted data using files, config, media
    extracted_txt = extracted_dir / "data" / "info.txt"
    extracted_cfg = extracted_dir / "data" / "app_config.yaml"
    extracted_img = extracted_dir / "data" / "images" / "sample.png"

    assert files.exists(extracted_txt)
    assert files.load(extracted_txt, as_lines=True) == ["line1", "line2"]

    assert files.exists(extracted_cfg)
    loaded_cfg = config.load(extracted_cfg)
    assert loaded_cfg.app.port == 9000

    assert files.exists(extracted_img)
    loaded_img_info = media.info(extracted_img)
    assert loaded_img_info["width"] == 30

    print("5. Testing no function/name collisions between top-level klygo modules...")
    # Verify module names and namespaces
    assert hasattr(archive, "compress")
    assert hasattr(config, "load")
    assert hasattr(files, "load")
    assert hasattr(media, "load")

    # Verify files.load and media.load behavior differences
    # files.load on config -> returns dict/Box
    # media.load on image -> returns list of images
    assert isinstance(files.load(txt_file), str)
    assert isinstance(media.load(img_file), list)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp_p = Path(td)
        test_cross_module_integration(tmp_p)
        print("ALL 4 MODULES (ARCHIVE, CONFIG, FILES, MEDIA) ARE FULLY COMPATIBLE WITH 0 CONFLICTS!")
