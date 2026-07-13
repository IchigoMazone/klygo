import json
import pathlib
import klygo.io as io

def test_io_operations():
    data = {"key": "value", "nested": {"a": 1}}

    # 1. Test basic write
    io.write_json("test_write.json", data, overwrite=True)

    # 2. Test overwrite: should raise FileExistsError if False and file exists
    try:
        io.write_json("test_write.json", data, overwrite=False)
        raise AssertionError("Should have raised FileExistsError")
    except FileExistsError:
        pass

    # 3. Test overwrite=True works
    io.write_json("test_write.json", data, overwrite=True)

    # 4. Test read progress bar and contents
    r_json = io.read_json("test_write.json")
    assert r_json == data

    # 5. Test auto write / read
    io.write_file("test_write.yaml", data, overwrite=True)
    r_yaml = io.read_file("test_write.yaml")
    assert dict(r_yaml) == data

    # 6. Test Config read() and to_dict() with no default root block
    cfg = io.Config("test_write.yaml")
    box_data = cfg.read()
    assert box_data.key == "value"
    assert cfg.to_dict() == data
    assert json.loads(cfg.to_json()) == data

    # 7. Test Config.create_default()
    cfg_def = io.Config.create_default("test_default_config.yaml", overwrite=True)
    box_def = cfg_def.read()
    assert box_def.model.name == "yolov8n"
    assert box_def.default.root == "./data"

    # 8. Test Config export overwrite
    cfg_def.export_file("exported_config", ".toml", overwrite=True)
    try:
        cfg_def.export_file("exported_config", ".toml", overwrite=False)
        raise AssertionError("Should have raised FileExistsError on export")
    except FileExistsError:
        pass

    # Clean up
    for f in ["test_write.json", "test_write.yaml", "test_default_config.yaml", "exported_config.toml"]:
        p = pathlib.Path(f)
        if p.exists():
            p.unlink()

if __name__ == "__main__":
    test_io_operations()
    print("ALL IO TESTS PASSED SUCCESSFULLY!")
