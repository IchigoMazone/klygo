import json
import pathlib
import klygo.io as io

data = {"key": "value", "nested": {"a": 1}}

# 1. Test basic write
io.write_json("test_write.json", data, overwrite=True)

# 2. Test overwrite: should raise FileExistsError if False and file exists
try:
    io.write_json("test_write.json", data, overwrite=False)
    raise AssertionError("Should have raised FileExistsError")
except FileExistsError:
    print("Caught expected FileExistsError for write_json")

# 3. Test overwrite=True works
io.write_json("test_write.json", data, overwrite=True)

# 4. Test read progress bar and contents
r_json = io.read_json("test_write.json")
assert r_json == data

# 5. Test auto write / read
io.write_file("test_write.yaml", data, overwrite=True)
r_yaml = io.read_file("test_write.yaml")
assert dict(r_yaml) == data

# 6. Test Config export overwrite
cfg = io.Config("test_write.yaml")
cfg.imread()

cfg.export_file("exported_config", ".toml", overwrite=True)
try:
    cfg.export_file("exported_config", ".toml", overwrite=False)
    raise AssertionError("Should have raised FileExistsError on export")
except FileExistsError:
    print("Caught expected FileExistsError for Config.export_file")

# Clean up
for f in ["test_write.json", "test_write.yaml", "exported_config.toml"]:
    p = pathlib.Path(f)
    if p.exists():
        p.unlink()

print("\nALL IO OVERWRITE & VERBOSE TESTS PASSED!")
