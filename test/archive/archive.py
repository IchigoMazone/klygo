from klygo.archive import *

if __name__ == "__main__":

    src = "test/data.zip"

    # 1. list_files
    files = list_files(src)
    print(f"list_files: {len(files)} files")

    # 2. get_info
    info = get_info(src)
    print(f"get_info: {info}")

    # 3. test integrity
    ok = test(src)
    print(f"test: {ok}")

    # 4. search
    found = search(src, "images/frame_1*.jpg")
    print(f"search 'frame_1*.jpg': {len(found)} matches -> {found[:3]}")

    # 5. extract_file
    extract_file(src, files[0], output="test_single", overwrite=True)
    print(f"extract_file: OK -> {files[0]}")

    # 6. compress
    compress("klygo/archive", "test_new.zip", overwrite=True, verbose=True)
    names = list_files("test_new.zip")
    print(f"archive contents: {names}")

    # 7. add
    add("test_new.zip", "klygo/__init__.py", verbose=True)
    print("after add:", list_files("test_new.zip"))

    # 8. remove
    to_remove = list_files("test_new.zip")[0]
    remove("test_new.zip", to_remove)
    print(f"remove {to_remove!r}: OK -> remaining:", list_files("test_new.zip"))

    # 9. merge
    compress("klygo/validators", "test_v.zip", overwrite=True, verbose=False)
    merge(["test_new.zip", "test_v.zip"], "test_merged.zip", overwrite=True, verbose=True)

    # 10. split
    parts = split(src, size=5_000_000, output_dir="test_parts", overwrite=True, verbose=True)
    print(f"split: {len(parts)} parts -> {parts}")

    print("\nALL TESTS PASSED")
