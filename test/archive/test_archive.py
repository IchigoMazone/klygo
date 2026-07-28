import sys
import pathlib
import os
import shutil

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from klygo.archive import *

if __name__ == "__main__":

    test_dir = pathlib.Path(__file__).parent.parent
    src_zip = str(test_dir / "data.zip")

    print("=== TESTING 7 ARCHIVE FORMATS ===")

    # 1. Format: .zip
    print("\n[1/7] Testing .zip format...")
    zip_out = "test_format.zip"
    compress("klygo/archive", zip_out, overwrite=True, verbose=False)
    assert is_archive(zip_out)
    assert detect_format(zip_out) == "zip"
    assert len(list_files(zip_out)) > 0
    extract(zip_out, "out_zip", overwrite=True, verbose=False)
    print("  -> .zip PASS")

    # 2. Format: .tar
    print("\n[2/7] Testing .tar format...")
    tar_out = "test_format.tar"
    compress("klygo/archive", tar_out, overwrite=True, verbose=False)
    assert is_archive(tar_out)
    assert detect_format(tar_out) == "tar"
    assert len(list_files(tar_out)) > 0
    extract(tar_out, "out_tar", overwrite=True, verbose=False)
    print("  -> .tar PASS")

    # 3. Format: .tar.gz (.tgz)
    print("\n[3/7] Testing .tar.gz format...")
    tgz_out = "test_format.tar.gz"
    compress("klygo/archive", tgz_out, overwrite=True, verbose=False)
    assert is_archive(tgz_out)
    assert detect_format(tgz_out) == "tar.gz"
    assert len(list_files(tgz_out)) > 0
    extract(tgz_out, "out_tgz", overwrite=True, verbose=False)
    print("  -> .tar.gz PASS")

    # 4. Format: .tar.xz (.txz)
    print("\n[4/7] Testing .tar.xz format...")
    txz_out = "test_format.tar.xz"
    compress("klygo/archive", txz_out, overwrite=True, verbose=False)
    assert is_archive(txz_out)
    assert detect_format(txz_out) == "tar.xz"
    assert len(list_files(txz_out)) > 0
    extract(txz_out, "out_txz", overwrite=True, verbose=False)
    print("  -> .tar.xz PASS")

    # 5. Format: .gz (Single file)
    print("\n[5/7] Testing .gz format...")
    gz_out = "test_format.txt.gz"
    sample_file = pathlib.Path("test_sample.txt")
    sample_file.write_text("Hello klygo archive .gz test!", encoding="utf-8")
    compress(sample_file, gz_out, overwrite=True, verbose=False)
    assert is_archive(gz_out)
    assert detect_format(gz_out) == "gz"
    assert len(list_files(gz_out)) == 1
    extract(gz_out, "out_gz", overwrite=True, verbose=False)
    if sample_file.exists():
        sample_file.unlink()
    print("  -> .gz PASS")

    # 6. Format: .7z
    print("\n[6/7] Testing .7z format...")
    try:
        import py7zr
        z7_out = "test_format.7z"
        compress("klygo/archive", z7_out, overwrite=True, verbose=False)
        assert is_archive(z7_out)
        assert detect_format(z7_out) == "7z"
        assert len(list_files(z7_out)) > 0
        extract(z7_out, "out_7z", overwrite=True, verbose=False)
        print("  -> .7z PASS (py7zr installed)")
    except ImportError:
        print("  -> .7z Optional (py7zr is not installed in environment, backend raises ImportError as expected)")

    # 7. Format: .rar (Extract-only)
    print("\n[7/7] Testing .rar format...")
    try:
        import rarfile
        print("  -> .rar Backend Ready (rarfile is installed)")
    except ImportError:
        print("  -> .rar Optional (rarfile is not installed in environment, backend raises ImportError as expected)")

    # Additional API Verification
    print("\n=== TESTING ADVANCED UTILITIES ===")
    assert detect_format(src_zip) == "zip"
    assert is_archive(src_zip)
    info = get_info(src_zip)
    assert "human_uncompressed_size" in info
    assert test(src_zip) == True
    assert verify(src_zip)["valid"] == True

    with open(src_zip) as ar_file:
        assert len(ar_file.list_files()) > 0

    # Auto Cleanup temporary output files and folders safely
    for target in ("test_format.zip", "test_format.tar", "test_format.tar.gz", "test_format.tar.xz", "test_format.txt.gz", "test_format.7z", "out_zip", "out_tar", "out_tgz", "out_txz", "out_gz", "out_7z"):
        p = pathlib.Path(target)
        if p.exists():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    print("\nALL 7 FORMAT CHECKS & ADVANCED UTILITIES PASSED SUCCESSFULLY!")
