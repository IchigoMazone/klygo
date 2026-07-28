"""
File Kiểm Thử Toàn Diện Module `klygo.archive`
=================================================
Chạy file này bằng lệnh:
    python test/archive/test_klygo_archive.py

File này lần lượt kiểm thử tất cả 22 hàm và tính năng của klygo.archive:
1. compress           12. remove
2. extract            13. merge
3. extract_file       14. split_by_size
4. extract_matching   15. convert
5. list_files         16. recompress
6. iter_files         17. compare
7. search             18. detect_format
8. get_info           19. is_archive
9. test               20. open (open_archive)
10. verify            21. ArchiveFile
11. add               22. human_size
"""

import os
import shutil
import pathlib
import sys

# Thiết lập UTF-8 cho console stdout để in tiếng Việt không bị lỗi cp1252 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Đảm bảo import được module klygo từ thư mục test/archive
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import klygo.archive as ar


def run_all_tests():
    print("=" * 60)
    print("  BAT DAU KIEM THU TOAN BO HAM TRONG KLYGO.ARCHIVE")
    print("=" * 60)

    # Thư mục tạm phục vụ kiểm thử bên trong test/
    work_dir = pathlib.Path(__file__).parent / "test_env"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Tạo một số file mẫu để test
    sample_dir = work_dir / "sample_folder"
    sample_dir.mkdir()
    (sample_dir / "file1.txt").write_text("Noi dung file 1", encoding="utf-8")
    (sample_dir / "file2.png").write_text("Noi dung file 2 png", encoding="utf-8")
    (sample_dir / "sub_folder").mkdir()
    (sample_dir / "sub_folder" / "file3.json").write_text('{"key": "value"}', encoding="utf-8")

    print("\n[OK] Da khoi tao du lieu mau thanh cong!")

    # -------------------------------------------------------------
    # 1. HÀM human_size
    # -------------------------------------------------------------
    print("\n--- [1/22] Testing: human_size ---")
    size_str = ar.human_size(1048576)
    print(f"human_size(1048576) -> {size_str}")
    assert size_str == "1.00 MB"

    # -------------------------------------------------------------
    # 2. HÀM compress
    # -------------------------------------------------------------
    print("\n--- [2/22] Testing: compress ---")
    zip_path = work_dir / "test_archive.zip"
    ar.compress(sample_dir, zip_path, overwrite=True, verbose=True)
    print(f"compress -> Tao file thanh cong: {zip_path}")
    assert zip_path.exists()

    # Nén dạng TAR.GZ
    tgz_path = work_dir / "test_archive.tar.gz"
    ar.compress(sample_dir, tgz_path, compresslevel=9, overwrite=True, verbose=True)
    print(f"compress (tar.gz) -> Tao file thanh cong: {tgz_path}")
    assert tgz_path.exists()

    # -------------------------------------------------------------
    # 3. HÀM detect_format
    # -------------------------------------------------------------
    print("\n--- [3/22] Testing: detect_format ---")
    fmt_zip = ar.detect_format(zip_path)
    fmt_tgz = ar.detect_format(tgz_path)
    print(f"detect_format('{zip_path.name}') -> '{fmt_zip}'")
    print(f"detect_format('{tgz_path.name}') -> '{fmt_tgz}'")
    assert fmt_zip == "zip"
    assert fmt_tgz == "tar.gz"

    # -------------------------------------------------------------
    # 4. HÀM is_archive
    # -------------------------------------------------------------
    print("\n--- [4/22] Testing: is_archive ---")
    print(f"is_archive('{zip_path.name}') -> {ar.is_archive(zip_path)}")
    print(f"is_archive('non_exist.xyz') -> {ar.is_archive('non_exist.xyz')}")
    assert ar.is_archive(zip_path) is True
    assert ar.is_archive("non_exist.xyz") is False

    # -------------------------------------------------------------
    # 5. HÀM list_files
    # -------------------------------------------------------------
    print("\n--- [5/22] Testing: list_files ---")
    files_list = ar.list_files(zip_path)
    print(f"list_files -> Tim thay {len(files_list)} file: {files_list}")
    assert len(files_list) >= 3

    # -------------------------------------------------------------
    # 6. HÀM iter_files
    # -------------------------------------------------------------
    print("\n--- [6/22] Testing: iter_files ---")
    iter_items = list(ar.iter_files(zip_path))
    print(f"iter_files -> Duyet generator thu duoc {len(iter_items)} phan tu")
    assert len(iter_items) == len(files_list)

    # -------------------------------------------------------------
    # 7. HÀM search
    # -------------------------------------------------------------
    print("\n--- [7/22] Testing: search ---")
    png_files = ar.search(zip_path, "*.png")
    print(f"search('*.png') -> {png_files}")
    assert len(png_files) >= 1

    regex_files = ar.search(zip_path, r".*file\d+\.txt", regex=True)
    print(f"search(regex=True) -> {regex_files}")
    assert len(regex_files) >= 1

    # -------------------------------------------------------------
    # 8. HÀM get_info
    # -------------------------------------------------------------
    print("\n--- [8/22] Testing: get_info ---")
    info = ar.get_info(zip_path)
    print(f"get_info -> {info}")
    assert info["format"] == "zip"
    assert info["file_count"] >= 3

    # -------------------------------------------------------------
    # 9. HÀM test
    # -------------------------------------------------------------
    print("\n--- [9/22] Testing: test ---")
    is_ok = ar.test(zip_path)
    print(f"test -> Kiem tra tinh toan ven: {is_ok}")
    assert is_ok is True

    # -------------------------------------------------------------
    # 10. HÀM verify
    # -------------------------------------------------------------
    print("\n--- [10/22] Testing: verify ---")
    verify_report = ar.verify(zip_path)
    print(f"verify -> Bao cao xac minh: {verify_report}")
    assert verify_report["valid"] is True

    # -------------------------------------------------------------
    # 11. HÀM extract
    # -------------------------------------------------------------
    print("\n--- [11/22] Testing: extract ---")
    out_extract = work_dir / "extracted_all"
    ar.extract(zip_path, output_dir=out_extract, overwrite=True, verbose=True)
    print(f"extract -> Da giai nen ra thu muc: {out_extract}")
    assert out_extract.exists()

    # -------------------------------------------------------------
    # 12. HÀM extract_file
    # -------------------------------------------------------------
    print("\n--- [12/22] Testing: extract_file ---")
    out_single = work_dir / "extracted_single"
    target_file = files_list[0]
    ar.extract_file(zip_path, filename=target_file, output_dir=out_single, overwrite=True)
    print(f"extract_file('{target_file}') -> Da giai nen file don le thanh cong!")
    assert (out_single / pathlib.Path(target_file).name).exists()

    # -------------------------------------------------------------
    # 13. HÀM extract_matching
    # -------------------------------------------------------------
    print("\n--- [13/22] Testing: extract_matching ---")
    out_matching = work_dir / "extracted_matching"
    ar.extract_matching(zip_path, pattern="*.png", output_dir=out_matching, overwrite=True)
    print(f"extract_matching('*.png') -> Da giai nen cac file khop mau!")

    # -------------------------------------------------------------
    # 14. HÀM add
    # -------------------------------------------------------------
    print("\n--- [14/22] Testing: add ---")
    new_file = work_dir / "extra_file.txt"
    new_file.write_text("File them moi", encoding="utf-8")
    ar.add(zip_path, files=[new_file], on_conflict="rename", verbose=True)
    files_after_add = ar.list_files(zip_path)
    print(f"add -> So luong file sau khi them: {len(files_after_add)}")
    assert len(files_after_add) == len(files_list) + 1

    # -------------------------------------------------------------
    # 15. HÀM remove
    # -------------------------------------------------------------
    print("\n--- [15/22] Testing: remove ---")
    ar.remove(zip_path, files=["extra_file.txt"])
    files_after_remove = ar.list_files(zip_path)
    print(f"remove -> So luong file sau khi xoa 'extra_file.txt': {len(files_after_remove)}")
    assert "extra_file.txt" not in files_after_remove

    # -------------------------------------------------------------
    # 16. HÀM merge
    # -------------------------------------------------------------
    print("\n--- [16/22] Testing: merge ---")
    zip2_path = work_dir / "test_archive2.zip"
    ar.compress(sample_dir, zip2_path, overwrite=True, verbose=False)

    merged_path = work_dir / "merged_output.zip"
    ar.merge([zip_path, zip2_path], output_path=merged_path, overwrite=True, verbose=True)
    print(f"merge -> Da gop thanh cong ra file: {merged_path}")
    assert merged_path.exists()

    # -------------------------------------------------------------
    # 17. HÀM split_by_size
    # -------------------------------------------------------------
    print("\n--- [17/22] Testing: split_by_size ---")
    parts = ar.split_by_size(zip_path, size=0.001, output_dir=work_dir / "parts", overwrite=True, verbose=True)
    print(f"split_by_size -> Da chia thanh {len(parts)} part(s): {parts}")
    assert len(parts) >= 1

    # -------------------------------------------------------------
    # 18. HÀM convert
    # -------------------------------------------------------------
    print("\n--- [18/22] Testing: convert ---")
    converted_tgz = work_dir / "converted_archive.tar.gz"
    ar.convert(zip_path, converted_tgz, overwrite=True, verbose=True)
    print(f"convert (.zip -> .tar.gz) -> Da chuyen doi thanh cong: {converted_tgz}")
    assert converted_tgz.exists()

    # -------------------------------------------------------------
    # 19. HÀM recompress
    # -------------------------------------------------------------
    print("\n--- [19/22] Testing: recompress ---")
    recompressed_path = work_dir / "recompressed.zip"
    ar.recompress(zip_path, recompressed_path, compresslevel=9, overwrite=True, verbose=True)
    print(f"recompress (level 9) -> Da nen lai thanh cong: {recompressed_path}")
    assert recompressed_path.exists()

    # -------------------------------------------------------------
    # 20. HÀM compare
    # -------------------------------------------------------------
    print("\n--- [20/22] Testing: compare ---")
    diff_report = ar.compare(zip_path, merged_path)
    print(f"compare -> Ket qua so sanh 2 archive:\n  {diff_report}")
    assert "common_files" in diff_report

    # -------------------------------------------------------------
    # 21. HÀM open (open_archive) & OOP ArchiveFile
    # -------------------------------------------------------------
    print("\n--- [21/22 & 22/22] Testing: open / ArchiveFile ---")
    with ar.open(zip_path) as archive:
        print(f"open() Context Manager -> Format: {archive.format}")
        print(f"  + list_files(): {len(archive.list_files())} files")
        print(f"  + get_info(): uncompressed size = {archive.get_info()['human_uncompressed_size']}")
        print(f"  + test(): {archive.test()}")

    print("\n" + "=" * 60)
    print("  TAT CA 22 HAM TRONG KLYGO.ARCHIVE DA CHAY HOAN HAO!")
    print("=" * 60)

    # Dọn dẹp thư mục tạm
    if work_dir.exists():
        shutil.rmtree(work_dir)
        print("\n[OK] Da don dep thu muc tam test_env thanh cong!")


if __name__ == "__main__":
    run_all_tests()
