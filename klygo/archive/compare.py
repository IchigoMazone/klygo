from pathlib import Path
from typing import Union, Dict, List

from klygo.archive.backend import get_backend


def compare(archive1: Union[str, Path], archive2: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Tác dụng:
    - So sánh nội dung danh sách các file bên trong 2 file lưu trữ khác nhau.

    Đầu vào:
    - archive1 [str | Path]: Đường dẫn file archive thứ nhất.
    - archive2 [str | Path]: Đường dẫn file archive thứ hai.

    Đầu ra:
    - [Dict[str, List[str]]] Từ điển chứa kết quả so sánh:
      + 'added_files': Các file có trong archive2 nhưng không có trong archive1
      + 'removed_files': Các file có trong archive1 nhưng không có trong archive2
      + 'common_files': Các file xuất hiện ở cả 2 archive

    Ví dụ:
    >>> import klygo.archive as ar

    # Ví dụ 1: So sánh hai phiên bản file ZIP
    >>> diff = ar.compare("v1.zip", "v2.zip")
    >>> print("File mới thêm:", diff['added_files'])
    File mới thêm: ['new_feature.py', 'docs/guide.md']
    >>> print("File đã xóa:", diff['removed_files'])
    File đã xóa: ['deprecated.py']
    >>> print("File xuất hiện ở cả hai:", diff['common_files'])
    File xuất hiện ở cả hai: ['main.py', 'config.yaml']

    Nguồn: TrinhNhuNhat_28072026.
    """
    path1 = Path(archive1)
    path2 = Path(archive2)

    backend1 = get_backend(path1)
    backend2 = get_backend(path2)

    set1 = set(backend1.list_files(path1))
    set2 = set(backend2.list_files(path2))

    return {
        "added_files": sorted(list(set2 - set1)),
        "removed_files": sorted(list(set1 - set2)),
        "common_files": sorted(list(set1 & set2)),
    }
