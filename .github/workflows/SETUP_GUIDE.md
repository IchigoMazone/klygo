# Hướng dẫn setup GitHub Actions → PyPI

## Bước 1: Setup Trusted Publisher trên PyPI (không cần token)

1. Vào https://pypi.org/manage/account/publishing/
2. Thêm publisher mới:
   - **PyPI Project name**: `klygo`
   - **Owner**: `IchigoMazone` (GitHub username)
   - **Repository name**: `klygo`
   - **Workflow name**: `build_wheels.yml`
   - **Environment name**: `pypi`
3. Click **Add**

## Bước 2: Tạo Environment trên GitHub

1. Vào repo GitHub → Settings → Environments → New environment
2. Đặt tên: `pypi`
3. (Tùy chọn) Thêm required reviewers để bảo mật

## Bước 3: Publish bằng cách push tag

```bash
# Bump cùng một version trong pyproject.toml và klygo/__init__.py
# Rồi:
git add -A
git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

GitHub Actions sẽ tự động:
1. Build pure-Python wheel và source distribution trên Linux.
2. Upload các package lên PyPI.

## Wheel files sẽ có

| Loại | File |
|------|------|
| Pure Python wheel | `klygo-x.y.z-py3-none-any.whl` |
| Source distribution | `klygo-x.y.z.tar.gz` |
