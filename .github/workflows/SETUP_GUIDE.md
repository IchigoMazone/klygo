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
# Bump version trong pyproject.toml, setup.py, __init__.py
# Rồi:
git add -A
git commit -m "chore: release v2.0.3"
git tag v2.0.3
git push origin main --tags
```

GitHub Actions sẽ tự động:
1. Build wheel trên Windows / Linux / macOS cho Python 3.10–3.13
2. Repair wheel (manylinux tag cho Linux, bundle dylib cho macOS)
3. Upload tất cả lên PyPI

## Wheel files sẽ có

| OS | File |
|----|------|
| Windows | `klygo-x.y.z-cp31x-cp31x-win_amd64.whl` |
| Linux | `klygo-x.y.z-cp31x-cp31x-manylinux_2_17_x86_64.whl` |
| macOS Intel | `klygo-x.y.z-cp31x-cp31x-macosx_10_9_x86_64.whl` |
| macOS Apple Silicon | `klygo-x.y.z-cp31x-cp31x-macosx_11_0_arm64.whl` |
