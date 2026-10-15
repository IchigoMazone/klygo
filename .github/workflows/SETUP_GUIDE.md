# GitHub Actions and PyPI release setup

## 1. Configure a trusted publisher on PyPI

In [PyPI publishing settings](https://pypi.org/manage/account/publishing/), add a publisher with these values:

- PyPI project: `klygo`
- GitHub owner: `IchigoMazone`
- Repository: `klygo`
- Workflow: `build_wheels.yml`
- Environment: `pypi`

## 2. Create the GitHub environment

In the repository, open Settings > Environments > New environment and create `pypi`. Add required reviewers if your release process needs them.

## 3. Publish a release

Update the version in `pyproject.toml` and `klygo/__init__.py`, commit the release, and push a matching tag:

```bash
git add -A
git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --tags
```

The workflow builds a Python wheel and source distribution, then publishes them to PyPI through trusted publishing.
