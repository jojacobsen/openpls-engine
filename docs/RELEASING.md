# Releasing `openpls-engine`

This is the maintainer workflow for cutting a release and publishing to PyPI.

## Versioning policy

- Semantic Versioning (`MAJOR.MINOR.PATCH`), with PEP 440 pre-release
  suffixes (`a1`, `b1`, `rc1`).
- Pre-1.0 releases on the `0.x` line while the API stabilizes.
- The single source of truth for the version is `version` in
  `pyproject.toml`. The runtime value `plspm.__version__` reads it via
  `importlib.metadata` on the installed package.

## One-time setup (first release only)

1. Reserve the project name on PyPI by either:
   a) doing an initial upload from a maintainer machine with
      `twine upload`, or
   b) configuring "pending" trusted publishing at
      <https://pypi.org/manage/account/publishing/> before any release exists.
2. Add a **trusted publisher** for the project on PyPI:
   - PyPI project page, "Publishing" tab, "Add a new publisher"
   - Owner: `jojacobsen`
   - Repository name: `openpls-engine`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
3. Same on TestPyPI (<https://test.pypi.org>) with environment name
   `testpypi`. TestPyPI is used by the workflow for any pre-release
   (versions containing `a`, `b`, or `rc`).
4. In GitHub repo settings, create environments `pypi` and `testpypi`.
   Optionally add required reviewers on `pypi` to force a manual approval
   gate before stable releases publish.

## Cutting a release

1. Update `CHANGELOG.md`: move `[Unreleased]` entries into a new section
   for the version, set the release date, and update the compare-link
   footers.
2. Bump `version` in `pyproject.toml` to the target version.
3. Commit with message `chore(release): vX.Y.Z`.
4. Tag the commit:
   ```sh
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
5. The `Release` GitHub Actions workflow takes it from here:
   - Builds sdist + wheel
   - Verifies the tag matches the pyproject version
   - Publishes to TestPyPI (pre-releases) or PyPI (stable)
   - Creates a GitHub Release with auto-generated notes

## Sanity check after publish

```sh
pip install openpls-engine==X.Y.Z
python -c "import plspm; print(plspm.__version__)"
```

## Rolling back

PyPI does not allow re-uploading the same version. If a release is broken,
yank it on PyPI and publish a `X.Y.Z+1` patch with the fix.
