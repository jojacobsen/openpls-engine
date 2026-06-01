# Releasing `openpls-engine`

This is the maintainer workflow for cutting a release and publishing to PyPI.

## Versioning policy

- Semantic Versioning (`MAJOR.MINOR.PATCH`), with PEP 440 pre-release
  suffixes (`a1`, `b1`, `rc1`).
- Public API is stable as of `1.0.0`; breaking changes require a major
  bump.
- The single source of truth for the version is `version` in
  `pyproject.toml`. The runtime value `openpls.__version__` reads it via
  `importlib.metadata` on the installed package.

## One-time setup (first release only)

The workflow publishes every tagged release (stable or pre-release like
`a1`, `b1`, `rc1`) directly to PyPI. PyPI handles pre-release semantics
natively: `pip install openpls-engine` skips pre-releases by default;
users opt in with `pip install --pre openpls-engine` or by pinning
`openpls-engine==X.Y.Za1`.

1. Configure a **pending trusted publisher** on PyPI before the first
   release exists. Sign in at <https://pypi.org/manage/account/publishing/>
   and add:
   - PyPI Project Name: `openpls-engine`
   - Owner: `jojacobsen`
   - Repository name: `openpls-engine`
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
2. In the GitHub repo settings, create an environment named `pypi`
   (Settings, Environments, New environment). Optionally add required
   reviewers to force a manual approval gate before each publish.

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
   - Publishes to PyPI (pre-releases and stable use the same target)
   - Creates a GitHub Release with auto-generated notes

## Sanity check after publish

```sh
pip install openpls-engine==X.Y.Z
python -c "import openpls; print(openpls.__version__)"
```

## Rolling back

PyPI does not allow re-uploading the same version. If a release is broken,
yank it on PyPI and publish a `X.Y.Z+1` patch with the fix.
