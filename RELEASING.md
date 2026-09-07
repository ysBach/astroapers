# Releasing astroapers

**Rehearse manually, then push the matching `vX.Y.Z` tag to publish.**
The tag must point to the exact commit that passed the rehearsal.

| Event | Build and test | Publish |
| --- | --- | --- |
| Branch push or pull request | Rust + installed wheel/sdist CI | Never |
| Manual `release.yml`, including on a tag | CI + full wheel matrix | Never |
| Matching `vX.Y.Z` tag push | CI + full wheel matrix | PyPI, crates.io, then GitHub |

This convention combines `imcombiners`' tag-triggered publishing, `reducers`'
early validation and reusable CI, and `pqfilt`'s installed-artifact checks.
`astroimred` uses the same release rules for PyPI; `astroapers` also publishes a
Rust crate. Cargo packages the validated commit again when publishing; the Python
jobs upload the exact wheel and sdist files that passed their checks.

## One-time setup

1. Create GitHub environments `pypi` and `crates-io`. Set deployment restrictions
   to **tags** matching `v*`, excluding branches. Configure required reviewers
   only if you want an additional approval before publication.
2. Register these trusted publishers in both registries:

   | Setting | PyPI | crates.io |
   | --- | --- | --- |
   | Package | `astroapers` | `astroapers` |
   | Owner/repository | `ysBach/astroapers` | `ysBach/astroapers` |
   | Workflow filename | `release.yml` | `release.yml` |
   | Environment | `pypi` | `crates-io` |

   Follow [PyPI setup](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
   and [crates.io setup](https://crates.io/docs/trusted-publishing). Both jobs
   use temporary OIDC credentials. If the crate does not exist yet, check the
   registry's first-publication requirements; an initial manual upload may be
   needed before its trusted publisher can be configured.
3. Enable Actions, run CI, and require its metadata, Rust, wheel, and sdist checks
   on `main` through branch protection or a ruleset. The existing `docs.yml`
   deployment remains independent of package releases.

## Each release

1. Choose the next version and set it in `Cargo.toml`, `pyproject.toml`, and the
   root `astroapers` entries in `Cargo.lock` and `uv.lock`. Move every change
   from `## Unreleased` into a nonempty dated entry:

   ```markdown
   ## X.Y.Z - YYYY-MM-DD

   - Describe the previous and new behavior.
   ```

   Replace the placeholders with the version and actual date. Empty `Unreleased`
   headings are allowed; the historical `## 0.1.3 (2026-07-15)` format also works.
   Use a project `.venv` with Python 3.11+ for release tooling:

   ```sh
   uv run --no-sync python scripts/check_release.py --tag vX.Y.Z
   ```

   The `../reducers` uv override is for local development. CI sets
   `UV_NO_SOURCES=true` and installs distribution artifacts with registry
   dependencies. Do not replace the development lock with a release-only lock,
   and never run project synchronization against `~/.venvs/ysbpy`.
2. Review and commit the intended files, including new files. Push the commit
   and run a rehearsal:

   ```sh
   uv run --no-sync python scripts/check_release.py --require-clean
   git push origin main
   gh workflow run release.yml --ref main
   gh run list --workflow release.yml
   gh run view RUN_ID
   ```

   Replace `RUN_ID` with the rehearsal run. All build/test jobs must pass and
   publication jobs must be skipped. Early branch rehearsals allow `Unreleased`.
3. Copy the successful run's full commit SHA, then tag that exact commit:

   ```sh
   gh run view RUN_ID --json headSha --jq .headSha
   git switch --detach REHEARSED_COMMIT_SHA
   uv run --no-sync python scripts/check_release.py --tag vX.Y.Z --require-clean
   git tag -a vX.Y.Z -m "Release X.Y.Z"
   git push origin vX.Y.Z
   git switch main
   ```

   **Pushing the tag publishes.** Replace `X.Y.Z` and `REHEARSED_COMMIT_SHA`;
   do not tag a later commit that has not passed the rehearsal.
4. Verify both registry version pages and the GitHub release. The release
   attaches the tested wheels and sdist and uses the validated changelog entry.

## What is checked

CI checks matching manifest and root-lock versions, the release-checker tests,
and formatting/linting of release tooling. A tag also requires an exact stable
version, valid dated notes, and no remaining `Unreleased` changes. Notes are
extracted before either registry receives credentials.

Rust checks run the tests without Python features, compile the extension feature,
reject NumPy/PyO3 in the Rust-only dependency graph, and dry-run crate publication.
Python checks install a wheel on 3.10 and the sdist on 3.13 into fresh environments.
They verify installed metadata and module locations, measure a circular aperture,
and run the test suite with published reducers, Photutils, SEP, Matplotlib, SciPy,
and astro-ndslice dependencies. The tested sdist is reused for publication.

The release matrix builds and tests **25 CPython wheels**: Python 3.10–3.14 on
Linux and macOS x86-64/ARM64, plus Windows x86-64. Linux wheels use native
manylinux2014 containers (glibc 2.17+). Wheels remain specific to each Python
version; this workflow does not change the extension's ABI. These are configured
targets; the first successful hosted run confirms the actual platform coverage.

## Failure recovery

- **Before publishing:** fix the source or metadata, commit, and rehearse again.
  If source changes are needed after pushing a release tag, use a new version/tag.
  A manual run, even on a tag, remains build-only.
- **Registry setup or network failure:** fix the external issue and rerun failed
  jobs in the original tag-push run, preserving successful jobs and artifacts.
  PyPI's `--check-url` skips identical uploaded files and rejects conflicting
  content; see [uv retry behavior](https://docs.astral.sh/uv/guides/package/).
- **Crate uploaded but its job later failed:** inspect crates.io before retrying;
  `cargo publish` is not idempotent. Verify the uploaded crate against the tagged
  source, finish any failed PyPI job, then create the GitHub release from the
  original run's `wheels-*`, `ci-sdist`, and `release-notes` artifacts if necessary.
- **Only GitHub release creation failed:** rerun that job. If a release already
  exists, inspect its tag, notes, and assets and complete it from the original
  artifacts. Changed package content needs a new version; never move a published tag.

The two registries and GitHub do not form one atomic transaction. Local checks
cannot verify registry permissions, environment restrictions, or hosted builds;
confirm those during setup and the first rehearsal.
