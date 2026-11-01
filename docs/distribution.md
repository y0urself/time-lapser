# Installing and distributing the package

The project builds an ordinary Python wheel and source distribution. Package
name: `castlestube-video-cleanup`; command: `castlestube-cleanup`.
Building the package does not publish it. Normal PR CI only validates; a separate
manually triggered release workflow creates the GitHub release and publishes to PyPI.

## pip installation on Windows

After a release has been published to PyPI, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "castlestube-video-cleanup[yolo]"
.\.venv\Scripts\castlestube-cleanup.exe --help
```

For MediaPipe replace `[yolo]` with `[mediapipe]`; use `[yolo,mediapipe]` for both,
or omit extras for plain dedupe. An isolated tool installation is another option:

```powershell
uv tool install --python 3.12 "castlestube-video-cleanup[yolo]"
castlestube-cleanup --help
```

If uv reports that the tools directory is missing from PATH, run
`uv tool update-shell` and open a new terminal. The above index commands require
an actual published release. Until then, install a checkout:

```powershell
py -3.12 -m pip install ".[yolo]"
```

Or install a locally built wheel (substitute the built version):

```powershell
py -3.12 -m pip install ".\dist\castlestube_video_cleanup-2.7.0-py3-none-any.whl[yolo]"
```

FFmpeg and FFprobe remain external executables; see [Windows setup](windows.md).
Model weights are a separate download. `pip` installs the selected Python backend
dependencies, not the `.pt` / `.task` files. A pip installation resolves declared
dependency ranges; uv project installs with `--locked` use the committed lockfile.

## Model redistribution and licenses

The repository's existing MIT license covers this project's own code. It does not
relicense detector libraries or downloaded weights. The wheel and sdist contain
no model weights; `.gitignore` also excludes them.

| Component | Primary source and distribution considerations |
| --- | --- |
| MediaPipe hand models | Google's [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf), linked from the [Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker), states Apache 2.0 for the hand detector/tracker pipeline. Redistribution is possible under that license: preserve relevant notices, provide the license, and identify changes; include applicable NOTICE content. |
| HaGRID YOLO hand weights | HaGRID's [README](https://github.com/hukenovs/hagrid) links the models and a [custom license](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf). The license explicitly says it is not a Creative Commons license, although adapted from CC BY-SA 4.0. It grants reproduction/sharing rights subject to attribution and other conditions, including conditions for adapted material. Do not label these files simply MIT or standard CC BY-SA. |
| Ultralytics runtime / YOLO integration | [Ultralytics](https://www.ultralytics.com/license) offers AGPL-3.0 and Enterprise licensing and states that its model/code integrations carry corresponding obligations. Review the integrated application's distribution terms before a public YOLO release; the repository's MIT notice alone does not settle those obligations. |

These notes record the sources checked for this packaging change, not a blanket
legal clearance for every model/version. The interaction between the HaGRID
weight license and the YOLO runtime needs particular attention if redistributing
the complete combination. Keeping weights outside the wheel, or having users
download them themselves, does **not** remove applicable license obligations.

Technically a wheel can carry data files, including models. Bundling them would
increase every package download and couple application releases to model releases.
PyPI also has [upload/storage limits](https://docs.pypi.org/project-management/storage-limits/).
This project therefore links original downloads instead. A possible future model
manager can download explicitly to a local cache, verify a pinned checksum and
retain license/source metadata; that command is not implemented in this release.

## One-time PyPI setup

1. Confirm the intended project name is available on PyPI and resolve the
   distribution-license choice for the optional YOLO integration. Model weights
   remain excluded.
2. Review and merge the implementation PR through the normal repository workflow.
3. Add a GitHub environment named `pypi`, restricted to the `main` branch.
   Configure required reviewers if releases should need a final human approval.
4. Configure PyPI [Trusted Publishing](https://docs.pypi.org/trusted-publishers/).
   For a new project, use a [pending publisher](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/):

   | PyPI field | Value |
   | --- | --- |
   | Project name | `castlestube-video-cleanup` |
   | Owner | `y0urself` |
   | Repository | `time-lapser` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

Trusted Publishing avoids storing a long-lived PyPI token in repository secrets.
This PR supplies the workflow; it does not configure a PyPI account, register a
project name, or trigger a release.

## Releasing through GitHub Actions

The workflow follows the Greenbone approach: shared uv setup, GitHub releases
and PyPA's Trusted Publishing action. Unlike Pontos's organization-specific
automation, it does not require Greenbone bot credentials or a signing key.

1. Update `castlestube_cleanup/version.py` and `CHANGELOG.md` in a PR; run
   `uv lock` to refresh the project version in the lockfile. Review and merge.
2. In **Actions → Release → Run workflow**, choose `main` and enter the exact
   version, for example `2.7.0`. Runs from other branches skip all release jobs.
3. The workflow runs the complete CI, including Windows tests, for the triggering
   commit. It checks that the requested version equals the source version.
4. It builds wheel and sdist once, creates `v2.7.0` and a GitHub release at that
   same tested commit, and attaches those distributions.
5. A separate job in the `pypi` environment publishes the same artifacts through
   OIDC. Only this job receives `id-token: write`; only the GitHub release job
   receives repository write permission. No release job pushes to `main`.

GitHub release creation and PyPI publishing are linked jobs in one workflow.
This avoids relying on a second workflow being triggered by a release created
with `GITHUB_TOKEN`. An existing release causes creation to fail instead of being
silently replaced. Tag creation also refuses existing tags, so a release cannot
silently point to a different commit. If creating the release fails after its tag
was created, inspect that tag and resolve the partial release before retrying.
If publishing alone fails, correct the configuration and use
**Re-run failed jobs** while the retained artifact is available; do not rerun the
successful release-creation job. Do not reuse a version already published on PyPI.

The artifact retention is seven days. After that, a failed release needs a new
deliberate recovery/release decision rather than an automatic rebuild-and-upload.
