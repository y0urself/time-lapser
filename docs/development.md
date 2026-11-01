# Development

The project uses PEP 621 metadata, Hatchling builds, uv dependency groups and a
committed `uv.lock`, following the structure used in
[greenbone/pontos](https://github.com/greenbone/pontos). The package stays flat so
the compatibility launcher can run directly from a checkout.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `cli.py` | Arguments, input discovery, output names and worker orchestration. |
| `hands.py` | Lazy MediaPipe / YOLO imports, local model loading and size filter. |
| `camera.py` | Global motion estimation from consecutive source images. |
| `activity.py` | Raw hand phases, camera selection, padding and protection. |
| `selection.py` | Source-frame interval operations and both dedupe strategies. |
| `video.py` | Decode/probe, signatures, rotation plan and FFmpeg pipe encoder. |
| `pipeline.py` | Per-file analysis/export flow, validation, publication and reports. |
| `runtime.py` | Progress logging and shared encoder cancellation tracking. |
| `version.py` | Single version source for metadata, CLI and reports. |

`__main__.py` supports `python -m castlestube_cleanup`; root `ffmpeg.py` delegates
to the same CLI while retaining model lookup beside the script. There is no
dependency from package modules back to the compatibility launcher.

## Local checks

Install FFmpeg first; see [Windows setup](windows.md). Then:

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked coverage run -m unittest discover -v
uv run --locked coverage report
uv build
```

For formatting changes use `uv run ruff format .`. Dependency changes belong in
`pyproject.toml` and `uv.lock` together; use `uv lock` after editing metadata.
For a targeted upgrade use `uv lock --upgrade-package PACKAGE`, review the diff,
and run the checks. Optional extras install actual detectors only when needed.

## Test strategy

Unit tests cover interval semantics, forward and center selection, backend
defaults, area filtering, context protection, model paths and file discovery.
Synthetic FFmpeg integrations check frame counts, FPS, baked rotation (including
orientation metadata), ROI consistency and output preservation on failure.
Camera tests distinguish global translation from static/local/lighting changes.

Fake hand models make tests deterministic and avoid network access or weight
downloads. These tests protect the selection/export contract; they do not measure
real-world hand recall or prove that every brick survives. Evaluate detection
changes separately against representative overhead recordings.

See [test data](test-data.md) for the separate real-clip regression dataset,
annotation format, metrics and CI scope.

## CI

`.github/workflows/ci.yml` runs on PRs, pushes to `main`, and manual dispatch.
It checks Ruff lint/format, builds wheel and sdist, and tests the installed wheel
outside the source import path. Tests run on Linux with Python 3.10, 3.12 and 3.13,
and Windows with Python 3.12. Both platforms install FFmpeg and run real encoding
tests. Coverage is reported in logs without needing a Codecov secret.

The workflow uses the shared
[greenbone/actions/uv](https://github.com/greenbone/actions/tree/main/uv) setup.
Dependencies are installed explicitly with `uv sync --locked`: the generic
Greenbone lint action installs all extras, which would bring large ML dependencies
into every lint run. Action revisions are pinned; CI has read-only repository
permissions. CI never merges changes. A separate manually triggered
[release workflow](distribution.md#releasing-through-github-actions) publishes
only after the full CI has passed for its commit.

## Dependabot

`.github/dependabot.yml` checks the native `uv` ecosystem and GitHub Actions
weekly. Minor/patch updates are grouped within each ecosystem; major updates
remain separate PRs. There is no automatic merge. The configuration takes effect
when present on the default branch. `uv` support maintains the lockfile alongside
project requirements; see [Astral's integration guide](https://docs.astral.sh/uv/guides/integration/dependabot/).

Model files and FFmpeg are not managed by Dependabot. Model source and license
changes need their own review; package updates do not replace downloaded weights.

## Packaging and publication

See [distribution](distribution.md) for pip installation, the release checklist
and the separate licenses for code, detector libraries and pretrained weights.
The source archive includes documentation, tests and the tiny public fixtures;
the wheel includes the Python package and its license, not videos or model files.
