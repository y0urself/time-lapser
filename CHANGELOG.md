# Changelog

## 0.4.0

- Package the existing cleanup script into focused modules with a
  `castlestube-cleanup` console command and module entry point.
- Preserve `ffmpeg.py` arguments and script-relative model lookup with a thin
  compatibility launcher; the installed CLI uses the working directory.
- Add uv project metadata and lockfile, optional YOLO/MediaPipe extras, and
  Hatchling wheel/sdist builds.
- Add Windows installation, usage, design and development documentation.
- Add unit and synthetic-video tests, Ruff, coverage, Linux/Windows CI using
  Greenbone's uv action, and Dependabot for uv and GitHub Actions.
- Add `--version`. Selection, rotation and context defaults remain unchanged.
- Include two small public video fixtures with hand-presence labels and an
  explicit optional detector evaluator; keep model inference out of normal CI.
- Add a manual release workflow that reruns CI, builds distributions once,
  creates the GitHub release and publishes to PyPI through Trusted Publishing.

## Earlier script versions

### 0.3.0

- Rotate decoded portrait exports clockwise by default; support counterclockwise
  and disabled rotation. Apply rotation to output pixels only.
- Write rotation-only exports even if all source frames are kept.

### 0.2.0

- Keep two original frames before and after activity by default.
- Add optional center dedupe within raw hand phases with protected endpoints,
  context and camera frames; remove only similar interior frames.

### 0.1.0

- Filter YOLO boxes smaller than a configurable fraction of the analysis image
  (default 0.5%) to suppress tiny false hand detections on LEGO details.
- Report size-filter statistics.
