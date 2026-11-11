# time-lapser

Turn long LEGO building recordings into shorter editing material: keep hand
activity, optionally preserve camera movement, remove similar frames, and rotate
portrait footage for a landscape timeline. Processing happens locally. The goal
is to reduce repetitive editing while keeping the build understandable.

The command is `castlestube-cleanup`; existing `ffmpeg.py` commands also work
from this complete checkout. Version 0.1.0 packages the existing script into modules
without changing its selection defaults.

## Installation

**Windows / PowerShell:** follow the [Windows installation guide](docs/windows.md)
for uv, FFmpeg, model downloads and a first preview. Python 3.12 is the recommended
starting point and is selected by `.python-version`.

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed and
`ffmpeg` / `ffprobe` on PATH:

```sh
git clone https://github.com/y0urself/time-lapser.git
cd time-lapser
uv sync --locked --extra yolo
uv run --locked --extra yolo castlestube-cleanup --help
```

Download a [hand model](docs/usage.md#hand-models) before using `hands` or
`hands-dedupe`. For MediaPipe, use `--extra mediapipe` instead. Plain deduplication
needs only `uv sync --locked`, with no model or machine-learning extra.

## First run

PowerShell, after installing YOLO and downloading `models\YOLOv10x_hands.pt`:

```powershell
uv run --locked --extra yolo castlestube-cleanup "D:\Lego Castle\clips" --mode hands --hand-detector yolo --hand-model .\models\YOLOv10x_hands.pt --keep-camera-motion --threads 4 --codec h264 --limit-seconds 30 --report
```

Inspect the preview in `D:\Lego Castle\clips\output` before removing
`--limit-seconds 30`. Use `--overwrite` to replace an existing export after a
successful run. Originals are never modified.

To additionally shorten similar parts inside long hand phases:

```powershell
uv run --locked --extra yolo castlestube-cleanup "D:\Lego Castle\clips" --mode hands-dedupe --hand-detector yolo --hand-model .\models\YOLOv10x_hands.pt --dedupe-order center --segment-frames 20 --keep-camera-motion --threads 4 --codec h264 --report
```

## Capabilities

| Feature | Purpose |
| --- | --- |
| `hands` | Keep detected hand activity, plus two source frames before and after each phase by default. |
| MediaPipe / YOLO selection | Try a different detector for overhead hands, occlusion and your lighting. |
| YOLO minimum box area | Filter tiny LEGO details incorrectly detected as hands. |
| `--keep-camera-motion` | Preserve detected pans, tilts, rotation and zoom with context. |
| `dedupe` / `hands-dedupe` | Remove similar frames globally or within selected activity intervals. |
| Center dedupe | Shorten long hand phases from the middle only where neighboring frames are similar. |
| Portrait rotation | Bake a 90° clockwise rotation into portrait exports; configurable or disabled. |
| Preview and JSON report | Compare settings using short outputs and explain retained frame counts. |
| Streaming export | Process long files without keeping decoded videos in memory. |

Hand phases are **not recognized LEGO placements**. Detection can miss hands or
mistake objects for hands; dedupe can remove small changes. Start with `hands`
and review representative previews before adding compression.

Exports are silent MOV files at a constant source-derived frame rate. The current
pipeline processes 8-bit SDR pixels; it does not preserve HDR, audio or timecode.
ProRes HQ is the default and produces large files; `--codec h264` is useful for
smaller previews. If no frames change and no rotation is needed, no duplicate
output is written.

## Documentation and development

- [Windows installation and troubleshooting](docs/windows.md)
- [Modes, arguments and model downloads](docs/usage.md)
- [Features and design decisions: what happens and why](docs/features-and-decisions.md)
- [Architecture, tests, CI and Dependabot](docs/development.md)
- [pip installation and model distribution](docs/distribution.md)
- [Real test clips and recognition-quality checks](docs/test-data.md)
- [Changelog](CHANGELOG.md)

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked coverage run -m unittest discover -v
uv run --locked coverage report
uv build
```

CI uses fake hand models, synthetic videos and two small real-video fixtures;
FFmpeg is required, model weights are not. Project code is [MIT licensed](LICENSE);
dependencies and model weights retain their own licenses. See the
[YOLO distribution considerations](docs/distribution.md#model-redistribution-and-licenses)
before redistributing the integrated backend.
