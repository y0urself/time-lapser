# Windows installation (PowerShell)

This guide uses Windows 10/11 on x64, PowerShell and Python 3.12 managed by uv.
You do not need to install Python separately or activate a virtual environment.
Run all project commands from the repository folder. Examples use CPU inference
so that an NVIDIA GPU is not required.

Prefer installing a released package with pip instead of a checkout? See the
[pip / uv tool installation instructions](distribution.md#pip-installation-on-windows).
The following checkout workflow works before the first PyPI release as well.

## 1. Install uv, FFmpeg and Git

Skip tools you already have. In PowerShell:

```powershell
winget install --id astral-sh.uv --exact
winget install --id Gyan.FFmpeg --exact
winget install --id Git.Git --exact
```

Open a **new PowerShell window** so it sees the updated PATH, then verify:

```powershell
uv --version
ffmpeg -version
ffprobe -version
git --version
```

If `winget` is unavailable, install Microsoft's
[App Installer](https://learn.microsoft.com/windows/package-manager/winget/).
Alternative installers are documented by
[uv](https://docs.astral.sh/uv/getting-started/installation/) and
[FFmpeg](https://ffmpeg.org/download.html#build-windows). A manual FFmpeg install
must expose the `bin` folder containing both `ffmpeg.exe` and `ffprobe.exe` on
PATH. You can instead pass absolute executable paths with `--ffmpeg` and
`--ffprobe`.

## 2. Get the project and install one backend

```powershell
git clone https://github.com/y0urself/time-lapser.git
Set-Location .\time-lapser
uv python install 3.12
uv sync --locked --extra yolo
uv run --locked --extra yolo castlestube-cleanup --version
```

`uv` creates `.venv` and installs the versions in `uv.lock`. The first YOLO
installation includes PyTorch and can take a while. No `Activate.ps1` or
PowerShell execution-policy change is needed.

For MediaPipe, substitute `--extra mediapipe` in both install and run commands.
For both backends use `--extra yolo --extra mediapipe`. Pass the selected extra
when running as well so uv keeps that dependency available. Plain dedupe needs
only `uv sync --locked` and no hand model.

## 3. Download the hand model once

YOLO / HaGRID:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Invoke-WebRequest -Uri "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10x_hands.pt" -OutFile .\models\YOLOv10x_hands.pt
```

Alternatively, MediaPipe:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile .\models\hand_landmarker.task
```

Use the original model sources linked in [model details](usage.md#hand-models).
Models are local files, excluded from Git. The application does not download
weights automatically.

## 4. Render a short preview

Replace the input path with your clip or folder. Keep paths containing spaces in
quotes, and omit a trailing backslash before the closing quote.

```powershell
uv run --locked --extra yolo castlestube-cleanup "D:\Lego Castle\clips" --mode hands --hand-detector yolo --hand-model .\models\YOLOv10x_hands.pt --keep-camera-motion --threads 4 --codec h264 --limit-seconds 30 --report
```

For MediaPipe:

```powershell
uv run --locked --extra mediapipe castlestube-cleanup "D:\Lego Castle\clips" --mode hands --hand-detector mediapipe --hand-model .\models\hand_landmarker.task --keep-camera-motion --threads 4 --codec h264 --limit-seconds 30 --report
```

Folder input searches recursively and puts results in that folder's `output`
subfolder. A single-file input uses `output` beside that file. Preview filenames
include `_preview30s_clean.mov`; reports use `.json`.

Review the preview for missed hands, retained empty stretches and orientation.
Remove `--limit-seconds 30` for a complete run. Add `--overwrite` when rerunning
settings against an existing output. Use `--rotate-portrait ccw` for the other
direction or `--rotate-portrait off` to retain portrait orientation.

## Existing ffmpeg.py commands

The launcher remains available with the same video arguments:

```powershell
uv run --locked --extra yolo python .\ffmpeg.py "D:\Lego Castle\clips" --mode hands --hand-detector yolo --hand-model .\models\YOLOv10x_hands.pt --keep-camera-motion --threads 4 --overwrite
```

Keep the **whole project**, including `castlestube_cleanup`, rather than copying
the new launcher alone. `py ffmpeg.py` still works if that Python has the project
and selected dependencies installed; uv avoids using a different global Python
accidentally. Without `--hand-model`, the legacy launcher looks beside itself;
the installed command looks in the current working directory.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `uv`, `ffmpeg` or `ffprobe` not found | Open a new terminal; use `Get-Command uv, ffmpeg, ffprobe`; fix PATH or pass executable paths. |
| `Hand model missing` | Check `Test-Path .\models\YOLOv10x_hands.pt` and the `--hand-model` path. MediaPipe needs `.task`, YOLO needs `.pt`. |
| Backend cannot load / module not found | Run `uv sync --locked --extra yolo` (or `mediapipe`), and use the same extra on `uv run`. |
| Output file is huge | Default is ProRes HQ. Use `--codec h264 --crf 18` for a smaller MOV; changing the file suffix is unnecessary. |
| Nothing exported | Read the status: existing output, no selected frames, or unchanged landscape content can intentionally produce no new file. Use `--report`. |
| Too much removed | Try `--mode hands`, `--hand-step 1`; inspect confidence and area thresholds before adding dedupe. Raising confidence rejects more detections. |
| Empty stretches retained | Inspect the JSON hand-size and camera statistics; compare with and without `--keep-camera-motion`. Lighting and false hand detections can affect selection. |
| DLL load error | Confirm x64 Python 3.12; inspect the named backend's error. Consult [OpenCV's Windows troubleshooting](https://pypi.org/project/opencv-python/). |
| NVENC fails | Use `--codec h264` first; NVENC needs an NVIDIA GPU, compatible driver and FFmpeg encoder support. |

`--hand-device 0` selects YOLO's CUDA device only when the installed PyTorch build
supports CUDA. `--codec h264_nvenc` selects video encoding separately. Neither is
required for the CPU setup above.

## Updating

After updating your checkout to the desired revision:

```powershell
uv sync --locked --extra yolo
uv run --locked --extra yolo castlestube-cleanup --version
```

Do not run `uv lock --upgrade` merely to install: it changes the tested dependency
set. Dependency changes are reviewed through PRs and CI.
