# Usage reference

Use `uv run --locked castlestube-cleanup --help` for the complete parser help.
When using hands, add `--extra yolo` or `--extra mediapipe` before the command.
After installing the package with pip, run `castlestube-cleanup` directly, or
`python -m castlestube_cleanup` with that same interpreter.

## Modes

| Mode | Selection |
| --- | --- |
| `dedupe` (default) | Compare the entire source; no hand model required. |
| `hands` | Keep hand phases and configured context, plus optional camera movement. |
| `hands-dedupe` | Select activity, then remove similar frames with forward or center dedupe. |

See [design decisions](features-and-decisions.md) for processing order and limits.

## Hand models

| Backend | Extra | Model | Default confidence |
| --- | --- | --- | --- |
| `mediapipe` (default) | `mediapipe` | `hand_landmarker.task` | 0.5 |
| `yolo` | `yolo` | `YOLOv10x_hands.pt` | 0.1 |

Original download locations:

- [MediaPipe Hand Landmarker](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task), [documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- [HaGRID YOLOv10x hands](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10x_hands.pt), [source project](https://github.com/hukenovs/hagrid).
- [HaGRID YOLOv10n hands](https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/models/YOLOv10n_hands.pt), a smaller alternative selected with `--hand-model`.

Models are not bundled in the wheel or Git. Download one locally and pass its
path with `--hand-model`; [Windows commands](windows.md#3-download-the-hand-model-once)
are provided. Without an explicit path, the installed CLI searches its working
directory and `ffmpeg.py` searches beside the launcher. Downloads do not happen
implicitly during inference. [Distribution and license notes](distribution.md)
explain why model files are handled separately.

YOLO requires a hand-specific detection model, not a general COCO `person` model.
All classes in the hand/gesture model count as hands. Each worker loads its own
selected backend; the other backend does not need to be installed.

## Selection options

| Option | Default | Effect |
| --- | --- | --- |
| `--threshold` | `8` | Similarity threshold, 0–255. Higher removes more; zero removes none by similarity. |
| `--max-frames` | `5` | Forward dedupe: retained frames per similar run. |
| `--compare` | `kept` | Forward reference: `kept` or `previous`. |
| `--dedupe-order` | `forward` | `center` is available only in `hands-dedupe`. |
| `--segment-frames` | `20` | Center target for long raw hand phases; at least 2, not a hard cap. |
| `--no-equalize` | disabled | Disable histogram equalization; retune threshold. |
| `--roi X Y W H` | whole image | Normalized hand/dedupe region inside decoded image. Export and camera use full frame. |
| `--pre-frames` / `--post-frames` | `2` / `2` | Protected source-frame context. |
| `--pre-roll` / `--post-roll` | `0` / `0` | Additional context in source seconds, rounded up to frames. |
| `--bridge-gap` | `0` | Fill gaps up to this many seconds after padding. |

`--max-frames` and `--compare` do not affect center dedupe. Different raw hand
phases stay separate for center comparisons even when context joins their output.

## Detection options

| Option | Default | Effect |
| --- | --- | --- |
| `--hand-detector` | `mediapipe` | Select `mediapipe` or `yolo`. |
| `--hand-model` | backend filename | Local model path. |
| `--hand-confidence` | backend-specific | Lower accepts more candidates; higher rejects more. |
| `--hand-min-area` | `0.005` | YOLO box area / analyzed image area; 0 disables. MediaPipe ignores this. |
| `--hand-step` | `1` | Run inference every N source frames; larger values approximate phase boundaries. |
| `--hand-width` | `640` | MediaPipe maximum side / YOLO inference size (stride-rounded). |
| `--hand-device` | `cpu` | YOLO only: `cpu`, `0` for CUDA, or `mps`; requires compatible runtime. |
| `--keep-camera-motion` | disabled | Select and protect detected camera movement in any mode. |
| `--camera-threshold` | `0.002` | Global displacement / image diagonal per frame; lower keeps slower motion and more shake. |
| `--camera-width` | `480` | Maximum analysis side, at least 96. |

## Export and operation

| Option | Default | Effect |
| --- | --- | --- |
| `input` | required | Video or recursively searched folder. |
| `--ext` | `.mp4 .mov` | Accepted input extensions, case-insensitive. |
| `--codec` | `prores` | `prores`, `h264`, `h264_nvenc`; all outputs use MOV. |
| `--prores-profile` | `3` | 0 Proxy, 1 LT, 2 422, 3 HQ. |
| `--crf` | `18` | H.264 CRF / NVENC CQ, 0–51; unused for ProRes. |
| `--rotate-portrait` | `cw` | `cw`, `ccw`, `off`; applied to decoded portrait output only. |
| `--threads` | min(8, CPU count) | Codec threads per video. |
| `--workers` | `1` | Concurrent videos; separate models and encoders. |
| `--limit-seconds` | none | Process the beginning with a distinct preview filename. |
| `--analyze-only` | disabled | Count/plan selection without encoding. |
| `--report` | disabled | Write or replace a JSON summary. |
| `--overwrite` | disabled | Replace an existing video only after successful processing. |
| `--ffmpeg` / `--ffprobe` | command names | Override executable paths. |
| `--version` | — | Print project version and exit. |

Folder scans skip `output`, temporary project folders and directory symlinks.
Exports go into `output` below the input root. Duplicate stems receive a stable
path hash. Reports sit beside exports; analysis-only reports end in
`_analysis.json`. Reports can be replaced independently of `--overwrite`.

If selection is empty, no MOV is created. If every frame survives and no portrait
rotation is needed, status is `unchanged` and no duplicate MOV is created. Existing
files remain untouched in both cases. Read reports/status when comparing runs.

## Examples

Plain dedupe, no ML extra:

```powershell
uv run --locked castlestube-cleanup "D:\Lego Castle\clips" --threshold 5 --max-frames 30 --threads 2 --codec h264 --report
```

Analysis only with YOLO:

```powershell
uv run --locked --extra yolo castlestube-cleanup "D:\Lego Castle\clip.mov" --mode hands --hand-detector yolo --hand-model .\models\YOLOv10x_hands.pt --analyze-only --report
```

Center dedupe with protected camera movement:

```powershell
uv run --locked --extra yolo castlestube-cleanup "D:\Lego Castle\clips" --mode hands-dedupe --hand-detector yolo --hand-model .\models\YOLOv10x_hands.pt --dedupe-order center --segment-frames 20 --keep-camera-motion --pre-frames 2 --post-frames 2 --codec h264 --report
```
