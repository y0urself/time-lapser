# Real building excerpts

These two one-second excerpts come from Jaspar Stach's own LEGO recordings
`IMG_8947.MOV` and `IMG_6199.MOV` and were selected for public regression testing. The complete
original recordings are not included. Repository licensing applies to these
fixture contributions.

| Clip | Source | Original frame range | Hand labels (fixture-relative) |
| --- | --- | --- | --- |
| `lego-building.mp4` | `IMG_8947.MOV` | `[480, 510)` (16–17 s) | `[0, 5)`, `[8, 12)`, `[18, 20)`, `[26, 30)` |
| `empty-lego-detail.mp4` | `IMG_6199.MOV` | `[240, 270)` | No hands |

`lego-building.mp4` replaces the missing `overhead-hands.mp4` fixture. It contains
15 frames with hands and 15 without; a sleeve alone is not labeled as a hand.
Its publication as a test contribution was requested by the recording owner.
The empty portrait clip retains the textured scene that previously produced
small false YOLO boxes.

All source frames in these intervals were visually inspected for hand presence.
Labels in `manifest.json` refer to fixture-relative, zero-based, half-open frame
intervals. They are independent of detector predictions. There are no camera or
individual-placement ground-truth labels yet.

Both fixtures retain their original decoded resolution and 30 fps: 1920×1080
for the new building clip and 1080×1920 for the empty portrait clip.
Source rotation is baked into the pixels. They were re-encoded with H.264 CRF 18
and have no audio or source metadata. They are portable test excerpts, not
bit-exact copies of the HEVC source; compression can affect detector scores.
The manifest records source ranges, source and fixture hashes and conversion.

Normal CI verifies hashes, decoding, frame counts, frame rate and a complete
rotation-only export. The new landscape clip also checks the exact selected
frame order, frame rate and two-frame context around controlled hand detections.
Actual detector inference is an explicit evaluation:

```sh
uv run --locked --extra yolo python -m tests.evaluate_models --hand-detector yolo --hand-model models/YOLOv10x_hands.pt
uv run --locked --extra mediapipe python -m tests.evaluate_models --hand-detector mediapipe --hand-model models/hand_landmarker.task
```

The evaluator prints JSON with model hash, runtime versions, settings and
per-clip/total confusion counts. It measures current quality without asserting
that every label must already be recognized. Model weights are not included.

Add new fixtures only for a specific failure case, inspect all labeled frames,
record provenance and conversion, and keep the dataset small. The clips are in
Git and the source archive for reproducible tests, but are not installed in the
application wheel.

To reproduce the new fixture from the owner-provided source (FFmpeg 6.1.1):

```sh
ffmpeg -v error -ss 16 -i IMG_8947.MOV -frames:v 30 -an -map_metadata -1 -c:v libx264 -crf 18 -pix_fmt yuv420p tests/data/lego-building.mp4
```

The checksum identifies the committed bytes; different encoder versions can
produce different bytes when repeating this conversion. Use the committed clip
for reproducible CI.
