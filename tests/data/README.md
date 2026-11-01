# Real building excerpts

These two one-second excerpts come from Jaspar Stach's own LEGO recording
`IMG_6199.MOV` and were selected for public regression testing. The complete
original recording is not included. Repository licensing applies to these
fixture contributions.

| Clip | Original frame range | Hand labels | Purpose |
| --- | --- | --- | --- |
| `overhead-hands.mp4` | `[0, 30)` | Hands in all 30 frames | Overhead hands manipulating pieces. |
| `empty-lego-detail.mp4` | `[240, 270)` | No hands in any frame | Textured LEGO details that previously produced small false YOLO boxes. |

All source frames in these intervals were visually inspected for hand presence.
Labels in `manifest.json` refer to fixture-relative, zero-based, half-open frame
intervals. They are independent of detector predictions. There are no camera or
individual-placement ground-truth labels yet.

Both fixtures retain the original decoded 1080×1920 resolution and 30 fps.
Source rotation is baked into the pixels. They were re-encoded with H.264 CRF 18
and have no audio or source metadata. They are portable test excerpts, not
bit-exact copies of the HEVC source; compression can affect detector scores.
The manifest records source ranges, source and fixture hashes and conversion.

Normal CI verifies hashes, decoding, frame counts, frame rate and a complete
rotation-only export. Actual detector inference is an explicit evaluation:

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
