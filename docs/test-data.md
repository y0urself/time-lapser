# Test data: pipeline correctness and detector quality

## Normal CI uses generated videos

The tests create short, tiny clips in temporary directories with FFmpeg and NumPy.
Known pixels and timestamps let us assert exact frame counts, ordering, source
frame rate, orientation and context boundaries. Hand detections are controlled
by fake detectors so selection tests do not change with model versions.

These fixtures are created during each test and cleaned up afterwards. CI does
not need the original building videos or downloaded hand weights. Synthetic
camera images exercise actual OpenCV motion estimation, including exposure
changes, local motion, global translation and the ORB fallback.

## Real clips measure recognition quality

The initial [public fixtures](../tests/data/README.md) contain two one-second
excerpts: overhead hands and an empty textured LEGO scene. Their 60 source frames
were visually reviewed for hand presence. `tests.evaluate_models` runs a selected
local detector and reports recall, precision and false-positive rate as JSON.
The normal CI checks fixture integrity, decoding and export without ML inference.

Synthetic tests cannot establish whether MediaPipe or YOLO recognizes a hand in
an overhead LEGO recording. Keep a separate, small regression collection for
that purpose. Useful cases include:

| Case | What to measure |
| --- | --- |
| Full hand viewed from above | Missed real hand frames. |
| Fingers behind a model or holding a brick | Recall under occlusion. |
| Distant or partly cropped hand | Interaction with YOLO minimum-area filtering. |
| Empty scene with LEGO details | False hand detections. |
| Empty scene with strong exposure changes | False retention from detection or dedupe. |
| Pan / tilt / zoom, including without hands | Intended camera preservation. |
| Two building actions separated by a pause | Separation of phases and visible build states. |
| Several placements without removing the hand | Limits of treating one hand phase as one action. |

Use short representative excerpts rather than entire recordings. Retain source
resolution, frame rate, codec and orientation metadata where they affect the
failure being tested. Re-encoding or resizing can change the problem. Only add
clips to a public dataset when their publication is explicitly intended.

## Ground truth and reproducibility

For each clip, record:

- Identifier, SHA-256, provenance and permission to redistribute, if applicable.
- Decoded frame count, FPS, dimensions and source orientation metadata.
- Manually checked hand-present intervals using zero-based half-open frame
  ranges: `[start, end)` includes start and excludes end.
- Expected camera-motion intervals and any ambiguous/occluded ranges.
- Selected build-state frames that a person wants to remain visible.

Record model filename/hash, backend versions, settings, platform and device for
each evaluation. Inspect the same decoded source frames for both annotations and
inference. Keep manual labels independent of the current detector's output;
otherwise a test would preserve its existing mistakes.

Useful metrics are hand-frame recall (missed activity), precision (false hand
positives), retained no-hand/no-camera frames, and retention of annotated build
states. Agree acceptable limits after establishing a baseline. Do not assert
exact floating-point model scores across different hardware.

## Where to run it

Keep the fast synthetic suite on every PR, including Dependabot PRs. Run a future
model evaluation suite deliberately for detector/model/dependency changes, or
on a separately scheduled job with a versioned dataset and cached weights.
Large/private clips and credentials do not belong in the normal public PR job.

The full user-provided recordings are not committed. Only the selected public
excerpts and their labels are included. More failure cases and agreed quality
thresholds are needed before model accuracy should gate a release.
