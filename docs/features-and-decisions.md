# What the tool does, and why

## Goal: less editing, understandable building

Long overhead LEGO recordings contain pauses, searching for pieces, hands
remaining still, exposure changes and occasional camera moves. Removing pauses
can save hours of repetitive editing. The current tool uses hand presence and
image similarity as practical signals; it does not recognize individual bricks
or understand whether a placement has finished.

## Selection happens on the original timeline

1. Decode the source orientation and inspect frame rate and dimensions.
2. For hand modes, detect hand presence and collect consecutive positive frames
   into raw hand phases. Optionally detect camera movement on every full frame.
3. Add context around each phase: two original frames before and after by
   default, plus any configured time-based roll. Merge overlapping intervals and
   optionally bridge short gaps. Protect context and camera frames from dedupe.
4. In `hands`, export the selection directly. In `hands-dedupe`, run the chosen
   dedupe strategy. In plain `dedupe`, all source frames are eligible, with camera
   protection if requested.
5. Export kept frames in chronological order, apply portrait rotation, and
   replace the destination only after successful encoding and validation.

There is no intermediate video with all no-hand frames removed. Selection keeps
source indices. Forward comparisons reset at each separate padded interval, so
separated building actions are not compared across a removed gap. Padding or
`--bridge-gap` can merge actions into one interval, however. Center dedupe uses
the raw phases from step 2, before those merges, to keep their comparisons apart.

## Why two frames before and after?

A detector can become positive slightly after a hand enters, or stop immediately
when it leaves. Source-frame context makes cuts less abrupt and preserves the
moment just outside the detected phase. It is not a guarantee against missed
hands. Increasing pre/post context adds material rather than improving detection.

## Why two hand backends?

MediaPipe combines palm detection, hand landmarks and tracking. YOLO/HaGRID
detects hand bounding boxes. Their failure modes differ with overhead views,
occlusion and small hands. Scores are model-specific: MediaPipe's default 0.5
and YOLO's 0.1 are not comparable confidence measurements.

Increasing `--hand-confidence` accepts fewer detections. This can reduce false
positives while removing real hand activity. Lowering it admits more candidates
and may keep empty stretches. No threshold ensures every hand is recognized.

YOLO also rejects boxes smaller than 0.5% of the analyzed image by default. This
was introduced because tiny LEGO details were being detected as hands. The
fraction is relative to `--roi` when set. Lower it for small/distant/cropped hands;
zero disables it. The report counts rejected boxes and affected samples.

## Forward dedupe

Each eligible frame becomes a 160×90 grayscale signature, histogram-equalized
unless disabled. Mean absolute difference below `--threshold` counts as similar.
The gate keeps up to `--max-frames` frames in a similar run. The default reference
is the last kept frame; `--compare previous` uses the previous eligible frame.

This is fast and suitable for removing long still stretches. It does not identify
brick placements. A tiny new brick may barely affect the signature, while a large
lighting change can look different. Equalization helps with some exposure changes
but does not make lighting irrelevant. Raising the threshold removes more.

## Center dedupe

`--mode hands-dedupe --dedupe-order center --segment-frames 20` treats each raw
hand phase separately. Phases up to 20 frames stay complete. In longer phases:

- Visit interior candidates from the original middle outwards.
- Remove a candidate only if it is similar to **both remaining neighbors**.
- Preserve phase endpoints, context and camera-protected frames.
- Stop at 20 retained phase frames, or when no further candidates qualify.

The target is not a hard limit. Different frames may keep a phase much longer.
The exported result stays chronological and is not deduped a second time.
This is deliberately conservative; it is not a fixed-duration center cut.

An isolated compressible phase can yield `2 + 20 + 2 = 24` frames. At 60 fps that
is 0.4 seconds, or 2.5 phases per second. A target of 16 gives 20 frames with
context, or 3 phases per second. Input frame rate is preserved, so at 30 fps the
same 24 frames take 0.8 seconds. Overlapping context is counted only once.

A single hand phase may contain several placements; a missed detection may split
one placement. Similarity protection reduces some losses but cannot promise that
every brick remains visible. Start with `hands` when checking detection quality.

## Why keep camera movement separately?

A pan or close-up often communicates the finished structure even without hands.
The camera detector looks for motion supported across the frame, using optical
flow with an ORB feature-matching fallback. It protects both frames of a detected
transition and their context. Local moving hands should not normally trigger it.

Camera analysis always uses the full source image and every frame, independently
of hand sampling and ROI. Unclassifiable frame pairs are not called camera motion.
Texture, lighting, blur and large moving objects can still cause mistakes.

## Rotation belongs to export

OpenCV applies available source orientation metadata before analysis. If that
decoded image is taller than wide, export rotates it 90° clockwise by default.
The rotation is baked into pixels for editors such as DaVinci Resolve. `ccw`
chooses the other direction; `off` keeps portrait orientation. This is a chosen
direction, not visual recognition of which way is up.

ROI, detection and similarity retain their original analysis coordinates.
Landscape and square images stay as they are. Rotation alone is enough to create
an output even if every frame is kept. H.264 may add a one-pixel edge to make
dimensions even; content is never cropped for rotation.

## Long-video processing and output tradeoffs

Decoding and encoding stream frames instead of buffering complete videos.
Center dedupe adds a decoding pass and stores signatures in temporary disk-backed
memory (14,400 bytes per frame of the phase); index arrays still grow with phase
length. Each concurrent worker owns a detector and an encoder. Start with one
worker and increase only after checking memory and GPU use.

ProRes HQ suits editing but is large. H.264 trades editability for smaller files;
NVENC can use supported NVIDIA hardware. Output is silent, constant-rate and
processed as 8-bit SDR, even when the selected ProRes format has a 10-bit storage
format. Audio, HDR precision, timecode and variable source timestamps are not
preserved by this pipeline.

Reports expose settings, frame counts, hand phases, camera statistics and export
orientation. They help compare settings but are not a complete per-frame model
debug trace. Original media stays untouched; temporary encodes are cleaned up on
failure or cancellation. An existing output is retained if the replacement fails.
