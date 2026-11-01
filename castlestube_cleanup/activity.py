"""Analyze hand/camera activity and protect source-frame context."""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2

from .camera import CameraMotionDetector
from .hands import create_hand_detector
from .selection import append_span, merge_spans
from .video import crop_for_analysis, open_capture


def analyze_activity(
    input_file: Path,
    args,
    fps: float,
    frame_limit: int | None,
    stop_event,
    progress,
):
    """Return eligibility, forced-retention intervals, hand stats and camera stats.

    Hands may be sampled, but camera motion always compares consecutive full
    source frames. A positive camera pair at i covers [i-1, i+1), preserving
    both images that define the transition even with zero context padding.
    The whole padded camera phase bypasses dedupe. Hand context is constructed
    independently so that merging activity spans never removes an inner neighbor.
    """
    started = time.perf_counter()
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Activity analysis requires a positive frame rate.")
    if args.hand_step < 1:
        raise ValueError("The hand sampling step must be positive.")
    if frame_limit is not None and frame_limit < 1:
        raise ValueError("The analysis frame limit must be positive.")

    hands_enabled = args.mode != "dedupe"
    pre_frames = args.pre_frames + math.ceil(args.pre_roll * fps)
    post_frames = args.post_frames + math.ceil(args.post_roll * fps)
    gap_frames = math.floor(args.bridge_gap * fps)
    intervals = []
    hand_intervals = []
    camera_intervals = []
    seen = samples = positives = 0
    last_sample_index = -1
    next_progress = started
    hand_detector = None
    camera_detector = None
    camera_seconds = 0.0
    cap = open_capture(input_file, args.threads)

    def select(start, end):
        append_span(
            intervals, max(0, start - pre_frames), end + post_frames, gap_frames
        )

    def selected_so_far():
        return sum(max(0, min(end, seen) - start) for start, end in intervals)

    try:
        reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count_metadata = (
            int(reported_count)
            if math.isfinite(reported_count) and reported_count > 0
            else None
        )
        if hands_enabled:
            hand_detector = create_hand_detector(args)
        if args.keep_camera_motion:
            camera_detector = CameraMotionDetector(
                args.camera_width, args.camera_threshold
            )
        stage = (
            "hands+camera"
            if hands_enabled and camera_detector
            else ("hands" if hands_enabled else "camera")
        )
        while frame_limit is None or seen < frame_limit:
            if stop_event.is_set():
                raise InterruptedError("Activity analysis was cancelled.")
            if not cap.grab():
                break
            source_index = seen
            seen += 1
            is_limit_frame = frame_limit is not None and seen == frame_limit
            hand_sample = hands_enabled and (
                source_index % args.hand_step == 0 or is_limit_frame
            )
            if hand_sample or camera_detector is not None:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Cannot retrieve frame {source_index} during activity analysis: {input_file}"
                    )
                if camera_detector is not None:
                    tick = time.perf_counter()
                    moving = camera_detector.moving(frame)
                    camera_seconds += time.perf_counter() - tick
                    if moving and source_index > 0:
                        append_span(
                            camera_intervals, source_index - 1, source_index + 1
                        )
                        select(source_index - 1, source_index + 1)
                if hand_sample:
                    has_hands = hand_detector.present(
                        crop_for_analysis(frame, args.roi), source_index, fps
                    )
                    samples += 1
                    last_sample_index = source_index
                    if has_hands:
                        positives += 1
                        hand_end = source_index + args.hand_step
                        append_span(hand_intervals, source_index, hand_end)
                        select(source_index, hand_end)
                del frame

            now = time.perf_counter()
            if now >= next_progress:
                progress(stage, seen, selected_so_far())
                next_progress = now + 1.0
        if seen == 0:
            raise RuntimeError(f"No video frames were decoded: {input_file}")
    finally:
        cap.release()
        if hand_detector is not None:
            hand_detector.close()

    intervals = merge_spans(intervals, seen)
    hand_intervals = merge_spans(hand_intervals, seen)
    camera_intervals = merge_spans(camera_intervals, seen)
    hand_spans = merge_spans(
        [(start - pre_frames, end + post_frames) for start, end in hand_intervals],
        seen,
        gap_frames,
    )
    camera_spans = merge_spans(
        [(start - pre_frames, end + post_frames) for start, end in camera_intervals],
        seen,
        gap_frames,
    )
    context = []
    for start, end in hand_intervals:
        context.extend(((start - pre_frames, start), (end, end + post_frames)))
    hand_context = merge_spans(context, seen)
    protected_spans = merge_spans(hand_context + camera_spans, seen)
    elapsed = round(time.perf_counter() - started, 4)
    hand_stats = None
    if hands_enabled:
        hand_stats = {
            "detector": args.hand_detector,
            "model": str(args.hand_model),
            "confidence": args.hand_confidence,
            "device": args.hand_device if args.hand_detector == "yolo" else "cpu",
            "input_frames": seen,
            "samples": samples,
            "positives": positives,
            "seconds": elapsed,
            "timing_note": "Elapsed shared analysis pass, including camera detection when enabled.",
            "spans": len(hand_spans),
            "hand_phases": len(hand_intervals),
            "phase_spans": hand_intervals,
            "interval_note": "Half-open, zero-based source indices before padding or gap bridging.",
            "selected_frames": sum(end - start for start, end in hand_spans),
            "protected_context_frames": sum(end - start for start, end in hand_context),
            "protected_spans": hand_context,
            "frame_count_metadata": frame_count_metadata,
            "unsampled_tail_frames": seen - last_sample_index - 1,
        }
        if args.hand_detector == "yolo":
            hand_stats["minimum_box_area_fraction"] = args.hand_min_area
            hand_stats["size_filter"] = dict(hand_detector.stats)
    camera_stats = None
    if camera_detector is not None:
        camera_stats = dict(camera_detector.stats)
        camera_stats.update(
            {
                "input_frames": seen,
                "seconds": round(camera_seconds, 4),
                "phases": len(camera_intervals),
                "selected_frames": sum(end - start for start, end in camera_spans),
                "phase_spans": camera_intervals,
                "protected_spans": camera_spans,
                "interval_note": "Half-open, zero-based source indices; phase spans include both pair endpoints.",
                "threshold_diagonal_fraction": args.camera_threshold,
                "maximum_analysis_side": args.camera_width,
            }
        )
    progress(stage, seen, sum(end - start for start, end in intervals))
    return intervals, protected_spans, hand_stats, camera_stats
