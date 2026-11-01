"""Interval operations and forward/center-first deduplication."""

from __future__ import annotations

import mmap
import tempfile
import time

import cv2
import numpy as np

from .video import open_capture, signature


class DedupeGate:
    def __init__(self, threshold, max_frames, compare):
        self.threshold = threshold
        self.max_frames = max_frames
        self.compare = compare
        self.reference = None
        self.similar_count = 0

    def keep(self, current, force=False):
        if force or self.reference is None:
            keep = True
            self.similar_count = 1
        elif cv2.mean(cv2.absdiff(self.reference, current))[0] >= self.threshold:
            keep = True
            self.similar_count = 1
        else:
            keep = self.similar_count < self.max_frames
            if keep:
                self.similar_count += 1
        if keep or self.compare == "previous":
            self.reference = current
        return keep


def append_span(spans, start, end, gap=0):
    """Add an interval in chronological order without storing individual frames."""
    if spans and start <= spans[-1][1] + gap:
        spans[-1] = (min(start, spans[-1][0]), max(end, spans[-1][1]))
    else:
        spans.append((start, end))


def merge_spans(spans, limit, gap=0):
    """Clip and union half-open source-frame intervals, optionally filling gaps."""
    merged = []
    for start, end in sorted(spans):
        start, end = max(0, start), min(limit, end)
        if start < end:
            append_span(merged, start, end, gap)
    return merged


def subtract_spans(spans, removed):
    """Subtract sorted, disjoint intervals without expanding frame indices."""
    result = []
    cursor = 0
    for start, end in spans:
        while cursor < len(removed) and removed[cursor][1] <= start:
            cursor += 1
        position = start
        while cursor < len(removed) and removed[cursor][0] < end:
            cut_start, cut_end = removed[cursor]
            if position < cut_start:
                result.append((position, cut_start))
            position = max(position, cut_end)
            if cut_end >= end:
                break
            cursor += 1
        if position < end:
            result.append((position, end))
    return result


def center_removals(signatures, protected, minimum, threshold, stop_event):
    """Return local removed intervals; compare only retained neighbors in ONE phase.

    Both endpoints stay. Visiting the original center first makes a static phase
    retain balanced opening/closing portions, while distinct changes stay visible.
    A failed candidate is kept; this conservative single visit avoids repeatedly
    relaxing a decision as other frames disappear.
    """
    count = len(signatures)
    if count <= minimum:
        return []
    previous = np.arange(count, dtype=np.int64) - 1
    following = np.arange(count, dtype=np.int64) + 1
    removed = np.zeros(count, dtype=bool)
    remaining = count
    for radius in range((count + 1) // 2):
        left = (count - 1) // 2 - radius
        right = count // 2 + radius
        candidates = (left,) if left == right else (left, right)
        for index in candidates:
            if stop_event.is_set():
                raise InterruptedError("Center dedupe was cancelled.")
            if remaining <= minimum:
                break
            if index <= 0 or index >= count - 1 or protected[index]:
                continue
            before, after = previous[index], following[index]
            if (
                cv2.mean(cv2.absdiff(signatures[index], signatures[before]))[0]
                < threshold
                and cv2.mean(cv2.absdiff(signatures[index], signatures[after]))[0]
                < threshold
            ):
                removed[index] = True
                remaining -= 1
                following[before] = after
                previous[after] = before
        if remaining <= minimum:
            break
    spans = []
    start = None
    for index, is_removed in enumerate(removed):
        if is_removed and start is None:
            start = index
        elif not is_removed and start is not None:
            spans.append((start, index))
            start = None
    # The last frame cannot be removed, so no open interval remains.
    return spans


def plan_center_dedupe(source, activity, phases, protected, args, stop_event, progress):
    """Plan center-first removal with bounded image RAM and no model rerun."""
    long_phases = [
        (start, end) for start, end in phases if end - start > args.segment_frames
    ]
    removed = []
    shortened = 0
    removed_count = 0
    selected_count = sum(end - start for start, end in activity)
    seen = 0
    protected_index = 0
    started = time.perf_counter()
    if long_phases:
        cap = open_capture(source, args.threads)
        try:
            for start, end in long_phases:
                count = end - start
                forced = np.zeros(count, dtype=bool)
                while (
                    protected_index < len(protected)
                    and protected[protected_index][1] <= start
                ):
                    protected_index += 1
                cursor = protected_index
                while cursor < len(protected) and protected[cursor][0] < end:
                    a, b = protected[cursor]
                    forced[max(0, a - start) : min(count, b - start)] = True
                    cursor += 1
                # Disk-backed signatures avoid buffering full frames or an
                # unbounded list of images when hands stay visible for minutes.
                with tempfile.TemporaryFile() as scratch:
                    size = count * 90 * 160
                    scratch.truncate(size)
                    with mmap.mmap(
                        scratch.fileno(), size, access=mmap.ACCESS_WRITE
                    ) as backing:
                        signatures = np.ndarray(
                            (count, 90, 160), dtype=np.uint8, buffer=backing
                        )
                        try:
                            while seen < end:
                                if stop_event.is_set():
                                    raise InterruptedError(
                                        "Center dedupe was cancelled."
                                    )
                                if not cap.grab():
                                    raise RuntimeError(
                                        "Video ended before its analyzed hand phase during center dedupe."
                                    )
                                index = seen
                                seen += 1
                                if index >= start:
                                    ok, frame = cap.retrieve()
                                    if not ok or frame is None:
                                        raise RuntimeError(
                                            f"Cannot retrieve frame {index} during center dedupe."
                                        )
                                    signatures[index - start] = signature(frame, args)
                                    del frame
                                progress(
                                    "center-dedupe",
                                    seen,
                                    selected_count - removed_count,
                                )
                            cuts = center_removals(
                                signatures,
                                forced,
                                args.segment_frames,
                                args.threshold,
                                stop_event,
                            )
                        finally:
                            del signatures
                if cuts:
                    shortened += 1
                    removed.extend((start + a, start + b) for a, b in cuts)
                    removed_count += sum(b - a for a, b in cuts)
        finally:
            cap.release()
    return subtract_spans(activity, removed), {
        "order": "center",
        "segment_frames": args.segment_frames,
        "hand_phases": len(phases),
        "long_phases": len(long_phases),
        "shortened_phases": shortened,
        "removed_frames": removed_count,
        "removed_spans": removed,
        "seconds": round(time.perf_counter() - started, 4),
        "note": "Independent raw hand phases, not detected LEGO placements. Similarity to both retained neighbors; endpoints/context/camera protected. Target is not a hard cap.",
    }
