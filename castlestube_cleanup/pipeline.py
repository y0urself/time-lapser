"""Per-file selection, export, reports and atomic publication."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from .activity import analyze_activity
from .runtime import Progress, log
from .selection import DedupeGate, plan_center_dedupe
from .version import __version__ as VERSION
from .video import (
    RawEncoder,
    check_frame_count,
    expected_frames,
    export_transform,
    frame_rate,
    open_capture,
    probe_video,
    signature,
)


def write_report(destination, report):
    report_path = destination.with_suffix(".json")
    if report["analyze_only"]:
        report_path = destination.with_name(destination.stem + "_analysis.json")
    fd, temporary = tempfile.mkstemp(
        prefix=".castlestube-", suffix=".json", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, report_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def process_file(source, destination, args, stop_event):
    if destination.exists() and not args.overwrite and not args.analyze_only:
        log(
            f"[SKIP] {source.name}: output exists ({destination.name}); use --overwrite to replace."
        )
        return "skipped"
    started = time.perf_counter()
    camera_label = ", camera motion enabled" if args.keep_camera_motion else ""
    log(f"[START] {source.name}: {args.mode}, {args.codec}{camera_label}")
    if args.mode != "dedupe":
        log(
            f"[MODEL] {args.hand_detector}: {args.hand_model.name}, confidence={args.hand_confidence:g}"
            + (
                f", device={args.hand_device}, min-area={args.hand_min_area:g}"
                if args.hand_detector == "yolo"
                else ""
            )
        )
    original_stat = source.stat()
    metadata = probe_video(source, args.ffprobe)
    cap = open_capture(source, args.threads)
    try:
        rate = frame_rate(metadata, cap)
        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            raise RuntimeError(f"Cannot decode the first video frame: {source}")
        frame_shape = first_frame.shape
        del first_frame
    finally:
        cap.release()
    transform = export_transform(frame_shape, args)
    will_rotate = transform["clockwise_degrees"] != 0
    if will_rotate:
        direction_label = (
            "clockwise" if transform["clockwise_degrees"] > 0 else "counterclockwise"
        )
        log(
            f"[ROTATE] {source.name}: {transform['decoded_width']}x{transform['decoded_height']} -> "
            f"{transform['output_width']}x{transform['output_height']}; 90 degrees {direction_label}"
        )
    fps = float(rate)
    limit = (
        max(1, math.ceil(args.limit_seconds * fps))
        if args.limit_seconds is not None
        else None
    )
    expected = expected_frames(metadata, limit)
    progress = Progress(source.name)
    pixel_format = metadata.get("pix_fmt", "")
    if metadata.get("color_transfer") in ("smpte2084", "arib-std-b67") or any(
        part in pixel_format for part in ("p10", "p12", "p16", "48", "64")
    ):
        log(
            f"[WARN] {source.name}: {pixel_format}/HDR input. This 8-bit SDR pipeline does not preserve HDR or 10-bit precision."
        )
    if metadata.get("sample_aspect_ratio", "1:1") not in ("1:1", "N/A", "0:1"):
        raise RuntimeError(
            "Non-square source pixels need a dedicated conversion; no output was created."
        )
    try:
        nominal = Fraction(metadata.get("r_frame_rate", "0/1"))
        if nominal > 0 and abs(float(nominal) - fps) / fps > 0.001:
            log(
                f"[WARN] {source.name}: differing nominal/average rates; output uses constant {rate} fps."
            )
    except (ValueError, ZeroDivisionError):
        pass

    hand_stats = None
    camera_stats = None
    analysis_count = None
    spans = None
    protected_spans = []
    if args.mode != "dedupe" or args.keep_camera_motion:
        activity_spans, protected_spans, hand_stats, camera_stats = analyze_activity(
            source, args, fps, limit, stop_event, progress
        )
        analysis_count = (hand_stats if hand_stats is not None else camera_stats)[
            "input_frames"
        ]
        check_frame_count(analysis_count, expected)
        if args.mode != "dedupe":
            spans = activity_spans
    if hand_stats is not None:
        log(
            f"[HANDS] {source.name}: {hand_stats['positives']}/{hand_stats['samples']} positive samples; {len(spans)} activity intervals"
        )
        if "size_filter" in hand_stats:
            sf = hand_stats["size_filter"]
            log(
                f"[HAND SIZE] {source.name}: {sf['rejected_small_boxes']} boxes below the minimum area; "
                f"{sf['frames_rejected_by_size']} samples had only rejected boxes"
            )
    if camera_stats is not None:
        log(
            f"[CAMERA] {source.name}: {camera_stats['detected_pairs']}/{camera_stats['comparisons']} moving pairs; "
            f"{camera_stats['phases']} phase(s); {camera_stats['selected_frames']} protected frames; "
            f"{camera_stats['unclassified_pairs']} unclassified pairs"
        )

    center_stats = None
    center_mode = args.mode == "hands-dedupe" and args.dedupe_order == "center"
    if center_mode:
        spans, center_stats = plan_center_dedupe(
            source,
            spans,
            hand_stats["phase_spans"],
            protected_spans,
            args,
            stop_event,
            progress,
        )
        log(
            f"[CENTER] {source.name}: {center_stats['shortened_phases']}/{center_stats['hand_phases']} "
            f"hand phases shortened; {center_stats['removed_frames']} frames removed; "
            f"phase target {args.segment_frames} plus protected context"
        )

    total_in = total_out = 0
    read_seconds = analysis_seconds = encode_wait_seconds = 0.0
    # The hands selection mode can count its activity union without a second pass.
    shortcut = spans is not None and (
        not spans or (args.analyze_only and (args.mode == "hands" or center_mode))
    )
    if shortcut:
        total_in = analysis_count
        total_out = sum(end - start for start, end in spans)
    else:
        gate = DedupeGate(args.threshold, args.max_frames, args.compare)
        span_index = 0
        protected_index = 0
        last_span_index = -1
        cap = open_capture(source, args.threads)
        encoder = None
        try:
            with tempfile.TemporaryDirectory(
                prefix=".castlestube-", dir=destination.parent
            ) as temporary:
                temp_output = Path(temporary) / "video.mov"
                try:
                    while limit is None or total_in < limit:
                        if stop_event.is_set():
                            raise InterruptedError("Cancelled.")
                        tick = time.perf_counter()
                        grabbed = cap.grab()
                        read_seconds += time.perf_counter() - tick
                        if not grabbed:
                            break
                        index = total_in
                        total_in += 1
                        if spans is not None:
                            while (
                                span_index < len(spans)
                                and index >= spans[span_index][1]
                            ):
                                span_index += 1
                            eligible = (
                                span_index < len(spans)
                                and index >= spans[span_index][0]
                            )
                            if not eligible:
                                progress("export", total_in, total_out)
                                continue
                            if span_index != last_span_index:
                                gate = DedupeGate(
                                    args.threshold, args.max_frames, args.compare
                                )
                                last_span_index = span_index
                        tick = time.perf_counter()
                        ok, frame = cap.retrieve()
                        read_seconds += time.perf_counter() - tick
                        if not ok or frame is None:
                            raise RuntimeError(
                                f"Cannot retrieve decoded frame {index}."
                            )
                        if frame.shape != frame_shape:
                            raise RuntimeError(
                                "Video dimensions changed during decoding."
                            )
                        tick = time.perf_counter()
                        if args.mode == "hands" or center_mode:
                            keep = True
                        else:
                            while (
                                protected_index < len(protected_spans)
                                and index >= protected_spans[protected_index][1]
                            ):
                                protected_index += 1
                            is_protected = (
                                protected_index < len(protected_spans)
                                and index >= protected_spans[protected_index][0]
                            )
                            keep = gate.keep(signature(frame, args), force=is_protected)
                        analysis_seconds += time.perf_counter() - tick
                        if keep:
                            total_out += 1
                            if not args.analyze_only:
                                tick = time.perf_counter()
                                if encoder is None:
                                    encoder = RawEncoder(
                                        temp_output, frame.shape, rate, args
                                    )
                                encoder.write(frame)
                                encode_wait_seconds += time.perf_counter() - tick
                        progress(
                            "analyze" if args.analyze_only else "export",
                            total_in,
                            total_out,
                        )
                    if total_in == 0:
                        raise RuntimeError("No video frames could be decoded.")
                    check_frame_count(total_in, expected)
                    if analysis_count is not None and total_in != analysis_count:
                        raise RuntimeError(
                            "The two decoding passes returned different frame counts."
                        )
                    if stop_event.is_set():
                        raise InterruptedError("Cancelled.")
                    current_stat = source.stat()
                    if (original_stat.st_size, original_stat.st_mtime_ns) != (
                        current_stat.st_size,
                        current_stat.st_mtime_ns,
                    ):
                        raise RuntimeError(
                            "Input changed during processing; no output was published."
                        )
                    if encoder:
                        tick = time.perf_counter()
                        encoder.finish()
                        encode_wait_seconds += time.perf_counter() - tick
                        if total_out > 0 and (total_out < total_in or will_rotate):
                            if destination.exists() and not args.overwrite:
                                raise FileExistsError(
                                    f"Output appeared during processing: {destination}"
                                )
                            os.replace(temp_output, destination)
                finally:
                    if encoder:
                        encoder.close()
        finally:
            cap.release()

    elapsed = time.perf_counter() - started
    reduction = 100 * (1 - total_out / total_in) if total_in else 0
    if total_out == 0:
        status = "no_selection"
    elif total_out >= total_in and not will_rotate:
        status = "unchanged"
    elif args.analyze_only:
        status = "analyzed"
    else:
        status = "written"
    report = {
        "version": VERSION,
        "source": str(source),
        "output": str(destination) if status == "written" else None,
        "status": status,
        "analyze_only": args.analyze_only,
        "mode": args.mode,
        "source_size": original_stat.st_size,
        "source_mtime_ns": original_stat.st_mtime_ns,
        "source_metadata": metadata,
        "output_fps": str(rate),
        "decoded_shape": frame_shape,
        "output_transform": transform,
        "input_frames": total_in,
        "selected_frames": total_out,
        "input_seconds_at_nominal_fps": total_in / fps,
        "selected_seconds": total_out / fps,
        "reduction_percent": reduction,
        "elapsed_seconds": elapsed,
        "timing_note": "Encoder work overlaps Python; pipe write/wait time is not total encoding CPU time.",
        "decode_and_retrieve_seconds": read_seconds,
        "selection_seconds": analysis_seconds,
        "encoder_write_wait_seconds": encode_wait_seconds,
        "hand_analysis": hand_stats,
        "camera_analysis": camera_stats,
        "center_dedupe": center_stats,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    if args.report:
        write_report(destination, report)
    log(
        f"[{status.upper()}] {source.name}: {total_in:,} -> {total_out:,} frames "
        f"({reduction:.1f}% removed), {total_in / fps:.2f}s -> {total_out / fps:.2f}s, "
        f"{elapsed:.2f}s elapsed ({total_in / max(elapsed, 0.001):.1f} input fps)"
    )
    if status == "no_selection":
        log(
            "  No MOV created. No activity was selected; inspect a short sample and its --report statistics."
        )
    elif status == "unchanged":
        log(
            "  No MOV created because the selection removed no frames and no portrait rotation was needed."
        )
    if status != "written" and destination.exists():
        log(
            f"  An older output still exists and was left unchanged: {destination.name}"
        )
    if status == "written":
        log(f"  Output: {destination}")
    return status
