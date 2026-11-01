"""Command-line arguments, input discovery and multi-video orchestration."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from .hands import HAND_DEFAULTS, import_yolo
from .pipeline import process_file
from .runtime import ACTIVE_ENCODERS, PROCESS_LOCK, log
from .version import __version__ as VERSION


def discover_files(source, extensions):
    if source.is_file():
        if source.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported input extension: {source.suffix}")
        return [source], source.parent
    files = []
    for folder, directories, names in os.walk(source, followlinks=False):
        directories[:] = [
            d
            for d in directories
            if d.casefold() != "output"
            and not d.startswith(".castlestube-")
            and not (Path(folder) / d).is_symlink()
        ]
        for name in names:
            file = Path(folder) / name
            if (
                file.suffix.lower() in extensions
                and file.is_file()
                and not file.is_symlink()
            ):
                files.append(file)
    return sorted(files, key=lambda p: str(p).casefold()), source


def output_names(files, root, output_dir, args):
    counts = Counter(p.stem.casefold() for p in files)
    used = set()
    pairs = []
    for file in files:
        base = file.stem
        digest = hashlib.sha256(file.relative_to(root).as_posix().encode()).hexdigest()[
            :10
        ]
        if counts[base.casefold()] > 1:
            base += "_" + digest
        if args.limit_seconds is not None:
            base += "_preview" + format(args.limit_seconds, "g").replace(".", "p") + "s"
        name = base + "_clean.mov"
        while name.casefold() in used:
            base += "_" + digest
            name = base + "_clean.mov"
        used.add(name.casefold())
        pairs.append((file, output_dir / name))
    return pairs


def arguments(argv=None, *, model_directory=None):
    parser = argparse.ArgumentParser(
        description="Clean build videos using streaming export, optional hand detection and camera-motion preservation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument(
        "input", type=Path, help="Input video or folder (searched recursively)"
    )
    parser.add_argument(
        "--ext", nargs="+", default=[".mp4", ".mov"], help="Input extensions"
    )
    parser.add_argument(
        "--mode", choices=["dedupe", "hands", "hands-dedupe"], default="dedupe"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="Higher values remove more similar frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Forward dedupe: keep up to this many frames in a similar run",
    )
    parser.add_argument(
        "--compare",
        choices=["kept", "previous"],
        default="kept",
        help="Forward dedupe reference frame",
    )
    parser.add_argument(
        "--dedupe-order",
        choices=["forward", "center"],
        default="forward",
        help="Center: hands-dedupe only; remove similar interior frames from each hand phase's middle outwards",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=20,
        help="Center dedupe: stop shortening a hand phase at this many frames, plus protected context; not a hard cap",
    )
    parser.add_argument(
        "--no-equalize",
        action="store_true",
        help="Disable histogram equalization; retune threshold",
    )
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Normalized hand/dedupe region, e.g. 0.15 0.10 0.70 0.80; camera detection/export use the full frame",
    )
    parser.add_argument(
        "--codec", choices=["prores", "h264", "h264_nvenc"], default="prores"
    )
    parser.add_argument(
        "--rotate-portrait",
        choices=["cw", "ccw", "off"],
        default="cw",
        help="Rotate decoded portrait output 90 degrees clockwise/counterclockwise, or off; landscape stays unchanged; analysis/ROI are unaffected",
    )
    parser.add_argument(
        "--prores-profile",
        type=int,
        choices=range(4),
        default=3,
        help="0=Proxy, 1=LT, 2=422, 3=HQ; applies only to ProRes",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 CRF / NVENC CQ; does not apply to ProRes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent videos; try 1 before increasing",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Codec threads per video; OpenCV image operations use one thread",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyze/count only; never encode a video",
    )
    parser.add_argument(
        "--limit-seconds",
        type=float,
        help="Process only the beginning; use a separate preview name",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write/replace a JSON summary alongside the output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs only after successful processing",
    )
    parser.add_argument(
        "--hand-detector",
        choices=tuple(HAND_DEFAULTS),
        default="mediapipe",
        help="Hand backend; existing commands continue to use MediaPipe",
    )
    parser.add_argument(
        "--hand-model",
        type=Path,
        default=argparse.SUPPRESS,
        help="Local model; default in working directory: hand_landmarker.task (MediaPipe) or YOLOv10x_hands.pt (YOLO)",
    )
    parser.add_argument(
        "--hand-step",
        type=int,
        default=1,
        help="Run hand inference every N source frames; 1 for exact context boundaries",
    )
    parser.add_argument(
        "--hand-width",
        type=int,
        default=640,
        help="MediaPipe max image side / YOLO imgsz (stride-rounded)",
    )
    parser.add_argument(
        "--hand-confidence",
        type=float,
        default=argparse.SUPPRESS,
        help="Override backend confidence: MediaPipe 0.5, YOLO 0.1; lower retains more candidates",
    )
    parser.add_argument(
        "--hand-device",
        default="cpu",
        help="YOLO inference device: cpu, 0 (CUDA GPU), or mps; independent of video codec",
    )
    parser.add_argument(
        "--hand-min-area",
        type=float,
        default=0.005,
        help="YOLO minimum box area / analysis image area: 0.005 = 0.5%%; 0 disables; applies to ROI if set",
    )
    parser.add_argument(
        "--keep-camera-motion",
        action="store_true",
        help="Keep detected global camera movement and context in every mode; analyze each full source frame",
    )
    parser.add_argument(
        "--camera-threshold",
        type=float,
        default=0.002,
        help="Minimum global displacement / image diagonal per frame; lower retains slower motion and more shake",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=480,
        help="Maximum side of camera analysis image; larger costs more processing",
    )
    parser.add_argument(
        "--pre-frames",
        type=int,
        default=2,
        help="Original frames before each hand/camera phase; protected from deduplication",
    )
    parser.add_argument(
        "--post-frames",
        type=int,
        default=2,
        help="Original frames after each hand/camera phase; protected from deduplication",
    )
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=0.0,
        help="Additional input seconds before each hand/camera phase",
    )
    parser.add_argument(
        "--post-roll",
        type=float,
        default=0.0,
        help="Additional input seconds after each hand/camera phase",
    )
    parser.add_argument(
        "--bridge-gap",
        type=float,
        default=0.0,
        help="Merge gaps up to this duration after padding",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="FFprobe executable")
    args = parser.parse_args(argv)
    model_name, default_confidence = HAND_DEFAULTS[args.hand_detector]
    if getattr(args, "hand_model", None) is None:
        args.hand_model = (
            Path(model_directory) if model_directory is not None else Path.cwd()
        ) / model_name
    if getattr(args, "hand_confidence", None) is None:
        args.hand_confidence = default_confidence
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255.")
    for name in ("max_frames", "workers", "threads", "hand_step", "hand_width"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in ("pre_frames", "post_frames"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative.")
    if args.segment_frames < 2:
        parser.error(
            "--segment-frames must be at least 2 to preserve both phase endpoints."
        )
    if args.dedupe_order == "center" and args.mode != "hands-dedupe":
        parser.error("--dedupe-order center requires --mode hands-dedupe.")
    if args.limit_seconds is not None and (
        not math.isfinite(args.limit_seconds) or args.limit_seconds <= 0
    ):
        parser.error("--limit-seconds must be a positive finite number.")
    for name in ("pre_roll", "post_roll", "bridge_gap"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative.")
    if not 0 <= args.hand_confidence <= 1 or not 0 <= args.crf <= 51:
        parser.error(
            "Confidence must be between 0 and 1; CRF/CQ must be between 0 and 51."
        )
    if not math.isfinite(args.hand_min_area) or not 0 <= args.hand_min_area <= 1:
        parser.error(
            "--hand-min-area must be a finite fraction between 0 and 1 (0.005 = 0.5%)."
        )
    if not math.isfinite(args.camera_threshold) or not 0 < args.camera_threshold < 1:
        parser.error("--camera-threshold must be finite and strictly between 0 and 1.")
    if args.camera_width < 96:
        parser.error("--camera-width must be at least 96 pixels.")
    if args.roi:
        x, y, w, h = args.roi
        if not (
            all(math.isfinite(v) for v in args.roi)
            and x >= 0
            and y >= 0
            and w > 0
            and h > 0
            and x + w <= 1
            and y + h <= 1
        ):
            parser.error(
                "ROI must fit inside the image: x,y >= 0, w,h > 0, x+w <= 1, y+h <= 1."
            )
    args.input = args.input.expanduser().resolve()
    args.hand_model = args.hand_model.expanduser().resolve()
    if args.mode != "dedupe":
        expected_suffix = ".pt" if args.hand_detector == "yolo" else ".task"
        if args.hand_model.suffix.lower() != expected_suffix:
            parser.error(
                f"--hand-detector {args.hand_detector} needs a {expected_suffix} model; check --hand-model."
            )
        if not args.hand_device.strip():
            parser.error("--hand-device must not be empty; use cpu, 0, or mps.")
    args.ext = {"." + ext.lstrip(".").lower() for ext in args.ext}
    # A list is easier to serialize in reports than a set.
    args.ext = sorted(args.ext)
    return args


def main(argv=None, *, model_directory=None):
    args = arguments(argv, model_directory=model_directory)
    if not args.input.exists():
        log(f"[ERROR] Input does not exist: {args.input}")
        return 1
    for executable in (args.ffprobe, args.ffmpeg):
        if shutil.which(executable) is None:
            log(f"[ERROR] Executable not found: {executable}")
            return 1
    if args.mode != "dedupe":
        if args.hand_step > 1:
            log(
                "[WARN] Sampled hand boundaries are approximate; use --hand-step 1 for exact neighboring frames."
            )
        if not args.hand_model.is_file():
            log(
                f"[ERROR] Hand model missing: {args.hand_model}\n"
                f"Download {HAND_DEFAULTS[args.hand_detector][0]} using the URL in docs/usage.md, "
                "place it in the working directory, or set --hand-model."
            )
            return 1
        try:
            if args.hand_detector == "yolo":
                import_yolo()
            else:
                import mediapipe  # noqa: F401 -- checked before starting workers
        except (ImportError, OSError, RuntimeError) as exc:
            log(
                f"[ERROR] {args.hand_detector} could not load: {exc}\n"
                f"Install the selected backend: uv sync --extra {args.hand_detector}"
            )
            return 1
    try:
        files, root = discover_files(args.input, args.ext)
    except ValueError as exc:
        log(f"[ERROR] {exc}")
        return 1
    if not files:
        log("[ERROR] No matching video files found.")
        return 1
    output_dir = root / "output"
    output_dir.mkdir(exist_ok=True)
    pairs = output_names(files, root, output_dir, args)
    cv2.setNumThreads(1)
    stop_event = threading.Event()
    errors = 0
    log(
        f"CastleStube cleanup {VERSION}: {len(files)} file(s), {args.workers} worker(s). "
        "Output is silent, constant frame rate, 8-bit SDR processing."
    )
    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(process_file, source, dest, args, stop_event): source
        for source, dest in pairs
    }
    try:
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors += 1
                log(f"[ERROR] {futures[future].name}: {exc}")
    except KeyboardInterrupt:
        stop_event.set()
        for future in futures:
            future.cancel()
        with PROCESS_LOCK:
            for proc in list(ACTIVE_ENCODERS):
                if proc.poll() is None:
                    proc.terminate()
        log("[CANCELLED] Cleaning up temporary outputs.")
        return 130
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    log(
        f"Finished: {len(files) - errors} file(s) processed or skipped, {errors} error(s)."
    )
    return 1 if errors else 0
