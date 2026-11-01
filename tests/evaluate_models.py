"""Explicit real-model evaluation; never downloaded or run by normal unit CI."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from castlestube_cleanup import cli, hands, video

DATA = Path(__file__).with_name("data")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def scores(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {
        **counts,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
    }


def main(argv=None):
    import sys

    flags = sys.argv[1:] if argv is None else argv
    # Reuse detector options and validation; fixed evaluation mode uses every frame.
    args = cli.arguments([str(DATA), *flags, "--mode", "hands", "--hand-step", "1"])
    if args.roi:
        raise ValueError("Fixture labels cover the whole image; omit --roi.")
    manifest = json.loads((DATA / "manifest.json").read_text())
    total = dict(tp=0, fp=0, fn=0, tn=0)
    clips = []
    for clip in manifest["clips"]:
        source = DATA / clip["file"]
        if digest(source) != clip["sha256"]:
            raise ValueError(f"Fixture checksum mismatch: {source.name}")
        detector = hands.create_hand_detector(args)
        cap = None
        counts = dict(tp=0, fp=0, fn=0, tn=0)
        index = 0
        try:
            cap = video.open_capture(source, args.threads)
            fps = float(video.frame_rate({"avg_frame_rate": clip["fps"]}, cap))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                expected = any(
                    start <= index < end for start, end in clip["hand_present_spans"]
                )
                actual = detector.present(frame, index, fps)
                key = (
                    ("tp" if expected else "fp")
                    if actual
                    else ("fn" if expected else "tn")
                )
                counts[key] += 1
                index += 1
        finally:
            if cap is not None:
                cap.release()
            detector.close()
        if index != clip["frames"]:
            raise ValueError(
                f"Fixture decode incomplete: {source.name}: {index} frames"
            )
        clips.append({"file": clip["file"], **scores(counts)})
        for key, value in counts.items():
            total[key] += value
    versions = {}
    for name in (
        "castlestube-video-cleanup",
        "numpy",
        "opencv-python",
        "mediapipe",
        "ultralytics",
        "torch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    print(
        json.dumps(
            {
                "backend": args.hand_detector,
                "model": args.hand_model.name,
                "model_sha256": digest(args.hand_model),
                "confidence": args.hand_confidence,
                "min_area": args.hand_min_area,
                "width": args.hand_width,
                "device": args.hand_device,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "versions": versions,
                "clips": clips,
                "total": scores(total),
                "note": "Two short re-encoded clips; exploratory metrics, not a general accuracy guarantee.",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
