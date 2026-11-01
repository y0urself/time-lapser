"""Installed entry point, model paths and Windows-safe output naming."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from castlestube_cleanup import __version__, cli


class CliTests(unittest.TestCase):
    def test_installed_and_legacy_model_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch.object(Path, "cwd", return_value=root):
                self.assertEqual(
                    cli.arguments(["clip.mov"]).hand_model,
                    root / "hand_landmarker.task",
                )
            legacy = root / "script"
            args = cli.arguments(
                ["clip.mov", "--hand-detector", "yolo"], model_directory=legacy
            )
            self.assertEqual(args.hand_model, legacy / "YOLOv10x_hands.pt")
            explicit = root / "custom.pt"
            args = cli.arguments(
                ["clip.mov", "--hand-detector", "yolo", "--hand-model", str(explicit)],
                model_directory=legacy,
            )
            self.assertEqual(args.hand_model, explicit)

    def test_module_and_compatibility_entry_points(self):
        launcher = Path(__file__).resolve().parents[1] / "ffmpeg.py"
        for invocation in (["-m", "castlestube_cleanup"], [str(launcher)]):
            with (
                self.subTest(invocation=invocation),
                tempfile.TemporaryDirectory() as tmp,
            ):
                result = subprocess.run(
                    [sys.executable, *invocation, "--version"],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(__version__, result.stdout)

    def test_scan_skips_outputs_and_accepts_uppercase_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / "build" / "CLIP.MOV",
                root / "clip.mp4",
                root / "output" / "clip_clean.mov",
                root / ".castlestube-test" / "partial.mov",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            files, found_root = cli.discover_files(root, [".mov", ".mp4"])
            self.assertEqual(set(files), set(paths[:2]))
            self.assertEqual(found_root, root)

    def test_duplicate_stems_and_preview_names_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / "a" / "clip.mov", root / "b" / "CLIP.mp4"]
            args = cli.arguments([str(root), "--limit-seconds", "1.5"])
            pairs = cli.output_names(files, root, root / "output", args)
            names = [p.name.casefold() for _, p in pairs]
            self.assertEqual(len(set(names)), 2)
            self.assertTrue(all(n.endswith("_preview1p5s_clean.mov") for n in names))
            self.assertEqual(
                pairs, cli.output_names(files, root, root / "output", args)
            )

    def test_invalid_region_and_nonfinite_options_fail_early(self):
        invalid = [
            ["--roi", "0.9", "0", "0.5", "1"],
            ["--threshold", "nan"],
            ["--limit-seconds", "inf"],
            ["--bridge-gap", "-1"],
            ["--threads", "0"],
        ]
        for flags in invalid:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.arguments(["clip.mov", *flags])
