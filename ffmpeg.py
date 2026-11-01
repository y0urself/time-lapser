#!/usr/bin/env python3
"""Compatibility launcher; use the complete project checkout, not this file alone."""

from pathlib import Path

from castlestube_cleanup.cli import main

if __name__ == "__main__":
    raise SystemExit(main(model_directory=Path(__file__).resolve().parent))
