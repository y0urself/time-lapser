"""Shared progress logging and encoder cancellation state."""

from __future__ import annotations

import subprocess
import threading
import time

PRINT_LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()
ACTIVE_ENCODERS: set[subprocess.Popen] = set()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


class Progress:
    def __init__(self, label):
        self.label = label
        self.last = time.monotonic()

    def __call__(self, stage, seen, selected):
        now = time.monotonic()
        if now - self.last >= 5:
            log(f"[{self.label}] {stage}: {seen:,} input frames; {selected:,} selected")
            self.last = now
