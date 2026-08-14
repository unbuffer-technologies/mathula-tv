#!/usr/bin/env python3
"""Run Mathula TV's Azure-STT production dubbing feedback loop."""

from __future__ import annotations

import sys

from mathula_tv.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["optimize-dub", *sys.argv[1:]]))
