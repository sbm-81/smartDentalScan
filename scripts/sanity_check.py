#!/usr/bin/env python3
"""Sanity check: import the Flask app and confirm the ML model loads.

This script is intentionally minimal and does NOT modify application code.
It measures the time spent importing `app` (which loads the model at import time)
then prints:
- model load/import time (seconds)
- number of classes

Exit code is non-zero on failure for easy CI/reporting.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # The application loads assets via relative paths at import time.
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    start = time.perf_counter()
    try:
        mod = importlib.import_module("app")
    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(f"FAIL: importing app raised {type(exc).__name__}: {exc}")
        print(f"Import elapsed: {elapsed:.3f}s")
        return 1

    elapsed = time.perf_counter() - start

    class_names = getattr(mod, "class_names", None)
    model = getattr(mod, "model", None)

    if not isinstance(class_names, list) or not class_names:
        print("FAIL: `class_names` missing or empty")
        print(f"Import elapsed: {elapsed:.3f}s")
        return 2

    if model is None:
        print("FAIL: `model` missing")
        print(f"Import elapsed: {elapsed:.3f}s")
        return 3

    # Basic consistency check: model final layer outputs match class count if available.
    try:
        out_features = getattr(getattr(model, "fc", None), "out_features", None)
    except Exception:
        out_features = None

    print(f"OK: import/model load time: {elapsed:.3f}s")
    print(f"OK: number of classes: {len(class_names)}")
    if out_features is not None:
        print(f"OK: model.fc.out_features: {out_features}")
        if int(out_features) != len(class_names):
            print("WARN: out_features does not match class count")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
