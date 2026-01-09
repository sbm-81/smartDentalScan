#!/usr/bin/env python3
"""Benchmark model inference time (CPU).

- Imports the Flask app module (`app`), which loads the trained model.
- Runs inference 20 times on the same preprocessed image tensor.
- Reports min/max/avg inference time.

This script does NOT modify model code; it only calls the existing `model` and
`transform` objects exposed by `app.py`.

Exit code is non-zero on failure.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path


def _make_input_tensor(app_module):
    from PIL import Image

    # Deterministic in-memory image; same input each run.
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    tensor = app_module.transform(img).unsqueeze(0)
    return tensor


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # app.py loads assets via relative paths at import time.
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    start_import = time.perf_counter()
    try:
        import importlib

        sys.modules.pop("app", None)
        app_module = importlib.import_module("app")
    except Exception as exc:
        elapsed = time.perf_counter() - start_import
        print(f"FAIL: importing app raised {type(exc).__name__}: {exc}")
        print(f"Import elapsed: {elapsed:.3f}s")
        return 1

    import_elapsed = time.perf_counter() - start_import

    try:
        import torch

        model = app_module.model
        model.eval()

        x = _make_input_tensor(app_module)

        # Make sure we can run at least once.
        with torch.no_grad():
            _ = model(x)

        times = []
        runs = 20
        for _i in range(runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            times.append(time.perf_counter() - t0)

        t_min = min(times)
        t_max = max(times)
        t_avg = statistics.fmean(times)

        print(f"Imported app/model in: {import_elapsed:.3f}s")
        print(f"Inference runs: {runs}")
        print(f"Inference time (s): min={t_min:.6f} max={t_max:.6f} avg={t_avg:.6f}")
        return 0

    except Exception as exc:
        print(f"FAIL: benchmarking raised {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
