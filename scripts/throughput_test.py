#!/usr/bin/env python3
"""Sequential /predict throughput test (no UI automation).

Requirements:
- Sends sequential prediction requests (n=30)
- Measures total time
- Calculates predictions/minute

Notes:
- Uses Flask test client (in-process requests; no network).
- Does NOT modify model code on disk.
- Avoids production DB modification by loading app.py with an in-memory SQLite URI.
- Disables audit logging at runtime to avoid writing audit.log during the benchmark.
"""

from __future__ import annotations

import io
import os
import sys
import time
import types
from pathlib import Path


def _load_app_module_inmemory_db(repo_root: Path):
    app_path = repo_root / "app.py"
    source = app_path.read_text(encoding="utf-8")

    needle = 'app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"'
    replacement = 'app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"'
    if needle not in source:
        raise RuntimeError("Expected SQLALCHEMY_DATABASE_URI assignment not found in app.py")
    source = source.replace(needle, replacement, 1)

    mod = types.ModuleType("smartdentalscan_app_throughput")
    mod.__file__ = str(app_path)
    code = compile(source, str(app_path), "exec")
    exec(code, mod.__dict__)
    return mod


def _png_payload_bytes() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(10, 20, 30))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    # app.py reads classes.txt/model.pth via relative paths at import time.
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    try:
        app_module = _load_app_module_inmemory_db(repo_root)
    except Exception as exc:
        print(f"FAIL: could not load app module: {type(exc).__name__}: {exc}")
        return 1

    # Test-client friendly configuration
    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)

    # Avoid writing to audit.log during the run.
    try:
        app_module.audit_logger.disabled = True
    except Exception:
        pass
    try:
        app_module.audit_event = lambda *args, **kwargs: None
    except Exception:
        pass

    client = app_module.app.test_client()

    image_bytes = _png_payload_bytes()
    n = 30

    t0 = time.perf_counter()
    for i in range(n):
        file_tuple = (io.BytesIO(image_bytes), f"scan_{i}.png", "image/png")
        resp = client.post(
            "/predict",
            data={"image": file_tuple},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        if resp.status_code != 200:
            print(f"FAIL: request {i+1}/{n} got status {resp.status_code}")
            return 2

        # Ensure we actually got a prediction page (basic guard).
        body = resp.data.decode("utf-8", errors="ignore")
        if "Top prediction" not in body:
            print(f"FAIL: request {i+1}/{n} did not return a prediction result")
            return 3

        # New client each iteration to avoid guest one-scan-per-session limit.
        client = app_module.app.test_client()

    total = time.perf_counter() - t0
    per_min = (n / total) * 60.0 if total > 0 else 0.0

    print(f"Requests: {n}")
    print(f"Total time (s): {total:.3f}")
    print(f"Predictions/minute: {per_min:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
