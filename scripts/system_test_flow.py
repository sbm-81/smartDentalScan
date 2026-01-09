#!/usr/bin/env python3
"""End-to-end system test flow (no UI automation).

Steps:
1) Register a new user (/signup)
2) Log in (/login)
3) Upload a valid image (/predict)
4) Receive prediction (response contains "Top prediction")
5) Confirm prediction saved in DB (Prediction row for the user)

Key constraints:
- Uses Flask test client
- Does NOT modify application code on disk
- Avoids production DB modification by running with a temp CWD, so sqlite:///app.db
  is created in a temporary directory.

Exit codes:
- 0: success
- non-zero: failure (printed reason)
"""

from __future__ import annotations

import importlib
import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _make_png_file_tuple() -> tuple[io.BytesIO, str, str]:
    from PIL import Image

    img = Image.new("RGB", (12, 12), color=(120, 10, 200))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio, "scan.png", "image/png"


def _prepare_temp_workdir(repo_root: Path) -> str:
    workdir = tempfile.mkdtemp(prefix="smartdentalscan_system_test_")

    # app.py loads these via relative paths at import time.
    for filename in ["classes.txt", "model.pth"]:
        src = repo_root / filename
        dst = Path(workdir) / filename
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copyfile(src, dst)

    return workdir


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    workdir = _prepare_temp_workdir(repo_root)

    print(f"Workdir: {workdir}")

    # Import app from repo root, but ensure sqlite:///app.db is created in workdir.
    os.chdir(workdir)
    sys.path.insert(0, str(repo_root))

    start_import = time.perf_counter()
    try:
        sys.modules.pop("app", None)
        app_module = importlib.import_module("app")
    except Exception as exc:
        elapsed = time.perf_counter() - start_import
        print(f"FAIL: importing app raised {type(exc).__name__}: {exc}")
        print(f"Import elapsed: {elapsed:.3f}s")
        return 1

    import_elapsed = time.perf_counter() - start_import
    print(f"Imported app in {import_elapsed:.3f}s")

    # Test-safe config: disable CSRF for test client POSTs.
    app_module.app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)

    # Make inference deterministic and fast for the system test.
    # This is runtime monkeypatching in the script only (no code changes on disk).
    try:
        import torch

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)

        app_module.class_names = ["ClassA", "ClassB", "ClassC"]
        app_module.model = DummyModel().eval()
        app_module.transform = lambda _img: torch.zeros((3, 224, 224), dtype=torch.float32)
    except Exception as exc:
        print(f"FAIL: could not patch model/transform for deterministic run: {exc}")
        return 2

    # Fresh DB.
    with app_module.app.app_context():
        app_module.db.drop_all()
        app_module.db.create_all()
        try:
            app_module.ensure_prediction_notes_column()
        except Exception:
            pass

    email = f"test_{int(time.time())}@example.com"
    password = "password123"

    client = app_module.app.test_client()

    # 1) Register
    resp = client.post(
        "/signup",
        data={"email": email, "password": password, "confirm_password": password},
        follow_redirects=True,
    )
    if resp.status_code != 200:
        print(f"FAIL: /signup status {resp.status_code}")
        return 3

    # 2) Login
    resp = client.post(
        "/login",
        data={"email": email, "password": password},
        follow_redirects=False,
    )
    if resp.status_code not in (301, 302, 303):
        print(f"FAIL: /login expected redirect, got {resp.status_code}")
        return 4

    # 3) Upload valid image
    start_predict = time.perf_counter()
    resp = client.post(
        "/predict",
        data={"image": _make_png_file_tuple()},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    predict_elapsed = time.perf_counter() - start_predict

    if resp.status_code != 200:
        print(f"FAIL: /predict status {resp.status_code}")
        return 5

    body = resp.data.decode("utf-8", errors="ignore")
    if "Top prediction" not in body:
        print("FAIL: prediction page did not include 'Top prediction'")
        return 6

    if "ClassA" not in body:
        print("FAIL: prediction page did not include expected label 'ClassA'")
        return 7

    print(f"Prediction request completed in {predict_elapsed:.3f}s")

    # 5) Confirm prediction saved in DB for this user.
    with app_module.app.app_context():
        user = app_module.User.query.filter_by(email=email.lower()).first()
        if user is None:
            print("FAIL: user not found in DB after signup")
            return 8

        preds = app_module.Prediction.query.filter_by(user_id=user.id).all()
        if len(preds) != 1:
            print(f"FAIL: expected 1 prediction row, found {len(preds)}")
            return 9

        pred = preds[0]
        if pred.predicted_label != "ClassA":
            print(f"FAIL: expected predicted_label=ClassA, got {pred.predicted_label}")
            return 10

        print(f"OK: DB saved prediction id={pred.id}, label={pred.predicted_label}, confidence={pred.confidence}")

    print("OK: system test flow passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
