import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def _load_app_module_with_inmemory_sqlite(repo_root: Path):
    """
    Load app.py into a fresh module, forcing an in-memory SQLite URI.
    This avoids touching the on-disk database.
    """

    app_path = repo_root / "app.py"
    source = app_path.read_text(encoding="utf-8")

    needle = 'app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"'
    replacement = 'app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"'
    if needle not in source:
        raise RuntimeError("Expected SQLALCHEMY_DATABASE_URI assignment not found in app.py")

    source = source.replace(needle, replacement, 1)

    module_name = "smartdentalscan_app_inmem"
    sys.modules.pop(module_name, None)

    mod = types.ModuleType(module_name)
    mod.__file__ = str(app_path)

    code = compile(source, str(app_path), "exec")
    exec(code, mod.__dict__)

    sys.modules[module_name] = mod
    return mod


@pytest.fixture(scope="module")
def app_module_inmem():
    repo_root = Path(__file__).resolve().parents[1]

    cwd = Path.cwd()
    try:
        import os

        os.chdir(repo_root)
        mod = _load_app_module_with_inmemory_sqlite(repo_root)

        # Test-safe runtime config
        mod.app.config.update(
            TESTING=True,
            WTF_CSRF_ENABLED=False,
        )

        # Disable audit logging if present
        try:
            mod.audit_logger.disabled = True
        except Exception:
            pass

        try:
            mod.audit_event = lambda *args, **kwargs: None
        except Exception:
            pass

        # Make /predict fast and deterministic
        import torch

        class DummyModel(torch.nn.Module):
            def forward(self, x):
                # Fixed logits for 3 classes
                return torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)

        mod.class_names = ["ClassA", "ClassB", "ClassC"]
        mod.model = DummyModel().eval()
        mod.transform = lambda _img: torch.zeros((3, 224, 224), dtype=torch.float32)

        return mod
    finally:
        cwd and __import__("os").chdir(cwd)


@pytest.fixture()
def client(app_module_inmem):
    return app_module_inmem.app.test_client()


def _png_file():
    """
    Create a real, valid PNG image entirely in memory.
    Pillow requires a proper PNG structure (IHDR/IDAT/IEND).
    """
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return (buf, "scan.png", "image/png")


def test_get_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_guest_can_predict_exactly_once(client):
    resp = client.post(
        "/predict",
        data={"image": _png_file()},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert resp.status_code == 200
    body = resp.data.decode("utf-8", errors="ignore")

    assert "Guest mode: results are not saved" in body
    assert "Top prediction" in body
    assert "ClassA" in body


def test_second_guest_prediction_is_blocked(client):
    # First prediction consumes the guest slot
    client.post(
        "/predict",
        data={"image": _png_file()},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    # Second attempt should redirect to /login
    resp2 = client.post(
        "/predict",
        data={"image": _png_file()},
        content_type="multipart/form-data",
        follow_redirects=False,
    )

    assert resp2.status_code in (301, 302, 303)
    assert "/login" in (resp2.headers.get("Location") or "")


def test_invalid_image_upload_returns_error(client):
    resp = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"GIF89a"), "bad.gif", "image/gif")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert resp.status_code == 200
    body = resp.data.decode("utf-8", errors="ignore")
    assert "Unsupported file type" in body
