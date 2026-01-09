import importlib
import io
import os
import shutil
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def app_module(tmp_path_factory: pytest.TempPathFactory, repo_root: Path):
    """Import the Flask app in an isolated working directory.

    The application loads assets and initializes SQLite at import time using
    relative paths (classes.txt, model.pth, sqlite:///app.db). To keep tests
    reproducible and avoid mutating the repo DB, we:
    - create a temp workdir
    - place/symlink required assets into it
    - chdir into it before importing app
    """

    workdir = tmp_path_factory.mktemp("smartdentalscan_workdir")

    for filename in ["classes.txt", "model.pth"]:
        src = repo_root / filename
        dst = workdir / filename
        if dst.exists():
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copyfile(src, dst)

    os.chdir(workdir)
    sys.path.insert(0, str(repo_root))
    sys.modules.pop("app", None)

    mod = importlib.import_module("app")

    # Test-safe config. CSRF would otherwise block all POSTs.
    mod.app.config.update(
        TESTING=True,
        WTF_CSRF_ENABLED=False,
    )

    # Avoid writing audit logs during tests.
    try:
        mod.audit_logger.disabled = True
    except Exception:
        pass

    # Ensure a clean DB for the session in the temp workdir.
    with mod.app.app_context():
        mod.db.drop_all()
        mod.db.create_all()
        try:
            mod.ensure_prediction_notes_column()
        except Exception:
            pass

    return mod


@pytest.fixture(autouse=True)
def clean_db(app_module):
    """Clear DB tables between tests for reproducibility."""
    with app_module.app.app_context():
        app_module.db.session.query(app_module.Prediction).delete()
        app_module.db.session.query(app_module.User).delete()
        app_module.db.session.commit()
    yield


@pytest.fixture()
def app(app_module):
    return app_module.app


@pytest.fixture()
def client(app):
    return app.test_client()


def _create_user(app_module, email: str, password: str):
    with app_module.app.app_context():
        user = app_module.User(email=email.strip().lower())
        user.set_password(password)
        app_module.db.session.add(user)
        app_module.db.session.commit()
        app_module.db.session.refresh(user)
        return user


@pytest.fixture()
def create_user(app_module):
    return lambda email="user@example.com", password="password123": _create_user(
        app_module, email, password
    )


@pytest.fixture()
def login(client, create_user):
    def _login(email="user@example.com", password="password123"):
        create_user(email=email, password=password)
        return client.post(
            "/login",
            data={"email": email, "password": password},
            follow_redirects=False,
        )

    return _login


@pytest.fixture()
def make_png_bytes():
    def _make_png_bytes():
        from PIL import Image

        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return bio

    return _make_png_bytes
