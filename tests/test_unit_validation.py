import io
import pytest
from PIL import Image


class DummyUpload:
    def __init__(self, filename: str, mimetype: str):
        self.filename = filename
        self.mimetype = mimetype


@pytest.mark.parametrize(
    "filename,mimetype",
    [
        ("scan.jpg", "image/jpeg"),
        ("scan.jpeg", "image/jpeg"),
        ("scan.png", "image/png"),
        ("SCAN.PNG", "image/png"),
        ("my scan.png", "image/png"),
    ],
)
def test_allowed_image_extensions_accept(app_module, filename, mimetype):
    allowed, ext, error = app_module.is_allowed_upload(DummyUpload(filename, mimetype))
    assert allowed is True
    assert ext in app_module.ALLOWED_IMAGE_EXTENSIONS
    assert error is None


@pytest.mark.parametrize(
    "filename,mimetype",
    [
        ("scan.gif", "image/gif"),
        ("scan.bmp", "image/bmp"),
        ("scan.tiff", "image/tiff"),
        ("scan", "image/png"),
        ("", "image/png"),
    ],
)
def test_rejected_extensions(app_module, filename, mimetype):
    allowed, ext, error = app_module.is_allowed_upload(DummyUpload(filename, mimetype))
    assert allowed is False
    assert error


@pytest.mark.parametrize(
    "mimetype",
    [
        "application/octet-stream",
        "image/gif",
        "text/plain",
        "",
        None,
    ],
)
def test_mime_type_validation_rejects_invalid(app_module, mimetype):
    upload = DummyUpload("scan.png", mimetype)
    allowed, ext, error = app_module.is_allowed_upload(upload)
    assert ext == ".png"
    assert allowed is False
    assert "MIME" in (error or "")


def test_image_preprocessing_transform_shape_and_dtype(app_module):
    img = Image.new("RGB", (10, 10), color=(10, 20, 30))
    tensor = app_module.transform(img)

    assert tuple(tensor.shape) == (3, 224, 224)

    import torch
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
