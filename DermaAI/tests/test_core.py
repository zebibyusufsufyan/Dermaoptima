import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.predict import assess_image_quality, compare_followup_images, validate_image_bytes


def png_bytes():
    buf = io.BytesIO(); Image.new("RGB", (16, 16), "red").save(buf, format="PNG"); return buf.getvalue()


def test_validate_png():
    image = validate_image_bytes(png_bytes(), "image/png")
    assert image.mode == "RGB"


def test_reject_bad_type():
    with pytest.raises(ValueError): validate_image_bytes(png_bytes(), "application/pdf")


def test_reject_corrupt():
    with pytest.raises(ValueError): validate_image_bytes(b"not an image", "image/png")


def test_reject_empty():
    with pytest.raises(ValueError): validate_image_bytes(b"", "image/png")


def test_quality_flags_small_image():
    quality = assess_image_quality(Image.new("RGB", (128, 128), "red"))
    assert quality["issues"]
    assert quality["acceptable"] is False


def test_followup_comparison_returns_cautious_signal():
    previous = Image.new("RGB", (64, 64), "red")
    current = Image.new("RGB", (64, 64), "red")
    comparison = compare_followup_images(previous, current)
    assert comparison["signal"] == "little visible change"
    assert "cannot determine healing" in comparison["limitation"]
