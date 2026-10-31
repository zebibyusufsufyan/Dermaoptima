"""Prediction helpers with no persistence of uploaded image bytes."""
from __future__ import annotations

from io import BytesIO
from typing import Any, BinaryIO

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

ALLOWED_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_BYTES = 10 * 1024 * 1024


def assess_image_quality(image: Image.Image) -> dict[str, Any]:
    """Return conservative capture-quality signals before model inference."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = rgb.mean(axis=2)
    height, width = gray.shape
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    focus_score = float((dx.var() + dy.var()) / 2.0)
    brightness = float(gray.mean())
    issues: list[str] = []
    if min(width, height) < 512:
        issues.append("The image is low resolution (short side below 512 px).")
    if focus_score < 18:
        issues.append("The image may be soft or out of focus.")
    if brightness < 35 or brightness > 225:
        issues.append("Lighting may be too dark or overexposed.")
    return {
        "width": width,
        "height": height,
        "megapixels": round((width * height) / 1_000_000, 2),
        "focus_score": round(focus_score, 2),
        "brightness": round(brightness, 1),
        "issues": issues,
        "acceptable": not issues,
    }


def compare_followup_images(previous: Image.Image, current: Image.Image) -> dict[str, Any]:
    """Compare captures as a review signal; this is not lesion segmentation."""
    a = np.asarray(previous.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    b = np.asarray(current.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
    mean_absolute_change = float(np.abs(a - b).mean())
    brightness_change = float(b.mean() - a.mean())
    # A broad image difference cannot establish healing; it only prioritizes review.
    if mean_absolute_change < 0.06:
        signal = "little visible change"
        action = "Continue the planned review schedule if the person is clinically well; confirm stability by examination."
    elif mean_absolute_change < 0.16:
        signal = "some visible change"
        action = "Arrange clinician review to decide whether the change represents healing, persistence, or a capture difference."
    else:
        signal = "substantial visible change"
        action = "Prioritise clinician review sooner, especially if symptoms or red flags are worsening."
    return {
        "change_score": round(mean_absolute_change, 3),
        "brightness_change": round(brightness_change, 3),
        "signal": signal,
        "action": action,
        "limitation": "This image comparison is sensitive to lighting, distance, angle, focus, and background. It cannot determine healing or diagnose progression.",
    }


def validate_image_bytes(data: bytes, mime_type: str | None = None) -> Image.Image:
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("The image exceeds the 10 MB upload limit.")
    if mime_type and mime_type.lower() not in ALLOWED_TYPES:
        raise ValueError("Only JPG, JPEG, and PNG images are supported.")
    try:
        image = Image.open(BytesIO(data))
        image.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    return image


def predict_image(bundle: Any, image: Image.Image, top_k: int = 3) -> dict[str, Any]:
    image = image.convert("RGB")
    inputs = bundle.processor(images=image, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = bundle.model(**inputs)
    logits = outputs.logits[0]
    scores = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(scores, k=min(top_k, scores.numel()))
    id2label = {int(k): str(v) for k, v in bundle.model.config.id2label.items()}
    top = [{"index": int(idx), "label": id2label[int(idx)], "score": float(value)} for value, idx in zip(values.cpu(), indices.cpu())]
    return {"predicted_index": top[0]["index"], "predicted_label": top[0]["label"], "top": top, "logits": logits.detach().cpu()}
