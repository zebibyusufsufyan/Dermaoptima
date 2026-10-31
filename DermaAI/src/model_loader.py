"""Model loading utilities for the DermaAI research prototype."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = os.getenv("DERMAAI_MODEL_ID", "LaurianeMD/vit-skin-disease")

@dataclass(frozen=True)
class ModelBundle:
    model: Any
    processor: Any
    device: torch.device
    model_id: str


def select_device() -> torch.device:
    """Prefer CUDA when available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalise_labels(config: Any) -> dict[int, str]:
    raw = getattr(config, "id2label", {}) or {}
    return {int(k): str(v) for k, v in raw.items()}


@st.cache_resource(show_spinner=False)
def load_model(model_id: str = MODEL_ID) -> ModelBundle:
    """Download/load the processor and classifier once per Streamlit process."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    device = select_device()
    model.to(device)
    model.eval()
    return ModelBundle(model=model, processor=processor, device=device, model_id=model_id)


def model_metadata(bundle: ModelBundle) -> dict[str, Any]:
    config = bundle.model.config
    image_size = getattr(config, "image_size", None)
    if image_size is None:
        image_size = getattr(bundle.processor, "size", None)
    return {
        "model_id": bundle.model_id,
        "architecture": getattr(config, "architectures", [config.__class__.__name__]),
        "num_labels": int(getattr(config, "num_labels", len(config.id2label))),
        "id2label": _normalise_labels(config),
        "image_size": image_size,
        "device": str(bundle.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
