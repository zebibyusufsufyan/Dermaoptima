"""Grad-CAM for Hugging Face ViT classifiers when patch-token hooks are available."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _last_vit_block(model: Any) -> Any:
    try:
        # The classifier consumes the final CLS token. Hooking the final block's
        # patch outputs gives zero spatial gradients because those tokens are no
        # longer mixed with CLS. The penultimate block retains a later attention
        # block through which patch-token influence can reach the classifier.
        layers = model.vit.encoder.layer
        return layers[-2] if len(layers) >= 2 else layers[-1]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("Grad-CAM is only implemented for this model's ViT encoder.") from exc


def gradcam(bundle: Any, image: Image.Image, class_index: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (RGB image, heatmap, overlay, limitation note)."""
    model = bundle.model
    device = bundle.device
    inputs = bundle.processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = _last_vit_block(model)

    def forward_hook(_module, _args, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        activations.append(tensor)
        tensor.register_hook(lambda grad: gradients.append(grad))

    handle = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        idx = int(outputs.logits.argmax(dim=-1).item()) if class_index is None else int(class_index)
        outputs.logits[0, idx].backward()
    finally:
        handle.remove()
    if not activations or not gradients:
        raise RuntimeError("The model did not expose gradients from the ViT target layer.")

    activation = activations[0][0]
    gradient = gradients[0][0]
    # CLS token is not a spatial patch; use the remaining patch tokens.
    if activation.shape[0] > 1:
        activation, gradient = activation[1:], gradient[1:]
    weights = gradient.mean(dim=0)
    cam = torch.relu((activation * weights).sum(dim=-1))
    grid = int(cam.numel() ** 0.5)
    if grid * grid != cam.numel():
        raise RuntimeError("Patch token count is not a square grid; Grad-CAM cannot be rendered safely.")
    cam = cam.reshape(1, 1, grid, grid)
    cam = F.interpolate(cam, size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    heat = np.zeros_like(rgb)
    heat[..., 0] = np.clip(cam * 255, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip((1 - np.abs(cam - 0.5) * 2) * 180, 0, 255).astype(np.uint8)
    overlay = np.clip(0.55 * rgb + 0.45 * heat, 0, 255).astype(np.uint8)
    note = "Grad-CAM uses gradients from penultimate ViT patch tokens so their influence can flow through the final attention block; it is an attribution aid, not a diagnostic explanation."
    return rgb, heat, overlay, note
