"""Model-agnostic explainability: occlusion sensitivity.

Slide a grey patch over the image and record how much the predicted probability of
the target class drops. Regions whose occlusion hurts most are the regions the model
relies on. Needs nothing but a predict function, so it works for every model here
(and is easy to test without TensorFlow).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def occlusion_sensitivity(predict_fn: Callable[[np.ndarray], np.ndarray],
                          image: np.ndarray, target_class: int,
                          patch: int = 32, stride: int = 16,
                          fill_value: float = 0.0) -> np.ndarray:
    """Return an (H, W) heat-map in [0, 1]; higher = occluding here hurts more.

    predict_fn : maps a batch (N, H, W, 3) in [0, 1] to class probabilities (N, C)
    image      : (H, W, 3) float array in [0, 1]
    """
    h, w, _ = image.shape
    base = float(predict_fn(image[None])[0, target_class])

    ys = list(range(0, max(h - patch, 0) + 1, stride))
    xs = list(range(0, max(w - patch, 0) + 1, stride))
    if ys[-1] + patch < h:
        ys.append(h - patch)          # make sure the bottom/right edges get covered
    if xs[-1] + patch < w:
        xs.append(w - patch)

    batch, coords = [], []
    for y in ys:
        for x in xs:
            occluded = image.copy()
            occluded[y:y + patch, x:x + patch, :] = fill_value
            batch.append(occluded)
            coords.append((y, x))
    probs = predict_fn(np.stack(batch))[:, target_class]

    heat = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for (y, x), p in zip(coords, probs):
        heat[y:y + patch, x:x + patch] += base - p
        count[y:y + patch, x:x + patch] += 1
    heat = heat / np.maximum(count, 1)
    heat = np.clip(heat, 0, None)      # only keep evidence FOR the class
    return heat / heat.max() if heat.max() > 0 else heat


def overlay_heatmap(image: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a 'jet' colour-mapped heat-map over the image; returns uint8 RGB."""
    from matplotlib import colormaps
    colour = colormaps["jet"](heat)[..., :3]
    blended = (1 - alpha) * image + alpha * colour
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
